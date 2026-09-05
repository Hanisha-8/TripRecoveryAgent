"""Offline gate for R1, plus an opt-in live run.

    python -m eval.smoke_r1            # offline, no API key, no spend
    python -m eval.smoke_r1 --live     # also runs the real agent once

Everything the plan compiler enforces is tested here by *effect*: a deliberately
broken plan is compiled and the run fails if it is accepted. A rule that has only
been read is a rule that has never run — and these particular rules exist to catch
outputs that look completely convincing.
"""

from __future__ import annotations

import sys
from datetime import datetime

from config import DATA_DIR, FAIL_MODE, FROZEN_CLOCK
from core.impact import impact_of_replacement, validate_itinerary
from core.plan_compiler import compile_plan, itinerary_version
from models import (
    ActionClass,
    ActionKind,
    DisruptionKind,
    PlanDraft,
    PlanOption,
    compute_plan_version,
)
from recovery.agent import (
    assess,
    flight_candidates,
    load_itinerary,
    redact,
    verify_disruption,
)
from recovery.subagents import check_tool_wiring
from recovery.tools import ORCHESTRATOR_TOOLS

from eval.harness import check, good_draft as _good_draft, report

ITINERARY = DATA_DIR / "sample_itinerary_sin.json"


# ===============================================================
# Offline suite
# ===============================================================
def offline() -> None:
    print("\n=== config ===")
    check("FROZEN_CLOCK is set", bool(FROZEN_CLOCK),
          "eval results drift between runs without it — copy .env.example to .env")
    check("FAIL_MODE is 'none' for the baseline", FAIL_MODE == "none",
          f"got {FAIL_MODE!r}; the uncertainty path is exercised separately")

    print("\n=== wiring ===")
    try:
        check_tool_wiring()
        check("tool wiring invariants hold", True)
    except AssertionError as exc:
        check("tool wiring invariants hold", False, str(exc))

    # `write_todos` sat in the orchestrator prompt for four phases without ever
    # being bound — deepagents does not install langchain's TodoListMiddleware.
    # Every run burned a turn on it and Groq rejected the call outright.
    from recovery import prompts as _prompts
    from recovery.subagents import check_prompt_tool_references

    check("no prompt names write_todos",
          "write_todos" not in _prompts.ORCHESTRATOR_PROMPT,
          "the tool is not bound, so the instruction wastes a turn and 400s on Groq")

    original = _prompts.ORCHESTRATOR_PROMPT
    try:
        _prompts.ORCHESTRATOR_PROMPT = original + "\n\nCall `invent_a_tool` first."
        check_prompt_tool_references()
        check("the guard catches a prompt naming an unbound tool", False,
              "a phantom tool reference passed — this is how write_todos survived")
    except AssertionError as exc:
        check("the guard catches a prompt naming an unbound tool",
              "invent_a_tool" in str(exc), str(exc)[:120])
    finally:
        _prompts.ORCHESTRATOR_PROMPT = original

    print("\n=== itinerary + redaction ===")
    itinerary = load_itinerary(ITINERARY)
    check("itinerary loads", len(itinerary.bookings) == 4,
          f"expected 4 bookings, got {len(itinerary.bookings)}")
    check("party size survives parsing", itinerary.party_size == 4)

    raw_pnr = itinerary.bookings[0].legs[0].booking_reference
    clean = redact(itinerary)
    check("source fixture carries a PNR to strip", raw_pnr is not None)
    check("redaction removes the PNR",
          clean.bookings[0].legs[0].booking_reference is None)
    check("redaction does not change the itinerary version",
          itinerary_version(clean) == itinerary_version(itinerary),
          "version must track bookings, not redaction state, or approval breaks")

    print("\n=== step 1: verification ===")
    disruption = verify_disruption(itinerary)
    check("a disruption is found", disruption is not None)
    assert disruption is not None
    check("it is matched to the right booking",
          disruption.booking_id == "bk_flight_sq123", disruption.booking_id)
    check("the flight code is normalised", disruption.flight_iata == "SQ123",
          disruption.flight_iata)

    from recovery.agent import flight_designator

    for carrier, number, expected in [
        ("SQ", "SQ123", "SQ123"),   # our fixture: prefix already present
        ("SQ", "123", "SQ123"),     # bare number: prefix needed
        ("SQ", "AI999", "AI999"),   # a different prefix must not be doubled up
        ("sq", " 123 ", "SQ123"),   # whitespace and case from an upload
    ]:
        check(f"designator({carrier!r}, {number!r}) == {expected!r}",
              flight_designator(carrier, number) == expected,
              flight_designator(carrier, number))
    check("the kind is cancelled", disruption.kind is DisruptionKind.CANCELLED)
    check("it is verified", disruption.verified)
    check("R1 has no live provider, so it is flagged uncertain", disruption.uncertain,
          "a fixture-sourced verification must not read as authoritative")

    print("\n=== steps 2-3: impact + candidates ===")
    impact = assess(itinerary, disruption)
    blocking = impact.blocking_node_ids()
    check("every downstream node is impacted", len(impact.impacted) == 4,
          f"got {[i.node_id for i in impact.impacted]}")
    check("a cancellation breaks all of them",
          all(i.severity == "broken" for i in impact.impacted))
    check("blocking set is non-empty", len(blocking) == 4, str(blocking))

    candidates = flight_candidates(itinerary, disruption)
    check("replacement flights are found", len(candidates) == 4, str(len(candidates)))
    check("candidates include the next day",
          any(c["departure"].startswith("2026-09-16") for c in candidates))

    print("\n=== residual feasibility ===")
    residual = impact_of_replacement(
        itinerary.bookings, "bk_flight_sq123",
        datetime.fromisoformat("2026-09-15T23:15:00+08:00"),
    )
    still = {i.booking_id for i in residual.impacted if i.severity in ("broken", "at_risk")}
    check("arriving 23:15 still breaks transfer, hotel and activity",
          still == {"bk_transfer_sin_hotel", "bk_hotel_marina", "bk_activity_night_safari"},
          str(sorted(still)))

    print("\n=== compiler: the good plan ===")
    good = _good_draft(blocking)
    result = compile_plan(good, itinerary=itinerary, verified_disruption=disruption,
                          impact=impact, flight_candidates=candidates)
    check("a correct two-option plan compiles", result.ok, "; ".join(result.violations))

    if result.plan is not None:
        plan = result.plan
        check("the compiled plan carries the graph's impacts",
              len(plan.impacts) == 4, str(len(plan.impacts)))
        check("plan_version is reproducible", plan.recompute_plan_version() == plan.plan_version)
        check("plan_version is namespaced", plan.plan_version.startswith("pv_"))
        check("both options survive", {o.option_id for o in plan.options} == {"A", "B"})

        # Approval binding: editing the text a human approved must invalidate the hash.
        tampered = plan.model_copy(deep=True)
        tampered.options[0].actions[1].message_body = "Cancel our reservation entirely."
        check("editing approved message text changes plan_version",
              tampered.recompute_plan_version() != plan.plan_version,
              "approval would survive a rewritten outbound message")

        # A benign regeneration must NOT invalidate it.
        same = compute_plan_version(
            trip_id=plan.trip_id, itinerary_version=plan.itinerary_version,
            disruption_summary=plan.disruption_summary, impacts=plan.impacts,
            options=plan.options,
        )
        check("regenerating an identical plan hashes the same", same == plan.plan_version,
              "a retry would invalidate a valid approval")

    print("\n=== the deterministic planner ===")
    # The demo and the upload path both run on this, so it needs the same
    # coverage as the hand-authored fixture — and it must clear the same gate.
    from core.planner import NoRecovery, build_draft, choose_flights

    built = build_draft(itinerary, disruption, impact, candidates)
    res = compile_plan(built, itinerary=itinerary, verified_disruption=disruption,
                        impact=impact, flight_candidates=candidates)
    check("the planner's draft compiles", res.ok, "; ".join(res.violations))

    if res.plan is not None:
        a, b = res.plan.options
        check("it picks two different flights",
              a.flight.flight_number != b.flight.flight_number,
              f"{a.flight.flight_number} vs {b.flight.flight_number}")
        check("A arrives before B", a.flight.arrival < b.flight.arrival,
              f"{a.flight.arrival} vs {b.flight.arrival}")
        check("B is the cheaper trade-off",
              (b.additional_cost_amount or 0) < (a.additional_cost_amount or 0),
              f"A={a.additional_cost_amount} B={b.additional_cost_amount}")
        check("every derived action that moves a booking carries a time",
              all(act.new_start is not None
                  for opt in res.plan.options for act in opt.actions
                  if act.needs_new_times()),
              "a rescheduling action with no time cannot be applied")
        check("the overnight option cites the accommodation rule",
              any("ACCOM" in c for c in b.citations), str(b.citations))
        check("payment stays human-required",
              all(act.action_class.value == "human_required"
                  for opt in res.plan.options for act in opt.actions
                  if act.kind is ActionKind.PAY_FOR_FLIGHT))
        check("the hotel message is written out in full, no placeholders",
              all("[" not in (act.message_body or "")
                  for opt in res.plan.options for act in opt.actions
                  if act.message_body),
              "a placeholder would make the action human-required")

    # Only one usable flight is not two options, and padding would be worse
    # than saying so.
    single = [c for c in candidates
              if c["flight_number"] == "SQ425"]
    try:
        build_draft(itinerary, disruption, impact, single)
        check("one usable flight yields no plan rather than a padded one", False,
              "a second option was invented")
    except NoRecovery as exc:
        check("one usable flight yields no plan rather than a padded one",
              "no second option" in str(exc), str(exc))

    try:
        choose_flights(itinerary, disruption, [])
        check("no candidates raises rather than returning nothing", False)
    except NoRecovery:
        check("no candidates raises rather than returning nothing", True)

    print("\n=== the draft can only say what the model decides ===")
    # Both R1 live failures were the model restating provider data and getting it
    # wrong: SQ425 with a departure of 16:10 instead of 16:30, and AI2380 given
    # SQ421's departure time. They are now unrepresentable rather than corrected.
    from models import FlightChoice, OptionDraft

    for field in ("departure", "arrival", "price_amount", "stops", "source",
                  "fetched_at", "uncertain", "fallback_used"):
        check(f"FlightChoice cannot state `{field}`",
              field not in FlightChoice.model_fields,
              "a field the compiler overwrites is a field the model can get wrong")
    check("FlightChoice states only carrier, number and date",
          set(FlightChoice.model_fields) == {"carrier", "flight_number",
                                             "departure_date"},
          str(sorted(FlightChoice.model_fields)))
    check("the draft cannot state an arrival delay",
          "arrival_delay_minutes" not in OptionDraft.model_fields,
          "it is arithmetic on two provider timestamps")
    check("the draft cannot state the impact list",
          "impacts" not in PlanDraft.model_fields,
          "the compiler discarded it every run in favour of the graph's")
    check("but the compiled option carries both",
          {"arrival_delay_minutes", "flight"} <= set(PlanOption.model_fields))

    print("\n=== compiler: flight canonicalisation ===")
    # The first live run failed here: gpt-4.1 picked exactly the right flight but
    # wrote carrier="Singapore Airlines" where the provider says "SQ", and a
    # string-equality check rejected two correct recommendations.
    variant = _good_draft(blocking)
    variant.options[0].flight.carrier = "Singapore Airlines"
    res = compile_plan(variant, itinerary=itinerary, verified_disruption=disruption,
                        impact=impact, flight_candidates=candidates)
    check("a flight named by airline rather than IATA code still resolves", res.ok,
          "; ".join(res.violations))
    if res.plan is not None:
        flown = res.plan.options[0].flight
        check("the carrier is rewritten from the provider record", flown.carrier == "SQ",
              flown.carrier)
        check("the fare comes from the provider, never the draft",
              flown.price_amount == 120.0, str(flown.price_amount))
        check("so does the stop count", flown.stops == 0, str(flown.stops))
        check("and the departure time the draft never stated",
              flown.departure == datetime.fromisoformat("2026-09-15T16:30:00+05:30"),
              flown.departure.isoformat())
        check("the arrival delay is derived",
              res.plan.options[0].arrival_delay_minutes == 420,
              str(res.plan.options[0].arrival_delay_minutes))
        check("canonicalisation is hashed, so approval binds the real fare",
              res.plan.recompute_plan_version() == res.plan.plan_version)

    unparseable = _good_draft(blocking)
    unparseable.options[0].flight.departure_date = "15 September"
    res = compile_plan(unparseable, itinerary=itinerary,
                        verified_disruption=disruption, impact=impact,
                        flight_candidates=candidates)
    check("a departure_date that is not YYYY-MM-DD is rejected",
          not res.ok and any("not a YYYY-MM-DD" in v for v in res.violations),
          str(res.violations))

    # The hedge is written in, not demanded. gpt-4.1 would not converge on it over
    # two attempts even with the phrase quoted verbatim in the rejection.
    unhedged = _good_draft(blocking)
    unhedged.options[0].rationale = "Arrives the same night"
    unhedged.options[1].rationale = "Cheapest way out."
    res = compile_plan(unhedged, itinerary=itinerary, verified_disruption=disruption,
                        impact=impact, flight_candidates=candidates)
    check("an unhedged rationale compiles rather than rejecting", res.ok,
          "; ".join(res.violations))
    if res.plan is not None:
        check("the hedge is appended to every uncertain option",
              all("verify at booking" in o.rationale.lower() for o in res.plan.options),
              str([o.rationale for o in res.plan.options]))
        check("appending it reads as a sentence, not a splice",
              res.plan.options[0].rationale.startswith("Arrives the same night. ")
              and res.plan.options[1].rationale.startswith("Cheapest way out. "),
              str([o.rationale for o in res.plan.options]))
        check("an already-hedged rationale is left alone",
              compile_plan(
                  _good_draft(blocking), itinerary=itinerary,
                  verified_disruption=disruption, impact=impact,
                  flight_candidates=candidates,
              ).plan.options[0].rationale.count("verify at booking") == 1,
              "the hedge must not be duplicated")

    wrong_day = _good_draft(blocking)
    wrong_day.options[0].flight.departure_date = "2026-09-16"
    res = compile_plan(wrong_day, itinerary=itinerary, verified_disruption=disruption,
                        impact=impact, flight_candidates=candidates)
    check("but the wrong DAY is still rejected",
          not res.ok and any("different day" in v for v in res.violations),
          f"ok={res.ok} violations={res.violations}")

    print("\n=== compiler: each rule rejects ===")

    def rejects(name: str, mutate, expect: str) -> None:
        draft = _good_draft(blocking)
        mutate(draft)
        res = compile_plan(draft, itinerary=itinerary, verified_disruption=disruption,
                            impact=impact, flight_candidates=candidates)
        hit = any(expect in v for v in res.violations)
        check(name, not res.ok and hit,
              f"ok={res.ok} violations={res.violations}")

    def _three_options(d: PlanDraft) -> None:
        extra = d.options[0].model_copy(deep=True)
        d.options.append(extra)

    rejects("C1 three options", _three_options, "exactly 2 options")

    rejects("C2 wrong option ids",
            lambda d: setattr(d.options[1], "option_id", "A"), "exactly A and B")

    rejects("C3 duplicate action ids",
            lambda d: setattr(d.options[1].actions[0], "action_id",
                              d.options[0].actions[0].action_id),
            "duplicate action_id")

    rejects("C4 invented flight",
            lambda d: setattr(d.options[0].flight, "flight_number", "SQ999"),
            "not in the provider's results")

    rejects("C7 payment marked agent-safe",
            lambda d: setattr(d.options[0].actions[0], "action_class",
                              ActionClass.AGENT_SAFE),
            "moves money")

    rejects("C8 outbound message with no approved text",
            lambda d: setattr(d.options[0].actions[1], "message_body", "   "),
            "requires the exact message_body")

    def _drop_transfer_action(d: PlanDraft) -> None:
        d.options[0].actions = [
            a for a in d.options[0].actions
            if a.target_booking_id != "bk_transfer_sin_hotel"
        ]

    rejects("C9 an impact left unhandled", _drop_transfer_action,
            "no action touches it")
    rejects("C9 is proven by simulation, not by inspecting the action list",
            _drop_transfer_action, "performing every action still leaves")

    def _useless_time(d: PlanDraft) -> None:
        # The action names the right booking and moves it somewhere useless. This
        # is what the old check-the-list version of C9 accepted.
        transfer = next(a for a in d.options[0].actions
                        if a.target_booking_id == "bk_transfer_sin_hotel")
        transfer.new_start = datetime.fromisoformat("2026-09-15T12:00:00+08:00")
        transfer.new_end = datetime.fromisoformat("2026-09-15T12:45:00+08:00")

    rejects("C9 an action that moves a booking to a useless time",
            _useless_time, "performing every action still leaves")

    def _no_new_time(d: PlanDraft) -> None:
        transfer = next(a for a in d.options[1].actions
                        if a.target_booking_id == "bk_transfer_sin_hotel")
        transfer.new_start = None
        transfer.new_end = None

    rejects("a rescheduling action with no new time is rejected",
            _no_new_time, "carries no new_start")

    def _drop_claim(d: PlanDraft) -> None:
        d.options[1].resolves = []

    rejects("C9 an impact not claimed", _drop_claim, "does not claim to resolve")

    print("\n=== agent assembly (no model call, no spend) ===")
    # Catches the mismatches a prompt review never will: a renamed
    # create_deep_agent kwarg, a middleware hook that no longer exists, a subagent
    # dict the wrong shape, or response_format being silently dropped.
    try:
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        from config import WORKSPACE_DIR
        from recovery.agent import build_recovery_agent, prepare_workspace

        prepare_workspace(itinerary, WORKSPACE_DIR)
        agent = build_recovery_agent(
            GenericFakeChatModel(messages=iter(["unused"])),
            itinerary=itinerary, disruption=disruption, workspace=WORKSPACE_DIR,
        )
        check("the agent assembles", agent is not None)

        output_keys = set(agent.get_output_jsonschema().get("properties", {}))
        check("response_format is wired, so PlanDraft comes back typed",
              "structured_response" in output_keys, str(sorted(output_keys)))

        node_names = set(agent.get_graph().nodes)
        check("the cost ceiling is in the graph",
              any("CostCeiling" in n for n in node_names), str(sorted(node_names)))

        # deepagents only inherits parent middleware into FORKED subagents
        # (graph.py:726). Ours are declarative specs, so the ceiling has to be
        # attached to each one or it counts the orchestrator's calls only — the
        # smaller half of a run, since the specialists do most of the work.
        from recovery.subagents import SUBAGENTS

        from recovery.agent import build_recovery_agent as _build  # noqa: F401
        seen: list[dict] = []
        import deepagents

        real = deepagents.create_deep_agent

        def _spy(**kwargs):
            seen.extend(kwargs.get("subagents") or [])
            return real(**kwargs)

        deepagents.create_deep_agent = _spy
        try:
            build_recovery_agent(
                GenericFakeChatModel(messages=iter(["unused"])),
                itinerary=itinerary, disruption=disruption, workspace=WORKSPACE_DIR,
            )
        finally:
            deepagents.create_deep_agent = real

        recorders = {
            s["name"]: [m for m in s.get("middleware", [])
                        if type(m).__name__ == "CostCeilingMiddleware"]
            for s in seen
        }
        check("every subagent spec is handed a cost recorder",
              len(seen) == len(SUBAGENTS)
              and all(len(v) == 1 for v in recorders.values()),
              str({k: len(v) for k, v in recorders.items()}))
        check("each recorder is labelled with its own agent",
              all(v[0].label == name for name, v in recorders.items()),
              str({k: v[0].label for k, v in recorders.items() if v}))
        check("but they all share one ledger, because there is one budget",
              len({id(v[0].ledger) for v in recorders.values() if v}) == 1,
              "five independent ceilings would each pass while the run blew past all")

        print("\n=== tool budget: what the model is actually shown ===")
        from langchain_core.utils.function_calling import convert_to_openai_tool

        from recovery.middleware import ToolBudgetMiddleware
        from recovery.subagents import TOOL_DENYLIST

        budgets = {
            s["name"]: [m for m in s.get("middleware", [])
                        if isinstance(m, ToolBudgetMiddleware)]
            for s in seen
        }
        check("every subagent gets a tool budget",
              all(len(v) == 1 for v in budgets.values()),
              str({k: len(v) for k, v in budgets.items()}))
        check("and it matches that agent's denylist",
              all(v[0].drop == TOOL_DENYLIST[name]
                  for name, v in budgets.items() if v),
              str({k: sorted(v[0].drop) for k, v in budgets.items() if v}))

        # The four filesystem tools nothing here uses. Advertising them costs
        # 1,369 tokens of schema on every call across five agents.
        never_used = {"grep", "glob", "edit_file", "delete"}
        check("no agent is shown grep, glob, edit_file or delete",
              all(never_used <= v[0].drop for v in budgets.values() if v))
        check("the critic cannot write",
              "write_file" in TOOL_DENYLIST["critic"],
              "a critic that can write can quietly fix what it was asked to judge")
        check("the specialists that write files still can",
              all("write_file" not in TOOL_DENYLIST[n]
                  for n in ("policy-checker", "impact-analyst", "options-finder")))

        # Denylist, not allowlist: an unfamiliar tool must pass through. An
        # allowlist would silently drop a provider's structured-output shim and
        # the failure would look like the model's fault.
        class _Req:
            def __init__(self, tools):
                self.tools = tools
                self.overridden = None

            def override(self, **kw):
                self.overridden = kw
                return self

        budget = ToolBudgetMiddleware(frozenset({"grep"}), "test")
        stub = _Req([
            type("T", (), {"name": "grep"})(),
            type("T", (), {"name": "__provider_structured_output__"})(),
        ])
        budget.wrap_model_call(stub, lambda r: r)
        kept = [t.name for t in (stub.overridden or {}).get("tools", [])]
        check("a tool nobody named survives the budget",
              kept == ["__provider_structured_output__"], str(kept))

        advertised = sum(
            len(str(convert_to_openai_tool(t))) for t in ORCHESTRATOR_TOOLS
        )
        check("the orchestrator still keeps its own tool", advertised > 0)
    except Exception as exc:  # noqa: BLE001
        check("the agent assembles", False, f"{type(exc).__name__}: {exc}")

    print("\n=== cost ledger ===")
    from recovery.middleware import CostLedger

    ledger = CostLedger(max_cost_usd=10.0, max_llm_calls=10)
    ledger.record("orchestrator", {
        "input_tokens": 6000, "output_tokens": 500,
        "input_token_details": {"cache_read": 5000},
    })
    ledger.record("options-finder", {
        # The raw OpenAI shape, which reaches us when usage_metadata is absent.
        "prompt_tokens": 3000, "completion_tokens": 200,
        "prompt_tokens_details": {"cached_tokens": 1024},
    })
    check("usage is attributed per agent",
          set(ledger.summary()["by_agent"]) == {"orchestrator", "options-finder"})
    check("totals add up across agents",
          ledger.input_tokens == 9000 and ledger.output_tokens == 700)
    check("cached tokens are read from the LangChain shape",
          ledger.by_agent["orchestrator"].cached_input_tokens == 5000)
    check("and from the raw OpenAI shape",
          ledger.by_agent["options-finder"].cached_input_tokens == 1024,
          "a provider reporting the raw dict would look like a 0% cache hit")

    tight = CostLedger(max_cost_usd=10.0, max_llm_calls=2)
    tight.record("orchestrator", {"input_tokens": 10, "output_tokens": 1})
    tight.record("critic", {"input_tokens": 10, "output_tokens": 1})
    try:
        tight.record("critic", {"input_tokens": 10, "output_tokens": 1})
        check("the call ceiling counts every agent, not just the orchestrator", False,
              "a third call passed a ceiling of two")
    except RuntimeError as exc:
        check("the call ceiling counts every agent, not just the orchestrator",
              "critic=" in str(exc),
              "the breach message must say where the calls went")

    broke = CostLedger(max_cost_usd=0.001, max_llm_calls=100)
    try:
        broke.record("orchestrator", {"input_tokens": 1_000_000, "output_tokens": 0})
        check("the cost ceiling raises rather than warns", False, "no RuntimeError")
    except RuntimeError:
        check("the cost ceiling raises rather than warns", True)

    print("\n=== R4 contract: validation ===")
    original = validate_itinerary(itinerary.trip_id, itinerary.bookings)
    check("the undisrupted itinerary is valid", original.valid,
          f"remaining={[r.node_id for r in original.remaining]}")
    check("its tight 45m airport transfer is surfaced as a warning, not invalidity",
          [w.booking_id for w in original.warnings] == ["bk_transfer_sin_hotel"],
          f"warnings={[w.booking_id for w in original.warnings]}")
    check("a pending handoff keeps it open",
          not validate_itinerary(
              itinerary.trip_id, itinerary.bookings, pending_handoffs=["a-pay"]
          ).valid,
          "an unfinished human action must block close-out")

    broken = [b.model_copy(deep=True) for b in itinerary.bookings]
    safari = next(b for b in broken if b.booking_id == "bk_activity_night_safari")
    safari.start = safari.start.replace(hour=16, minute=0)  # before the flight lands
    check("an itinerary with an infeasible dependency is not valid",
          not validate_itinerary(itinerary.trip_id, broken).valid,
          "close-out must reject a genuinely impossible schedule")


# ===============================================================
# Live run — opt in, costs money
# ===============================================================
def live() -> None:
    from config import LLM_MODEL
    from recovery.agent import run_recovery
    from recovery.llm import available

    print(f"\n=== live run ({LLM_MODEL}) ===")
    if not available(LLM_MODEL):
        check("model key configured", False, f"no key for {LLM_MODEL!r}; skipping")
        return

    # A provider outage, an exhausted quota or a breached cost ceiling is a fact
    # about the environment, not a failed assertion. Letting it raise buried the
    # offline results under a traceback the first time it happened.
    try:
        result = run_recovery(load_itinerary(ITINERARY))
    except Exception as exc:  # noqa: BLE001
        check("live run reached the provider", False,
              f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        return
    cost = result.cost
    print(f"  attempts={result.attempts}")
    print(f"  calls={cost.get('llm_calls')} in={cost.get('input_tokens')} "
          f"out={cost.get('output_tokens')} cached={cost.get('cached_input_tokens')} "
          f"({cost.get('cache_hit_rate', 0):.0%}) "
          f"${cost.get('estimated_cost_usd', 0):.4f}")
    print(f"  {'agent':<17}{'calls':>6}{'in':>9}{'avg/call':>10}{'cached':>9}{'hit':>6}")
    for label, usage in (cost.get("by_agent") or {}).items():
        print(f"  {label:<17}{usage['calls']:>6}{usage['input_tokens']:>9}"
              f"{usage['avg_input_per_call']:>10}{usage['cached_input_tokens']:>9}"
              f"{usage['cache_hit_rate']:>6.0%}")
    if result.stopped_reason:
        check("run produced a plan", False, f"stopped: {result.stopped_reason}")
        return

    check("run produced a compiled plan", result.ok, "; ".join(result.violations))
    if result.plan is not None:
        for option in result.plan.options:
            agent_safe = [a.kind.value for a in option.actions
                          if a.action_class is ActionClass.AGENT_SAFE]
            human = [a.kind.value for a in option.actions
                     if a.action_class is ActionClass.HUMAN_REQUIRED]
            print(
                f"  Option {option.option_id}: "
                f"{option.flight.carrier}{option.flight.flight_number} "
                f"+{option.arrival_delay_minutes}m "
                f"{option.additional_cost_currency}{option.additional_cost_amount:.0f} "
                f"risk={option.risk}\n"
                f"    agent: {agent_safe}\n    human: {human}"
            )
        check("first attempt was accepted", result.attempts == 1,
              f"took {result.attempts} attempts — see the logged rejection report")


def main() -> int:
    offline()
    if "--live" in sys.argv:
        live()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
