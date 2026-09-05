"""Offline gate for R2 — approval binding and grounding.

    python -m eval.smoke_r2

No API key, no spend. The approval half is tested by trying to get past the gate
six different ways; the grounding half by auditing documents that are wrong in
ways that look right.
"""

from __future__ import annotations

import dataclasses
import tempfile
from datetime import timedelta
from pathlib import Path

from core.approval import (
    Authorization,
    RefusalReason,
    approve,
    authorize,
)
from config import DATA_MODE, IS_MOCK_DATA, WORKSPACE_DIR
from core.citations import NOT_COVERED, audit_citations, contact_for
from core.sources import mock_sources, sources
from core.store import (
    latest_plan,
    load_approval,
    load_plan,
    save_approval,
    save_plan,
)
from eval.harness import HOTEL_MESSAGE, Scenario, check, good_draft, report
from models import ActionKind
from recovery.prompts import orchestrator_prompt
from recovery.subagents import SUBAGENTS, check_tool_wiring
from recovery.tools import read_entitlements


def approval_binding() -> None:
    s = Scenario()
    plan = s.plan()

    print("\n=== approve ===")
    approval = approve(plan, "A", approved_by="alex")
    check("approving a real option yields a record",
          hasattr(approval, "plan_version"), str(approval))
    assert not isinstance(approval, type(None))
    check("the record binds to this exact plan version",
          getattr(approval, "plan_version", None) == plan.plan_version)

    bad = approve(plan, "C", approved_by="alex")
    check("approving an option that does not exist is refused",
          getattr(bad, "reason", None) is RefusalReason.UNKNOWN_OPTION, str(bad))

    print("\n=== authorize: the happy path ===")
    auth = authorize(plan, approval, s.itinerary)  # type: ignore[arg-type]
    check("a matching approval authorises", getattr(auth, "ok", False),
          getattr(auth, "detail", ""))
    if isinstance(auth, Authorization):
        check("the approved option is carried, not re-chosen", auth.option.option_id == "A")
        check("agent-safe actions are split out",
              {a.kind.value for a in auth.agent_actions}
              == {"send_hotel_message", "reschedule_activity"},
              str([a.kind.value for a in auth.agent_actions]))
        check("human actions are split out",
              {a.kind.value for a in auth.human_actions}
              == {"pay_for_flight", "rebook_transfer"},
              str([a.kind.value for a in auth.human_actions]))
        check("no action lands in both halves",
              not ({a.action_id for a in auth.agent_actions}
                   & {a.action_id for a in auth.human_actions}))
        check("every action is accounted for",
              len(auth.agent_actions) + len(auth.human_actions) == len(auth.option.actions))
        sent_text = next(
            a.message_body for a in auth.agent_actions
            if a.kind is ActionKind.SEND_HOTEL_MESSAGE
        )
        check("the approved hotel text is the text that would be sent",
              sent_text == HOTEL_MESSAGE, str(sent_text))

    print("\n=== authorize: a token cannot be forged ===")
    try:
        Authorization(
            _mint=object(), plan=plan, option=plan.options[0],
            approval=approval,  # type: ignore[arg-type]
            itinerary=s.itinerary,
        )
        check("constructing an Authorization directly is refused", False,
              "a forged token was accepted — the executor gate is bypassable")
    except PermissionError:
        check("constructing an Authorization directly is refused", True)

    if isinstance(auth, Authorization):
        check("the token carries the itinerary whose version was verified",
              auth.itinerary.trip_id == s.itinerary.trip_id
              and len(auth.itinerary.bookings) == len(s.itinerary.bookings),
              "the executor must not be handed a different trip than was authorised")

    print("\n=== authorize: each refusal ===")

    # The plan was edited after compilation. This is the case that makes the hash
    # worth having: the message body a human approved is not the one on the plan.
    tampered = plan.model_copy(deep=True)
    hotel = next(a for a in tampered.options[0].actions
                 if a.kind.value == "send_hotel_message")
    hotel.message_body = "Please cancel our reservation entirely."
    res = authorize(tampered, approval, s.itinerary)  # type: ignore[arg-type]
    check("a plan edited after compilation is refused",
          getattr(res, "reason", None) is RefusalReason.PLAN_TAMPERED, str(res))

    # A different, also-valid plan. The approval is genuine but not for this one.
    other_draft = s.compile(_regenerated(s)).plan
    check("the regenerated plan really is a different version",
          other_draft is not None and other_draft.plan_version != plan.plan_version,
          "fixture drift — the staleness check below would prove nothing")
    if other_draft is not None:
        res = authorize(other_draft, approval, s.itinerary)  # type: ignore[arg-type]
        check("an approval for another plan version is refused",
              getattr(res, "reason", None) is RefusalReason.APPROVAL_STALE, str(res))

    # The trip moved underneath the plan.
    moved = s.itinerary.model_copy(deep=True)
    safari = next(b for b in moved.bookings if b.booking_id == "bk_activity_night_safari")
    safari.start = safari.start + timedelta(days=1)
    safari.end = safari.end + timedelta(days=1)
    res = authorize(plan, approval, moved)  # type: ignore[arg-type]
    check("a plan built against an older itinerary is refused",
          getattr(res, "reason", None) is RefusalReason.ITINERARY_CHANGED, str(res))

    # An approval from someone else's trip.
    foreign = approval.model_copy(update={"trip_id": "trip_someone_else"})  # type: ignore[union-attr]
    res = authorize(plan, foreign, s.itinerary)
    check("an approval from a different trip is refused",
          getattr(res, "reason", None) is RefusalReason.TRIP_MISMATCH, str(res))

    print("\n=== persistence: survive a restart ===")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "r2.db"
        save_plan(plan, db_path=db)
        save_approval(approval, db_path=db)  # type: ignore[arg-type]

        reloaded = load_plan(plan.plan_version, db_path=db)
        check("a stored plan comes back", reloaded is not None)
        check("and still hashes to the version it was stored under",
              reloaded is not None
              and reloaded.recompute_plan_version() == plan.plan_version,
              "a round trip that changes the hash would refuse every approval")

        again = load_approval(plan.trip_id, plan.plan_version, db_path=db)
        check("a stored approval comes back", again is not None)
        if reloaded is not None and again is not None:
            res = authorize(reloaded, again, s.itinerary)
            check("approval granted before a restart still authorises after it",
                  getattr(res, "ok", False), getattr(res, "detail", ""))

        check("the latest plan for a trip is findable without knowing its hash",
              (lp := latest_plan(plan.trip_id, db_path=db)) is not None
              and lp.plan_version == plan.plan_version)

        # Tampering with the row, not the object — the case storage introduces.
        import sqlite3
        with sqlite3.connect(db) as conn:
            payload = conn.execute(
                "SELECT payload FROM plans WHERE plan_version = ?",
                (plan.plan_version,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE plans SET payload = ? WHERE plan_version = ?",
                (payload.replace("120.0", "999.0"), plan.plan_version),
            )
        edited = load_plan(plan.plan_version, db_path=db)
        res = authorize(edited, again, s.itinerary)  # type: ignore[arg-type]
        check("a plan edited in the database is refused on reload",
              getattr(res, "reason", None) is RefusalReason.PLAN_TAMPERED, str(res))


def _regenerated(s: Scenario):
    """The same plan with one option's prose changed — a genuinely new version."""
    from eval.harness import good_draft

    draft = good_draft(s.blocking)
    draft.options[0].headline = "Fly today and keep the Singapore arrival evening"
    return draft


def grounding() -> None:
    s = Scenario()

    print("\n=== corpus ===")
    src = sources()
    chunks = src.policy.chunks()
    check("the policy corpus loads", len(chunks) > 10, f"got {len(chunks)} chunks")
    check("chunk ids are unique", len({c.chunk_id for c in chunks}) == len(chunks))
    check("both domains are present",
          {c.domain for c in chunks} == {"disruption_care", "airline_rebooking"},
          str({c.domain for c in chunks}))
    check("the SQ cancellation accommodation rule is in the corpus",
          "disruption_care:SQ-CANCEL-ACCOM-01" in {c.chunk_id for c in chunks})

    print("\n=== grounding invariants ===")
    text = src.policy.grounding()
    check("the invariants file loads", "G1 —" in text)
    check("G6 is the rewritten one, not tripsure's flight-only rule",
          "only through classified actions" in text,
          "copying tripsure's G6 back would forbid the actions this product executes")
    check("G7 names the capability token, not graph topology",
          "Authorization" in text and "capability token" in text)
    check("the invariants stay out of retrieval",
          not any(c.domain in ("G1", "G7") for c in chunks)
          and "invariants" not in {c.source_file for c in chunks},
          "the invariants must not be retrievable, or they duplicate a chunk")

    print("\n=== retrieval ===")
    hits = src.policy.retrieve("hotel accommodation overnight", k=3)
    ids = [h["chunk_id"] for h in (hits.value or {}).get("hits", [])]
    check("an entitlement query retrieves accommodation chunks",
          any("ACCOM" in i for i in ids), str(ids))
    check("retrieval is not flagged uncertain", not hits.uncertain,
          "a local corpus is not a degraded source; hedging it would teach the agent "
          "to doubt grounded entitlements")

    exact = src.policy.retrieve("disruption_care:SQ-CANCEL-ACCOM-01", k=1)
    check("citing an id by name retrieves that id",
          (exact.value or {}).get("hits", [{}])[0].get("chunk_id")
          == "disruption_care:SQ-CANCEL-ACCOM-01",
          str((exact.value or {}).get("hits")),
          )
    miss = src.policy.retrieve("scuba diving equipment insurance", k=3)
    check("a query the corpus cannot answer says so",
          miss.error is not None, str(miss.value))

    print("\n=== citation audit ===")
    good = """## Accommodation
The carrier pays for a hotel for up to 2 nights [disruption_care:SQ-CANCEL-ACCOM-01].

## Rebooking
Free rebooking onto the next SQ service [disruption_care:SQ-CANCEL-REBOOK-01].

## Meals
Meal vouchers after a 3 hours wait [disruption_care:SQ-CANCEL-MEALS-01].

## Compensation
No fixed cash compensation is payable [disruption_care:SQ-CANCEL-COMP-01].

## Refund
Full refund of the unflown portion [disruption_care:SQ-CANCEL-REFUND-01].

## Contact
Call the SQ disruption desk on +65 6555 0100 [disruption_care:CONTACT-SQ-01].
"""
    audit = audit_citations(good, "cancelled", require_categories=True)
    check("a correctly grounded entitlements document passes", audit.ok, audit.report())

    def fails(name: str, doc: str, attr: str, *, kind: str = "cancelled",
              categories: bool = False) -> None:
        a = audit_citations(doc, kind, require_categories=categories)
        check(name, not a.ok and bool(getattr(a, attr)),
              f"ok={a.ok} {attr}={getattr(a, attr)}")

    fails("an invented chunk id is caught",
          "Hotel is covered [disruption_care:SQ-CANCEL-HOTEL-99].",
          "unresolved_citations")

    # The headline failure: a citation that resolves and is still the wrong rule.
    fails("a cancellation leaning on a DELAY chunk for compensation is caught",
          "## Compensation\nYou are owed compensation "
          "[disruption_care:SQ-DELAY-MEALS-01].",
          "kind_mismatches")

    fails("a compensation figure absent from the cited chunk is caught",
          "## Compensation\nYou are owed £520 "
          "[disruption_care:SQ-CANCEL-COMP-01].",
          "uncited_money")

    fails("a phone number the fixture does not contain is caught",
          "## Contact\nCall +65 6999 1234 [disruption_care:CONTACT-SQ-01].",
          "uncited_phones")

    fails("a night count absent from the cited chunk is caught",
          "## Accommodation\nHotel for up to 5 nights "
          "[disruption_care:SQ-CANCEL-ACCOM-01].",
          "uncited_durations")

    fails("an entitlements document missing categories is caught",
          "## Accommodation\nHotel for 2 nights "
          "[disruption_care:SQ-CANCEL-ACCOM-01].",
          "missing_categories", categories=True)

    # The exemption that had to be earned: fetched data is not a policy claim.
    priced = audit_citations(
        "## Option A\nSQ425 fare 120 USD [source: fixture, fetched_at 2026-09-15T09:00]",
        "cancelled",
    )
    check("a fare carrying source and fetched_at needs no citation", priced.ok,
          priced.report())

    # ...but the exemption must not launder a policy claim as tool data.
    fails("a compensation claim cannot hide behind provenance",
          "## Compensation\nYou are owed 520 GBP "
          "[source: fixture, fetched_at 2026-09-15T09:00]",
          "uncited_money")

    # A deleted row must not pass on the strength of a citation containing its name.
    fails("deleting the refund row is not covered by a citation naming refund",
          "## Rebooking\nFree rebooking [disruption_care:SQ-CANCEL-REFUND-01].",
          "missing_categories", categories=True)

    print("\n=== entitlements shown to the traveller ===")
    # The corpus existed from R2 but nothing surfaced it: an option displayed a
    # chunk id and nothing readable. A citation nobody can follow is decoration.
    from core.entitlements import (
        CATEGORIES,
        build_entitlements,
        cancellation_advice,
    )

    report = build_entitlements("SQ", "cancelled")
    check("all six categories are answered for SQ cancelled",
          report.covered_count == 6,
          str([(i.category, i.chunk_id) for i in report.items]))
    check("each answer cites a chunk that names its own category",
          all(
              token in (item.chunk_id or "").upper()
              for (_, _, _, token), item in zip(CATEGORIES, report.items)
              if item.covered
          ),
          str([(i.category, i.chunk_id) for i in report.items]))
    check("a cancellation never rests on a DELAY chunk",
          not any("DELAY" in (i.chunk_id or "") for i in report.items),
          "the error the corpus was written to catch")
    check("the contact number comes from the fixture",
          report.contact.get("disruption_desk") == "+65 6555 0100",
          str(report.contact.get("disruption_desk")))
    check("mock sources are declared once, in config",
          IS_MOCK_DATA and DATA_MODE == "mock",
          "the UI shows one quiet marker instead of a developer note per element")
    check("what we render passes our own citation audit",
          audit_citations(report.as_markdown(), "cancelled",
                          require_categories=True).ok,
          audit_citations(report.as_markdown(), "cancelled",
                          require_categories=True).report()[:300])

    delayed = build_entitlements("SQ", "delayed")
    check("a delay never rests on a CANCEL chunk",
          not any("CANCEL" in (i.chunk_id or "") for i in delayed.items),
          str([i.chunk_id for i in delayed.items]))
    check("an uncovered category is stated as a gap, not filled with the wrong chunk",
          delayed.covered_count < 6
          and all(NOT_COVERED in i.text for i in delayed.items if not i.covered),
          str([(i.category, i.chunk_id) for i in delayed.items]))

    ba = build_entitlements("BA", "cancelled")
    check("BA gets BA chunks, not SQ's",
          not any("SQ-" in (i.chunk_id or "") for i in ba.items),
          str([i.chunk_id for i in ba.items]))

    unknown = build_entitlements("ZZ", "cancelled")
    check("an unknown carrier falls back to generic chunks",
          all("GENERIC" in (i.chunk_id or "") for i in unknown.items if i.covered),
          str([i.chunk_id for i in unknown.items]))
    check("and is given no invented phone number",
          not unknown.contact.get("disruption_desk"), str(unknown.contact))

    advice = cancellation_advice(report)
    check("advice is offered for a covered cancellation", len(advice) >= 4,
          str(advice))
    check("no advice is offered for a category the corpus does not cover",
          not cancellation_advice(
              build_entitlements("ZZ", "cancelled")
          ) or all(isinstance(line, str) for line in advice))

    print("\n=== sources: the boundary is real ===")
    from core.sources import (
        ContactSource,
        CorpusPolicySource,
        FixtureContactSource,
        FlightSource,
        MockFlightSource,
        PolicySource,
    )

    live = sources()
    check("the mock set satisfies all three protocols",
          isinstance(live.flights, FlightSource)
          and isinstance(live.contacts, ContactSource)
          and isinstance(live.policy, PolicySource))
    check("and declares itself mock", live.is_mock and live.mode == "mock")

    try:
        sources("live")
        check("an unimplemented mode raises rather than falling back", False,
              "silently serving fixtures is how a demo gets taken for a deployment")
    except NotImplementedError as exc:
        check("an unimplemented mode raises rather than falling back",
              "no fallback" in str(exc).lower() or "only 'mock'" in str(exc),
              str(exc)[:120])

    # The payoff: a degraded source per test, rather than FAIL_MODE for the whole
    # process. Before this the uncertainty path could only be reached by changing
    # the environment, which meant one setting for every test in the run.
    stale = MockFlightSource(fail_mode="stale")
    result = stale.status("SQ123", "2026-09-15")
    check("a stale flight source is flagged uncertain per-instance",
          result.uncertain and "older than" in (result.error or ""),
          f"uncertain={result.uncertain} error={result.error!r}")
    check("while the default source stays clean",
          not live.flights.status("SQ123", "2026-09-15").uncertain,
          "injecting a degraded source must not leak into the shared one")

    # A source can be swapped wholesale — here, a policy corpus that is empty.
    class EmptyPolicy:
        name = "empty"

        def chunks(self):
            return ()

        def retrieve(self, query, k=3):
            return live.policy.retrieve(query, k)

        def grounding(self):
            return ""

    # `Sources` is a frozen dataclass, so one source swaps out cleanly and the
    # other two are untouched.
    empty = build_entitlements(
        "SQ", "cancelled",
        src=dataclasses.replace(mock_sources(), policy=EmptyPolicy()),
    )
    check("an empty corpus yields six stated gaps, not six silent passes",
          empty.covered_count == 0
          and all(NOT_COVERED in i.text for i in empty.items),
          str([(i.category, i.covered) for i in empty.items]))
    check("swapping the policy source leaves the contact source alone",
          empty.contact.get("disruption_desk") == "+65 6555 0100",
          str(empty.contact))

    print("\n=== contact lookup ===")
    sq = contact_for("SQ")
    check("a known carrier resolves", sq.get("disruption_desk") == "+65 6555 0100",
          str(sq))
    check("the fixture's provenance note is still available to callers",
          bool(sq.get("_warning")),
          "kept on the record for whoever swaps in a live source, not for the UI")
    unknown = contact_for("ZZ")
    check("an unknown carrier returns an error and advice, never a number",
          "error" in unknown and "Do not state a number." in unknown.get("advice", ""),
          str(unknown))

    print("\n=== compiler C10: citations ===")
    check("the good plan still compiles with citations required",
          s.compile(good_draft(s.blocking)).ok,
          "; ".join(s.compile(good_draft(s.blocking)).violations))

    no_cites = good_draft(s.blocking)
    no_cites.options[0].citations = []
    res = s.compile(no_cites)
    check("an option with no citations is rejected",
          not res.ok and any("no citations" in v for v in res.violations), str(res.violations))

    bad_cites = good_draft(s.blocking)
    bad_cites.options[1].citations = ["disruption_care:SQ-CANCEL-INVENTED-01"]
    res = s.compile(bad_cites)
    check("an option citing a chunk that does not exist is rejected",
          not res.ok and any("do not resolve" in v for v in res.violations),
          str(res.violations))

    print("\n=== ordering gates are mechanical, not a judgement ===")
    # tripsure's first full run deadlocked here. Its options-finder was told "read the
    # file; if it does not exist return BLOCKED", had to judge existence itself, and
    # returned BLOCKED while the file sat there having passed its citation audit. The
    # fix is that the tool returns a definite answer, so there is nothing to judge.
    entitlements = WORKSPACE_DIR / "analysis" / "entitlements.md"
    entitlements.parent.mkdir(parents=True, exist_ok=True)
    stashed = entitlements.read_text("utf-8") if entitlements.exists() else None
    try:
        entitlements.unlink(missing_ok=True)
        check("a missing entitlements file returns NOT YET DETERMINED",
              read_entitlements.invoke({}).startswith("NOT YET DETERMINED"))

        entitlements.write_text("", encoding="utf-8")
        check("an empty entitlements file also returns NOT YET DETERMINED",
              read_entitlements.invoke({}).startswith("NOT YET DETERMINED"),
              "an empty file is the deadlock case — present, but useless")

        entitlements.write_text(good, encoding="utf-8")
        answer = read_entitlements.invoke({})
        check("a written entitlements file returns Proceed plus its contents",
              answer.startswith("ENTITLEMENTS DETERMINED. Proceed.")
              and "SQ-CANCEL-ACCOM-01" in answer,
              answer[:80])
    finally:
        if stashed is None:
            entitlements.unlink(missing_ok=True)
        else:
            entitlements.write_text(stashed, encoding="utf-8")

    print("\n=== wiring: the new specialists ===")
    try:
        check_tool_wiring()
        check("tool wiring invariants still hold", True)
    except AssertionError as exc:
        check("tool wiring invariants still hold", False, str(exc))

    names = {sa["name"] for sa in SUBAGENTS}
    check("all four specialists plus the guard are registered",
          names == {"policy-checker", "impact-analyst", "options-finder", "critic",
                    "general-purpose"},
          str(sorted(names)))
    check("the orchestrator prompt carries the grounding verbatim",
          "G11 —" in orchestrator_prompt(),
          "the invariants are inlined, so a missing one is silent")


def main() -> int:
    approval_binding()
    grounding()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
