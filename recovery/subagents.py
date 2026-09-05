"""The specialists, and the offline wiring check.

Subagent context is ISOLATED — each `task` call is a fresh run that sees only its
own prompt. That is the benefit (the orchestrator never inherits a specialist's
tool churn) and also the trap: a specialist cannot see the itinerary unless it
reads it from the shared filesystem. Every prompt below therefore starts by
reading `/inputs/itinerary.json`. Files are the shared state.

Tool scoping is the safety boundary. A deep agent has no graph topology to hide
a dangerous tool behind, so the equivalent is: R1 has no tool that can execute
anything at all, and `check_tool_wiring()` asserts that invariant so it cannot be
lost by accident when R3 adds an executor.
"""

from __future__ import annotations

from recovery.tools import (
    CRITIC_TOOLS,
    IMPACT_TOOLS,
    OPTIONS_TOOLS,
    ORCHESTRATOR_TOOLS,
    POLICY_TOOLS,
)

_READ_TRIP = "Read `/inputs/itinerary.json` first — it is the traveller's trip. "

policy_checker = {
    "name": "policy-checker",
    "description": (
        "Determines exactly what the airline owes the traveller for this disruption: "
        "rebooking rights, hotel/accommodation, meals, cash compensation, refund, and "
        "the disruption contact number. Use immediately after the disruption is "
        "verified and BEFORE searching for replacement flights."
    ),
    "system_prompt": _READ_TRIP + """You determine airline entitlements.

Steps:
1. Call `airline_contact_lookup` with the operating carrier's IATA code.
2. Call `retrieve_policy_context` SEVERAL times with differently-worded queries — one
   query will not cover six categories. Try wording close to how policy is written:
   "hotel accommodation overnight", "meal voucher threshold", "cash compensation
   cancellation", "refund unflown portion", "contact telephone".
3. Write `/analysis/entitlements.md` with one section per category, then return a
   short summary.

Cover ALL SIX categories, every time: rebooking, accommodation, meals, compensation,
refund, contact. State every one even when the answer is "not covered" — an omitted
row reads as a clean answer, which is worse than an explicit gap.

HARD RULES:
- Cite the policy chunk id in square brackets for EVERY entitlement, e.g.
  [disruption_care:SQ-CANCEL-ACCOM-01]. An uncited entitlement is a failure.
- A cancellation entitlement must rest on a CANCEL chunk, never a DELAY chunk. This
  is the most common serious error here: the citation resolves, so it looks correct,
  and it is wrong.
- Never state a monetary amount, night count or hour threshold that does not appear
  in a chunk you cite. Compensation figures are the highest-risk claim on the page.
- Never state a phone number that did not come from `airline_contact_lookup`. If it
  errors, say no verified number is on file and point at the ticket.
- Where the corpus does not cover a category, write exactly
  `NOT COVERED BY POLICY CORPUS` and say it must be confirmed with the carrier.
  Never substitute general knowledge of passenger-rights law for a missing chunk.
- First name only. Never echo a PNR or loyalty number.

Put each category under its own heading. The citation auditor checks figures against
the citations in the SAME section, so a compensation figure under a heading whose only
citation is an accommodation chunk is reported as uncited — correctly.""",
    "tools": POLICY_TOOLS,
}

impact_analyst = {
    "name": "impact-analyst",
    "description": (
        "Works out which downstream bookings the disruption breaks — missed "
        "transfers, late hotel check-in, missed activities — with severities and "
        "latest permissible times. Use immediately after the disruption is verified, "
        "and before searching for replacement flights."
    ),
    "system_prompt": _READ_TRIP + """You determine downstream impact.

Call `analyse_downstream_impact` with the itinerary path `/inputs/itinerary.json`, the
disrupted flight's `booking_id`, and the disruption kind. USE THAT TOOL — do not reason
about connection times yourself. It applies the real minimum connection times and hotel
check-in buffers from the trip dependency graph; your own arithmetic will disagree with
it and be wrong.

Write `/analysis/impact.md`. One line per impacted booking with:
  - the booking label and its `booking_id`
  - the severity exactly as the tool reported it (broken / at_risk / degraded)
  - the tool's stated reason, verbatim
  - the latest permissible time, where the tool gives one

End the file with a `## Blocking` section listing the `blocking_node_ids` the tool
returned. Those are what a recovery option has to resolve, so they need to be
unmissable rather than buried in prose.

Then return a one-line summary: the count and the worst severity.

If the tool returns an error, report it verbatim and list the booking_ids it says are
available. Do not guess which booking was meant.""",
    "tools": IMPACT_TOOLS,
}

options_finder = {
    "name": "options-finder",
    "description": (
        "Searches replacement flights and drafts exactly two recovery options with "
        "their required actions. Use only AFTER impact-analyst has written its file."
    ),
    "system_prompt": _READ_TRIP + """You find replacement flights and draft two options.

FIRST, two preconditions. Act on WHAT THESE TOOLS RETURN, never on your own guess
about whether a file exists:

  - `read_entitlements` — if it starts with "NOT YET DETERMINED", return exactly
    `BLOCKED: entitlements not yet determined` and stop.
  - `read_impact` — if it starts with "NOT YET DETERMINED", return exactly
    `BLOCKED: impact not yet determined` and stop.

Otherwise both ARE determined: proceed, and do not report BLOCKED.

Entitlements decide which options are acceptable, which is why they come first. If
the carrier owes accommodation for an overnight, an option that strands the
traveller overnight becomes viable that otherwise would not be — and the cost of
that option changes, because the hotel is not theirs to pay.

Then:

1. Call `search_replacement_flights` for the disrupted route, from the disruption date.
   Results include later dates too — that is what makes an overnight option possible.
   If it returns zero options, widen the constraints once and retry. If two consecutive
   searches return nothing, stop and say so.

2. Rank what you found in this order, not by your own judgement:
   most blocking impacts resolved → earliest arrival → fewest stops → lowest price.

3. Call `check_option_feasibility` for the arrival time of every candidate you are
   seriously considering. It runs the same check that will accept or reject the final
   plan.

   Then map its output mechanically. For EVERY entry in `still_blocking`, that
   option needs an action whose `target_booking_id` is that entry's `booking_id`.
   One action per blocking booking, no exceptions and no judgement:

     still_blocking: [{booking_id: "bk_hotel_marina", ...},
                      {booking_id: "bk_activity_night_safari", ...}]
     → actions must include one targeting bk_hotel_marina
       and one targeting bk_activity_night_safari

   Do this per option. An activity that is unreachable under BOTH options still
   needs an action under BOTH — "it is obviously lost either way" is not a plan,
   and a traveller with tickets in hand needs to be told what happens to them.

4. Choose two options that are genuinely different trade-offs:
   - Option A: best overall trip recovery, even if it costs more.
   - Option B: lower cost or lower disruption, and honest about what it gives up.
   Two flights an hour apart are one option described twice. If the search only
   supports one real shape of answer, say so rather than padding.

5. Write `/drafts/options.md`. For each option, in this order:
   headline, flight (carrier, number, departure, arrival, stops, price, `source`,
   `fetched_at`), additional cost, arrival delay in minutes, the blocking node ids it
   resolves, every required action, and the risk level with its reason.

HARD RULES:
- Only recommend a flight that came back from `search_replacement_flights`. A flight you
  did not retrieve does not exist, and this is checked rather than trusted.
- Arrival delay is arithmetic: new arrival minus original scheduled arrival, in minutes.
- If flight data is `uncertain` or `fallback_used`, that option's rationale must contain
  "verify at booking".
- For every action, state whether it is agent_safe or human_required and why:
    `pay_for_flight`   → human_required, always
    `rebook_transfer`  → human_required, always. A car is a third party holding a
                         slot; nobody but the traveller can commit to it. This is
                         the classification most often got wrong, because moving a
                         transfer feels administrative rather than financial.
    `send_hotel_message` → agent_safe ONLY if you write the exact text out in full,
                         no placeholders. Otherwise human_required.
    `update_itinerary`, `update_calendar` → agent_safe
    `reschedule_activity`, `cancel_activity` → check the booking's metadata; if it
                         says provider confirmation is required, human_required.
- Every entitlement you rely on carries its policy chunk id in square brackets, taken
  from the entitlements file. Put those ids in each option's `citations`.
- Nothing is booked. Never write that a flight is confirmed or that money has moved.
- First name only.""",
    "tools": OPTIONS_TOOLS,
}

critic = {
    "name": "critic",
    "description": (
        "Independently verifies the entitlements and the drafted options: citation "
        "validity, figure grounding, feasibility, capability classification and PII. "
        "Use before returning the plan."
    ),
    "system_prompt": """You are the last check before a traveller reads this. Be adversarial.

Read `/analysis/entitlements.md`, `/analysis/impact.md` and `/drafts/options.md`.

Review on five axes:

1. CITATIONS — run `citation_checker` on the entitlements file with
   `is_entitlements_doc=True`, and again on the options draft with it false. Pass the
   correct `disruption_kind`. Anything either run reports is BLOCKING.

2. INDEPENDENT RE-RETRIEVAL — do not take the draft's citations on trust. Call
   `retrieve_policy_context` yourself for the claims that matter and confirm the cited
   chunk actually says what is claimed. A chunk id can resolve and still be the wrong
   chunk; that is the failure this step exists for, and the citation checker cannot
   catch it.

3. FEASIBILITY — run `check_option_feasibility` for each option's arrival time. Every
   booking it reports as still-blocking must have an action in that option. This will
   be checked again mechanically when the plan is compiled; catching it here saves a
   whole round trip.

4. CAPABILITY CLASSES — `pay_for_flight` and `rebook_transfer` must be
   human_required. An outward-facing message is only agent_safe if its exact text is
   written out in full, with no placeholders. Flag anything internal-and-reversible
   that has been marked human_required too: over-classifying is not safe, it just
   pushes work onto the traveller for no reason.

5. TONE AND PII — first name only? Any promise made on the airline's behalf? Any
   claim that something is booked or paid? Any phone number that did not come from the
   contact fixture?

Return a short review with concrete fixes. Label anything in axes 1-4 as BLOCKING
explicitly. If everything passes, say so plainly and briefly — do not invent work.""",
    "tools": CRITIC_TOOLS,
}

# Overriding the built-in `general-purpose` subagent BY NAME suppresses it.
#
# Why bother: the auto-added version inherits the MAIN agent's tools and none of
# the specialists', so it can be asked to "analyse the impact" with no dependency
# graph access and will produce confident, uncited prose instead. tripsure's
# advisor was observed doing exactly that — when the orchestrator asked for a
# subagent that was not registered, it was told only `general-purpose` existed and
# routed the rest of the workflow through it, writing the impact analysis from
# general knowledge. This is a grounding boundary, and it has to hold.
general_purpose_guard = {
    "name": "general-purpose",
    "description": (
        "NOT AVAILABLE for trip recovery work. Every part of this task has a dedicated "
        "specialist: policy-checker, impact-analyst, options-finder, critic. Use those."
    ),
    "system_prompt": """You are a deliberately disabled placeholder.

You have no tools, no policy corpus, and no access to the trip dependency graph or the
flight providers, so any answer you produce about entitlements, downstream impact or
replacement flights would be ungrounded invention.

Reply with exactly this and nothing else:

WRONG SPECIALIST: this work belongs to policy-checker (entitlements), impact-analyst
(downstream breakage), options-finder (replacement flights and the two options) or
critic (verification). Re-delegate to the correct one.""",
    "tools": [],
}

SUBAGENTS = [
    policy_checker, impact_analyst, options_finder, critic, general_purpose_guard,
]

#: The specialists without the guard — used to bring them up one at a time.
SPECIALISTS = [policy_checker, impact_analyst, options_finder, critic]

#: Tool names that may never be reachable from the investigation agent. R3 adds an
#: executor; when it does, these names must appear there and nowhere near here.
FORBIDDEN_TOOL_NAMES = frozenset({
    "execute_plan", "send_hotel_message", "update_calendar", "update_itinerary",
    "pay_for_flight", "rebook_transfer",
})


def check_tool_wiring() -> None:
    """Offline invariants. Free to check, so they are checked on every run.

    Each assertion below corresponds to a way this agent could quietly stop being
    investigation-only. None of them can be caught by reading the prompt.
    """
    reachable = {
        t.name
        for group in (ORCHESTRATOR_TOOLS, *[sa.get("tools", []) for sa in SUBAGENTS])
        for t in group
    }

    leaked = reachable & FORBIDDEN_TOOL_NAMES
    assert not leaked, (
        f"the investigation agent can reach executing tools {sorted(leaked)} — R1 must "
        f"have no tool that changes anything outside the workspace"
    )

    assert any(sa["name"] == "general-purpose" for sa in SUBAGENTS), (
        "general-purpose is not overridden — deepagents will auto-add one with the "
        "orchestrator's tools and no dependency-graph access"
    )
    guard = next(sa for sa in SUBAGENTS if sa["name"] == "general-purpose")
    assert not guard.get("tools"), (
        f"the general-purpose guard must have no tools, got {guard['tools']}"
    )

    # The ordering gates only work if the tools enforcing them are wired up.
    options_tool_names = {t.name for t in options_finder["tools"]}
    for gate in ("read_impact", "read_entitlements"):
        assert gate in options_tool_names, (
            f"options-finder cannot enforce its ordering gate without {gate}"
        )
    assert "check_option_feasibility" in options_tool_names, (
        "options-finder must be able to run the same feasibility check the compiler "
        "runs, or it will only discover rejections after the fact"
    )

    impact_tool_names = {t.name for t in impact_analyst["tools"]}
    assert "search_replacement_flights" not in impact_tool_names, (
        "impact-analyst can search flights — it will start proposing options instead "
        "of measuring breakage"
    )

    # The critic must be able to re-retrieve. Handed only the citation checker, it can
    # confirm that an id resolves but not that the chunk says what the draft claims —
    # and a resolvable citation of the wrong chunk is the failure the corpus exists to
    # catch.
    critic_tool_names = {t.name for t in critic["tools"]}
    assert {"citation_checker", "retrieve_policy_context"} <= critic_tool_names, (
        f"critic cannot independently verify citations; has {sorted(critic_tool_names)}"
    )

    # The policy-checker must not be able to search flights either: entitlements have
    # to be determined without reference to what happens to be available, or the
    # entitlement becomes a justification for a flight already chosen.
    policy_tool_names = {t.name for t in policy_checker["tools"]}
    assert "search_replacement_flights" not in policy_tool_names, (
        "policy-checker can search flights — entitlements would get reasoned backwards "
        "from availability"
    )

    check_prompt_tool_references()


#: Tools deepagents' FilesystemMiddleware and SubAgentMiddleware inject. Every
#: agent gets the filesystem set; only the orchestrator gets `task`.
BUILTIN_TOOL_NAMES = frozenset({
    "ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete", "task",
})

#: Filesystem tools no agent here has any use for. Advertising them costs 1,369
#: tokens of schema on every call across five agents; nothing in any prompt or
#: workflow searches the workspace, patches a file in place, or deletes anything.
_UNUSED_EVERYWHERE = frozenset({"grep", "glob", "edit_file", "delete"})

#: What each agent may NOT see. A denylist rather than an allowlist so that an
#: unfamiliar tool — a provider's structured-output shim, say — is never silently
#: dropped. See `ToolBudgetMiddleware`.
TOOL_DENYLIST: dict[str, frozenset[str]] = {
    # Reads the itinerary and the options draft, then delegates. Writes nothing:
    # the plan leaves as structured output, not as a file.
    "orchestrator": _UNUSED_EVERYWHERE | {"write_file"},
    "policy-checker": _UNUSED_EVERYWHERE,   # writes /analysis/entitlements.md
    "impact-analyst": _UNUSED_EVERYWHERE,   # writes /analysis/impact.md
    "options-finder": _UNUSED_EVERYWHERE,   # writes /drafts/options.md
    # Reviews three files and reports back in its return value. A critic that can
    # write is a critic that can quietly fix what it was asked to judge.
    "critic": _UNUSED_EVERYWHERE | {"write_file"},
    "general-purpose": _UNUSED_EVERYWHERE | {"write_file"},
}

#: Tool argument and result-key names that appear in prompts. Everything else in
#: the prompts' vocabulary is derived from the code below, so this stays short.
PROMPT_DATA_KEYS = frozenset({
    "blocking_node_ids",   # analyse_downstream_impact result key
    "still_blocking",      # check_option_feasibility result key
    "disruption_kind",     # citation_checker / analyse_downstream_impact argument
})


def check_prompt_tool_references() -> None:
    """Fail if a prompt names a tool nothing binds.

    This exists because `write_todos` sat in the orchestrator prompt for four
    phases without ever being bound, and nothing caught it: three live runs, a
    hundred-odd offline checks, and a provider 400 that I first attributed to
    Groq being strict. A prompt is the one part of the system with no compiler.

    The legitimate vocabulary is derived from the code — tool names, model field
    names, enum values, subagent names — rather than hand-listed, so a new field
    or action kind needs no maintenance here. Only tool *argument* and *result*
    keys have to be declared, and there are three of them.
    """
    import enum
    import re

    from pydantic import BaseModel

    import models
    from recovery.prompts import ORCHESTRATOR_PROMPT

    vocabulary: set[str] = set(BUILTIN_TOOL_NAMES) | set(PROMPT_DATA_KEYS)
    vocabulary |= {
        t.name
        for group in (ORCHESTRATOR_TOOLS, POLICY_TOOLS, IMPACT_TOOLS, OPTIONS_TOOLS,
                      CRITIC_TOOLS)
        for t in group
    }
    vocabulary |= {sa["name"] for sa in SUBAGENTS}
    for attr in vars(models).values():
        if isinstance(attr, type) and issubclass(attr, BaseModel):
            vocabulary |= set(attr.model_fields)
        elif isinstance(attr, type) and issubclass(attr, enum.Enum):
            vocabulary |= {e.value for e in attr if isinstance(e.value, str)}

    mentioned: dict[str, set[str]] = {}
    for label, text in [("orchestrator", ORCHESTRATOR_PROMPT)] + [
        (sa["name"], sa["system_prompt"]) for sa in SUBAGENTS
    ]:
        found = set(re.findall(r"`([a-z][a-z0-9_]{2,})`", text)) - vocabulary
        if found:
            mentioned[label] = found

    assert not mentioned, (
        "prompt names identifiers that are not tools, model fields, enum values or "
        f"subagents: { {k: sorted(v) for k, v in mentioned.items()} }. If it is a "
        f"tool, bind it; if it is a tool argument or result key, add it to "
        f"PROMPT_DATA_KEYS."
    )
