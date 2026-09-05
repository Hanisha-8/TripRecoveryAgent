"""The deterministic gate.

The deep agent proposes; this module decides. It takes the agent's `PlanDraft`
and either compiles it into a version-bound `RecoveryPlan` or rejects it with a
list of violations the orchestrator must fix and resubmit.

Why the checks live here and not in the prompt: a prompt rule is a request, and
the model satisfies it in prose. "Both options resolve every identified impact"
is not a style guide — it is arithmetic on the dependency graph, and it either
holds or it does not. Every rule below is checked against recomputed facts, never
against what the draft claims about itself.

Rules, in the order they run:

  C1  exactly two options
  C2  option ids are exactly A and B
  C3  action ids are unique across the plan
  C4  every proposed flight exists in the provider's result set, on the stated day
  C7  capability classes are correct (money and third parties are never agent-safe)
  C8  outward-facing agent-safe actions carry verbatim approved text
  C9  performing the option's actions leaves a VALID itinerary (by simulation)
  C10 every citation resolves to a real policy chunk

There is no C5 or C6. Both began as rejections and became canonicalisation, on the
same principle: **a check the model can fail should only exist where there is a
judgement we cannot make ourselves.** C5 was "the stated arrival delay matches the
flight's arrival" — that is arithmetic on two provider timestamps. C6 was "an
uncertain option carries a hedging phrase" — that is a fixed sentence gated on a
provider flag. Both are now computed and written in, so they are guaranteed rather
than usually present. What survives as a rule is identity (C1-C4, C10) and
judgement (C7-C9): which flight, what grounds it, who may act, and whether the
trip is actually recovered.

Everything provenance-bearing on a recommended flight — carrier, arrival, price,
stops, source — is rewritten from the provider record. The model chooses WHICH
flight; it does not get to restate what that flight is.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from core.citations import unresolved_citation_ids
from core.clock import now
from core.dependency_graph import build_dependency_graph, find_node_by_booking
from core.impact import validate_itinerary
from core.itinerary_ops import ActionApplyError, apply_actions
from models import (
    ALWAYS_HUMAN,
    OUTWARD_FACING,
    Action,
    ActionClass,
    FlightChoice,
    FlightRef,
    ImpactSummary,
    Itinerary,
    OptionDraft,
    PlanDraft,
    PlanOption,
    RecoveryPlan,
    TripNodeKind,
    VerifiedDisruption,
    compute_plan_version,
    itinerary_content_hash,
)

HEDGE_PHRASE = "verify at booking"
#: How far the stated arrival delay may differ from the computed one. A stated
#: delay is a headline number a traveller decides on, so the tolerance is tight.
DELAY_TOLERANCE_MINUTES = 15


class CompileResult:
    """Either a compiled plan or the reasons there isn't one."""

    def __init__(self, plan: RecoveryPlan | None, violations: list[str]) -> None:
        self.plan = plan
        self.violations = violations

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.violations

    def report(self) -> str:
        """Feedback text to hand back to the orchestrator on rejection."""
        if self.ok:
            return "PLAN ACCEPTED"
        lines = "\n".join(f"- {v}" for v in self.violations)
        return (
            "PLAN REJECTED — fix every item below and resubmit the full plan.\n"
            f"{lines}"
        )


def itinerary_version(itinerary: Itinerary) -> str:
    """Content hash of the itinerary's schedule. See `itinerary_content_hash`."""
    return itinerary_content_hash(itinerary.bookings)


def _norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _designators(carrier: str, number: str) -> set[str]:
    """Every reasonable spelling of one flight's identity.

    Models write the carrier inconsistently — "SQ", "Singapore Airlines", or
    nothing at all with the code folded into the number. All three name the same
    flight, and rejecting a correct recommendation over spelling would make the
    gate an obstacle instead of a control. The departure instant does the real
    disambiguating; this only has to survive naming variance.
    """
    c, n = _norm(carrier), _norm(number)
    forms = {n, c + n}
    if n.startswith(c) and len(n) > len(c):
        forms.add(n[len(c):])
    return {f for f in forms if f}


def _match_candidate(
    choice: FlightChoice, candidates: list[dict]
) -> tuple[dict | None, str | None]:
    """Find the provider record the option points at.

    Identity is the flight designator plus the departure DATE. A flight number
    flies once a day, so that pair is unique, and the clock time is an attribute
    of the flight rather than part of its name — which is why `FlightChoice` has
    no way to state one.

    The date is NOT forgiven the same way: flying the next morning instead of the
    same evening is the whole difference between Option A and Option B.
    """
    try:
        wanted = date.fromisoformat(choice.departure_date.strip()[:10])
    except ValueError:
        return None, (
            f"departure_date {choice.departure_date!r} is not a YYYY-MM-DD date"
        )

    proposed = _designators(choice.carrier, choice.flight_number)
    same_designator: list[dict] = []

    for candidate in candidates:
        if not (proposed & _designators(candidate["carrier"], candidate["flight_number"])):
            continue
        same_designator.append(candidate)
        if datetime.fromisoformat(candidate["departure"]).date() == wanted:
            return candidate, None

    if same_designator:
        dates = ", ".join(
            datetime.fromisoformat(c["departure"]).date().isoformat()
            for c in same_designator
        )
        return None, (
            f"{choice.carrier}{choice.flight_number} does not depart on "
            f"{wanted.isoformat()} — it departs {dates}. Flying on a different day "
            f"is a different option, not a typo"
        )
    return None, None


def _canonical_flight(candidate: dict) -> FlightRef:
    """Rebuild the flight from the provider record, discarding the restatement.

    This is stronger than checking the model's numbers and telling it off: the
    model chooses WHICH flight, and every fact about that flight — arrival, price,
    stops, provenance — comes from the provider. A misquoted fare cannot reach a
    traveller, because the misquote is simply overwritten.
    """
    return FlightRef(
        carrier=candidate["carrier"],
        flight_number=candidate["flight_number"],
        origin_iata=candidate["origin_iata"],
        destination_iata=candidate["destination_iata"],
        departure=datetime.fromisoformat(candidate["departure"]),
        arrival=datetime.fromisoformat(candidate["arrival"]),
        stops=int(candidate.get("stops", 0)),
        price_amount=candidate.get("price_amount"),
        price_currency=candidate.get("price_currency"),
        source=candidate.get("source", ""),
        fetched_at=datetime.fromisoformat(candidate["fetched_at"]),
        uncertain=bool(candidate.get("uncertain", False)),
        fallback_used=bool(candidate.get("fallback_used", False)),
    )


def _check_citations(option: PlanOption) -> list[str]:
    """C10 — the option's grounding actually exists.

    Only resolvability is checked here. Whether a chunk *says* what the option
    claims is a judgement, and it belongs to the critic's independent
    re-retrieval — a resolvable citation of the wrong chunk is precisely the
    failure a mechanical check cannot see.
    """
    tag = f"option {option.option_id}"
    if not option.citations:
        return [
            f"{tag}: no citations. Every option rests on an entitlement — at minimum "
            f"the rebooking right that makes the replacement free or paid — and that "
            f"entitlement's chunk id belongs here"
        ]
    unresolved = unresolved_citation_ids(option.citations)
    if unresolved:
        return [
            f"{tag}: citations {unresolved} do not resolve to any chunk in the policy "
            f"corpus. Cite ids you actually retrieved, or state "
            f"'NOT COVERED BY POLICY CORPUS'"
        ]
    return []


def _check_actions(option: PlanOption, seen_ids: set[str]) -> list[str]:
    """C3, C7, C8 — action identity and capability classification."""
    problems: list[str] = []
    for action in option.actions:
        tag = f"option {option.option_id} action {action.action_id!r}"

        if action.action_id in seen_ids:
            problems.append(f"{tag}: duplicate action_id")
        seen_ids.add(action.action_id)

        if action.kind in ALWAYS_HUMAN and action.action_class is not ActionClass.HUMAN_REQUIRED:
            problems.append(
                f"{tag}: {action.kind.value} moves money or needs provider "
                f"confirmation, so it cannot be {action.action_class.value}"
            )

        if action.requires_payment and action.action_class is not ActionClass.HUMAN_REQUIRED:
            problems.append(
                f"{tag}: requires_payment is true, so action_class must be human_required"
            )

        if (
            action.kind in OUTWARD_FACING
            and action.action_class is ActionClass.AGENT_SAFE
            and not (action.message_body or "").strip()
        ):
            problems.append(
                f"{tag}: {action.kind.value} reaches a third party and cannot be "
                f"unsent, so agent_safe requires the exact message_body to approve"
            )

    return problems


def _resolve_flight(
    option: OptionDraft, candidates: list[dict], original_arrival: datetime | None
) -> tuple[PlanOption | None, list[str]]:
    """C4 — compile an `OptionDraft` into a `PlanOption`, or say why not.

    C4 (the flight was really retrieved, on the stated day) is the only thing the
    model can fail here. Everything else this function does is *construction*:
    the flight comes from the provider record, the arrival delay is arithmetic on
    two provider timestamps, and the hedging phrase is appended when a provider
    flag says the data was a fallback.

    Returning None means C4 failed and there is nothing to compile.
    """
    problems: list[str] = []
    tag = f"option {option.option_id}"

    candidate, reason = _match_candidate(option.flight, candidates)
    if candidate is None:
        problems.append(
            f"{tag}: " + (reason or (
                f"flight {option.flight.carrier} {option.flight.flight_number} is not "
                f"in the provider's results — it cannot be recommended. Recommend one "
                f"of: " + ", ".join(
                    f"{c['carrier']}{c['flight_number']} dep {c['departure']}"
                    for c in candidates
                )
            ))
        )
        return None, problems

    flight = _canonical_flight(candidate)

    # Derived, not requested — arithmetic on two provider timestamps. Asking the
    # model created only a way to be wrong, and a rejection over 375-vs-420 cost a
    # full extra round trip to fix a number we compute exactly.
    delay = (
        max(0, int((flight.arrival - original_arrival).total_seconds() // 60))
        if original_arrival is not None else 0
    )

    # Guaranteed, not requested — same reasoning. This was the one rule gpt-4.1
    # would not converge on: it kept the hedge in `/drafts/options.md` and dropped
    # it when transcribing structured output, twice, even with the exact phrase
    # quoted back in the rejection. Whether to recommend uncertain data at all is
    # a judgement; appending a fixed sentence when a provider flag is set is
    # bookkeeping, and doing it here makes the hedge certain rather than usual.
    rationale = option.rationale.rstrip()
    if (flight.uncertain or flight.fallback_used) and HEDGE_PHRASE not in rationale.lower():
        joiner = " " if rationale.endswith((".", "!", "?")) else ". "
        rationale = (
            f"{rationale}{joiner}Availability and fare came from "
            f"{flight.source}, so {HEDGE_PHRASE}."
        )

    if not flight.source.strip():
        problems.append(f"{tag}: provider record has no `source` — cannot be cited")

    resolved = PlanOption(
        **option.model_dump(exclude={"flight", "rationale"}),
        flight=flight,
        rationale=rationale,
        arrival_delay_minutes=delay,
    )
    return resolved, problems


def _check_coverage(
    option: PlanOption,
    itinerary: Itinerary,
    blocking: list[str],
    node_to_booking: dict[str, str],
) -> list[str]:
    """C9 — simulate the option and require the trip to come out valid.

    This began as "every still-blocking booking has an action targeting it",
    which is weaker than it sounds: an action naming the right booking and moving
    it to a useless time passed. Now the option's actions are actually applied
    (via `core.itinerary_ops`, the same code the executor runs) and the result is
    put through the same validator that closes the case in R4.

    So "both options resolve every identified impact" stops being a claim to
    check and becomes a thing to do. An option is feasible when performing it
    leaves a trip that holds together.
    """
    problems: list[str] = []
    tag = f"option {option.option_id}"

    unknown = [n for n in option.resolves if n not in node_to_booking]
    if unknown:
        problems.append(f"{tag}: `resolves` names unknown node ids {unknown}")

    missing_claim = [n for n in blocking if n not in option.resolves]
    if missing_claim:
        problems.append(
            f"{tag}: does not claim to resolve blocking impacts {missing_claim} — "
            f"both options must resolve every identified impact"
        )

    try:
        after = apply_actions(itinerary, option)
    except ActionApplyError as exc:
        return problems + [f"{tag}: {exc}"]

    result = validate_itinerary(itinerary.trip_id, after.bookings)
    if not result.valid:
        for node in result.remaining:
            problems.append(
                f"{tag}: performing every action still leaves "
                f"{node.label or node.node_id} broken ({node.reason}). Either move "
                f"that booking to a workable time or cancel it"
            )

    # A booking the disruption broke that no action touches is worth naming even
    # when the graph happens to validate, because the usual cause is an option
    # quietly abandoning something the traveller paid for. Cancelling the Night
    # Safari is a legitimate answer; saying nothing about it is not.
    touched = {a.target_booking_id for a in option.actions}
    broken_bookings = {
        node_to_booking[n] for n in blocking if n in node_to_booking
    }
    for booking_id in sorted(broken_bookings - touched):
        problems.append(
            f"{tag}: the disruption breaks {booking_id!r} and no action touches it — "
            f"move it, cancel it, or say explicitly what happens to it"
        )

    return problems


def compile_plan(
    draft: PlanDraft,
    *,
    itinerary: Itinerary,
    verified_disruption: VerifiedDisruption,
    impact: ImpactSummary,
    flight_candidates: list[dict],
) -> CompileResult:
    """Validate `draft` against recomputed facts and bind it to a version.

    `impact` and `flight_candidates` come from the deterministic core, not from
    the agent. That asymmetry is the safety property: the agent chose what to
    investigate, and the numbers it is checked against are ours.
    """
    violations: list[str] = []

    # C1, C2 — the two-option contract
    if len(draft.options) != 2:
        violations.append(
            f"expected exactly 2 options, got {len(draft.options)} — Option A is the "
            f"best overall recovery, Option B a meaningful alternative"
        )
    ids = [o.option_id for o in draft.options]
    if sorted(ids) != ["A", "B"]:
        violations.append(f"option ids must be exactly A and B, got {ids}")

    graph = build_dependency_graph(itinerary.bookings)
    node_to_booking = {n.node_id: n.booking_id for n in graph.nodes}
    blocking = impact.blocking_node_ids()

    original = find_node_by_booking(
        graph, verified_disruption.booking_id, kind=TripNodeKind.FLIGHT
    )
    original_arrival = original.scheduled_at if original else None

    seen_action_ids: set[str] = set()
    resolved_options: list[PlanOption] = []
    for option in draft.options:
        resolved, problems = _resolve_flight(option, flight_candidates, original_arrival)
        violations += problems
        if resolved is None:
            # No provider record, so there is nothing to check the rest against.
            # Reporting coverage against a flight that does not exist would add a
            # confusing second violation about a booking when the real problem is
            # the flight.
            continue
        violations += _check_actions(resolved, seen_action_ids)
        violations += _check_citations(resolved)
        # Coverage is checked against the CANONICAL arrival, which the model now
        # has no way to misstate — `FlightChoice` carries no times at all.
        violations += _check_coverage(resolved, itinerary, blocking, node_to_booking)
        resolved_options.append(resolved)

    if violations:
        return CompileResult(None, violations)

    iv = itinerary_version(itinerary)
    # Impacts on the compiled plan are OURS, not the draft's, and the options carry
    # provider-truthful flights. The draft contributes judgement — which options,
    # what trade-off, which actions — and nothing that can be independently checked.
    plan = RecoveryPlan(
        trip_id=itinerary.trip_id,
        itinerary_version=iv,
        plan_version=compute_plan_version(
            trip_id=itinerary.trip_id,
            itinerary_version=iv,
            disruption_summary=draft.disruption_summary,
            impacts=impact.impacted,
            options=resolved_options,
        ),
        verified_disruption=verified_disruption,
        disruption_summary=draft.disruption_summary,
        impacts=impact.impacted,
        options=resolved_options,
        generated_at=now(),
    )
    return CompileResult(plan, [])


def agent_safe_actions(option: PlanOption) -> list[Action]:
    """Actions the executor may perform without a human. Used in R3."""
    return [a for a in option.actions if a.action_class is ActionClass.AGENT_SAFE]


def human_actions(option: PlanOption) -> list[Action]:
    """Actions that become handoffs. Used in R3."""
    return [a for a in option.actions if a.action_class is ActionClass.HUMAN_REQUIRED]
