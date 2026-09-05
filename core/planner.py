"""Deterministic plan builder — two options and their actions, with no model call.

This is C2 of the token plan brought forward, because the demo needed it: a
hand-authored draft only fits the trip it was written for, so "upload your own
itinerary" is meaningless without a planner that works on any trip.

**What it derives, and why that is the right split.** Given a chosen arrival
time and the dependency graph, the required actions and their new times are
arithmetic: the transfer must meet the arrival, the hotel check-in must follow
the transfer, the activity must fall in the next slot the traveller can actually
reach. The live runs showed a model getting exactly these wrong, repeatedly —
that is what C9-by-simulation kept catching. So they are computed here.

What is left as judgement — which flight is the *better* trade-off, whether an
activity is worth moving or writing off — is decided by explicit, stated rules
below rather than by a model. That makes this planner honest but not clever: it
ranks by whole-trip recovery and picks a genuinely different alternative, and it
will say so plainly when there is only one shape of answer.

The output is a `PlanDraft`, so it goes through exactly the same
`compile_plan` gate as anything a model produces. Nothing here is trusted.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from core.dependency_graph import DEFAULT_BUFFER_MINUTES, build_dependency_graph
from core.impact import impact_of_replacement
from core.sources import Sources, sources
from models import (
    Action,
    ActionClass,
    ActionKind,
    Booking,
    BookingType,
    FlightChoice,
    ImpactSummary,
    Itinerary,
    OptionDraft,
    PlanDraft,
    TripNodeKind,
    VerifiedDisruption,
)

FLIGHT_TO_TRANSFER = DEFAULT_BUFFER_MINUTES[
    (TripNodeKind.FLIGHT, TripNodeKind.TRANSFER)
]
FLIGHT_TO_HOTEL = DEFAULT_BUFFER_MINUTES[
    (TripNodeKind.FLIGHT, TripNodeKind.HOTEL_CHECK_IN)
]
FLIGHT_TO_ACTIVITY = DEFAULT_BUFFER_MINUTES[
    (TripNodeKind.FLIGHT, TripNodeKind.ACTIVITY)
]
#: Slack added on top of a minimum buffer. A plan that clears the minimum by
#: zero minutes is technically feasible and reads as reckless, and the validator
#: flags it as a tight connection — so leave room deliberately.
COMFORT_MINUTES = 30


class NoRecovery(Exception):
    """No option can be built — usually because no replacement flight exists."""


# ===============================================================
# Choosing the two flights
# ===============================================================
def _residual_count(itinerary: Itinerary, booking_id: str, arrival: datetime) -> int:
    residual = impact_of_replacement(itinerary.bookings, booking_id, arrival)
    return sum(1 for i in residual.impacted if i.severity in ("broken", "at_risk"))


def choose_flights(
    itinerary: Itinerary, disruption: VerifiedDisruption, candidates: list[dict]
) -> tuple[dict, dict | None]:
    """Pick Option A's flight and a meaningfully different Option B.

    A is ranked the way the product promises: most of the trip preserved first,
    then earliest arrival, then fewest stops, then lowest price.

    B has to be a different *shape* of answer, not the runner-up by a few
    minutes. Two flights an hour apart are one option described twice, so B is
    chosen as the cheapest candidate that either leaves on a different day or
    undercuts A materially. If nothing qualifies, B is None and the caller has to
    say there is only one real option rather than padding the list.
    """
    if not candidates:
        raise NoRecovery("no replacement flights on this route")

    def rank(candidate: dict) -> tuple:
        arrival = datetime.fromisoformat(candidate["arrival"])
        return (
            _residual_count(itinerary, disruption.booking_id, arrival),
            arrival,
            int(candidate.get("stops", 0)),
            float(candidate.get("price_amount") or 0.0),
        )

    ordered = sorted(candidates, key=rank)
    best = ordered[0]
    best_day = datetime.fromisoformat(best["departure"]).date()
    best_price = float(best.get("price_amount") or 0.0)

    alternatives = [
        c for c in ordered[1:]
        if datetime.fromisoformat(c["departure"]).date() != best_day
        or float(c.get("price_amount") or 0.0) <= best_price * 0.75
    ]
    alternatives.sort(key=lambda c: float(c.get("price_amount") or 0.0))
    return best, (alternatives[0] if alternatives else None)


# ===============================================================
# Deriving the actions
# ===============================================================
def _booking(itinerary: Itinerary, booking_id: str) -> Booking | None:
    return next((b for b in itinerary.bookings if b.booking_id == booking_id), None)


def _hotel_message(itinerary: Itinerary, check_in: datetime) -> str:
    """The exact text that would be sent, composed once and hashed into the plan.

    Written out in full deliberately: an outward-facing action is only
    agent-safe when the approved bytes are the sent bytes, so a placeholder here
    would make the action human-required instead.
    """
    destination = next(
        (
            booking.legs[0].destination_iata
            for booking in itinerary.bookings
            if booking.type is BookingType.FLIGHT and booking.legs
        ),
        "our destination",
    )
    return (
        f"Hello, our flight to {destination} was cancelled and we have rebooked. "
        f"We now expect to reach the hotel at about "
        f"{check_in.strftime('%H:%M')} on {check_in.strftime('%d %B')}. "
        f"Please hold the reservation for a late arrival. Thank you."
    )


def derive_actions(
    itinerary: Itinerary,
    disruption: VerifiedDisruption,
    arrival: datetime,
    *,
    prefix: str,
) -> list[Action]:
    """Compute the actions that make `arrival` work, in dependency order.

    Order matters for readability rather than correctness — every action writes
    an absolute time to a distinct booking, so applying them is
    order-independent (see `core.itinerary_ops`).
    """
    actions: list[Action] = [
        Action(
            action_id=f"{prefix}-pay",
            kind=ActionKind.PAY_FOR_FLIGHT,
            action_class=ActionClass.HUMAN_REQUIRED,
            target_booking_id=disruption.booking_id,
            description="Complete payment for the replacement fare.",
            requires_payment=True,
        )
    ]

    residual = impact_of_replacement(itinerary.bookings, disruption.booking_id, arrival)
    broken = {
        i.booking_id for i in residual.impacted
        if i.severity in ("broken", "at_risk")
    }

    # --- the transfer meets the arrival --------------------------------
    transfer_end: datetime | None = None
    for booking in itinerary.bookings:
        if booking.type is not BookingType.TRANSFER or booking.booking_id not in broken:
            continue
        start = arrival + timedelta(minutes=FLIGHT_TO_TRANSFER + COMFORT_MINUTES)
        end = start + (booking.end - booking.start)
        transfer_end = end
        actions.append(Action(
            action_id=f"{prefix}-transfer",
            kind=ActionKind.REBOOK_TRANSFER,
            action_class=ActionClass.HUMAN_REQUIRED,
            target_booking_id=booking.booking_id,
            description="Move the airport transfer to the new arrival time.",
            new_start=start, new_end=end,
        ))

    # --- the hotel holds the room --------------------------------------
    for booking in itinerary.bookings:
        if booking.type is not BookingType.HOTEL or booking.booking_id not in broken:
            continue
        earliest = arrival + timedelta(minutes=FLIGHT_TO_HOTEL)
        check_in = max(earliest, transfer_end) if transfer_end else earliest
        actions.append(Action(
            action_id=f"{prefix}-hotel",
            kind=ActionKind.SEND_HOTEL_MESSAGE,
            action_class=ActionClass.AGENT_SAFE,
            target_booking_id=booking.booking_id,
            description="Tell the hotel to hold the room for a late arrival.",
            message_body=_hotel_message(itinerary, check_in),
            # Only check-in moves. Pushing check-out too would quietly turn a
            # three-night stay into four.
            new_start=check_in,
        ))

    # --- activities move, or are written off ---------------------------
    for booking in itinerary.bookings:
        if booking.type is not BookingType.ACTIVITY or booking.booking_id not in broken:
            continue
        slot = _next_activity_slot(itinerary, booking, arrival)
        if slot is None:
            actions.append(Action(
                action_id=f"{prefix}-{_slug(booking.booking_id)}",
                kind=ActionKind.CANCEL_ACTIVITY,
                action_class=ActionClass.HUMAN_REQUIRED
                if booking.metadata.get("reschedule_requires_provider_confirmation")
                else ActionClass.AGENT_SAFE,
                target_booking_id=booking.booking_id,
                description=(
                    f"Cancel {booking.title} — no reachable slot remains before "
                    f"the trip ends."
                ),
            ))
            continue
        start, end = slot
        actions.append(Action(
            action_id=f"{prefix}-{_slug(booking.booking_id)}",
            kind=ActionKind.RESCHEDULE_ACTIVITY,
            action_class=ActionClass.HUMAN_REQUIRED
            if booking.metadata.get("reschedule_requires_provider_confirmation")
            else ActionClass.AGENT_SAFE,
            target_booking_id=booking.booking_id,
            description=(
                f"Move {booking.title} to {start.strftime('%a %d %b, %H:%M')}."
            ),
            new_start=start, new_end=end,
        ))

    return actions


def _slug(booking_id: str) -> str:
    return booking_id.replace("bk_", "").replace("_", "-")[:24]


def _next_activity_slot(
    itinerary: Itinerary, activity: Booking, arrival: datetime
) -> tuple[datetime, datetime] | None:
    """The next day, at the same clock time, that the traveller can reach.

    Keeping the original time of day is the point: a Night Safari booked for the
    evening is an evening activity, and moving it to breakfast would be feasible
    on the graph and useless to the traveller.
    """
    earliest = arrival + timedelta(minutes=FLIGHT_TO_ACTIVITY + COMFORT_MINUTES)
    window = int(activity.metadata.get("reschedule_window_days", 3))
    duration = activity.end - activity.start
    last_moment = _trip_ends(itinerary)

    for day in range(0, window + 1):
        start = activity.start + timedelta(days=day)
        if start < earliest:
            continue
        end = start + duration
        if last_moment is not None and end > last_moment:
            return None
        return start, end
    return None


def _trip_ends(itinerary: Itinerary) -> datetime | None:
    """When the traveller stops being there — hotel check-out if there is one."""
    checkouts = [
        b.end for b in itinerary.bookings if b.type is BookingType.HOTEL
    ]
    return max(checkouts) if checkouts else None


# ===============================================================
# Assembling the draft
# ===============================================================
def _citations(
    disruption: VerifiedDisruption, overnight: bool, src: Sources
) -> list[str]:
    """Retrieve the chunk ids that actually ground this option.

    Retrieval is lexical and local, so this costs nothing and needs no key — see
    `core.policy`. An option with no citation is rejected by compiler rule C10,
    and inventing an id would be rejected too, so the ids come from the corpus.
    """
    queries = ["cancellation free rebooking next available flight"]
    if overnight:
        queries.append("hotel accommodation overnight stranded")

    found: list[str] = []
    for query in queries:
        result = src.policy.retrieve(f"{disruption.flight_iata[:2]} {query}", k=1)
        for hit in (result.value or {}).get("hits", []):
            if hit["chunk_id"] not in found:
                found.append(hit["chunk_id"])
    return found


def _option(
    itinerary: Itinerary,
    disruption: VerifiedDisruption,
    candidate: dict,
    *,
    option_id: str,
    prefix: str,
    original_arrival: datetime | None,
    src: Sources,
) -> OptionDraft:
    arrival = datetime.fromisoformat(candidate["arrival"])
    departure = datetime.fromisoformat(candidate["departure"])
    actions = derive_actions(itinerary, disruption, arrival, prefix=prefix)

    delay = (
        max(0, int((arrival - original_arrival).total_seconds() // 60))
        if original_arrival else 0
    )
    overnight = bool(original_arrival and arrival.date() > original_arrival.date())
    cancels = [a for a in actions if a.kind is ActionKind.CANCEL_ACTIVITY]

    if overnight:
        headline = "Fly tomorrow — cheaper, but the arrival day is gone"
        risk, reason = "medium", (
            "An overnight delay costs the whole arrival day, and availability "
            "was checked at search time rather than held."
        )
    elif delay > 240:
        headline = "Fly today, keep the trip on the same day"
        risk, reason = "low", (
            "Same carrier and no connection, so there is one thing to go wrong "
            "rather than two."
        )
    else:
        headline = "Fly today with the least disruption"
        risk, reason = "low", "A short delay with the rest of the day intact."

    if cancels:
        risk = "medium" if risk == "low" else risk
        reason += (
            f" {len(cancels)} booking{'s' if len(cancels) != 1 else ''} cannot be "
            f"salvaged and would be written off."
        )

    rationale = (
        f"Arrives {arrival.strftime('%a %d %b at %H:%M')}, "
        f"{_human_delay(delay)} later than planned. "
        + (
            f"{len(actions) - 1} downstream booking"
            f"{'s' if len(actions) - 1 != 1 else ''} shift to match."
            if len(actions) > 1 else "Nothing downstream needs to move."
        )
    )

    return OptionDraft(
        option_id=option_id,  # type: ignore[arg-type]
        headline=headline,
        flight=FlightChoice(
            carrier=candidate["carrier"],
            flight_number=candidate["flight_number"],
            departure_date=departure.date().isoformat(),
        ),
        additional_cost_amount=float(candidate.get("price_amount") or 0.0),
        additional_cost_currency=candidate.get("price_currency") or "USD",
        resolves=[],  # filled in by the caller from the server-owned impact list
        actions=actions,
        risk=risk,  # type: ignore[arg-type]
        risk_reason=reason,
        rationale=rationale,
        citations=_citations(disruption, overnight, src),
    )


def _human_delay(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} minutes"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h" if not mins else f"{hours}h {mins}m"
    days, rem = divmod(hours, 24)
    return f"{days} day{'s' if days != 1 else ''}" + (f" {rem}h" if rem else "")


def build_draft(
    itinerary: Itinerary,
    disruption: VerifiedDisruption,
    impact: ImpactSummary,
    candidates: list[dict],
    *,
    src: Sources | None = None,
) -> PlanDraft:
    """Build a two-option `PlanDraft` for any itinerary. Raises `NoRecovery`."""
    src = src or sources()
    best, alternative = choose_flights(itinerary, disruption, candidates)
    if alternative is None:
        raise NoRecovery(
            "only one usable replacement flight was found, so there is no second "
            "option worth offering — padding the list would be worse than saying so"
        )

    graph = build_dependency_graph(itinerary.bookings)
    original = next(
        (n for n in graph.nodes
         if n.booking_id == disruption.booking_id and n.kind is TripNodeKind.FLIGHT),
        None,
    )
    original_arrival = original.scheduled_at if original else None
    blocking = impact.blocking_node_ids()

    options = [
        _option(itinerary, disruption, best, option_id="A", prefix="a",
                original_arrival=original_arrival, src=src),
        _option(itinerary, disruption, alternative, option_id="B", prefix="b",
                original_arrival=original_arrival, src=src),
    ]
    for option in options:
        option.resolves = list(blocking)

    return PlanDraft(
        disruption_summary=(
            f"{disruption.flight_iata} on {disruption.date} was "
            f"{disruption.kind.value} by the carrier."
        ),
        options=options,
    )
