"""Applying a plan's actions to an itinerary. Pure functions.

This module is used by **both** the plan compiler and the executor, and that is
the point rather than a convenience. The compiler accepts an option only if
applying its actions here produces a valid itinerary; the executor then applies
the same actions with the same code. So what was validated is literally what
happens — there is no second implementation free to disagree with the one that
was approved.

Nothing here touches a database, a provider or a clock. `apply_actions` returns a
new `Itinerary`; the caller decides whether that is a simulation or the truth.
"""

from __future__ import annotations

from models import (
    Action,
    ActionKind,
    Booking,
    Itinerary,
    Leg,
    PlanOption,
)


class ActionApplyError(ValueError):
    """The action cannot be applied — a missing booking or absent new times."""


def _booking(itinerary: Itinerary, booking_id: str) -> Booking | None:
    return next((b for b in itinerary.bookings if b.booking_id == booking_id), None)


def apply_action(
    itinerary: Itinerary, action: Action, option: PlanOption
) -> Itinerary:
    """Return a new itinerary with `action` applied.

    `option` is needed because the replacement flight's times are the option's
    canonical (provider-sourced) ones, not something the action restates.
    """
    updated = itinerary.model_copy(deep=True)

    if action.kind is ActionKind.UPDATE_CALENDAR:
        # The traveller's calendar is outside the itinerary. Executing this
        # changes their calendar, not the trip, so the schedule is untouched —
        # and pretending otherwise would make the validator pass on a trip whose
        # bookings had not actually moved.
        return updated

    target = _booking(updated, action.target_booking_id)
    if target is None:
        raise ActionApplyError(
            f"action {action.action_id!r} targets booking "
            f"{action.target_booking_id!r}, which is not in the itinerary"
        )

    if action.kind is ActionKind.CANCEL_ACTIVITY:
        updated.bookings = [
            b for b in updated.bookings if b.booking_id != action.target_booking_id
        ]
        return updated

    if action.kind in (ActionKind.UPDATE_ITINERARY, ActionKind.PAY_FOR_FLIGHT):
        # Both mean "the trip now uses the replacement flight". They differ in who
        # performs them, not in their effect: `update_itinerary` records the intent
        # locally, `pay_for_flight` is the traveller actually committing. The
        # schedule effect is the same, and applying it twice is idempotent.
        flight = option.flight
        target.start = flight.departure
        target.end = flight.arrival
        target.location_iata = flight.origin_iata
        target.title = (
            f"{flight.origin_iata} → {flight.destination_iata} on "
            f"{flight.carrier}{flight.flight_number}"
        )
        target.legs = [Leg(
            carrier=flight.carrier,
            flight_number=flight.flight_number,
            origin_iata=flight.origin_iata,
            destination_iata=flight.destination_iata,
            scheduled_departure=flight.departure,
            scheduled_arrival=flight.arrival,
            cabin=target.legs[0].cabin if target.legs else None,
        )]
        return updated

    if action.needs_new_times():
        if action.new_start is None:
            raise ActionApplyError(
                f"action {action.action_id!r} ({action.kind.value}) moves "
                f"{action.target_booking_id!r} in time but carries no new_start"
            )
        # An end with no explicit value keeps the booking's original duration,
        # which is what a moved transfer or a rescheduled activity means. Hotels
        # are the exception worth noticing: holding a check-in later must not
        # push the check-out out too, or a three-night stay silently becomes four.
        duration = target.end - target.start
        target.start = action.new_start
        if action.new_end is not None:
            target.end = action.new_end
        elif action.kind is not ActionKind.SEND_HOTEL_MESSAGE:
            target.end = action.new_start + duration
        return updated

    return updated


def apply_actions(
    itinerary: Itinerary, option: PlanOption, actions: list[Action] | None = None
) -> Itinerary:
    """Apply `actions` (default: all of the option's) in order.

    Order matters only in that the flight update should land before anything
    measured against it; the option's action list is authored in that order and
    the compiler simulates the same sequence the executor will run.
    """
    result = itinerary
    for action in actions if actions is not None else option.actions:
        result = apply_action(result, action, option)
    return result
