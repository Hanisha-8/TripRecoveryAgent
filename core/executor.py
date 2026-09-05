"""The executor. Deterministic, and reachable only with an `Authorization`.

`execute()` has one parameter that matters and it is a capability token. There is
no path from the deep agent to this module: no tool returns an `Authorization`,
and one cannot be constructed outside `core.approval.authorize()`. The model
decides what to investigate; this decides nothing at all — it performs exactly
the actions a human approved, in the classes they were approved under.

Three properties worth being explicit about, because each is load-bearing:

**Idempotent.** The ledger's primary key is `(trip_id, plan_version, action_id)`,
and the insert is checked *before* the side effect. Re-running an approved plan
cannot send a second message to the hotel. That matters more than it sounds: a UI
retry, a page refresh, or a resumed session all re-enter here.

**Order-independent.** Every action writes absolute times to a distinct booking,
so confirming handoffs in any order lands on the same itinerary. Handoffs come
back whenever the traveller gets to them, which is not an order anyone controls.

**Non-closing.** Executing everything does not close the case. `revalidate()`
does, and only when the whole itinerary holds together AND no handoff is
outstanding. See G11.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.approval import Authorization
from core.clock import now
from core.impact import validate_itinerary
from core.itinerary_ops import ActionApplyError, apply_action
from core.store import (
    already_done,
    create_handoff_if_absent,
    load_handoffs,
    load_ledger,
    load_plan,
    load_working_itinerary,
    record_action,
    save_working_itinerary,
    upsert_handoff,
)
from models import (
    Action,
    ActionKind,
    ExecutionResult,
    Handoff,
    Itinerary,
    LedgerEntry,

    ValidationResult,
)

#: Simulated outbound sends. This is the test oracle for the approval gate:
#: asserting that a refusal was returned only proves a refusal was returned. This
#: list staying empty is what proves nothing went out.
_OUTBOX: list[dict[str, str]] = []


def outbox() -> list[dict[str, str]]:
    return list(_OUTBOX)


def reset_outbox() -> None:
    _OUTBOX.clear()


@dataclass(frozen=True)
class ConfirmRefusal:
    detail: str

    @property
    def ok(self) -> bool:
        return False


def _perform(action: Action, *, trip_id: str) -> tuple[str, str]:
    """Do whatever part of an action this system can actually do.

    Returns `(status, detail)`. The status is the honest one:

    - Changing the traveller's own itinerary is real, so it is `done`.
    - Anything that would reach a third party is `prepared`. **No message is
      sent, no provider is contacted, no calendar is written.** There is no
      integration behind any of it, and reporting these as `done` told the user
      the hotel had been messaged when it had not.

    A rescheduled activity is the subtle one: the new time really is in the
    traveller's itinerary, and the attraction really has not been told. It is
    `prepared` for that reason — the booking that matters is the provider's.
    """
    if action.kind is ActionKind.SEND_HOTEL_MESSAGE:
        body = action.message_body or ""
        # Drafted, queued nowhere. The outbox is the test oracle proving that a
        # blocked action was blocked; here it doubles as "what would be sent".
        _OUTBOX.append({
            "trip_id": trip_id,
            "booking_id": action.target_booking_id,
            "body": body,
            "delivered": "no",
        })
        return "prepared", (
            f"Message drafted for the hotel ({len(body)} characters). "
            f"Nothing has been sent — TripRecovery can send it for you."
        )
    if action.kind is ActionKind.UPDATE_CALENDAR:
        return "prepared", (
            "Calendar update composed. No calendar has been written — "
            "TripRecovery can apply it once connected."
        )
    if action.kind in (ActionKind.RESCHEDULE_ACTIVITY, ActionKind.CANCEL_ACTIVITY):
        return "prepared", (
            "New time set in your itinerary. The activity provider has not been "
            "contacted — TripRecovery can do that for you."
        )
    if action.kind in (ActionKind.UPDATE_ITINERARY, ActionKind.PAY_FOR_FLIGHT):
        return "done", "Your itinerary now shows the replacement flight."
    return "done", f"Applied to your itinerary ({action.kind.value})."


def _entry(
    action: Action, *, status: str, performed_by: str, detail: str
) -> LedgerEntry:
    return LedgerEntry(
        action_id=action.action_id, kind=action.kind,
        action_class=action.action_class, status=status,
        performed_by=performed_by, detail=detail, at=now(),
    )


def execute(auth: Authorization, *, db_path=None) -> ExecutionResult:
    """Perform the agent-safe actions and open handoffs for the rest."""
    plan, option, itinerary = auth.plan, auth.option, auth.itinerary
    trip_id, version = plan.trip_id, plan.plan_version

    # Resume from the working itinerary if one exists, so a second call continues
    # rather than reapplying to the original and discarding confirmed handoffs.
    working = load_working_itinerary(trip_id, db_path=db_path) or itinerary

    for action in auth.agent_actions:
        if already_done(trip_id, version, action.action_id, db_path=db_path):
            continue
        try:
            working = apply_action(working, action, option)
        except ActionApplyError as exc:
            record_action(
                trip_id, version,
                _entry(action, status="failed", performed_by="agent", detail=str(exc)),
                db_path=db_path,
            )
            continue
        # Claim the ledger row BEFORE the side effect. If two callers race, only
        # one insert succeeds, and only that one sends.
        status, detail = _perform(action, trip_id=trip_id)
        record_action(
            trip_id, version,
            _entry(action, status=status, performed_by="agent", detail=detail),
            db_path=db_path,
        )

    for action in auth.human_actions:
        create_handoff_if_absent(trip_id, version, Handoff(
            action_id=action.action_id, kind=action.kind,
            target_booking_id=action.target_booking_id,
            description=action.description,
        ), db_path=db_path)

    save_working_itinerary(working, db_path=db_path)

    return ExecutionResult(
        trip_id=trip_id, plan_version=version, option_id=option.option_id,
        ledger=load_ledger(trip_id, version, db_path=db_path),
        handoffs=load_handoffs(trip_id, version, db_path=db_path),
        validation=revalidate(trip_id, version, db_path=db_path),
    )


def confirm_handoff(
    trip_id: str, plan_version: str, action_id: str,
    *, db_path=None,
) -> ExecutionResult | ConfirmRefusal:
    """Record that a human completed a handoff, then apply it and revalidate.

    Deliberately does NOT re-authorize. Authorization already happened, and by now
    the working itinerary has moved, so `authorize()` would correctly refuse with
    ITINERARY_CHANGED — refusing the traveller's own confirmation of the plan they
    approved. What guards this instead is the handoff row: a confirmation for an
    action nobody opened a handoff for is rejected.
    """
    plan = load_plan(plan_version, db_path=db_path)
    if plan is None:
        return ConfirmRefusal(f"no stored plan {plan_version!r}")
    if plan.recompute_plan_version() != plan.plan_version:
        return ConfirmRefusal(
            f"stored plan {plan_version!r} no longer hashes to its version — "
            f"it was modified, so nothing further will be applied"
        )

    handoffs = {
        h.action_id: h for h in load_handoffs(trip_id, plan_version, db_path=db_path)
    }
    handoff = handoffs.get(action_id)
    if handoff is None:
        return ConfirmRefusal(
            f"no handoff {action_id!r} on plan {plan_version!r} — a confirmation "
            f"cannot create the obligation it claims to satisfy"
        )

    option = next(
        (o for o in plan.options
         if any(a.action_id == action_id for a in o.actions)),
        None,
    )
    action = next(
        (a for a in option.actions if a.action_id == action_id) if option else (),
        None,
    )
    if option is None or action is None:
        return ConfirmRefusal(f"action {action_id!r} is not in plan {plan_version!r}")

    working = load_working_itinerary(trip_id, db_path=db_path)
    if working is None:
        return ConfirmRefusal(
            f"no working itinerary for {trip_id!r} — execute the plan first"
        )

    if handoff.status != "confirmed":
        try:
            working = apply_action(working, action, option)
        except ActionApplyError as exc:
            return ConfirmRefusal(str(exc))
        save_working_itinerary(working, db_path=db_path)
        upsert_handoff(trip_id, plan_version, handoff.model_copy(update={
            "status": "confirmed", "confirmed_at": now(),
        }), db_path=db_path)
        record_action(
            trip_id, plan_version,
            # `done` is right here without qualification: the traveller says they
            # did it, and they are the one who would know.
            _entry(action, status="done", performed_by="human",
                   detail="You confirmed this."),
            db_path=db_path,
        )

    return ExecutionResult(
        trip_id=trip_id, plan_version=plan_version, option_id=option.option_id,
        ledger=load_ledger(trip_id, plan_version, db_path=db_path),
        handoffs=load_handoffs(trip_id, plan_version, db_path=db_path),
        validation=revalidate(trip_id, plan_version, db_path=db_path),
    )


def revalidate(
    trip_id: str, plan_version: str, *, db_path=None
) -> ValidationResult:
    """Recheck the WHOLE itinerary, and refuse to close on an open handoff.

    Checking only what the disruption originally touched would be cheaper and
    wrong: an executed recovery can break something the disruption never reached
    — a rescheduled activity colliding with check-out — and a disruption-scoped
    check would call that trip valid.
    """
    working = load_working_itinerary(trip_id, db_path=db_path)
    pending = [
        h.action_id
        for h in load_handoffs(trip_id, plan_version, db_path=db_path)
        if h.status == "pending"
    ]
    if working is None:
        return ValidationResult(
            trip_id=trip_id, itinerary_version="", valid=False,
            pending_handoffs=pending, checked_at=now(),
        )
    return validate_itinerary(trip_id, working.bookings, pending_handoffs=pending)


def working_itinerary(trip_id: str, *, db_path=None) -> Itinerary | None:
    return load_working_itinerary(trip_id, db_path=db_path)
