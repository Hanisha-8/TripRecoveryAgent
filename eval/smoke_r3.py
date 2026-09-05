"""Offline gate for R3 — executor, ledger, handoffs, close-out.

    python -m eval.smoke_r3

No API key, no spend. The property that matters most is tested by its absence:
`core.executor.outbox()` stays empty unless a message actually went out, so a
blocked action is proven blocked rather than merely reported as refused.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

from core.approval import Authorization, approve, authorize
from core.dependency_graph import build_dependency_graph
from core.executor import (
    ConfirmRefusal,
    confirm_handoff,
    execute,
    outbox,
    reset_outbox,
    revalidate,
    working_itinerary,
)
from core.itinerary_ops import apply_actions
from core.store import save_plan
from eval.harness import HOTEL_MESSAGE, Scenario, check, report
from models import OUTWARD_FACING, ActionClass, ActionKind


def _authorized(s: Scenario, db: Path, option_id: str = "A") -> Authorization:
    plan = s.plan()
    save_plan(plan, db_path=db)
    approval = approve(plan, option_id, approved_by="alex")
    auth = authorize(plan, approval, s.itinerary)  # type: ignore[arg-type]
    assert isinstance(auth, Authorization), auth
    return auth


def graph_edges() -> None:
    s = Scenario()

    print("\n=== presence edges: you cannot be somewhere before you land ===")
    graph = build_dependency_graph(s.itinerary.bookings)
    explicit = {
        e.to_node for e in graph.edges if "you must have landed" in e.rationale
    }
    check("every SIN booking is tied to the SIN arrival",
          explicit == {"bk_transfer_sin_hotel", "bk_hotel_marina:checkin",
                       "bk_activity_night_safari"},
          str(sorted(explicit)))
    check("check-out is not tied to the inbound arrival",
          "bk_hotel_marina:checkout" not in explicit,
          "that would double-count check-in and forbid a same-day turnaround")

    # Hole 1: move the flight later, leave the transfer. Chronological sorting puts
    # the car before the plane — a comfortable positive gap, and a traveller whose
    # transfer turns up six hours early.
    moved = s.itinerary.model_copy(deep=True)
    flight = next(b for b in moved.bookings if b.booking_id == "bk_flight_sq123")
    flight.end = flight.end + timedelta(hours=7)
    flight.legs[0].scheduled_arrival = flight.legs[0].scheduled_arrival + timedelta(hours=7)
    result = revalidate_pure(moved)
    check("a stale transfer left before the new arrival is caught",
          not result.valid
          and any(r.booking_id == "bk_transfer_sin_hotel" for r in result.remaining),
          f"remaining={[(r.booking_id, r.reason) for r in result.remaining]}")

    # Hole 2: move the activity to before the trip arrives. It sorts first, every
    # gap after it is enormous, and a purely chronological check calls it valid.
    early = s.itinerary.model_copy(deep=True)
    safari = next(b for b in early.bookings
                  if b.booking_id == "bk_activity_night_safari")
    safari.start = safari.start - timedelta(days=1)
    safari.end = safari.end - timedelta(days=1)
    result = revalidate_pure(early)
    check("an activity scheduled before the traveller lands is caught",
          not result.valid
          and any(r.booking_id == "bk_activity_night_safari"
                  for r in result.remaining),
          f"remaining={[(r.booking_id, r.reason) for r in result.remaining]}")


def revalidate_pure(itinerary):
    from core.impact import validate_itinerary
    return validate_itinerary(itinerary.trip_id, itinerary.bookings)


def itinerary_ops() -> None:
    s = Scenario()
    plan = s.plan()
    option = plan.options[0]

    print("\n=== applying actions ===")
    after = apply_actions(s.itinerary, option)
    flight = next(b for b in after.bookings if b.booking_id == "bk_flight_sq123")
    check("the flight is replaced with the approved option's flight",
          flight.legs[0].flight_number == "SQ425"
          and flight.end == option.flight.arrival,
          f"{flight.legs[0].flight_number} arriving {flight.end}")

    hotel = next(b for b in after.bookings if b.booking_id == "bk_hotel_marina")
    check("the hotel check-in moves to the held time",
          hotel.start.isoformat() == "2026-09-16T01:00:00+08:00", hotel.start.isoformat())
    check("but the check-out does NOT move with it",
          hotel.end == next(b for b in s.itinerary.bookings
                            if b.booking_id == "bk_hotel_marina").end,
          "a three-night stay must not silently become four")

    safari = next(b for b in after.bookings if b.booking_id == "bk_activity_night_safari")
    check("the activity keeps its duration when rescheduled",
          safari.end - safari.start == timedelta(hours=3),
          str(safari.end - safari.start))

    print("\n=== application is order-independent ===")
    forwards = apply_actions(s.itinerary, option, option.actions)
    backwards = apply_actions(s.itinerary, option, list(reversed(option.actions)))
    check("applying the actions in reverse lands on the same itinerary",
          {b.booking_id: (b.start, b.end) for b in forwards.bookings}
          == {b.booking_id: (b.start, b.end) for b in backwards.bookings},
          "handoffs come back in whatever order the traveller gets to them")

    check("and the fully-applied plan validates",
          revalidate_pure(forwards).valid,
          str([(r.booking_id, r.reason) for r in revalidate_pure(forwards).remaining]))


def executor() -> None:
    s = Scenario()

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "r3.db"
        reset_outbox()
        auth = _authorized(s, db)
        plan_version = auth.plan_version

        print("\n=== execute: agent half only ===")
        result = execute(auth, db_path=db)

        done = {e.action_id: e for e in result.ledger}
        check("both agent-safe actions are in the ledger",
              {"a-hotel", "a-safari"} <= set(done), str(sorted(done)))
        check("and are recorded as performed by the agent",
              all(done[a].performed_by == "agent" for a in ("a-hotel", "a-safari")))
        check("neither human action was performed",
              not {"a-pay", "a-transfer"} & set(done), str(sorted(done)))

        check("exactly one message was drafted", len(outbox()) == 1, str(outbox()))
        check("and it was the approved bytes, not a re-composition",
              outbox()[0]["body"] == HOTEL_MESSAGE, outbox()[0]["body"][:60])
        check("the draft is explicitly marked undelivered",
              outbox()[0].get("delivered") == "no", str(outbox()[0]))

        print("\n=== nothing outward is reported as done ===")
        # The UI used to say "Done by agent · sent 174 chars to bk_hotel_marina"
        # for a message that was never sent. There is no hotel integration, no
        # provider integration and no calendar; claiming otherwise is a lie about
        # the one thing this system exists to be careful about.
        for action in auth.option.actions:
            entry = done.get(action.action_id)
            if entry is None:
                continue
            if action.kind in OUTWARD_FACING or action.kind in (
                ActionKind.RESCHEDULE_ACTIVITY, ActionKind.CANCEL_ACTIVITY,
                ActionKind.UPDATE_CALENDAR,
            ):
                check(f"{action.kind.value} is 'prepared', never 'done'",
                      entry.status == "prepared" and not entry.settled,
                      f"status={entry.status} detail={entry.detail!r}")
                check(f"{action.kind.value} says so in its detail line",
                      "not been" in entry.detail.lower()
                      or "nothing has been sent" in entry.detail.lower(),
                      entry.detail)
            else:
                check(f"{action.kind.value} really is done",
                      entry.status == "done" and entry.settled,
                      f"status={entry.status}")

        check("no ledger detail claims a message was sent",
              not any("sent " in e.detail.lower()
                      and "nothing has been sent" not in e.detail.lower()
                      for e in result.ledger),
              str([e.detail for e in result.ledger]))
        check("progress counts only what really settled",
              len([e for e in result.ledger if e.settled]) == 0,
              "both agent actions are outward-facing, so nothing is settled yet")

        pending = {h.action_id for h in result.pending}
        check("both human actions became pending handoffs",
              pending == {"a-pay", "a-transfer"}, str(sorted(pending)))

        print("\n=== execute: the case does not close ===")
        check("the trip is not valid with handoffs outstanding",
              not result.closed,
              "executing the agent half must not read as recovery complete")
        check("and the reason given is the handoffs, not a broken schedule",
              result.validation is not None
              and sorted(result.validation.pending_handoffs) == ["a-pay", "a-transfer"],
              str(result.validation))

        print("\n=== execute: idempotent ===")
        again = execute(auth, db_path=db)
        check("re-executing does not send a second message", len(outbox()) == 1,
              f"{len(outbox())} messages — a UI retry would have double-sent")
        check("and does not duplicate ledger entries",
              len(again.ledger) == len(result.ledger),
              f"{len(again.ledger)} vs {len(result.ledger)}")

        print("\n=== confirm handoffs ===")
        bad = confirm_handoff(s.itinerary.trip_id, plan_version, "a-nonexistent", db_path=db)
        check("confirming an action with no handoff is refused",
              isinstance(bad, ConfirmRefusal)
              and "cannot create the obligation" in bad.detail, str(bad))

        first = confirm_handoff(s.itinerary.trip_id, plan_version, "a-transfer", db_path=db)
        check("confirming the transfer is accepted", not isinstance(first, ConfirmRefusal),
              getattr(first, "detail", ""))
        assert not isinstance(first, ConfirmRefusal)
        check("it is recorded as performed by the human",
              next(e for e in first.ledger if e.action_id == "a-transfer").performed_by
              == "human")
        check("the trip is still not valid — one handoff remains",
              not first.closed and [h.action_id for h in first.pending] == ["a-pay"],
              str([h.action_id for h in first.pending]))

        moved = working_itinerary(s.itinerary.trip_id, db_path=db)
        assert moved is not None
        transfer = next(b for b in moved.bookings if b.booking_id == "bk_transfer_sin_hotel")
        check("confirming applied the transfer's new time",
              transfer.start.isoformat() == "2026-09-16T00:15:00+08:00",
              transfer.start.isoformat())

        print("\n=== the case closes ===")
        final = confirm_handoff(s.itinerary.trip_id, plan_version, "a-pay", db_path=db)
        assert not isinstance(final, ConfirmRefusal)
        check("with every handoff confirmed, the trip is valid again", final.closed,
              f"remaining={[(r.booking_id, r.reason) for r in final.validation.remaining]} "
              f"pending={final.validation.pending_handoffs}"
              if final.validation else "no validation")
        check("all four actions are on the ledger", len(final.ledger) == 4,
              str([e.action_id for e in final.ledger]))
        check("two by agent, two by human",
              sorted(e.performed_by for e in final.ledger)
              == ["agent", "agent", "human", "human"],
              str([(e.action_id, e.performed_by) for e in final.ledger]))
        check("no handoff is left pending", not final.pending,
              str([h.action_id for h in final.pending]))

        print("\n=== confirming twice is harmless ===")
        repeat = confirm_handoff(s.itinerary.trip_id, plan_version, "a-pay", db_path=db)
        assert not isinstance(repeat, ConfirmRefusal)
        check("a duplicate confirmation does not re-apply or unbalance the ledger",
              len(repeat.ledger) == 4 and repeat.closed,
              str([e.action_id for e in repeat.ledger]))


def executor_refusals() -> None:
    s = Scenario()

    print("\n=== nothing executes without a token ===")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "refuse.db"
        reset_outbox()

        # The gate proven by its effect: a refused authorization leaves the outbox
        # empty. Asserting that `authorize` returned a Refusal only proves it
        # returned a Refusal.
        plan = s.plan()
        save_plan(plan, db_path=db)
        approval = approve(plan, "A", approved_by="alex")

        tampered = plan.model_copy(deep=True)
        hotel = next(a for a in tampered.options[0].actions
                     if a.kind is ActionKind.SEND_HOTEL_MESSAGE)
        hotel.message_body = "Please cancel our reservation entirely."
        refusal = authorize(tampered, approval, s.itinerary)  # type: ignore[arg-type]
        check("a tampered plan does not authorise", not getattr(refusal, "ok", False))
        check("and nothing was sent", not outbox(),
              f"{len(outbox())} messages went out despite the refusal")

        # A cancellation message cannot be executed because there is no execute()
        # path that accepts anything but an Authorization.
        try:
            execute(refusal, db_path=db)  # type: ignore[arg-type]
            check("execute() rejects a Refusal in place of a token", False,
                  "a Refusal was accepted as authorization")
        except AttributeError:
            check("execute() rejects a Refusal in place of a token", True)
        check("still nothing sent", not outbox(), str(outbox()))

    print("\n=== a stale confirmation cannot resurrect a modified plan ===")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "stale.db"
        reset_outbox()
        auth = _authorized(s, db)
        execute(auth, db_path=db)

        # Edit the stored plan row, then try to confirm against it.
        import sqlite3
        with sqlite3.connect(db) as conn:
            payload = conn.execute(
                "SELECT payload FROM plans WHERE plan_version = ?",
                (auth.plan_version,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE plans SET payload = ? WHERE plan_version = ?",
                (payload.replace("Marina Bay", "Somewhere Else"), auth.plan_version),
            )
        res = confirm_handoff(s.itinerary.trip_id, auth.plan_version, "a-pay", db_path=db)
        check("confirming against an edited stored plan is refused",
              isinstance(res, ConfirmRefusal) and "no longer hashes" in res.detail,
              str(res))


def option_b_closes_too() -> None:
    """Both options must be executable, not just the recommended one."""
    s = Scenario()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "b.db"
        reset_outbox()
        auth = _authorized(s, db, option_id="B")

        print("\n=== option B executes and closes ===")
        result = execute(auth, db_path=db)
        check("B's agent actions run", len(result.ledger) == 2,
              str([e.action_id for e in result.ledger]))
        check("B opens the same two handoffs",
              {h.action_id for h in result.pending} == {"b-pay", "b-transfer"},
              str([h.action_id for h in result.pending]))

        for action_id in ("b-transfer", "b-pay"):
            out = confirm_handoff(s.itinerary.trip_id, auth.plan_version, action_id,
                                  db_path=db)
            assert not isinstance(out, ConfirmRefusal), out
        final = revalidate(s.itinerary.trip_id, auth.plan_version, db_path=db)
        check("B also ends with a valid trip", final.valid,
              f"remaining={[(r.booking_id, r.reason) for r in final.remaining]}")


def main() -> int:
    graph_edges()
    itinerary_ops()
    executor()
    executor_refusals()
    option_b_closes_too()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
