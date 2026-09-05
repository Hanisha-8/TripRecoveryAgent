"""The demo scenario: SQ123 BOM→SIN cancelled, and a plan that compiles.

This exists so the UI is fully explorable with no API key and no spend, and it
lives here rather than in `eval/` for a specific reason: `eval/harness.py`
imports these same fixtures, so **the plan the demo shows is the plan the gates
test**. If the two had separate definitions, the demo could drift into showing a
plan the compiler would reject.

Everything below is authored by hand in the shape the deep agent's `PlanDraft`
would arrive in. It is then put through the real `compile_plan`, so the demo plan
is version-bound, its flights are canonicalised from the provider fixture, and it
has passed every rule C1-C10 exactly as a live plan would.
"""

from __future__ import annotations

from datetime import datetime

from config import DATA_DIR
from core.plan_compiler import CompileResult, compile_plan
from models import (
    Action,
    ActionClass,
    ActionKind,
    FlightChoice,
    ImpactSummary,
    Itinerary,
    OptionDraft,
    PlanDraft,
    RecoveryPlan,
    VerifiedDisruption,
)
from recovery.agent import assess, flight_candidates, load_itinerary, verify_disruption

ITINERARY_PATH = DATA_DIR / "sample_itinerary_sin.json"


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


class Scenario:
    """Everything the deterministic core derives from a trip, once.

    Works on any `Itinerary`, not just the bundled sample, because the UI lets a
    visitor upload their own and the whole view has to follow from it.
    """

    def __init__(self, itinerary_path=ITINERARY_PATH) -> None:
        self._init(load_itinerary(itinerary_path))

    @classmethod
    def from_itinerary(cls, itinerary: Itinerary) -> Scenario:
        """Build a scenario from an already-parsed itinerary (an upload)."""
        scenario = cls.__new__(cls)
        scenario._init(itinerary)
        return scenario

    def _init(self, itinerary: Itinerary) -> None:
        self.itinerary: Itinerary = itinerary
        self.disruption: VerifiedDisruption | None = verify_disruption(itinerary)
        self.impact: ImpactSummary = (
            assess(itinerary, self.disruption) if self.disruption
            else ImpactSummary(disruption_node="", impacted=[])
        )
        self.candidates: list[dict] = (
            flight_candidates(itinerary, self.disruption) if self.disruption else []
        )

    @property
    def has_disruption(self) -> bool:
        return self.disruption is not None and self.disruption.verified

    def builder_plan(self):
        """The plan a deterministic planner produces for this trip.

        Goes through the same `compile_plan` gate as anything a model produces —
        the planner gets no more trust than the agent does.
        """
        from core.planner import build_draft

        assert self.disruption is not None
        return self.compile(
            build_draft(self.itinerary, self.disruption, self.impact, self.candidates)
        )

    @property
    def blocking(self) -> list[str]:
        return self.impact.blocking_node_ids()

    def compile(self, draft: PlanDraft) -> CompileResult:
        return compile_plan(
            draft,
            itinerary=self.itinerary,
            verified_disruption=self.disruption,
            impact=self.impact,
            flight_candidates=self.candidates,
        )

    def plan(self) -> RecoveryPlan:
        """The compiled demo plan. Raises if the fixtures have drifted."""
        result = self.compile(good_draft(self.blocking))
        assert result.plan is not None, f"fixture no longer compiles: {result.violations}"
        return result.plan


# ===============================================================
# Flights — must exist in data/mock_flights.json or C4 rejects them
# ===============================================================
def flight_a() -> FlightChoice:
    """SQ425: same day, arrives 23:15 SGT — 7h after the original.

    Only the designator and the date: the arrival, fare and stop count come from
    the provider record when the compiler resolves this, so a fixture cannot
    disagree with the data it is meant to represent.
    """
    return FlightChoice(
        carrier="SQ", flight_number="SQ425", departure_date="2026-09-15"
    )


def flight_b() -> FlightChoice:
    """AI2380: next morning, arrives 15:30 SGT on the 16th — cheaper, loses a day."""
    return FlightChoice(
        carrier="AI", flight_number="AI2380", departure_date="2026-09-16"
    )


HOTEL_MESSAGE = (
    "Hello, our flight to Singapore was cancelled and we have rebooked. "
    "We now expect to reach the hotel after midnight. Please hold the "
    "reservation for a late arrival. Thank you."
)


def actions(
    prefix: str,
    *,
    transfer: tuple[str, str],
    hotel_check_in: str,
    safari: tuple[str, str],
) -> list[Action]:
    """One action per booking the disruption breaks, correctly classed and timed.

    The times matter: the compiler applies these and requires the resulting
    itinerary to validate, so an action naming the right booking and moving it to
    a useless time does not pass.
    """
    return [
        Action(
            action_id=f"{prefix}-pay", kind=ActionKind.PAY_FOR_FLIGHT,
            action_class=ActionClass.HUMAN_REQUIRED,
            target_booking_id="bk_flight_sq123",
            description="Complete payment for the replacement fare.",
            requires_payment=True,
        ),
        Action(
            action_id=f"{prefix}-hotel", kind=ActionKind.SEND_HOTEL_MESSAGE,
            action_class=ActionClass.AGENT_SAFE,
            target_booking_id="bk_hotel_marina",
            description="Tell the hotel to hold the room for a late arrival.",
            message_body=HOTEL_MESSAGE,
            new_start=_dt(hotel_check_in),
        ),
        Action(
            action_id=f"{prefix}-transfer", kind=ActionKind.REBOOK_TRANSFER,
            action_class=ActionClass.HUMAN_REQUIRED,
            target_booking_id="bk_transfer_sin_hotel",
            description="Move the airport transfer to the new arrival time.",
            new_start=_dt(transfer[0]), new_end=_dt(transfer[1]),
        ),
        Action(
            action_id=f"{prefix}-safari", kind=ActionKind.RESCHEDULE_ACTIVITY,
            action_class=ActionClass.AGENT_SAFE,
            target_booking_id="bk_activity_night_safari",
            description="Move the Night Safari tickets to the following evening.",
            new_start=_dt(safari[0]), new_end=_dt(safari[1]),
        ),
    ]


def actions_a() -> list[Action]:
    """Option A — land 23:15 on the 15th, everything shifts past midnight."""
    return actions(
        "a",
        transfer=("2026-09-16T00:15:00+08:00", "2026-09-16T01:00:00+08:00"),
        hotel_check_in="2026-09-16T01:00:00+08:00",
        safari=("2026-09-16T19:30:00+08:00", "2026-09-16T22:30:00+08:00"),
    )


def actions_b() -> list[Action]:
    """Option B — land 15:30 on the 16th, the arrival day is gone."""
    return actions(
        "b",
        transfer=("2026-09-16T16:30:00+08:00", "2026-09-16T17:15:00+08:00"),
        hotel_check_in="2026-09-16T17:15:00+08:00",
        safari=("2026-09-16T19:30:00+08:00", "2026-09-16T22:30:00+08:00"),
    )


def good_draft(blocking: list[str]) -> PlanDraft:
    return PlanDraft(
        disruption_summary="SQ123 BOM→SIN on 2026-09-15 was cancelled by the carrier.",
        options=[
            OptionDraft(
                option_id="A", headline="Fly today, keep the trip on the same day",
                flight=flight_a(), additional_cost_amount=120.0,
                resolves=list(blocking),
                actions=actions_a(), risk="low",
                risk_reason="Single carrier, no connection, seats confirmed at search time.",
                rationale=(
                    "Arrives the same night, so only the evening is lost rather than a "
                    "whole day. Fare and seat count are from a local fallback, so "
                    "verify at booking."
                ),
                citations=["disruption_care:SQ-CANCEL-REBOOK-01"],
            ),
            OptionDraft(
                option_id="B", headline="Fly tomorrow morning for less",
                flight=flight_b(), additional_cost_amount=70.0,
                resolves=list(blocking),
                actions=actions_b(), risk="medium",
                risk_reason="Loses the whole arrival day; only 4 seats were showing.",
                rationale=(
                    "Cheapest way out and a full night's rest, at the cost of the "
                    "arrival day. Availability came from a local fallback, so "
                    "verify at booking."
                ),
                citations=[
                    "disruption_care:SQ-CANCEL-REBOOK-01",
                    "disruption_care:SQ-CANCEL-ACCOM-01",
                ],
            ),
        ],
    )
