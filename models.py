"""Domain models and the recovery-plan contract.

Two families live here, and the split is the whole architecture:

**Model-facing** (`PlanOption`, `Action`, `PlanDraft`) — what the deep agent is
allowed to author. It proposes options and the actions they require.

**Server-owned** (`RecoveryPlan`, `ApprovalRecord`, `LedgerEntry`, `Handoff`,
`ValidationResult`) — what deterministic Python owns. The agent never computes
`itinerary_version` or `plan_version`, because those are the values approval is
bound to. A model that can compute its own approval hash can invalidate the gate
by restating the plan.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ===============================================================
# Provenance envelope — every tool result is wrapped
# ===============================================================
class ToolResult(BaseModel):
    """Wraps every tool return so the plan compiler can verify data lineage.

    `uncertain` / `fallback_used` are the fields that matter most: they say the
    live provider was unreachable, which every downstream claim inherits.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    value: Any = None
    fetched_at: datetime
    source: str
    uncertain: bool = False
    fallback_used: bool = False
    error: str | None = None


# ===============================================================
# Booking primitives
# ===============================================================
class BookingType(str, Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"
    ACTIVITY = "activity"
    TRANSFER = "transfer"


class Leg(BaseModel):
    """A single flight segment inside a Booking."""

    carrier: str
    flight_number: str
    origin_iata: str
    destination_iata: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    cabin: str | None = None
    booking_reference: str | None = None  # PNR — redacted before the agent sees it


class Booking(BaseModel):
    booking_id: str
    type: BookingType
    title: str
    start: datetime
    end: datetime
    location_iata: str | None = None
    location_name: str | None = None
    legs: list[Leg] = Field(default_factory=list)
    cancellable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Itinerary(BaseModel):
    trip_id: str
    traveller_first_name: str
    party_size: int = 1
    bookings: list[Booking] = Field(default_factory=list)


# ===============================================================
# Trip dependency graph
# ===============================================================
class TripNodeKind(str, Enum):
    FLIGHT = "flight"
    TRANSFER = "transfer"
    HOTEL_CHECK_IN = "hotel_check_in"
    HOTEL_CHECK_OUT = "hotel_check_out"
    ACTIVITY = "activity"


class TripNode(BaseModel):
    node_id: str
    booking_id: str
    kind: TripNodeKind
    label: str
    scheduled_at: datetime


class TripEdge(BaseModel):
    """Directed edge: `from_node` must precede `to_node` by `min_gap_minutes`."""

    from_node: str
    to_node: str
    min_gap_minutes: int
    rationale: str


class TripDependencyGraph(BaseModel):
    nodes: list[TripNode] = Field(default_factory=list)
    edges: list[TripEdge] = Field(default_factory=list)


# ===============================================================
# Disruption + impact
# ===============================================================
class DisruptionKind(str, Enum):
    CANCELLED = "cancelled"
    DELAYED = "delayed"
    DIVERTED = "diverted"
    GATE_CHANGE = "gate_change"
    UNKNOWN = "unknown"


class DisruptionEvent(BaseModel):
    booking_id: str
    node_id: str
    kind: DisruptionKind
    delta_minutes: int = 0
    detected_at: datetime
    source: str


class VerifiedDisruption(BaseModel):
    """A disruption the agent confirmed with a tool, not one the user asserted.

    `verified` is false when the provider chain fell all the way through to the
    local fixture. Step 1 of the flow says stop or escalate when verification is
    insufficient — this field is what that decision reads.
    """

    booking_id: str
    flight_iata: str
    date: str
    kind: DisruptionKind
    delta_minutes: int = 0
    verified: bool
    source: str
    fetched_at: datetime
    uncertain: bool = False
    note: str | None = None


Severity = Literal["broken", "at_risk", "degraded"]


class ImpactedNode(BaseModel):
    node_id: str
    booking_id: str = ""
    label: str = ""
    severity: Severity
    reason: str


class ImpactSummary(BaseModel):
    disruption_node: str
    impacted: list[ImpactedNode] = Field(default_factory=list)

    def blocking_node_ids(self) -> list[str]:
        """Nodes an option must resolve. `degraded` is still feasible, so it is
        reported but does not block."""
        return [i.node_id for i in self.impacted if i.severity in ("broken", "at_risk")]


# ===============================================================
# Actions — the capability classification
# ===============================================================
class ActionClass(str, Enum):
    """Who may perform an action.

    Note the axis is NOT financial vs non-financial. Messaging a hotel costs
    nothing and cannot be unsent — it is outward-facing and irreversible. An
    action is AGENT_SAFE only if it is internal, or outward-facing with its exact
    text approved in advance (see `Action.message_body`, which is hashed into
    `plan_version`).
    """

    AGENT_SAFE = "agent_safe"
    HUMAN_REQUIRED = "human_required"


class ActionKind(str, Enum):
    # Internal + reversible
    UPDATE_ITINERARY = "update_itinerary"
    UPDATE_CALENDAR = "update_calendar"
    # Outward-facing, requires verbatim pre-approved text
    SEND_HOTEL_MESSAGE = "send_hotel_message"
    # Money moves or a third party must confirm
    PAY_FOR_FLIGHT = "pay_for_flight"
    REBOOK_TRANSFER = "rebook_transfer"
    RESCHEDULE_ACTIVITY = "reschedule_activity"
    CANCEL_ACTIVITY = "cancel_activity"


#: Actions that reach outside the system. AGENT_SAFE is only granted to these
#: when `message_body` is present, so the approved bytes are what gets sent.
OUTWARD_FACING: frozenset[ActionKind] = frozenset({ActionKind.SEND_HOTEL_MESSAGE})

#: Actions that can never be AGENT_SAFE, whatever the model proposes.
ALWAYS_HUMAN: frozenset[ActionKind] = frozenset({
    ActionKind.PAY_FOR_FLIGHT,
    ActionKind.REBOOK_TRANSFER,
})


#: Actions that move a booking in time and must therefore say where to. Without
#: these, executing a plan could not change the schedule, and the revalidation
#: pass would forever report the original breakage.
RESCHEDULING: frozenset[ActionKind] = frozenset({
    ActionKind.SEND_HOTEL_MESSAGE,
    ActionKind.REBOOK_TRANSFER,
    ActionKind.RESCHEDULE_ACTIVITY,
})


class Action(BaseModel):
    """One step required to make an option real."""

    action_id: str
    kind: ActionKind
    action_class: ActionClass
    target_booking_id: str
    description: str
    #: Verbatim text for outward-facing actions. Hashed into `plan_version`, so
    #: approving the plan approves these exact bytes and nothing else.
    message_body: str | None = None
    #: True when the traveller must pay or a provider must confirm.
    requires_payment: bool = False
    #: Where this action moves the target booking to. Required for `RESCHEDULING`
    #: kinds. For a hotel this is the held check-in time, which is what a
    #: late-arrival message means in schedule terms rather than merely in prose.
    #: The disrupted flight's new times are NOT taken from here — they come from
    #: the option's canonical flight, so there is one less thing to restate wrong.
    new_start: datetime | None = None
    new_end: datetime | None = None

    def needs_new_times(self) -> bool:
        return self.kind in RESCHEDULING


# ===============================================================
# Recovery options — the two-plan contract
# ===============================================================
class FlightRef(BaseModel):
    """A replacement flight, always carrying its provenance."""

    carrier: str
    flight_number: str
    origin_iata: str
    destination_iata: str
    departure: datetime
    arrival: datetime
    stops: int = 0
    price_amount: float | None = None
    price_currency: str | None = None
    source: str
    fetched_at: datetime
    uncertain: bool = False
    fallback_used: bool = False


class FlightChoice(BaseModel):
    """Which flight the model picked — and nothing else about it.

    The compiler rewrites every other fact from the provider record, so asking
    for them served only to create ways of being wrong. Both live failures in R1
    were exactly that: gpt-4.1 named SQ425 correctly and wrote its departure as
    16:10 instead of 16:30, then gave AI2380 the departure time belonging to
    SQ421. Neither is possible to express here.

    Identity is the designator plus the departure DATE. A flight number flies once
    a day, so that pair is unique; and the day is the one part that is a real
    decision, since flying tomorrow instead of tonight is the whole A/B choice.
    """

    # Descriptions are deliberately terse. The first version of this model spelled
    # each field out and cost 317 schema tokens for three strings — more than the
    # 13-field `FlightRef` it replaced, which wiped out the saving it existed to
    # make. Only `departure_date` carries one, because its format is the one thing
    # a model cannot infer from the search results it is copying from.
    carrier: str
    flight_number: str
    departure_date: str = Field(description="YYYY-MM-DD, from the search results")


class OptionDraft(BaseModel):
    """One of exactly two options, as the model authors it.

    Everything here is a judgement the model is the right author of: which flight,
    what the traveller pays, what has to happen, how risky it is and why. Nothing
    here is derivable from provider data — see `FlightChoice` for what was removed
    and why.
    """

    option_id: Literal["A", "B"]
    headline: str
    flight: FlightChoice
    #: Not simply the new fare — entitlements may mean the carrier covers it,
    #: which is why this stays a judgement rather than being derived. Kept as a
    #: comment rather than a `description` so it costs no schema tokens; the
    #: prompt is where the model is told this.
    additional_cost_amount: float = Field(ge=0)
    additional_cost_currency: str = "USD"
    #: Node ids from `ImpactSummary` this option puts back into a valid state.
    #: Must cover every blocking node or the compiler rejects the plan.
    resolves: list[str] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"]
    risk_reason: str
    rationale: str
    citations: list[str] = Field(default_factory=list)


class PlanOption(OptionDraft):
    """A compiled option: the draft plus the facts the compiler owns.

    `flight` is narrowed to a full `FlightRef` taken from the provider record, and
    `arrival_delay_minutes` is arithmetic on two provider timestamps. Only
    `core.plan_compiler` builds one of these.
    """

    flight: FlightRef  # type: ignore[assignment]
    arrival_delay_minutes: int = Field(ge=0)


class PlanDraft(BaseModel):
    """The deep agent's structured output — `response_format` target.

    Deliberately smaller than `RecoveryPlan`: no versions, no trip id, no
    timestamps, and no impact list. The agent authors judgement; the compiler
    authors identity and recomputes every checkable fact.

    There used to be an `impacts` field here. The compiler discarded it on every
    run in favour of the dependency graph's own list, so it was pure cost —
    schema tokens on every call plus output tokens spent filling a field that was
    thrown away.
    """

    disruption_summary: str
    options: list[OptionDraft] = Field(default_factory=list)


# ===============================================================
# Server-owned plan identity
# ===============================================================
def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    if isinstance(payload, BaseModel):
        payload = json.loads(payload.model_dump_json())
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(payload: Any, *, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def itinerary_content_hash(bookings: list[Booking]) -> str:
    """Version an itinerary over the fields that determine trip validity.

    PNRs, loyalty numbers and free-form metadata are excluded on purpose. Two
    reasons, and the first is a real bug this projection fixes: redaction strips
    a PNR before the agent ever sees the itinerary, so hashing the raw booking
    made the version the agent was told differ from the version approval is bound
    to. The second is plain correctness — a PNR reissue is not an itinerary change,
    and it must not invalidate a pending approval.

    Bookings are sorted, so re-serialising the same trip in a different order
    hashes the same.
    """
    projection = [
        {
            "booking_id": b.booking_id,
            "type": b.type.value,
            "title": b.title,
            "start": b.start.isoformat(),
            "end": b.end.isoformat(),
            "location": b.location_iata or b.location_name or "",
            "legs": [
                {
                    "carrier": leg.carrier,
                    "flight_number": leg.flight_number,
                    "origin_iata": leg.origin_iata,
                    "destination_iata": leg.destination_iata,
                    "scheduled_departure": leg.scheduled_departure.isoformat(),
                    "scheduled_arrival": leg.scheduled_arrival.isoformat(),
                }
                for leg in b.legs
            ],
        }
        for b in sorted(bookings, key=lambda b: (b.start, b.booking_id))
    ]
    return content_hash(projection, prefix="iv_")


class RecoveryPlan(BaseModel):
    """A compiled, version-bound plan. Only `core.plan_compiler` builds one."""

    trip_id: str
    itinerary_version: str
    plan_version: str
    verified_disruption: VerifiedDisruption
    disruption_summary: str
    impacts: list[ImpactedNode] = Field(default_factory=list)
    options: list[PlanOption] = Field(default_factory=list)
    generated_at: datetime

    def option(self, option_id: str) -> PlanOption | None:
        return next((o for o in self.options if o.option_id == option_id), None)

    def recompute_plan_version(self) -> str:
        """Recompute the hash from current content.

        The executor calls this and compares against the stored `plan_version`.
        A mismatch means the plan was edited after approval, and execution is
        refused. This is the version binding the flow asks for.
        """
        return compute_plan_version(
            trip_id=self.trip_id,
            itinerary_version=self.itinerary_version,
            disruption_summary=self.disruption_summary,
            impacts=self.impacts,
            options=self.options,
        )


def compute_plan_version(
    *,
    trip_id: str,
    itinerary_version: str,
    disruption_summary: str,
    impacts: list[ImpactedNode],
    options: list[PlanOption],
) -> str:
    """Hash everything a human is being asked to approve — and nothing else.

    `generated_at` is excluded on purpose: an identical plan regenerated a minute
    later must hash the same, or a benign retry would invalidate an approval.
    Action `message_body` IS included, so the approved text is the sent text.
    """
    return content_hash(
        {
            "trip_id": trip_id,
            "itinerary_version": itinerary_version,
            "disruption_summary": disruption_summary,
            "impacts": [json.loads(i.model_dump_json()) for i in impacts],
            "options": [json.loads(o.model_dump_json()) for o in options],
        },
        prefix="pv_",
    )


# ===============================================================
# Approval, ledger, handoffs, validation — contracts for R2-R4
# ===============================================================
class ApprovalRecord(BaseModel):
    """Approval is bound to (option, plan_version). Never inferred from chat."""

    trip_id: str
    option_id: Literal["A", "B"]
    plan_version: str
    approved_at: datetime
    approved_by: str


class LedgerEntry(BaseModel):
    """One executed action. Append-only; this is the receipt the UI renders.

    `status` distinguishes what actually happened in the world:

    - `done` — really applied. The traveller's own records changed.
    - `prepared` — composed and ready, but **nothing left the system**. Every
      outward-facing action is this, because the send is simulated.
    - `failed` / `skipped` — self-explanatory.

    The distinction exists because the UI was reporting "Done by agent · sent 174
    chars to bk_hotel_marina" for a message that was never sent. A demo that
    claims to have contacted a hotel is telling the user something false about
    the one thing this system is supposed to be careful about.
    """

    action_id: str
    kind: ActionKind
    action_class: ActionClass
    status: Literal["done", "prepared", "failed", "skipped"]
    performed_by: Literal["agent", "human"]
    detail: str
    at: datetime

    @property
    def settled(self) -> bool:
        """True only when something really changed. `prepared` is not progress."""
        return self.status == "done"


class Handoff(BaseModel):
    """An action only a human can complete. Survives process restarts."""

    action_id: str
    kind: ActionKind
    target_booking_id: str
    description: str
    status: Literal["pending", "confirmed", "abandoned"] = "pending"
    confirmed_at: datetime | None = None


class ExecutionResult(BaseModel):
    """What executing an approved option did, and where the trip stands after."""

    trip_id: str
    plan_version: str
    option_id: str
    ledger: list[LedgerEntry] = Field(default_factory=list)
    handoffs: list[Handoff] = Field(default_factory=list)
    validation: "ValidationResult | None" = None

    @property
    def pending(self) -> list[Handoff]:
        return [h for h in self.handoffs if h.status == "pending"]

    @property
    def closed(self) -> bool:
        return bool(self.validation and self.validation.valid)


class ValidationResult(BaseModel):
    """Output of the deterministic revalidation pass. `valid` closes the case.

    `remaining` blocks close-out; `warnings` does not. The split matters: a
    dependency with 15 minutes of slack is tight but feasible, and plenty of
    real itineraries are booked that way. Treating tightness as invalidity would
    hold the recovery to a higher standard than the trip the traveller booked,
    and no recovery could ever close.
    """

    trip_id: str
    itinerary_version: str
    valid: bool
    remaining: list[ImpactedNode] = Field(default_factory=list)
    warnings: list[ImpactedNode] = Field(default_factory=list)
    pending_handoffs: list[str] = Field(default_factory=list)
    checked_at: datetime
