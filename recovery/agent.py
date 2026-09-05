"""Run entrypoint: build the agent, run it, compile what it returns.

The order of operations here is the architecture, so it is worth being explicit
about why verification and search happen BEFORE the model is built:

- **Verification is a lookup, not a judgement.** Doing it deterministically means
  "stop or escalate if verification is insufficient" is a gate that runs before
  any tokens are spent, rather than an instruction the model may skip. The agent
  still calls `get_flight_status` itself — that is how it sees `uncertain` and
  knows to hedge — but the authoritative record is this module's.

- **The flight candidate set is ours.** The compiler rejects any recommended
  flight that is not in it (rule C4). If the agent supplied that set, the check
  would be circular.

- **The impact list is ours.** The compiled plan carries the graph's impacts, not
  the draft's, so the UI, the executor and the R4 validator all read the same
  numbers.

What the agent contributes is the part none of the above can: which options are
worth offering, what trade-off each represents, and what has to happen to make
them real.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import LLM_MODEL, WORKSPACE_DIR
from core.clock import now
from core.dependency_graph import build_dependency_graph, find_node_by_booking
from core.impact import assess_downstream_impact
from core.plan_compiler import CompileResult, compile_plan, itinerary_version
from core.sources import Sources, sources
from models import (
    Booking,
    BookingType,
    DisruptionEvent,
    DisruptionKind,
    ImpactSummary,
    Itinerary,
    PlanDraft,
    RecoveryPlan,
    TripNodeKind,
    VerifiedDisruption,
)
from recovery.middleware import (
    CostCeilingMiddleware,
    CostLedger,
    ToolBudgetMiddleware,
    TripContextMiddleware,
)
from recovery.prompts import orchestrator_prompt
from recovery.subagents import SUBAGENTS, TOOL_DENYLIST, check_tool_wiring
from recovery.tools import ORCHESTRATOR_TOOLS

logger = logging.getLogger(__name__)

#: Statuses that mean the trip is proceeding as booked.
_HEALTHY = {"on_time", "scheduled", "landed", "active"}

#: How many times the compiler's rejection report is fed back before giving up.
#: One retry catches the ordinary miss (a forgotten action, a wrong delay figure);
#: a model that fails twice on the same recomputed facts is not going to converge,
#: and the violations are more useful to a human than to a third attempt.
MAX_COMPILE_ATTEMPTS = 2


# ===============================================================
# Workspace preparation
# ===============================================================
def redact(itinerary: Itinerary) -> Itinerary:
    """Strip identifiers the agent has no use for before it can ever see them.

    Redaction is here rather than in the prompt because "never echo a PNR" only
    works if a PNR is in the context to echo. Removing it makes the rule
    unbreakable instead of merely instructed.
    """
    clean = itinerary.model_copy(deep=True)
    for booking in clean.bookings:
        for leg in booking.legs:
            leg.booking_reference = None
        booking.metadata = {
            k: v for k, v in booking.metadata.items()
            if k not in {"pnr", "loyalty_number", "passport", "card_last4", "phone"}
        }
    return clean


def prepare_workspace(itinerary: Itinerary, workspace: Path) -> Path:
    """Create the agent's sandbox and mount the redacted itinerary."""
    for sub in ("inputs", "analysis", "drafts", "final"):
        (workspace / sub).mkdir(parents=True, exist_ok=True)
    path = workspace / "inputs" / "itinerary.json"
    path.write_text(redact(itinerary).model_dump_json(indent=2), encoding="utf-8")
    return path


# ===============================================================
# Step 1 — deterministic verification
# ===============================================================
def flight_designator(carrier: str, flight_number: str) -> str:
    """Combine a carrier and flight number into one designator, e.g. 'SQ123'.

    Real itinerary data is inconsistent about whether the flight number already
    carries the carrier prefix: our own fixture stores `SQ123`, plenty of feeds
    store `123`. Naive concatenation gives `SQSQ123` for the first.

    The previous version collapsed a *doubled* prefix, which handled that case
    and mangled the other one: carrier `SQ` with number `AI999` produced
    `SQAI999`, a flight that does not exist. That only mattered once uploaded
    itineraries became possible, and it is the kind of thing that produces a
    confident lookup failure rather than an obvious error.
    """
    carrier = (carrier or "").strip().upper()
    number = (flight_number or "").strip().upper().replace(" ", "")
    if not number:
        return carrier
    # A number that already opens with letters carries its own prefix.
    if number[0].isalpha():
        return number
    return f"{carrier}{number}"


def verify_disruption(
    itinerary: Itinerary,
    *,
    booking_id: str | None = None,
    src: Sources | None = None,
) -> VerifiedDisruption | None:
    """Confirm a disruption against the provider and match it to a booking.

    Returns None when no flight in the itinerary is disrupted. Returns a record
    with `verified=False` when a disruption is indicated but the evidence is too
    weak to act on — the caller must stop or escalate rather than proceed.
    """
    src = src or sources()
    flights = [
        b for b in itinerary.bookings
        if b.type == BookingType.FLIGHT and (booking_id is None or b.booking_id == booking_id)
    ]

    for booking in flights:
        if not booking.legs:
            continue
        leg = booking.legs[0]
        flight_iata = flight_designator(leg.carrier, leg.flight_number)
        date = leg.scheduled_departure.date().isoformat()

        result = src.flights.status(flight_iata, date)
        if result.error and result.value is None:
            return VerifiedDisruption(
                booking_id=booking.booking_id, flight_iata=flight_iata, date=date,
                kind=DisruptionKind.UNKNOWN, verified=False, source=result.source,
                fetched_at=result.fetched_at, uncertain=True,
                note=f"verification insufficient: {result.error}",
            )

        status = str((result.value or {}).get("status", "")).lower()
        if status in _HEALTHY:
            continue

        try:
            kind = DisruptionKind(status)
        except ValueError:
            kind = DisruptionKind.UNKNOWN

        return VerifiedDisruption(
            booking_id=booking.booking_id,
            flight_iata=flight_iata,
            date=date,
            kind=kind,
            delta_minutes=int((result.value or {}).get("delay_minutes") or 0),
            # A disruption of unknown kind is not something to build a plan on:
            # cancelled and delayed lead to different entitlements and different
            # options, so an unrecognised status has to escalate rather than guess.
            verified=kind is not DisruptionKind.UNKNOWN,
            source=result.source,
            fetched_at=result.fetched_at,
            uncertain=result.uncertain or result.fallback_used,
            note=result.error,
        )

    return None


def assess(itinerary: Itinerary, disruption: VerifiedDisruption) -> ImpactSummary:
    """Build the graph and measure the damage. Server-owned, not agent-owned."""
    graph = build_dependency_graph(itinerary.bookings)
    node = find_node_by_booking(graph, disruption.booking_id, kind=TripNodeKind.FLIGHT)
    if node is None:
        return ImpactSummary(disruption_node="", impacted=[])
    return assess_downstream_impact(graph, DisruptionEvent(
        booking_id=disruption.booking_id,
        node_id=node.node_id,
        kind=disruption.kind,
        delta_minutes=disruption.delta_minutes,
        detected_at=disruption.fetched_at,
        source="recovery.agent.assess",
    ))


def flight_candidates(
    itinerary: Itinerary,
    disruption: VerifiedDisruption,
    *,
    src: Sources | None = None,
) -> list[dict[str, Any]]:
    """The authoritative set of flights an option may recommend (compiler rule C4)."""
    booking = next(
        (b for b in itinerary.bookings if b.booking_id == disruption.booking_id), None
    )
    if booking is None or not booking.legs:
        return []
    leg = booking.legs[0]
    result = (src or sources()).flights.search(
        leg.origin_iata, leg.destination_iata,
        leg.scheduled_departure.date().isoformat(),
    )
    # Stamp the envelope's provenance onto each candidate. The compiler rewrites
    # every recommended flight from these records, so `uncertain` and
    # `fallback_used` have to travel with the flight rather than staying on the
    # envelope — otherwise the hedging rule would have nothing to read.
    return [
        {
            **candidate,
            "source": result.source,
            "fetched_at": result.fetched_at.isoformat(),
            "uncertain": result.uncertain,
            "fallback_used": result.fallback_used,
        }
        for candidate in (result.value or {}).get("options", [])
    ]


# ===============================================================
# Agent construction
# ===============================================================
def build_recovery_agent(
    model: Any = None,
    *,
    itinerary: Itinerary,
    disruption: VerifiedDisruption,
    workspace: Path,
    subagents: list[dict] | None = None,
    checkpointer: Any = None,
    cost_ledger: CostLedger | None = None,
) -> Any:
    """Assemble the investigation agent. `subagents=None` uses all of them."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    check_tool_wiring()

    if model is None:
        model = LLM_MODEL
    if isinstance(model, str):
        from recovery.llm import build_model
        model = build_model(model)

    root = workspace.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"workspace {root} does not exist — create it first")

    # Sandboxed deliberately. root_dir="." would put .env and this source tree
    # inside the agent's writable filesystem.
    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    trip_context = TripContextMiddleware(
        trip_id=itinerary.trip_id,
        traveller_first_name=itinerary.traveller_first_name,
        party_size=itinerary.party_size,
        itinerary_version=itinerary_version(itinerary),
        disruption_line=(
            f"{disruption.flight_iata} on {disruption.date} is "
            f"{disruption.kind.value} (source={disruption.source}, "
            f"uncertain={str(disruption.uncertain).lower()})"
        ),
        today=now().isoformat(),
    )

    # Attach the trip context and a cost recorder to every subagent spec
    # ourselves. deepagents inherits the parent's middleware into a subagent only
    # when that subagent is a fork (graph.py:726), and ours are declarative
    # specs — so without this the ceiling counts orchestrator calls only, which is
    # the smaller half of a run, and the trip context never reaches a specialist.
    #
    # One ledger, one recorder per agent. The ledger has to be shared (there is
    # one budget for the run) and the recorders have to be separate (a single
    # instance across five agents can only ever report a total, which tells you
    # nothing about where the tokens went).
    ledger = cost_ledger or CostLedger()

    def stack(label: str) -> list[Any]:
        """The middleware every agent gets, in the order it must run.

        `ToolBudgetMiddleware` is last so it runs innermost, after deepagents'
        own filesystem middleware has injected the tools it is there to strip.
        """
        return [
            trip_context,
            CostCeilingMiddleware(ledger, label),
            ToolBudgetMiddleware(TOOL_DENYLIST.get(label, frozenset()), label),
        ]

    specs = [
        {**spec, "middleware": [*stack(spec["name"]), *spec.get("middleware", [])]}
        for spec in (subagents if subagents is not None else SUBAGENTS)
    ]

    return create_deep_agent(
        name="trip-recovery-orchestrator",
        model=model,
        system_prompt=orchestrator_prompt(),
        tools=list(ORCHESTRATOR_TOOLS),
        subagents=specs,
        backend=backend,
        response_format=PlanDraft,
        middleware=stack("orchestrator"),
        checkpointer=checkpointer,
    )


# ===============================================================
# Draft extraction
# ===============================================================
def extract_draft(result: dict, workspace: Path) -> PlanDraft | None:
    """Pull the `PlanDraft` out of an agent result.

    `response_format` lands the parsed object in `structured_response`; a dict is
    accepted too, since a provider without native structured output round-trips
    through JSON.

    There used to be a fallback here that read `workspace/final/plan.json`,
    justified as insurance for a run that died before returning. It was removed
    once the orchestrator stopped being handed `write_file`: no prompt instructed
    anything to write that path, so the file could never exist. Dead insurance is
    worse than none, because it reads like coverage.
    """
    structured = result.get("structured_response")
    if isinstance(structured, PlanDraft):
        return structured
    if isinstance(structured, dict):
        try:
            return PlanDraft.model_validate(structured)
        except Exception as exc:  # noqa: BLE001
            logger.warning("structured_response did not validate as PlanDraft: %s", exc)
    return None


# ===============================================================
# The run
# ===============================================================
@dataclass
class RunResult:
    """Everything a caller (eval script in R1, UI in R5) needs to render."""

    trip_id: str
    verified_disruption: VerifiedDisruption | None = None
    impact: ImpactSummary | None = None
    plan: RecoveryPlan | None = None
    violations: list[str] = field(default_factory=list)
    draft: PlanDraft | None = None
    attempts: int = 0
    cost: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.plan is not None


def load_itinerary(path: str | Path) -> Itinerary:
    return Itinerary.model_validate(json.loads(Path(path).read_text("utf-8")))


def run_recovery(
    itinerary: Itinerary,
    *,
    model: Any = None,
    workspace: Path | None = None,
    booking_id: str | None = None,
    checkpointer: Any = None,
) -> RunResult:
    """The R1 workflow: verify → investigate → two options → compile.

    No execution and no approval — those are R2 and R3. A successful run ends with
    a version-bound `RecoveryPlan` and nothing having changed in the world.
    """
    workspace = workspace or WORKSPACE_DIR
    out = RunResult(trip_id=itinerary.trip_id)

    # --- Step 1: verify, deterministically, before spending anything ---------
    disruption = verify_disruption(itinerary, booking_id=booking_id)
    if disruption is None:
        out.stopped_reason = "no disruption found on any flight in this itinerary"
        return out
    out.verified_disruption = disruption

    if not disruption.verified:
        out.stopped_reason = (
            f"verification insufficient for {disruption.flight_iata} on "
            f"{disruption.date} — {disruption.note or 'unrecognised status'}. "
            f"Escalate rather than plan a recovery."
        )
        return out

    # --- Steps 2-3: server-owned facts the agent will be checked against -----
    out.impact = assess(itinerary, disruption)
    candidates = flight_candidates(itinerary, disruption)
    if not candidates:
        out.stopped_reason = (
            "no replacement flights available on this route — there is no recovery "
            "to offer, and inventing one is worse than saying so"
        )
        return out

    prepare_workspace(itinerary, workspace)

    # --- Steps 4-5: the agent investigates, the compiler decides -------------
    # A checkpointer is not optional here, whatever the caller passed. The retry
    # below feeds the rejection back as a new turn on the same `thread_id`, and
    # without persistence that thread does not exist: the second attempt starts
    # cold, holding a list of criticisms and no copy of the plan being criticised.
    # Observed live — attempt 1 missed only a hedging phrase, and attempt 2, asked
    # to "fix every item below" with no memory of its own work, dropped every
    # action and invented a flight. The comment claiming a continued thread was
    # aspirational until this line existed.
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        # `PlanDraft` and its enums go into checkpoint state as structured output.
        # LangGraph warns on deserialising unregistered types today and will block
        # them in a future version, so the allowlist is declared now rather than
        # discovered by a broken resume after an upgrade.
        checkpointer = MemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=["models"])
        )

    ledger = CostLedger()
    agent = build_recovery_agent(
        model, itinerary=itinerary, disruption=disruption,
        workspace=workspace, checkpointer=checkpointer, cost_ledger=ledger,
    )
    config = {"configurable": {"thread_id": f"{itinerary.trip_id}:{disruption.date}"}}

    task = (
        f"{disruption.flight_iata} on {disruption.date} has been reported "
        f"{disruption.kind.value}. Verify it, work out everything it breaks, and "
        f"produce exactly two recovery options. Follow your workflow exactly."
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": task}]
    compiled: CompileResult | None = None

    for attempt in range(1, MAX_COMPILE_ATTEMPTS + 1):
        out.attempts = attempt
        result = agent.invoke({"messages": messages}, config=config)

        draft = extract_draft(result, workspace)
        if draft is None:
            out.violations = ["the agent returned no parseable PlanDraft"]
            break
        out.draft = draft

        compiled = compile_plan(
            draft,
            itinerary=itinerary,
            verified_disruption=disruption,
            impact=out.impact,
            flight_candidates=candidates,
        )
        if compiled.ok:
            out.plan = compiled.plan
            out.violations = []
            break

        logger.info("attempt %d rejected:\n%s", attempt, compiled.report())
        out.violations = compiled.violations
        # A new turn on the same thread, so the agent keeps its messages, files and
        # todos. The framing is deliberate: told only what is wrong, a model will
        # rebuild the whole plan and lose the parts that were already correct.
        messages = [{"role": "user", "content": (
            f"{compiled.report()}\n\n"
            "Resubmit the SAME plan with only these items fixed. Keep both options, "
            "the same flights, and every action you already had — anything not "
            "listed above was accepted, and rebuilding it risks losing it."
        )}]

    out.cost = ledger.summary()
    return out
