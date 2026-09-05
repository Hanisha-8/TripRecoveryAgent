"""LLM-facing tools. Thin wrappers over `core/`.

Three rules every wrapper here follows, each one learned from a real bug in
tripsure's equivalent layer:

1. **Return the serialised provenance envelope.** Handing back a pydantic object
   gives the model a Python repr and silently drops `source` / `fetched_at` /
   `uncertain` / `fallback_used` — the exact fields the plan compiler enforces.

2. **Resolve workspace paths.** The agent's filesystem is virtual and rooted at
   the workspace, so it naturally passes `/inputs/itinerary.json`. A plain
   `Path(...).read_text()` on that resolves to the OS root and fails with ENOENT.

3. **Never build a tool inside a loop.** Late-binding closures make every wrapper
   call whichever callable the loop bound last. Everything here is module scope.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.tools import tool

from config import WORKSPACE_DIR
from core import citations
from core.sources import sources
from core.dependency_graph import (
    build_dependency_graph,
    find_node_by_booking,
    latest_permissible,
)
from core.impact import assess_downstream_impact, impact_of_replacement
from core.clock import now
from models import (
    DisruptionEvent,
    DisruptionKind,
    Itinerary,
    ToolResult,
    TripNodeKind,
)

WORKSPACE_ROOT = Path(WORKSPACE_DIR)


def resolve_workspace_path(path: str) -> Path:
    """Map an agent-visible path onto a real one.

    A path that already exists is used as-is, so tests and callers outside the
    agent can pass real paths; anything else is treated as workspace-relative.
    """
    p = Path(path)
    return p if p.exists() else WORKSPACE_ROOT / str(path).lstrip("/")


def _envelope(result: Any) -> str:
    if isinstance(result, ToolResult):
        return result.model_dump_json()
    return json.dumps(result, default=str)


def _load_itinerary(path: str) -> Itinerary | str:
    """Return the parsed itinerary, or a JSON error string for the model."""
    resolved = resolve_workspace_path(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return json.dumps({
            "error": f"cannot read itinerary at {path!r} (resolved {str(resolved)!r}): {exc}",
            "hint": "Pass the path exactly as you see it, e.g. /inputs/itinerary.json",
        })
    try:
        return Itinerary.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — surface validation detail to the model
        return json.dumps({"error": f"itinerary failed validation: {exc}"})


# ===============================================================
# Orchestrator
# ===============================================================
@tool
def get_flight_status(flight_iata: str, date: str) -> str:
    """Get the current status of a flight. `flight_iata` e.g. 'SQ123', `date` 'YYYY-MM-DD'.

    Returns the full provenance envelope. Always read `source`, `uncertain` and
    `fallback_used` before trusting it: a fallback means the live providers were
    unreachable, and every claim you build on it inherits that uncertainty.
    """
    return _envelope(sources().flights.status(flight_iata, date))


# ===============================================================
# impact-analyst
# ===============================================================
@tool
def analyse_downstream_impact(
    itinerary_path: str,
    disrupted_booking_id: str,
    disruption_kind: str = "cancelled",
    delay_minutes: int = 0,
) -> str:
    """Compute which downstream bookings break, from the trip dependency graph.

    `disruption_kind` is one of cancelled, delayed, diverted, gate_change. Returns
    each impacted node with its severity (broken / at_risk / degraded), the reason,
    and the latest time it could still happen.

    Use this rather than reasoning about connection times yourself — it applies the
    real minimum connection times and hotel check-in buffers, and your own
    arithmetic will disagree with it and be wrong.
    """
    itinerary = _load_itinerary(itinerary_path)
    if isinstance(itinerary, str):
        return itinerary

    graph = build_dependency_graph(itinerary.bookings)
    node = find_node_by_booking(graph, disrupted_booking_id, kind=TripNodeKind.FLIGHT)
    if node is None:
        return json.dumps({
            "error": f"no flight node for booking_id {disrupted_booking_id!r}",
            "known_booking_ids": [b.booking_id for b in itinerary.bookings],
        })

    try:
        kind = DisruptionKind(disruption_kind.strip().lower())
    except ValueError:
        kind = DisruptionKind.UNKNOWN

    summary = assess_downstream_impact(graph, DisruptionEvent(
        booking_id=disrupted_booking_id,
        node_id=node.node_id,
        kind=kind,
        delta_minutes=int(delay_minutes or 0),
        detected_at=now(),
        source="recovery.tools.analyse_downstream_impact",
    ))

    return json.dumps({
        "disruption_node": summary.disruption_node,
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "blocking_node_ids": summary.blocking_node_ids(),
        "impacted": [
            {
                "node_id": i.node_id,
                "booking_id": i.booking_id,
                "label": i.label,
                "severity": i.severity,
                "reason": i.reason,
                "latest_permissible": (
                    lp.isoformat()
                    if (lp := latest_permissible(graph, i.node_id)) else None
                ),
            }
            for i in summary.impacted
        ],
    }, default=str)


# ===============================================================
# options-finder
# ===============================================================
@tool
def read_impact() -> str:
    """Read the impact analysis. Call this BEFORE searching for flights.

    Returns "NOT YET DETERMINED" if the impact-analyst has not written its file.
    Act on what this returns, never on your own guess about whether a file exists.
    """
    path = WORKSPACE_ROOT / "analysis" / "impact.md"
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return (
            "NOT YET DETERMINED — the impact analysis has not been written. "
            "Return exactly `BLOCKED: impact not yet determined` and stop."
        )
    return path.read_text(encoding="utf-8")


@tool
def search_replacement_flights(
    origin: str, destination: str, date: str, constraints: dict | None = None
) -> str:
    """Search replacement flights. IATA `origin`/`destination`, `date` 'YYYY-MM-DD'.

    Results include flights on later dates too, which is how an overnight Option B
    becomes possible. Optional `constraints`: max_stops, price_ceiling, depart_after.

    You may ONLY recommend a flight that appears in these results. If two
    consecutive searches return nothing, stop and say so — never invent a flight,
    a price or a time.

    Read `required_hedge_phrase` on the result. When it is set, every option built
    on these flights must contain that exact phrase in its rationale.
    """
    result = sources().flights.search(origin, destination, date, constraints or {})
    payload = json.loads(result.model_dump_json())
    # The hedging requirement rides along with the data that triggers it. Asking
    # the model to infer it from `uncertain` / `fallback_used` two layers up was
    # the one rule it kept missing live — not because it disagreed, but because
    # the obligation lived in the prompt while the trigger lived in the payload.
    payload["required_hedge_phrase"] = (
        "verify at booking" if (result.uncertain or result.fallback_used) else None
    )
    return json.dumps(payload, default=str)


@tool
def check_option_feasibility(
    itinerary_path: str, disrupted_booking_id: str, new_arrival: str
) -> str:
    """Check what a candidate arrival time still leaves broken.

    `new_arrival` is an ISO-8601 datetime with offset, e.g.
    '2026-09-15T23:15:00+08:00'. Returns the bookings that remain broken or at
    risk if the traveller lands then.

    This runs the same check the plan compiler runs when it accepts or rejects
    your plan, so use it on every option before you commit to one. Anything it
    reports still-blocking needs an action in that option, or the plan is rejected.
    """
    itinerary = _load_itinerary(itinerary_path)
    if isinstance(itinerary, str):
        return itinerary

    try:
        arrival = datetime.fromisoformat(new_arrival.strip())
    except ValueError:
        return json.dumps({
            "error": f"unparseable new_arrival {new_arrival!r}",
            "hint": "use ISO-8601 with offset, e.g. 2026-09-15T23:15:00+08:00",
        })
    if arrival.tzinfo is None:
        return json.dumps({
            "error": f"new_arrival {new_arrival!r} has no UTC offset",
            "hint": "the trip spans timezones, so a naive datetime is ambiguous",
        })

    residual = impact_of_replacement(itinerary.bookings, disrupted_booking_id, arrival)
    still_blocking = [
        {
            "node_id": i.node_id, "booking_id": i.booking_id, "label": i.label,
            "severity": i.severity, "reason": i.reason,
        }
        for i in residual.impacted if i.severity in ("broken", "at_risk")
    ]
    return json.dumps({
        "new_arrival": arrival.isoformat(),
        "feasible_without_action": not still_blocking,
        "still_blocking": still_blocking,
        "degraded": [
            {"node_id": i.node_id, "label": i.label, "reason": i.reason}
            for i in residual.impacted if i.severity == "degraded"
        ],
    }, default=str)


# ===============================================================
# policy-checker
# ===============================================================
@tool
def retrieve_policy_context(query: str, k: int = 3) -> str:
    """Retrieve the top-k airline policy chunks relevant to `query`.

    Each hit carries a `chunk_id` you MUST cite in square brackets, e.g.
    `[disruption_care:SQ-CANCEL-ACCOM-01]`. Run several differently-worded
    queries — one query rarely covers every entitlement category. Try wording
    close to how policy is written: "hotel accommodation overnight", "meal
    voucher threshold", "cash compensation cancellation", "refund unflown
    portion", "contact telephone".

    If nothing relevant comes back, the corpus does not cover it. Say so and
    never fill the gap from memory.
    """
    return _envelope(sources().policy.retrieve(query, k))


@tool
def airline_contact_lookup(carrier_iata: str) -> str:
    """Look up an airline's disruption-desk contact by IATA code (e.g. 'SQ').

    Never guess or recall a phone number. If this returns an error, say no
    verified number is on file and point the traveller at the number printed on
    their ticket. Numbers you state that did not come from here are rejected by
    the citation audit.
    """
    return json.dumps(citations.contact_for(carrier_iata), default=str)


# ===============================================================
# critic
# ===============================================================
@tool
def citation_checker(
    text: str, disruption_kind: str = "cancelled", is_entitlements_doc: bool = False
) -> str:
    """Audit every citation and figure in `text` against the live policy corpus.

    Checks that each `[domain:ID]` resolves; that each monetary amount, duration
    and phone number appears in a chunk cited in the same section (phone numbers
    must come from the contact fixture); and that a cancellation does not cite a
    delay rule for compensation. Set `is_entitlements_doc=True` when checking the
    entitlements file to additionally require all six categories.

    Returns PASS or an itemised FAIL report. Anything it reports is BLOCKING.
    """
    return citations.audit_citations(
        text, disruption_kind, require_categories=is_entitlements_doc
    ).report()


@tool
def read_entitlements() -> str:
    """Read the determined entitlements. Call this BEFORE searching for flights.

    Returns the entitlements document, or a string starting with "NOT YET
    DETERMINED" if the policy-checker has not produced it. Report
    `BLOCKED: entitlements not yet determined` only if THIS TOOL says so — never
    on your own assumption about whether a file exists.
    """
    path = WORKSPACE_ROOT / "analysis" / "entitlements.md"
    # A mechanical answer, not a judgement. tripsure's first full run deadlocked
    # here: its subagent was told "read the file; if it does not exist, return
    # BLOCKED", had to judge existence itself, and returned BLOCKED while the file
    # sat there having already passed its citation audit.
    if not path.is_file():
        return (
            "NOT YET DETERMINED — the policy-checker has not written "
            "/analysis/entitlements.md yet. Report exactly "
            "'BLOCKED: entitlements not yet determined' and stop."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return "NOT YET DETERMINED — the entitlements file is empty."
    return f"ENTITLEMENTS DETERMINED. Proceed. Contents follow.\n\n{text}"


# ---------------------------------------------------------------
# Tool loadouts — scoping is the safety boundary, so it is explicit.
# ---------------------------------------------------------------
ORCHESTRATOR_TOOLS = [get_flight_status]
IMPACT_TOOLS = [analyse_downstream_impact]
POLICY_TOOLS = [retrieve_policy_context, airline_contact_lookup]
OPTIONS_TOOLS = [
    read_entitlements, read_impact, search_replacement_flights,
    check_option_feasibility,
]
CRITIC_TOOLS = [
    citation_checker, retrieve_policy_context, read_entitlements,
    check_option_feasibility,
]
