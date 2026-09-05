"""Deterministic downstream-impact analyzer.

- CANCELLED / DIVERTED: every reachable downstream node is `broken`.
- DELAYED: propagate the delta through outgoing edges and classify by remaining
  slack — broken (< 0m), at_risk (< AT_RISK_MINUTES), degraded (feasible).
- GATE_CHANGE / UNKNOWN: informational unless `delta_minutes` is set.

Ported from tripsure's `tools/impact_analyzer.py`, with two additions: impacted
nodes carry `booking_id` and `label` so the UI never has to re-join against the
graph, and `validate_itinerary` reuses the same walk for the R4 close-out check.
"""

from __future__ import annotations

from datetime import timedelta

from config import AT_RISK_MINUTES
from core.clock import now
from core.dependency_graph import (
    build_dependency_graph,
    find_node,
    find_node_by_booking,
    outgoing_edges,
)
from models import (
    Booking,
    DisruptionEvent,
    DisruptionKind,
    ImpactedNode,
    ImpactSummary,
    Severity,
    TripDependencyGraph,
    TripNode,
    TripNodeKind,
    ValidationResult,
    itinerary_content_hash,
)

BROKEN_MINUTES = 0


def _classify(slack_minutes: int) -> tuple[Severity, str]:
    if slack_minutes < BROKEN_MINUTES:
        return "broken", f"missed by {abs(slack_minutes)}m"
    if slack_minutes < AT_RISK_MINUTES:
        return "at_risk", f"only {slack_minutes}m slack left (< {AT_RISK_MINUTES}m)"
    return "degraded", f"{slack_minutes}m slack retained"


def _impacted(node: TripNode, severity: Severity, reason: str) -> ImpactedNode:
    return ImpactedNode(
        node_id=node.node_id,
        booking_id=node.booking_id,
        label=node.label,
        severity=severity,
        reason=reason,
    )


def assess_downstream_impact(
    graph: TripDependencyGraph, disruption: DisruptionEvent
) -> ImpactSummary:
    """Walk from the disrupted node and classify every reachable node."""
    if disruption.kind in (DisruptionKind.CANCELLED, DisruptionKind.DIVERTED):
        return _all_downstream_broken(graph, disruption)
    if disruption.kind == DisruptionKind.DELAYED or disruption.delta_minutes:
        return _propagate_delay(graph, disruption)
    return ImpactSummary(disruption_node=disruption.node_id, impacted=[])


def _all_downstream_broken(
    graph: TripDependencyGraph, disruption: DisruptionEvent
) -> ImpactSummary:
    impacted: list[ImpactedNode] = []
    seen: set[str] = set()
    queue: list[str] = [disruption.node_id]
    while queue:
        for edge in outgoing_edges(graph, queue.pop(0)):
            if edge.to_node in seen:
                continue
            seen.add(edge.to_node)
            node = find_node(graph, edge.to_node)
            if node is None:
                continue
            impacted.append(
                _impacted(node, "broken", f"upstream {disruption.kind.value}")
            )
            queue.append(node.node_id)
    return ImpactSummary(disruption_node=disruption.node_id, impacted=impacted)


def _propagate_delay(
    graph: TripDependencyGraph, disruption: DisruptionEvent
) -> ImpactSummary:
    """Push the delay forward through the DAG, carrying accumulated slippage."""
    impacted: list[ImpactedNode] = []
    eff: dict[str, timedelta] = {
        disruption.node_id: timedelta(minutes=disruption.delta_minutes)
    }
    seen: set[str] = {disruption.node_id}
    queue: list[str] = [disruption.node_id]

    while queue:
        curr_id = queue.pop(0)
        curr = find_node(graph, curr_id)
        if curr is None:
            continue
        upstream_delta = eff.get(curr_id, timedelta(0))
        for edge in outgoing_edges(graph, curr_id):
            target = find_node(graph, edge.to_node)
            if target is None:
                continue
            required = (
                curr.scheduled_at + upstream_delta
                + timedelta(minutes=edge.min_gap_minutes)
            )
            slack_minutes = int((target.scheduled_at - required).total_seconds() // 60)
            severity, fragment = _classify(slack_minutes)

            slippage = max(timedelta(0), required - target.scheduled_at)
            if slippage > eff.get(target.node_id, timedelta(-1)):
                eff[target.node_id] = slippage

            if target.node_id not in seen:
                seen.add(target.node_id)
                impacted.append(_impacted(
                    target, severity,
                    f"upstream delay {disruption.delta_minutes}m — {fragment}",
                ))
                queue.append(target.node_id)

    return ImpactSummary(disruption_node=disruption.node_id, impacted=impacted)


# ---------------------------------------------------------------
# Feasibility check for a *proposed* replacement flight
# ---------------------------------------------------------------
def impact_of_replacement(
    bookings: list[Booking],
    disrupted_booking_id: str,
    new_arrival,
) -> ImpactSummary:
    """What still breaks if the traveller lands at `new_arrival` instead.

    This is how an option is checked for feasibility rather than plausibility:
    substitute the arrival time, rebuild the graph, and re-run the same walk. An
    option that leaves a blocking node behind has not recovered the trip, however
    convincing its prose.
    """
    graph = build_dependency_graph(bookings)
    node = find_node_by_booking(graph, disrupted_booking_id, kind=TripNodeKind.FLIGHT)
    if node is None:
        return ImpactSummary(disruption_node="", impacted=[])

    delta = int((new_arrival - node.scheduled_at).total_seconds() // 60)
    if delta <= 0:
        return ImpactSummary(disruption_node=node.node_id, impacted=[])

    return _propagate_delay(graph, DisruptionEvent(
        booking_id=disrupted_booking_id,
        node_id=node.node_id,
        kind=DisruptionKind.DELAYED,
        delta_minutes=delta,
        detected_at=now(),
        source="core.impact.impact_of_replacement",
    ))


# ---------------------------------------------------------------
# R4 close-out check
# ---------------------------------------------------------------
def validate_itinerary(
    trip_id: str,
    bookings: list[Booking],
    *,
    pending_handoffs: list[str] | None = None,
) -> ValidationResult:
    """Is the whole itinerary internally consistent again?

    Deliberately checks the *entire* graph rather than only the nodes the
    disruption touched: an executed recovery can break something the original
    disruption never reached — a rescheduled activity colliding with check-out,
    say — and a disruption-scoped check would call that trip valid.
    """
    graph = build_dependency_graph(bookings)
    remaining: list[ImpactedNode] = []
    warnings: list[ImpactedNode] = []

    for edge in graph.edges:
        src, dst = find_node(graph, edge.from_node), find_node(graph, edge.to_node)
        if src is None or dst is None:
            continue
        gap = int((dst.scheduled_at - src.scheduled_at).total_seconds() // 60)
        slack = gap - edge.min_gap_minutes
        if slack >= AT_RISK_MINUTES:
            continue
        severity, fragment = _classify(slack)
        node = _impacted(dst, severity, f"{edge.rationale} — {fragment}")
        # Only an infeasible dependency blocks close-out. See ValidationResult.
        (remaining if severity == "broken" else warnings).append(node)

    pending = list(pending_handoffs or [])
    return ValidationResult(
        trip_id=trip_id,
        itinerary_version=itinerary_content_hash(bookings),
        valid=not remaining and not pending,
        remaining=remaining,
        warnings=warnings,
        pending_handoffs=pending,
        checked_at=now(),
    )
