"""Deterministic trip dependency graph.

Converts bookings into nodes (flight leg, transfer, hotel check-in/out, activity)
and directed "must-precede-by-N-minutes" edges. The impact analyzer walks it; the
revalidation pass in R4 walks the same graph against the updated itinerary, which
is why this is a pure function with no agent coupling.

Ported from tripsure's `tools/dependency_graph.py` — the buffer policy is proven
against its eval scenarios and is not worth re-deriving.

Known limitation: edges are inferred from chronological order only, which is
correct for single-traveller linear itineraries. A branching itinerary (two
travellers splitting up) needs a location-aware pass and is out of scope here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from config import HOTEL_CHECK_IN_BUFFER_MINUTES, TRANSFER_PICKUP_BUFFER_MINUTES
from models import (
    Booking,
    BookingType,
    TripDependencyGraph,
    TripEdge,
    TripNode,
    TripNodeKind,
)

#: Minutes that must sit between two adjacent node kinds for the downstream node
#: to remain feasible. Missing pairs fall back to 30m.
DEFAULT_BUFFER_MINUTES: dict[tuple[TripNodeKind, TripNodeKind], int] = {
    (TripNodeKind.FLIGHT, TripNodeKind.FLIGHT): 120,
    (TripNodeKind.FLIGHT, TripNodeKind.TRANSFER): TRANSFER_PICKUP_BUFFER_MINUTES,
    (TripNodeKind.FLIGHT, TripNodeKind.HOTEL_CHECK_IN): HOTEL_CHECK_IN_BUFFER_MINUTES,
    (TripNodeKind.FLIGHT, TripNodeKind.ACTIVITY): 90,
    (TripNodeKind.TRANSFER, TripNodeKind.HOTEL_CHECK_IN): 0,
    (TripNodeKind.TRANSFER, TripNodeKind.ACTIVITY): 0,
    (TripNodeKind.TRANSFER, TripNodeKind.FLIGHT): 30,
    (TripNodeKind.HOTEL_CHECK_IN, TripNodeKind.ACTIVITY): 0,
    (TripNodeKind.HOTEL_CHECK_IN, TripNodeKind.HOTEL_CHECK_OUT): 0,
    (TripNodeKind.HOTEL_CHECK_OUT, TripNodeKind.FLIGHT): 180,
    (TripNodeKind.HOTEL_CHECK_OUT, TripNodeKind.TRANSFER): 0,
    (TripNodeKind.HOTEL_CHECK_OUT, TripNodeKind.ACTIVITY): 0,
    (TripNodeKind.ACTIVITY, TripNodeKind.ACTIVITY): 0,
    (TripNodeKind.ACTIVITY, TripNodeKind.HOTEL_CHECK_IN): 0,
    (TripNodeKind.ACTIVITY, TripNodeKind.HOTEL_CHECK_OUT): 0,
    (TripNodeKind.ACTIVITY, TripNodeKind.FLIGHT): 180,
    (TripNodeKind.ACTIVITY, TripNodeKind.TRANSFER): 0,
}


def _buffer_for(from_kind: TripNodeKind, to_kind: TripNodeKind) -> int:
    return DEFAULT_BUFFER_MINUTES.get((from_kind, to_kind), 30)


def _nodes_from(bookings: list[Booking]) -> list[TripNode]:
    """Explode bookings into per-segment nodes.

    A flight node is timed at its *arrival*, because what breaks downstream is
    when the traveller lands, not when they were meant to leave.
    """
    nodes: list[TripNode] = []
    for b in bookings:
        if b.type == BookingType.FLIGHT:
            if b.legs:
                for i, leg in enumerate(b.legs):
                    nodes.append(TripNode(
                        node_id=f"{b.booking_id}:leg{i}",
                        booking_id=b.booking_id,
                        kind=TripNodeKind.FLIGHT,
                        label=f"{leg.carrier}{leg.flight_number} "
                              f"{leg.origin_iata}→{leg.destination_iata}",
                        scheduled_at=leg.scheduled_arrival,
                    ))
            else:
                nodes.append(TripNode(
                    node_id=b.booking_id, booking_id=b.booking_id,
                    kind=TripNodeKind.FLIGHT, label=b.title, scheduled_at=b.end,
                ))
        elif b.type == BookingType.TRANSFER:
            nodes.append(TripNode(
                node_id=b.booking_id, booking_id=b.booking_id,
                kind=TripNodeKind.TRANSFER, label=b.title, scheduled_at=b.start,
            ))
        elif b.type == BookingType.HOTEL:
            nodes.append(TripNode(
                node_id=f"{b.booking_id}:checkin", booking_id=b.booking_id,
                kind=TripNodeKind.HOTEL_CHECK_IN, label=f"Check in: {b.title}",
                scheduled_at=b.start,
            ))
            nodes.append(TripNode(
                node_id=f"{b.booking_id}:checkout", booking_id=b.booking_id,
                kind=TripNodeKind.HOTEL_CHECK_OUT, label=f"Check out: {b.title}",
                scheduled_at=b.end,
            ))
        elif b.type == BookingType.ACTIVITY:
            nodes.append(TripNode(
                node_id=b.booking_id, booking_id=b.booking_id,
                kind=TripNodeKind.ACTIVITY, label=b.title, scheduled_at=b.start,
            ))
    return nodes


def build_dependency_graph(bookings: list[Booking]) -> TripDependencyGraph:
    """Build the graph. Pure function of `bookings`."""
    if not bookings:
        return TripDependencyGraph()

    nodes = _nodes_from(bookings)
    nodes.sort(key=lambda n: n.scheduled_at)

    edges: list[TripEdge] = []
    for prev, curr in zip(nodes, nodes[1:]):
        # Check-in → check-out of the same hotel: the gap is the stay length, so
        # adding a buffer on top would be nonsense.
        if prev.booking_id == curr.booking_id and prev.kind != curr.kind:
            edges.append(TripEdge(
                from_node=prev.node_id, to_node=curr.node_id,
                min_gap_minutes=0, rationale="intra-booking",
            ))
            continue
        gap = _buffer_for(prev.kind, curr.kind)
        edges.append(TripEdge(
            from_node=prev.node_id, to_node=curr.node_id, min_gap_minutes=gap,
            rationale=f"{prev.kind.value} → {curr.kind.value} min buffer {gap}m",
        ))

    # Explicit edges go first so that on a tie they keep their rationale. In an
    # undisrupted itinerary a presence edge often duplicates the chronological one
    # exactly; "you must have landed at SIN first" is the more useful thing to read
    # in a violation message than "flight → transfer min buffer 30m".
    return TripDependencyGraph(
        nodes=nodes,
        edges=_dedupe(_presence_edges(bookings, nodes) + edges),
    )


def _dedupe(edges: list[TripEdge]) -> list[TripEdge]:
    """One edge per (from, to), keeping the strictest buffer; first wins a tie."""
    best: dict[tuple[str, str], TripEdge] = {}
    for edge in edges:
        key = (edge.from_node, edge.to_node)
        if key not in best or edge.min_gap_minutes > best[key].min_gap_minutes:
            best[key] = edge
    return list(best.values())


def _presence_edges(
    bookings: list[Booking], nodes: list[TripNode]
) -> list[TripEdge]:
    """You cannot be somewhere before you have arrived there.

    Chronological inference has a hole that only shows up once times start moving.
    Two instances, both found by testing rather than by reading:

    - Move the flight from 16:15 to 23:15 and leave the airport transfer at 17:00.
      Re-sorting puts the transfer *before* the arrival, where the gap is
      comfortably positive and the graph reports no problem — while the
      traveller's car turns up six hours before their plane.
    - Move the Night Safari to the 14th. The trip arrives on the 15th, so the
      activity is simply impossible; chronologically it sorts first and every gap
      after it is enormous, so the itinerary "validates".

    Both are the same missing fact: a booking at location X depends on the flight
    that brings the traveller to X, and that dependency does not stop being true
    when the times stop agreeing. Making it explicit is what lets the close-out
    check refuse to close, and what lets the plan compiler reject a time that
    would strand someone.

    Matching is by `location_iata`, so a booking without one contributes no edge.
    That is the honest limit here: real coverage needs the destination resolved
    from a place name, which is a geocoding problem rather than a graph one.
    """
    by_id = {n.node_id: n for n in nodes}
    arrivals: dict[str, list[TripNode]] = {}
    for booking in bookings:
        if booking.type != BookingType.FLIGHT:
            continue
        for i, leg in enumerate(booking.legs):
            node = by_id.get(f"{booking.booking_id}:leg{i}")
            if node is not None:
                arrivals.setdefault(leg.destination_iata, []).append(node)

    out: list[TripEdge] = []
    for booking in bookings:
        if booking.type == BookingType.FLIGHT or not booking.location_iata:
            continue
        for node in (n for n in nodes if n.booking_id == booking.booking_id):
            # Check-out is not a presence constraint: it happens after the stay,
            # and tying it to the inbound arrival would double-count the check-in
            # edge and forbid a same-day turnaround.
            if node.kind is TripNodeKind.HOTEL_CHECK_OUT:
                continue
            gap = _buffer_for(TripNodeKind.FLIGHT, node.kind)
            for arrival in arrivals.get(booking.location_iata, []):
                out.append(TripEdge(
                    from_node=arrival.node_id, to_node=node.node_id,
                    min_gap_minutes=gap,
                    rationale=(
                        f"you must have landed at {booking.location_iata} "
                        f"at least {gap}m before this"
                    ),
                ))
    return out


# ---------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------
def outgoing_edges(graph: TripDependencyGraph, node_id: str) -> list[TripEdge]:
    return [e for e in graph.edges if e.from_node == node_id]


def find_node(graph: TripDependencyGraph, node_id: str) -> TripNode | None:
    return next((n for n in graph.nodes if n.node_id == node_id), None)


def find_node_by_booking(
    graph: TripDependencyGraph, booking_id: str, *, kind: TripNodeKind | None = None
) -> TripNode | None:
    for n in graph.nodes:
        if n.booking_id == booking_id and (kind is None or n.kind == kind):
            return n
    return None


def latest_permissible(
    graph: TripDependencyGraph, node_id: str
) -> datetime | None:
    """Latest time `node_id` can happen without breaking its first successor.

    This is the "latest permissible change time" the impact graph is meant to
    carry. Returns None for a terminal node.
    """
    node = find_node(graph, node_id)
    if node is None:
        return None
    successors = outgoing_edges(graph, node_id)
    if not successors:
        return None
    limits: list[datetime] = []
    for edge in successors:
        target = find_node(graph, edge.to_node)
        if target is not None:
            limits.append(target.scheduled_at - timedelta(minutes=edge.min_gap_minutes))
    return min(limits) if limits else None
