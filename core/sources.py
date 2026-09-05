"""Where the outside world comes from — three protocols and one mock set.

Everything the system cannot compute for itself arrives through here: flight
status, seat availability and fares, airline contact details, and the policy
corpus. Each is a `Protocol`, so an implementation is anything with the right
shape rather than a subclass of ours.

**Why this exists.** "Mock" used to be a fallback baked into each module —
`core.providers` read `data/mock_flights.json` directly, `core.citations` opened
`data/airline_contacts.json` at import, `core.policy` had the corpus path as a
module constant. Everything worked, but mock was not *one implementation among
several*: it was the only one, wired in at four separate places. Adding a live
provider meant editing business logic rather than choosing a source.

Two things fall out of the boundary being explicit:

1. A live source can be written and selected without touching the planner, the
   compiler, the entitlement builder or the agent's tools.
2. Tests can inject a source per case. The uncertainty path in particular was
   only reachable by setting `FAIL_MODE` for the whole process; now a single
   test can hand over a degraded flight source and leave everything else alone.

`DATA_MODE` selects the set. Only `mock` is implemented; asking for anything
else raises rather than silently falling back, because a silent fallback to
fixtures is exactly how a demo gets mistaken for a deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from config import DATA_MODE, DATA_DIR, FAIL_MODE, ROOT_DIR
from core.clock import now
from core.policy import PolicyChunk, load_chunks, score_chunk
from models import ToolResult


# ===============================================================
# Protocols
# ===============================================================
@runtime_checkable
class FlightSource(Protocol):
    """Flight status and replacement availability.

    Both methods return a `ToolResult`, so `source`, `fetched_at`, `uncertain`
    and `fallback_used` travel with the data. The plan compiler rewrites every
    recommended flight from these records, so their provenance is not decoration
    — it is what makes a fare in front of a traveller checkable.
    """

    name: str

    def status(self, flight_iata: str, date: str) -> ToolResult: ...

    def search(
        self, origin: str, destination: str, date: str,
        constraints: dict[str, Any] | None = None,
    ) -> ToolResult: ...


@runtime_checkable
class ContactSource(Protocol):
    """Airline disruption-desk details. A lookup, never a recollection."""

    name: str

    def contact(self, carrier_iata: str) -> dict[str, Any]: ...


@runtime_checkable
class PolicySource(Protocol):
    """The retrievable policy corpus, plus the non-retrievable invariants."""

    name: str

    def chunks(self) -> tuple[PolicyChunk, ...]: ...

    def retrieve(self, query: str, k: int = 3) -> ToolResult: ...

    def grounding(self) -> str: ...


@dataclass(frozen=True)
class Sources:
    """One configured set. Passed around rather than imported per module."""

    mode: str
    flights: FlightSource
    contacts: ContactSource
    policy: PolicySource

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"


# ===============================================================
# Mock: flight status and search
# ===============================================================
class MockFlightSource:
    """Fixture-backed flights, with optional injected failure.

    `FAIL_MODE` is a constructor argument rather than a module-level read, so a
    test can build a degraded source without changing the environment for every
    other test in the process.

    `stale` is the nastiest of the three failures and the reason the option
    exists: the data comes back, looks complete, and is simply out of date. It
    is the case most likely to survive review.
    """

    name = "fixture"
    _NOTES = {
        "http_500": "live provider returned 500; served from local fixture",
        "timeout": "live provider timed out; served from local fixture",
        "stale": "live provider data older than the cache TTL",
    }

    def __init__(
        self,
        fixture: Path | None = None,
        fail_mode: str = FAIL_MODE,
    ) -> None:
        self._fixture_path = fixture or (DATA_DIR / "mock_flights.json")
        self.fail_mode = (fail_mode or "none").lower()

    @property
    def _data(self) -> dict[str, Any]:
        return _read_json(self._fixture_path)

    def _degraded(self) -> tuple[bool, str | None]:
        if self.fail_mode == "none":
            return False, None
        return True, self._NOTES.get(
            self.fail_mode, f"unknown FAIL_MODE {self.fail_mode!r}"
        )

    def status(self, flight_iata: str, date: str) -> ToolResult:
        uncertain, note = self._degraded()
        fetched = now()
        if self.fail_mode == "stale":
            fetched = fetched - timedelta(hours=6)

        match = next(
            (
                f for f in self._data.get("status", [])
                if f["flight_iata"].upper() == flight_iata.strip().upper()
                and f["date"] == date.strip()
            ),
            None,
        )
        if match is None:
            return ToolResult(
                tool_name="get_flight_status", value=None, fetched_at=fetched,
                source=self.name, uncertain=True, fallback_used=True,
                error=f"no status on file for {flight_iata} on {date}",
            )
        return ToolResult(
            tool_name="get_flight_status", value=match, fetched_at=fetched,
            source=self.name, uncertain=uncertain,
            # There is no live provider behind this source, so every answer it
            # gives is a fallback and every claim built on it inherits that.
            fallback_used=True, error=note,
        )

    def search(
        self, origin: str, destination: str, date: str,
        constraints: dict[str, Any] | None = None,
    ) -> ToolResult:
        constraints = constraints or {}
        uncertain, note = self._degraded()

        try:
            from_date = datetime.fromisoformat(date.strip()).date()
        except ValueError:
            return ToolResult(
                tool_name="search_replacement_flights", value=None,
                fetched_at=now(), source=self.name, uncertain=True,
                fallback_used=True,
                error=f"unparseable date {date!r}, expected YYYY-MM-DD",
            )

        max_stops = int(constraints.get("max_stops", 9))
        ceiling = constraints.get("price_ceiling")
        depart_after = constraints.get("depart_after")

        options: list[dict[str, Any]] = []
        for flight in self._data.get("search", []):
            if flight["origin_iata"] != origin.strip().upper():
                continue
            if flight["destination_iata"] != destination.strip().upper():
                continue
            departure = datetime.fromisoformat(flight["departure"])
            # Later dates are kept deliberately: an overnight Option B is only
            # possible if tomorrow's flights are in the result set.
            if departure.date() < from_date:
                continue
            if flight["stops"] > max_stops:
                continue
            if ceiling is not None and flight["price_amount"] > float(ceiling):
                continue
            if depart_after and departure < datetime.fromisoformat(depart_after):
                continue
            options.append(flight)

        options.sort(key=lambda f: f["arrival"])
        return ToolResult(
            tool_name="search_replacement_flights",
            value={
                "options": options,
                "search": {
                    "origin": origin, "destination": destination,
                    "from_date": from_date.isoformat(), "constraints": constraints,
                },
            },
            fetched_at=now(), source=self.name, uncertain=uncertain,
            fallback_used=True, error=note,
        )


# ===============================================================
# Mock: airline contacts
# ===============================================================
class FixtureContactSource:
    """Contacts from `data/airline_contacts.json`.

    An unknown carrier gets an error and advice, never a number. The one thing
    this must not do is help: a plausible-looking invented phone number is worse
    than no number at all.
    """

    name = "fixture"

    def __init__(self, fixture: Path | None = None) -> None:
        self._fixture_path = fixture or (DATA_DIR / "airline_contacts.json")

    def contact(self, carrier_iata: str) -> dict[str, Any]:
        try:
            contacts = _read_json(self._fixture_path)
        except (OSError, json.JSONDecodeError) as exc:
            return {"error": f"contact fixture unreadable: {exc}"}

        hit = contacts.get((carrier_iata or "").strip().upper())
        if not hit:
            return {
                "error": f"no verified contact on file for carrier {carrier_iata!r}",
                "advice": "Direct the traveller to the number printed on their "
                          "ticket. Do not state a number.",
                "cite": "disruption_care:CONTACT-GENERIC-01",
            }
        out = dict(hit)
        # Provenance kept on the record for whoever swaps in a live source. It is
        # deliberately NOT rendered in the UI: it is a note to a maintainer, and
        # showing it to a traveller was a wall of red text about drama-range
        # phone numbers.
        out["_warning"] = contacts.get("_meta", {}).get("WARNING", "")
        out["_source"] = self.name
        return out


# ===============================================================
# Mock: the policy corpus
# ===============================================================
class CorpusPolicySource:
    """The markdown corpus under `policies/`, with lexical retrieval.

    Retrieval is IDF over unigrams rather than embeddings, and that is a choice
    rather than a stopgap — see `core.policy`. It needs no key, so the offline
    gates can exercise every grounding rule, and it is deterministic, so a
    scenario replays.
    """

    name = "lexical_local"

    def __init__(
        self, policies_dir: Path | None = None, grounding_path: Path | None = None
    ) -> None:
        self._dir = policies_dir or (ROOT_DIR / "policies")
        self._grounding = grounding_path or (ROOT_DIR / "grounding" / "invariants.md")

    def chunks(self) -> tuple[PolicyChunk, ...]:
        return _load_corpus(self._dir)

    def grounding(self) -> str:
        if not self._grounding.exists():
            return "(grounding file missing — be maximally conservative)"
        return self._grounding.read_text(encoding="utf-8")

    def retrieve(self, query: str, k: int = 3) -> ToolResult:
        chunks = self.chunks()
        scored = sorted(
            ((score_chunk(query, c, chunks), c) for c in chunks),
            key=lambda pair: -pair[0],
        )
        hits = [
            {
                "chunk_id": c.chunk_id, "domain": c.domain,
                "source_file": c.source_file, "score": score, "text": c.text,
            }
            for score, c in scored[: max(1, k)] if score > 0
        ]
        return ToolResult(
            tool_name="retrieve_policy_context",
            value={"query": query, "hits": hits, "corpus_size": len(chunks)},
            fetched_at=now(), source=self.name,
            # A local corpus is not a degraded substitute for anything here, so
            # flagging it would train the agent to hedge entitlement claims that
            # are in fact fully grounded.
            uncertain=False, fallback_used=False,
            error=None if hits else f"no chunk in the corpus matches {query!r}",
        )


# ===============================================================
# Caching helpers — the fixtures are immutable within a run
# ===============================================================
@lru_cache(maxsize=8)
def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_corpus(policies_dir: Path) -> tuple[PolicyChunk, ...]:
    return tuple(load_chunks(policies_dir))


# ===============================================================
# Resolution
# ===============================================================
def mock_sources(*, fail_mode: str = FAIL_MODE) -> Sources:
    return Sources(
        mode="mock",
        flights=MockFlightSource(fail_mode=fail_mode),
        contacts=FixtureContactSource(),
        policy=CorpusPolicySource(),
    )


@lru_cache(maxsize=4)
def _resolve(mode: str, fail_mode: str) -> Sources:
    if mode == "mock":
        return mock_sources(fail_mode=fail_mode)
    raise NotImplementedError(
        f"DATA_MODE={mode!r} has no source set. Only 'mock' is implemented. "
        f"There is deliberately no fallback: silently serving fixtures when a "
        f"live source was asked for is how a demo gets mistaken for a deployment."
    )


def sources(mode: str | None = None, *, fail_mode: str | None = None) -> Sources:
    """The configured source set. Callers may inject their own instead."""
    return _resolve(
        (mode or DATA_MODE).lower(),
        (fail_mode if fail_mode is not None else FAIL_MODE).lower(),
    )
