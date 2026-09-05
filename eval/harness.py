"""Shared check harness for the phase smoke gates.

The plan fixtures live in `demo.py`, not here, and are re-exported below. That
direction is deliberate: the UI's demo scenario and the gates' fixture are the
same object, so the demo cannot drift into showing a plan the compiler would
reject, and a fixture change cannot pass the gates while breaking the demo.

Each fixture builds a plan that SHOULD compile, so every test mutates exactly one
thing and isolates one rule.
"""

from __future__ import annotations

from demo import (  # noqa: F401 — re-exported for the smoke gates
    HOTEL_MESSAGE,
    ITINERARY_PATH,
    Scenario,
    actions,
    actions_a,
    actions_b,
    flight_a,
    flight_b,
    good_draft,
)

_passed = 0
_failed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed.append(f"{name} — {detail}" if detail else name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def report() -> int:
    print(f"\n{'=' * 60}\n{_passed} passed, {len(_failed)} failed")
    for failure in _failed:
        print(f"  - {failure}")
    return 1 if _failed else 0
