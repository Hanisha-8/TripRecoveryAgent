"""Timezone-aware clock.

`FROZEN_CLOCK` freezes time so eval scenarios are reproducible — without it, a
scenario dated 2026-09-15 drifts from "today" and every latest-permissible-change
calculation changes answer between runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import FROZEN_CLOCK


class ClockError(Exception):
    """Raised when FROZEN_CLOCK is set but unparseable."""


def _frozen() -> datetime | None:
    if not FROZEN_CLOCK:
        return None
    try:
        parsed = datetime.fromisoformat(FROZEN_CLOCK)
    except ValueError as exc:
        raise ClockError(f"invalid FROZEN_CLOCK: {FROZEN_CLOCK!r}") from exc
    if parsed.tzinfo is None:
        raise ClockError(f"FROZEN_CLOCK needs a UTC offset, got {FROZEN_CLOCK!r}")
    return parsed


def now() -> datetime:
    """Current time, UTC, honouring FROZEN_CLOCK."""
    return (_frozen() or datetime.now(timezone.utc)).astimezone(timezone.utc)


def now_in_tz(tz_name: str) -> datetime:
    """Current time in an IANA zone. Falls back to UTC on an unknown zone —
    callers should mark the resulting ToolResult `uncertain` when that happens."""
    base = _frozen() or datetime.now(timezone.utc)
    try:
        return base.astimezone(ZoneInfo(tz_name))
    except ZoneInfoNotFoundError:
        return base.astimezone(timezone.utc)
