"""SQLite persistence for plans and approvals.

Plans are stored because approval binds to a hash, and a hash is only useful if
the content it covers can be produced again later. Without the stored plan, a
restart between "traveller approves" and "actions execute" leaves an
`ApprovalRecord` pointing at a version nobody can reconstruct — which fails
closed, but fails uninformatively.

Tables are created on demand with `IF NOT EXISTS`, so R3 adding a ledger and a
handoff table needs no migration.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import SQLITE_DB
from core.clock import now
from models import ApprovalRecord, Handoff, Itinerary, LedgerEntry, RecoveryPlan

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    plan_version      TEXT PRIMARY KEY,
    trip_id           TEXT NOT NULL,
    itinerary_version TEXT NOT NULL,
    payload           TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS plans_trip ON plans (trip_id, created_at DESC);

CREATE TABLE IF NOT EXISTS approvals (
    trip_id      TEXT NOT NULL,
    plan_version TEXT NOT NULL,
    option_id    TEXT NOT NULL,
    approved_by  TEXT NOT NULL,
    approved_at  TEXT NOT NULL,
    PRIMARY KEY (trip_id, plan_version)
);

-- The primary key is what makes execution idempotent. Re-running an approved
-- plan must not send a second message to the hotel, and the cheapest way to
-- guarantee that is to make a second write impossible rather than to remember
-- not to try.
CREATE TABLE IF NOT EXISTS ledger (
    trip_id      TEXT NOT NULL,
    plan_version TEXT NOT NULL,
    action_id    TEXT NOT NULL,
    kind         TEXT NOT NULL,
    action_class TEXT NOT NULL,
    status       TEXT NOT NULL,
    performed_by TEXT NOT NULL,
    detail       TEXT NOT NULL,
    at           TEXT NOT NULL,
    PRIMARY KEY (trip_id, plan_version, action_id)
);

CREATE TABLE IF NOT EXISTS handoffs (
    trip_id           TEXT NOT NULL,
    plan_version      TEXT NOT NULL,
    action_id         TEXT NOT NULL,
    kind              TEXT NOT NULL,
    target_booking_id TEXT NOT NULL,
    description       TEXT NOT NULL,
    status            TEXT NOT NULL,
    confirmed_at      TEXT,
    PRIMARY KEY (trip_id, plan_version, action_id)
);

-- The itinerary as it stands after everything executed so far. This is what the
-- close-out check validates, and it is stored rather than recomputed because a
-- traveller may confirm one handoff today and another tomorrow.
CREATE TABLE IF NOT EXISTS working_itineraries (
    trip_id    TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection with the schema applied."""
    path = Path(db_path or SQLITE_DB)
    if path.parent != Path("") and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------
# Plans
# ---------------------------------------------------------------
def save_plan(plan: RecoveryPlan, *, db_path: Path | str | None = None) -> None:
    """Persist a compiled plan, keyed by its version.

    `INSERT OR REPLACE` is safe precisely because the key is a content hash: a
    second write under the same key is by definition the same plan.
    """
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO plans "
            "(plan_version, trip_id, itinerary_version, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                plan.plan_version, plan.trip_id, plan.itinerary_version,
                plan.model_dump_json(), plan.generated_at.isoformat(),
            ),
        )


def load_plan(
    plan_version: str, *, db_path: Path | str | None = None
) -> RecoveryPlan | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload FROM plans WHERE plan_version = ?", (plan_version,)
        ).fetchone()
    return RecoveryPlan.model_validate(json.loads(row["payload"])) if row else None


def latest_plan(
    trip_id: str, *, db_path: Path | str | None = None
) -> RecoveryPlan | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload FROM plans WHERE trip_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (trip_id,),
        ).fetchone()
    return RecoveryPlan.model_validate(json.loads(row["payload"])) if row else None


# ---------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------
def save_approval(
    approval: ApprovalRecord, *, db_path: Path | str | None = None
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO approvals "
            "(trip_id, plan_version, option_id, approved_by, approved_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                approval.trip_id, approval.plan_version, approval.option_id,
                approval.approved_by, approval.approved_at.isoformat(),
            ),
        )


def load_approval(
    trip_id: str, plan_version: str, *, db_path: Path | str | None = None
) -> ApprovalRecord | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM approvals WHERE trip_id = ? AND plan_version = ?",
            (trip_id, plan_version),
        ).fetchone()
    if row is None:
        return None
    return ApprovalRecord(
        trip_id=row["trip_id"],
        option_id=row["option_id"],
        plan_version=row["plan_version"],
        approved_at=row["approved_at"],
        approved_by=row["approved_by"],
    )


# ---------------------------------------------------------------
# Ledger — append-only, and the primary key is the idempotency guard
# ---------------------------------------------------------------
def record_action(
    trip_id: str, plan_version: str, entry: LedgerEntry,
    *, db_path: Path | str | None = None,
) -> bool:
    """Append a ledger entry. Returns False if this action was already recorded.

    The caller checks the return value BEFORE performing a side effect, which is
    the wrong way round for a normal write and the right way round here: an
    `INSERT OR IGNORE` that reports "already there" is how we know not to send a
    second message.
    """
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO ledger (trip_id, plan_version, action_id, kind, "
            "action_class, status, performed_by, detail, at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trip_id, plan_version, entry.action_id, entry.kind.value,
                entry.action_class.value, entry.status, entry.performed_by,
                entry.detail, entry.at.isoformat(),
            ),
        )
        return cursor.rowcount > 0


def already_done(
    trip_id: str, plan_version: str, action_id: str,
    *, db_path: Path | str | None = None,
) -> bool:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM ledger WHERE trip_id = ? AND plan_version = ? "
            "AND action_id = ?",
            (trip_id, plan_version, action_id),
        ).fetchone()
    return row is not None


def load_ledger(
    trip_id: str, plan_version: str, *, db_path: Path | str | None = None
) -> list[LedgerEntry]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM ledger WHERE trip_id = ? AND plan_version = ? ORDER BY at",
            (trip_id, plan_version),
        ).fetchall()
    return [
        LedgerEntry(
            action_id=r["action_id"], kind=r["kind"], action_class=r["action_class"],
            status=r["status"], performed_by=r["performed_by"], detail=r["detail"],
            at=r["at"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------
# Handoffs
# ---------------------------------------------------------------
def upsert_handoff(
    trip_id: str, plan_version: str, handoff: Handoff,
    *, db_path: Path | str | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO handoffs (trip_id, plan_version, action_id, kind, "
            "target_booking_id, description, status, confirmed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trip_id, plan_version, handoff.action_id, handoff.kind.value,
                handoff.target_booking_id, handoff.description, handoff.status,
                handoff.confirmed_at.isoformat() if handoff.confirmed_at else None,
            ),
        )


def create_handoff_if_absent(
    trip_id: str, plan_version: str, handoff: Handoff,
    *, db_path: Path | str | None = None,
) -> bool:
    """Create a handoff, leaving an existing one untouched.

    Re-executing must not reset a handoff the traveller has already confirmed
    back to pending, which is exactly what an unconditional upsert would do.
    """
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO handoffs (trip_id, plan_version, action_id, kind, "
            "target_booking_id, description, status, confirmed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trip_id, plan_version, handoff.action_id, handoff.kind.value,
                handoff.target_booking_id, handoff.description, handoff.status, None,
            ),
        )
        return cursor.rowcount > 0


def load_handoffs(
    trip_id: str, plan_version: str, *, db_path: Path | str | None = None
) -> list[Handoff]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM handoffs WHERE trip_id = ? AND plan_version = ? "
            "ORDER BY action_id",
            (trip_id, plan_version),
        ).fetchall()
    return [
        Handoff(
            action_id=r["action_id"], kind=r["kind"],
            target_booking_id=r["target_booking_id"], description=r["description"],
            status=r["status"], confirmed_at=r["confirmed_at"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------
# Working itinerary
# ---------------------------------------------------------------
def save_working_itinerary(
    itinerary: Itinerary, *, db_path: Path | str | None = None
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO working_itineraries (trip_id, payload, updated_at) "
            "VALUES (?, ?, ?)",
            (itinerary.trip_id, itinerary.model_dump_json(), now().isoformat()),
        )


def load_working_itinerary(
    trip_id: str, *, db_path: Path | str | None = None
) -> Itinerary | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload FROM working_itineraries WHERE trip_id = ?", (trip_id,)
        ).fetchone()
    return Itinerary.model_validate(json.loads(row["payload"])) if row else None


# ---------------------------------------------------------------
# Reset — for the demo, and for starting a trip over
# ---------------------------------------------------------------
def clear_trip(trip_id: str, *, db_path: Path | str | None = None) -> None:
    """Forget everything about one trip. Plans included.

    Leaving the plans behind would look tidier and be wrong: `plan_version` is a
    content hash, so a re-run produces the identical version, and a stale approval
    row would then authorise it. Clearing the approval but keeping the plan is the
    subtle version of the same bug.
    """
    with connect(db_path) as conn:
        conn.execute("DELETE FROM plans WHERE trip_id = ?", (trip_id,))
        conn.execute("DELETE FROM approvals WHERE trip_id = ?", (trip_id,))
        conn.execute("DELETE FROM ledger WHERE trip_id = ?", (trip_id,))
        conn.execute("DELETE FROM handoffs WHERE trip_id = ?", (trip_id,))
        conn.execute("DELETE FROM working_itineraries WHERE trip_id = ?", (trip_id,))


def current_approval(
    trip_id: str, *, db_path: Path | str | None = None
) -> ApprovalRecord | None:
    """The most recent approval for a trip, whatever plan version it names."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM approvals WHERE trip_id = ? ORDER BY approved_at DESC LIMIT 1",
            (trip_id,),
        ).fetchone()
    if row is None:
        return None
    return ApprovalRecord(
        trip_id=row["trip_id"], option_id=row["option_id"],
        plan_version=row["plan_version"], approved_at=row["approved_at"],
        approved_by=row["approved_by"],
    )
