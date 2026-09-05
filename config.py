"""TripRecovery configuration.

Loads `.env` and exposes typed constants. Same shape as tripsure's `config.py`
so the two projects stay readable side by side.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
WORKSPACE_DIR = ROOT_DIR / os.getenv("WORKSPACE_DIR", "workspace")

# ---------------------------------------------------------------
# LLM
# ---------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or None
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY") or None
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or None

# ---------------------------------------------------------------
# Spend ceilings — enforced by CostCeilingMiddleware, not by hope.
# A four-subagent orchestrator with its own tool loops can spend a lot before
# anyone notices, so the ceiling is a hard RuntimeError rather than a warning.
# ---------------------------------------------------------------
MAX_COST_USD = float(os.getenv("MAX_COST_USD", "1.00"))
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "40"))
COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_INPUT", "0.002"))
COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_OUTPUT", "0.008"))
#: Per-call output cap. A `PlanDraft` with two options and eight actions runs
#: ~2.5k tokens, so 4k leaves headroom without leaving the ceiling open.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------
SQLITE_DB = Path(os.getenv("SQLITE_DB", "./triprecovery.db"))

# ---------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------
#: Where flight status, availability, airline contacts and policy come from.
#: `mock` means the fixtures under `data/` and `policies/`. Every provider goes
#: through the same interface, so flipping this is a wiring change rather than a
#: rewrite — see `core/sources.py`.
#:
#: Declared in one place on purpose. Scattering "this is fixture data" notices
#: through the UI put a developer's warning string in front of travellers; one
#: honest mode marker is both truer and quieter.
DATA_MODE = os.getenv("DATA_MODE", "mock").lower()
IS_MOCK_DATA = DATA_MODE == "mock"

# ---------------------------------------------------------------
# Demo controls
# ---------------------------------------------------------------
FROZEN_CLOCK = os.getenv("FROZEN_CLOCK") or None
FAIL_MODE = os.getenv("FAIL_MODE", "none").lower()

# ---------------------------------------------------------------
# Deterministic thresholds — the impact graph's buffer policy
# ---------------------------------------------------------------
AT_RISK_MINUTES = 30
HOTEL_CHECK_IN_BUFFER_MINUTES = 60
TRANSFER_PICKUP_BUFFER_MINUTES = 30


def summary() -> dict[str, object]:
    """Config summary with secrets masked. Rendered by the UI in R5."""

    def _mask(v: str | None) -> str:
        return "SET" if v else "MISSING"

    return {
        "llm_model": LLM_MODEL,
        "openai_key": _mask(OPENAI_API_KEY),
        "groq_key": _mask(GROQ_API_KEY),
        "max_cost_usd": MAX_COST_USD,
        "max_llm_calls": MAX_LLM_CALLS,
        "frozen_clock": FROZEN_CLOCK,
        "fail_mode": FAIL_MODE,
        "data_mode": DATA_MODE,
        "workspace_dir": str(WORKSPACE_DIR),
        "sqlite_db": str(SQLITE_DB),
    }
