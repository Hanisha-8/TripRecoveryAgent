"""Live-validate one specialist at a time, on a small free-tier model.

    python -m eval.live_specialist                    # all four, in workflow order
    python -m eval.live_specialist impact-analyst      # just one

Why this exists: the full orchestrator does not fit Groq's free tier. Measured on
`qwen/qwen3.8-27b`, a single request is capped at 7,000 input tokens (HTTP 413,
not retried) and the orchestrator carries ~4,940 of fixed overhead before any
message history; the run dies at ~7,974. The `PlanDraft` structured output needs
1,081 tokens against a 1,000-per-minute output ceiling, so it misses by 8%.

A specialist on its own fits comfortably — 1,400–2,500 input tokens and a short
return value. So this runs each one standalone, in the order the orchestrator
would delegate them, which live-validates the parts that do not depend on the
delegation loop:

  - the tools actually work against a real model, not a fake one
  - `TripContextMiddleware` reaches a specialist (deepagents only inherits
    parent middleware into *forked* subagents, so we attach it ourselves)
  - `ToolBudgetMiddleware` leaves a usable toolset behind
  - the ordering gates (`read_entitlements`, `read_impact`) fire for real
  - `CostLedger` attribution and cached-token reporting produce numbers

It does NOT cover the orchestrator's delegation, the retry loop, or the
structured-output assembly. Those need a provider that can serve them.

The specialist is rebuilt here with `create_agent` + `FilesystemMiddleware`
rather than `create_deep_agent`, because the latter auto-adds a general-purpose
subagent and with it the `task` tool (558 tokens) that a real subagent never
sees. The point is to reproduce the subagent's environment, not to approximate it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from config import LLM_MODEL, WORKSPACE_DIR
from core.clock import now
from core.plan_compiler import itinerary_version
from demo import Scenario
from eval.harness import check, report
from recovery.agent import prepare_workspace
from recovery.middleware import (
    CostCeilingMiddleware,
    CostLedger,
    ToolBudgetMiddleware,
    TripContextMiddleware,
)
from recovery.subagents import SPECIALISTS, TOOL_DENYLIST

#: What each specialist must have produced, and the task the orchestrator sends.
EXPECTATIONS: dict[str, tuple[str | None, str]] = {
    "policy-checker": (
        "analysis/entitlements.md",
        "SQ123 on 2026-09-15 was cancelled. Determine what Singapore Airlines "
        "owes the traveller and write /analysis/entitlements.md.",
    ),
    "impact-analyst": (
        "analysis/impact.md",
        "SQ123 on 2026-09-15 was cancelled; its booking_id is bk_flight_sq123. "
        "Work out what it breaks and write /analysis/impact.md.",
    ),
    "options-finder": (
        "drafts/options.md",
        "SQ123 BOM→SIN on 2026-09-15 was cancelled. Find replacement flights and "
        "draft exactly two options in /drafts/options.md.",
    ),
    "critic": (
        None,  # the critic reports in its return value; it cannot write
        "Review the entitlements and the drafted options. Report anything BLOCKING.",
    ),
}


def build_specialist(spec: dict, scenario: Scenario, ledger: CostLedger, workspace: Path):
    """Rebuild one subagent as a standalone agent, with the same environment."""
    from langchain.agents import create_agent
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware

    from recovery.llm import build_model

    backend = FilesystemBackend(root_dir=str(workspace.resolve()), virtual_mode=True)
    disruption = scenario.disruption
    return create_agent(
        model=build_model(LLM_MODEL),
        tools=list(spec.get("tools") or []),
        system_prompt=spec["system_prompt"],
        middleware=[
            TripContextMiddleware(
                trip_id=scenario.itinerary.trip_id,
                traveller_first_name=scenario.itinerary.traveller_first_name,
                party_size=scenario.itinerary.party_size,
                itinerary_version=itinerary_version(scenario.itinerary),
                disruption_line=(
                    f"{disruption.flight_iata} on {disruption.date} is "
                    f"{disruption.kind.value} (source={disruption.source})"
                ),
                today=now().isoformat(),
            ),
            CostCeilingMiddleware(ledger, spec["name"]),
            ToolBudgetMiddleware(TOOL_DENYLIST.get(spec["name"], frozenset()), spec["name"]),
            FilesystemMiddleware(backend=backend),
        ],
    )


def run_one(spec: dict, scenario: Scenario, ledger: CostLedger, workspace: Path) -> None:
    name = spec["name"]
    artifact, task = EXPECTATIONS[name]
    print(f"\n=== {name} ===")

    agent = build_specialist(spec, scenario, ledger, workspace)
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    except Exception as exc:  # noqa: BLE001
        check(f"{name} reached the provider", False,
              f"{type(exc).__name__}: {str(exc).splitlines()[0][:220]}")
        return

    final = ""
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", None) == "ai" and getattr(message, "content", None):
            final = message.content if isinstance(message.content, str) else str(message.content)
            break

    check(f"{name} ran and returned something", bool(final.strip()), "empty response")
    print(f"  → {final.strip()[:300]}")

    if artifact is not None:
        path = workspace / artifact
        check(f"{name} wrote /{artifact}", path.is_file() and path.stat().st_size > 0,
              "the file the orchestrator reads next does not exist")
        if path.is_file():
            body = path.read_text("utf-8")
            print(f"  /{artifact}: {len(body)} chars")
            if name == "policy-checker":
                from core.citations import REQUIRED_CATEGORIES, audit_citations

                audit = audit_citations(body, "cancelled", require_categories=True)
                covered = [c for c in REQUIRED_CATEGORIES if c in body.lower()]
                check("its entitlements cover all six categories",
                      len(covered) == 6, f"covered {covered}")
                check("and pass the citation audit", audit.ok, audit.report()[:400])
            if name == "impact-analyst":
                check("its impact file names the blocking bookings",
                      all(b in body for b in ("bk_transfer_sin_hotel",
                                              "bk_activity_night_safari")),
                      "the orchestrator cannot act on impacts it cannot identify")
    else:
        check(f"{name} was denied write_file and wrote nothing",
              not (workspace / "critic.md").exists())

    usage = ledger.by_agent.get(name)
    if usage:
        print(f"  calls={usage.calls} in={usage.input_tokens} "
              f"avg={usage.input_tokens // max(1, usage.calls)} "
              f"out={usage.output_tokens} cached={usage.cached_input_tokens} "
              f"({usage.cache_hit_rate:.0%})")
        check(f"{name}'s usage was attributed to it", usage.calls > 0)
        check(f"{name} stayed under Groq's 7,000-per-request input cap",
              usage.input_tokens / max(1, usage.calls) < 7000,
              f"avg {usage.input_tokens // max(1, usage.calls)} per call")


def main() -> int:
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    scenario = Scenario()
    workspace = WORKSPACE_DIR
    prepare_workspace(scenario.itinerary, workspace)

    # A stale entitlements or impact file would make the ordering gates pass for
    # the wrong reason, so the run starts from a clean workspace every time.
    for stale in ("analysis/entitlements.md", "analysis/impact.md", "drafts/options.md"):
        (workspace / stale).unlink(missing_ok=True)

    print(f"model: {LLM_MODEL}")
    ledger = CostLedger(max_cost_usd=5.0, max_llm_calls=200)

    # Workflow order, so the ordering gates are exercised rather than bypassed:
    # options-finder must find the entitlements and impact files already written.
    order = [s for s in SPECIALISTS if not wanted or s["name"] in wanted]
    for spec in order:
        run_one(spec, scenario, ledger, workspace)

    print("\n=== run totals ===")
    summary = ledger.summary()
    print(f"  calls={summary['llm_calls']} in={summary['input_tokens']} "
          f"out={summary['output_tokens']} "
          f"cached={summary['cached_input_tokens']} "
          f"({summary['cache_hit_rate']:.0%}) "
          f"${summary['estimated_cost_usd']:.4f}")
    for label, usage in summary["by_agent"].items():
        print(f"  {label:<17}{usage['calls']:>4} calls "
              f"{usage['avg_input_per_call']:>6} avg in "
              f"{usage['cache_hit_rate']:>5.0%} cached")
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
