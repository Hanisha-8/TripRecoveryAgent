"""The deep agent — investigation only.

The orchestrator decides what to investigate next and delegates to scoped
specialists. It has no authority to execute anything: its structured output is a
`PlanDraft`, which `core.plan_compiler` either accepts or rejects.

R1 ships two specialists (impact-analyst, options-finder) plus a guard that
disables the built-in `general-purpose` subagent. policy-checker and critic land
in R2 — the deterministic pieces they defend are worth building first, so their
findings have something to be checked against.
"""
