"""Approval binding — the gate between investigating and acting.

The R1 plan said this phase would use deepagents' `interrupt_on`. It does not,
and the reason is worth stating: `interrupt_on` pauses a *tool call the model
makes*. Here the executor is not a tool. The model has no way to reach it, so
there is no call to interrupt. Guarding a door the model cannot walk through
would look like a safety control while protecting nothing.

What replaces it is a capability token. `execute()` in R3 takes an
`Authorization` and nothing else, and only `authorize()` can mint one — after
checking that the plan is untampered, the approval matches this exact plan
version, and the itinerary has not moved underneath it. There is no prompt to
bypass and no config key to typo: without a token there is no code path that
acts.

Approval is never inferred from conversation. It arrives as an `ApprovalRecord`
naming one option and one `plan_version`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.clock import now
from core.plan_compiler import agent_safe_actions, human_actions, itinerary_version
from models import Action, ApprovalRecord, Itinerary, PlanOption, RecoveryPlan

#: Minting key for `Authorization`. Module-private, so an `Authorization` cannot
#: be constructed anywhere but `authorize()` below — including by a tool handing
#: back JSON, which is the case that matters.
_MINT = object()


class RefusalReason(str, Enum):
    PLAN_TAMPERED = "plan_tampered"
    APPROVAL_STALE = "approval_stale"
    UNKNOWN_OPTION = "unknown_option"
    ITINERARY_CHANGED = "itinerary_changed"
    TRIP_MISMATCH = "trip_mismatch"


@dataclass(frozen=True)
class Refusal:
    """Why nothing will be executed. Rendered to the user verbatim."""

    reason: RefusalReason
    detail: str

    @property
    def ok(self) -> bool:
        return False


@dataclass(frozen=True)
class Authorization:
    """Proof that a specific plan version was approved by a specific person.

    Holding one of these is the only way to execute anything. It carries the
    action split so the executor never re-derives it: an executor that decided
    for itself which actions were agent-safe would be a second implementation of
    the capability rules, free to disagree with the one that was approved.
    """

    _mint: Any
    plan: RecoveryPlan
    option: PlanOption
    approval: ApprovalRecord
    #: The exact itinerary whose version was verified. Carried rather than passed
    #: separately to the executor, so there is no way to authorise against one
    #: trip and then execute against another.
    itinerary: Itinerary
    agent_actions: list[Action] = field(default_factory=list)
    human_actions: list[Action] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise PermissionError(
                "Authorization cannot be constructed directly — call "
                "core.approval.authorize(), which verifies the plan hash first"
            )

    @property
    def ok(self) -> bool:
        return True

    @property
    def plan_version(self) -> str:
        return self.plan.plan_version


def approve(
    plan: RecoveryPlan, option_id: str, *, approved_by: str
) -> ApprovalRecord | Refusal:
    """Record a human's choice of one option on one plan version.

    Refuses an unknown option here rather than at execution time, so a bad
    selection surfaces while the person who made it is still looking at it.
    """
    if plan.option(option_id) is None:
        return Refusal(
            RefusalReason.UNKNOWN_OPTION,
            f"option {option_id!r} is not in plan {plan.plan_version} "
            f"(has {[o.option_id for o in plan.options]})",
        )
    return ApprovalRecord(
        trip_id=plan.trip_id,
        option_id=option_id,  # type: ignore[arg-type]
        plan_version=plan.plan_version,
        approved_at=now(),
        approved_by=approved_by,
    )


def authorize(
    plan: RecoveryPlan, approval: ApprovalRecord, itinerary: Itinerary
) -> Authorization | Refusal:
    """Mint an `Authorization`, or refuse and say why.

    Four independent checks, because they fail for four different reasons and a
    single "invalid" would tell the traveller nothing about what to do next.
    """
    if approval.trip_id != plan.trip_id:
        return Refusal(
            RefusalReason.TRIP_MISMATCH,
            f"approval is for trip {approval.trip_id!r}, plan is for {plan.trip_id!r}",
        )

    # The plan's own content no longer hashes to the version it claims. Something
    # edited it after compilation — most usefully caught when the plan came back
    # from storage rather than straight from the compiler.
    recomputed = plan.recompute_plan_version()
    if recomputed != plan.plan_version:
        return Refusal(
            RefusalReason.PLAN_TAMPERED,
            f"plan content hashes to {recomputed} but claims {plan.plan_version} — "
            f"it was modified after compilation and cannot be executed",
        )

    # The approval names a different version. Benign cause: the plan was
    # regenerated after the traveller approved. Either way the person approved
    # something other than what is in front of us, so they have to look again.
    if approval.plan_version != plan.plan_version:
        return Refusal(
            RefusalReason.APPROVAL_STALE,
            f"approval is bound to {approval.plan_version} but this plan is "
            f"{plan.plan_version} — the plan changed after approval, so it must be "
            f"re-approved",
        )

    # The trip itself moved. The plan's actions were computed against a schedule
    # that no longer exists, so acting on them could break something new.
    current_iv = itinerary_version(itinerary)
    if current_iv != plan.itinerary_version:
        return Refusal(
            RefusalReason.ITINERARY_CHANGED,
            f"itinerary is now {current_iv} but the plan was built against "
            f"{plan.itinerary_version} — re-run recovery against the current trip",
        )

    option = plan.option(approval.option_id)
    if option is None:
        return Refusal(
            RefusalReason.UNKNOWN_OPTION,
            f"approved option {approval.option_id!r} is not in the plan",
        )

    return Authorization(
        _mint=_MINT,
        plan=plan,
        option=option,
        approval=approval,
        itinerary=itinerary,
        agent_actions=agent_safe_actions(option),
        human_actions=human_actions(option),
    )
