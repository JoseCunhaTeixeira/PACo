"""What the stage tools' retry loops share (option B of docs/qc_workflow.md): a gate's changes for
the stage being run, merged into the parameters of the attempt before, and whether a unit may
have one more try. Changes a gate asks of an earlier stage are left to the agent: going back is
its call."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, cast

from paco.qc.budgets import run_budget
from paco.qc.log import INVERSION_GATES, retries_at_gate, retries_in_run, retries_of_inversion
from paco.qc.models import Attempt, Budgets, GateResult, Override, Stage


def deep_merge(base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """`changes` over `base`, stage by stage: a nested mapping is merged, anything else replaced
    (a stage given with another method is then replaced whole by the preset's rules)."""
    merged = dict(base)
    for key, value in changes.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(
                cast(Mapping[str, Any], previous), cast(Mapping[str, Any], value)
            )
        else:
            merged[key] = value
    return merged


def stage_changes(result: GateResult, stage: Stage) -> tuple[dict[str, Any], str] | None:
    """The changes `result`'s flags ask of `stage`, merged, and the first flag asking; None when
    none do."""
    changes: dict[str, Any] = {}
    first: str | None = None
    for flag in result.flags:
        if isinstance(flag.action, Override) and flag.action.stage == stage:
            changes = deep_merge(changes, flag.action.overrides)
            first = first or flag.name
    return (changes, first) if first is not None else None


class RetryBudget:
    """What is left of the budgets (rule 4) for one batch of retries: each retry granted counts
    at once, so that a batch of windows cannot go beyond the run's budget together. A window's
    inversion has its own budget, whichever gate asks."""

    def __init__(self, attempts: Iterable[Attempt], budgets: Budgets, n_units: int) -> None:
        self._attempts = tuple(attempts)
        self._budgets = budgets
        self.left = run_budget(budgets, n_units) - retries_in_run(self._attempts)
        self._inversions: Counter[str] = Counter()

    def grant(self, unit: str, gate: str) -> bool:
        """One more retry of `unit` at `gate`, if its budgets allow it."""
        if gate in INVERSION_GATES:
            spent = retries_of_inversion(self._attempts, unit) + self._inversions[unit]
            if spent >= self._budgets.inversion_per_window:
                return False
            self._inversions[unit] += 1
            return True
        if self.left <= 0:
            return False
        if retries_at_gate(self._attempts, unit, gate) >= self._budgets.per_gate_and_unit:
            return False
        self.left -= 1
        return True


def next_try(
    result: GateResult, stage: Stage, budget: RetryBudget, previous: Mapping[str, Any]
) -> tuple[dict[str, Any], str] | None:
    """The parameters of `result`'s unit's next attempt at `stage` (`previous`, the latest
    attempt's, with the gate's changes over them) and what triggers it ("<gate>:<flag>"); None
    when the gate asks nothing of this stage, or `budget` has no retry left for it."""
    wanted = stage_changes(result, stage)
    if wanted is None or result.verdict != "retry" or not budget.grant(result.unit, result.gate):
        return None
    changes, flag = wanted
    return deep_merge(previous, changes), f"{result.gate}:{flag}"


def spent(result: GateResult, stage: Stage) -> bool:
    """Whether `result` asks for a change of `stage`: once `next_try` gave it none, the budget
    is spent."""
    return result.verdict == "retry" and stage_changes(result, stage) is not None
