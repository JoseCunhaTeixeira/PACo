"""Bounded retries (rule 4): a budget per gate and unit, and one for the whole run; a window's
inversion has its own, whichever gate asks. A unit whose budget is spent is rejected with
"budget spent" and the last attempt's flags."""

from collections.abc import Iterable

from paco.qc.log import INVERSION_GATES, retries_at_gate, retries_in_run, retries_of_inversion
from paco.qc.models import Attempt, Budgets, Flag, GateResult, Reject


def run_budget(budgets: Budgets, n_xmids: int) -> int:
    """Retries the whole run may spend, shared by its xmids."""
    return budgets.per_xmid_of_the_run * n_xmids


def can_retry(
    attempts: Iterable[Attempt], budgets: Budgets, n_xmids: int, unit: str, gate: str
) -> bool:
    """Whether `unit` may be retried once more at `gate`: its budgets have something left."""
    attempts = tuple(attempts)
    if gate in INVERSION_GATES:
        return retries_of_inversion(attempts, unit) < budgets.inversion_per_window
    return retries_at_gate(attempts, unit, gate) < budgets.per_gate_and_unit and retries_in_run(
        attempts
    ) < run_budget(budgets, n_xmids)


def budget_spent(result: GateResult) -> GateResult:
    """The result of a unit that asked for a retry it cannot have: rejected, with its flags."""
    flags = (
        Flag(
            name="budget_spent",
            message=f"{result.gate}: the retry budget is spent; the last attempt raised "
            f"{', '.join(flag.name for flag in result.flags) or 'no flag'}.",
            stage=result.flags[0].stage if result.flags else "picking",
            action=Reject(reason="budget spent"),
            fixable=False,
        ),
        *result.flags,
    )
    return result.model_copy(update={"verdict": "reject", "flags": flags})
