"""Bounded retries (rule 4): a budget per gate and unit, and one for the whole run; a window's
inversion has its own, whichever gate asks. A unit whose budget is spent is rejected with
"budget spent" and the last attempt's flags."""

from collections.abc import Iterable
from typing import Literal

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


# Why a unit is refused the retry it asks: its budget is spent; the retry would run as the
# attempt before (loops.unchanged); the earlier stage it blames was done again once already, with
# the change it asks (redo.settle_earlier).
type Refusal = Literal["budget", "unchanged", "redone"]
_REFUSED: dict[Refusal, tuple[str, str]] = {
    "budget": ("budget_spent", "the retry budget is spent; the last attempt raised {raised}."),
    "unchanged": (
        "nothing_to_try",
        "its retry would run as the attempt before, which raised {raised}: nothing left to try.",
    ),
    "redone": (
        "redone_once",
        "the stage it blames was done again once with the change it asks, and it still raises "
        "{raised}.",
    ),
}


def budget_spent(result: GateResult, why: Refusal = "budget") -> GateResult:
    """The result of a unit that asked for a retry it cannot have (`why`): rejected, with its
    flags."""
    raised = ", ".join(flag.name for flag in result.flags) or "no flag"
    name, said = _REFUSED[why]
    flags = (
        Flag(
            name=name,
            message=f"{result.gate}: {said.format(raised=raised)}",
            stage=result.flags[0].stage if result.flags else "picking",
            action=Reject(reason=name.replace("_", " ")),
            fixable=False,
        ),
        *result.flags,
    )
    return result.model_copy(update={"verdict": "reject", "flags": flags})
