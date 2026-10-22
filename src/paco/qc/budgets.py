"""Bounded retries: a budget per gate and unit, and one for the whole run; a window's
inversion has its own, whichever gate asks. A unit refused the retry it asks (its budget spent,
nothing left to try, or a setting the user gave) stays flagged, a note first saying why, and goes
on as every flagged unit does: only a rejected one stops."""

from collections.abc import Iterable
from typing import Literal

from paco.qc.log import INVERSION_GATES, retries_at_gate, retries_in_run, retries_of_inversion
from paco.qc.models import Attempt, Budgets, Flag, GateResult, Keep


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
# attempt before (loops.unchanged); it asks only to change settings the user gave (qc.given: kept
# as given).
type Refusal = Literal["budget", "unchanged", "locked"]
_REFUSED: dict[Refusal, tuple[str, str]] = {
    "budget": ("budget_spent", "The retry budget is spent; the last attempt raised {raised}."),
    "unchanged": (
        "nothing_to_try",
        "Nothing left to try; its retry would run as the attempt before, which raised {raised}.",
    ),
    "locked": (
        "locked",
        "Its retry changes a setting the user gave, which stays as given; it asks {held} (it "
        "raised {raised}). Suggest the change to the user.",
    ),
}


# The notes saying why a unit was refused its retry.
REFUSALS = frozenset(name for name, _ in _REFUSED.values())


def refused(result: GateResult, why: Refusal = "budget", held: str = "") -> GateResult:
    """The result of a unit that asked for a retry it cannot have (`why`; `held`, the changes
    of the user's settings it asks, in words): still flagged, a note first saying why."""
    raised = ", ".join(flag.name for flag in result.flags) or "no flag"
    name, said = _REFUSED[why]
    note = Flag(
        name=name,
        message=said.format(raised=raised, held=held),
        stage=result.flags[0].stage if result.flags else "picking",
        # A locked setting's: the change it asks, for the window's line in the answer.
        action=Keep(note=f"{name}, asks {held}" if held else name.replace("_", " ")),
        fixable=False,
    )
    return result.model_copy(update={"flags": (note, *result.flags)})


def was_refused(result: GateResult) -> bool:
    """Whether `result` says already why its retry was refused: refused once, never again."""
    return any(flag.name in REFUSALS for flag in result.flags)
