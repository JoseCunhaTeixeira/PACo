"""Budgets, and what is spent of them (docs/autonomous_plan.md §9). Each effort's budgets are the plan's
starting values. What is spent is counted from the study's records: the experiments and the
commits from the journal; the model's calls and the tools' time from the host's usage log
(usage.jsonl, a line a call); CPU time in P2 in multiples of the line's first processing, its
first commit's time. The host refuses a call that would go past a budget, saying what to do
instead, and stops the run once its model calls are spent; at 80 % of a budget the state card
says so."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from paco.autonomous.tools import journal
from paco.autonomous.tools.trials import Study

USAGE = "usage.jsonl"  # in the study's folder
WARN = 0.8  # the share of a budget past which the state card says so
type Effort = Literal["quick", "normal", "thorough"]


class Budgets(BaseModel):
    """An effort's budgets: experiments and line commits in P2, CPU time in P2 in multiples of
    the line's first processing, each window's inversion retries (M7), the model's calls over
    the run."""

    model_config = ConfigDict(frozen=True)

    experiments: int
    commits: int
    cpu: float
    inversion_retries: int
    model_calls: int


EFFORTS: dict[Effort, Budgets] = {
    "quick": Budgets(experiments=8, commits=2, cpu=6, inversion_retries=2, model_calls=60),
    "normal": Budgets(experiments=20, commits=4, cpu=20, inversion_retries=6, model_calls=150),
    "thorough": Budgets(experiments=40, commits=6, cpu=40, inversion_retries=6, model_calls=300),
}


class Use(BaseModel):
    """A call the host made: the model's (in the run's loop, or filling a form) or a tool's."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["model", "form", "tool"]
    phase: str
    at: datetime
    seconds: float
    name: str | None = None  # the tool's
    refused: bool = False  # the host did not call it
    failed: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def record(study: Study, use: Use) -> None:
    """`use` appended to the study's usage log."""
    with (study.folder / USAGE).open("a") as file:
        file.write(use.model_dump_json() + "\n")


def uses(study: Study) -> list[Use]:
    """The study's usage log, in its order."""
    path = study.folder / USAGE
    if not path.exists():
        return []
    return [Use.model_validate_json(line) for line in path.read_text().splitlines() if line]


@dataclass(frozen=True)
class Spent:
    """What is spent of `budgets`: experiments, commits, CPU time in P2 (None before the first
    commit, its unit), and model calls."""

    budgets: Budgets
    experiments: int
    commits: int
    cpu: float | None
    model_calls: int

    def said(self) -> str:
        """Each budget and what is spent of it, in a line."""
        budgets = self.budgets
        cpu = f"{self.cpu:.1f}" if self.cpu is not None else "0"
        return (
            f"{self.experiments} of {budgets.experiments} experiments, {self.commits} of "
            f"{budgets.commits} commits, CPU {cpu} of {budgets.cpu:g} times the first "
            f"processing, {self.model_calls} of {budgets.model_calls} model calls"
        )

    def near(self) -> list[str]:
        """The budgets at WARN or past, in words."""
        said: list[str] = []
        for name, used, budget in self._counts():
            if used >= budget:
                said.append(f"{name} spent ({used:g} of {budget:g})")
            elif used >= WARN * budget:
                said.append(f"{name} at {used / budget:.0%} ({used:g} of {budget:g})")
        return said

    @property
    def calls_spent(self) -> bool:
        return self.model_calls >= self.budgets.model_calls

    def _counts(self) -> list[tuple[str, float, float]]:
        budgets = self.budgets
        counts = [
            ("experiments", float(self.experiments), float(budgets.experiments)),
            ("commits", float(self.commits), float(budgets.commits)),
            ("model calls", float(self.model_calls), float(budgets.model_calls)),
        ]
        if self.cpu is not None:
            counts.insert(2, ("CPU", round(self.cpu, 1), budgets.cpu))
        return counts


def spent(study: Study, effort: Effort) -> Spent:
    """What the study spent of `effort`'s budgets, from its journal and its usage log."""
    found = journal.entries(study)
    used = uses(study)
    return Spent(
        budgets=EFFORTS[effort],
        experiments=sum(1 for entry in found if entry.kind == "experiment"),
        commits=sum(1 for entry in found if entry.kind == "commit"),
        cpu=_cpu(used),
        model_calls=sum(1 for one in used if one.kind == "model"),
    )


def refusal(spent: Spent, tool: str) -> str | None:
    """Why the host does not call `tool` now: a budget it would go past, and what to do instead;
    none within the budgets."""
    budgets = spent.budgets
    if tool == "try_settings" and spent.experiments >= budgets.experiments:
        return (
            f"Not called: the experiments' budget is spent ({spent.experiments} of "
            f"{budgets.experiments}). Commit your best settings and move on."
        )
    if tool == "try_settings" and spent.cpu is not None and spent.cpu >= budgets.cpu:
        return (
            f"Not called: the CPU budget of the curves is spent ({spent.cpu:.1f} of "
            f"{budgets.cpu:g} times the line's first processing). Commit your best settings "
            "and move on."
        )
    if tool == "commit" and spent.commits >= budgets.commits:
        return (
            f"Not called: the line commits' budget is spent ({spent.commits} of "
            f"{budgets.commits}). Pick and decide the windows of the line as it is."
        )
    return None


def _cpu(used: Sequence[Use]) -> float | None:
    """The time of the tools called in P2, in multiples of the first commit's; None before it."""
    first = next(
        (
            one.seconds
            for one in used
            if one.kind == "tool" and one.name == "commit" and not one.refused and not one.failed
        ),
        None,
    )
    if not first:
        return None
    total = sum(
        one.seconds for one in used if one.kind == "tool" and one.phase == "P2" and not one.refused
    )
    return total / first
