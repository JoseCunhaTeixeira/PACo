"""What an evaluation records: each check, the judge's score, and the report of a whole suite."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type Kind = Literal[
    "look around",
    "process and judge",
    "recover",
    "the loop",
    "stuck",
    "hand work",
    "positions",
    "work there",
    "wording",
]


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str  # what was checked, e.g. "called inspect(what=profile, profile=active_p1)"
    passed: bool
    detail: str = ""  # why it failed


class JudgeScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: int | None = Field(ge=1, le=5)  # None: the judge's answer could not be read
    reason: str


class ScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Kind
    attempt: int = 1  # which play of the scenario, when the evaluation repeats them
    checks: tuple[CheckResult, ...]
    judge: JudgeScore | None  # None: no judge model configured
    tool_calls: int
    failed_calls: int  # errors and refused calls, which the model had to read and handle
    max_prompt_tokens: int | None  # the largest context the model read, when the API says
    duration_s: float
    answer: str  # the model's last answer to the user
    # The calls the host refused as outside the message's scope (L4), the answers a cap or a
    # repeat ended (L1, L3), and the retries the gates asked in the play's runs, by gate (O3).
    scope_blocks: int = 0
    caps_reached: int = 0
    retries: dict[str, int] = {}
    # Why the play was cut short by the model's server (unreachable, failing): not the model's
    # doing, so out of the pass rates; its transcript kept as far as it went.
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(check.passed for check in self.checks)


class EvaluationReport(BaseModel):
    """One run of the suite: written as report.json, with one transcript per play of each
    scenario."""

    model_config = ConfigDict(frozen=True)

    eval_id: str
    model: str
    judge_model: str | None
    # The prompts' version the plays read (paco.prompts): pass rates are per model and prompts.
    prompt_version: str | None = None
    started_at: datetime
    # Plays of each scenario: the model samples, so a single play is a noisy measure.
    repeat: int = 1
    results: tuple[ScenarioResult, ...]  # every play, scenario by scenario
    # Each scenario's pass-rate threshold when it was played (E4).
    thresholds: dict[str, float] = {}

    def pass_rates(self) -> dict[str, tuple[int, int]]:
        """Each scenario's plays passed and played, in the order played; a play its model's
        server cut short not counted."""
        rates: dict[str, tuple[int, int]] = {}
        for result in self.results:
            if result.error is not None:
                continue
            passed, played = rates.get(result.name, (0, 0))
            rates[result.name] = (passed + result.passed, played + 1)
        return rates

    def meets(self, scenario: str) -> bool | None:
        """Whether `scenario`'s pass rate meets its threshold; None without one recorded."""
        threshold = self.thresholds.get(scenario)
        passed, played = self.pass_rates().get(scenario, (0, 0))
        return None if threshold is None or not played else passed / played >= threshold
