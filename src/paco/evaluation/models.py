"""What an evaluation records: each check, the judge's score, and the report of a whole suite."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type Kind = Literal["look around", "process and judge", "recover", "the loop", "stuck"]


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str  # what was checked, e.g. "called inspect_profile(profile=active_p1)"
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

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class EvaluationReport(BaseModel):
    """One run of the suite: written as report.json, with one transcript per play of each
    scenario."""

    model_config = ConfigDict(frozen=True)

    eval_id: str
    model: str
    judge_model: str | None
    started_at: datetime
    # Plays of each scenario: the model samples, so a single play is a noisy measure.
    repeat: int = 1
    results: tuple[ScenarioResult, ...]  # every play, scenario by scenario
