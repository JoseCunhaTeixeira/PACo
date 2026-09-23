"""Rule checks on a conversation: what the agent called, what it answered, what it left on disk."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paco.agent.record import ToolStep, Transcript
from paco.evaluation.models import CheckResult
from paco.inversion import InversionRecord
from paco.quality import RunQuality


@dataclass(frozen=True)
class Trial:
    """One scenario, played: the conversation, and what the server wrote."""

    transcript: Transcript
    output_dir: Path  # the server's output directory during the scenario
    questions_to_user: tuple[str, ...]  # approval questions the simulated user was asked


type Check = Callable[[Trial], CheckResult]
# A fact the answer must mention: fixed, or read from what the server wrote.
type Fact = str | Callable[[Trial], str]


def called(tool: str, **arguments: Any) -> Check:  # noqa: ANN401
    """The agent called `tool`, with at least these arguments (nested objects may hold more)."""
    shown = ", ".join(f"{name}={json.dumps(value)}" for name, value in arguments.items())
    name = f"called {tool}({shown})"

    def check(trial: Trial) -> CheckResult:
        steps = [step for step in _called(trial) if step.name == tool]
        if not steps:
            return CheckResult(name=name, passed=False, detail=f"{tool} was never called")
        if any(_contains(json.loads(step.arguments or "{}"), arguments) for step in steps):
            return CheckResult(name=name, passed=True)
        seen = "; ".join(step.arguments for step in steps)
        return CheckResult(name=name, passed=False, detail=f"called with {seen}")

    return check


def only_called(tool: str, **arguments: Any) -> Check:  # noqa: ANN401
    """`tool` succeeded, and every call of it that succeeded had at least these arguments: the
    agent kept what the user asked for, even when a tool advised otherwise."""
    shown = ", ".join(f"{name}={json.dumps(value)}" for name, value in arguments.items())
    name = f"only {tool}({shown})"

    def check(trial: Trial) -> CheckResult:
        steps = [step for step in _called(trial) if step.name == tool and not step.is_error]
        if not steps:
            return CheckResult(name=name, passed=False, detail=f"{tool} never succeeded")
        others = [
            step.arguments
            for step in steps
            if not _contains(json.loads(step.arguments or "{}"), arguments)
        ]
        detail = f"called with {'; '.join(others)}" if others else ""
        return CheckResult(name=name, passed=not others, detail=detail)

    return check


def succeeded(tool: str) -> Check:
    """At least one call of `tool` succeeded."""

    def check(trial: Trial) -> CheckResult:
        steps = [step for step in _called(trial) if step.name == tool]
        passed = any(not step.is_error for step in steps)
        detail = "" if passed else f"{len(steps)} call(s), none succeeded"
        return CheckResult(name=f"{tool} succeeded", passed=passed, detail=detail)

    return check


def not_succeeded(tool: str) -> Check:
    """No call of `tool` succeeded (it may have been tried, and refused by PACo)."""

    def check(trial: Trial) -> CheckResult:
        passed = not any(step.name == tool and not step.is_error for step in _called(trial))
        detail = "" if passed else f"{tool} succeeded"
        return CheckResult(name=f"{tool} did not succeed", passed=passed, detail=detail)

    return check


def never_called(tool: str) -> Check:
    """`tool` was never called: the user did not ask for what it does."""

    def check(trial: Trial) -> CheckResult:
        calls = sum(step.name == tool for step in _called(trial))
        detail = f"called {calls} time(s)" if calls else ""
        return CheckResult(name=f"{tool} never called", passed=not calls, detail=detail)

    return check


def in_order(*tools: str) -> Check:
    """The first successful call of each tool comes in this order."""
    name = "in order: " + ", ".join(tools)

    def check(trial: Trial) -> CheckResult:
        firsts: list[int] = []
        for tool in tools:
            index = next(
                (
                    position
                    for position, step in enumerate(_called(trial))
                    if step.name == tool and not step.is_error
                ),
                None,
            )
            if index is None:
                return CheckResult(name=name, passed=False, detail=f"{tool} never succeeded")
            firsts.append(index)
        passed = firsts == sorted(firsts)
        return CheckResult(name=name, passed=passed, detail="" if passed else f"order {firsts}")

    return check


def at_most_calls(limit: int) -> Check:
    def check(trial: Trial) -> CheckResult:
        count = len(trial.transcript.tool_steps)
        passed = count <= limit
        detail = "" if passed else f"{count} calls"
        return CheckResult(name=f"at most {limit} tool calls", passed=passed, detail=detail)

    return check


def answer_mentions(*facts: Fact) -> Check:
    """The final answer mentions each fact (case aside; a number as a whole number)."""

    def check(trial: Trial) -> CheckResult:
        answer = trial.transcript.answer
        values = [fact(trial) if callable(fact) else fact for fact in facts]
        missing = [value for value in values if not _mentions(answer, value)]
        name = "answer mentions " + ", ".join(values)
        detail = "" if not missing else "missing " + ", ".join(missing)
        return CheckResult(name=name, passed=not missing, detail=detail)

    return check


def asked_the_user() -> Check:
    def check(trial: Trial) -> CheckResult:
        passed = bool(trial.questions_to_user)
        detail = "" if passed else "no approval question reached the user"
        return CheckResult(name="the user was asked to approve", passed=passed, detail=detail)

    return check


def no_inversion_started() -> Check:
    def check(trial: Trial) -> CheckResult:
        records = list(trial.output_dir.glob("*/*/inversion.json"))
        detail = "" if not records else f"{len(records)} inversion(s) on disk"
        return CheckResult(name="no inversion started", passed=not records, detail=detail)

    return check


def inversion_succeeded() -> Check:
    """The inversion the agent started ran to its end, with a model for every window.

    `invert` succeeding only means the job started: it can still fail in every window.
    """
    name = "the inversion gave every window a model"

    def check(trial: Trial) -> CheckResult:
        paths = sorted(trial.output_dir.glob("*/*/inversion.json"))
        if not paths:
            return CheckResult(name=name, passed=False, detail="no inversion on disk")
        problems: list[str] = []
        for path in paths:
            record = InversionRecord.model_validate_json(path.read_text())
            failed = sum(window.status == "failed" for window in record.windows)
            if record.state != "succeeded" or failed or len(record.windows) < record.total:
                problems.append(
                    f"{record.job_id} {record.state}, {failed} of {record.total} windows failed"
                )
        return CheckResult(name=name, passed=not problems, detail="; ".join(problems))

    return check


def good_windows(trial: Trial) -> str:
    """The number of good windows of the latest run the agent judged."""
    paths = sorted(trial.output_dir.glob("*/*/quality.json"), key=lambda path: path.parent.name)
    if not paths:
        return "(no judged run)"
    record = RunQuality.model_validate_json(paths[-1].read_text())
    return str(sum(window.quality.verdict == "good" for window in record.windows))


def job_id(trial: Trial) -> str:
    """The ID of the inversion job the agent started."""
    paths = sorted(trial.output_dir.glob("*/*/inversion.json"))
    if not paths:
        return "(no job)"
    return InversionRecord.model_validate_json(paths[-1].read_text()).job_id


def _called(trial: Trial) -> list[ToolStep]:
    return [step for step in trial.transcript.tool_steps if step.called]


def _contains(actual: Any, expected: Any) -> bool:  # noqa: ANN401
    if isinstance(expected, dict) and isinstance(actual, str):
        # An object sent as JSON text: the SDK decodes it, and the server gets the object.
        try:
            actual = json.loads(actual)
        except json.JSONDecodeError:
            return False
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _mentions(answer: str, value: str) -> bool:
    if re.fullmatch(r"\d+(\.\d+)?", value):
        # 4 must not match 24, nor 0.25 match 10.25.
        return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d]|\.\d)", answer) is not None
    return value.lower() in answer.lower()
