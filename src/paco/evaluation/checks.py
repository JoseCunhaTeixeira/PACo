"""Rule checks on a conversation: what the agent called, what it answered, what it left on disk."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from paco.agent.record import ToolStep, Transcript
from paco.evaluation.models import CheckResult
from paco.inversion import InversionRecord
from paco.qc import QCConfig, QCReport, Stage, read_attempts, read_length_choice
from paco.runs import RunManifest


@dataclass(frozen=True)
class Trial:
    """One scenario, played: the conversation, and what the server wrote."""

    transcript: Transcript
    output_dir: Path  # the server's output directory during the scenario


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


def any_of(*checks: Check) -> Check:
    """One of `checks` passes: ways of doing right that the scenario accepts alike."""

    def check(trial: Trial) -> CheckResult:
        results = [check(trial) for check in checks]
        name = " or ".join(result.name for result in results)
        if any(result.passed for result in results):
            return CheckResult(name=name, passed=True)
        detail = "; ".join(f"{result.name}: {result.detail}" for result in results)
        return CheckResult(name=name, passed=False, detail=detail)

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
    """The final answer asks the user something: the agent was stuck, and said so."""

    def check(trial: Trial) -> CheckResult:
        passed = _asks(trial.transcript.answer)
        detail = "" if passed else "the answer asks nothing"
        return CheckResult(name="the agent asked the user", passed=passed, detail=detail)

    return check


def asked_nothing() -> Check:
    """The final answer asks the user nothing: the data decided (docs/qc_workflow.md)."""

    def check(trial: Trial) -> CheckResult:
        passed = not _asks(trial.transcript.answer)
        detail = "" if passed else "the answer asks the user"
        return CheckResult(name="the agent asked nothing", passed=passed, detail=detail)

    return check


def thresholds_unchanged() -> Check:
    """Every run used the server's configuration of thresholds: the judge stays fixed."""
    name = "the thresholds stayed the configuration's"

    def check(trial: Trial) -> CheckResult:
        expected = QCConfig()
        changed = [
            path.parent.name
            for path in trial.output_dir.glob("*/*/qc_config.json")
            if QCConfig.model_validate_json(path.read_text()) != expected
        ]
        detail = f"changed in {', '.join(changed)}" if changed else ""
        return CheckResult(name=name, passed=not changed, detail=detail)

    return check


def loop_retried(trigger: str) -> Check:
    """The QC loop retried a unit for `trigger` ("<gate>:<flag>", or "backtrack": the agent
    went back a stage), in some run of the scenario."""
    name = f"the loop retried for {trigger}"

    def check(trial: Trial) -> CheckResult:
        seen = {
            attempt.triggered_by
            for path in trial.output_dir.glob("*/*/qc_log.jsonl")
            for attempt in read_attempts(path.parent)
        }
        passed = any(one.startswith(trigger) for one in seen)
        detail = (
            "" if passed else f"triggers seen: {', '.join(sorted(seen - {'initial'})) or 'none'}"
        )
        return CheckResult(name=name, passed=passed, detail=detail)

    return check


def excluded(record: str, trace: int) -> Check:
    """The latest run left trace `trace` of `record` out of its windows (G1's fix)."""
    name = f"trace {trace} of {record} left out"

    def check(trial: Trial) -> CheckResult:
        manifest = _latest_manifest(trial)
        if manifest is None:
            return CheckResult(name=name, passed=False, detail="no run on disk")
        passed = trace in manifest.exclusions.traces.get(record, ())
        detail = "" if passed else f"exclusions: {manifest.exclusions.model_dump()}"
        return CheckResult(name=name, passed=passed, detail=detail)

    return check


def processed_in_mode(mode: str) -> Check:
    """Every run on disk was processed in `mode`, however the agent asked for it (run_processing's
    `mode`, or "mode" in its overrides)."""
    name = f"processed in mode {mode}"

    def check(trial: Trial) -> CheckResult:
        paths = sorted(trial.output_dir.glob("*/*/run.json"))
        if not paths:
            return CheckResult(name=name, passed=False, detail="no run on disk")
        modes = [RunManifest.model_validate_json(path.read_text()).preset.mode for path in paths]
        others = [str(one) for one in modes if one != mode]
        detail = f"runs in {', '.join(others)}" if others else ""
        return CheckResult(name=name, passed=not others, detail=detail)

    return check


def no_settings_invented(tool: str) -> Check:
    """Every successful call of `tool` gave no overrides but the window length: with no setting
    from the user, every parameter comes from the data, and the length is the agent's to
    choose (the user's decision of 2026-09-25)."""
    name = f"{tool} with no settings but the window length"

    def check(trial: Trial) -> CheckResult:
        steps = [step for step in _called(trial) if step.name == tool and not step.is_error]
        if not steps:
            return CheckResult(name=name, passed=False, detail=f"{tool} never succeeded")
        given = [
            step.arguments
            for step in steps
            if _beyond_length(json.loads(step.arguments or "{}").get("overrides"))
        ]
        detail = f"called with {'; '.join(given)}" if given else ""
        return CheckResult(name=name, passed=not given, detail=detail)

    return check


def _beyond_length(overrides: Any) -> bool:  # noqa: ANN401
    """Whether `overrides` set anything but masw's window length."""
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    if not overrides:
        return False
    if not isinstance(overrides, dict) or set(overrides) != {"masw"}:
        return True
    masw = overrides["masw"]
    return not isinstance(masw, dict) or set(masw) - {"length"} != set()


def windows_than_proposed(longer: Literal["longer", "shorter"]) -> Check:
    """The agent ran the line again with windows `longer` (or shorter) than the ladder
    proposed in the first run: it chose the length for what the user asked (depth, or lateral
    detail)."""
    name = f"windows {longer} than the ladder proposed"

    def check(trial: Trial) -> CheckResult:
        folders = sorted(
            (path.parent for path in trial.output_dir.glob("*/*/run.json")),
            key=lambda folder: folder.name,
        )
        first = read_length_choice(folders[0]) if folders else None
        if first is None:
            return CheckResult(name=name, passed=False, detail="no run proposed a length")
        last = RunManifest.model_validate_json((folders[-1] / "run.json").read_text())
        chosen = last.preset.masw.length
        passed = chosen > first.length if longer == "longer" else chosen < first.length
        detail = f"proposed {first.length}, last run {chosen}"
        return CheckResult(name=name, passed=passed, detail="" if passed else detail)

    return check


def no_inversion_started() -> Check:
    def check(trial: Trial) -> CheckResult:
        records = list(trial.output_dir.glob("*/*/inversion.json"))
        detail = "" if not records else f"{len(records)} inversion(s) on disk"
        return CheckResult(name="no inversion started", passed=not records, detail=detail)

    return check


def inversion_succeeded() -> Check:
    """The inversion the agent started ran to its end, with a model for every window it took.

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


def curves(trial: Trial) -> str:
    """How many curves of the latest run passed G3 and G4."""
    report = _latest_report(trial)
    if report is None:
        return "(no judged run)"
    passed = [
        unit
        for unit in report.units
        if unit.xmid is not None
        and unit.verdicts.get("G3") == "pass"
        and unit.verdicts.get("G4") == "pass"
    ]
    return str(len(passed))


def models(trial: Trial) -> str:
    """How many windows the latest inversion gave a model."""
    paths = sorted(trial.output_dir.glob("*/*/inversion.json"))
    if not paths:
        return "(no inversion)"
    record = InversionRecord.model_validate_json(paths[-1].read_text())
    return str(sum(window.status == "succeeded" for window in record.windows))


def retried_value(stage: Stage, *path: str) -> Callable[[Trial], str]:
    """The value at `path` in the parameters of the first retry of `stage` a gate made (the
    setting a gate changed, as the summary's example shows it), e.g. ("dispersion", "vmax")."""

    def fact(trial: Trial) -> str:
        value: Any = None
        for log in sorted(trial.output_dir.glob("*/*/qc_log.jsonl")):
            for attempt in read_attempts(log.parent):
                if (
                    value is None
                    and attempt.stage == stage
                    and attempt.triggered_by.startswith("G")
                ):
                    found: Any = attempt.parameters
                    for key in path:
                        found = found.get(key) if isinstance(found, dict) else None
                    if found is not None:
                        value = found
        return (
            "(no retry)"
            if value is None
            else f"{value:g}"
            if isinstance(value, float)
            else str(value)
        )

    return fact


def job_id(trial: Trial) -> str:
    """The ID of the inversion job the agent started."""
    paths = sorted(trial.output_dir.glob("*/*/inversion.json"))
    if not paths:
        return "(no job)"
    return InversionRecord.model_validate_json(paths[-1].read_text()).job_id


# A question without a question mark: options offered for the user to pick (seen from Qwen3-8B:
# "Choose one to proceed.", an <options> block).
_CHOICE = re.compile(r"\bchoose\b|\bwhich (one|option)\b|<options>|\breply with\b", re.IGNORECASE)


def _asks(answer: str) -> bool:
    """Whether an answer asks the user something: a question, or options to choose from."""
    return "?" in answer or _CHOICE.search(answer) is not None


def _latest_manifest(trial: Trial) -> RunManifest | None:
    paths = sorted(trial.output_dir.glob("*/*/run.json"), key=lambda path: path.parent.name)
    return RunManifest.model_validate_json(paths[-1].read_text()) if paths else None


def _latest_report(trial: Trial) -> QCReport | None:
    paths = sorted(trial.output_dir.glob("*/*/qc_report.json"), key=lambda path: path.parent.name)
    return QCReport.model_validate_json(paths[-1].read_text()) if paths else None


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
        # Thousands written with a separator count ("17,000", "17 000"); 4 must not match 24,
        # nor 0.25 match 10.25.
        plain = re.sub(r"(?<=\d)[,\u202f\u00a0 ](?=\d{3}\b)", "", answer)
        return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d]|\.\d)", plain) is not None
    return value.lower() in answer.lower()
