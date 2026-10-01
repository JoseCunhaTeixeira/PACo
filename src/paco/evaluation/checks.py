"""Rule checks on a conversation: what the agent called, what it answered, what it left on disk."""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sigpipe.masw.runs import RunManifest

from paco.agent.record import ToolStep, Transcript
from paco.evaluation.models import CheckResult
from paco.inversion import InversionRecord
from paco.qc import (
    QCConfig,
    QCReport,
    Stage,
    load_qc_config,
    read_attempts,
    read_length_choice,
    run_work,
)
from paco.qc.attempts import HAND_FOLDER
from paco.qc.log import JUDGED
from paco.qc.models import Reject
from paco.settings import get_settings


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
    """Every run used the server's configuration of thresholds (PACO_QC_CONFIG's, else PACo's
    defaults): the judge stays fixed."""
    name = "the thresholds stayed the configuration's"

    def check(trial: Trial) -> CheckResult:
        expected = load_qc_config(get_settings().qc_config)
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


def kept_as_given(stage: Stage, *path: str | int, value: float) -> Check:
    """No attempt of `stage` ran with another value than `value` at `path` (e.g. "vs_layers",
    -1, "vs_max"): the setting the user gave stayed as given (qc.given)."""
    shown = " ".join(str(key) for key in path)
    name = f"{shown} stayed {value:g}, as given"

    def check(trial: Trial) -> CheckResult:
        other = sorted(
            {
                f"{found:g}" if isinstance(found, float) else str(found)
                for log in trial.output_dir.glob("*/*/qc_log.jsonl")
                for attempt in read_attempts(log.parent)
                if attempt.stage == stage
                and (found := _at(attempt.parameters, path)) is not None
                and found != value
            }
        )
        detail = f"ran with {', '.join(other)}" if other else ""
        return CheckResult(name=name, passed=not other, detail=detail)

    return check


def top_vs_kept(value: float) -> Check:
    """No inversion ran with another upper Vs bound than `value`, as given: the half-space's
    of the layers given (`vs_layers`), or the free layering's (`free.vs_max`), whichever ran."""
    name = f"the upper Vs bound stayed {value:g}, as given"

    def check(trial: Trial) -> CheckResult:
        other = sorted(
            {
                f"{found:g}"
                for log in trial.output_dir.glob("*/*/qc_log.jsonl")
                for attempt in read_attempts(log.parent)
                if attempt.stage == "inversion"
                and (found := _top_vs(attempt.parameters)) is not None
                and found != value
            }
        )
        detail = f"ran with {', '.join(other)}" if other else ""
        return CheckResult(name=name, passed=not other, detail=detail)

    return check


def _top_vs(parameters: dict[str, Any]) -> float | None:
    """An inversion's upper Vs bound, of the layering it ran with."""
    if parameters.get("layering", "fixed" if "vs_layers" in parameters else "free") == "free":
        return _at(parameters, ("free", "vs_max"))
    return _at(parameters, ("vs_layers", -1, "vs_max"))


def refused_as_locked(gate: str) -> Check:
    """`gate` asked a change of a setting the user gave, and its window was left out for it
    ("locked"), in some run of the scenario."""
    name = f"{gate} left a window out over a setting given"

    def check(trial: Trial) -> CheckResult:
        passed = _locked_flag(trial, gate) is not None
        detail = "" if passed else f"no {gate} result refused as locked"
        return CheckResult(name=name, passed=passed, detail=detail)

    return check


def locked_asks(gate: str) -> Callable[[Trial], str]:
    """The value `gate` asked of a setting the user gave, in its first refusal as locked: "900"
    of "locked, asks dispersion vmax 900 (given: 250)"."""

    def fact(trial: Trial) -> str:
        reason = _locked_flag(trial, gate)
        found = _ASKED.search(reason) if reason is not None else None
        return found.group(1) if found is not None else "(no locked refusal)"

    return fact


def checks_ask(trial: Trial) -> str:
    """The value the checks before the inversion asked of a bound the user gave, kept as given:
    "450" of "kept as given (the check sets 450 m/s)"."""
    for log in sorted(trial.output_dir.glob("*/*/qc_log.jsonl")):
        for attempt in read_attempts(log.parent):
            for note in attempt.notes if attempt.stage == "inversion" else ():
                if (found := _CHECK_SETS.search(note)) is not None:
                    return found.group(1)
    return "(no bound kept as given)"


def line_muted(vmin: float | None = None, vmax: float | None = None) -> Check:
    """The latest run's line is muted: with `vmin` and `vmax`, exactly those (a muting given)."""
    bounds = f" {vmin:g} to {vmax:g} m/s" if vmin is not None and vmax is not None else ""
    name = f"the line muted{bounds}"

    def check(trial: Trial) -> CheckResult:
        manifest = _latest_manifest(trial)
        if manifest is None:
            return CheckResult(name=name, passed=False, detail="no run on disk")
        muting = manifest.preset.model_dump(mode="json").get("muting") or {}
        muted = muting.get("method") == "mute"
        exact = vmin is None or (muting.get("vmin"), muting.get("vmax")) == (vmin, vmax)
        detail = "" if muted and exact else f"muting: {muting or 'none'}"
        return CheckResult(name=name, passed=muted and exact, detail=detail)

    return check


def compare_best(trial: Trial) -> str:
    """The metric's value of the best variant of the latest comparison the agent made."""
    for step in reversed(_called(trial)):
        if step.name != "compare" or step.is_error:
            continue
        try:
            compared = json.loads(step.result)
        except json.JSONDecodeError:
            break
        best = next(
            (
                one
                for one in compared.get("variants", ())
                if one.get("label") == compared.get("best")
            ),
            None,
        )
        if best is not None and best.get("value") is not None:
            return f"{best['value']:g}"
        break
    return "(no comparison)"


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
    from the user, every parameter comes from the data, and the length is the agent's to choose."""
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


def nothing_given(tool: str, argument: str) -> Check:
    """Every successful call of `tool` left `argument` empty: the user gave no value for it."""
    name = f"{tool} with no {argument}"

    def check(trial: Trial) -> CheckResult:
        given = [
            step.arguments
            for step in _called(trial)
            if step.name == tool
            and not step.is_error
            and json.loads(step.arguments or "{}").get(argument)
        ]
        detail = f"called with {'; '.join(given)}" if given else ""
        return CheckResult(name=name, passed=not given, detail=detail)

    return check


def kept_by_hand(unit: str) -> Check:
    """Window `unit`'s M0, picked by hand before the conversation, is the person's still."""
    name = f"{unit}'s curve picked by hand kept"

    def check(trial: Trial) -> CheckResult:
        manifest = _latest_manifest(trial)
        if manifest is None:
            return CheckResult(name=name, passed=False, detail="no run on disk")
        run_folder = next(trial.output_dir.glob(f"*/{manifest.run_id}"))
        state = run_work(run_folder, manifest)[unit].m0
        return CheckResult(name=name, passed=state == "user", detail=f"its M0: {state}")

    return check


def nothing_redone(*stages: Stage) -> Check:
    """No attempt at `stages` started during the conversation (a judgement aside): the work
    already there was gone on from, whatever the agent called."""
    name = "nothing redone: " + ", ".join(stages)

    def check(trial: Trial) -> CheckResult:
        started = trial.transcript.started_at
        redone = [
            f"{attempt.unit} {attempt.stage}"
            for folder in trial.output_dir.glob("*/*")
            if folder.is_dir()
            for attempt in read_attempts(folder)
            if attempt.stage in stages
            and attempt.started_at >= started
            and attempt.triggered_by != JUDGED
        ]
        detail = f"{len(redone)} attempt(s), as {redone[0]}" if redone else ""
        return CheckResult(name=name, passed=not redone, detail=detail)

    return check


def replaced_by_hand(unit: str) -> Check:
    """Window `unit`'s M0, picked by hand before the conversation, was picked again as the
    message asked, the person's set aside in the window's by_hand folder."""
    name = f"{unit}'s curve picked by hand replaced, set aside"

    def check(trial: Trial) -> CheckResult:
        manifest = _latest_manifest(trial)
        if manifest is None:
            return CheckResult(name=name, passed=False, detail="no run on disk")
        run_folder = next(trial.output_dir.glob(f"*/{manifest.run_id}"))
        state = run_work(run_folder, manifest)[unit].m0
        aside = list((run_folder / unit / HAND_FOLDER).glob("picking_*"))
        passed = state != "user" and len(aside) == 1
        return CheckResult(
            name=name, passed=passed, detail=f"its M0: {state}, {len(aside)} set aside"
        )

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


def inverted_windows(count: int) -> Check:
    """The latest inversion took `count` windows, and gave each a model."""
    name = f"the inversion took {count} window(s)"

    def check(trial: Trial) -> CheckResult:
        paths = sorted(
            trial.output_dir.glob("*/*/inversion.json"), key=lambda one: one.stat().st_mtime
        )
        if not paths:
            return CheckResult(name=name, passed=False, detail="no inversion on disk")
        record = InversionRecord.model_validate_json(paths[-1].read_text())
        done = sum(window.status != "failed" for window in record.windows)
        return CheckResult(
            name=name,
            passed=record.total == count and done == count,
            detail=f"{record.total} taken, {done} with a model",
        )

    return check


def inverted_every_curve() -> Check:
    """Every window of the latest run holding an M0 curve got a model: the curves picked by hand
    taken as they are, none left out for want of a gate's verdict."""
    name = "every curve was inverted"

    def check(trial: Trial) -> CheckResult:
        manifest = _latest_manifest(trial)
        if manifest is None:
            return CheckResult(name=name, passed=False, detail="no run on disk")
        run_folder = next(trial.output_dir.glob(f"*/{manifest.run_id}"))
        work = run_work(run_folder, manifest)
        missing = [unit for unit, one in work.items() if one.m0 is not None and one.model is None]
        return CheckResult(
            name=name,
            passed=not missing,
            detail=f"no model at {', '.join(missing)}" if missing else "",
        )

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


def water_table(trial: Trial) -> str:
    """The shallowest water table of the latest run's petrophysical models G7 and G8 passed."""
    report = _latest_report(trial)
    paths = sorted(trial.output_dir.glob("*/*/qc_report.json"), key=lambda path: path.parent.name)
    if report is None or not paths:
        return "(no judged run)"
    depths = [
        float(json.loads(measures.read_text())["water_table_m"])
        for unit in report.units
        if unit.verdicts.get("G7") == "pass"
        and unit.verdicts.get("G8") == "pass"
        and (
            measures := paths[-1].parent / unit.unit / "PetroInversion_Measures_0000.json"
        ).exists()
    ]
    return f"{min(depths):g}" if depths else "(no petrophysical model)"


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


# A question without a question mark: options offered for the user to pick (e.g. "Choose one to
# proceed.", an <options> block).
# An answer that asks without a question mark: options to choose from, or a preference asked.
_CHOICE = re.compile(
    r"\bchoose\b|\bwhich (one|option|you prefer)\b|<options>|\breply with\b"
    r"|\byour (choice|choices|preference)\b|\bselect (one|your)\b|\byou prefer\b",
    re.IGNORECASE,
)


# The value a refusal as locked asks, and the value a check asks of a bound kept as given.
_ASKED = re.compile(r"asks .*?(-?\d+(?:\.\d+)?) \(given: ")
_CHECK_SETS = re.compile(r"kept as given \(the check sets (-?\d+(?:\.\d+)?)")


def _locked_flag(trial: Trial, gate: str) -> str | None:
    """The reason of `gate`'s first refusal as locked, if any."""
    for log in sorted(trial.output_dir.glob("*/*/qc_log.jsonl")):
        for attempt in read_attempts(log.parent):
            result = attempt.results.get(gate)
            for flag in result.flags if result is not None else ():
                if flag.name == "locked" and isinstance(flag.action, Reject):
                    return flag.action.reason
    return None


def _at(values: Any, path: Sequence[str | int]) -> Any:  # noqa: ANN401
    """The value at `path` in nested mappings and lists; None where it is missing."""
    for key in path:
        if isinstance(key, int) and isinstance(values, list) and -len(values) <= key < len(values):
            values = values[key]
        elif isinstance(key, str) and isinstance(values, dict):
            values = values.get(key)
        else:
            return None
    return values


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
