"""The evaluation report, as a table for the terminal; and the pass rates of every evaluation
kept, by model and prompt version (E4)."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from paco.evaluation.models import EvaluationReport, ScenarioResult


def format_report(report: EvaluationReport) -> str:
    judge = report.judge_model or "none"
    repeated = report.repeat > 1
    title = f"Evaluation {report.eval_id} of {report.model} (judge: {judge})"
    if report.prompt_version:
        title += f" on {report.prompt_version}"
    if repeated:
        title += f", {report.repeat} plays of each scenario"
    labels = [_label(result, repeated) for result in report.results]
    width = max([20, *map(len, labels)])
    lines = [
        title,
        "",
        f"{'scenario':{width}} {'kind':18} {'checks':>6} {'judge':>5} {'calls':>5} "
        f"{'failed':>6} {'max prompt':>10} {'time':>7}",
    ]
    for label, result in zip(labels, report.results, strict=True):
        passed = sum(check.passed for check in result.checks)
        score = (
            "-" if result.judge is None or result.judge.score is None else str(result.judge.score)
        )
        tokens = "-" if result.max_prompt_tokens is None else f"{result.max_prompt_tokens:,}"
        lines.append(
            f"{label:{width}} {result.kind:18} {passed:>3}/{len(result.checks):<2} {score:>5} "
            f"{result.tool_calls:>5} {result.failed_calls:>6} {tokens:>10} "
            f"{result.duration_s:>6.1f}s"
        )

    passed = sum(result.passed for result in report.results)
    if repeated:
        lines += ["", f"Passed {passed} of {len(report.results)} plays:"]
        for name, plays in _by_scenario(report.results).items():
            lines.append(
                f"  {name:{width}} {sum(play.passed for play in plays)}/{len(plays)}"
                + _verdict(report, name)
            )
        judged = [name for name in report.pass_rates() if report.meets(name) is not None]
        if judged:
            met = sum(bool(report.meets(name)) for name in judged)
            lines.append(f"{met} of {len(judged)} scenarios meet their pass-rate threshold.")
    else:
        lines += ["", f"Passed {passed} of {len(report.results)} scenarios."]
    scores = [
        result.judge.score
        for result in report.results
        if result.judge is not None and result.judge.score is not None
    ]
    if scores:
        lines.append(f"Mean judge score: {sum(scores) / len(scores):.1f} of 5.")
    failures = [
        f"  {label}: {check.name} ({check.detail})"
        for label, result in zip(labels, report.results, strict=True)
        for check in result.checks
        if not check.passed
    ]
    if failures:
        lines += ["Failed checks:", *failures]
    return "\n".join(lines)


def read_reports(root: Path) -> list[EvaluationReport]:
    """Every evaluation's report under `root`, oldest first; one that does not read, skipped."""
    reports: list[EvaluationReport] = []
    for path in sorted(root.glob("*/report.json")):
        try:
            reports.append(EvaluationReport.model_validate_json(path.read_text()))
        except ValidationError:
            continue
    return sorted(reports, key=lambda report: report.started_at)


def format_history(reports: Iterable[EvaluationReport], model: str | None = None) -> str:
    """Each scenario's pass rate by model and prompt version, the plays of every evaluation of
    that pair summed (E4); for each model, its latest prompt version against the one before."""
    rates: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    order: dict[str, list[str]] = {}
    for report in reports:
        if model is not None and report.model != model:
            continue
        version = report.prompt_version or "unversioned"
        key = (report.model, version)
        if version not in order.setdefault(report.model, []):
            order[report.model].append(version)
        into = rates.setdefault(key, {})
        for name, (passed, played) in report.pass_rates().items():
            before = into.get(name, (0, 0))
            into[name] = (before[0] + passed, before[1] + played)
    if not rates:
        return "No evaluation kept" + (f" of {model}." if model else ".")
    lines: list[str] = []
    for name, versions in order.items():
        shown = versions[-2:]
        names = sorted({scenario for version in shown for scenario in rates[name, version]})
        width = max([20, *map(len, names)])
        lines += ["", name, f"{'scenario':{width}} " + " ".join(f"{one:>18}" for one in shown)]
        for scenario in names:
            cells = []
            for version in shown:
                passed, played = rates[name, version].get(scenario, (0, 0))
                cells.append(f"{passed:>2}/{played:<2} {passed / played:>6.0%}" if played else "-")
            lines.append(f"{scenario:{width}} " + " ".join(f"{cell:>18}" for cell in cells))
    return "\n".join(lines).strip("\n")


def _verdict(report: EvaluationReport, scenario: str) -> str:
    meets = report.meets(scenario)
    if meets is None:
        return ""
    threshold = report.thresholds[scenario]
    return f"  ({'meets' if meets else 'below'} {threshold:.0%})"


def _label(result: ScenarioResult, repeated: bool) -> str:
    return f"{result.name} #{result.attempt}" if repeated else result.name


def _by_scenario(results: tuple[ScenarioResult, ...]) -> dict[str, list[ScenarioResult]]:
    plays: dict[str, list[ScenarioResult]] = {}
    for result in results:
        plays.setdefault(result.name, []).append(result)
    return plays
