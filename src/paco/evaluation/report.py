"""The evaluation report, as a table for the terminal."""

from paco.evaluation.models import EvaluationReport, ScenarioResult


def format_report(report: EvaluationReport) -> str:
    judge = report.judge_model or "none"
    repeated = report.repeat > 1
    title = f"Evaluation {report.eval_id} of {report.model} (judge: {judge})"
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
            lines.append(f"  {name:{width}} {sum(play.passed for play in plays)}/{len(plays)}")
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


def _label(result: ScenarioResult, repeated: bool) -> str:
    return f"{result.name} #{result.attempt}" if repeated else result.name


def _by_scenario(results: tuple[ScenarioResult, ...]) -> dict[str, list[ScenarioResult]]:
    plays: dict[str, list[ScenarioResult]] = {}
    for result in results:
        plays.setdefault(result.name, []).append(result)
    return plays
