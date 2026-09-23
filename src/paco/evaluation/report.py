"""The evaluation report, as a table for the terminal."""

from paco.evaluation.models import EvaluationReport


def format_report(report: EvaluationReport) -> str:
    judge = report.judge_model or "none"
    lines = [
        f"Evaluation {report.eval_id} of {report.model} (judge: {judge})",
        "",
        f"{'scenario':20} {'kind':18} {'checks':>6} {'judge':>5} {'calls':>5} {'failed':>6} "
        f"{'max prompt':>10} {'time':>7}",
    ]
    for result in report.results:
        passed = sum(check.passed for check in result.checks)
        score = (
            "-" if result.judge is None or result.judge.score is None else str(result.judge.score)
        )
        tokens = "-" if result.max_prompt_tokens is None else f"{result.max_prompt_tokens:,}"
        lines.append(
            f"{result.name:20} {result.kind:18} {passed:>3}/{len(result.checks):<2} {score:>5} "
            f"{result.tool_calls:>5} {result.failed_calls:>6} {tokens:>10} "
            f"{result.duration_s:>6.1f}s"
        )

    passed = sum(result.passed for result in report.results)
    scores = [
        result.judge.score
        for result in report.results
        if result.judge is not None and result.judge.score is not None
    ]
    lines += ["", f"Passed {passed} of {len(report.results)} scenarios."]
    if scores:
        lines.append(f"Mean judge score: {sum(scores) / len(scores):.1f} of 5.")
    failures = [
        f"  {result.name}: {check.name} ({check.detail})"
        for result in report.results
        for check in result.checks
        if not check.passed
    ]
    if failures:
        lines += ["Failed checks:", *failures]
    return "\n".join(lines)
