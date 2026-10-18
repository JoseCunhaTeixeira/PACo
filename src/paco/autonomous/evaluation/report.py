"""The A/B report (docs/autonomous_plan.md §12), for the terminal: each scenario's table, an agent a column
(PACo's, then each variant of the autonomous agent), the plays' checks and the judge's grades,
then the results, the cost and the process as the median of the plays, the stability as their
spread, and a known-truth scenario's scores; then each scenario's two agents at the same CPU
budget, the failed checks and the plays lost."""

import statistics
from collections.abc import Callable, Sequence

from paco.autonomous.evaluation.measures import Measures, spread
from paco.autonomous.evaluation.running import ABReport, Play

type Row = tuple[str, Callable[[Sequence[Play]], str]]
_WIDTH = 30  # the rows' labels
_CELL = 20  # each column's


def format_ab(report: ABReport, prices: tuple[float, float, float] | None = None) -> str:
    """The report as a table; `prices`: the model's, in dollars a million prompt tokens (those
    read from the cache apart), completion tokens and cached tokens, for the money each play
    cost."""
    workflow = f"the workflow agent ({report.workflow_model}, {report.workflow_prompts})"
    title = (
        f"A/B evaluation {report.eval_id}: "
        f"{workflow if report.workflow_model else 'the workflow agent not played'} against the "
        f"autonomous agent ({report.autonomous_model}, {report.autonomous_prompts}), {report.repeat} play"
        f"{'s' if report.repeat != 1 else ''} of each; judge: {report.judge_model or 'none'}; "
        f"inversions: {_INVERSIONS[report.inversions]}."
    )
    lines = [title]
    for scenario, questions in report.questions.items():
        plays = [play for play in report.plays if play.scenario == scenario]
        if not plays:
            continue
        said = " / ".join(f'"{question}"' for question in questions)
        lines += ["", f"{scenario}, on {report.profiles.get(scenario, '?')}: {said}"]
        lines += _table(plays, prices)
    lines += _same_cpu(report)
    failed = [
        f"  {play.scenario} {play.label} #{play.attempt}: {check.name} ({check.detail})"
        for play in report.plays
        for check in play.checks
        if not check.passed
    ]
    if failed:
        lines += ["", "Failed checks:", *failed]
    lost = [
        f"  {play.scenario} {play.label} #{play.attempt}: {play.error}"
        for play in report.plays
        if play.error is not None
    ]
    if lost:
        lines += ["", "Plays the model's server cut short, not counted:", *lost]
    return "\n".join(lines)


_INVERSIONS = {
    "scenario": "quick, but where a scenario judges the models (PAC's effort)",
    "quick": "every play's quick",
    "full": "every play's at PAC's effort",
}


def _table(plays: Sequence[Play], prices: tuple[float, float, float] | None) -> list[str]:
    """A scenario's table: a column for each agent, in the order played."""
    labels = list(dict.fromkeys(play.label for play in plays))
    columns = [[play for play in plays if play.label == label] for label in labels]
    rows: list[Row] = [*_ROWS]
    if prices is not None:
        rows.insert(_ROWS.index(_MONEY_AFTER) + 1, ("  money ($)", _money(prices)))
    if any(play.truth is not None for play in plays):
        rows += _TRUTH_ROWS
    lines = [f"{'':{_WIDTH}}" + "".join(f" {label:>{_CELL - 1}}" for label in labels)]
    for label, cell in rows:
        if not label.startswith(" "):  # a heading
            lines.append(label)
            continue
        cells = [cell([play for play in column if play.error is None]) for column in columns]
        lines.append(f"{label:{_WIDTH}}" + "".join(f" {one:>{_CELL - 1}}" for one in cells))
    return lines


def _median(name: str, digits: int = 2) -> Callable[[Sequence[Play]], str]:
    """The median over the plays of measure `name`."""

    def cell(plays: Sequence[Play]) -> str:
        values = [getattr(play.measures, name) for play in plays]
        known = [value for value in values if value is not None]
        return _number(statistics.median(known), digits) if known else "-"

    return cell


def _spread(name: str, digits: int = 2) -> Callable[[Sequence[Play]], str]:
    """The smallest and largest over the plays of measure `name` (or property)."""

    def cell(plays: Sequence[Play]) -> str:
        found = spread([getattr(play.measures, name) for play in plays])
        if found is None:
            return "-"
        low, high = found
        return f"{_number(low, digits)}-{_number(high, digits)}"

    return cell


def _counted(plays: Sequence[Play]) -> str:
    return str(len(plays))


def _passed(plays: Sequence[Play]) -> str:
    return f"{sum(play.passed for play in plays)}/{len(plays)}" if plays else "-"


def _judged(kind: str) -> Callable[[Sequence[Play]], str]:
    """The judge's mean grade over the plays, of the answer or of the journal's reasoning."""

    def cell(plays: Sequence[Play]) -> str:
        grades = [getattr(play, kind) for play in plays]
        scores = [grade.score for grade in grades if grade is not None and grade.score is not None]
        return f"{statistics.mean(scores):.1f}" if scores else "-"

    return cell


def _accepted(plays: Sequence[Play]) -> str:
    accepted = [play.measures.accepted for play in plays if play.measures.accepted is not None]
    windows = [play.measures.windows for play in plays if play.measures.windows is not None]
    if not accepted or not windows:
        return "-"
    return f"{_number(statistics.median(accepted), 1)}/{_number(statistics.median(windows), 1)}"


def _tokens(plays: Sequence[Play]) -> str:
    if not plays:
        return "-"
    prompt = statistics.median(play.measures.prompt_tokens for play in plays)
    cached = statistics.median(play.measures.cached_tokens for play in plays)
    completion = statistics.median(play.measures.completion_tokens for play in plays)
    return f"{_thousands(prompt)} ({_thousands(cached)})/{_thousands(completion)}"


def _minutes(name: str) -> Callable[[Sequence[Play]], str]:
    def cell(plays: Sequence[Play]) -> str:
        values = [getattr(play.measures, name) / 60 for play in plays]
        return f"{statistics.median(values):.1f}" if values else "-"

    return cell


def _calls(plays: Sequence[Play]) -> str:
    if not plays:
        return "-"
    refused = statistics.median(play.measures.refused for play in plays)
    failed = statistics.median(play.measures.failed for play in plays)
    return f"{_number(refused, 1)}/{_number(failed, 1)}"


def _money(prices: tuple[float, float, float]) -> Callable[[Sequence[Play]], str]:
    def cell(plays: Sequence[Play]) -> str:
        costs = [_cost(play.measures, prices) for play in plays]
        return f"{statistics.median(costs):.2f}" if costs else "-"

    return cell


def _cost(measures: Measures, prices: tuple[float, float, float]) -> float:
    prompt, completion, cached = prices
    fresh = measures.prompt_tokens - measures.cached_tokens
    spent = (
        fresh * prompt + measures.cached_tokens * cached + measures.completion_tokens * completion
    )
    return spent / 1e6


def _truth(name: str) -> Callable[[Sequence[Play]], str]:
    def cell(plays: Sequence[Play]) -> str:
        values = [getattr(play.truth, name) for play in plays if play.truth is not None]
        known = [value for value in values if value is not None]
        return _number(statistics.median(known), 3) if known else "-"

    return cell


_MONEY_AFTER: Row = ("  tokens, prompt (cached)/out", _tokens)
_ROWS: list[Row] = [
    ("  plays counted", _counted),
    ("  plays passed", _passed),
    ("  judge: the answer (1-5)", _judged("judge")),
    ("  judge: the reasoning (1-5)", _judged("reasoning")),
    ("Results (the plays' median)", _counted),
    ("  windows accepted", _accepted),
    ("  gaps", _median("gaps", 1)),
    ("  lambda max/2 (m)", _median("depth_m")),
    ("  uncertainty", _median("uncertainty", 3)),
    ("  G4 outliers", _median("outliers", 1)),
    ("  G5 pass share", _median("g5_pass")),
    ("  misfit", _median("misfit")),
    ("  depth informed (m)", _median("depth_informed_m")),
    ("Cost (the plays' median)", _counted),
    ("  CPU (min)", _minutes("cpu_s")),
    ("  wall-clock (min)", _minutes("wall_s")),
    ("  model calls", _median("model_calls", 1)),
    _MONEY_AFTER,
    ("Process (the plays' median)", _counted),
    ("  experiments", _median("experiments", 1)),
    ("  commits", _median("commits", 1)),
    ("  returns after the picks", _median("returns", 1)),
    ("  calls refused/failed", _calls),
    ("Stability (the plays' spread)", _counted),
    ("  window length (receivers)", _spread("length", 1)),
    ("  nearest shot (m)", _spread("distance_min_m")),
    ("  farthest shot (m)", _spread("distance_max_m")),
    ("  coverage", _spread("coverage")),
    ("  lambda max/2 (m)", _spread("depth_m")),
]
_TRUTH_ROWS: list[Row] = [
    ("Known truth (the plays' median)", _counted),
    ("  models scored", _truth("windows")),
    ("  Vs error, to the depth informed", _truth("error")),
    ("  truth within P10-P90", _truth("in_band")),
]


def _same_cpu(report: ABReport) -> list[str]:
    """Each scenario's PACo agent against the autonomous agent's variant whose median CPU time is
    nearest its own: the two at the same CPU budget (§12)."""
    lines: list[str] = []
    for scenario in report.questions:
        plays = [play for play in report.plays if play.scenario == scenario and play.error is None]
        workflow = [play for play in plays if play.agent == "workflow"]
        labels = list(dict.fromkeys(play.label for play in plays if play.agent == "autonomous"))
        if not workflow or not labels:
            continue
        budget = statistics.median(play.measures.cpu_s for play in workflow)
        cpus = {
            label: statistics.median(play.measures.cpu_s for play in plays if play.label == label)
            for label in labels
        }
        nearest = min(labels, key=lambda label: abs(cpus[label] - budget))
        autonomous = [play for play in plays if play.label == nearest]
        said = "; ".join(
            f"{name} {cell(workflow)} against {cell(autonomous)}"
            for name, cell in (
                ("windows accepted", _accepted),
                ("lambda max/2", _median("depth_m")),
                ("G5 pass share", _median("g5_pass")),
                ("depth informed", _median("depth_informed_m")),
            )
        )
        lines.append(
            f"  {scenario}: PACo {budget / 60:.1f} CPU min, {nearest} {cpus[nearest] / 60:.1f}: "
            f"{said}"
        )
    if not lines:
        return []
    return [
        "",
        "At the same CPU budget (the workflow agent against the autonomous agent's variant nearest it "
        "in CPU time; --variants quick normal thorough brackets it):",
        *lines,
    ]


def _number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{value:.0f}"


def _thousands(value: float) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"
