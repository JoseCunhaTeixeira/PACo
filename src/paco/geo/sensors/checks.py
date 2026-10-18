"""What a window's picks say about the line's settings (docs/geo_plan.md §4.2, §4.6): from each
curve's G3 result, its longest wavelength against the window's length, its points beyond the
window's reach and in the aliasing zone, its nearest shot against its near field, the depth it
informs and its precision; then their medians over the line, and the runs of windows that do not
pass. The measures and their limits are G3's (sigpipe's 2·dx and 3·L, half the longest wavelength
for the near field): no threshold of the package's own."""

import statistics
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

from paco.qc.models import GateResult, Metric
from paco.qc.report import xmid_of


@dataclass(frozen=True, slots=True)
class Checks:
    """One window's curve, as its G3 result measures it. None where G3 could not measure."""

    xmid: float
    shortest_m: float  # the curve's shortest wavelength
    longest_m: float  # its longest
    reach: float  # its longest wavelength over the window's length: over 3, beyond its reach
    beyond: float | None  # the share of its points beyond 3 window lengths (beyond_reach_points)
    aliased: float | None  # the share under twice the receiver spacing (aliased_points)
    # Its nearest shot's distance over half its longest wavelength: under 1, the near field.
    nearest_shot: float | None
    uncertainty: float | None  # its points' median relative uncertainty

    @property
    def depth_m(self) -> float:
        """The depth the curve informs: half its longest wavelength, PACo's rule for the
        inversion's deepest interface."""
        return self.longest_m / 2

    @property
    def span(self) -> float:
        """Its longest wavelength over its shortest."""
        return self.longest_m / self.shortest_m

    @property
    def in_near_field(self) -> bool:
        """Whether its nearest shot lies within half its longest wavelength (G3's near_field)."""
        return self.nearest_shot is not None and self.nearest_shot < 1


@dataclass(frozen=True, slots=True)
class LineChecks:
    """The checks over the line's curves: their medians, and how many curves each limit
    concerns."""

    curves: int
    reach: float | None  # the median longest wavelength over the window's length
    beyond_windows: int  # curves with points beyond the window's reach
    beyond: float | None  # the median share of points beyond it
    aliased_windows: int  # curves with points in the aliasing zone
    aliased: float | None  # the median share in it
    near_field_windows: int  # curves whose nearest shot lies in their near field
    depth_m: float | None  # the median depth informed
    reaching: int | None  # curves informing the target depth, when the objective has one
    uncertainty: float | None  # the median of the curves' median relative uncertainty


@dataclass(frozen=True, slots=True)
class Run:
    """Consecutive windows of the line, in its order, that do not pass: at its ends, windows with
    near shots only are the likely cause; inside, noise or geology (docs/geo_plan.md §4.6)."""

    xmids: tuple[float, ...]
    at_end: bool  # it holds the line's first or last window
    near_only: tuple[float, ...]  # those of its windows that stack near shots only


def window_checks(g3: GateResult, length_m: float, shots: bool = True) -> Checks | None:
    """The checks of the curve G3 judged in `g3`, on a window `length_m` long from its first
    receiver to its last (sigpipe's window_length); None when the window has no curve. Without
    `shots` (a passive line's windows, whose virtual source is their first receiver), no nearest
    shot."""
    xmid = xmid_of(g3.unit)
    wavelengths = g3.kept.wavelength_m
    if xmid is None or wavelengths is None:
        return None
    shortest, longest = wavelengths
    near = _metric(g3, "near_offset")
    nearest_shot = (
        near.value / near.threshold
        if shots and near is not None and near.value is not None and near.threshold
        else None
    )
    return Checks(
        xmid=xmid,
        shortest_m=shortest,
        longest_m=longest,
        reach=longest / length_m,
        beyond=_value(g3, "beyond_reach_points"),
        aliased=_value(g3, "aliased_points"),
        nearest_shot=nearest_shot,
        uncertainty=_value(g3, "uncertainty"),
    )


def shot_note(flag_name: str, mode: str) -> bool:
    """Whether a flag G3 raises reads a window's shots where the line has none: its near field
    on a passive line, whose windows' virtual source is their first receiver."""
    return mode == "passive" and flag_name == "near_field"


def line_checks(checks: Collection[Checks], target_depth_m: float | None = None) -> LineChecks:
    """The checks over `checks`, one a curve; with `target_depth_m`, how many curves inform it."""
    return LineChecks(
        curves=len(checks),
        reach=median_known(check.reach for check in checks),
        beyond_windows=sum(1 for check in checks if check.beyond),
        beyond=median_known(check.beyond for check in checks),
        aliased_windows=sum(1 for check in checks if check.aliased),
        aliased=median_known(check.aliased for check in checks),
        near_field_windows=sum(1 for check in checks if check.in_near_field),
        depth_m=median_known(check.depth_m for check in checks),
        reaching=sum(1 for check in checks if check.depth_m >= target_depth_m)
        if target_depth_m is not None
        else None,
        uncertainty=median_known(check.uncertainty for check in checks),
    )


def failing_runs(
    passing: Mapping[float, bool], near_only: Collection[float] = ()
) -> tuple[Run, ...]:
    """The runs of consecutive windows that do not pass, in the line's order. `passing`: every
    window of the line, by xmid, and whether it passes; `near_only`: the windows that stack near
    shots only (sigpipe's near-field rule keeps them, §4.3)."""
    order = sorted(passing)
    runs: list[list[float]] = []
    for index, xmid in enumerate(order):
        if passing[xmid]:
            continue
        if runs and runs[-1][-1] == order[index - 1]:
            runs[-1].append(xmid)
        else:
            runs.append([xmid])
    ends = {order[0], order[-1]} if order else set[float]()
    return tuple(
        Run(
            xmids=tuple(run),
            at_end=bool(ends.intersection(run)),
            near_only=tuple(xmid for xmid in run if xmid in near_only),
        )
        for run in runs
    )


def _metric(result: GateResult, name: str) -> Metric | None:
    return next((metric for metric in result.metrics if metric.name == name), None)


def _value(result: GateResult, name: str) -> float | None:
    metric = _metric(result, name)
    return metric.value if metric is not None else None


def median_known(values: Iterable[float | None]) -> float | None:
    """The median of the known `values`; None when none is."""
    known = [value for value in values if value is not None]
    return statistics.median(known) if known else None
