"""G4, the curve profile QC over the whole line (docs/qc_workflow.md): the pseudo-section
Vr(wavelength, xmid) of the curves that passed G3, each curve against its neighbours on either
side, an isolated outlier (off neighbours that agree with each other, fitting no side) told
from a change shared with one side (geology), and the coverage. Its verdict is the go/no-go for
the inversion."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import rankdata
from sigpipe.masw.quality.line import Series, median_series, neighbourhoods, spread

from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Override
from paco.qc.report import line_step, stretches

GATE = "G4"
LINE = "line"  # the unit of the line-level result
GUIDE_POINTS = 12  # the neighbours' median curve given to the picker for an outlier


class ProfileThresholds(BaseModel):
    """G4's limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    neighbours: int = Field(
        default=4, ge=2, description="Curves compared with: half of them on each side of the xmid."
    )
    max_misfit: float = Field(
        default=0.15,
        gt=0,
        description="Median relative misfit to a side's median curve, over the shared "
        "wavelengths, at most: beyond it the curve is off that side. The side's own curves must "
        "agree within half of it to count.",
    )
    min_shared_points: int = Field(
        default=3, ge=1, description="Wavelengths shared with a side's median, to compare at all."
    )
    max_depth_spread: float = Field(
        default=0.5,
        gt=0,
        description="MAD over median of the curves' longest wavelength along the line, at most: "
        "beyond it the depths of investigation are not comparable.",
    )


def judge_profile(
    curves: Sequence[Series],
    thresholds: ProfileThresholds,
    without: Sequence[float] = (),
) -> tuple[GateResult, ...]:
    """G4's verdicts: one per curve (pass; retry with the neighbours' median as the picking's
    guide when the curve is off agreeing neighbours and fits no side; a change shared with one
    side is kept), then one for the line (coverage), unit "line". `without` lists the xmids
    without a curve, for the coverage."""
    ordered = sorted(curves, key=lambda curve: curve.xmid)
    per_side = max(1, thresholds.neighbours // 2)
    found = neighbourhoods(
        ordered, thresholds.neighbours, thresholds.max_misfit, thresholds.min_shared_points
    )
    results: list[GateResult] = []
    for curve, near in zip(ordered, found, strict=True):
        worst = near.worst
        metrics = [
            Metric(
                name="misfit",
                value=None if np.isnan(worst) else round(worst, 3),
                threshold=thresholds.max_misfit,
                bound="max",
                passed=near.standing != "outlier",
            ),
            Metric(
                name="sides_compared",
                value=len(near.compared),
                threshold=1,
                bound="min",
                passed=bool(near.compared),
            ),
        ]
        flags: list[Flag] = []
        if near.standing == "outlier":
            grid, median = median_series([other for side in near.off for other in side.series])
            known = ~np.isnan(median)
            wavelengths, velocities = grid[known], median[known]
            # Thinned: the picker interpolates between the guide's points, and the summary the
            # agent reads shows them all.
            chosen = np.unique(
                np.linspace(0, wavelengths.size - 1, min(wavelengths.size, GUIDE_POINTS))
                .round()
                .astype(int)
            )
            guide = sorted(
                {
                    (
                        round(float(velocities[i] / wavelengths[i]), 2),
                        round(float(velocities[i]), 1),
                    )
                    for i in chosen
                }
            )
            flags.append(
                Flag(
                    name="outlier",
                    message=f"The curve is {worst:.0%} off its neighbours, which agree with each "
                    f"other, {near.where}: pick it again along their median curve.",
                    stage="picking",
                    action=Override(stage="picking", overrides={"guide": guide}),
                )
            )
        elif near.standing == "shared_change":
            flags.append(
                Flag(
                    name="shared_change",
                    message=f"The curve is {near.off[0].misfit:.0%} off the neighbours on one side "
                    "and fits the other: a change in the ground, kept.",
                    stage="picking",
                    action=Keep(note="a change shared with one side is geology"),
                )
            )
        elif near.standing == "no_neighbours":
            flags.append(
                Flag(
                    name="no_neighbours",
                    message="The curve fits neither side of its neighbours, and neither side "
                    f"holds {per_side} curves that agree with each other over "
                    f"{thresholds.min_shared_points} shared wavelengths: nothing to judge it "
                    "against.",
                    stage="picking",
                    action=Keep(note="no agreeing neighbours to compare with"),
                )
            )
        acted = [flag for flag in flags if not isinstance(flag.action, Keep)]
        results.append(
            GateResult(
                gate=GATE,
                unit=curve.unit,
                verdict="retry" if acted else "pass",
                metrics=tuple(metrics),
                flags=tuple(flags),
                kept=Kept(
                    wavelength_m=(float(curve.x[0]), float(curve.x[-1])),
                    n_points=int(curve.x.size),
                ),
            )
        )
    results.append(_line_result(ordered, without, thresholds))
    return tuple(results)


def _line_result(
    curves: Sequence[Series], without: Sequence[float], thresholds: ProfileThresholds
) -> GateResult:
    """Coverage: the xmids without a curve, and how comparable the depths of investigation are
    along the line (the spread of the curves' longest wavelengths); and where velocity falls
    with wavelength along the line (said once, kept)."""
    longest = [float(curve.x[-1]) for curve in curves]
    depth_spread = spread(longest)
    inverse = [curve.xmid for curve in curves if _falls(curve)]
    metrics = [
        Metric(name="curves", value=len(curves), threshold=1, bound="min", passed=len(curves) >= 1),
        Metric(name="without_curve", value=len(without), passed=True),
        Metric(
            name="depth_spread",
            value=round(depth_spread, 3),
            threshold=thresholds.max_depth_spread,
            bound="max",
            passed=depth_spread <= thresholds.max_depth_spread,
        ),
        Metric(name="inverse_curves", value=len(inverse), passed=True),
    ]
    step = line_step([*(one.xmid for one in curves), *without])
    flags: list[Flag] = []
    if without:
        flags.append(
            Flag(
                name="gaps",
                message=f"No curve at {stretches(without, step)}: the section has gaps there.",
                stage="picking",
                action=Keep(note="the gaps stay in the report"),
            )
        )
    if depth_spread > thresholds.max_depth_spread:
        flags.append(
            Flag(
                name="uneven_depth",
                message=f"The longest wavelengths vary by {depth_spread:.0%} along the line: the "
                "depths "
                "of investigation are not comparable.",
                stage="picking",
                action=Keep(note="the models will not reach the same depth everywhere"),
            )
        )
    if len(inverse) >= 2:
        flags.append(
            Flag(
                name="inverse_trend",
                message=f"Velocity falls with wavelength on {len(inverse)} of {len(curves)} "
                f"curves, at {stretches(inverse, step)}: a stiff layer over a softer one there, "
                "if the picks are right.",
                stage="picking",
                action=Keep(note="an inverse trend can be geology"),
            )
        )
    return GateResult(
        gate=GATE,
        unit=LINE,
        verdict="pass" if curves else "reject",
        metrics=tuple(metrics),
        flags=tuple(flags),
        kept=Kept(n_points=len(longest)),
    )


def _falls(curve: Series) -> bool:
    """Whether velocity falls with wavelength along the curve (a negative rank correlation, as
    G3's trend)."""
    if curve.x.size < 3 or np.ptp(curve.x) == 0 or np.ptp(curve.values) == 0:
        return False
    return float(np.corrcoef(rankdata(curve.x), rankdata(curve.values))[0, 1]) < 0
