"""G6, the model profile QC over the whole line (docs/qc_workflow.md): the monitored model's Vs
(the ensemble's) at fixed depths down to its depth informed (G5's), each model against its
neighbours on either side within a few of the line's steps (the sides of G4). A jump the curves
do not show (G4 found the window's curve fits its neighbours) is non-uniqueness, kept with its
flag: G5 passed the model converged, and sampling longer draws the same posterior again (the
user, 2026-09-29). A jump the curves show too is kept. And how evenly the depth informed runs
along the line. No lateral smoothing: neither models edited, nor neighbours used as priors."""

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.masw.quality.line import Series, neighbourhoods, spread

from paco.qc.models import Flag, GateResult, Keep, Kept, Metric
from paco.qc.report import line_step, stretches

GATE = "G6"
LINE = "line"  # the unit of the line-level result, as G4's


class ModelProfileThresholds(BaseModel):
    """G6's limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    neighbours: int = Field(
        default=4, ge=2, description="Models compared with: half of them on each side of the xmid."
    )
    max_misfit: float = Field(
        default=0.15,
        gt=0,
        description="Median relative difference of Vs to a side's median model, over the depths "
        "they share, at most: beyond it the model is off that side. The side's own models must "
        "agree within half of it to count.",
    )
    min_shared_depths: int = Field(
        default=2, ge=1, description="Depths shared with a side's median model, to compare at all."
    )
    max_depth_informed_spread: float = Field(
        default=0.5,
        gt=0,
        description="MAD over median of the models' depths informed (where their Vs spread U "
        "stays under its limit, G5's) along the line, at most: beyond it the models do not see "
        "equally deep.",
    )
    max_gap_steps: float = Field(
        default=3.0,
        gt=0,
        description="Neighbours are the windows within this many of the line's steps: windows "
        "far apart, across a gap, are not neighbours (as G8's).",
    )


def judge_model_profile(
    models: Sequence[Series],
    curves: Mapping[str, GateResult],
    depths_informed: Mapping[str, float | None],
    thresholds: ModelProfileThresholds,
    without: Sequence[float] = (),
) -> tuple[GateResult, ...]:
    """G6's verdicts: one per model (the ensemble's Vs by depth, down to its depth informed:
    below it the priors speak, not the data), then one for the line, unit "line". `curves`
    holds G4's result on each window's curve, `depths_informed` each window's (None: the whole
    model), `without` the xmids without a model. Neighbours are the models within
    `max_gap_steps` of the line's steps."""
    ordered = sorted(models, key=lambda model: model.xmid)
    step = line_step([*(one.xmid for one in ordered), *without])
    found = neighbourhoods(
        ordered,
        thresholds.neighbours,
        thresholds.max_misfit,
        thresholds.min_shared_depths,
        max_distance=thresholds.max_gap_steps * step if step > 0 else None,
    )
    per_side = max(1, thresholds.neighbours // 2)
    results: list[GateResult] = []
    for model, near in zip(ordered, found, strict=True):
        worst = near.worst
        metrics = [
            Metric(
                name="neighbour_misfit",
                value=None if np.isnan(worst) else round(worst, 3),
                threshold=thresholds.max_misfit,
                bound="max",
                passed=near.standing != "outlier",
                of="neighbours",
                over=f"{per_side} models a side at most, within {thresholds.max_gap_steps:g} "
                "line steps: Vs down to their depth informed, the median relative difference to "
                "a side's median, the worst side",
            ),
            Metric(
                name="sides_compared",
                value=len(near.compared),
                threshold=1,
                bound="min",
                passed=bool(near.compared),
                of="neighbours",
                over=f"its {len(near.sides)} sides: those whose models agree with each other",
            ),
        ]
        flags: list[Flag] = []
        if near.standing == "outlier":
            curve = curves.get(model.unit)
            shown = {flag.name for flag in curve.flags} if curve is not None else set()
            if "shared_change" in shown:
                flags.append(
                    Flag(
                        name="model_change",
                        message=f"The model is {worst:.0%} off its neighbours {near.where}, and "
                        "the curves show the change too: kept.",
                        stage="inversion",
                        action=Keep(note="the curves show the same change"),
                    )
                )
            elif shown:
                flags.append(
                    Flag(
                        name="model_change",
                        message=f"The model is {worst:.0%} off its neighbours {near.where}; its "
                        "curve had no agreeing neighbours to say whether the change is real.",
                        stage="inversion",
                        action=Keep(note="the curves cannot tell"),
                    )
                )
            else:
                # Its chains agree (G5 passed it converged): sampling longer draws the same
                # posterior again (the user, 2026-09-29: no retry that could not change it).
                flags.append(
                    Flag(
                        name="non_unique",
                        message=f"The model is {worst:.0%} off its neighbours {near.where}, while "
                        "its curve fits theirs: another model fits the same data. Kept, to look "
                        "at.",
                        stage="inversion",
                        action=Keep(note="a converged posterior: sampling longer gives it again"),
                    )
                )
        elif near.standing == "shared_change":
            flags.append(
                Flag(
                    name="model_change",
                    message=f"The model is {near.off[0].misfit:.0%} off the neighbours on one side "
                    "and fits the other: a change in the ground, kept.",
                    stage="inversion",
                    action=Keep(note="a change shared with one side is geology"),
                )
            )
        elif near.standing == "no_neighbours":
            flags.append(
                Flag(
                    name="no_neighbours",
                    message="The model fits neither side of its neighbours, and neither side holds "
                    "models that agree with each other: nothing to judge it against.",
                    stage="inversion",
                    action=Keep(note="no agreeing neighbours to compare with"),
                )
            )
        acted = [flag for flag in flags if not isinstance(flag.action, Keep)]
        results.append(
            GateResult(
                gate=GATE,
                unit=model.unit,
                verdict="retry" if acted else "pass",
                metrics=tuple(metrics),
                flags=tuple(flags),
                kept=Kept(n_points=int(model.x.size)),
            )
        )
    results.append(_line_result(ordered, depths_informed, without, thresholds))
    return tuple(results)


def _line_result(
    models: Sequence[Series],
    depths_informed: Mapping[str, float | None],
    without: Sequence[float],
    thresholds: ModelProfileThresholds,
) -> GateResult:
    """Coverage: the xmids without a model, and how evenly the depth informed runs along the
    line (a model informed down to its bottom counts at its deepest compared depth)."""
    depths = [
        depth if (depth := depths_informed.get(model.unit)) is not None else float(model.x[-1])
        for model in models
    ]
    depth_spread = spread(depths)
    metrics = [
        Metric(
            name="models",
            value=len(models),
            threshold=1,
            bound="min",
            passed=bool(models),
            of="line",
            over=f"the line's {len(models) + len(without)} windows: a model G5 passed",
        ),
        Metric(
            name="without_model",
            value=len(without),
            passed=True,
            of="line",
            over=f"the line's {len(models) + len(without)} windows",
        ),
        Metric(
            name="depth_informed_spread",
            value=round(depth_spread, 3),
            threshold=thresholds.max_depth_informed_spread,
            bound="max",
            passed=depth_spread <= thresholds.max_depth_informed_spread,
            of="line",
            over=f"the {len(models)} models' depths informed: MAD over median",
        ),
    ]
    step = line_step([*(one.xmid for one in models), *without])
    flags: list[Flag] = []
    if without:
        flags.append(
            Flag(
                name="gaps",
                message=f"No model at {stretches(without, step)}: the section has gaps there.",
                stage="inversion",
                action=Keep(note="the gaps stay in the report"),
            )
        )
    if depth_spread > thresholds.max_depth_informed_spread:
        flags.append(
            Flag(
                name="uneven_depth_informed",
                message=f"The depth informed varies by {depth_spread:.0%} along the line: the "
                "models do not see equally deep.",
                stage="inversion",
                action=Keep(note="compare the models only down to the shallowest depth informed"),
            )
        )
    return GateResult(
        gate=GATE,
        unit=LINE,
        verdict="pass" if models else "reject",
        metrics=tuple(metrics),
        flags=tuple(flags),
        kept=Kept(n_points=len(models)),
    )
