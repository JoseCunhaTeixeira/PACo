"""G8, the petrophysical profile QC over the whole line (the user's decision of 2026-09-26: range,
fit, line): the Vs of each window's rock physics at fixed depths, and its water table, against
its neighbours on either side, the sides of G4 and G6, among the windows within a few of the
line's steps: the range check leaves gaps, and models far apart are not neighbours. A window off
neighbours that agree while its curve fits theirs (G4) is left out of the sections: a model
predicts one soil column per curve, there is nothing to sample again. A change the curves show
too is kept."""

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.masw.quality.line import Neighbourhood, Series, neighbourhoods

from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Reject
from paco.qc.report import line_step, stretches

GATE = "G8"
LINE = "line"  # the unit of the line-level result, as G4's and G6's


class PetroLineThresholds(BaseModel):
    """G8's limits: G6's for Vs; provisional (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    neighbours: int = Field(
        default=4, ge=2, description="Windows compared with: half of them on each side."
    )
    max_misfit: float = Field(
        default=0.15,
        gt=0,
        description="Median relative difference of the rock physics' Vs to a side's median, over "
        "the depths they share, at most (G6's limit).",
    )
    max_water_table_jump: float = Field(
        default=1.0,
        gt=0,
        description="Difference of the water table's depth to a side's median, at most (m): one "
        "of the model's 1 m steps. Relative, 0.5 m against 1.5 m was 67 % (active_p2, "
        "2026-09-26).",
    )
    max_gap_steps: float = Field(
        default=3.0,
        gt=0,
        description="Neighbours are the windows within this many of the line's steps: on "
        "active_p2, whose curves the model mostly does not cover, windows 75 m apart were "
        "compared (2026-09-26).",
    )
    min_shared_depths: int = Field(
        default=2, ge=1, description="Depths shared with a side's median, to compare at all."
    )


def judge_petro_line(
    profiles: Sequence[Series],
    water_tables: Mapping[str, float],
    curves: Mapping[str, GateResult],
    thresholds: PetroLineThresholds,
    without: Sequence[float] = (),
) -> tuple[GateResult, ...]:
    """G8's verdicts: one per window (its rock physics' Vs by depth, down to its curve's depth
    of investigation, and its water table), then one for the line, unit "line". `curves` holds
    G4's result on each window's curve, `without` the xmids without a model G7 passed."""
    ordered = sorted(profiles, key=lambda profile: profile.xmid)
    step = line_step([*(one.xmid for one in ordered), *without])
    reach = thresholds.max_gap_steps * step if step > 0 else None
    by_vs = neighbourhoods(
        ordered,
        thresholds.neighbours,
        thresholds.max_misfit,
        thresholds.min_shared_depths,
        max_distance=reach,
    )
    tables = [
        Series(one.unit, one.xmid, np.array([0.0]), np.array([water_tables[one.unit]]))
        for one in ordered
    ]
    by_table = neighbourhoods(
        tables,
        thresholds.neighbours,
        thresholds.max_water_table_jump,
        1,
        relative=False,
        max_distance=reach,
    )
    results = [
        _window_result(profile, vs, table, curves.get(profile.unit), thresholds)
        for profile, vs, table in zip(ordered, by_vs, by_table, strict=True)
    ]
    results.append(_line_result(ordered, water_tables, without))
    return tuple(results)


def _window_result(
    profile: Series,
    vs: Neighbourhood,
    table: Neighbourhood,
    curve: GateResult | None,
    thresholds: PetroLineThresholds,
) -> GateResult:
    metrics = [
        Metric(
            name="misfit",
            value=None if np.isnan(vs.worst) else round(vs.worst, 3),
            threshold=thresholds.max_misfit,
            bound="max",
            passed=vs.standing != "outlier",
        ),
        Metric(
            name="water_table_jump",
            value=None if np.isnan(table.worst) else round(table.worst, 2),
            threshold=thresholds.max_water_table_jump,
            bound="max",
            passed=table.standing != "outlier",
            unit="m",
        ),
    ]
    shown = {flag.name for flag in curve.flags} if curve is not None else set()
    flags: list[Flag] = []
    for what, near, off in (
        ("rock physics' Vs", vs, f"{vs.worst:.0%}"),
        ("water table", table, f"{table.worst:g} m"),
    ):
        if near.standing == "outlier" and "shared_change" in shown:
            flags.append(
                Flag(
                    name="petro_change",
                    message=f"The {what} is {off} off the neighbours {near.where}, "
                    "and the curves show the change too: kept.",
                    stage="petro_inversion",
                    action=Keep(note="the curves show the same change"),
                )
            )
        elif near.standing == "outlier":
            flags.append(
                Flag(
                    name="petro_outlier",
                    message=f"The {what} is {off} off the neighbours {near.where}, "
                    "while the curves do not show such a change: left out of the sections.",
                    stage="petro_inversion",
                    action=Reject(reason=f"the {what} is off its neighbours"),
                    fixable=False,
                )
            )
    return GateResult(
        gate=GATE,
        unit=profile.unit,
        verdict="reject" if any(isinstance(flag.action, Reject) for flag in flags) else "pass",
        metrics=tuple(metrics),
        flags=tuple(flags),
        kept=Kept(n_points=int(profile.x.size)),
    )


def _line_result(
    profiles: Sequence[Series], water_tables: Mapping[str, float], without: Sequence[float]
) -> GateResult:
    """Coverage: the xmids without a model G7 passed, and the water table's range."""
    depths = [water_tables[one.unit] for one in profiles]
    metrics = [
        Metric(
            name="models", value=len(profiles), threshold=2, bound="min", passed=len(profiles) >= 2
        ),
        Metric(name="without_model", value=len(without), passed=True),
        Metric(name="water_table_min", value=min(depths, default=None), passed=True, unit="m"),
        Metric(name="water_table_max", value=max(depths, default=None), passed=True, unit="m"),
    ]
    flags: list[Flag] = []
    if without:
        step = line_step([*(one.xmid for one in profiles), *without])
        flags.append(
            Flag(
                name="gaps",
                message=f"No petrophysical model at {stretches(without, step)}: the sections "
                "have gaps there.",
                stage="petro_inversion",
                action=Keep(note="the gaps stay in the report"),
            )
        )
    return GateResult(
        gate=GATE,
        unit=LINE,
        # A section needs two windows.
        verdict="pass" if len(profiles) >= 2 else "reject",
        metrics=tuple(metrics),
        flags=tuple(flags),
        kept=Kept(n_points=len(profiles)),
    )
