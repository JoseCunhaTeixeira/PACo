"""From PACo, src/paco/qc/g4_profile.py, the guide `judge_profile` gives an outlier (its neighbours'
median curve, thinned), at commit e5f1a24.

Why: inline in judge_profile, which gives it only to a curve G4 finds off its neighbours. pick
(paco.autonomous.tools.pick) guides any window along its neighbours' median curve (docs/autonomous_plan.md §4.6),
so that a guide reads the same wherever it comes from. Copied as it is, made a function; changed: no
guide without a curve to take it from (judge_profile never asks one: a window guided along its
neighbours may have none)."""

from collections.abc import Sequence

import numpy as np
from sigpipe.masw.quality.line import Series, median_series

from paco.qc.g4_profile import GUIDE_POINTS


def median_guide(series: Sequence[Series]) -> tuple[tuple[float, float], ...]:
    """The median of `series` (curves as velocities by wavelength) as the picker's guide: points
    (frequency in Hz, velocity in m/s), GUIDE_POINTS at most; none where the median has none, or
    without a curve to take it from."""
    if not series:
        return ()
    grid, median = median_series(series)
    known = ~np.isnan(median)
    wavelengths, velocities = grid[known], median[known]
    if not wavelengths.size:
        return ()
    # Thinned: the picker interpolates between the guide's points, and the summary the agent
    # reads shows them all.
    chosen = np.unique(
        np.linspace(0, wavelengths.size - 1, min(wavelengths.size, GUIDE_POINTS))
        .round()
        .astype(int)
    )
    return tuple(
        sorted(
            {
                (
                    round(float(velocities[i] / wavelengths[i]), 2),
                    round(float(velocities[i]), 1),
                )
                for i in chosen
            }
        )
    )
