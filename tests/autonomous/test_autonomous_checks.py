"""The checks of a window's picks (docs/autonomous_plan.md §4.6), on active_test_profile's curves, and on cases with
a known answer its curves do not hold."""

import pytest
from autonomous_judged import JudgedLine

from paco.autonomous.sensors.checks import Run, failing_runs, line_checks, window_checks
from paco.qc.models import GateResult, Kept, Metric

# active_test_profile's four windows: 24 receivers 1.5 m apart, 34.5 m long. By xmid: the shortest and
# longest wavelengths (m), the share of points beyond 3 window lengths, the share in the aliasing
# zone, the nearest shot's distance over half the longest wavelength, the median relative
# uncertainty, and whether the nearest shot is in the near field (each window has one 0.75 m off
# an end). The window at 53.25 m holds one point, under twice the receiver spacing.
LINE = {
    17.25: (11.0, 15.0, 0.0, 0.0, 0.75 / 7.5, 0.079, True),
    53.25: (1.72603, 1.72603, 0.0, 1.0, 0.75 / 0.86, 0.119, True),  # half its wavelength, rounded
    89.25: (8.0, 16.0, 0.0, 0.0, 0.75 / 8, 0.073, True),
    125.25: (9.0, 13.0, 0.0, 0.0, 0.75 / 6.5, 0.067, True),
}


@pytest.mark.slow
def test_each_curves_checks_on_the_line(judged_line: JudgedLine) -> None:
    assert judged_line.length_m == 34.5
    for xmid, (shortest, longest, beyond, aliased, nearest, uncertainty, near) in LINE.items():
        check = window_checks(judged_line.g3[xmid], judged_line.length_m)

        assert check is not None
        assert (check.shortest_m, check.longest_m) == pytest.approx((shortest, longest), abs=1e-5)
        assert check.reach == pytest.approx(longest / 34.5, abs=1e-5)
        assert check.depth_m == pytest.approx(longest / 2, abs=1e-5)
        assert check.span == pytest.approx(longest / shortest, abs=1e-5)
        assert (check.beyond, check.aliased, check.uncertainty) == (beyond, aliased, uncertainty)
        assert check.nearest_shot == pytest.approx(nearest)
        assert check.in_near_field == near


@pytest.mark.slow
def test_the_lines_checks(judged_line: JudgedLine) -> None:
    checks = [window_checks(judged_line.g3[xmid], judged_line.length_m) for xmid in LINE]

    line = line_checks([check for check in checks if check is not None], target_depth_m=7.0)

    assert line.curves == 4
    # Medians of four: halfway between the two middle ones.
    assert line.reach == pytest.approx((13 + 15) / 2 / 34.5, abs=1e-5)
    assert (line.beyond_windows, line.beyond) == (0, 0.0)
    assert (line.aliased_windows, line.aliased) == (1, 0.0)
    assert line.near_field_windows == 4
    assert line.depth_m == pytest.approx((6.5 + 7.5) / 2, abs=1e-5)
    # Two curves inform 7 m: half of 15 and 16 m.
    assert line.reaching == 2
    assert line.uncertainty == pytest.approx((0.073 + 0.079) / 2)


def test_points_in_the_aliasing_zone_and_a_far_nearest_shot() -> None:
    # A curve partly in the aliasing zone, its nearest shot far.
    result = _g3("xmid_6.00", (0.4, 12.0), aliased=0.25, near=(9.0, 6.0))

    check = window_checks(result, 10.0)

    assert check is not None
    assert (check.aliased, check.reach, check.depth_m, check.nearest_shot) == (0.25, 1.2, 6.0, 1.5)
    assert not check.in_near_field
    line = line_checks([check])
    assert (line.aliased_windows, line.aliased, line.near_field_windows) == (1, 0.25, 0)
    assert line.reaching is None


def test_what_g3_did_not_measure_stays_unknown() -> None:
    # No shot distance given to G3, no uncertainty measured; no curve at all.
    check = window_checks(_g3("xmid_6.00", (2.0, 8.0), uncertainty=None), 10.0)

    assert check is not None
    assert (check.nearest_shot, check.uncertainty, check.in_near_field) == (None, None, False)
    assert line_checks([check]).uncertainty is None
    assert window_checks(_g3("xmid_6.00", None), 10.0) is None


def test_the_runs_of_windows_that_do_not_pass() -> None:
    failing = {1.0, 2.0, 5.0, 6.0, 7.0, 10.0}
    passing = {float(xmid): float(xmid) not in failing for xmid in range(1, 11)}

    runs = failing_runs(passing, near_only={1.0, 10.0})

    assert runs == (
        Run(xmids=(1.0, 2.0), at_end=True, near_only=(1.0,)),
        Run(xmids=(5.0, 6.0, 7.0), at_end=False, near_only=()),
        Run(xmids=(10.0,), at_end=True, near_only=(10.0,)),
    )
    assert failing_runs({1.0: True, 2.0: True}) == ()


def _g3(
    unit: str,
    wavelengths: tuple[float, float] | None,
    *,
    aliased: float = 0.0,
    uncertainty: float | None = 0.2,
    near: tuple[float, float] | None = None,
) -> GateResult:
    """A passing G3 result of window `unit`: its curve's shortest and longest `wavelengths` (None:
    no curve), its share of points in the aliasing zone, its uncertainty, and `near`, its nearest
    shot's distance and half its longest wavelength (None: no shot distance given)."""
    metrics = [
        Metric(name="aliased_points", value=aliased, threshold=0.0, passed=True),
        Metric(name="beyond_reach_points", value=0.0, threshold=0.0, passed=True),
        Metric(name="uncertainty", value=uncertainty, passed=True),
    ]
    if near is not None:
        metrics.append(
            Metric(
                name="near_offset",
                value=near[0],
                threshold=near[1],
                bound="min",
                passed=near[0] >= near[1],
            )
        )
    return GateResult(
        gate="G3",
        unit=unit,
        verdict="pass",
        metrics=tuple(metrics),
        kept=Kept(wavelength_m=wavelengths),
    )
