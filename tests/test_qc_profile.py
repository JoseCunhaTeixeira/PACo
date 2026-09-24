"""G4 on synthetic lines: curves rising with wavelength along the xmids, with one outlier, a
step in the ground, a run of changed windows, and gaps."""

import numpy as np

from paco.qc.g4_profile import LINE, ProfileThresholds, judge_profile
from paco.qc.models import GateResult
from paco.qc.sides import Series

THRESHOLDS = ProfileThresholds()
WAVELENGTHS = np.linspace(3.0, 30.0, 20)


def _curve(xmid: float, scale: float = 1.0, longest: float = 30.0) -> Series:
    wavelengths = WAVELENGTHS[longest >= WAVELENGTHS]
    return Series(
        unit=f"xmid_{xmid:.2f}",
        xmid=xmid,
        x=wavelengths,
        values=scale * (150.0 + 8.0 * wavelengths),
    )


def _line(scales: dict[int, float] | None = None, n: int = 8) -> list[Series]:
    scales = scales or {}
    return [_curve(float(xmid), scales.get(xmid, 1.0)) for xmid in range(n)]


def _by_unit(results: tuple[GateResult, ...]) -> dict[str, GateResult]:
    return {result.unit: result for result in results}


def test_a_smooth_line_passes_every_curve_and_the_line() -> None:
    results = _by_unit(judge_profile(_line(), THRESHOLDS))

    assert len(results) == 9
    assert all(result.verdict == "pass" and result.flags == () for result in results.values())
    line = results[LINE]
    assert {metric.name: metric.value for metric in line.metrics} == {
        "curves": 8,
        "without_curve": 0,
        "depth_spread": 0.0,
    }
    window = results["xmid_3.00"]
    assert {metric.name: metric.value for metric in window.metrics} == {
        "misfit": 0.0,
        "sides_compared": 2,
    }
    assert window.kept.wavelength_m == (3.0, 30.0) and window.kept.n_points == 20
    # The first curve has one side only, and still passes.
    assert next(m.value for m in results["xmid_0.00"].metrics if m.name == "sides_compared") == 1


def test_an_isolated_outlier_is_picked_again_along_the_neighbours_median() -> None:
    results = _by_unit(judge_profile(_line({4: 1.3}), THRESHOLDS))

    outlier = results["xmid_4.00"]
    assert outlier.verdict == "retry"
    (flag,) = outlier.flags
    assert flag.name == "outlier" and flag.stage == "picking"
    assert "both sides" in flag.message
    action = flag.action.model_dump()
    assert action["kind"] == "override" and action["stage"] == "picking"
    guide = np.array(action["overrides"]["guide"])
    # The guide is the neighbours' median: (frequency, velocity) pairs along 150 + 8 wavelength.
    assert guide[:, 0].tolist() == sorted(guide[:, 0])
    wavelengths = guide[:, 1] / guide[:, 0]
    assert np.allclose(guide[:, 1], 150 + 8 * wavelengths, rtol=0.01)
    assert wavelengths.min() >= 3.0 and wavelengths.max() <= 30.0
    # Its neighbours are not off: one side each holds the outlier, and does not agree.
    assert all(results[f"xmid_{x}.00"].verdict == "pass" for x in (2, 3, 5, 6))
    assert results["xmid_3.00"].flags == ()


def test_an_outlier_at_the_end_of_the_line_is_off_its_only_side() -> None:
    results = _by_unit(judge_profile(_line({0: 0.7}), THRESHOLDS))

    (flag,) = results["xmid_0.00"].flags
    assert flag.name == "outlier" and "its only side" in flag.message


def test_a_step_in_the_ground_is_kept_on_both_edges() -> None:
    results = _by_unit(judge_profile(_line({4: 1.5, 5: 1.5, 6: 1.5, 7: 1.5}), THRESHOLDS))

    assert all(result.verdict == "pass" for result in results.values())
    for xmid in ("xmid_3.00", "xmid_4.00"):
        (flag,) = results[xmid].flags
        assert flag.name == "shared_change"
        assert flag.action.model_dump() == {
            "kind": "keep",
            "note": "a change shared with one side is geology",
        }
    assert results["xmid_2.00"].flags == () and results["xmid_5.00"].flags == ()
    # The line's ends have one full side each; the second and second to last, one full side too.
    assert results["xmid_0.00"].flags == () and results["xmid_7.00"].flags == ()


def test_a_run_of_three_changed_windows_has_no_outlier() -> None:
    results = _by_unit(judge_profile(_line({3: 1.5, 4: 1.5, 5: 1.5}), THRESHOLDS))

    assert all(result.verdict == "pass" for result in results.values())
    for xmid in ("xmid_2.00", "xmid_3.00", "xmid_5.00", "xmid_6.00"):
        assert {flag.name for flag in results[xmid].flags} == {"shared_change"}
    # The middle one: the neighbours on each side disagree with each other.
    (flag,) = results["xmid_4.00"].flags
    assert flag.name == "no_neighbours" and flag.action.model_dump()["kind"] == "keep"
    assert results["xmid_0.00"].flags == () and results["xmid_1.00"].flags == ()


def test_curves_sharing_too_few_wavelengths_are_not_compared() -> None:
    curves = [_curve(0.0), _curve(1.0), _curve(2.0, longest=5.0), _curve(3.0), _curve(4.0)]
    results = _by_unit(judge_profile(curves, THRESHOLDS))

    # The short curve has 2 points, below the 3 to share: nothing to compare it with.
    (flag,) = results["xmid_2.00"].flags
    assert flag.name == "no_neighbours"
    assert results["xmid_2.00"].verdict == "pass"
    assert next(m.value for m in results["xmid_2.00"].metrics if m.name == "sides_compared") == 0


def test_a_side_of_one_curve_is_no_evidence() -> None:
    # Three curves: every side holds one curve, which agrees with itself; nothing is compared.
    results = _by_unit(judge_profile(_line({1: 1.3}, n=3), THRESHOLDS))

    assert all(result.verdict == "pass" for result in results.values())
    assert {flag.name for flag in results["xmid_1.00"].flags} == {"no_neighbours"}
    # With two on each side, the same curve is an outlier.
    results = _by_unit(judge_profile(_line({2: 1.3}, n=5), THRESHOLDS))
    assert {flag.name for flag in results["xmid_2.00"].flags} == {"outlier"}


def test_the_line_reports_the_gaps_and_uneven_depths() -> None:
    curves = [_curve(float(x), longest=30.0 if x < 3 else 6.0) for x in range(6)]
    results = _by_unit(judge_profile(curves, THRESHOLDS, without=[6.0, 7.0, 10.0]))

    line = results[LINE]
    assert line.verdict == "pass"
    assert {flag.name for flag in line.flags} == {"gaps", "uneven_depth"}
    assert "xmid 6.00-7.00 (2), 10.00 (1)" in next(
        f.message for f in line.flags if f.name == "gaps"
    )
    assert next(m.value for m in line.metrics if m.name == "without_curve") == 3


def test_a_line_without_a_curve_is_rejected() -> None:
    (line,) = judge_profile([], THRESHOLDS, without=[1.0, 2.0])

    assert line.unit == LINE and line.verdict == "reject"
    assert {flag.name for flag in line.flags} == {"gaps"}
