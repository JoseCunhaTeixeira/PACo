"""G3 on the analytic images of test_quality.py: the quality metrics, said in the gates' language,
and the curve's own rules (wavelengths, jumps, the air wave, the trend, points, uncertainties)
on curves given with the pick."""

import math
from dataclasses import replace

import numpy as np
from sigpipe.algorithms.picking.dispersion.tracking import PickedMode, PickingParameters, pick_modes
from sigpipe.base import (
    Coordinate,
    DispersionCurve,
    DispersionImage,
    LinearAcquisition,
    Mode,
    VelocityType,
)

from paco.qc.g3_curve import CurveThresholds, judge_curve
from paco.qc.models import Flag, GateResult

FREQUENCIES = np.arange(10.0, 40.5, 0.5)
VELOCITIES = np.arange(1.0, 1000.5, 0.5)
M0 = 150 + 250 * np.exp(-FREQUENCIES / 15)
CONSTANT_WAVELENGTH = 6.0 * FREQUENCIES  # a 6 m wavelength at every frequency
WINDOW_LENGTH = 47.0
ACQUISITION = LinearAcquisition(
    source=Coordinate(0.0, 0.0, 0.0),
    receivers=tuple(Coordinate(2.0 + k, 0.0, 0.0) for k in range(48)),
)
NOISE_FLOOR = 1 / math.sqrt(48)
THRESHOLDS = CurveThresholds()
QUALITY_METRICS = {"sharpness", "prominence", "on_data", "constant_wavelength", "n_points"}
CURVE_METRICS = {
    "aliased_points",
    "curve_points",
    "max_jump",
    "air_wave_share",
    "trend",
    "uncertainty",
}


def _ridge(velocities: np.ndarray, height: float, width: float) -> np.ndarray:
    f, c = FREQUENCIES[:, None], velocities[:, None]
    sigma = width * c * (c / f) / WINDOW_LENGTH / (2 * math.sqrt(2 * math.log(2)))
    return height * np.exp(-0.5 * ((VELOCITIES - c) / sigma) ** 2)


def _image(*ridges: np.ndarray, background: float = 0.0) -> DispersionImage:
    flat = np.full((FREQUENCIES.size, VELOCITIES.size), background)
    return DispersionImage(
        fv_map=NOISE_FLOOR + np.maximum.reduce([flat, *ridges]),
        fs=FREQUENCIES,
        vs=VELOCITIES,
        type=VelocityType.PHASE,
        acquisition=ACQUISITION,
    )


def _pick(image: DispersionImage, velocities: np.ndarray) -> PickedMode:
    """An M0 pick at `velocities`, on the image's grid, every point kept (as test_quality.py)."""
    cells = np.abs(VELOCITIES[None, :] - velocities[:, None]).argmin(axis=1)
    n = FREQUENCIES.size
    return PickedMode(
        number=0,
        frequencies=FREQUENCIES,
        velocities=VELOCITIES[cells],
        coherence=image.fv_map[np.arange(n), cells].astype(float),
        pinned=np.zeros(n, dtype=bool),
        kept=np.ones(n, dtype=bool),
        noise_floor=NOISE_FLOOR,
        curve=None,
    )


def _picked(
    image: DispersionImage,
    velocities: np.ndarray,
    fs: np.ndarray | None = None,
    vs: np.ndarray | None = None,
    vs_err: np.ndarray | None = None,
) -> PickedMode:
    """The pick at `velocities`, with the curve saved from it: the pick's own points, or the
    (`fs`, `vs`) given."""
    mode = _pick(image, velocities)
    fs = FREQUENCIES if fs is None else fs
    curve = DispersionCurve(
        fs=fs,
        vs=mode.velocities if vs is None else vs,
        mode=Mode("M", 0),
        acquisition=ACQUISITION,
        vs_err=vs_err,
        type=VelocityType.PHASE,
    )
    return replace(mode, curve=curve)


def _judge(image: DispersionImage, band: tuple[float, float] | None = None):  # noqa: ANN202
    """G3 on the real picker's M0."""
    modes = pick_modes(image, PickingParameters())
    return judge_curve("xmid_12.50", image, modes[0] if modes else None, THRESHOLDS, band)


def _flags(result: GateResult) -> dict[str, Flag]:
    return {flag.name: flag for flag in result.flags}


# ---------------------------------------------------------------- the quality metrics


def test_a_clean_ridge_passes_with_what_its_curve_keeps() -> None:
    result = _judge(_image(_ridge(M0, 0.8, 1.5)))

    assert result.verdict == "pass"
    assert result.flags == ()
    assert {metric.name for metric in result.metrics} == QUALITY_METRICS | CURVE_METRICS
    assert all(metric.passed for metric in result.metrics)
    # The curve that goes to the inversion: resampled in wavelength, fewer than the tracked points.
    assert result.kept.n_points is not None and 15 < result.kept.n_points < 40
    assert result.kept.band_hz is not None
    assert result.kept.wavelength_m is not None
    shortest, longest = result.kept.wavelength_m
    assert 3 < shortest < 6 and 25 < longest < 45  # v / f from about 165 / 40 to 400 / 10
    assert result.kept.n_traces == 48
    # The picker's Lorentzian uncertainties, for a 47 m window: well within what the inversion uses.
    uncertainty = next(metric for metric in result.metrics if metric.name == "uncertainty")
    assert uncertainty.value is not None and 0.05 < uncertainty.value < 0.5


def test_peaks_too_narrow_for_the_window_reject_the_window() -> None:
    result = _judge(_image(_ridge(M0, 0.8, 0.3)))

    assert result.verdict == "reject"
    (flag,) = result.flags
    assert flag.name == "sharpness" and not flag.fixable
    assert flag.action.model_dump() == {
        "kind": "reject",
        "reason": "peaks narrower than the window can resolve",
    }


def test_a_ridge_barely_above_the_image_asks_for_a_filter_on_the_band() -> None:
    image = _image(_ridge(M0, 0.8, 1.5), background=0.6)
    result = judge_curve("xmid_12.50", image, _pick(image, M0), THRESHOLDS, (12.0, 38.0))

    assert result.verdict == "retry"
    (flag,) = result.flags
    assert flag.name == "prominence" and flag.stage == "preprocessing"
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "preprocessing",
        "overrides": {"filtering": {"method": "iir", "fmin": 12.0, "fmax": 38.0}},
    }
    # No curve with the pick: only the quality metrics.
    assert {metric.name for metric in result.metrics} == QUALITY_METRICS


def test_a_brighter_ridge_above_asks_to_narrow_the_band_to_the_pick() -> None:
    image = _image(_ridge(M0, 0.6, 1.5), _ridge(1.8 * M0, 0.9, 1.5))
    result = judge_curve("xmid_12.50", image, _pick(image, M0), THRESHOLDS)

    assert result.verdict == "retry"
    (flag,) = result.flags
    assert flag.name == "on_data" and flag.stage == "phase_shift"
    # No band given: the pick's own.
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "phase_shift",
        "overrides": {"dispersion": {"fmin": 10.0, "fmax": 40.0}},
    }


def test_a_pick_following_the_resolution_edge_asks_to_cut_long_wavelengths() -> None:
    image = _image(_ridge(CONSTANT_WAVELENGTH, 0.8, 1.5))
    result = judge_curve("xmid_12.50", image, _pick(image, CONSTANT_WAVELENGTH), THRESHOLDS)

    assert result.verdict == "retry"
    (flag,) = result.flags
    # Cut below where the stretch starts: 6 m, 0.12 of the 47 m window.
    assert "0.12 window lengths" in flag.message
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "picking",
        "overrides": {"max_wavelength": 0.12},
    }


def test_no_pick_asks_for_a_looser_mode_rule() -> None:
    result = judge_curve("xmid_12.50", _image(), None, THRESHOLDS)

    assert result.verdict == "retry"
    (flag,) = result.flags
    assert flag.name == "no_ridge"
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "picking",
        "overrides": {"mode_min_ratio": 1.2},
    }
    assert result.kept.n_points == 0
    assert result.kept.wavelength_m is None


# ---------------------------------------------------------------- the curve's own rules


def test_a_curve_given_with_the_pick_passes_the_curve_rules() -> None:
    image = _image(_ridge(M0, 0.8, 1.5))
    result = judge_curve("xmid_12.50", image, _picked(image, M0), THRESHOLDS)

    assert result.verdict == "pass" and result.flags == ()
    values = {metric.name: metric.value for metric in result.metrics}
    assert values["aliased_points"] == 0.0
    assert values["curve_points"] == 61
    jump = values["max_jump"]
    assert jump is not None and jump < 0.1
    assert values["air_wave_share"] == 0.0
    assert values["trend"] == 1.0  # velocity rises with wavelength, every point
    assert values["uncertainty"] is None  # none given with the curve
    assert result.kept.wavelength_m is not None
    assert math.isclose(result.kept.wavelength_m[1], 27.85, rel_tol=0.01)  # M0(10 Hz) / 10 Hz


def test_a_jump_between_consecutive_points_means_another_mode() -> None:
    image = _image(_ridge(M0, 0.8, 1.5))
    vs = M0.copy()
    vs[FREQUENCIES >= 30.5] *= 1.6
    result = judge_curve(
        "xmid_12.50", image, _picked(image, M0, vs=vs), THRESHOLDS, picking=PickingParameters()
    )

    assert result.verdict == "retry"
    flags = _flags(result)
    assert "mode_jump" in flags
    # First the band is cut where the jump sits (30 to 30.5 Hz), the side with fewer points
    # (20 above against 41 below) going.
    assert flags["mode_jump"].action.model_dump() == {
        "kind": "override",
        "stage": "picking",
        "overrides": {"fmax": 30.2},
    }
    jump = next(metric for metric in result.metrics if metric.name == "max_jump")
    assert jump.value is not None and jump.value > 0.3
    # With the band already cut, the corridor is halved.
    cut = PickingParameters(fmax=30.2)
    again = judge_curve("xmid_12.50", image, _picked(image, M0, vs=vs), THRESHOLDS, picking=cut)
    assert _flags(again)["mode_jump"].action.model_dump() == {
        "kind": "override",
        "stage": "picking",
        "overrides": {"corridor": 0.1},
    }


def test_points_at_the_air_waves_speed_ask_for_a_mute() -> None:
    image = _image(_ridge(M0, 0.8, 1.5))
    air = 345.0 - 15.0 * (FREQUENCIES - 10.0) / 30.0  # 345 at 10 Hz to 330 at 40 Hz
    result = judge_curve("xmid_12.50", image, _picked(image, M0, vs=air), THRESHOLDS)

    assert result.verdict == "retry"
    (flag,) = result.flags
    assert flag.name == "air_wave" and flag.stage == "preprocessing"
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "preprocessing",
        "overrides": {"muting": {"method": "mute", "vmin": 80.0, "vmax": 320.0}},
    }


def test_velocity_falling_with_wavelength_is_flagged_and_kept() -> None:
    image = _image(_ridge(M0, 0.8, 1.5))
    result = judge_curve("xmid_12.50", image, _picked(image, M0, vs=600.0 - M0), THRESHOLDS)

    assert result.verdict == "pass"  # a kept flag does not change the verdict
    (flag,) = result.flags
    assert flag.name == "inverse_dispersion"
    assert flag.action.model_dump() == {"kind": "keep", "note": "an inverse trend can be geology"}
    trend = next(metric for metric in result.metrics if metric.name == "trend")
    assert trend.value == -1.0 and not trend.passed


def test_too_few_points_ask_to_keep_more_of_the_ridge() -> None:
    image = _image(_ridge(M0, 0.8, 1.5))
    fs = np.array([30.0, 31.0, 32.0])
    vs = 150 + 250 * np.exp(-fs / 15)
    result = judge_curve(
        "xmid_12.50",
        image,
        _picked(image, M0, fs=fs, vs=vs),
        THRESHOLDS,
        picking=PickingParameters(min_relative_coherence=0.5),
    )

    assert result.verdict == "retry"
    (flag,) = result.flags
    assert flag.name == "too_few_points"
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "picking",
        "overrides": {"min_relative_coherence": 0.3},
    }
    assert result.kept.n_points == 3


def test_uncertainties_beyond_what_the_inversion_can_use_reject() -> None:
    image = _image(_ridge(M0, 0.8, 1.5))
    picked = _picked(image, M0, vs_err=0.6 * M0)
    result = judge_curve("xmid_12.50", image, picked, THRESHOLDS)

    assert result.verdict == "reject"
    (flag,) = result.flags
    assert flag.name == "uncertain" and not flag.fixable and flag.stage == "phase_shift"
    assert flag.action.model_dump()["kind"] == "reject"
    uncertainty = next(metric for metric in result.metrics if metric.name == "uncertainty")
    assert uncertainty.value is not None and math.isclose(uncertainty.value, 0.6, abs_tol=0.01)


def test_a_shot_closer_than_half_the_longest_wavelength_is_reported() -> None:
    image = _image(_ridge(M0, 0.8, 1.5))
    picked = _picked(image, M0)  # wavelengths up to 27.85 m: the near field ends at 13.9 m

    near = judge_curve("xmid_12.50", image, picked, THRESHOLDS, nearest_offset=0.75)
    assert near.verdict == "pass"
    (flag,) = near.flags
    assert flag.name == "near_field" and flag.stage == "phase_shift"
    assert flag.message.startswith("The nearest shot is 0.75 m from the window, under half the")
    assert flag.action.model_dump()["kind"] == "keep"
    metric = next(metric for metric in near.metrics if metric.name == "near_offset")
    assert (metric.value, metric.threshold, metric.bound, metric.passed) == (
        0.75,
        13.93,
        "min",
        False,
    )
    # A shot far enough, or no geometry at all: nothing to say.
    far = judge_curve("xmid_12.50", image, picked, THRESHOLDS, nearest_offset=20.0)
    assert far.flags == () and "near_offset" in {metric.name for metric in far.metrics}
    unknown = judge_curve("xmid_12.50", image, picked, THRESHOLDS)
    assert "near_offset" not in {metric.name for metric in unknown.metrics}


def test_the_thresholds_are_the_configuration_of_the_gate() -> None:
    thresholds = CurveThresholds(min_points=10, max_jump=0.5)

    assert thresholds.metrics.min_sharpness == 0.8
    assert thresholds.air_wave_band == (330.0, 345.0)
    assert CurveThresholds.model_validate_json(thresholds.model_dump_json()) == thresholds
