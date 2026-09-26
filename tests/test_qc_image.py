"""G2 on analytic images with a known answer: Gaussian ridges over the noise floor, as in
test_quality.py, each defect planted alone."""

import math

import numpy as np
from sigpipe.base import Coordinate, DispersionImage, LinearAcquisition, VelocityType
from sigpipe.masw.quality.image import (
    aliased,
    coherent_columns,
    competing_ridges,
    edge_peaks,
    noise_floor,
)

from paco.qc.g2_image import ImageThresholds, judge_image

FREQUENCIES = np.arange(10.0, 40.5, 0.5)  # Hz
VELOCITIES = np.arange(1.0, 1000.5, 0.5)  # m/s
M0 = 150 + 250 * np.exp(-FREQUENCIES / 15)  # m/s, the fundamental mode of test_picking.py
THRESHOLDS = ImageThresholds()


def _line(n_receivers: int, spacing: float = 1.0) -> LinearAcquisition:
    return LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=tuple(Coordinate(2.0 + k * spacing, 0.0, 0.0) for k in range(n_receivers)),
    )


def _ridge(
    velocities: np.ndarray,
    height: float,
    width: float = 1.5,
    band: tuple[float, float] = (15.0, 35.0),
    window_length: float = 47.0,
) -> np.ndarray:
    """A Gaussian ridge along `velocities`, `height` above the floor, `width` times the window's
    resolution wide, present only inside `band` (so a clean ridge fades before the grid's
    edges)."""
    f, c = FREQUENCIES[:, None], velocities[:, None]
    sigma = width * c * (c / f) / window_length / (2 * math.sqrt(2 * math.log(2)))
    ridge = height * np.exp(-0.5 * ((VELOCITIES - c) / sigma) ** 2)
    inside = (band[0] <= FREQUENCIES) & (band[1] >= FREQUENCIES)
    return ridge * inside[:, None]


def _image(*ridges: np.ndarray, acquisition: LinearAcquisition | None = None) -> DispersionImage:
    acquisition = acquisition or _line(48)
    floor = 1 / math.sqrt(len(acquisition.receivers))
    flat = np.zeros((FREQUENCIES.size, VELOCITIES.size))
    return DispersionImage(
        fv_map=floor + np.maximum.reduce([flat, *ridges]),
        fs=FREQUENCIES,
        vs=VELOCITIES,
        type=VelocityType.PHASE,
        acquisition=acquisition,
    )


def _flags(image: DispersionImage, usable: tuple[float, float] | None = None) -> dict[str, object]:
    result = judge_image("xmid_12.50", image, THRESHOLDS, usable)
    return {flag.name: flag.action.model_dump() for flag in result.flags}


def test_a_clean_ridge_passes_with_its_band() -> None:
    image = _image(_ridge(M0, 0.8))

    result = judge_image("xmid_12.50", image, THRESHOLDS)

    assert result.verdict == "pass"
    assert result.flags == ()
    assert result.kept == result.kept.model_copy(update={"band_hz": (15.0, 35.0), "n_traces": 48})
    by_name = {metric.name: metric for metric in result.metrics}
    assert by_name["coherent_columns"].value == round(41 / 61, 3)
    assert noise_floor(image) == 1 / math.sqrt(48)
    assert coherent_columns(image, THRESHOLDS.coherent_level).sum() == 41


def test_a_ridge_beyond_the_top_velocity_asks_for_a_wider_range() -> None:
    image = _image(_ridge(M0 + 900, 0.8))  # peaks above 1,000 m/s: the columns peak on the edge

    result = judge_image("xmid_12.50", image, THRESHOLDS)

    # The 47 m window tells 1,000 m/s from an infinite velocity above 21.3 Hz (f L > vmax): 28
    # of the 41 columns count.
    assert edge_peaks(image, coherent_columns(image, 0.3), 0.02) == (0, 28)
    assert _flags(image) == {
        "ridge_at_vmax": {
            "kind": "override",
            "stage": "phase_shift",
            "overrides": {"dispersion": {"vmax": 1500.0}},
        }
    }
    assert result.verdict == "retry"
    assert result.flags[0].stage == "phase_shift"


def test_energy_with_no_moveout_on_a_short_window_is_not_a_ridge_beyond_the_grid() -> None:
    # 24 receivers 0.25 m apart (5.75 m): at 10 to 40 Hz the window cannot tell 1,000 m/s from
    # an infinite velocity (f L at most 230 m/s), so energy at the grid's top is noise common to
    # every trace, not a ridge to widen the range for.
    short = _line(24, spacing=0.25)
    image = _image(_ridge(M0 + 900, 0.8, window_length=5.75), acquisition=short)

    assert edge_peaks(image, coherent_columns(image, 0.3), 0.02)[1] == 0
    assert "ridge_at_vmax" not in _flags(image)


def test_a_band_reaching_either_end_of_the_image_is_kept() -> None:
    image = _image(_ridge(M0, 0.8, band=(15.0, 40.0)))

    # Kept, not widened: on the demo line a wider band let other ridges compete.
    assert _flags(image) == {
        "band_at_fmax": {
            "kind": "keep",
            "note": "fmax stays: a wider band let other ridges compete",
        }
    }
    assert judge_image("xmid_12.50", image, THRESHOLDS).verdict == "pass"
    # The bottom too since 2026-09-25: the picker stops where the ridge breaks.
    low = _image(_ridge(M0, 0.8, band=(10.0, 35.0)))
    assert _flags(low) == {
        "band_at_fmin": {
            "kind": "keep",
            "note": "fmin stays: the picker stops where the ridge breaks",
        }
    }
    assert judge_image("xmid_12.50", low, THRESHOLDS).verdict == "pass"


def test_a_second_ridge_of_similar_strength_is_a_higher_mode() -> None:
    image = _image(_ridge(M0, 0.8), _ridge(M0 * 1.8, 0.7))

    competing = competing_ridges(image, coherent_columns(image, 0.3), 0.7, 0.15)

    assert competing.sum() == 41
    assert _flags(image) == {
        "competing_ridges": {"kind": "override", "stage": "picking", "overrides": {"max_modes": 2}}
    }
    # Weaker, it is no competition.
    assert _flags(_image(_ridge(M0, 0.8), _ridge(M0 * 1.8, 0.3))) == {}


def test_a_second_ridge_below_the_aliasing_limit_cuts_the_band() -> None:
    coarse = _line(10, spacing=5.0)  # the alias limit 2 dx f is 100 to 400 m/s here
    image = _image(_ridge(M0, 0.8), _ridge(M0 + 300, 0.7), acquisition=coarse)
    coherent = coherent_columns(image, 0.3)

    alias = aliased(image, coherent, 5.0)
    flags = _flags(image)

    # M0 = 150 + 250 exp(-f/15) drops below 10 f between 21 and 22 Hz.
    first = FREQUENCIES[alias][0]
    assert first == 21.5
    assert flags == {
        "aliasing": {
            "kind": "override",
            "stage": "phase_shift",
            "overrides": {"dispersion": {"fmax": float(first)}},
        }
    }


def test_weak_or_absent_coherence_suggests_a_mute() -> None:
    mute = {
        "kind": "override",
        "stage": "preprocessing",
        "overrides": {"muting": {"method": "mute", "vmin": 80.0, "vmax": 1500.0}},
    }

    assert _flags(_image(_ridge(M0, 0.8, band=(15.0, 20.0)))) == {"weak_coherence": mute}
    empty = judge_image("xmid_12.50", _image(), THRESHOLDS)
    assert empty.verdict == "retry"
    assert [flag.name for flag in empty.flags] == ["no_coherent_energy"]
    assert empty.kept.band_hz is None


def test_a_band_much_narrower_than_the_usable_one_is_reported() -> None:
    image = _image(_ridge(M0, 0.8))  # coherent from 15 to 35 Hz

    assert _flags(image, usable=(12.0, 38.0)) == {}
    narrow = judge_image("xmid_12.50", image, THRESHOLDS, (5.0, 80.0))
    (flag,) = narrow.flags
    assert flag.name == "narrower_than_usable"
    assert flag.stage == "phase_shift"
    assert "27%" in flag.message
    assert flag.action.model_dump()["kind"] == "keep" and narrow.verdict == "pass"


def test_a_peak_at_a_vmin_below_any_wave_is_an_artifact_to_start_above() -> None:
    image = _image(_ridge(np.full(FREQUENCIES.size, 1.0), 0.8))  # energy at 1 m/s, the grid's vmin

    assert _flags(image) == {
        "ridge_at_vmin": {
            "kind": "override",
            "stage": "phase_shift",
            "overrides": {"dispersion": {"vmin": 30.0}},
        }
    }
    (flag,) = judge_image("xmid_12.50", image, THRESHOLDS).flags
    assert "artifact" in flag.message
