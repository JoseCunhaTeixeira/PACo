import math
from dataclasses import replace

import numpy as np
import pytest
from sigpipe.algorithms.dispersion.phase_shift import phase_shift
from sigpipe.algorithms.picking.dispersion.tracking import PickedMode, pick_modes
from sigpipe.base import Coordinate, DispersionImage, LinearAcquisition, VelocityType

from paco.quality import Flag, ImageQuality, QualityParameters, Verdict, measure_quality


def _line(n_receivers: int) -> LinearAcquisition:
    """Receivers 1 m apart, the first 2 m from the source."""
    return LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=tuple(Coordinate(2.0 + k, 0.0, 0.0) for k in range(n_receivers)),
    )


def _m0(frequencies: np.ndarray) -> np.ndarray:
    """The fundamental mode of sigpipe's picking tests."""
    return 150 + 250 * np.exp(-frequencies / 15)


# The synthetic line of sigpipe's picking tests: 48 receivers, a 47 m window.
ACQUISITION = _line(48)
WINDOW_LENGTH = 47.0  # m
NOISE_FLOOR = 1 / math.sqrt(48)
FREQUENCIES = np.arange(10.0, 40.5, 0.5)  # Hz
VELOCITIES = np.arange(1.0, 1000.5, 0.5)  # m/s
M0 = _m0(FREQUENCIES)  # m/s
CONSTANT_WAVELENGTH = 6.0 * FREQUENCIES  # m/s, a 6 m wavelength at every frequency


def _ridge(velocities: np.ndarray, height: float, width: float) -> np.ndarray:
    """A Gaussian ridge along `velocities`, `height` above the noise floor at its peak, and `width`
    times the resolution (v * wavelength / window length) wide at half that height."""
    f, c = FREQUENCIES[:, None], velocities[:, None]
    sigma = width * c * (c / f) / WINDOW_LENGTH / (2 * math.sqrt(2 * math.log(2)))
    return height * np.exp(-0.5 * ((VELOCITIES - c) / sigma) ** 2)


def _image(*ridges: np.ndarray, background: float = 0.0) -> DispersionImage:
    """Ridges over a flat background, both measured above the noise floor."""
    flat = np.full((FREQUENCIES.size, VELOCITIES.size), background)
    return DispersionImage(
        fv_map=NOISE_FLOOR + np.maximum.reduce([flat, *ridges]),
        fs=FREQUENCIES,
        vs=VELOCITIES,
        type=VelocityType.PHASE,
        acquisition=ACQUISITION,
    )


def _pick(image: DispersionImage, velocities: np.ndarray) -> PickedMode:
    """An M0 pick at `velocities`, on the image's grid, every point kept."""
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


CLEAN = _image(_ridge(M0, height=0.8, width=1.5))


# ---------------------------------------------------------------- one image


def test_a_clean_ridge_is_good() -> None:
    quality = measure_quality(CLEAN, _pick(CLEAN, M0))

    assert quality.model_dump(exclude={"sharpness", "prominence"}) == {
        "verdict": "good",
        "flags": (),
        "n_points": 61,
        "band_hz": (10.0, 40.0),
        "on_data": 1.0,
        "constant_wavelength": 0.0,
    }
    # 1.5 resolutions wide, against a perfect plane wave's 1.09 on this array (48 receivers, 47 m).
    assert quality.sharpness == pytest.approx(1.5 / 1.085, abs=0.03)


@pytest.mark.parametrize(
    ("image", "velocities", "flag", "measured"),
    [
        pytest.param(
            _image(_ridge(M0, height=0.8, width=0.5)),
            M0,
            "sharpness",
            0.5 / 1.085,
            id="narrower-than-the-window-resolves",
        ),
        pytest.param(
            _image(_ridge(M0, height=0.8, width=1.5), background=0.6),
            M0,
            "prominence",
            0.8 / 0.6 / 4,  # a long array's plane wave counts as prominent enough: 4
            id="barely-above-the-rest",
        ),
        pytest.param(
            _image(_ridge(M0, height=0.6, width=1.5), _ridge(1.8 * M0, height=0.9, width=1.5)),
            M0,
            "on_data",
            0.0,
            id="brighter-ridge-above",
        ),
        pytest.param(
            _image(_ridge(CONSTANT_WAVELENGTH, height=0.8, width=1.5)),
            CONSTANT_WAVELENGTH,
            "constant_wavelength",
            1.0,
            id="constant-wavelength",
        ),
    ],
)
def test_each_measurement_raises_its_own_flag(
    image: DispersionImage, velocities: np.ndarray, flag: Flag, measured: float
) -> None:
    quality = measure_quality(image, _pick(image, velocities))

    assert (quality.verdict, quality.flags) == ("doubtful", (flag,))
    assert getattr(quality, flag) == pytest.approx(measured, abs=0.03)


@pytest.mark.parametrize(
    ("thresholds", "verdict", "flags"),
    [
        ({}, "good", ()),
        ({"min_sharpness": 2.0}, "doubtful", ("sharpness",)),
        ({"min_sharpness": 2.0, "min_prominence": 1e6}, "bad", ("sharpness", "prominence")),
    ],
)
def test_the_verdict_counts_the_flags(
    thresholds: dict[str, float], verdict: Verdict, flags: tuple[Flag, ...]
) -> None:
    quality = measure_quality(CLEAN, _pick(CLEAN, M0), QualityParameters(**thresholds))

    assert (quality.verdict, quality.flags) == (verdict, flags)


def test_no_pick_is_bad() -> None:
    assert measure_quality(CLEAN, None) == ImageQuality(
        verdict="bad", flags=("no_ridge",), n_points=0, band_hz=None
    )


def test_a_pick_needs_two_kept_points() -> None:
    kept = np.zeros(FREQUENCIES.size, dtype=bool)
    kept[10] = True

    quality = measure_quality(CLEAN, replace(_pick(CLEAN, M0), kept=kept))

    assert quality == ImageQuality(verdict="bad", flags=("no_ridge",), n_points=1, band_hz=None)


def test_0_hz_is_not_measured() -> None:
    # 0 Hz has no wavelength.
    pick = _pick(CLEAN, M0)
    with_0_hz = replace(
        pick,
        frequencies=np.r_[0.0, pick.frequencies],
        velocities=np.r_[300.0, pick.velocities],
        coherence=np.r_[1.0, pick.coherence],
        pinned=np.r_[False, pick.pinned],
        kept=np.r_[True, pick.kept],
    )

    assert measure_quality(CLEAN, with_0_hz) == measure_quality(CLEAN, pick)


def _through_phase_shift(traces: np.ndarray, acquisition: LinearAcquisition) -> DispersionImage:
    """The dispersion image of 2 s traces at 500 Hz, on PAC's default dispersion grid."""
    fs, vs, fv_map = phase_shift(
        traces,
        500.0,
        acquisition.offsets.astype(float),
        fmin=0.0,
        fmax=100.0,
        vmin=1.0,
        vmax=1_000.0,
    )
    return DispersionImage(
        fv_map=fv_map, fs=fs, vs=vs, type=VelocityType.PHASE, acquisition=acquisition
    )


@pytest.mark.parametrize("n_receivers", [24, 48])
def test_a_perfect_plane_wave_is_good(n_receivers: int) -> None:
    # Its sharpness, 0.98 to 1.08 depending on the array, is the physical limit: min_sharpness
    # must stay below it.
    acquisition = _line(n_receivers)
    offsets = acquisition.offsets.astype(float)
    frequencies = np.fft.rfftfreq(1_000, 1 / 500.0)
    spectra = np.exp(-2j * np.pi * offsets[:, None] * frequencies / _m0(frequencies))
    image = _through_phase_shift(np.fft.irfft(spectra, n=1_000, axis=1), acquisition)
    (m0,) = pick_modes(image)

    assert measure_quality(image, m0).verdict == "good"


@pytest.mark.parametrize("seed", range(5))
def test_pure_noise_is_bad(seed: int) -> None:
    # Random traces: the picker can still find an M0 in them.
    traces = np.random.default_rng(seed).standard_normal((len(ACQUISITION.receivers), 1_000))
    image = _through_phase_shift(traces, ACQUISITION)
    modes = pick_modes(image)

    assert measure_quality(image, modes[0] if modes else None).verdict == "bad"
