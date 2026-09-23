import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import get_args

import numpy as np
import pytest

from paco.picking import PickedMode, PickingParameters, pick_modes
from paco.quality import (
    ADVICE,
    Flag,
    ImageQuality,
    QualityParameters,
    QualitySummary,
    RunQuality,
    Verdict,
    WindowQuality,
    dispersion_quality,
    measure_quality,
    summarize_quality,
)
from paco.runs import RunError, find_run, load_manifest, run_processing
from paco.settings import Settings
from sigpipe.algorithms.dispersion.phase_shift import phase_shift
from sigpipe.base import Coordinate, DispersionImage, LinearAcquisition, VelocityType


def _line(n_receivers: int) -> LinearAcquisition:
    """Receivers 1 m apart, the first 2 m from the source."""
    return LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=tuple(Coordinate(2.0 + k, 0.0, 0.0) for k in range(n_receivers)),
    )


def _m0(frequencies: np.ndarray) -> np.ndarray:
    """The fundamental mode of test_picking.py."""
    return 150 + 250 * np.exp(-frequencies / 15)


# The synthetic line of test_picking.py: 48 receivers, a 47 m window.
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
    # The ridge's width, to within the velocity step.
    assert quality.sharpness == pytest.approx(1.5, abs=0.03)


@pytest.mark.parametrize(
    ("image", "velocities", "flag", "measured"),
    [
        pytest.param(
            _image(_ridge(M0, height=0.8, width=0.5)),
            M0,
            "sharpness",
            0.5,
            id="narrower-than-the-window-resolves",
        ),
        pytest.param(
            _image(_ridge(M0, height=0.8, width=1.5), background=0.6),
            M0,
            "prominence",
            0.8 / 0.6,
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


# ---------------------------------------------------------------- the summary on its own


def _window(xmid: float, *flags: Flag) -> WindowQuality:
    verdict: Verdict = "good" if not flags else "doubtful" if len(flags) == 1 else "bad"
    quality = ImageQuality(verdict=verdict, flags=flags, n_points=10, band_hz=(10.0, 40.0))
    return WindowQuality(xmid=xmid, folder=f"xmid_{xmid:.2f}", quality=quality)


def test_summary_counts_verdicts_flags_and_good_stretches() -> None:
    record = RunQuality(
        run_id="20260923-100000-abcd",
        picking=PickingParameters(),
        thresholds=QualityParameters(),
        windows=(
            _window(1.0),
            _window(2.0),
            _window(3.0, "sharpness", "on_data", "prominence", "constant_wavelength"),
            _window(4.0),
            _window(5.0, "sharpness"),
            _window(6.0, "sharpness", "on_data", "prominence"),
            _window(7.0, "sharpness", "on_data"),
            _window(8.0),
            _window(9.0),
        ),
    )

    summary = summarize_quality(record, "active_p1")

    assert summary == QualitySummary(
        run_id="20260923-100000-abcd",
        profile="active_p1",
        n_windows=9,
        good=5,
        doubtful=1,
        bad=3,
        good_xmids=("1.00-2.00 m (2)", "4.00 m (1)", "8.00-9.00 m (2)"),
        flags={"sharpness": 4, "on_data": 3, "prominence": 2, "constant_wavelength": 1},
        # Advice for the three most frequent flags only.
        advice=(
            f"sharpness (4 windows): {ADVICE['sharpness']}",
            f"on_data (3 windows): {ADVICE['on_data']}",
            f"prominence (2 windows): {ADVICE['prominence']}",
        ),
    )
    assert list(summary.flags) == ["sharpness", "on_data", "prominence", "constant_wavelength"]


def test_every_flag_has_advice() -> None:
    assert set(ADVICE) == set(get_args(Flag.__value__))


# ---------------------------------------------------------------- whole runs

# Four 24-receiver windows along each demo line, as in test_runs.py.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}


@dataclass(frozen=True)
class Assessed:
    run_id: str
    folder: Path
    summary: QualitySummary


@dataclass(frozen=True)
class Outputs:
    settings: Settings
    working_dir: Path  # where the tools were called from
    runs: dict[str, Assessed]  # by preset: active, passive


# Real runs are the slow part: both demo profiles are processed and assessed once.
@pytest.fixture(scope="module")
def outputs(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Outputs:
    root = tmp_path_factory.mktemp("quality")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    working_dir = root / "working_dir"
    working_dir.mkdir()

    runs: dict[str, Assessed] = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(working_dir)
        for preset in ("active", "passive"):
            run_id = run_processing(f"{preset}_p1", preset, SMALL_WINDOWS, settings).run_id
            summary = dispersion_quality(run_id, settings)
            runs[preset] = Assessed(run_id, find_run(run_id, settings), summary)
    return Outputs(settings, working_dir, runs)


def test_active_windows_are_good(outputs: Outputs) -> None:
    assert outputs.runs["active"].summary.model_dump(exclude={"run_id"}) == {
        "profile": "active_p1",
        "n_windows": 4,
        "good": 4,
        "doubtful": 0,
        "bad": 0,
        "good_xmids": ("2.88-20.88 m (4)",),
        "flags": {},
        "advice": (),
    }


def test_no_passive_window_is_good(outputs: Outputs) -> None:
    # With the preset's defaults, passive images show no clean ridge: every window raises a flag,
    # though xmid 2.88 raises only one (constant wavelength), so it is doubtful, not bad.
    summary = outputs.runs["passive"].summary

    assert (summary.n_windows, summary.good, summary.doubtful, summary.bad) == (4, 0, 1, 3)
    assert summary.good_xmids == ()


@pytest.mark.parametrize("preset", ["active", "passive"])
def test_quality_files_hold_what_the_summary_says(outputs: Outputs, preset: str) -> None:
    run = outputs.runs[preset]
    manifest = load_manifest(run.run_id, outputs.settings)

    record = RunQuality.model_validate_json((run.folder / "quality.json").read_text())

    assert summarize_quality(record, manifest.profile.name) == run.summary
    assert (record.picking, record.thresholds) == (PickingParameters(), QualityParameters())
    assert [(window.xmid, window.folder) for window in record.windows] == [
        (window.xmid, window.folder) for window in manifest.windows
    ]
    for window in record.windows:
        saved = (run.folder / window.folder / "quality.json").read_text()
        assert ImageQuality.model_validate_json(saved) == window.quality


@pytest.mark.parametrize("preset", ["active", "passive"])
def test_summary_stays_short(outputs: Outputs, preset: str) -> None:
    # The summary is what the agent reads: it must stay far below the tool-output budget.
    assert len(outputs.runs[preset].summary.model_dump_json()) < 1_000


def test_nothing_is_written_in_the_working_directory(outputs: Outputs) -> None:
    assert list(outputs.working_dir.iterdir()) == []


def test_parameters_are_recorded(outputs: Outputs, tmp_path: Path) -> None:
    # On a copy of the active run, so the shared run keeps its default assessment.
    active = outputs.runs["active"]
    settings = Settings(input_dir=outputs.settings.input_dir, output_dir=tmp_path)
    folder = shutil.copytree(active.folder, tmp_path / "active_p1" / active.run_id)
    picking = PickingParameters(smoothness=2.0)
    thresholds = QualityParameters(min_sharpness=2.0)

    summary = dispersion_quality(active.run_id, settings, picking, thresholds)

    record = RunQuality.model_validate_json((folder / "quality.json").read_text())
    assert (record.picking, record.thresholds) == (picking, thresholds)
    # No demo window is twice as wide as the resolution.
    assert (summary.good, summary.flags) == (0, {"sharpness": 4})


@pytest.mark.parametrize("run_id", ["20260923-000000-0000", "../active_p1", "*"])
def test_unknown_runs_are_refused(outputs: Outputs, run_id: str) -> None:
    active, passive = outputs.runs["active"].run_id, outputs.runs["passive"].run_id

    with pytest.raises(RunError) as error:
        dispersion_quality(run_id, outputs.settings)

    assert str(error.value) == (
        f"Unknown run '{run_id}'. Latest runs: {passive} (passive_p1), {active} (active_p1)."
    )


def test_a_run_without_images_is_refused(demo_settings: Settings) -> None:
    # A single 3-receiver window, with a dispersion band that makes it fail: cheap to run.
    overrides = {"masw": {"length": 3, "step": 94}, "dispersion": {"fmin": 10.1, "fmax": 10.2}}
    run_id = run_processing("active_p1", "active", overrides, demo_settings).run_id

    with pytest.raises(RunError, match=f"^Run '{run_id}' has no dispersion image to assess"):
        dispersion_quality(run_id, demo_settings)
