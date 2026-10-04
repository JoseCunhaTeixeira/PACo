"""G2 on analytic images with a known answer: Gaussian ridges over the noise floor, as in
test_quality.py, each defect planted alone."""

import math

import numpy as np
import pytest
from sigpipe.base import Coordinate, DispersionImage, LinearAcquisition, VelocityType
from sigpipe.dataio.selection_plotting import SelectionScores
from sigpipe.masw.quality.image import (
    aliased,
    coherent_columns,
    competing_ridges,
    edge_peaks,
    noise_floor,
)
from sigpipe.masw.quality.measures import (
    Measure,
    SignalLimits,
    SignalReport,
    selection_measures,
)

from paco.qc.g2_image import ImageThresholds, judge_image, more_data
from paco.qc.models import GateResult

FREQUENCIES = np.arange(10.0, 40.5, 0.5)  # Hz
VELOCITIES = np.arange(1.0, 1000.5, 0.5)  # m/s
M0 = 150 + 250 * np.exp(-FREQUENCIES / 15)  # m/s, the fundamental mode of sigpipe's picking tests
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
    # The bottom too: the picker stops where the ridge breaks.
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
    # Kept: the fundamental mode is picked as the slowest ridge whatever the modes picked; two
    # picked would change neither.
    assert _flags(image) == {
        "competing_ridges": {
            "kind": "keep",
            "note": "the fundamental mode picked as the slowest ridge",
        }
    }
    assert judge_image("xmid_12.50", image, THRESHOLDS).verdict == "pass"
    # Weaker, it is no competition.
    assert _flags(_image(_ridge(M0, 0.8), _ridge(M0 * 1.8, 0.3))) == {}


def test_a_second_ridge_below_the_aliasing_limit_stops_the_windows_picking() -> None:
    coarse = _line(10, spacing=5.0)  # the alias limit 2 dx f is 100 to 400 m/s here
    image = _image(_ridge(M0, 0.8), _ridge(M0 + 300, 0.7), acquisition=coarse)
    coherent = coherent_columns(image, 0.3)

    alias = aliased(image, coherent, 5.0)
    flags = _flags(image)

    # M0 = 150 + 250 exp(-f/15) drops below 10 f between 21 and 22 Hz.
    first = FREQUENCIES[alias][0]
    assert first == 21.5
    # The image is the line's: the window's picking stops below the alias, the image stands.
    assert flags == {
        "aliasing": {"kind": "override", "stage": "picking", "overrides": {"fmax": float(first)}}
    }
    assert judge_image("xmid_12.50", image, THRESHOLDS).verdict == "pass"


def test_weak_or_absent_coherence_suggests_a_mute() -> None:
    mute = {
        "kind": "override",
        "stage": "preprocessing",
        # No width: each record's own pulse, filled when the records are done again (shots.py).
        "overrides": {"muting": {"method": "mute", "vmin": 80.0, "vmax": 1500.0}},
    }

    assert _flags(_image(_ridge(M0, 0.8, band=(15.0, 20.0)))) == {"weak_coherence": mute}
    empty = judge_image("xmid_12.50", _image(), THRESHOLDS)
    assert empty.verdict == "retry"
    assert [flag.name for flag in empty.flags] == ["no_coherent_energy"]
    assert empty.kept.band_hz is None


def test_an_image_mostly_noise_of_muted_records_is_flagged_and_kept() -> None:
    # Muted already, a mute could not change it: nothing to try, the window goes on to its pick,
    # the later gates judging it.
    image = _image(_ridge(M0, 0.8, band=(15.0, 20.0)))
    weak = judge_image("xmid_12.50", image, THRESHOLDS, muted=True)
    (flag,) = weak.flags
    assert flag.name == "weak_coherence" and flag.action.model_dump()["kind"] == "keep"
    assert weak.verdict == "pass" and "mute" not in flag.message
    empty = judge_image("xmid_12.50", _image(), THRESHOLDS, muted=True)
    assert empty.verdict == "pass" and empty.flags[0].action.model_dump()["kind"] == "keep"


def test_a_ridge_at_the_grids_floor_is_kept_with_nothing_lower_to_try() -> None:
    # Under the floor (30 m/s) a peak is an artifact: a grid starting there can go no lower, and
    # lowered from 45 m/s it stops at the floor (not 1-30-20-30 m/s again and again).
    vs = np.arange(30.0, 1000.5, 0.5)

    def at(vmin: float) -> DispersionImage:
        grid = np.arange(vmin, 1000.5, 0.5)
        ridge = np.exp(-0.5 * ((grid[None, :] - vmin) / 2.0) ** 2) * np.ones((FREQUENCIES.size, 1))
        return DispersionImage(
            fv_map=1 / math.sqrt(48) + 0.8 * ridge,
            fs=FREQUENCIES,
            vs=grid,
            type=VelocityType.PHASE,
            acquisition=_line(48),
        )

    assert vs[0] == THRESHOLDS.vmin_floor
    floor = judge_image("xmid_12.50", at(30.0), THRESHOLDS)
    (flag,) = [one for one in floor.flags if one.name == "ridge_at_vmin"]
    assert flag.action.model_dump() == {"kind": "keep", "note": "the range at its floor"}
    above = judge_image("xmid_12.50", at(45.0), THRESHOLDS)
    (flag,) = [one for one in above.flags if one.name == "ridge_at_vmin"]
    assert flag.action.model_dump()["overrides"] == {"dispersion": {"vmin": 30.0}}


def test_a_passive_image_mostly_noise_is_flagged_and_kept() -> None:
    # A passive line has no muting: nothing to try, the window goes on to its pick, the later
    # gates judging it.
    image = _image(_ridge(M0, 0.8, band=(15.0, 20.0)))
    weak = judge_image("xmid_12.50", image, THRESHOLDS, mode="passive")
    (flag,) = weak.flags
    assert flag.name == "weak_coherence" and flag.action.model_dump()["kind"] == "keep"
    assert weak.verdict == "pass" and "mute" not in flag.message
    empty = judge_image("xmid_12.50", _image(), THRESHOLDS, mode="passive")
    assert [flag.name for flag in empty.flags] == ["no_coherent_energy"]
    assert empty.verdict == "pass"


def _correlations(snr: float, coherence: float = 0.9) -> SignalReport:
    """A window's stacked correlations as measure_signal reports them: their SNR (against the one
    limit of every signal) and coherence, their band and traces fine."""
    limits = SignalLimits()
    alive = np.zeros(5, dtype=bool)
    measures = (
        Measure(name="dead_traces", value=0, threshold=0, bound="max", passed=True, of="signal"),
        Measure(
            name="snr_db",
            value=snr,
            threshold=limits.min_snr_db,
            bound="min",
            passed=snr >= limits.min_snr_db,
            unit="dB",
            of="signal",
        ),
        Measure(
            name="usable_band_hz",
            value=40.0,
            threshold=0,
            bound="min",
            passed=True,
            unit="Hz",
            of="spectrum",
        ),
        Measure(
            name="lateral_coherence",
            value=coherence,
            threshold=limits.min_coherence,
            bound="min",
            passed=coherence >= limits.min_coherence,
            of="signal",
        ),
    )
    return SignalReport(measures, alive, alive, alive, alive)


PASSIVE = {
    "selection": {"method": "fk", "threshold": 0.1},
    "slicing": {"segment_duration": 2.0, "segment_step": 2.0},
}


def test_a_windows_correlations_are_judged_as_a_record_is() -> None:
    # Measured as a shot's from the virtual source, against the 6 dB of every signal: falling
    # short, the phase shift again with more of the data; a passive-active window stacks every
    # shot within reach already, and is rejected.
    image = _image(_ridge(M0, 0.8))
    more = more_data("passive", PASSIVE)

    clean = judge_image("xmid_12.50", image, THRESHOLDS, mode="passive",
                        correlations=_correlations(14.0), more=more)  # fmt: skip
    noisy = judge_image("xmid_12.50", image, THRESHOLDS, mode="passive",
                        correlations=_correlations(2.1), more=more)  # fmt: skip
    lost = judge_image("xmid_12.50", image, THRESHOLDS, mode="passive-active",
                       correlations=_correlations(2.1, 0.4),
                       more=more_data("passive-active", {}))  # fmt: skip

    assert clean.flags == () and clean.verdict == "pass"
    (flag,) = noisy.flags
    assert flag.name == "noisy_correlations" and noisy.verdict == "retry"
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "phase_shift",
        "overrides": {"selection": {"threshold": 0.05}},
    }
    (flag,) = lost.flags
    assert flag.name == "noisy_correlations" and lost.verdict == "reject"
    assert "median SNR 2.1 dB (at least 6)" in flag.message
    assert "coherence 0.4 (at least 0.5)" in flag.message
    # Every measure says its object: the correlations' signal and spectrum, the image's.
    assert {metric.of for metric in noisy.metrics} == {"signal", "spectrum", "image"}


def _kept(count: int, segments: int) -> SelectionScores:
    """An fk selection keeping `count` of `segments`, each kept one flipped."""
    return SelectionScores(threshold=0.1, flip=True,
                           ratios=(0.3,) * count + (0.01,) * (segments - count),
                           kept=(True,) * count + (False,) * (segments - count),
                           segments=segments, kept_count=count, flipped_count=count,
                           kept_share=count / segments)  # fmt: skip


def test_a_passive_windows_fk_selection_is_judged() -> None:
    image = _image(_ridge(M0, 0.8))

    def judged(selection: SelectionScores) -> GateResult:
        return judge_image("xmid_12.50", image, THRESHOLDS, mode="passive",
                           selection=selection_measures(selection, SignalLimits()),
                           more=more_data("passive", PASSIVE))  # fmt: skip

    few = judged(_kept(1, 200))
    # Good segments are rare: 2 % kept is enough, 0.5 % is not.
    rare = judged(_kept(4, 200))

    (flag,) = few.flags
    assert flag.name == "few_segments_kept" and few.verdict == "retry"
    kept = next(metric for metric in few.metrics if metric.name == "fk_kept")
    assert (kept.value, kept.threshold, kept.of) == (0.005, 0.01, "selection")
    assert rare.flags == () and rare.verdict == "pass"


def test_more_data_halves_the_fk_threshold_and_stops_when_it_keeps_every_segment() -> None:
    fk = more_data("passive", PASSIVE)
    every = more_data(
        "passive",
        {
            "selection": {"method": "none"},
            "slicing": {"segment_duration": 2.0, "segment_step": 1.0},
        },
    )

    assert fk is not None and fk.overrides == {"selection": {"threshold": 0.05}}
    # No selection: every segment stacked already. Longer segments are no more data, the same
    # record cut in fewer pieces.
    assert every is None
    assert more_data("passive-active", PASSIVE) is None


def test_a_band_much_narrower_than_the_usable_one_is_reported() -> None:
    # Against the part of the records' usable band the image spans (10 to 40 Hz here): beyond
    # its frequencies, nothing it could use.
    image = _image(_ridge(M0, 0.8))  # coherent from 15 to 35 Hz

    assert _flags(image, usable=(12.0, 38.0)) == {}
    wide = judge_image("xmid_12.50", image, THRESHOLDS, (5.0, 80.0))
    share = next(metric for metric in wide.metrics if metric.name == "band_share_of_usable")
    assert wide.flags == () and share.value == pytest.approx(20 / 30, abs=0.001)
    assert share.over.endswith("usable band within the image's, 10-40 Hz")
    narrow = judge_image("xmid_12.50", _image(_ridge(M0, 0.8, band=(15.0, 25.0))), THRESHOLDS,
                         (10.0, 40.0))  # fmt: skip
    (flag,) = [one for one in narrow.flags if one.name == "narrower_than_usable"]
    assert flag.stage == "phase_shift" and "33%" in flag.message
    assert flag.action.model_dump()["kind"] == "keep"


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
