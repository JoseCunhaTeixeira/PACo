"""G1 on synthetic records with a known answer: a surface wave of known velocity and known noise,
with defects planted one at a time."""

from dataclasses import replace

import numpy as np
import pytest
from sigpipe.base import Coordinate, LinearAcquisition, Stream

from paco.qc.g1_signal import (
    SignalThresholds,
    dead_clipped_nan,
    first_breaks,
    judge_signal,
    lateral_coherence,
    rms_decay_outliers,
    signal_windows,
    snr_db,
    trigger_shift,
    usable_band,
)
from paco.transformers import ShiftTrigger

SAMPLING = 1000.0
N_TRACES = 24
DX = 1.0
VELOCITY = 200.0  # m/s, the surface wave
FREQUENCY = 20.0  # Hz, the wavelet's centre


def _shot(
    noise: float = 0.02,
    t0: float = 0.0,
    duration_s: float = 2.0,
    seed: int = 0,
) -> Stream:
    """A causal wavelet (Berlage: a damped, delayed cosine, starting at the arrival) moving out
    at VELOCITY from a source 2 m before the first receiver, decaying with offset, over white
    noise of RMS `noise`; it peaks at 1 on the nearest trace."""
    rng = np.random.default_rng(seed)
    ts = np.arange(int(duration_s * SAMPLING)) / SAMPLING
    receivers = tuple(Coordinate(2.0 + i * DX, 0.0, 0.0) for i in range(N_TRACES))
    acquisition = LinearAcquisition(source=Coordinate(0.0, 0.0, 0.0), receivers=receivers)
    xt = rng.standard_normal((N_TRACES, ts.size)) * noise
    for i, offset in enumerate(acquisition.offsets):
        tau = ts - (t0 + offset / VELOCITY)
        pulse = np.where(
            tau >= 0, tau * np.exp(-2 * FREQUENCY * tau) * np.cos(2 * np.pi * FREQUENCY * tau), 0.0
        )
        xt[i] += pulse / (np.exp(-1) / (2 * FREQUENCY)) / np.sqrt(offset / 2.0)
    return Stream(
        xt=xt.astype(np.float32),
        ts=ts.astype(np.float32),
        sampling_freq=SAMPLING,
        acquisition=acquisition,
    )


def _with_trace(stream: Stream, index: int, trace: np.ndarray) -> Stream:
    xt = stream.xt.copy()
    xt[index] = trace
    return replace(stream, xt=xt)


THRESHOLDS = SignalThresholds()


# ---------------------------------------------------------------- each metric on a known input


def test_a_clean_shot_passes_with_its_wave_in_the_window() -> None:
    result = judge_signal("1.dat", _shot(), THRESHOLDS)

    assert result.verdict == "pass"
    assert result.flags == ()
    by_name = {metric.name: metric for metric in result.metrics}
    # A 0.1 s pulse over a 0.36 s window, against noise of RMS 0.02: about 16 dB.
    assert by_name["snr_db"].value is not None and by_name["snr_db"].value > 10
    assert (
        by_name["lateral_coherence"].value is not None and by_name["lateral_coherence"].value > 0.9
    )
    assert (
        by_name["trigger_shift_s"].value is not None
        and abs(by_name["trigger_shift_s"].value) < THRESHOLDS.max_trigger_shift_s
    )
    assert result.kept.n_traces == N_TRACES
    assert result.kept.band_hz is not None
    fmin, fmax = result.kept.band_hz
    assert fmin < FREQUENCY < fmax


def test_the_windows_follow_the_moveout_and_the_noise_comes_after_the_slowest_arrival() -> None:
    stream = _shot()
    windows = signal_windows(
        np.asarray(stream.acquisition.offsets), np.asarray(stream.ts), THRESHOLDS
    )

    assert windows is not None
    assert windows.where.startswith("after the slowest arrival")
    # The far trace's window ends later than the near one's, and both hold their arrival.
    near, far = windows.signal[0], windows.signal[-1]
    ts = np.asarray(stream.ts)
    assert ts[near][-1] < ts[far][-1]
    for i in (0, -1):
        arrival = stream.acquisition.offsets[i] / VELOCITY
        assert windows.signal[i][int(arrival * SAMPLING)]
    assert not (windows.signal & windows.noise).any()
    # A record that ends before the slowest arrival has no noise window.
    assert (
        signal_windows(
            np.asarray(stream.acquisition.offsets), np.asarray(stream.ts[:200]), THRESHOLDS
        )
        is None
    )


def test_dead_clipped_and_nan_traces_are_found_and_excluded() -> None:
    stream = _shot()
    stream = _with_trace(stream, 3, np.zeros(stream.xt.shape[1], dtype=np.float32))
    stream = _with_trace(stream, 7, np.clip(stream.xt[7], -0.02, 0.02))
    nan_trace = stream.xt[11].copy()
    nan_trace[100] = np.nan
    stream = _with_trace(stream, 11, nan_trace)

    dead, clipped, nan = dead_clipped_nan(stream.xt, THRESHOLDS)
    result = judge_signal("1.dat", stream, THRESHOLDS)

    assert list(np.flatnonzero(dead)) == [3]
    assert list(np.flatnonzero(clipped)) == [7]
    assert list(np.flatnonzero(nan)) == [11]
    assert result.verdict == "retry"
    excluded = {flag.name: flag.action for flag in result.flags}
    assert excluded["dead_traces"].model_dump() == {
        "kind": "exclude_traces",
        "record": "1.dat",
        "traces": (3,),
    }
    assert excluded["clipped_traces"].model_dump()["traces"] == (7,)
    assert excluded["nan_traces"].model_dump()["traces"] == (11,)
    assert all(not flag.fixable for flag in result.flags)
    assert result.kept.n_traces == N_TRACES - 3


def test_an_amplitude_off_the_decay_is_an_outlier() -> None:
    stream = _shot()
    stream = _with_trace(stream, 10, stream.xt[10] * 30)
    rms = np.sqrt(np.mean(stream.xt.astype(float) ** 2, axis=1))

    outliers = rms_decay_outliers(
        rms, np.asarray(stream.acquisition.offsets), np.ones(N_TRACES, bool), 3.0
    )
    result = judge_signal("1.dat", stream, THRESHOLDS)

    assert list(np.flatnonzero(outliers)) == [10]
    (flag,) = result.flags
    assert flag.name == "rms_outliers" and not flag.fixable
    assert flag.action.model_dump()["traces"] == (10,)


def test_the_snr_matches_the_noise_planted() -> None:
    quiet, loud = _shot(noise=0.001), _shot(noise=0.05)
    windows = signal_windows(
        np.asarray(quiet.acquisition.offsets), np.asarray(quiet.ts), THRESHOLDS
    )
    assert windows is not None

    quiet_snr = np.median(snr_db(quiet.xt.astype(float), windows))
    loud_snr = np.median(snr_db(loud.xt.astype(float), windows))

    # 50 x less noise RMS is 34 dB more, within the window's own share of the wavelet.
    assert 25 < quiet_snr - loud_snr < 40
    assert judge_signal("1.dat", loud, THRESHOLDS).verdict == "pass"
    weak = _shot(noise=1.0)
    result = judge_signal("1.dat", weak, THRESHOLDS)
    assert "low_snr" in {flag.name for flag in result.flags}


def test_the_usable_band_holds_the_wavelet_and_no_more() -> None:
    stream = _shot(noise=0.05)
    windows = signal_windows(
        np.asarray(stream.acquisition.offsets), np.asarray(stream.ts), THRESHOLDS
    )
    assert windows is not None

    band = usable_band(stream.xt.astype(float), SAMPLING, windows, THRESHOLDS.band_db)

    assert band is not None
    fmin, fmax = band
    # A 20 Hz Berlage pulse has its energy between a few Hz and about 60 Hz.
    assert 0 <= fmin <= 12  # a causal pulse has energy down to 0 Hz
    assert 35 <= fmax <= 80


def test_neighbours_correlate_unless_one_is_reversed() -> None:
    stream = _shot()
    windows = signal_windows(
        np.asarray(stream.acquisition.offsets), np.asarray(stream.ts), THRESHOLDS
    )
    assert windows is not None
    max_lag_s = DX / THRESHOLDS.vg_min

    coherence, polarity = lateral_coherence(stream.xt.astype(float), windows, SAMPLING, max_lag_s)
    reversed_stream = _with_trace(stream, 5, -stream.xt[5])
    result = judge_signal("1.dat", reversed_stream, THRESHOLDS)

    assert coherence.min() > 0.9
    assert (polarity > 0).all()
    (flag,) = result.flags
    assert flag.name == "reversed_polarity"
    assert flag.action.model_dump()["traces"] == (5,)


def test_a_shifted_trigger_rejects_the_record() -> None:
    stream = _shot()
    windows = signal_windows(
        np.asarray(stream.acquisition.offsets), np.asarray(stream.ts), THRESHOLDS
    )
    assert windows is not None
    breaks = first_breaks(
        stream.xt.astype(float), np.asarray(stream.ts), windows, THRESHOLDS.first_break_ratio
    )
    fit = trigger_shift(breaks, np.asarray(stream.acquisition.offsets))

    assert fit is not None
    t0, velocity, scatter = fit
    assert abs(t0) < THRESHOLDS.max_trigger_shift_s
    assert 150 < velocity < 250
    assert scatter < 0.005
    shifted = judge_signal("1.dat", _shot(t0=0.05), THRESHOLDS)
    assert shifted.verdict == "retry"
    (flag,) = shifted.flags
    assert flag.name == "shifted_trigger" and flag.fixable
    action = flag.action.model_dump()
    assert action["kind"] == "override" and action["stage"] == "preprocessing"
    assert action["overrides"]["trigger"]["t0"] == pytest.approx(0.05, abs=0.005)
    # Corrected by that t0, the record passes.
    (corrected,) = ShiftTrigger(action["overrides"]["trigger"]["t0"]).transform([_shot(t0=0.05)])
    assert judge_signal("1.dat", corrected, THRESHOLDS).verdict == "pass"


def test_a_trigger_that_differs_from_trace_to_trace_rejects_the_record() -> None:
    stream = _shot()
    rng = np.random.default_rng(1)
    xt = np.stack(
        [np.roll(trace, int(rng.integers(-150, 151))) for trace in stream.xt]
    )  # each trace shifted by its own -150 to 150 ms: no line through the first breaks
    jittered = replace(stream, xt=xt)

    result = judge_signal("1.dat", jittered, THRESHOLDS)

    assert result.verdict == "reject"
    names = {flag.name for flag in result.flags}
    assert "inconsistent_trigger" in names
    flag = next(flag for flag in result.flags if flag.name == "inconsistent_trigger")
    assert not flag.fixable
    assert flag.action.model_dump() == {"kind": "exclude_record", "record": "1.dat"}


def test_a_passive_record_gets_only_the_checks_without_a_trigger() -> None:
    stream = _with_trace(_shot(), 3, np.zeros(2000, dtype=np.float32))

    result = judge_signal("1.dat", stream, THRESHOLDS, active=False)

    assert {metric.name for metric in result.metrics} == {
        "dead_traces",
        "clipped_traces",
        "nan_traces",
    }
    assert [flag.name for flag in result.flags] == ["dead_traces"]


def test_the_energy_a_mute_removed_is_reported() -> None:
    stream = _shot()
    muted = replace(
        stream, xt=np.where(np.asarray(stream.ts)[None, :] > 0.5, 0.0, stream.xt).astype(np.float32)
    )

    result = judge_signal("1.dat", muted, THRESHOLDS, raw=stream)

    removed = next(metric for metric in result.metrics if metric.name == "energy_removed")
    assert removed.value is not None
    assert 0.0 < removed.value < 0.5


@pytest.mark.parametrize("field", ["dead_ratio", "clip_share", "band_db"])
def test_thresholds_refuse_nonsense(field: str) -> None:
    with pytest.raises(ValueError):
        SignalThresholds(**{field: 0})
