"""G1 on synthetic records with a known answer: a surface wave of known velocity and known noise,
with defects planted one at a time. The measures themselves are sigpipe's, tested there
(sigpipe.masw.quality.signal)."""

from dataclasses import replace

import numpy as np
import pytest
from sigpipe.base import Coordinate, LinearAcquisition, Stream
from sigpipe.transformers import Shift

from paco.qc.g1_signal import (
    SignalThresholds,
    decay_outliers,
    judge_receivers,
    judge_signal,
    judge_spectra,
)
from paco.qc.models import GateResult

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


def test_dead_clipped_and_nan_traces_are_found_and_excluded() -> None:
    stream = _shot()
    stream = _with_trace(stream, 3, np.zeros(stream.xt.shape[1], dtype=np.float32))
    stream = _with_trace(stream, 7, np.clip(stream.xt[7], -0.02, 0.02))
    nan_trace = stream.xt[11].copy()
    nan_trace[100] = np.nan
    stream = _with_trace(stream, 11, nan_trace)

    result = judge_signal("1.dat", stream, THRESHOLDS)

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


def test_an_amplitude_off_the_decay_is_reported_not_left_out() -> None:
    stream = _shot()
    stream = _with_trace(stream, 10, stream.xt[10] * 30)

    result = judge_signal("1.dat", stream, THRESHOLDS)

    # One record's trace off the decay is no bad geophone: the line judges each receiver over
    # every record (judge_receivers).
    assert result.flags == () and result.verdict == "pass"
    metrics = {metric.name: metric for metric in result.metrics}
    assert metrics["rms_outliers"].value == 1 and metrics["rms_outliers"].threshold is None
    outliers, judged = decay_outliers(stream, THRESHOLDS)
    assert np.flatnonzero(outliers).tolist() == [10] and judged.all()


def _metric(result: GateResult, name: str) -> float | None:
    return next(metric.value for metric in result.metrics if metric.name == name)


def test_a_record_with_too_much_noise_has_a_low_snr() -> None:
    loud = _shot(noise=0.05)

    assert judge_signal("1.dat", loud, THRESHOLDS).verdict == "pass"
    weak = _shot(noise=1.0)
    result = judge_signal("1.dat", weak, THRESHOLDS)
    assert "low_snr" in {flag.name for flag in result.flags}


def test_a_reversed_trace_is_not_judged() -> None:
    stream = _shot()
    reversed_stream = _with_trace(stream, 5, -stream.xt[5])

    # The polarity is measured, but not judged: a reversed geophone hardly ever happens, and
    # near the source the check would flag whole blocks of traces.
    assert "reversed_polarity" not in {
        flag.name for flag in judge_signal("1.dat", reversed_stream, THRESHOLDS).flags
    }


def test_leaving_traces_out_adds_no_amplitude_outlier() -> None:
    # The decay is fitted on every trace alive: leaving the nearest six out does not move it,
    # so no new trace falls off it.
    outliers, judged = decay_outliers(_shot(), THRESHOLDS, excluded=range(6))

    assert not outliers.any() and not judged[:6].any() and judged[6:].all()


def test_a_shifted_trigger_asks_for_its_correction() -> None:
    # Muted with no shift (applied_s 0): the shot should be at 0, the first breaks put it 50 ms on.
    shifted = judge_signal("1.dat", _shot(t0=0.05), THRESHOLDS, applied_s=0.0)
    assert shifted.verdict == "retry"
    (flag,) = shifted.flags
    assert flag.name == "shifted_trigger" and flag.fixable
    action = flag.action.model_dump()
    assert action["kind"] == "override" and action["stage"] == "preprocessing"
    assert action["overrides"]["trigger"]["t0"] == pytest.approx(0.05, abs=0.005)
    # Corrected by that t0, the record passes.
    (corrected,) = Shift(t0=action["overrides"]["trigger"]["t0"]).transform([_shot(t0=0.05)])
    assert judge_signal("1.dat", corrected, THRESHOLDS, applied_s=0.05).verdict == "pass"


def test_a_shot_before_the_records_start_sets_the_trigger_to_0() -> None:
    # The trigger never below 0 (the user, 2026-09-28). Moved by 30 ms, the first breaks put the
    # shot 40 ms before the time origin, 10 ms before the record's start: the trigger at 0.
    (flag,) = judge_signal("1.dat", _shot(t0=-0.04), THRESHOLDS, applied_s=0.03).flags
    assert flag.name == "shifted_trigger" and "its trigger set to 0" in flag.message
    assert flag.action.model_dump()["overrides"]["trigger"]["t0"] == 0.0
    # Not moved, the shot 20 ms before the start: nothing to correct, only said.
    before = judge_signal("1.dat", _shot(t0=-0.02), THRESHOLDS, applied_s=0.0)
    (flag,) = before.flags
    assert flag.name == "shifted_trigger" and "No trigger corrects it" in flag.message
    assert flag.action.model_dump()["kind"] == "keep" and before.verdict == "pass"


def test_a_trigger_not_muted_is_only_reported_against_the_files() -> None:
    # The trigger is part of the muting: off, nothing to correct. Where the file says the shot
    # is, the record passes; elsewhere its measure fails, with no retry.
    assert judge_signal("1.dat", _shot(t0=0.05), THRESHOLDS, shot_s=0.05).verdict == "pass"
    off = judge_signal("1.dat", _shot(t0=0.05), THRESHOLDS)
    assert off.verdict == "pass" and not off.flags
    (metric,) = [metric for metric in off.metrics if metric.name == "trigger_shift_s"]
    assert not metric.passed and metric.value == pytest.approx(0.05, abs=0.005)


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


# ---------------------------------------------------------------- the line's reach


def test_the_shots_pulse_is_measured_on_a_record_not_muted() -> None:
    # The wavelet's energy falls to a tenth of its peak about 120 ms after the first break, on
    # the traces nearest the shot. Muted, the record's pulse is not measured: the mute cuts it.
    pulse = _metric(judge_signal("1.dat", _shot(), THRESHOLDS), "pulse_s")
    assert pulse == pytest.approx(0.12, abs=0.015)
    muted = judge_signal("1.dat", _shot(), THRESHOLDS, applied_s=0.0)
    assert "pulse_s" not in {metric.name for metric in muted.metrics}


def test_a_record_is_judged_within_the_reach() -> None:
    # The far two thirds of the line carry only noise, as on a long line: judged on every trace
    # the median SNR fails; within the reach, the near traces carry the wave.
    shot = _shot()
    rng = np.random.default_rng(1)
    xt = shot.xt.copy()
    xt[8:] = (rng.standard_normal(xt[8:].shape) * 0.02).astype(np.float32)
    far_noise = replace(shot, xt=xt)
    offsets = np.asarray(far_noise.acquisition.offsets)

    whole = judge_signal("1.dat", far_noise, THRESHOLDS)
    near = judge_signal("1.dat", far_noise, THRESHOLDS, reach_m=float(offsets[7]))

    assert "low_snr" in {flag.name for flag in whole.flags}
    assert "low_snr" not in {flag.name for flag in near.flags}
    assert "low_coherence" in {flag.name for flag in whole.flags}
    assert near.flags == ()


def test_the_decay_is_fitted_within_the_reach() -> None:
    # Six traces carry the wave, eighteen only noise: over the whole line the noise floor
    # flattens the decay, and the six read too loud (active_p2's nearest traces).
    shot = _shot()
    rng = np.random.default_rng(1)
    xt = shot.xt.copy()
    xt[6:] = (rng.standard_normal(xt[6:].shape) * 0.02).astype(np.float32)
    far_noise = replace(shot, xt=xt)
    offsets = np.asarray(far_noise.acquisition.offsets)

    whole, _ = decay_outliers(far_noise, THRESHOLDS)
    near, judged = decay_outliers(far_noise, THRESHOLDS, reach_m=float(offsets[5]))

    assert np.flatnonzero(whole).tolist() == [0, 1, 2, 3, 4, 5]
    assert not near.any() and np.flatnonzero(judged).tolist() == [0, 1, 2, 3, 4, 5]
    assert judge_signal(
        "1.dat", far_noise, THRESHOLDS, reach_m=float(offsets[5])
    ).kept.n_traces == (N_TRACES)


POSITIONS = [1.5 * i for i in range(N_TRACES)]


def test_a_receiver_off_the_decay_in_most_records_leaves_every_window() -> None:
    # Receiver 10 off in 5 of the 8 records that reach it; receiver 3 in 1 of 8 (the record
    # whose shot stands next to it); receiver 4 in none.
    result = judge_receivers({10: 5, 3: 1}, {10: 8, 3: 8, 4: 8}, POSITIONS, THRESHOLDS)

    assert result.unit == "line" and result.verdict == "pass"
    (flag,) = result.flags
    assert flag.name == "off_decay_receivers" and not flag.fixable
    assert flag.action.model_dump() == {"kind": "exclude_traces", "record": "line", "traces": (10,)}
    assert flag.message.startswith(
        "The receiver at 15 m (5 of 8 records) is too weak or too strong for its distance"
    )


def test_a_receiver_off_in_a_few_records_stays() -> None:
    # Off in 3 of 8 records, under half: those records' own. Off in 2 of 2: too few to judge.
    result = judge_receivers({3: 3, 7: 2}, {3: 8, 7: 2}, POSITIONS, THRESHOLDS)

    assert result.flags == ()
    assert result.metrics[0].value == 0 and result.metrics[0].passed
    several = judge_receivers({3: 4, 9: 8}, {3: 8, 9: 8}, POSITIONS, THRESHOLDS)
    assert several.flags[0].message.startswith(
        "The receivers at 4.5 m (4 of 8 records) and 13.5 m (8 of 8 records) are too weak or"
    )


def _noise(notched: int | None = None) -> Stream:
    """A passive record: white noise on every trace, 20 s; `notched`, a trace with nothing
    between 20 and 120 Hz (a dead band, a quarter of the band compared)."""
    rng = np.random.default_rng(3)
    xt = rng.standard_normal((N_TRACES, int(20 * SAMPLING)))
    if notched is not None:
        spectrum = np.fft.rfft(xt[notched])
        fs = np.fft.rfftfreq(xt.shape[1], d=1 / SAMPLING)
        spectrum[(fs >= 20) & (fs <= 120)] = 0
        xt[notched] = np.fft.irfft(spectrum, n=xt.shape[1])
    receivers = tuple(Coordinate(i * DX, 0.0, 0.0) for i in range(N_TRACES))
    acquisition = LinearAcquisition(source=receivers[0], receivers=receivers)
    ts = np.arange(xt.shape[1]) / SAMPLING
    return Stream(xt=xt.astype(np.float32), ts=ts.astype(np.float32),
                  sampling_freq=SAMPLING, acquisition=acquisition)  # fmt: skip


def test_a_noise_records_trace_off_its_neighbours_spectra_is_counted() -> None:
    # On a passive line (the user, 2026-09-28): reported, nothing left out.
    clean = judge_signal("1.dat", _noise(), THRESHOLDS, active=False, spectra=True)
    notched = judge_signal("1.dat", _noise(notched=7), THRESHOLDS, active=False, spectra=True)

    assert _metric(clean, "spectral_outliers") == 0
    assert _metric(notched, "spectral_outliers") == 1
    assert notched.verdict == "pass" and notched.kept.n_traces == N_TRACES


def test_a_receiver_off_its_neighbours_spectra_in_most_records_is_flagged_and_kept() -> None:
    positions = [i * DX for i in range(N_TRACES)]
    # Receiver 7 off in 3 of the 4 records judging it, receiver 2 in 1 of 4.
    metric, flag = judge_spectra({7: 3, 2: 1}, dict.fromkeys(range(N_TRACES), 4), positions,
                                 THRESHOLDS)  # fmt: skip

    assert metric.value == 1 and not metric.passed
    assert flag is not None and flag.action.model_dump()["kind"] == "keep"
    assert "7 m (3 of 4 records)" in flag.message
    assert judge_spectra({2: 1}, {2: 4}, positions, THRESHOLDS)[1] is None
