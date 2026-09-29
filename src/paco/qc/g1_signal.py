"""G1, the signal QC of a preprocessed record (docs/qc_workflow.md): dead, clipped and NaN
traces; amplitudes off the decay with offset; the SNR and the usable band,
from a surface-wave window against a noise window; lateral coherence; the trigger; and what a
mute removed. On a passive record, only the checks that need no trigger. The measures are
sigpipe's (sigpipe.masw.quality.signal); G1 judges them. A trace off the decay in one record is
reported, not left out: G1 over the line (`judge_receivers`) leaves out the receivers off it in
most of the records that reach them, a bad geophone."""

import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.base import Stream
from sigpipe.masw.quality.signal import (
    Windows,
    dead_clipped_nan,
    first_breaks,
    lateral_coherence,
    pulse_durations,
    rms_decay_outliers,
    signal_windows,
    snr_db,
    spectral_deviations,
    trigger_shift,
    usable_band,
)

from paco.qc.models import (
    ExcludeRecord,
    ExcludeTraces,
    Flag,
    GateResult,
    Keep,
    Kept,
    Metric,
    Override,
)

GATE = "G1"
LINE = "line"  # the unit of the line-level result, as G4's
# The modes whose records' traces are compared with their neighbours' spectra (the user,
# 2026-09-28): a noise record's, and a shot's correlated whole.
SPECTRA_MODES = frozenset({"passive", "passive-active"})


class SignalThresholds(BaseModel):
    """G1's window and limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vg_min: float = Field(default=80.0, gt=0, description="m/s, the slowest surface wave kept")
    vg_max: float = Field(default=1500.0, gt=0, description="m/s, the fastest arrival kept")
    pad_s: float = Field(default=0.05, ge=0, description="s, added on both sides of the window")
    mute_width_s: float = Field(
        default=0.05,
        ge=0,
        description="s, kept after the slowest arrival by a mute when no record's pulse could be "
        "measured (each record's own, else the line's median, first)",
    )
    pulse_ratio: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="The shot's pulse ends where its envelope falls back below this share of its "
        "peak",
    )
    pulse_traces: int = Field(
        default=3, ge=1, description="Traces nearest the shot whose pulses give the record's"
    )
    max_pulse_s: float = Field(
        default=0.5, gt=0, description="s, the longest pulse searched: longer, not measured"
    )
    # A noise record's traces, and a shot's (passive-active), against their neighbours' spectra
    # (the user, 2026-09-28): flagged, never left out. On the demo, a passive trace sits 1 dB
    # from its neighbours (3.4 at the 99th percentile); a shot's, its own level taken out and
    # the five nearest the shot aside, 1.7 (5.6).
    spectra_fmin_hz: float = Field(
        default=2.0, ge=0, description="Hz, where the traces' spectra are compared from"
    )
    spectra_fmax_share: float = Field(
        default=0.8, gt=0, le=1, description="Share of Nyquist the spectra are compared up to"
    )
    spectra_neighbours: int = Field(
        default=2, ge=1, description="Traces on each side a trace's spectrum is compared with"
    )
    max_spectral_deviation_db: float = Field(
        default=6.0,
        gt=0,
        description="dB, a trace's spectrum's median distance from its neighbours', at most: "
        "beyond, a gain or a response of its own",
    )
    dead_band_db: float = Field(
        default=15.0, gt=0, description="dB under the neighbours' that makes a frequency dead"
    )
    max_dead_band_share: float = Field(
        default=0.1, gt=0, le=1, description="Share of the band a trace may have dead, at most"
    )
    spectra_near_traces: int = Field(
        default=5,
        ge=0,
        description="Traces nearest a shot left out of the comparison: louder and brighter by "
        "their distance alone",
    )
    spectra_line_share: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Share of the records reaching it in which a receiver is off its "
        "neighbours' spectra, for the line to flag it",
    )
    dead_ratio: float = Field(
        default=0.01,
        gt=0,
        description="A trace whose RMS is below this share of the median is dead.",
    )
    clip_share: float = Field(
        default=0.005,
        gt=0,
        description="Share of samples at a trace's extreme that means clipping.",
    )
    rms_outlier_mads: float = Field(
        default=3.0, gt=0, description="RMS off the decay with offset by this many MADs, in log."
    )
    rms_outlier_factor: float = Field(
        default=2.0,
        gt=1,
        description="...and by at least this factor: a smooth decay has tiny MADs.",
    )
    rms_outlier_share: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="A receiver off the decay in at least this share of the records that reach "
        "it is left out of every window: a bad geophone. Off in fewer, the records' own.",
    )
    rms_outlier_min_records: int = Field(
        default=3, ge=1, description="...over at least this many records."
    )
    min_snr_db: float = Field(default=6.0, description="Median SNR of the traces, in dB.")
    reach_snr_db: float = Field(
        default=2.0,
        description="dB: the traces' median SNR, by distance from the shot, under which they carry "
        "no wave: the line's reach, which sets masw.distance_max.",
    )
    band_db: float = Field(
        default=6.0, gt=0, description="Signal above noise, for the usable band."
    )
    peak_db: float = Field(
        default=20.0,
        gt=0,
        description="...and within this much of the signal spectrum's peak: after the wave, the\n"
        "noise window is quieter than the signal window at every frequency.",
    )
    min_coherence: float = Field(
        default=0.5, ge=0, le=1, description="Median correlation of neighbouring traces."
    )
    max_trigger_shift_s: float = Field(
        default=0.01,
        ge=0,
        description="s, the first breaks' shot time off where it should be: 0 once the muting "
        "moved the time origin, the file's trigger otherwise.",
    )
    max_trigger_scatter_s: float = Field(
        default=0.05,
        ge=0,
        description="s, the first breaks' scatter about their line: beyond it they are no line, "
        "and no correction fixes the record.",
    )
    first_break_ratio: float = Field(
        default=5.0, gt=0, description="A first break is the first sample above this x noise RMS."
    )


def judge_signal(
    record: str,
    preprocessed: Stream,
    thresholds: SignalThresholds,
    raw: Stream | None = None,
    active: bool = True,
    excluded: Collection[int] = (),
    reach_m: float | None = None,
    shot_s: float = 0.0,
    applied_s: float | None = None,
    spectra: bool = False,
    before_muting: tuple[Stream, float] | None = None,
) -> GateResult:
    """G1's verdict on one record: the metrics, the flags with their actions, and what is kept.
    The traces `excluded` already (receiver indices) are left out of every measure and flag;
    the decay with offset, the SNR, the usable band and the lateral coherence are measured on
    the traces within `reach_m` of the shot (the line's reach, beyond which the traces carry no
    wave). The first breaks should put the shot at `shot_s` (`trigger_context`); `applied_s`,
    the shift the muting applied, None with the muting off: the trigger is part of the muting,
    so a record not muted has no trigger to correct, its shot time only reported. The SNR and
    the usable band measure the noise on the record before its muting (`before_muting`: it, and
    where its shot is on it; the user, 2026-09-29: a muting zeroes the noise window after the
    slowest arrival), when muted; the other measures, `preprocessed`."""
    xt = preprocessed.xt
    left_out = _left_out(xt.shape[0], excluded)
    dead, clipped, nan = (
        mask & ~left_out
        for mask in dead_clipped_nan(xt, thresholds.dead_ratio, thresholds.clip_share)
    )
    bad_traces = dead | clipped | nan | left_out
    metrics: list[Metric] = [
        Metric(
            name="dead_traces",
            value=int(dead.sum()),
            threshold=0,
            bound="max",
            passed=not dead.any(),
        ),
        Metric(
            name="clipped_traces",
            value=int(clipped.sum()),
            threshold=0,
            bound="max",
            passed=not clipped.any(),
        ),
        Metric(
            name="nan_traces", value=int(nan.sum()), threshold=0, bound="max", passed=not nan.any()
        ),
    ]
    flags: list[Flag] = []
    for name, mask, what in (
        ("dead_traces", dead, "dead"),
        ("clipped_traces", clipped, "clipped"),
        ("nan_traces", nan, "not a number"),
    ):
        if mask.any():
            traces = tuple(int(i) for i in np.flatnonzero(mask))
            flags.append(
                Flag(
                    name=name,
                    message=f"Traces {list(traces)} are {what}: no parameter improves them.",
                    stage="preprocessing",
                    action=ExcludeTraces(record=record, traces=traces),
                    fixable=False,
                )
            )
    kept = Kept(n_traces=int((~bad_traces).sum()))
    # With `spectra` (passive and passive-active lines): the traces off their neighbours'
    # spectra, reported; the line flags a receiver off in most records (judge_spectra).
    if spectra:
        off_spectra, _ = spectral_outliers(preprocessed, thresholds, excluded, shots=active)
        metrics.append(Metric(name="spectral_outliers", value=int(off_spectra.sum()), passed=True))
    if not active:
        return _result(record, metrics, flags, kept)

    offsets = np.asarray(preprocessed.acquisition.offsets, dtype=float)
    # Times from the shot where it should be: its windows on its arrivals, muted or not.
    ts = np.asarray(preprocessed.ts, dtype=float) - shot_s
    windows = signal_windows(
        offsets,
        ts,
        thresholds.vg_min,
        thresholds.vg_max,
        thresholds.pad_s,
    )
    if windows is None:
        flags.append(
            Flag(
                name="record_too_short",
                message="The record ends before the slowest arrival leaves room for a noise "
                "window: no SNR, no usable band.",
                stage="preprocessing",
                action=ExcludeRecord(record=record),
                fixable=False,
            )
        )
        return _result(record, metrics, flags, kept)

    finite = np.nan_to_num(xt)
    rms = np.sqrt(np.mean(finite**2, axis=1))
    within = _within(offsets, reach_m)
    # Reported, not left out: a trace off the decay in this record alone (the one nearest the
    # shot, where the fitted decay overshoots; a burst of noise) is no bad geophone. The line
    # judges each receiver over every record (judge_receivers). The record's own measures leave
    # them out.
    outliers, _ = _off_decay(rms, offsets, ~(dead | clipped | nan), within, left_out, thresholds)
    metrics.append(Metric(name="rms_outliers", value=int(outliers.sum()), passed=True))

    usable = ~(bad_traces | outliers)
    measured = usable & within if (usable & within).any() else usable
    noisy, noise = _before_muting(before_muting, finite, windows, offsets, thresholds)
    snr = snr_db(noisy, noise)
    median_snr = float(np.median(snr[measured])) if measured.any() else float("nan")
    snr_ok = median_snr >= thresholds.min_snr_db
    metrics.append(
        Metric(
            name="snr_db",
            value=_finite(median_snr),
            threshold=thresholds.min_snr_db,
            bound="min",
            passed=snr_ok,
            unit="dB",
        )
    )
    band = (
        usable_band(
            noisy[measured],
            preprocessed.sampling_freq,
            Windows(noise.signal[measured], noise.noise[measured], noise.where),
            thresholds.band_db,
            thresholds.peak_db,
        )
        if measured.any()
        else None
    )
    metrics.append(
        Metric(
            name="usable_band_hz",
            value=None if band is None else band[1] - band[0],
            threshold=0,
            bound="min",
            passed=band is not None,
            unit="Hz",
        )
    )
    if not snr_ok or band is None:
        flags.append(
            Flag(
                name="low_snr",
                message=f"Median SNR {median_snr:.1f} dB in the surface-wave window ({noise.where})"
                + (
                    f"; usable band {band[0]:.1f}-{band[1]:.1f} Hz."
                    if band
                    else "; no usable band."
                ),
                stage="preprocessing",
                action=Override(
                    stage="preprocessing",
                    overrides={"filtering": {"method": "iir", "fmin": band[0], "fmax": band[1]}}
                    if band
                    else {
                        "muting": {
                            "method": "mute",
                            "vmin": thresholds.vg_min,
                            "vmax": thresholds.vg_max,
                        }
                    },
                ),
            )
        )

    max_lag_s = (
        float(np.diff(np.sort(offsets)).min()) / thresholds.vg_min if offsets.size > 1 else 0.0
    )
    coherence, _ = lateral_coherence(finite, windows, preprocessed.sampling_freq, max_lag_s)
    # Within the reach too: pairs of noise traces beyond it are not the record's fault.
    pair_ok = measured[:-1] & measured[1:]
    median_coherence = float(np.median(coherence[pair_ok])) if pair_ok.any() else float("nan")
    coherence_ok = median_coherence >= thresholds.min_coherence
    metrics.append(
        Metric(
            name="lateral_coherence",
            value=_finite(median_coherence),
            threshold=thresholds.min_coherence,
            bound="min",
            passed=coherence_ok,
        )
    )
    if not coherence_ok:
        flags.append(
            Flag(
                name="low_coherence",
                message=f"Neighbouring traces correlate at {median_coherence:.2f} in the surface-wave window: noisy or scattered.",
                stage="preprocessing",
                action=Override(
                    stage="preprocessing",
                    overrides={
                        "muting": {
                            "method": "mute",
                            "vmin": thresholds.vg_min,
                            "vmax": thresholds.vg_max,
                        }
                    },
                ),
            )
        )
    # No reversed-polarity check: a reversed geophone hardly ever happens, and near the source,
    # neighbours shifted by more than half a period look reversed.

    breaks = first_breaks(finite, ts, windows, thresholds.first_break_ratio)
    # Only the traces whose own SNR passes: a noisy trace's envelope crosses the threshold on
    # noise, early or late.
    breaks[~usable | (snr < thresholds.min_snr_db)] = np.nan
    # The shot's pulse (the user, 2026-09-28): how long its energy lasts after the first break,
    # the median over the traces nearest the shot that show one; the width a mute keeps after
    # the slowest arrival (shots.py). Measured on a record not muted: a mute would cut it.
    if applied_s is None:
        pulses = pulse_durations(finite, ts, breaks, thresholds.pulse_ratio, thresholds.max_pulse_s)
        nearest = [i for i in np.argsort(offsets) if np.isfinite(pulses[i])]
        pulse = float(np.median(pulses[nearest[: thresholds.pulse_traces]])) if nearest else None
        metrics.append(Metric(name="pulse_s", value=_finite(pulse), passed=True, unit="s"))
    fit = trigger_shift(breaks, offsets)
    # Without four first breaks, or on a noisy record whose breaks come late, the trigger is
    # not measured, not wrong.
    shift = None if fit is None or not snr_ok else fit[0]
    scatter = None if fit is None or not snr_ok else fit[2]
    consistent = scatter is None or scatter <= thresholds.max_trigger_scatter_s
    # Where the first breaks put the shot, off where it should be (times from it).
    off = shift
    shift_ok = off is None or abs(off) <= thresholds.max_trigger_shift_s
    metrics.append(
        Metric(
            name="trigger_shift_s",
            value=_finite(off),
            threshold=thresholds.max_trigger_shift_s,
            bound="max",
            passed=shift_ok,
            unit="s",
        )
    )
    metrics.append(
        Metric(
            name="trigger_scatter_s",
            value=_finite(scatter),
            threshold=thresholds.max_trigger_scatter_s,
            bound="max",
            passed=consistent,
            unit="s",
        )
    )
    if shift is not None and scatter is not None and not consistent:
        flags.append(
            Flag(
                name="inconsistent_trigger",
                message=f"The first breaks scatter by {scatter * 1000:.0f} ms about their line: the "
                "delay differs from trace to trace, which no correction fixes.",
                stage="preprocessing",
                action=ExcludeRecord(record=record),
                fixable=False,
            )
        )
    elif shift is not None and off is not None and not shift_ok and applied_s is not None:
        # The trigger never below 0 (the user, 2026-09-28): a shot before the record's start, the
        # recording started late; the trigger at 0 and what is left said, or, at 0 already, said
        # alone.
        t0 = round(applied_s + off, 4)
        said = (
            f"The first breaks put the shot {off * 1000:+.0f} ms from the time origin the muting "
            "moved, the same on every trace"
        )
        early = f"{-t0 * 1000:.0f} ms before the record starts: it began after the shot"
        flags.append(
            Flag(
                name="shifted_trigger",
                message=f"{said}: correct its trigger by them."
                if t0 >= 0
                else f"{said}, {early}; its trigger set to 0."
                if applied_s > 0
                else f"{said}, {early}. No trigger corrects it.",
                stage="preprocessing",
                action=Override(stage="preprocessing", overrides={"trigger": {"t0": max(t0, 0.0)}})
                if t0 >= 0 or applied_s > 0
                else Keep(note="the shot before the record's start"),
            )
        )

    if raw is not None:
        removed = 1 - float(np.sum(finite**2) / max(np.sum(np.nan_to_num(raw.xt) ** 2), 1e-30))
        metrics.append(Metric(name="energy_removed", value=round(removed, 3), passed=True))

    kept = Kept(band_hz=band, n_traces=int(usable.sum()))
    return _result(record, metrics, flags, kept)


def _before_muting(
    before_muting: tuple[Stream, float] | None,
    finite: np.ndarray,
    windows: Windows,
    offsets: np.ndarray,
    thresholds: SignalThresholds,
) -> tuple[np.ndarray, Windows]:
    """The traces and windows G1 measures the noise on: the record before its muting's, its
    times from its shot (`before_muting`); the preprocessed record's (`finite`, `windows`) when
    not muted, or when the record before its muting leaves no room for a noise window."""
    if before_muting is None:
        return finite, windows
    stream, shot_s = before_muting
    found = signal_windows(
        offsets,
        np.asarray(stream.ts, dtype=float) - shot_s,
        thresholds.vg_min,
        thresholds.vg_max,
        thresholds.pad_s,
    )
    if found is None or stream.xt.shape != finite.shape:
        return finite, windows
    return np.nan_to_num(stream.xt), found


def decay_outliers(
    preprocessed: Stream,
    thresholds: SignalThresholds,
    excluded: Collection[int] = (),
    reach_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The record's traces off the decay with offset (receiver indices as a mask), and those
    judged for it: alive, within `reach_m` of the shot, not `excluded`."""
    xt = preprocessed.xt
    left_out = _left_out(xt.shape[0], excluded)
    dead, clipped, nan = dead_clipped_nan(xt, thresholds.dead_ratio, thresholds.clip_share)
    offsets = np.asarray(preprocessed.acquisition.offsets, dtype=float)
    rms = np.sqrt(np.mean(np.nan_to_num(xt) ** 2, axis=1))
    return _off_decay(
        rms, offsets, ~(dead | clipped | nan), _within(offsets, reach_m), left_out, thresholds
    )


def spectral_outliers(
    stream: Stream,
    thresholds: SignalThresholds,
    excluded: Collection[int] = (),
    shots: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """The traces off their neighbours' spectra (a gain or a response of their own, a dead
    band), and those judged: a noise record's by level and shape, a shot's by shape alone with
    the traces nearest its shot aside; the traces `excluded`, dead, clipped or NaN neither
    judged nor compared with."""
    xt = stream.xt
    dead, clipped, nan = dead_clipped_nan(xt, thresholds.dead_ratio, thresholds.clip_share)
    usable = ~(dead | clipped | nan | _left_out(xt.shape[0], excluded))
    if shots:
        offsets = np.asarray(stream.acquisition.offsets, dtype=float)
        usable[np.argsort(offsets)[: thresholds.spectra_near_traces]] = False
    band = (
        thresholds.spectra_fmin_hz,
        thresholds.spectra_fmax_share * stream.sampling_freq / 2,
    )
    deviation, dropped = spectral_deviations(
        np.nan_to_num(np.asarray(xt, dtype=float)),
        stream.sampling_freq,
        band,
        thresholds.spectra_neighbours,
        thresholds.dead_band_db,
        usable,
        shape=shots,
    )
    judged = np.isfinite(deviation)
    off = judged & (
        (deviation > thresholds.max_spectral_deviation_db)
        | (dropped > thresholds.max_dead_band_share)
    )
    return off, judged


def judge_spectra(
    off: Mapping[int, int],
    reached: Mapping[int, int],
    positions: Sequence[float],
    thresholds: SignalThresholds,
) -> tuple[Metric, Flag | None]:
    """G1 over the line, a passive or passive-active one: the receivers off their neighbours'
    spectra (`off`: in how many records) in at least `spectra_line_share` of the records judging
    them (`reached`). Flagged, kept (the user, 2026-09-28: flag, not alter the workflow)."""
    bad = [
        receiver
        for receiver, count in sorted(reached.items())
        if count and off.get(receiver, 0) >= math.ceil(thresholds.spectra_line_share * count)
    ]
    metric = Metric(
        name="spectral_receivers", value=len(bad), threshold=0, bound="max", passed=not bad
    )
    if not bad:
        return metric, None
    each = [f"{positions[r]:g} m ({off[r]} of {reached[r]} records)" for r in bad]
    which, its = (
        (f"The receiver at {each[0]} is", "its")
        if len(bad) == 1
        else (f"The receivers at {', '.join(each[:-1])} and {each[-1]} are", "their")
    )
    return metric, Flag(
        name="spectral_receivers",
        message=f"{which} off {its} neighbours' spectra in most of the records that reach "
        f"{'it' if len(bad) == 1 else 'them'}: a dead band, a gain or a response of {its} own. "
        "Kept, to look at.",
        stage="preprocessing",
        action=Keep(note="a geophone of its own"),
        fixable=False,
    )


def judge_receivers(
    off: Mapping[int, int],
    reached: Mapping[int, int],
    positions: Sequence[float],
    thresholds: SignalThresholds,
) -> GateResult:
    """G1 over the line: the receivers off the decay with offset (`off`: in how many records)
    in at least `rms_outlier_share` of the records that reach them (`reached`), and in
    `rms_outlier_min_records` at least, left out of every window: a noisy or badly coupled
    geophone. `positions`: each receiver's x."""
    bad = [
        receiver
        for receiver, count in sorted(reached.items())
        if count >= thresholds.rms_outlier_min_records
        and off.get(receiver, 0) >= math.ceil(thresholds.rms_outlier_share * count)
    ]
    metrics = [
        Metric(
            name="off_decay_receivers",
            value=len(bad),
            threshold=0,
            bound="max",
            passed=not bad,
        )
    ]
    flags: list[Flag] = []
    if bad:
        each = [
            f"{positions[receiver]:g} m ({off[receiver]} of {reached[receiver]} records)"
            for receiver in bad
        ]
        which, its = (
            (f"The receiver at {each[0]} is", "its")
            if len(bad) == 1
            else (f"The receivers at {', '.join(each[:-1])} and {each[-1]} are", "their")
        )
        flags.append(
            Flag(
                name="off_decay_receivers",
                message=f"{which} too weak or too strong for {its} distance from the shot (off "
                "the amplitude decay with offset) in most of the records that reach "
                f"{'it' if len(bad) == 1 else 'them'}: a noisy or badly coupled geophone, left "
                "out of every window.",
                stage="preprocessing",
                action=ExcludeTraces(record=LINE, traces=tuple(bad)),
                fixable=False,
            )
        )
    # The receivers left out, the line's records stand.
    return GateResult(
        gate=GATE, unit=LINE, verdict="pass", metrics=tuple(metrics), flags=tuple(flags)
    )


def _off_decay(
    rms: np.ndarray,
    offsets: np.ndarray,
    alive: np.ndarray,
    within: np.ndarray,
    left_out: np.ndarray,
    thresholds: SignalThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    """The traces off the decay with offset, and those judged. The decay is fitted on every
    trace `alive` within the reach, those already left out among them: excluding a trace must
    not move the fit, or each round excludes more. Within the reach only: over the whole line
    the noise floor flattens the fit, and the traces nearest the shots, the strongest, read too
    loud."""
    fit_on = alive & within
    outliers = (
        rms_decay_outliers(
            rms, offsets, fit_on, thresholds.rms_outlier_mads, thresholds.rms_outlier_factor
        )
        & ~left_out
        & within
    )
    return outliers, fit_on & ~left_out


def _within(offsets: np.ndarray, reach_m: float | None) -> np.ndarray:
    """The traces within the line's reach: beyond it they carry no wave, and no window stacks
    them."""
    if reach_m is not None and (offsets <= reach_m).any():
        return offsets <= reach_m
    return np.ones(offsets.size, dtype=bool)


def _left_out(n_traces: int, excluded: Collection[int]) -> np.ndarray:
    left_out = np.zeros(n_traces, dtype=bool)
    left_out[[index for index in excluded if 0 <= index < n_traces]] = True
    return left_out


def _result(record: str, metrics: list[Metric], flags: list[Flag], kept: Kept) -> GateResult:
    # A kept flag is information (a shot before the record's start): it never changes the
    # verdict, as G2's and G3's.
    acted = [flag for flag in flags if not isinstance(flag.action, Keep)]
    if any(not flag.fixable and isinstance(flag.action, ExcludeRecord) for flag in acted):
        verdict = "reject"
    elif acted:
        verdict = "retry"
    else:
        verdict = "pass"
    return GateResult(
        gate=GATE,
        unit=record,
        verdict=verdict,
        metrics=tuple(metrics),
        flags=tuple(flags),
        kept=kept,
    )


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, 4)
