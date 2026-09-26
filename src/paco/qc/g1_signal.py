"""G1, the signal QC of a preprocessed record (docs/qc_workflow.md): dead, clipped and NaN
traces; amplitudes off the decay with offset; the SNR and the usable band,
from a surface-wave window against a noise window; lateral coherence; the trigger; and what a
mute removed. On a passive record, only the checks that need no trigger."""

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, hilbert
from scipy.stats import theilslopes
from sigpipe.base import Stream

from paco.qc.models import ExcludeRecord, ExcludeTraces, Flag, GateResult, Kept, Metric, Override

GATE = "G1"


class SignalThresholds(BaseModel):
    """G1's window and limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vg_min: float = Field(default=80.0, gt=0, description="m/s, the slowest surface wave kept")
    vg_max: float = Field(default=1500.0, gt=0, description="m/s, the fastest arrival kept")
    pad_s: float = Field(default=0.05, ge=0, description="s, added on both sides of the window")
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
    min_snr_db: float = Field(default=6.0, description="Median SNR of the traces, in dB.")
    reach_snr_db: float = Field(
        default=2.0,
        description="dB: the traces' median SNR, by distance from the shot, under which they carry "
        "no wave: the line's reach, which sets masw.distance_max (the user's choice of "
        "2026-09-25: at 6 dB the demo's end windows lost a far shot that helped them).",
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
        default=0.01, ge=0, description="s, the first breaks' time at zero offset."
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


@dataclass(frozen=True)
class Windows:
    """The samples of each trace in the surface-wave window and in the noise window."""

    signal: np.ndarray  # bool, (n_traces, n_samples)
    noise: np.ndarray  # bool, (n_traces, n_samples)
    where: str  # which noise window was used


def signal_windows(
    offsets: np.ndarray, ts: np.ndarray, thresholds: SignalThresholds
) -> Windows | None:
    """The surface-wave window of each trace, between the arrivals at vg_max and vg_min, and a
    noise window: before the trigger when the record has samples there, else after the slowest
    arrival when the record is long enough; None when it is not."""
    starts = offsets / thresholds.vg_max - thresholds.pad_s
    stops = offsets / thresholds.vg_min + thresholds.pad_s
    signal = (ts[None, :] >= starts[:, None]) & (ts[None, :] <= stops[:, None])
    if ts[0] < -thresholds.pad_s:
        noise = np.broadcast_to(ts < 0.0, signal.shape)
        return Windows(signal, np.array(noise), "before the trigger")
    after = float(stops.max())
    if ts[-1] - after < (stops - starts).mean():
        return None
    noise = np.broadcast_to(ts > after, signal.shape)
    return Windows(signal, np.array(noise), f"after the slowest arrival, {after:.2f} s on")


def dead_clipped_nan(xt: np.ndarray, thresholds: SignalThresholds) -> tuple[np.ndarray, ...]:
    """Masks of the dead, clipped and NaN traces."""
    nan = np.isnan(xt).any(axis=1)
    finite = np.nan_to_num(xt)
    rms = np.sqrt(np.mean(finite**2, axis=1))
    dead = rms < thresholds.dead_ratio * np.median(rms[~nan]) if (~nan).any() else rms == 0
    peak = np.abs(finite).max(axis=1, keepdims=True)
    at_peak = np.abs(finite) >= (1 - 1e-3) * np.where(peak > 0, peak, 1.0)
    clipped = (at_peak.mean(axis=1) > thresholds.clip_share) & ~dead
    return dead, clipped, nan


def rms_decay_outliers(
    rms: np.ndarray, offsets: np.ndarray, usable: np.ndarray, mads: float, factor: float = 2.0
) -> np.ndarray:
    """Traces whose RMS is far from the fit of log RMS against log offset (the decay with
    distance): by more than `mads` median absolute deviations, and by at least `factor`; among
    the usable traces."""
    outliers = np.zeros(rms.size, dtype=bool)
    fit_on = usable & (rms > 0) & (offsets > 0)
    if fit_on.sum() < 4:
        return outliers
    x, y = np.log(offsets[fit_on]), np.log(rms[fit_on])
    fit = np.asarray(theilslopes(y, x), dtype=float)  # robust: one bad trace does not tilt it
    slope, intercept = fit[0], fit[1]
    residuals = y - (slope * x + intercept)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    limit = max(mads * 1.4826 * mad, math.log(factor))
    outliers[fit_on] = np.abs(residuals - np.median(residuals)) > limit
    return outliers


def trace_snrs(
    stream: Stream, thresholds: SignalThresholds
) -> tuple[np.ndarray, np.ndarray] | None:
    """Each trace's distance from the shot and its SNR in dB (surface-wave window against noise
    window); None when the record leaves no room for a noise window."""
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    windows = signal_windows(offsets, np.asarray(stream.ts, dtype=float), thresholds)
    if windows is None:
        return None
    return offsets, snr_db(np.nan_to_num(stream.xt), windows)


def snr_reach(
    measured: Sequence[tuple[np.ndarray, np.ndarray]], min_db: float, bin_m: float
) -> float | None:
    """How far from the shot the traces still carry the wave: over every record's traces
    `measured` (offsets, SNR), binned every `bin_m` m from the shot, the distance where the
    bins' median SNR first falls below `min_db`, between the centres of the last bin above and
    the first below; None when no bin falls below. Measured on 2026-09-25 at 2 dB: 63.35 m on
    the 142.5 m active_p2 (53 dB at 0-10 m, 5 dB at 50-60 m), 24.34 m on the 24 m demo."""
    if not measured or bin_m <= 0:
        return None
    offsets = np.concatenate([one[0] for one in measured])
    snrs = np.concatenate([one[1] for one in measured])
    bins = np.floor(offsets / bin_m).astype(int)
    previous: tuple[float, float] | None = None
    for index in range(int(bins.max()) + 1):
        values = snrs[bins == index]
        if values.size == 0:
            continue
        centre, median = (index + 0.5) * bin_m, float(np.median(values))
        if median < min_db:
            if previous is None:
                return round(centre, 2)
            before, above = previous
            return round(before + (above - min_db) / (above - median) * (centre - before), 2)
        previous = (centre, median)
    return None


def snr_db(xt: np.ndarray, windows: Windows) -> np.ndarray:
    """Each trace's energy per sample in the signal window over the noise window, in dB."""
    signal = _energy_per_sample(xt, windows.signal)
    noise = _energy_per_sample(xt, windows.noise)
    return 10 * np.log10((signal + 1e-30) / (noise + 1e-30))


def usable_band(
    xt: np.ndarray,
    sampling_freq: float,
    windows: Windows,
    band_db: float,
    peak_db: float = 20.0,
) -> tuple[float, float] | None:
    """The band of frequencies, around the signal spectrum's peak, where the signal windows'
    spectrum exceeds the noise windows' by `band_db` and stays within `peak_db` of its peak;
    None when no frequency does."""
    signal = _mean_spectrum(xt, windows.signal, sampling_freq)
    noise = _mean_spectrum(xt, windows.noise, sampling_freq)
    if signal is None or noise is None:
        return None
    freqs, signal_power = signal
    _, noise_power = noise
    above = 10 * np.log10((signal_power + 1e-30) / (noise_power + 1e-30)) > band_db
    above &= 10 * np.log10((signal_power + 1e-30) / (signal_power.max() + 1e-30)) > -peak_db
    if not above.any():
        return None
    peak = int(np.argmax(np.where(above, signal_power, -np.inf)))
    low = peak
    while low > 0 and above[low - 1]:
        low -= 1
    high = peak
    while high < above.size - 1 and above[high + 1]:
        high += 1
    return float(freqs[low]), float(freqs[high])


def lateral_coherence(
    xt: np.ndarray, windows: Windows, sampling_freq: float, max_lag_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """For each pair of neighbouring traces, the peak of their normalized cross-correlation in
    the signal window within `max_lag_s`, and its sign: the coherence, and the polarity."""
    max_lag = max(1, round(max_lag_s * sampling_freq))
    peaks = np.zeros(xt.shape[0] - 1)
    for i in range(xt.shape[0] - 1):
        mask = windows.signal[i] | windows.signal[i + 1]
        a, b = xt[i][mask], xt[i + 1][mask]
        norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if norm == 0:
            continue
        full = correlate(a, b, mode="full") / norm
        centre = a.size - 1
        window = full[max(0, centre - max_lag) : centre + max_lag + 1]
        peaks[i] = window[np.argmax(np.abs(window))]
    return np.abs(peaks), np.sign(peaks)


def first_breaks(xt: np.ndarray, ts: np.ndarray, windows: Windows, ratio: float) -> np.ndarray:
    """Each trace's first break: where its envelope, smoothed over 5 ms, first rises above
    `ratio` times its noise RMS; NaN when it never does."""
    noise_rms = np.sqrt(_energy_per_sample(xt, windows.noise))
    span = max(3, round(0.005 / max(float(ts[1] - ts[0]), 1e-9)))
    envelope = uniform_filter1d(np.abs(hilbert(xt, axis=1)), span, axis=1)
    breaks = np.full(xt.shape[0], np.nan)
    for i, trace in enumerate(envelope):
        above = np.flatnonzero(trace > ratio * noise_rms[i])
        if above.size:
            breaks[i] = ts[above[0]]
    return breaks


def trigger_shift(breaks: np.ndarray, offsets: np.ndarray) -> tuple[float, float, float] | None:
    """The first breaks fitted as t = t0 + offset / v, robustly (Theil-Sen): (t0, v, the
    breaks' scatter about the line, as 1.4826 x their median absolute deviation); None with
    fewer than 4 breaks, or breaks spanning less than half the offsets (a noisy far half fakes
    a shift)."""
    valid = ~np.isnan(breaks)
    if valid.sum() < 4:
        return None
    span = offsets[valid].max() - offsets[valid].min()
    if span < 0.5 * (offsets.max() - offsets.min()):
        return None
    fit = np.asarray(theilslopes(breaks[valid], offsets[valid]), dtype=float)
    slowness, t0 = float(fit[0]), float(fit[1])
    if slowness <= 0:
        return None
    residuals = breaks[valid] - (t0 + slowness * offsets[valid])
    scatter = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    return float(t0), float(1 / slowness), scatter


def judge_signal(
    record: str,
    preprocessed: Stream,
    thresholds: SignalThresholds,
    raw: Stream | None = None,
    active: bool = True,
    excluded: Collection[int] = (),
    reach_m: float | None = None,
) -> GateResult:
    """G1's verdict on one record: the metrics, the flags with their actions, and what is kept.
    The traces `excluded` already (receiver indices) are left out of every measure and flag;
    the decay with offset, the SNR, the usable band and the lateral coherence are measured on
    the traces within `reach_m` of the shot (the line's reach, beyond which the traces carry no
    wave: the user's decisions of 2026-09-25)."""
    xt = preprocessed.xt
    left_out = np.zeros(xt.shape[0], dtype=bool)
    left_out[[index for index in excluded if 0 <= index < xt.shape[0]]] = True
    dead, clipped, nan = (mask & ~left_out for mask in dead_clipped_nan(xt, thresholds))
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
    if not active:
        return _result(record, metrics, flags, kept)

    offsets = np.asarray(preprocessed.acquisition.offsets, dtype=float)
    windows = signal_windows(offsets, np.asarray(preprocessed.ts, dtype=float), thresholds)
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
    # The line's reach: beyond it the traces carry no wave, and no window stacks them.
    within = np.ones(offsets.size, dtype=bool)
    if reach_m is not None and (offsets <= reach_m).any():
        within = offsets <= reach_m
    # The decay is fitted on every trace within the reach that is neither dead, clipped nor NaN,
    # those already left out among them: excluding a trace must not move the fit, or each round
    # excludes more (21 traces around each shot of active_p2, 2026-09-25). Within the reach only
    # (the user's decision of 2026-09-25): fitted over active_p2's whole line, the noise floor
    # flattened it, and 259 of the traces nearest the shots, the strongest, read too loud.
    outliers = (
        rms_decay_outliers(
            rms,
            offsets,
            ~(dead | clipped | nan) & within,
            thresholds.rms_outlier_mads,
            thresholds.rms_outlier_factor,
        )
        & ~left_out
        & within
    )
    metrics.append(
        Metric(
            name="rms_outliers",
            value=int(outliers.sum()),
            threshold=0,
            bound="max",
            passed=not outliers.any(),
        )
    )
    if outliers.any():
        traces = tuple(int(i) for i in np.flatnonzero(outliers))
        flags.append(
            Flag(
                name="rms_outliers",
                message=f"Traces {list(traces)} have an amplitude far off the decay with offset: "
                "bad coupling, not a parameter.",
                stage="preprocessing",
                action=ExcludeTraces(record=record, traces=traces),
                fixable=False,
            )
        )

    usable = ~(bad_traces | outliers)
    measured = usable & within if (usable & within).any() else usable
    snr = snr_db(finite, windows)
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
            finite[measured],
            preprocessed.sampling_freq,
            Windows(windows.signal[measured], windows.noise[measured], windows.where),
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
                message=f"Median SNR {median_snr:.1f} dB in the surface-wave window ({windows.where})"
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
    # No reversed-polarity check (the user, 2026-09-25): a reversed geophone hardly ever happens,
    # and at 1.5 m spacing near the source, neighbours shifted by more than half a period
    # flagged whole blocks of active_p2's traces.

    breaks = first_breaks(
        finite, np.asarray(preprocessed.ts, dtype=float), windows, thresholds.first_break_ratio
    )
    # Only the traces whose own SNR passes: a noisy trace's envelope crosses the threshold on
    # noise, early or late.
    breaks[~usable | (snr < thresholds.min_snr_db)] = np.nan
    fit = trigger_shift(breaks, offsets)
    # Without four first breaks, or on a noisy record whose breaks come late, the trigger is
    # not measured, not wrong.
    shift = None if fit is None or not snr_ok else fit[0]
    scatter = None if fit is None or not snr_ok else fit[2]
    consistent = scatter is None or scatter <= thresholds.max_trigger_scatter_s
    shift_ok = shift is None or abs(shift) <= thresholds.max_trigger_shift_s
    metrics.append(
        Metric(
            name="trigger_shift_s",
            value=_finite(shift),
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
    elif shift is not None and not shift_ok:
        flags.append(
            Flag(
                name="shifted_trigger",
                message=f"The first breaks put the trigger at {shift * 1000:.0f} ms, the same on "
                "every trace: correct the record's time origin by it.",
                stage="preprocessing",
                action=Override(
                    stage="preprocessing", overrides={"trigger": {"t0": round(shift, 4)}}
                ),
            )
        )

    if raw is not None:
        removed = 1 - float(np.sum(finite**2) / max(np.sum(np.nan_to_num(raw.xt) ** 2), 1e-30))
        metrics.append(Metric(name="energy_removed", value=round(removed, 3), passed=True))

    kept = Kept(band_hz=band, n_traces=int(usable.sum()))
    return _result(record, metrics, flags, kept)


def _result(record: str, metrics: list[Metric], flags: list[Flag], kept: Kept) -> GateResult:
    if any(not flag.fixable and isinstance(flag.action, ExcludeRecord) for flag in flags):
        verdict = "reject"
    elif flags:
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


def _energy_per_sample(xt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = mask.sum(axis=1)
    energy = np.sum(np.where(mask, xt, 0.0) ** 2, axis=1)
    return np.where(counts > 0, energy / np.maximum(counts, 1), 0.0)


def _mean_spectrum(
    xt: np.ndarray, mask: np.ndarray, sampling_freq: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """The mean power spectrum of the traces' masked samples, each trace's window cut to its
    samples and zero-padded to the record's length."""
    if xt.shape[0] == 0 or not mask.any():
        return None
    n = xt.shape[1]
    power = np.zeros(n // 2 + 1)
    counted = 0
    for trace, keep in zip(xt, mask, strict=True):
        if not keep.any():
            continue
        segment = np.where(keep, trace, 0.0)
        spectrum = np.abs(np.fft.rfft(segment, n=n)) ** 2 / max(int(keep.sum()), 1)
        power += spectrum
        counted += 1
    if counted == 0:
        return None
    return np.fft.rfftfreq(n, d=1 / sampling_freq), power / counted


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, 4)
