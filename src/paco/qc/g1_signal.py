"""G1, the signal QC of a preprocessed record (docs/qc_workflow.md): dead, clipped and NaN
traces; amplitudes off the decay with offset; the SNR and the usable band,
from a surface-wave window against a noise window; lateral coherence; the trigger; and what a
mute removed. On a passive record, only the checks that need no trigger. The measures are
sigpipe's (sigpipe.masw.quality.signal); G1 judges them."""

import math
from collections.abc import Collection

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.base import Stream
from sigpipe.masw.quality.signal import (
    Windows,
    dead_clipped_nan,
    first_breaks,
    lateral_coherence,
    rms_decay_outliers,
    signal_windows,
    snr_db,
    trigger_shift,
    usable_band,
)

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
    if not active:
        return _result(record, metrics, flags, kept)

    offsets = np.asarray(preprocessed.acquisition.offsets, dtype=float)
    windows = signal_windows(
        offsets,
        np.asarray(preprocessed.ts, dtype=float),
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


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, 4)
