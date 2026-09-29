"""G1, the signal QC of a preprocessed record (docs/qc_workflow.md): dead, clipped and NaN
traces; amplitudes off the decay with offset; the usable band, and the SNR and the lateral
coherence in it, from a surface-wave window against a noise window; the trigger. On a passive
record, only the checks that need no shot. The measures are sigpipe's
(sigpipe.masw.quality.measures, PAC's alike); G1 judges them. A trace off the decay in one record
is reported, not left out: G1 over the line (`judge_receivers`) leaves out the receivers off it
in most of the records that reach them, a bad geophone.

A record under its limits is left out at once: its SNR and coherence are measured in its usable
band, which a filter cannot change (nor can a filter change the dispersion image: the phase shift
divides each trace's spectrum by its own amplitude), its noise before its muting, which a mute
cannot; a retry would measure them again. Its trigger alone is corrected: a muted record's, the
trigger being the muting's."""

import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np
from pydantic import ConfigDict, Field
from sigpipe.base import Stream
from sigpipe.masw.quality.measures import (
    SignalLimits,
    measure_signal,
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
# The modes whose records' traces are compared with their neighbours' spectra: a noise record's,
# a shot's correlated whole and an active shot's (the same measures for every line).
SPECTRA_MODES = frozenset({"active", "passive", "passive-active"})


class SignalThresholds(SignalLimits):
    """G1's window and limits (sigpipe's SignalLimits: how a signal is measured, the same for
    PAC), and G1's own over the line: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mute_width_s: float = Field(
        default=0.05,
        ge=0,
        description="s, kept after the slowest arrival by a mute when no record's pulse could be "
        "measured (each record's own, else the line's median, first)",
    )
    spectra_line_share: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Share of the records reaching it in which a receiver is off its "
        "neighbours' spectra, for the line to flag it",
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


def judge_signal(
    record: str,
    preprocessed: Stream,
    thresholds: SignalThresholds,
    active: bool = True,
    excluded: Collection[int] = (),
    reach_m: float | None = None,
    shot_s: float = 0.0,
    applied_s: float | None = None,
    spectra: bool = False,
    before_muting: tuple[Stream, float] | None = None,
    image_band: tuple[float, float] | None = None,
) -> GateResult:
    """G1's verdict on one record: the metrics, the flags with their actions, and what is kept.
    The measures are sigpipe's (measure_signal: PAC measures a record it made alike), each with
    what it covers; G1 judges them. The traces `excluded` already (receiver indices) are left
    out of every measure and flag; the decay with offset, the SNR, the usable band and the
    lateral coherence are measured on the traces within `reach_m` of the shot (the line's
    reach, beyond which the traces carry no wave). The first breaks should put the shot at
    `shot_s` (`trigger_context`); `applied_s`, the shift the muting applied, None with the
    muting off: the trigger is part of the muting, so a record not muted has no trigger to
    correct, its shot time only reported. The noise, the usable band, the first breaks and the
    shot's pulse are measured on the record before its muting (`before_muting`: it, and where
    its shot is on it; a muting zeroes the noise window, and cuts the first breaks), when
    muted. The SNR and the coherence in the part of its usable band the dispersion images use
    (`image_band`, image_band's). Under its limits (no usable band there, its SNR or its
    coherence in it), the record is left out at once: nothing a retry could change would change
    them."""
    report = measure_signal(
        preprocessed,
        thresholds,
        source="shot" if active else "none",
        shot_s=shot_s,
        applied_s=applied_s,
        excluded=excluded,
        reach_m=reach_m,
        before_muting=before_muting,
        spectra=spectra,
        image_band=image_band,
    )
    metrics = [Metric(**measure.model_dump()) for measure in report.measures]
    flags: list[Flag] = []
    for name, mask, what in (
        ("dead_traces", report.dead, "dead"),
        ("clipped_traces", report.clipped, "clipped"),
        ("nan_traces", report.nan, "not a number"),
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
    bad_traces = report.dead | report.clipped | report.nan | report.left_out
    kept = Kept(n_traces=int((~bad_traces).sum()))
    if not active:
        return _result(record, metrics, flags, kept)
    if report.windows is None:
        flags.append(
            Flag(
                name="record_too_short",
                message="The record ends before the slowest arrival within the line's reach "
                "leaves room for a noise window: no SNR, no usable band.",
                stage="preprocessing",
                action=ExcludeRecord(record=record),
                fixable=False,
            )
        )
        return _result(record, metrics, flags, kept)

    band, imaged = report.band, report.imaged
    median_snr = report.median_snr
    noise_where = f"noise {report.noise_where}"
    if imaged is None:
        found = (
            f"its usable band {band[0]:.1f}-{band[1]:.1f} Hz misses the images' band "
            if band is not None and image_band is not None
            else f"its surface waves stand {thresholds.band_db:g} dB over its {noise_where} "
            f"across less than {thresholds.min_band_hz:g} Hz"
        )
        flags.append(
            Flag(
                name="no_usable_band",
                message=f"No usable band: {found.rstrip()}, nothing to image. Left out: a filter "
                "or a mute would measure the same.",
                stage="preprocessing",
                action=ExcludeRecord(record=record),
                fixable=False,
            )
        )
    elif median_snr is None or median_snr < thresholds.min_snr_db:
        said = float("nan") if median_snr is None else median_snr
        flags.append(
            Flag(
                name="low_snr",
                message=f"Median SNR {said:.1f} dB in its usable band {imaged[0]:.1f}-"
                f"{imaged[1]:.1f} Hz ({noise_where}), under {thresholds.min_snr_db:g} dB. Left "
                "out: a filter or a mute would measure the same.",
                stage="preprocessing",
                action=ExcludeRecord(record=record),
                fixable=False,
            )
        )
    coherence = report.median_coherence
    if coherence is None or coherence < thresholds.min_coherence:
        said = float("nan") if coherence is None else coherence
        flags.append(
            Flag(
                name="low_coherence",
                message=f"Neighbouring traces correlate at {said:.2f} in the surface-wave window"
                + (" and its usable band" if imaged is not None else "")
                + f", under {thresholds.min_coherence:g}: noisy or scattered. Left out: a "
                "filter or a mute would measure the same.",
                stage="preprocessing",
                action=ExcludeRecord(record=record),
                fixable=False,
            )
        )
    # No reversed-polarity check: a reversed geophone hardly ever happens, and near the source,
    # neighbours shifted by more than half a period look reversed.

    # Without four first breaks, or on a noisy record whose breaks come late, the trigger is
    # not measured, not wrong (measure_signal: no fit then). Fitted on the breaks nearest the
    # shot (the direct wave's), on the record before its muting.
    error = None if report.fit is None else report.fit[0]
    scatter = report.scatter
    consistent = scatter is None or scatter <= thresholds.max_trigger_scatter_s
    if scatter is not None and not consistent:
        flags.append(
            Flag(
                name="inconsistent_trigger",
                message=f"The first breaks scatter by {scatter * 1000:.0f} ms about their line: "
                "the delay differs from trace to trace, which no correction fixes.",
                stage="preprocessing",
                action=ExcludeRecord(record=record),
                fixable=False,
            )
        )
    elif (
        error is not None and applied_s is not None and abs(error) > thresholds.max_trigger_error_s
    ):
        # The trigger never below 0: a shot before the record's start, the recording started late;
        # the trigger at 0 and what is left said, or, at 0 already, said alone.
        t0 = round(applied_s + error, 4)
        said_error = (
            f"The first breaks put the shot {error * 1000:+.0f} ms from the time origin the "
            "muting's trigger moved, the same on every trace"
        )
        early = f"{-t0 * 1000:.0f} ms before the record starts: it began after the shot"
        flags.append(
            Flag(
                name="shifted_trigger",
                message=f"{said_error}: correct its trigger by them."
                if t0 >= 0
                else f"{said_error}, {early}; its trigger set to 0."
                if applied_s > 0
                else f"{said_error}, {early}. No trigger corrects it.",
                stage="preprocessing",
                action=Override(stage="preprocessing", overrides={"trigger": {"t0": max(t0, 0.0)}})
                if t0 >= 0 or applied_s > 0
                else Keep(note="the shot before the record's start"),
            )
        )

    usable = report.usable if report.usable is not None else ~bad_traces
    kept = Kept(band_hz=band, n_traces=int(usable.sum()))
    return _result(record, metrics, flags, kept)


def judge_spectra(
    off: Mapping[int, int],
    reached: Mapping[int, int],
    positions: Sequence[float],
    thresholds: SignalThresholds,
) -> tuple[Metric, Flag | None]:
    """G1 over the line: the receivers off their neighbours' spectra (`off`: in how many records)
    in at least `spectra_line_share` of the records judging them (`reached`). Flagged, kept: the
    workflow unchanged."""
    bad = [
        receiver
        for receiver, count in sorted(reached.items())
        if count and off.get(receiver, 0) >= math.ceil(thresholds.spectra_line_share * count)
    ]
    metric = Metric(
        name="spectral_receivers",
        value=len(bad),
        threshold=0,
        bound="max",
        passed=not bad,
        of="line",
        over=f"the line's {len(reached)} receivers: off their neighbours' spectra in "
        f"{thresholds.spectra_line_share:.0%} of the records reaching them",
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
            of="line",
            over=f"the line's {len(reached)} receivers: off the amplitude decay in "
            f"{thresholds.rms_outlier_share:.0%} of the records reaching them, "
            f"{thresholds.rms_outlier_min_records} at least",
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
