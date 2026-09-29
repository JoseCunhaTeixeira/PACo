"""G2, the QC of a dispersion image before picking (docs/qc_workflow.md): coherent energy
against the noise floor, the energy maximum on the grid's edges, competing ridges (a higher mode
or aliasing), and a coherent band much narrower than the record's usable band. A passive or
passive-active window's stacked correlations, the signal its image is made of, measured as a
record is (sigpipe's measure_signal, from the virtual source), and a passive window's fk segment
selection: judged too, the phase shift again with more of the data when they fall short, then
rejected. The measures are sigpipe's (sigpipe.masw.quality.image, .measures); G2 judges them."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.base import DispersionImage, Stream
from sigpipe.masw.quality.image import aliased, coherent_columns, competing_ridges, edge_peaks
from sigpipe.masw.quality.measures import Measure, SignalLimits, SignalReport, measure_signal

from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Override, Reject

GATE = "G2"


class ImageThresholds(BaseModel):
    """G2's limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coherent_level: float = Field(
        default=0.3,
        gt=0,
        lt=1,
        description="A column is coherent when its peak is this far from the noise floor\n"
        "1 / sqrt(N) towards 1, the most a plane wave gives: the same for every N.",
    )
    min_coherent_columns: float = Field(
        default=0.5, ge=0, le=1, description="Share of the image's columns that must be coherent."
    )
    edge_share: float = Field(
        default=0.02,
        gt=0,
        lt=0.5,
        description="Share of the velocity range that counts as an edge.",
    )
    max_edge_columns: float = Field(
        default=0.2, ge=0, le=1, description="Share of coherent columns peaking on an edge."
    )
    competing_ratio: float = Field(
        default=0.7, gt=0, lt=1, description="A second ridge at least this share of the first."
    )
    competing_separation: float = Field(
        default=0.15, gt=0, description="Its velocity at least this share away from the first's."
    )
    max_competing_columns: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Share of coherent columns with a second ridge: good 24-receiver windows of the\n"
        "demo hold one in 52 to 60 % of theirs, 5-receiver ones in 87 to 95 %.",
    )
    min_band_share: float = Field(
        default=0.5, gt=0, le=1, description="Coherent band over the usable band, at least."
    )
    vmin_floor: float = Field(
        default=30.0,
        gt=0,
        description="m/s: energy peaking at a vmin below this is an artifact, not a slower wave.",
    )
    vg_min: float = Field(default=80.0, gt=0, description="m/s, for the mute a retry suggests")
    vg_max: float = Field(default=1500.0, gt=0, description="m/s, for the mute a retry suggests")


def judge_image(
    unit: str,
    image: DispersionImage,
    thresholds: ImageThresholds,
    usable_band: tuple[float, float] | None = None,
    mode: str = "active",
    muted: bool = False,
    correlations: SignalReport | None = None,
    selection: Sequence[Measure] = (),
    more: Override | None = None,
) -> GateResult:
    """G2's verdict on one window's image, with G1's usable band of its records when known. A
    passive-active image (`mode`) peaking at the grid's top velocity, its records not `muted`,
    tries the surface-wave mute before a wider grid: correlated whole, a record's noise common
    to every trace peaks there. A passive or passive-active window's stacked `correlations`,
    measured as a record is (their SNR, coherence and usable band), and a passive one's fk
    `selection` (the share of segments kept): falling short, the phase shift again with more of
    the data (`more`: more_data), else rejected. An image mostly noise tries the surface-wave
    mute when its records are not muted yet; else (a passive line has no muting) it is flagged
    and kept, the later gates judging its pick: no retry that could not change it (the user,
    2026-09-29)."""
    fs = np.asarray(image.fs, dtype=float)
    vs = np.asarray(image.vs, dtype=float)
    coherent = coherent_columns(image, thresholds.coherent_level)
    n_coherent = int(coherent.sum())
    share = n_coherent / max(coherent.size, 1)
    metrics = [
        Metric(
            name="coherent_columns",
            value=round(share, 3),
            threshold=thresholds.min_coherent_columns,
            bound="min",
            passed=share >= thresholds.min_coherent_columns,
        )
    ]
    flags: list[Flag] = []
    if correlations is not None:
        metrics += [Metric(**one.model_dump()) for one in correlations.measures]
        short = [one for one in correlations.measures if one.name in _JUDGED and not one.passed]
        if short:
            flags.append(_short("noisy_correlations", "Its stacked correlations", short, more))
        bad = [one for one in correlations.measures if one.name in _TRACES and not one.passed]
        if bad:
            flags.append(
                Flag(
                    name="bad_correlations",
                    message="Its stacked correlations hold "
                    + ", ".join(f"{one.value:g} {_WORDS[one.name]}" for one in bad)
                    + ": a receiver dead or clipped in its records, the line's to leave out.",
                    stage="phase_shift",
                    action=Keep(note="a receiver the records' checks leave to the line"),
                    fixable=False,
                )
            )
    if selection:
        metrics += [Metric(**one.model_dump()) for one in selection]
        kept = [one for one in selection if one.name == "fk_kept" and not one.passed]
        if kept:
            flags.append(_short("few_segments_kept", "Its fk selection", kept, more))
    mute = Override(
        stage="preprocessing",
        overrides={
            "muting": {
                "method": "mute",
                "vmin": thresholds.vg_min,
                "vmax": thresholds.vg_max,
            }
        },
    )
    # A passive line has no muting (the user, 2026-09-28), and muted records have theirs: an image
    # mostly noise is flagged and kept there, the later gates judging its pick.
    mutable = mode != "passive" and not muted
    if n_coherent == 0:
        said = (
            f"No column of the image rises {thresholds.coherent_level:.0%} of the way from the "
            "noise floor to 1: nothing to pick."
        )
        flags.append(
            Flag(
                name="no_coherent_energy",
                message=f"{said} Try a surface-wave mute." if mutable else said,
                stage="preprocessing",
                action=mute
                if mutable
                else Keep(note="no coherent energy, its records muted or no muting to try"),
                fixable=mutable,
            )
        )
        return _result(
            unit, metrics, flags, Kept(band_hz=None, n_traces=len(image.acquisition.receivers))
        )

    band = (float(fs[coherent].min()), float(fs[coherent].max()))
    if share < thresholds.min_coherent_columns:
        said = (
            f"Only {share:.0%} of the columns are coherent ({band[0]:.1f}-{band[1]:.1f} Hz): the "
            "image is mostly noise."
        )
        flags.append(
            Flag(
                name="weak_coherence",
                message=f"{said} Try a surface-wave mute." if mutable else said,
                stage="preprocessing",
                action=mute
                if mutable
                else Keep(note="mostly noise, its records muted or no muting to try"),
                fixable=mutable,
            )
        )

    low, high = edge_peaks(image, coherent, thresholds.edge_share, thresholds.vmin_floor)
    # A peak at a vmin below the floor is an artifact (a correlation's zero lag, a mute's edge):
    # start the range at the floor; at a real vmin, lower it, never under the floor (where the
    # artifacts are, and where it would be raised again). At the floor already, nothing lower is
    # left to try, and the flag is kept.
    artifact = vs.min() < thresholds.vmin_floor
    vmin = (
        thresholds.vmin_floor
        if artifact
        else max(thresholds.vmin_floor, round(float(vs.min()) / 1.5, 1))
    )
    at_floor = not artifact and vmin >= round(float(vs.min()), 1)
    for name, count, edge, override in (
        ("ridge_at_vmin", low, vs.min(), {"vmin": vmin}),
        ("ridge_at_vmax", high, vs.max(), {"vmax": round(vs.max() * 1.5, 1)}),
    ):
        edge_share = count / n_coherent
        metrics.append(
            Metric(
                name=name,
                value=round(edge_share, 3),
                threshold=thresholds.max_edge_columns,
                bound="max",
                passed=edge_share <= thresholds.max_edge_columns,
            )
        )
        if edge_share > thresholds.max_edge_columns:
            common_noise = name == "ridge_at_vmax" and mode == "passive-active" and not muted
            floor = name == "ridge_at_vmin" and at_floor
            flags.append(
                Flag(
                    name=name,
                    message=f"{edge_share:.0%} of the coherent columns peak at {edge:g} m/s, the "
                    "grid's edge: "
                    + (
                        "an artifact below any surface wave; start the range above it."
                        if name == "ridge_at_vmin" and artifact
                        else "correlated whole, a record's noise common to every trace peaks "
                        "there: try a surface-wave mute first."
                        if common_noise
                        else f"the range starts at {thresholds.vmin_floor:g} m/s already, under "
                        "which the peaks are artifacts. Kept, to look at."
                        if floor
                        else "the velocity range is too narrow."
                    ),
                    stage="preprocessing" if common_noise else "phase_shift",
                    action=mute
                    if common_noise
                    else Keep(note="the range at its floor")
                    if floor
                    else Override(stage="phase_shift", overrides={"dispersion": override}),
                )
            )

    rows = np.flatnonzero(coherent)
    at_fmin, at_fmax = rows[0] == 0, rows[-1] == fs.size - 1
    # A band reaching fmax is kept, not widened: a wider band lets a second ridge compete, and
    # fewer curves pass G3. The coherence rules cap fmax at the records' usable band.
    for name, touches, action in (
        # Kept too: the picker stops where its ridge breaks or the window resolves no velocity.
        (
            "band_at_fmin",
            at_fmin,
            Keep(note="fmin stays: the picker stops where the ridge breaks"),
        ),
        ("band_at_fmax", at_fmax, Keep(note="fmax stays: a wider band let other ridges compete")),
    ):
        metrics.append(
            Metric(name=name, value=float(touches), threshold=0, bound="max", passed=not touches)
        )
        if touches:
            where = "lowest" if name == "band_at_fmin" else "highest"
            flags.append(
                Flag(
                    name=name,
                    message=f"The coherent band reaches the image's {where} frequency: the band "
                    "may go on beyond it.",
                    stage="phase_shift",
                    action=action,
                )
            )

    competing = competing_ridges(
        image,
        coherent,
        thresholds.competing_ratio,
        thresholds.competing_separation,
        thresholds.vmin_floor,
    )
    competing_share = int(competing.sum()) / n_coherent
    metrics.append(
        Metric(
            name="competing_ridges",
            value=round(competing_share, 3),
            threshold=thresholds.max_competing_columns,
            bound="max",
            passed=competing_share <= thresholds.max_competing_columns,
        )
    )
    # On a grid too narrow, a truncated ridge makes second ridges and aliases of its own: the
    # velocity range first.
    grid_first = any(flag.name in ("ridge_at_vmin", "ridge_at_vmax") for flag in flags)
    if competing_share > thresholds.max_competing_columns and not grid_first:
        alias = aliased(image, competing, thresholds.vmin_floor)
        if alias.sum() > competing.sum() / 2:
            first = float(fs[alias][0])
            flags.append(
                Flag(
                    name="aliasing",
                    message=f"{competing_share:.0%} of the coherent columns hold a second ridge, "
                    f"mostly below the aliasing limit 2 dx f: an alias from {first:.1f} Hz on.",
                    stage="phase_shift",
                    action=Override(
                        stage="phase_shift", overrides={"dispersion": {"fmax": round(first, 1)}}
                    ),
                )
            )
        else:
            # Kept: the fundamental mode is picked as the slowest ridge whatever the modes
            # picked, and only it is judged; picking two changes neither.
            flags.append(
                Flag(
                    name="competing_ridges",
                    message=f"{competing_share:.0%} of the coherent columns hold a second ridge of "
                    "similar strength: a higher mode may dominate. Kept, the pick's checks "
                    "judging it.",
                    stage="picking",
                    action=Keep(note="the fundamental mode picked as the slowest ridge"),
                )
            )

    if usable_band is not None:
        usable_width = usable_band[1] - usable_band[0]
        band_share = (band[1] - band[0]) / usable_width if usable_width > 0 else 1.0
        metrics.append(
            Metric(
                name="band_share_of_usable",
                value=round(band_share, 3),
                threshold=thresholds.min_band_share,
                bound="min",
                passed=band_share >= thresholds.min_band_share,
            )
        )
        if band_share < thresholds.min_band_share:
            flags.append(
                Flag(
                    name="narrower_than_usable",
                    message=f"The coherent band {band[0]:.1f}-{band[1]:.1f} Hz is {band_share:.0%} "
                    f"of the records' usable band {usable_band[0]:.1f}-{usable_band[1]:.1f} Hz: "
                    "the image does not use all the data offers.",
                    stage="phase_shift",
                    action=Keep(note="the band is capped, never widened: see band_at_fmax"),
                )
            )

    # The image's own measures said so, beside its correlations' and its selection's.
    metrics = [one if one.of else one.model_copy(update={"of": "image"}) for one in metrics]
    return _result(
        unit, metrics, flags, Kept(band_hz=band, n_traces=len(image.acquisition.receivers))
    )


# The correlations' measures judged, and those of their traces (the records' checks' to fix).
_JUDGED = ("snr_db", "usable_band_hz", "lateral_coherence")
_TRACES = ("dead_traces", "clipped_traces", "nan_traces")
_WORDS = {
    "dead_traces": "dead traces",
    "clipped_traces": "clipped traces",
    "nan_traces": "traces not a number",
    "snr_db": "median SNR",
    "usable_band_hz": "usable band",
    "lateral_coherence": "coherence",
    "fk_kept": "share of segments kept",
}


def _short(name: str, said: str, short: Sequence[Measure], more: Override | None) -> Flag:
    """A flag of `short` measures (of the correlations, or the selection): the phase shift again
    with more of the data (`more`), else rejected."""
    found = "; ".join(
        f"{_WORDS[one.name]} {'none' if one.value is None else f'{one.value:g}'}"
        f"{' ' + one.unit if one.unit and one.value is not None else ''}"
        + ("" if one.threshold is None else f" (at least {one.threshold:g})")
        for one in short
    )
    if more is not None:
        return Flag(
            name=name,
            message=f"{said} fall short, {found}: the phase shift again with more of the data.",
            stage="phase_shift",
            action=more,
        )
    return Flag(
        name=name,
        message=f"{said} fall short, {found}: all of its data is stacked already.",
        stage="phase_shift",
        action=Reject(reason="the window's correlations fall short, with all its data"),
        fixable=False,
    )


def more_data(mode: str, values: Mapping[str, Any]) -> Override | None:
    """The phase shift of a passive window again with more of the data (`values`: its stage
    values as it last ran): its fk selection keeping more segments (its threshold halved). None
    when it keeps them all already (no selection, or a threshold at 0.01), and for a
    passive-active window, which stacks every shot within reach: nothing more to stack. Longer
    segments are no more data: the same record cut in fewer pieces."""
    if mode != "passive":
        return None
    selection = values.get("selection") or {}
    threshold = selection.get("threshold")
    if selection.get("method") == "fk" and isinstance(threshold, int | float) and threshold > 0.01:
        return Override(
            stage="phase_shift", overrides={"selection": {"threshold": round(threshold / 2, 3)}}
        )
    return None


def _result(unit: str, metrics: list[Metric], flags: list[Flag], kept: Kept) -> GateResult:
    # A kept flag is information: it never changes the verdict; one no change fixes rejects, as
    # G3's.
    acted = [flag for flag in flags if not isinstance(flag.action, Keep)]
    verdict = (
        "pass" if not acted else "reject" if any(not flag.fixable for flag in acted) else "retry"
    )
    return GateResult(
        gate=GATE, unit=unit, verdict=verdict, metrics=tuple(metrics), flags=tuple(flags), kept=kept
    )


def virtual_shot_snr(stream: Stream, vmin: float, vmax: float, pad_s: float) -> float | None:
    """A passive or passive-active window's virtual shot (its stacked correlations): the median
    SNR of its traces, dB, measured as a record's (measure_signal from the virtual source, its
    own trace aside); None when the lags end before a noise window."""
    limits = SignalLimits(vg_min=vmin, vg_max=vmax, pad_s=pad_s)
    return measure_signal(stream, limits, source="virtual").median_snr
