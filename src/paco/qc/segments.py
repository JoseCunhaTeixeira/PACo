"""S2's segments on a passive line: PACo optimizes the parameters the passive workflow has. The
segments' length matters, set by the window's span; the FK selection shapes the dispersion image;
the correlograms must converge. On a few trial windows of the chosen length, each candidate
segment length (a number of times the slowest wave's crossing of the window) slices the noise,
and each segment is whitened, normalized, tapered and correlated once, as the pipeline does,
flipped and not; each FK selection (none, or a velocity band and a threshold) then keeps the
segments whose f-k energy is lopsided enough in the band, flipped where it runs the other way,
and stacks them. A candidate is judged by the dispersion image its stack makes: the span of
wavelengths its M0 pick holds (the picker's, as S3 runs it: how much of the curve, and of the
depth, the image gives), then the pick's coherence; provided its correlograms converged (the
virtual shots of the kept segments' two halves agree over their arrivals) and it keeps enough
segments. The best wins when it widens the line's own span by `min_gain`; the trials are kept in
segments.json. (G2's share of coherent columns tells nothing here: a passive line's stacked
images come near 100 % whatever the settings.)"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.algorithms.flipping.flipping import FlipAxis, flip
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.algorithms.selection.stream.fk import fk_ratio
from sigpipe.base import DispersionImage, Stream
from sigpipe.masw.pipelines.common import load_preprocessed, stage_kwargs
from sigpipe.masw.presets import PassivePreset, apply_overrides, resolve_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.quality.signal import signal_windows
from sigpipe.masw.runs.processing import GEOMETRIES, RECORDS_FOLDER
from sigpipe.masw.windows import Exclusions, MASWWindow, apply_exclusions, build_windows
from sigpipe.transformers import Apodize, Correlate, Dispersion, Normalize, Slice, Stack, Whiten

from paco.qc.g1_signal import SignalThresholds
from paco.qc.g2_image import virtual_shot_snr

SEGMENTS_FILE = "segments.json"  # the trials, in the run folder
# The stages whose settings S2 tries on a passive line, unless the user set them.
SEGMENT_STAGES = frozenset({"slicing", "selection"})

type Band = tuple[float | None, float | None]


class SegmentRules(BaseModel):
    """The candidates S2 tries for a passive line's segments, and how one wins: provisional,
    measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    crossings: tuple[float, ...] = Field(
        default=(10.0, 20.0, 40.0, 80.0),
        description="The candidate segment lengths, in times the slowest wave (G1's vg_min) "
        "takes to cross the window's receivers; the line's own length is tried too",
    )
    thresholds: tuple[float, ...] = Field(
        default=(0.05, 0.1, 0.2, 0.3),
        description="How lopsided a segment's f-k energy must be to be kept, the candidates "
        "(the demo's segments sit mostly under 0.3)",
    )
    bands: tuple[Band, ...] = Field(
        default=((None, None), (80.0, 1500.0)),
        description="m/s, the velocity bands the lopsidedness is measured in (None: no bound)",
    )
    trial_windows: int = Field(
        default=3, ge=1, description="Windows along the line the candidates are tried on"
    )
    min_convergence: float = Field(
        default=0.7,
        gt=-1,
        le=1,
        description="How closely the virtual shots of the kept segments' two halves agree over "
        "their arrivals (the median correlation of their traces), at least: the correlograms "
        "converged",
    )
    min_gain: float = Field(
        default=0.2,
        ge=0,
        description="How much wider than the line's own the M0 pick's span of wavelengths must "
        "be (a share of it) for a candidate to replace its settings",
    )


@dataclass(frozen=True)
class Candidate:
    """One pair of settings tried: the segments' length, the FK selection (None: no selection),
    and what they made over the trial windows (medians; None: not measured)."""

    segment_s: float
    vmin: float | None
    vmax: float | None
    threshold: float | None
    wavelength_span: float | None  # the M0 pick's longest wavelength over its shortest
    coherence: float | None  # the image's value along the pick
    convergence: float | None
    snr_db: float | None
    kept_share: float


@dataclass(frozen=True)
class _Segments:
    """A trial window's segments, correlated as the pipeline does, flipped and not, with their
    f-k lopsidedness in each band."""

    straight: list[Stream]
    flipped: list[Stream]
    ratios: dict[Band, np.ndarray]


def choose_segments(
    profile: Profile,
    preset: PassivePreset,
    run_folder: Path,
    rules: SegmentRules,
    signal: SignalThresholds,
    picking: PickingParameters,
    exclusions: Exclusions,
    given: frozenset[str] = frozenset(),
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """The segments' length and FK selection for the line's windows (the stages of `given`, the
    user's, left as they are), and a note of the choice: None when the line's own settings stay
    (no candidate converged and kept enough segments with a better image), or the line cannot be
    tried (receivers not evenly spaced, no window)."""
    windows = [
        window
        for built in _spread(build_windows(profile, preset.masw), rules.trial_windows)
        if (window := apply_exclusions(built, exclusions, GEOMETRIES["passive"])) is not None
    ]
    if not windows:
        return None, ()
    own = stage_kwargs(preset, "slicing")
    own_length = round(float(own["segment_duration"]), 3)
    own_selection = _given_selection(preset)
    lengths = (
        [own_length]
        if "slicing" in given
        else lengths_tried(own_length, windows, signal.vg_min, rules, profile)
    )
    selections: list[tuple[Band, float] | None] = [own_selection]
    if "selection" not in given:
        for one in [None, *((band, t) for band in rules.bands for t in rules.thresholds)]:
            if one not in selections:
                selections.append(one)
    stack = Stack(**stage_kwargs(preset, "stacking"))
    dispersion = Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
    candidates: list[Candidate] = []
    skipped: list[str] = []
    for length in lengths:
        try:
            # Segments end to end (PACo's step, always its segments' length); a length the
            # whitening's band or the records cannot take refused as sigpipe checks it. The user's
            # own slicing, as it is.
            sliced = (
                preset
                if "slicing" in given
                else resolve_preset(
                    apply_overrides(
                        preset,
                        {"slicing": {"segment_duration": length, "segment_step": length}},
                    ),
                    profile,
                )
            )
            bands = [one[0] for one in selections if one is not None]
            trials = [_segments(window, sliced, run_folder, bands) for window in windows]
        except ValueError as error:  # refused by sigpipe; receivers not evenly spaced
            skipped.append(f"{length:g} s: {error}")
            continue
        for chosen in selections:
            candidates.append(_judge(trials, length, chosen, stack, dispersion, signal, picking))
    baseline = next(
        (
            one
            for one in candidates
            if one.segment_s == own_length and _selection_of(one) == own_selection
        ),
        None,
    )
    if baseline is None:
        return None, (f"The segments' settings not tried: {'; '.join(skipped)}.",)
    valid = [
        one
        for one in candidates
        if one.wavelength_span is not None
        and one.kept_share >= signal.min_fk_kept_share
        and one.convergence is not None
        and one.convergence >= rules.min_convergence
    ]
    best = max(
        valid, key=lambda one: (one.wavelength_span or 0.0, one.coherence or 0.0), default=None
    )
    converged = baseline in valid
    better = (
        best is not None
        and best != baseline
        and (
            not converged
            or (best.wavelength_span or 0.0)
            >= (baseline.wavelength_span or 0.0) * (1 + rules.min_gain)
        )
    )
    chosen = best if better else None
    (run_folder / SEGMENTS_FILE).write_text(
        json.dumps(
            {
                "windows": [window.xmid for window in windows],
                "baseline": asdict(baseline),
                "candidates": [asdict(one) for one in candidates],
                "chosen": asdict(chosen) if chosen is not None else None,
            },
            indent=2,
        )
    )
    span = median(_span(window) for window in windows)
    if chosen is None:
        return None, (_kept_note(baseline, converged, rules, signal.min_fk_kept_share),)
    changes: dict[str, object] = {}
    if chosen.segment_s != own_length:
        changes["slicing"] = {
            "segment_duration": chosen.segment_s,
            "segment_step": chosen.segment_s,
        }
    if _selection_of(chosen) != own_selection:
        changes["selection"] = (
            {"method": "none"}
            if chosen.threshold is None
            else {
                "method": "fk",
                "threshold": chosen.threshold,
                "vmin": chosen.vmin,
                "vmax": chosen.vmax,
            }
        )
    crossings = chosen.segment_s * signal.vg_min / span if span > 0 else float("nan")
    said = (
        f"Segments of {chosen.segment_s:g} s ({crossings:.0f} crossings of the windows' "
        f"{span:g} m at {signal.vg_min:g} m/s)"
        + (
            ", no FK selection"
            if chosen.threshold is None
            else f", FK selection at {chosen.threshold:g} {_written(chosen.vmin, chosen.vmax)}"
        )
        + f": M0 picked over {chosen.wavelength_span or 0:.1f} times its shortest wavelength "
        f"against {baseline.wavelength_span or 0:.1f} with the line's own settings"
        + ("" if converged else " (whose correlograms had not converged)")
        + f", correlograms converged ({chosen.convergence or 0:.2f}), "
        f"{chosen.kept_share:.0%} of the segments kept ({len(windows)} trial windows)."
    )
    return changes, (said,)


def _judge(
    trials: Sequence[_Segments],
    length: float,
    chosen: tuple[Band, float] | None,
    stack: Stack[Stream],
    dispersion: Dispersion,
    signal: SignalThresholds,
    picking: PickingParameters,
) -> Candidate:
    """One pair of settings over the trial windows: the M0 pick's span of wavelengths and its
    coherence on each image, the correlograms' convergence and the virtual shots' SNR (medians;
    a window picking nothing counts a span of 1), the segments kept."""
    spans: list[float] = []
    coherences: list[float] = []
    agreements: list[float] = []
    snrs: list[float] = []
    kept = total = 0
    for trial in trials:
        total += len(trial.straight)
        if chosen is None:
            picked = list(trial.straight)
        else:
            band, threshold = chosen
            ratios = trial.ratios[band]
            picked = [
                (trial.flipped if ratios[j] > 0 else trial.straight)[j]
                for j in np.flatnonzero(np.abs(ratios) > threshold)
            ]
        kept += len(picked)
        if len(picked) < 2:
            continue
        (shot,) = stack.transform(picked)
        (made,) = cast(list[DispersionImage], dispersion.transform([shot]))
        modes = pick_modes(made, picking)
        m0 = modes[0] if modes else None
        curve = m0.curve if m0 is not None else None
        if curve is not None and len(curve.fs) > 1:
            wavelengths = np.asarray(curve.vs, dtype=float) / np.asarray(curve.fs, dtype=float)
            spans.append(float(wavelengths.max() / wavelengths.min()))
        else:
            spans.append(1.0)
        if m0 is not None and m0.kept.any():
            coherences.append(float(np.median(m0.coherence[m0.kept])))
        (first,) = stack.transform(picked[0::2])
        (second,) = stack.transform(picked[1::2])
        agreement = halves_agreement(first, second, signal)
        if agreement is not None:
            agreements.append(agreement)
        snr = virtual_shot_snr(shot, signal.vg_min, signal.vg_max, signal.pad_s)
        if snr is not None:
            snrs.append(snr)
    band, threshold = chosen if chosen is not None else ((None, None), None)
    return Candidate(
        segment_s=length,
        vmin=band[0],
        vmax=band[1],
        threshold=threshold,
        wavelength_span=round(median(spans), 2) if spans else None,
        coherence=round(median(coherences), 3) if coherences else None,
        convergence=round(median(agreements), 3) if agreements else None,
        snr_db=round(median(snrs), 1) if snrs else None,
        kept_share=round(kept / max(total, 1), 3),
    )


def halves_agreement(first: Stream, second: Stream, signal: SignalThresholds) -> float | None:
    """How closely two virtual shots agree over their arrivals (between G1's velocities, from
    the virtual source): the median over their traces (the source's own aside) of the
    correlation of their samples there; None without arrivals to compare."""
    offsets = np.asarray(first.acquisition.offsets, dtype=float)
    windows = signal_windows(
        offsets, np.asarray(first.ts, dtype=float), signal.vg_min, signal.vg_max, signal.pad_s
    )
    if windows is None:
        return None
    a = np.asarray(first.xt, dtype=float)
    b = np.asarray(second.xt, dtype=float)
    correlations: list[float] = []
    for i in np.flatnonzero(offsets > 0):
        mask = windows.signal[i]
        x, y = a[i][mask], b[i][mask]
        if x.size > 2 and x.std() > 0 and y.std() > 0:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return median(correlations) if correlations else None


def lengths_tried(
    own: float,
    windows: Sequence[MASWWindow],
    vmin: float,
    rules: SegmentRules,
    profile: Profile,
) -> list[float]:
    """The segment lengths tried: the line's own, and `rules.crossings` times the slowest wave's
    crossing of the windows' span, within a tenth of a second (PAC's least) and the shortest
    record."""
    crossing = median(_span(window) for window in windows) / vmin
    shortest = min(record.duration_s for record in profile.records)
    tried = {round(own, 3)}
    for times in rules.crossings:
        length = round(times * crossing, 2)
        if 0.1 <= length <= shortest:
            tried.add(length)
    return sorted(tried)


def _span(window: MASWWindow) -> float:
    """The distance the window's receivers span, m."""
    xs = [receiver.x for receiver in window.acquisitions[0].receivers]
    return float(max(xs) - min(xs)) if xs else 0.0


def _selection_of(candidate: Candidate) -> tuple[Band, float] | None:
    """A candidate's FK selection, as `selections` lists them: None when off."""
    if candidate.threshold is None:
        return None
    return (candidate.vmin, candidate.vmax), candidate.threshold


def _given_selection(preset: PassivePreset) -> tuple[Band, float] | None:
    """The preset's own FK selection, as a candidate: None when off."""
    selection = stage_kwargs(preset, "selection")
    if selection["method"] != "fk":
        return None
    return (selection.get("vmin"), selection.get("vmax")), float(selection["threshold"])


def _kept_note(
    baseline: Candidate, converged: bool, rules: SegmentRules, min_kept_share: float
) -> str:
    """Why the line's own segments stay (`min_kept_share`: G2's floor on the share of segments
    the fk selection keeps, the trials' too)."""
    span = f"{baseline.wavelength_span:.1f}" if baseline.wavelength_span is not None else "none"
    if converged:
        return (
            f"The segments' settings kept: no candidate widened the M0 pick's span of "
            f"wavelengths by {rules.min_gain:.0%} over the line's own ({span} times its shortest)."
        )
    return (
        "The segments' settings kept, though their correlograms had not converged: no candidate "
        f"converged keeping {min_kept_share:.0%} of the segments."
    )


def _spread(windows: Sequence[MASWWindow], n: int) -> list[MASWWindow]:
    """`n` windows spread along the line: its ends and between."""
    if len(windows) <= n:
        return list(windows)
    return [windows[round(i)] for i in np.linspace(0, len(windows) - 1, n)]


def _segments(
    window: MASWWindow, preset: PassivePreset, run_folder: Path, bands: Sequence[Band]
) -> _Segments:
    """The window's noise segments, correlated as the pipeline does (whitened, normalized,
    tapered, against its first receiver), flipped and not, with their lopsidedness per band."""
    # A passive window's records load as streams.
    streams = cast(list[Stream], load_preprocessed(window, run_folder / RECORDS_FOLDER).transform())
    segments = Slice(**stage_kwargs(preset, "slicing")).transform(streams)
    prepare = [
        Whiten(**stage_kwargs(preset, "whitening")),
        Normalize(**stage_kwargs(preset, "normalization")),
        Apodize(method="hanning", frac=0.1),
    ]
    correlate = Correlate(method="cross", virtual_source_index=0, part="causal")
    ratios = {band: np.array([fk_ratio(one, *band) for one in segments]) for band in set(bands)}
    ready = list(segments)
    for step in prepare:
        ready = list(step.transform(ready))
    turned = [flip(stream=one, axis=FlipAxis.SPACE, flip_acquisition=False) for one in ready]
    return _Segments(
        straight=list(correlate.transform(ready)),
        flipped=list(correlate.transform(turned)),
        ratios=ratios,
    )


def _written(vmin: float | None, vmax: float | None) -> str:
    """A velocity band in words."""
    if vmin is None and vmax is None:
        return "at any velocity"
    if vmin is None:
        return f"up to {vmax:g} m/s"
    if vmax is None:
        return f"from {vmin:g} m/s"
    return f"{vmin:g}-{vmax:g} m/s"
