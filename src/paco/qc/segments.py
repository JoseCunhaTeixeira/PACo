"""S2's segments on a passive line. Their length follows the window's span: the slowest wave
(G1's vg_min) crosses the window `segment_crossings` times in a segment, end to end
(sized_slicing); the ladder tries each window length with its own, and the line keeps its
length's. For segments the user gave, their FK selection is tried on the ladder's sample windows
at the chosen length (`trials`): each segment is whitened, normalized, tapered and correlated once, as the pipeline
does, flipped and not; each FK selection (a velocity band and a threshold, the selection always
on) then keeps the segments whose f-k energy is lopsided enough in the band, flipped where it runs
the other way, and stacks them. A candidate is judged by the dispersion image its stack makes:
the span of wavelengths its M0 pick holds (the picker's, as S3 runs it: how much of the curve,
and of the depth, the image gives), then the pick's coherence; provided its correlograms
converged (the virtual shots of the kept segments' two halves agree over their arrivals) and it
keeps enough segments. The best wins when it widens the line's own span by `min_gain`; the trials
are kept in segments.json. (G2's share of coherent columns tells nothing here: a passive line's
stacked images come near 100 % whatever the settings.)"""

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
from sigpipe.masw.presets import PassivePreset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.quality.signal import signal_windows
from sigpipe.masw.runs.processing import GEOMETRIES, RECORDS_FOLDER
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions, MASWWindow, apply_exclusions, build_windows
from sigpipe.transformers import Apodize, Correlate, Dispersion, Normalize, Slice, Stack, Whiten

from paco.qc.g1_signal import SignalThresholds
from paco.qc.g2_image import virtual_shot_snr

SEGMENTS_FILE = "segments.json"  # the trials, in the run folder
# A passive line's segment stages, which S2 sets (their length) or tries (their FK selection),
# unless the user set them.
SEGMENT_STAGES = frozenset({"slicing", "selection"})

type Band = tuple[float | None, float | None]


class SegmentRules(BaseModel):
    """A passive line's segments: their length, the FK selections S2 tries for them and how one
    wins, measured on PAC's test lines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_crossings: float = Field(
        default=10.0,
        gt=0,
        description="The segments' length, end to end, in times the slowest wave (G1's vg_min) "
        "takes to cross the window's receivers",
    )
    thresholds: tuple[float, ...] = Field(
        default=(0.05, 0.1, 0.2, 0.3),
        description="How lopsided a segment's f-k energy must be to be kept, the candidates "
        "(the passive test line's segments sit mostly under 0.3)",
    )
    bands: tuple[Band, ...] = Field(
        default=((None, None), (80.0, 1500.0)),
        description="m/s, the velocity bands the lopsidedness is measured in (None: no bound)",
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
    trials: int,
    given: frozenset[str] = frozenset(),
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """The FK selection for the line's segments, as long as the line has them, and a note of
    the choice: None when the line's own selection stays (no candidate converged and kept enough
    segments with a better image), the user set it (`given`), or the line cannot be tried
    (receivers not evenly spaced, no window)."""
    windows = [
        window
        for built in _spread(build_windows(profile, preset.masw), trials)
        if (window := apply_exclusions(built, exclusions, GEOMETRIES["passive"])) is not None
    ]
    if not windows or "selection" in given:
        return None, ()
    length = round(float(stage_kwargs(preset, "slicing")["segment_duration"]), 3)
    own = _given_selection(preset)
    # The FK selection always on: its thresholds and bands tried, never none.
    selections: list[tuple[Band, float] | None] = [own]
    selections += [
        one for one in ((band, t) for band in rules.bands for t in rules.thresholds) if one != own
    ]
    stack = Stack(**stage_kwargs(preset, "stacking"))
    dispersion = Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
    try:
        bands = [one[0] for one in selections if one is not None]
        made = [_segments(window, preset, run_folder, bands) for window in windows]
    except ValueError as error:  # refused by sigpipe; receivers not evenly spaced
        return None, (f"The segments' FK selection not tried: {error}.",)
    candidates = [
        _judge(made, length, chosen, stack, dispersion, signal, picking) for chosen in selections
    ]
    baseline = candidates[0]
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
    write_atomic(
        run_folder / SEGMENTS_FILE,
        json.dumps(
            {
                "windows": [window.xmid for window in windows],
                "baseline": asdict(baseline),
                "candidates": [asdict(one) for one in candidates],
                "chosen": asdict(chosen) if chosen is not None else None,
            },
            indent=2,
        ),
    )
    if chosen is None or chosen.threshold is None:
        return None, (_kept_note(baseline, converged, rules, signal.min_fk_kept_share),)
    said = (
        f"FK selection at {chosen.threshold:g} {_written(chosen.vmin, chosen.vmax)} on the "
        f"line's {length:g} s segments: M0 picked over {chosen.wavelength_span or 0:.1f} times "
        f"its shortest wavelength against {baseline.wavelength_span or 0:.1f} with the line's own "
        "selection"
        + ("" if converged else " (whose correlograms had not converged)")
        + f", correlograms converged ({chosen.convergence or 0:.2f}), "
        f"{chosen.kept_share:.0%} of the segments kept ({len(windows)} trial windows)."
    )
    selection = {
        "method": "fk",
        "threshold": chosen.threshold,
        "vmin": chosen.vmin,
        "vmax": chosen.vmax,
    }
    return {"selection": selection}, (said,)


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


def sized_slicing(
    length: int, profile: Profile, rules: SegmentRules, vmin: float
) -> dict[str, dict[str, float]]:
    """Segments for windows of `length` receivers, end to end, as long as the slowest wave takes
    to cross such a window `rules.segment_crossings` times, within a tenth of a second and the
    shortest record."""
    xs = sorted(receiver.x for receiver in profile.receivers)
    spacing = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0
    shortest = min(record.duration_s for record in profile.records)
    seconds = min(
        max(round(rules.segment_crossings * (length - 1) * spacing / vmin, 2), 0.1), shortest
    )
    return {"slicing": {"segment_duration": seconds, "segment_step": seconds}}


def _given_selection(preset: PassivePreset) -> tuple[Band, float] | None:
    """The preset's own FK selection, as a candidate: None when off."""
    selection = stage_kwargs(preset, "selection")
    if selection["method"] != "fk":
        return None
    return (selection.get("vmin"), selection.get("vmax")), float(selection["threshold"])


def _kept_note(
    baseline: Candidate, converged: bool, rules: SegmentRules, min_kept_share: float
) -> str:
    """Why the line's own FK selection stays (`min_kept_share`: G2's floor on the share of
    segments the fk selection keeps, the trials' too)."""
    span = f"{baseline.wavelength_span:.1f}" if baseline.wavelength_span is not None else "none"
    if converged:
        return (
            f"The line's FK selection kept: no candidate widened the M0 pick's span of "
            f"wavelengths by {rules.min_gain:.0%} over its own ({span} times its shortest)."
        )
    return (
        "The line's FK selection kept, though its correlograms had not converged: no candidate "
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
