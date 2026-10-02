"""The coherence rules for S2 (docs/qc_workflow.md), checked before the phase shift runs: the
image's band within what G1 found usable in the records, and one window
length for the whole line, the shortest at which most trial windows give a curve G3 passes
(lateral resolution first)."""

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.masw.presets import ActivePreset, PassivePreset, apply_overrides, resolve_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.runs import RecordOutcome, RunError, WindowOutcome, load_image
from sigpipe.masw.runs.processing import RECORDS_FOLDER, process_windows
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions, MASWWindow, build_windows
from sigpipe.masw.windows import nearest_offset as window_nearest_offset

from paco import stopping
from paco.qc.g2_image import ImageThresholds, judge_image
from paco.qc.g3_curve import CurveThresholds, judge_curve
from paco.qc.loops import deep_merge
from paco.qc.models import Override

TRIALS_FOLDER = "coherence"  # inside the run folder: the ladder's trial windows, by length
COHERENCE_FILE = "coherence.json"


class CoherenceRules(BaseModel):
    """How S2's parameters come from the data: provisional, measured on the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lengths: tuple[int, ...] = Field(
        default=(5, 7, 9, 11),
        description="Window lengths tried, in receivers, shortest first: short windows only, for "
        "the lateral detail; a line where none passes takes the one that passed most.",
    )
    trials: int = Field(
        default=27,
        ge=1,
        description="Trial windows per length, spread evenly along the whole line, its ends "
        "included: enough that their share passing G3 is the line's.",
    )
    min_pass_share: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Share of the trial windows G3 must pass for a length to be kept.",
    )
    max_line_share: float = Field(
        default=0.5, gt=0, le=1, description="Longest window, as a share of the line's receivers."
    )
    max_uncertainty: float = Field(
        default=0.2,
        gt=0,
        description="Median velocity uncertainty of the curves G3 passed at which the climb "
        "stops: short of it, the ladder keeps the most precise length that passes. On p2, 5 "
        "receivers passed with picks at 40 %, 7 at 30 %.",
    )
    min_precision_gain: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        description="A longer length is worth the lateral detail it costs while its picks are "
        "this much more precise than the best so far: past it the climb stops. On the demo, "
        "picks stay at 36 to 40 % from 7 to 11 receivers.",
    )


@dataclass(frozen=True)
class TrialJudge:
    """What judges the ladder's trial windows: the rules, the run's G2 (for the velocity grid),
    G3 and picking."""

    rules: CoherenceRules
    curve: CurveThresholds
    picking: PickingParameters
    image: ImageThresholds = field(default_factory=ImageThresholds)


class LengthTrial(BaseModel):
    """One length of the ladder, tried on a few windows."""

    model_config = ConfigDict(frozen=True)

    length: int
    xmids: tuple[float, ...]
    verdicts: tuple[str, ...]  # G3's, in the order of xmids
    flags: tuple[str, ...]  # G3's flags over the trials, most frequent first
    passed: int
    metres: float = 0.0  # the window's span
    windows: int = 0  # windows the line gets at this length, at the run's step
    # The median shortest and longest wavelengths of the curves G3 passed, m: the depth the
    # length reaches is about half the longest.
    wavelengths_m: tuple[float, float] | None = None
    # The median velocity uncertainty (G3's) of the curves G3 passed: the array's precision.
    uncertainty: float | None = None
    compared: bool = False  # tried past the kept length, for the agent to compare


class LengthChoice(BaseModel):
    """The window length the ladder kept, and why: its proposal, the agent's to change."""

    model_config = ConfigDict(frozen=True)

    length: int
    trials: tuple[LengthTrial, ...]
    notes: tuple[str, ...]
    receivers: int = 0  # the line's
    spacing_m: float = 0.0
    longest: int = 0  # the longest length allowed, in receivers


def cap_band(
    preset: ActivePreset | PassivePreset,
    usable: Sequence[tuple[float, float] | None],
    nyquist: float,
    given: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The dispersion band within the records' median usable band (G1) and below Nyquist:
    overrides for the dispersion stage (empty when the band already fits), and notes. The
    median, not the worst record: one record's narrow band would narrow the whole line's. An
    edge the user gave (`given`, the dispersion's) stays as given, said in a note."""
    dispersion = preset.model_dump()["dispersion"]
    mine = dict(given or {})
    known = [band for band in usable if band is not None]
    low = float(np.median([band[0] for band in known])) if known else 0.0
    high = min(float(np.median([band[1] for band in known])) if known else nyquist, nyquist)
    if high <= low:
        # No usable band left: no cap.
        return {}, (
            f"The records' median usable band is empty ({low:.1f} to {high:.1f} Hz): the band "
            "stays the preset's.",
        )
    overrides: dict[str, Any] = {}
    notes: list[str] = []
    for edge, outside in (("fmax", dispersion["fmax"] > high), ("fmin", dispersion["fmin"] < low)):
        if edge in mine and outside:
            notes.append(
                f"dispersion {edge} {mine[edge]:g} Hz, as given, lies outside the records' median "
                f"usable band ({low:.1f}-{high:.1f} Hz): kept as given."
            )
    if dispersion["fmax"] > high and "fmax" not in mine:
        overrides["fmax"] = round(high, 1)
        notes.append(
            f"dispersion fmax {dispersion['fmax']:g} Hz is above the records' median usable band "
            f"(up to {high:.1f} Hz): set to {high:.1f}."
        )
    if dispersion["fmin"] < low and "fmin" not in mine:
        overrides["fmin"] = round(low, 1)
        notes.append(
            f"dispersion fmin {dispersion['fmin']:g} Hz is below the records' median usable band "
            f"(from {low:.1f} Hz): set to {low:.1f}."
        )
    if "fmax" not in mine and overrides.get("fmax", dispersion["fmax"]) <= overrides.get(
        "fmin", dispersion["fmin"]
    ):
        # The preset's band lies wholly below what the records keep usable: capped, it would be
        # empty. The one case the band is widened: up to the usable band's top.
        overrides["fmax"] = round(high, 1)
        notes.append(
            f"dispersion fmax {dispersion['fmax']:g} Hz lies below the records' median usable band "
            f"({low:.1f}-{high:.1f} Hz): set to {high:.1f}."
        )
    return ({"dispersion": overrides} if overrides else {}), tuple(notes)


def choose_length(
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    records: tuple[RecordOutcome, ...],
    run_folder: Path,
    judge: TrialJudge,
    workers: int,
    first: int | None = None,
    exclusions: Exclusions | None = None,
) -> LengthChoice:
    """The window length, from trial windows spread along the line: up the ladder while G3
    passes `min_pass_share` of them and each length's picks are more precise than the best so
    far (`min_precision_gain`), the first length whose passed curves are precise
    (`max_uncertainty`), else the most precise that passes, the shorter on a tie (the most
    passes when none does), and one length past it for comparison; or `first`, a length given (by the user or the agent),
    kept as it is with its trial windows' result (the ladder proposes, a length given decides).
    A short window's picks may pass G3 yet be too loose for the inversion: precision is what
    longer windows buy, until the line's lateral changes blur their images. Trial windows go to
    <run_folder>/coherence/<length>/, on the run's records."""
    rules = judge.rules
    longest = max(3, int(len(profile.receivers) * rules.max_line_share))

    def attempt(length: int) -> LengthTrial | None:
        return _try_length(profile, preset, records, run_folder, judge, workers, length, exclusions)

    def passes(trial: LengthTrial) -> bool:
        return trial.passed >= math.ceil(rules.min_pass_share * len(trial.xmids))

    def precise(trial: LengthTrial) -> bool:
        return trial.uncertainty is not None and trial.uncertainty <= rules.max_uncertainty

    trials: list[LengthTrial] = []
    why = "as given"
    if first is not None:
        given = attempt(first)
        if given is None:
            raise RunError(
                f"No window of {first} receivers has a shot: check masw's length and distances."
            )
        trials.append(given)
        kept = given
    else:
        for length in [length for length in rules.lengths if length <= longest]:
            trial = attempt(length)
            if trial is None:
                continue
            trials.append(trial)
            if passes(trial) and precise(trial):
                break
            earlier = [before for before in trials[:-1] if passes(before)]
            if not passes(trial):
                if earlier:
                    break  # past the lengths that pass
                continue
            best = min(earlier, key=_loose, default=None)
            if best is not None and not _gains(trial, best, rules.min_precision_gain):
                break  # longer windows no longer buy precision: keep the lateral detail
        if not trials:
            raise RunError("No window length gives a window with a shot: check masw's distances.")
        passing = [trial for trial in trials if passes(trial)]
        if not passing:
            kept = max(trials, key=lambda trial: (trial.passed, -trial.length))
            why = f"the most passes (none passed {rules.min_pass_share:.0%})"
        elif precise(passing[-1]):
            kept = passing[-1]
            why = f"the first that passed with picks within {rules.max_uncertainty:.0%}"
        else:
            kept = min(passing, key=lambda trial: (_loose(trial), trial.length))
            why = f"the most precise that passed (none within {rules.max_uncertainty:.0%})"
        if passes(kept) and not any(trial.length > kept.length for trial in trials):
            # One length more, for the agent to weigh the depth it reaches against detail.
            longer = next(
                (length for length in rules.lengths if kept.length < length <= longest), None
            )
            compared = attempt(longer) if longer is not None else None
            if compared is not None:
                trials.append(compared)
        trials = [
            trial.model_copy(update={"compared": True}) if trial.length > kept.length else trial
            for trial in trials
        ]
    tried = ", ".join(_tried(trial) for trial in trials)
    note = f"masw length {kept.length} for the whole line, {why}: trial windows G3 passed {tried}."
    receivers = profile.receivers
    choice = LengthChoice(
        length=kept.length,
        trials=tuple(trials),
        notes=(note,),
        receivers=len(receivers),
        spacing_m=receiver_spacing(profile),
        longest=longest,
    )
    write_atomic(run_folder / COHERENCE_FILE, choice.model_dump_json(indent=2))
    return choice


def _tried(trial: LengthTrial) -> str:
    """A trial in the length note: "25/27 at 5 (picks 40%)", "14/27 at 16 (compared)"."""
    said = [f"picks {trial.uncertainty:.0%}"] if trial.uncertainty is not None else []
    said += ["compared"] if trial.compared else []
    return f"{trial.passed}/{len(trial.xmids)} at {trial.length}" + (
        f" ({'; '.join(said)})" if said else ""
    )


def _gains(trial: LengthTrial, best: LengthTrial, gain: float) -> bool:
    """Whether `trial`'s picks are at least `gain` more precise than `best`'s."""
    return (
        trial.uncertainty is not None
        and best.uncertainty is not None
        and trial.uncertainty <= best.uncertainty * (1 - gain)
    )


def _loose(trial: LengthTrial) -> float:
    """A trial's uncertainty, for sorting: unknown last."""
    return trial.uncertainty if trial.uncertainty is not None else math.inf


def read_length_choice(run_folder: Path) -> LengthChoice | None:
    """The ladder's choice for the run in `run_folder`, when it made one."""
    path = run_folder / COHERENCE_FILE
    return LengthChoice.model_validate_json(path.read_text()) if path.exists() else None


def describe_lengths(choice: LengthChoice) -> tuple[str, ...]:
    """The ladder's trials in words, for the agent to choose the window length from: the line,
    then each length tried with what its trial windows gave."""
    span = (choice.receivers - 1) * choice.spacing_m
    said = [
        f"line: {choice.receivers} receivers {choice.spacing_m:g} m apart ({span:.2f} m); "
        f"windows of up to {choice.longest} receivers (half the line)"
    ]
    for trial in choice.trials:
        text = (
            f"{trial.length} receivers ({trial.metres:.2f} m): {trial.passed}/{len(trial.xmids)} "
            "trial windows passed G3"
        )
        if trial.wavelengths_m is not None:
            # MASW's depth of investigation: about half the longest wavelength.
            text += (
                f", wavelengths {trial.wavelengths_m[0]:.1f}-{trial.wavelengths_m[1]:.1f} m "
                f"(models down to about {trial.wavelengths_m[1] / 2:.1f} m)"
            )
        if trial.uncertainty is not None:
            text += f", picks within {trial.uncertainty:.0%}"
        text += f", {trial.windows} windows on the line"
        if trial.length == choice.length:
            text += " (proposed)"
        said.append(text)
    return tuple(said)


def length_hint(choice: LengthChoice, then: str) -> str:
    """The ladder's length, said as kept, with the depth its trial curves reach, then `then`, the
    next step. The length whose curves are best stays whatever depth or lateral detail a request
    asks: the curves first."""
    kept = next((trial for trial in choice.trials if trial.length == choice.length), None)
    depth = (
        f", its trial curves reaching about {kept.wavelengths_m[1] / 2:.1f} m deep"
        if kept is not None and kept.wavelengths_m is not None
        else ""
    )
    return (
        f"The window length is the ladder's ({choice.length} receivers), the best curves of the "
        f"lengths tried{depth}: kept, whatever depth or detail the request asks. {then}"
    )


def receiver_spacing(profile: Profile) -> float:
    """The receivers' spacing along the line, m."""
    positions = sorted(receiver.x for receiver in profile.receivers)
    return round(float(np.median(np.diff(positions))), 3) if len(positions) > 1 else 0.0


def _try_length(
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    records: tuple[RecordOutcome, ...],
    run_folder: Path,
    judge: TrialJudge,
    workers: int,
    length: int,
    exclusions: Exclusions | None = None,
) -> LengthTrial | None:
    """S2, the picking and G3 on `trials` windows of `length` receivers spread along the line;
    None when no window of that length has a shot."""
    trial_preset = resolve_preset(
        apply_overrides(preset, {"masw": {"length": length, "step": 1}}), profile
    )
    windows = build_windows(profile, trial_preset.masw)
    if not windows:
        return None
    chosen = [windows[index] for index in trial_indices(len(windows), judge.rules.trials)]
    tried = try_windows(
        trial_preset,
        profile,
        chosen,
        records,
        run_folder / RECORDS_FOLDER,
        run_folder / TRIALS_FOLDER / f"{length}",
        judge,
        workers,
        exclusions,
    )
    on_line = resolve_preset(apply_overrides(preset, {"masw": {"length": length}}), profile)
    return LengthTrial(
        length=length,
        xmids=tried.xmids,
        verdicts=tried.verdicts,
        flags=tried.flags,
        passed=tried.passed,
        metres=round((length - 1) * receiver_spacing(profile), 2),
        windows=len(build_windows(profile, on_line.masw)),
        wavelengths_m=tried.wavelengths_m,
        uncertainty=tried.uncertainty,
    )


@dataclass(frozen=True)
class TriedWindows:
    """What trial windows gave: G3's verdict on each ("failed": no image), the flags it raised
    (most frequent first), and over the curves it passed, the median shortest and longest
    wavelengths, m, the median band, Hz, and the median uncertainty."""

    xmids: tuple[float, ...]
    verdicts: tuple[str, ...]
    flags: tuple[str, ...]
    wavelengths_m: tuple[float, float] | None
    uncertainty: float | None
    band_hz: tuple[float, float] | None = None

    @property
    def passed(self) -> int:
        return self.verdicts.count("pass")


def try_windows(
    trial_preset: ActivePreset | PassivePreset,
    profile: Profile,
    windows: list[MASWWindow],
    records: tuple[RecordOutcome, ...],
    records_folder: Path,
    folder: Path,
    judge: TrialJudge,
    workers: int,
    exclusions: Exclusions | None = None,
) -> TriedWindows:
    """S2 on `windows` from the preprocessed records in `records_folder`, into `folder` (made
    afresh), their grids fixed as G2 asks (_fix_grids), then the picking and G3 on each: the
    trials of the window length and of the muting alike."""
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    outcomes = process_windows(
        trial_preset,
        windows,
        records,
        folder,
        workers,
        records_folder=records_folder,
        exclusions=exclusions,
        stop=stopping.current(),
    )
    outcomes = _fix_grids(
        trial_preset,
        profile,
        windows,
        outcomes,
        records,
        records_folder,
        folder,
        judge,
        workers,
        exclusions,
    )
    verdicts: list[str] = []
    # A passive line has no muting: a flag only a mute fixes rejects (G3).
    mutable = "muting" in type(trial_preset).model_fields
    flags: dict[str, int] = {}
    ranges: list[tuple[float, float]] = []
    bands: list[tuple[float, float]] = []
    uncertainties: list[float] = []
    for outcome in outcomes:
        if outcome.status != "succeeded":
            verdicts.append("failed")
            continue
        image = load_image(folder / outcome.folder)
        modes = pick_modes(image, judge.picking)
        m0 = modes[0] if modes else None
        offset = nearest_offset(folder / outcome.folder)
        g3 = judge_curve(
            outcome.folder, image, m0, judge.curve, None, judge.picking, offset, mutable
        )
        verdicts.append(g3.verdict)
        for flag in g3.flags:
            flags[flag.name] = flags.get(flag.name, 0) + 1
        if g3.verdict == "pass" and g3.kept.wavelength_m is not None:
            ranges.append(g3.kept.wavelength_m)
        if g3.verdict == "pass" and g3.kept.band_hz is not None:
            bands.append(g3.kept.band_hz)
        if g3.verdict == "pass":
            uncertainties += [
                metric.value
                for metric in g3.metrics
                if metric.name == "uncertainty" and metric.value is not None
            ]
    return TriedWindows(
        xmids=tuple(outcome.xmid for outcome in outcomes),
        verdicts=tuple(verdicts),
        flags=tuple(sorted(flags, key=lambda name: -flags[name])),
        wavelengths_m=_medians(ranges),
        uncertainty=round(float(np.median(uncertainties)), 3) if uncertainties else None,
        band_hz=_medians(bands),
    )


def _medians(ranges: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """The median low and high ends of `ranges`, rounded to 0.1; None without any."""
    if not ranges:
        return None
    return (
        round(float(np.median([low for low, _ in ranges])), 1),
        round(float(np.median([high for _, high in ranges])), 1),
    )


def trial_indices(n_windows: int, trials: int) -> list[int]:
    """`trials` windows spread evenly along the whole line, its ends included: without the ends,
    the trials pass G3 more often than the line's windows do."""
    positions = np.linspace(0, n_windows - 1, trials).round().astype(int)
    return [int(index) for index in np.unique(positions)]


def nearest_offset(window_folder: Path) -> float | None:
    """The distance from the window's nearest shot to its nearest receiver, over the records it
    uses (window.json, sigpipe's nearest_offset); None when the window has none."""
    return window_nearest_offset(
        MASWWindow.model_validate_json((window_folder / "window.json").read_text())
    )


def _fix_grids(
    preset: ActivePreset | PassivePreset,
    profile: Profile,
    windows: list[MASWWindow],
    outcomes: tuple[WindowOutcome, ...],
    records: tuple[RecordOutcome, ...],
    records_folder: Path,
    folder: Path,
    judge: TrialJudge,
    workers: int,
    exclusions: Exclusions | None,
) -> tuple[WindowOutcome, ...]:
    """The trial images whose ridge G2 finds on the velocity grid's edge, made again with the
    range G2 asks for (twice at most): a grid too narrow for the ground is no fault of the
    window's length, and must not send the ladder to longer windows."""
    by_folder = {f"xmid_{window.xmid:.2f}": window for window in windows}
    current = {outcome.folder: outcome for outcome in outcomes}
    changes: dict[str, dict[str, Any]] = {}
    for _ in range(2):
        groups: dict[str, tuple[dict[str, Any], list[MASWWindow]]] = {}
        for name, outcome in current.items():
            if outcome.status != "succeeded":
                continue
            g2 = judge_image(name, load_image(folder / name), judge.image)
            asked = {}
            for flag in g2.flags:
                if flag.name in ("ridge_at_vmin", "ridge_at_vmax") and isinstance(
                    flag.action, Override
                ):
                    asked = deep_merge(asked, flag.action.overrides)
            if asked:
                changes[name] = deep_merge(changes.get(name, {}), asked)
                key = json.dumps(changes[name], sort_keys=True)
                groups.setdefault(key, (changes[name], []))[1].append(by_folder[name])
        if not groups:
            break
        for values, group in groups.values():
            again = resolve_preset(apply_overrides(preset, values), profile)
            for outcome in process_windows(
                again,
                group,
                records,
                folder,
                workers,
                records_folder=records_folder,
                exclusions=exclusions,
                stop=stopping.current(),
            ):
                current[outcome.folder] = outcome
    return tuple(sorted(current.values(), key=lambda outcome: outcome.xmid))


def given_length(overrides: Mapping[str, object] | None) -> int | None:
    """The window length the user gave in `overrides`, if any."""
    masw = (overrides or {}).get("masw")
    if isinstance(masw, Mapping):
        length = cast(Mapping[str, object], masw).get("length")
        if isinstance(length, int):
            return length
    return None


def near_field(
    overrides: Mapping[str, object] | None, mode: str, choice: LengthChoice
) -> tuple[float | None, float | None]:
    """How far from a window's nearest receiver a shot must stand to be out of the near field:
    half the longest wavelength the line's trial curves reached (Park et al.: nearer, the wave
    is not yet a plane surface wave and the long wavelengths read slow), and that wavelength.
    None for shots not imaged as recorded (another mode than active), and where the user gave
    masw.distance_min: theirs rules."""
    masw = (overrides or {}).get("masw")
    if mode != "active" or (isinstance(masw, Mapping) and "distance_min" in masw):
        return None, None
    kept = next((trial for trial in choice.trials if trial.length == choice.length), None)
    if kept is None or kept.wavelengths_m is None:
        return None, None
    longest = kept.wavelengths_m[1]
    return round(longest / 2, 2), longest


def near_field_windows(
    windows: Sequence[MASWWindow], distance_m: float
) -> tuple[list[MASWWindow], int]:
    """`windows` without their shots nearer than `distance_m` to their nearest receiver, where
    they keep a farther one: a window with near shots only keeps them (G3 flags its near
    field). Returns the windows, and how many kept near shots only."""
    found: list[MASWWindow] = []
    near_only = 0
    for window in windows:
        xs = [receiver.x for receiver in window.acquisitions[0].receivers]
        first, last = min(xs), max(xs)
        far = [
            index
            for index, acquisition in enumerate(window.acquisitions)
            if max(first - acquisition.source.x, acquisition.source.x - last) >= distance_m
        ]
        if not far:
            near_only += 1
            found.append(window)
            continue
        found.append(
            window.model_copy(
                update={
                    "selected_files": [window.selected_files[index] for index in far],
                    "acquisitions": [window.acquisitions[index] for index in far],
                }
            )
        )
    return found, near_only


def near_note(distance_m: float, longest: float, near_only: int) -> str:
    """The near-field rule, as the line's note says it."""
    return (
        f"near_field distance_m {distance_m:g} m: a window stacks no shot nearer than that to its "
        f"nearest receiver, half the longest wavelength its trial curves reached ({longest:g} m), "
        "where it has a farther one"
        + (
            f"; {near_only} windows with near shots only keep them (G3 flags their near field)"
            if near_only
            else ""
        )
        + "."
    )
