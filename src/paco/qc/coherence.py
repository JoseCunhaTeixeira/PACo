"""The coherence rules for S2 (docs/qc_workflow.md), checked before the phase shift runs: the
image's band within what G1 found usable in the records (capped, never widened: a decision of
milestone 13), and one window length for the whole line, the shortest at which most trial
windows give a curve G3 passes (lateral resolution first)."""

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from paco.picking import PickingParameters, pick_modes
from paco.presets import ActivePreset, PassivePreset, apply_overrides, resolve_preset
from paco.profiles import Profile
from paco.qc.g2_image import ImageThresholds, judge_image
from paco.qc.g3_curve import CurveThresholds, judge_curve
from paco.qc.loops import deep_merge
from paco.qc.models import Override
from paco.runs import RecordOutcome, RunError, WindowOutcome, load_image
from paco.runs.processing import RECORDS_FOLDER, process_windows
from paco.windows import Exclusions, MASWWindow, build_windows

TRIALS_FOLDER = "coherence"  # inside the run folder: the ladder's trial windows, by length
COHERENCE_FILE = "coherence.json"


class CoherenceRules(BaseModel):
    """How S2's parameters come from the data: provisional, measured on the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lengths: tuple[int, ...] = Field(
        default=(5, 7, 9, 11, 16, 24, 32, 48, 64, 96, 128),
        description="Window lengths tried, in receivers, shortest first: the short ones the "
        "user works with, then longer ones for a line where none of them passes.",
    )
    trials: int = Field(default=9, ge=1, description="Trial windows per length, along the line.")
    min_pass_share: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Share of the trial windows G3 must pass for a length to be kept.",
    )
    max_line_share: float = Field(
        default=0.5, gt=0, le=1, description="Longest window, as a share of the line's receivers."
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
    compared: bool = False  # tried past the kept length, for the agent to compare


class LengthChoice(BaseModel):
    """The window length the ladder kept, and why: its proposal, the agent's to change (the
    user's decision of 2026-09-25)."""

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
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The dispersion band within what every record keeps usable (G1) and below Nyquist:
    overrides for the dispersion stage (empty when the band already fits), and notes."""
    dispersion = preset.model_dump()["dispersion"]
    known = [band for band in usable if band is not None]
    low = max((band[0] for band in known), default=0.0)
    high = min([band[1] for band in known] + [nyquist])
    overrides: dict[str, Any] = {}
    notes: list[str] = []
    if dispersion["fmax"] > high:
        overrides["fmax"] = round(high, 1)
        notes.append(
            f"dispersion fmax {dispersion['fmax']:g} Hz is above the records' usable band "
            f"(up to {high:.1f} Hz): set to {high:.1f}."
        )
    if dispersion["fmin"] < low:
        overrides["fmin"] = round(low, 1)
        notes.append(
            f"dispersion fmin {dispersion['fmin']:g} Hz is below the records' usable band "
            f"(from {low:.1f} Hz): set to {low:.1f}."
        )
    if overrides.get("fmax", dispersion["fmax"]) <= overrides.get("fmin", dispersion["fmin"]):
        # The preset's band lies wholly below what the records keep usable: capped, it would be
        # empty. The one case the band is widened: up to the usable band's top.
        overrides["fmax"] = round(high, 1)
        notes.append(
            f"dispersion fmax {dispersion['fmax']:g} Hz lies below the records' usable band "
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
    """The shortest window length up the ladder at which G3 passes `min_pass_share` of the
    trial windows spread along the line (the most passes when none does), and one length more
    for comparison; or `first`, a length given (by the user or the agent), kept as it is with
    its trial windows' result (the user's decision of 2026-09-25: the ladder proposes, a
    length given decides). Trial windows go to <run_folder>/coherence/<length>/, on the run's
    records."""
    rules = judge.rules
    longest = max(3, int(len(profile.receivers) * rules.max_line_share))

    def attempt(length: int) -> LengthTrial | None:
        return _try_length(profile, preset, records, run_folder, judge, workers, length, exclusions)

    def passes(trial: LengthTrial) -> bool:
        return trial.passed >= math.ceil(rules.min_pass_share * len(trial.xmids))

    trials: list[LengthTrial] = []
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
            if passes(trial):
                break
        if not trials:
            raise RunError("No window length gives a window with a shot: check masw's distances.")
        kept = trials[-1]
        if not passes(kept):
            kept = max(trials, key=lambda trial: (trial.passed, -trial.length))
        else:
            # One length more, for the agent to weigh the depth it reaches against detail.
            longer = next(
                (length for length in rules.lengths if kept.length < length <= longest), None
            )
            compared = attempt(longer) if longer is not None else None
            if compared is not None:
                trials.append(compared.model_copy(update={"compared": True}))
    tried = ", ".join(
        f"{trial.passed}/{len(trial.xmids)} at {trial.length}{' (compared)' if trial.compared else ''}"
        for trial in trials
    )
    how = "as given" if first is not None else "by length in receivers"
    note = f"masw length {kept.length} for the whole line: trial windows G3 passed, {how}: {tried}."
    receivers = profile.receivers
    choice = LengthChoice(
        length=kept.length,
        trials=tuple(trials),
        notes=(note,),
        receivers=len(receivers),
        spacing_m=_spacing(profile),
        longest=longest,
    )
    (run_folder / COHERENCE_FILE).write_text(choice.model_dump_json(indent=2))
    return choice


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
            text += f", wavelengths {trial.wavelengths_m[0]:.1f}-{trial.wavelengths_m[1]:.1f} m"
        text += f", {trial.windows} windows on the line"
        if trial.length == choice.length:
            text += " (proposed)"
        said.append(text)
    return tuple(said)


def _spacing(profile: Profile) -> float:
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
    picks = np.unique(np.linspace(0, len(windows) - 1, judge.rules.trials).round().astype(int))
    chosen = [windows[index] for index in picks]
    folder = run_folder / TRIALS_FOLDER / f"{length}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    outcomes = process_windows(
        trial_preset,
        chosen,
        records,
        folder,
        workers,
        records_folder=run_folder / RECORDS_FOLDER,
        exclusions=exclusions,
    )
    outcomes = _fix_grids(
        trial_preset,
        profile,
        chosen,
        outcomes,
        records,
        run_folder,
        folder,
        judge,
        workers,
        exclusions,
    )
    verdicts: list[str] = []
    flags: dict[str, int] = {}
    ranges: list[tuple[float, float]] = []
    for outcome in outcomes:
        if outcome.status != "succeeded":
            verdicts.append("failed")
            continue
        image = load_image(folder / outcome.folder)
        modes = pick_modes(image, judge.picking)
        m0 = modes[0] if modes else None
        offset = nearest_offset(folder / outcome.folder)
        g3 = judge_curve(outcome.folder, image, m0, judge.curve, None, judge.picking, offset)
        verdicts.append(g3.verdict)
        for flag in g3.flags:
            flags[flag.name] = flags.get(flag.name, 0) + 1
        if g3.verdict == "pass" and g3.kept.wavelength_m is not None:
            ranges.append(g3.kept.wavelength_m)
    on_line = resolve_preset(apply_overrides(preset, {"masw": {"length": length}}), profile)
    return LengthTrial(
        length=length,
        xmids=tuple(outcome.xmid for outcome in outcomes),
        verdicts=tuple(verdicts),
        flags=tuple(sorted(flags, key=lambda name: -flags[name])),
        passed=verdicts.count("pass"),
        metres=round((length - 1) * _spacing(profile), 2),
        windows=len(build_windows(profile, on_line.masw)),
        wavelengths_m=(
            (
                round(float(np.median([low for low, _ in ranges])), 1),
                round(float(np.median([high for _, high in ranges])), 1),
            )
            if ranges
            else None
        ),
    )


def nearest_offset(window_folder: Path) -> float | None:
    """The distance from the window's nearest shot to its nearest receiver, over the records it
    uses (window.json); None when the window has none."""
    window = MASWWindow.model_validate_json((window_folder / "window.json").read_text())
    distances = [
        math.dist(
            (acquisition.source.x, acquisition.source.y, acquisition.source.z),
            (receiver.x, receiver.y, receiver.z),
        )
        for acquisition in window.acquisitions
        for receiver in acquisition.receivers
    ]
    return round(min(distances), 3) if distances else None


def _fix_grids(
    preset: ActivePreset | PassivePreset,
    profile: Profile,
    windows: list[MASWWindow],
    outcomes: tuple[WindowOutcome, ...],
    records: tuple[RecordOutcome, ...],
    run_folder: Path,
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
                records_folder=run_folder / RECORDS_FOLDER,
                exclusions=exclusions,
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
