"""The mute trial (docs/qc_workflow.md): whether a surface-wave
mute makes better dispersion images, measured once the line's window length is chosen, at that
length: a mute changes the images, never the windows.

The cone comes from the shots' gathers (sigpipe's masw.quality.cone: the interquartile range
of the traces' envelope peaks, from the shot, and their median). Five candidates, from the
least cutting to the most, each on the same windows spread along the line, their records
preprocessed with it, imaged, picked and judged by G3 as the window-length ladder does: no
mute; the standard 80-1500 m/s mute G2 and G3 repair with; the cone widened by the square of
`widening` each side; the cone widened by `widening`; and the tight cone, the median peak
velocity divided and multiplied by `widening` (strong refractions pull the peaks' fast
quartile up; their median holds). The most windows passing G3 wins, the tightest mute on a tie
(the narrowest velocity window). Before it, the line's trigger where the first breaks put the
shots, when G1 finds them off by more than its limit: a mute cuts around the shot. Each muted
candidate's records judged by G1 as the line's are once a mute is kept, so that the windows each
counts are those the line will have. Active and passive-active lines (a passive line has no
muting); never a muting the user gave, which stays as given."""

import math
import statistics
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.masw.pipelines.preprocessing import shot_time_s, unmuted_record
from sigpipe.masw.presets import ActivePreset, PassivePreset, apply_overrides, resolve_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.quality.cone import Cone, peak_velocities, surface_wave_cone
from sigpipe.masw.runs import RecordOutcome
from sigpipe.masw.runs.processing import RECORDS_FOLDER, preprocess_records
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions, build_windows

from paco import stopping
from paco.qc.coherence import TrialJudge, receiver_spacing, trial_indices, try_windows
from paco.qc.models import GateResult

TRIALS_FOLDER = "mute_trials"  # inside the run folder: each candidate's records and windows
MUTE_FILE = "mute.json"
# From the least cutting to the most: ties go to the earlier.
type Candidate = Literal["none", "standard", "wider", "cone", "tight"]
# G1 on a candidate's muted records (its trial folder, its preset, its records), as the line
# judges its records once a mute is kept: the traces and records it leaves out.
type Settle = Callable[[Path, ActivePreset | PassivePreset, tuple[RecordOutcome, ...]], Exclusions]


class MuteRules(BaseModel):
    """How the mute trial decides, measured on PAC's active test line."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    length: int = Field(
        default=9,
        ge=3,
        description="Receivers of the trial windows when the user gave no length: the "
        "window-length ladder's middle.",
    )
    records: int = Field(
        default=12, ge=1, description="Records the cone is estimated from, spread on the line."
    )
    min_offset_spacings: float = Field(
        default=3.0,
        ge=0,
        description="Traces nearer the shot than this many receiver spacings are left out of "
        "the cone: their envelope peaks sit in the near field.",
    )
    widening: float = Field(
        default=1.5,
        gt=1,
        description="The cone's interquartile range widened by this factor each side: the "
        "wave train's onset and tail around its peaks; the wider candidate by its square; the "
        "tight one, the median peak velocity divided and multiplied by it.",
    )
    standard: tuple[float, float] = Field(
        default=(80.0, 1500.0), description="The standard mute's velocities, m/s: G2's and G3's."
    )
    width_s: float = Field(
        default=0.05,
        ge=0,
        description="The mute's width after its slow edge, s: the wave train's tail (PACo's "
        "pulse width before G1 measures one).",
    )
    taper_s: float = Field(default=0.01, ge=0, description="The mute's taper, s.")


class MuteTrial(BaseModel):
    """One candidate on the trial windows: its muting, G3's verdicts, and over the curves G3
    passed, their median shortest and longest wavelengths, m."""

    model_config = ConfigDict(frozen=True)

    candidate: Candidate
    muting: dict[str, Any] | None  # the muting stage it ran with; None: no mute
    verdicts: tuple[str, ...]
    passed: int
    flags: tuple[str, ...] = ()
    wavelengths_m: tuple[float, float] | None = None
    # Not kept: why, for a mute that had the most windows passing.
    refused: str | None = None


class MuteChoice(BaseModel):
    """The trial's outcome: the candidate kept and its muting (None: no mute), the cone the
    gathers gave, the trigger every candidate ran with, the trial windows, every candidate's
    result, and the line's note."""

    model_config = ConfigDict(frozen=True)

    chosen: Candidate
    muting: dict[str, Any] | None
    cone_m_s: tuple[float, float, float] | None  # its slow edge, median and fast edge
    # The trigger stage set before the trial, where the first breaks put the shots; None: the
    # records' files'.
    trigger: dict[str, Any] | None = None
    length: int
    xmids: tuple[float, ...]
    trials: tuple[MuteTrial, ...]
    notes: tuple[str, ...]


def mutable(preset: ActivePreset | PassivePreset) -> bool:
    """Whether `preset`'s line can be muted: its shots' records (active, passive-active)."""
    return "muting" in type(preset).model_fields


def given_muting(overrides: Mapping[str, object] | None) -> bool:
    """Whether the user gave a muting in `overrides`: it stays as given, no trial."""
    return isinstance((overrides or {}).get("muting"), Mapping)


def choose_mute(
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    run_folder: Path,
    judge: TrialJudge,
    rules: MuteRules,
    workers: int,
    cone: Cone,
    own: Mapping[str, Mapping[str, Any]] | None = None,
    exclusions: Exclusions | None = None,
    settle: Settle | None = None,
    trigger: Mapping[str, Any] | None = None,
) -> MuteChoice | None:
    """The mute trial on `profile`'s line, with `preset` (resolved, unmuted, at the line's
    window length) and the gathers' `cone`: each record with its `own` changes (G1's, by record
    name); without a mute, the traces and records G1 left out (`exclusions`) left out, with one,
    those G1 leaves out of its muted records (`settle`), the records it left out before staying
    out, as the line preprocesses them no more (without `settle`, `exclusions`); `trigger`, the
    trigger stage `preset` holds when set before the trial. Its outcome saved in the run's
    mute.json; None when the line has no trial window."""
    length = preset.masw.length
    trial_preset = resolve_preset(apply_overrides(preset, {"masw": {"step": 1}}), profile)
    windows = build_windows(profile, trial_preset.masw)
    if not windows:
        return None
    # The ladder's sample windows at the line's length.
    chosen = [windows[index] for index in trial_indices(len(windows), judge.rules.trials)]
    needed = sorted({path.name for window in chosen for path in window.selected_files})
    trials: list[MuteTrial] = []
    for candidate, muting in mute_candidates(cone, rules, profile).items():
        stopping.check()
        tried_preset = (
            resolve_preset(apply_overrides(trial_preset, {"muting": muting}), profile)
            if muting is not None
            else trial_preset
        )
        folder = run_folder / TRIALS_FOLDER / candidate
        records = preprocess_records(
            tried_preset,
            profile,
            folder,
            workers,
            presets={
                name: resolve_preset(
                    apply_overrides(tried_preset, dict((own or {}).get(name, {}))), profile
                )
                for name in needed
            },
            stop=stopping.current(),
        )
        # The records as G1 leaves them on the line with this muting: judged again once muted,
        # those it left out before staying out.
        judged = exclusions
        if settle is not None and muting is not None:
            judged = settle(folder, tried_preset, records)
            for name in exclusions.records if exclusions is not None else ():
                judged = judged.with_record(name)
        tried = try_windows(
            tried_preset,
            profile,
            chosen,
            records,
            folder / RECORDS_FOLDER,
            folder / "windows",
            judge,
            workers,
            judged,
        )
        trials.append(
            MuteTrial(
                candidate=candidate,
                muting=muting,
                verdicts=tried.verdicts,
                passed=tried.passed,
                flags=tried.flags,
                wavelengths_m=tried.wavelengths_m,
            )
        )
    kept, trials = _decide(trials)
    choice = MuteChoice(
        chosen=kept.candidate,
        muting=kept.muting,
        cone_m_s=(cone.vmin, cone.median, cone.vmax),
        trigger=dict(trigger) if trigger else None,
        length=length,
        xmids=tuple(window.xmid for window in chosen),
        trials=tuple(trials),
        notes=(_note(kept, trials, cone, length, receiver_spacing(profile)),),
    )
    write_atomic(run_folder / MUTE_FILE, choice.model_dump_json(indent=2))
    return choice


def line_trigger(
    results: Mapping[str, GateResult], triggers: Mapping[str, float | None], limit_s: float
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The line's trigger before its mute trial, and its note: where the first breaks put the
    shots, the median over the records G1 judged (`results`, unmuted: their trigger error
    measured, not judged) of their file's trigger (`triggers`) and that error; none when the
    median error is within G1's `limit_s` (the files' triggers stay). Never before the records'
    start."""
    errors: list[float] = []
    shots: list[float] = []
    for name, result in results.items():
        measured = next(
            (metric.value for metric in result.metrics if metric.name == "trigger_error_s"), None
        )
        if isinstance(measured, int | float) and math.isfinite(measured):
            errors.append(float(measured))
            shots.append((triggers.get(name) or 0.0) + float(measured))
    if not errors or abs(error := statistics.median(errors)) <= limit_s:
        return {}, ()
    t0 = round(max(statistics.median(shots), 0.0), 4)
    side = "before" if error < 0 else "after"
    return {"trigger": {"t0": t0}}, (
        f"trigger t0 {t0:g} s: where the first breaks put the shots, {abs(error) * 1000:.0f} ms "
        f"{side} their files' trigger (the median of the {len(errors)} records G1 kept), set "
        "before the mute trial so that every mute cuts around the shot.",
    )


def read_mute_choice(run_folder: Path) -> MuteChoice | None:
    """The run's mute trial, None when it had none (a passive line, a muting given)."""
    path = run_folder / MUTE_FILE
    return MuteChoice.model_validate_json(path.read_text()) if path.exists() else None


def describe_mutes(choice: MuteChoice) -> tuple[str, ...]:
    """The candidates the trial compared, one line each, for the agent: their muting, the trial
    windows G3 passed, the wavelengths reached, and why a mute was not kept."""
    lines: list[str] = []
    for trial in choice.trials:
        reach = (
            f", wavelengths {trial.wavelengths_m[0]:g} to {trial.wavelengths_m[1]:g} m"
            if trial.wavelengths_m is not None
            else ""
        )
        mark = " (kept)" if trial.candidate == choice.chosen else ""
        why = f"; not kept: {trial.refused}" if trial.refused else ""
        lines.append(
            f"{_named(trial)}{mark}: {trial.passed} of {len(trial.verdicts)} trial windows "
            f"passed G3{reach}{why}"
        )
    return tuple(lines)


def estimate_cone(
    profile: Profile, preset: ActivePreset | PassivePreset, rules: MuteRules
) -> Cone | None:
    """The surface waves' cone of `rules.records` records spread along the line, from their
    gathers before any muting; None when they give too few traces."""
    count = len(profile.records)
    picked = sorted(
        {round(index) for index in np.linspace(0, count - 1, min(rules.records, count))}
    )
    nearest = rules.min_offset_spacings * receiver_spacing(profile)
    velocities = []
    for index in picked:
        record = profile.records[index]
        stream = unmuted_record(preset, record, profile)
        velocities.append(peak_velocities(stream, shot_time_s(preset, record), nearest))
    return surface_wave_cone(velocities)


def mute_candidates(
    cone: Cone, rules: MuteRules, profile: Profile
) -> dict[Candidate, dict[str, Any] | None]:
    """Each candidate's muting stage, from the least cutting to the most: none, the standard,
    the wider cone, the cone, the tight cone."""
    sampling_hz = profile.sampling_rate_hz

    def mute(vmin: float, vmax: float) -> dict[str, Any]:
        return {
            "method": "mute",
            "vmin": round(vmin, 1),
            "vmax": round(vmax, 1),
            "width": rules.width_s,
            "taper": max(0, round(rules.taper_s * sampling_hz)),
        }

    factor = rules.widening
    return {
        "none": None,
        "standard": mute(*rules.standard),
        "wider": mute(cone.vmin / factor**2, cone.vmax * factor**2),
        "cone": mute(cone.vmin / factor, cone.vmax * factor),
        "tight": mute(cone.median / factor, cone.median * factor),
    }


def _decide(trials: list[MuteTrial]) -> tuple[MuteTrial, list[MuteTrial]]:
    """The candidate kept, and the trials: the most windows passing G3, the tightest mute on a
    tie (the narrowest velocity window, its vmax over its vmin; no mute the widest)."""

    def rank(trial: MuteTrial) -> tuple[int, float]:
        muting = trial.muting
        low = float(muting["vmin"]) if muting is not None else 0.0
        width = float(muting["vmax"]) / low if muting is not None and low > 0 else math.inf
        return trial.passed, -width

    return max(trials, key=rank), trials


def _named(trial: MuteTrial) -> str:
    muting = trial.muting
    if muting is None:
        return "no mute"
    names = {
        "standard": "the standard mute",
        "wider": "the wider cone",
        "cone": "the cone",
        "tight": "the tight cone",
    }
    return f"{names[trial.candidate]} {muting['vmin']:g} to {muting['vmax']:g} m/s"


def _note(
    kept: MuteTrial,
    trials: list[MuteTrial],
    cone: Cone,
    length: int,
    spacing: float,
) -> str:
    """The line's note on the muting: what was kept and why, against the others."""
    none = trials[0]
    windows = (
        f"{len(kept.verdicts)} trial windows of {length} receivers ({(length - 1) * spacing:g} m)"
    )
    others = "; ".join(f"{_named(trial)} {trial.passed}" for trial in trials if trial is not kept)
    gathers = (
        f"the gathers' surface waves {cone.vmin:g} to {cone.vmax:g} m/s, median {cone.median:g}"
    )
    if kept.muting is None:
        return (
            f"muting none: the mute trial ({gathers}): no mute, {none.passed} of {windows} "
            f"passing G3, kept; {others}."
        )
    return (
        f"muting mute {kept.muting['vmin']:g} to {kept.muting['vmax']:g} m/s: the mute trial "
        f"({gathers}): {_named(kept)}, {kept.passed} of {windows} passing G3, against "
        f"{none.passed} without a mute; {others}."
    )


def changed_muting(choice: MuteChoice | None) -> dict[str, Any] | None:
    """The overrides the trial's choice adds to the line's preset; None when it keeps no mute."""
    if choice is None or choice.muting is None:
        return None
    return {"muting": dict(cast(Mapping[str, Any], choice.muting))}
