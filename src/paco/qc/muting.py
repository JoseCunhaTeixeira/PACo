"""The mute trial (L10 of PACo's agent guidelines; docs/qc_workflow.md): whether a surface-wave
mute makes better dispersion images, measured once the line's window length is chosen, at that
length: a mute changes the images, never the windows.

The cone comes from the shots' gathers (sigpipe's masw.quality.cone: the interquartile range
of the traces' envelope peaks, from the shot, and their median). Five candidates, from the
least cutting to the most, each on the same windows spread along the line, their records
preprocessed with it, imaged, picked and judged by G3 as the window-length ladder does: no
mute; the standard 80-1500 m/s mute G2 and G3 repair with; the cone widened by the square of
`widening` each side; the cone widened by `widening`; and the tight cone, the median peak
velocity divided and multiplied by `widening` (strong refractions pull the peaks' fast
quartile up; their median holds). The most windows passing G3 wins, ties to the earlier, the
less cutting. A mute is kept only with `min_gain` more windows passing than no mute, and
without losing the long wavelengths (`max_wavelength_loss`): over-muting cuts the far offsets
and the low band the image needs. Active and passive-active lines (a passive line has no
muting); never a muting the user gave, which stays as given."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.masw.pipelines.preprocessing import shot_time_s, unmuted_record
from sigpipe.masw.presets import ActivePreset, PassivePreset, apply_overrides, resolve_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.quality.cone import Cone, peak_velocities, surface_wave_cone
from sigpipe.masw.runs.processing import RECORDS_FOLDER, preprocess_records
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions, build_windows

from paco import stopping
from paco.qc.coherence import TrialJudge, receiver_spacing, trial_indices, try_windows

TRIALS_FOLDER = "mute_trials"  # inside the run folder: each candidate's records and windows
MUTE_FILE = "mute.json"
# From the least cutting to the most: ties go to the earlier.
type Candidate = Literal["none", "standard", "wider", "cone", "tight"]


class MuteRules(BaseModel):
    """How the mute trial decides: provisional, measured on active_p1 and active_p2."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trials: int = Field(default=15, ge=3, description="Windows each candidate is tried on.")
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
    min_gain: int = Field(
        default=2,
        ge=1,
        description="Windows passing G3 a mute must add over no mute: one is within what the "
        "trial windows vary by.",
    )
    max_wavelength_loss: float = Field(
        default=0.2,
        ge=0,
        lt=1,
        description="The share of no mute's median longest wavelength a mute may lose: more, "
        "it cut the far offsets or the low band (over-muting).",
    )
    guard_share: float = Field(
        default=1 / 3,
        gt=0,
        le=1,
        description="The share of the trial windows no mute must pass for its longest "
        "wavelengths to judge a mute's: fewer curves say little of what the line reaches.",
    )


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
    gathers gave, the trial windows, every candidate's result, and the line's note."""

    model_config = ConfigDict(frozen=True)

    chosen: Candidate
    muting: dict[str, Any] | None
    cone_m_s: tuple[float, float, float] | None  # its slow edge, median and fast edge
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
) -> MuteChoice | None:
    """The mute trial on `profile`'s line, with `preset` (resolved, unmuted, at the line's
    window length) and the gathers' `cone`: each record with its `own` changes (G1's, by record
    name), the traces and records G1 left out left out. Its outcome saved in the run's
    mute.json; None when the line has no trial window."""
    length = preset.masw.length
    trial_preset = resolve_preset(apply_overrides(preset, {"masw": {"step": 1}}), profile)
    windows = build_windows(profile, trial_preset.masw)
    if not windows:
        return None
    chosen = [windows[index] for index in trial_indices(len(windows), rules.trials)]
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
        tried = try_windows(
            tried_preset,
            profile,
            chosen,
            records,
            folder / RECORDS_FOLDER,
            folder / "windows",
            judge,
            workers,
            exclusions,
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
    kept, trials = _decide(trials, rules)
    choice = MuteChoice(
        chosen=kept.candidate,
        muting=kept.muting,
        cone_m_s=(cone.vmin, cone.median, cone.vmax),
        length=length,
        xmids=tuple(window.xmid for window in chosen),
        trials=tuple(trials),
        notes=(_note(kept, trials, cone, length, receiver_spacing(profile)),),
    )
    write_atomic(run_folder / MUTE_FILE, choice.model_dump_json(indent=2))
    return choice


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


def _decide(trials: list[MuteTrial], rules: MuteRules) -> tuple[MuteTrial, list[MuteTrial]]:
    """The candidate kept, and the trials with why each mute that passed most was refused: the
    most windows passing G3, ties to the earlier; a mute only with `min_gain` more than no mute
    and its longest wavelengths kept (judged when no mute passes `guard_share` of the trial
    windows)."""
    none = trials[0]
    guarded = none.passed >= rules.guard_share * len(none.verdicts)
    checked: list[MuteTrial] = [none]
    for trial in trials[1:]:
        refused = None
        if trial.passed < none.passed + rules.min_gain:
            refused = (
                f"{trial.passed - none.passed:+d} trial windows passing against no mute, "
                f"{rules.min_gain} more needed"
            )
        elif (
            guarded
            and none.wavelengths_m is not None
            and trial.wavelengths_m is not None
            and trial.wavelengths_m[1] < none.wavelengths_m[1] * (1 - rules.max_wavelength_loss)
        ):
            refused = (
                f"its longest wavelengths ({trial.wavelengths_m[1]:g} m) below no mute's "
                f"({none.wavelengths_m[1]:g} m): it cuts the far offsets or the low band"
            )
        checked.append(trial.model_copy(update={"refused": refused}))
    kept = max(
        (trial for trial in checked if trial.refused is None),
        key=lambda trial: trial.passed,  # max keeps the first of equals: no mute first
    )
    return kept, checked


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
