"""advise: today's automatic choices, as PACo's workflow would make them, its picks within the
window's reach, given as a second opinion with their evidence and never applied: the ladder's
window length, the mute trial's muting (a line with shots), a passive line's segments (as long as
its windows ask; the FK selection tried for segments given), the line loop's changes of the
line's settings, and G3's changes of each window's picking. What to do stays the model's."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

from sigpipe.masw.profiles import load_profile
from sigpipe.masw.runs import RunManifest
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import MASWWindow

from paco.autonomous.tools.given import given_of, with_line
from paco.autonomous.tools.trials import (
    Study,
    check_open,
    concrete,
    fast_cone,
    settled,
    survey_reach,
    with_records,
)
from paco.qc.coherence import TrialJudge, choose_length, describe_lengths
from paco.qc.config import QCConfig
from paco.qc.line_loop import candidates
from paco.qc.log import latest, read_attempts
from paco.qc.loops import deep_merge
from paco.qc.models import Attempt, GateResult, Override, Stage
from paco.qc.muting import choose_mute, describe_mutes, estimate_cone, mutable
from paco.qc.report import stretches, xmid_of
from paco.qc.segments import SEGMENT_STAGES, choose_segments, sized_slicing
from paco.settings import Settings

type Topic = Literal["window_length", "mute", "segments", "line_changes", "picking", "inversion"]
# The answers where a line has no such trial.
NO_MUTE = (
    "No mute on a passive line, which has no shot: its levers are the segments, the FK "
    "selection, the whitening and the stacking (advise segments)."
)
NO_SEGMENTS = (
    "PACo's segments are a passive line's: this line's windows stack its shots (the "
    "passive-active mode correlates them, without segments)."
)
ADVICE = "advice"  # in the study's folder: one folder a topic and settings
LENGTH = "length.txt"  # beside the ladder's advice: the length it chooses


def advise(
    study: Study,
    topic: Topic,
    current: Mapping[str, Any],
    settings: Settings,
    config: QCConfig,
    run_folder: Path | None = None,
) -> str:
    """PACo's choice on `topic`, from the line's `current` settings: the ladder's and the mute
    trial's made on the study's records (kept in the study, made once a settings); the line
    loop's and G3's read from the committed line in `run_folder`. The curves' topics are
    closed once the curves are frozen."""
    if topic != "inversion":
        check_open(study)
    match topic:
        case "window_length" | "mute" | "segments":
            return _trial(study, topic, current, settings, config)
        case "line_changes" | "picking":
            if run_folder is None:
                return f"No line committed yet: {topic} reads the committed line's gates."
            if topic == "line_changes":
                return _line_changes(run_folder)
            return _picking(run_folder)
        case "inversion":
            if run_folder is None:
                return "No line committed yet: inversion reads its models' gates."
            return _inversion(run_folder)


def _trial(
    study: Study,
    topic: Literal["window_length", "mute", "segments"],
    current: Mapping[str, Any],
    settings: Settings,
    config: QCConfig,
) -> str:
    """The ladder's, the mute trial's or the segments' outcome, made once for these settings,
    the user's among them, picked as the workflow picks. Without a window length, the mute trial
    runs at the one the ladder chooses, as the workflow orders them."""
    user = given_of(study)
    current = with_line(current, user)
    laddered = topic == "mute" and "length" not in (current.get("masw") or {})
    if laddered:
        current = deep_merge(
            current, {"masw": {"length": _ladder(study, current, settings, config)}}
        )
    folder = study.folder / ADVICE / topic / _key(current)
    saved = folder / "advice.txt"
    if saved.exists() and (topic != "window_length" or (folder / LENGTH).exists()):
        return saved.read_text()
    profile = load_profile(study.profile, settings)
    trial = concrete(current, profile, config, survey_reach(study))
    if topic == "mute" and not mutable(trial.preset):
        return NO_MUTE
    if topic == "segments" and trial.preset.mode != "passive":
        return NO_SEGMENTS
    # A passive line's segments as long as its windows ask, as the workflow sizes them, unless
    # the line's settings give them.
    sized = (
        partial(sized_slicing, profile=profile, rules=config.segments, vmin=config.signal.vg_min)
        if trial.preset.mode == "passive" and "slicing" not in current
        else None
    )
    if topic == "segments" and sized is not None:
        return _sized(sized(trial.preset.masw.length), trial.preset.masw.length, config)
    records, _ = with_records(study, folder, trial.preset, profile, settings, config)
    cone = fast_cone(profile, trial.preset, config)
    exclusions, _, _ = settled(folder, profile, trial.preset, records, config, cone, study)
    judge = TrialJudge(config.coherence, config.curve, trial.picking, config.image)
    if topic == "window_length":
        choice = choose_length(
            profile,
            trial.preset,
            records.outcomes,
            folder,
            judge,
            settings.workers,
            None,
            exclusions,
            sized,
        )
        said = [
            "PACo's ladder, on trial windows spread along the line (not applied):",
            *describe_lengths(choice),
            f"It would choose windows of {choice.length} receivers.",
        ]
        write_atomic(folder / LENGTH, str(choice.length))
    elif topic == "segments":
        # The line's segments kept, their FK selection tried unless the user gave it.
        picked, notes = choose_segments(
            profile,
            trial.preset,
            folder,
            config.segments,
            config.signal,
            trial.picking,
            exclusions,
            config.coherence.trials,
            frozenset(user.line) & SEGMENT_STAGES,
        )
        said = [
            "PACo's FK selection trial, on trial windows spread along the line (not applied):",
            *notes,
            f"It would keep: {json.dumps(picked)}."
            if picked is not None
            else "It would keep the line's own segments and FK selection.",
        ]
    else:
        cone = estimate_cone(profile, trial.preset, config.mute)
        if cone is None:
            return "PACo's mute trial: the gathers give too few traces for the surface waves' cone."
        muting = trial.preset.model_dump(mode="json").get("muting", {})
        muted = muting.get("method", "none") != "none"
        mute = choose_mute(
            profile,
            trial.preset,
            folder,
            judge,
            config.mute,
            settings.workers,
            cone,
            None,
            exclusions,
        )
        if mute is None:
            return "PACo's mute trial: the line has no trial window."
        whose = "its ladder's" if laddered else "the line's"
        said = [
            f"PACo's mute trial at {trial.preset.masw.length} receivers ({whose}), on trial "
            "windows spread along the line (not applied)"
            + (", from the line unmuted, as the trial starts:" if muted else ":"),
            *describe_mutes(mute),
            f"It would keep: {mute.chosen} ({json.dumps(mute.muting)}).",
        ]
    text = "\n".join(said)
    write_atomic(saved, text)
    return text


def _ladder(study: Study, current: Mapping[str, Any], settings: Settings, config: QCConfig) -> int:
    """The window length PACo's ladder chooses from `current` (advise window_length, made once)."""
    _trial(study, "window_length", current, settings, config)
    return int((study.folder / ADVICE / "window_length" / _key(current) / LENGTH).read_text())


def _line_changes(run_folder: Path) -> str:
    """The changes of the line's settings PACo's line loop would try, from the committed line's
    settings (run.json) and its G1, G2 and G3: the one the most failing windows ask first."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    attempts = read_attempts(run_folder)
    records = _latest(attempts, [record.name for record in manifest.records], "preprocessing", "G1")
    folders = [window.folder for window in manifest.windows]
    images = _latest(attempts, folders, "phase_shift", "G2")
    picks = _latest(attempts, folders, "picking", "G3")
    uses: dict[str, set[str]] = {}
    for folder in folders:
        path = run_folder / folder / "window.json"
        if path.exists():
            for file in MASWWindow.model_validate_json(path.read_text()).selected_files:
                uses.setdefault(file.name, set()).add(folder)
    found = candidates(
        records,
        images,
        picks,
        {name: frozenset(units) for name, units in uses.items()},
        manifest.preset.model_dump(mode="json"),
        {},
        frozenset(),
    )
    if not found:
        return "PACo's line loop: no change of the line's settings asked by the gates."
    lines = ["PACo's line loop would try, in this order (not applied):"]
    lines += [
        f"- {candidate.stage} {json.dumps(candidate.overrides)}: asked by {candidate.flag} on "
        f"{_where(candidate.windows)}, {candidate.failing} of them failing G3"
        for candidate in found
    ]
    return "\n".join(lines)


def _picking(run_folder: Path) -> str:
    """G3's changes of the windows' picking, on the committed line, by change."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    attempts = read_attempts(run_folder)
    picks = _latest(attempts, [window.folder for window in manifest.windows], "picking", "G3")
    asked: dict[tuple[str, str], list[str]] = {}
    for unit, result in picks.items():
        for flag in result.flags:
            if isinstance(flag.action, Override) and flag.action.stage == "picking":
                change = json.dumps(flag.action.overrides, sort_keys=True)
                asked.setdefault((flag.name, change), []).append(unit)
    if not asked:
        return "G3 asks no window's picking to change."
    lines = ["G3's changes of the picking, by window (not applied):"]
    lines += [
        f"- {flag} on {_where(units)}: picking {change}" for (flag, change), units in asked.items()
    ]
    return "\n".join(lines)


def _inversion(run_folder: Path) -> str:
    """G5's changes of the windows' inversions, on their latest models, by change."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    attempts = read_attempts(run_folder)
    models = _latest(attempts, [window.folder for window in manifest.windows], "inversion", "G5")
    if not models:
        return "No model yet: G5's changes come once the frozen curves are inverted (P4)."
    asked: dict[tuple[str, str], list[str]] = {}
    for unit, result in models.items():
        for flag in result.flags:
            if isinstance(flag.action, Override) and flag.action.stage == "inversion":
                change = json.dumps(flag.action.overrides, sort_keys=True)
                asked.setdefault((flag.name, change), []).append(unit)
    if not asked:
        return "G5 asks no window's inversion to change."
    lines = ["G5's changes of the inversion, by window (not applied):"]
    lines += [
        f"- {flag} on {_where(units)}: inversion {change}"
        for (flag, change), units in asked.items()
    ]
    return "\n".join(lines)


def _latest(
    attempts: tuple[Attempt, ...], units: Sequence[str], stage: Stage, gate: str
) -> dict[str, GateResult]:
    """Each unit's latest result of `gate`, on its latest attempt of `stage`."""
    found: dict[str, GateResult] = {}
    for unit in units:
        attempt = latest(attempts, unit, stage)
        if attempt is not None and (result := attempt.results.get(gate)) is not None:
            found[unit] = result
    return found


def _where(units: Sequence[str]) -> str:
    """The windows `units` names, as stretches of xmids."""
    xmids = [xmid for unit in units if (xmid := xmid_of(unit)) is not None]
    return stretches(xmids) if xmids else ", ".join(units)


def _sized(made: Mapping[str, Mapping[str, float]], length: int, config: QCConfig) -> str:
    """PACo's segments for a passive line's windows of `length` receivers, sized as its ladder
    tries them, with no trial."""
    return (
        f"PACo's segments for windows of {length} receivers (not applied): "
        f"{made['slicing']['segment_duration']:g} s end to end, "
        f"{config.segments.segment_crossings:g} crossings of the window by the slowest wave "
        f"({config.signal.vg_min:g} m/s), as its ladder tried that length; the FK selection the "
        f"line's own.\nIt would keep: {json.dumps(made)}."
    )


def _key(current: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()[:16]
