"""advise (docs/geo_plan.md §6): today's automatic choices, as PACo's workflow would make them,
given as a second opinion with their evidence and never applied: the ladder's window length, the
mute trial's muting (a line with shots), the segment trial's segments and FK selection (a passive
line), the line loop's changes of the line's settings, and G3's changes of each window's picking.
What to do stays the model's (§16, the starting point)."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from sigpipe.masw.profiles import load_profile
from sigpipe.masw.runs import RunManifest
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import MASWWindow

from paco.geo.tools.given import given_of, with_line
from paco.geo.tools.trials import Study, check_open, concrete, fast_cone, settled, with_records
from paco.qc.coherence import TrialJudge, choose_length, describe_lengths
from paco.qc.config import QCConfig
from paco.qc.line_loop import candidates
from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt, GateResult, Override, Stage
from paco.qc.muting import choose_mute, describe_mutes, estimate_cone, mutable
from paco.qc.report import stretches, xmid_of
from paco.qc.segments import SEGMENT_STAGES, choose_segments
from paco.settings import Settings

type Topic = Literal["window_length", "mute", "segments", "line_changes", "picking", "inversion"]
# The answers where a line has no such trial.
NO_MUTE = (
    "No mute on a passive line, which has no shot: its levers are the segments, the FK "
    "selection, the whitening and the stacking (advise segments)."
)
NO_SEGMENTS = (
    "PACo's segment trial is a passive line's: this line's windows stack its shots (the "
    "passive-active mode correlates them, without segments)."
)
ADVICE = "advice"  # in the study's folder: one folder a topic and settings


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
    """The ladder's, the mute trial's or the segment trial's outcome, made once for these
    settings, the user's among them."""
    user = given_of(study)
    current = with_line(current, user)
    folder = study.folder / ADVICE / topic / _key(current)
    saved = folder / "advice.txt"
    if saved.exists():
        return saved.read_text()
    profile = load_profile(study.profile, settings)
    trial = concrete(current, profile, config)
    if topic == "mute" and not mutable(trial.preset):
        return NO_MUTE
    if topic == "segments" and trial.preset.mode != "passive":
        return NO_SEGMENTS
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
        )
        said = [
            "PACo's ladder, on trial windows spread along the line (not applied):",
            *describe_lengths(choice),
            f"It would choose windows of {choice.length} receivers.",
        ]
    elif topic == "segments":
        # The stages of the trial the user gave stay as they gave them.
        picked, notes = choose_segments(
            profile,
            trial.preset,
            folder,
            config.segments,
            config.signal,
            trial.picking,
            exclusions,
            frozenset(user.line) & SEGMENT_STAGES,
        )
        said = [
            "PACo's segment trial, on trial windows spread along the line (not applied):",
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
        said = [
            "PACo's mute trial, on trial windows spread along the line (not applied)"
            + (", from the line unmuted, as the trial starts:" if muted else ":"),
            *describe_mutes(mute),
            f"It would keep: {mute.chosen} ({json.dumps(mute.muting)}).",
        ]
    text = "\n".join(said)
    write_atomic(saved, text)
    return text


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


def _key(current: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()[:16]
