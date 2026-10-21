"""The state card and the study's state. Code writes the card before
each of the model's calls, replacing the one before: the phase, the objective and the budgets, the
settings kept as the user gave them, the survey, the line now with its scorecard, the open issues,
the checks, the last experiments and the windows' decisions; after the freeze, the frozen curves.
About 1,500 tokens at most. Both are read from the study's records alone (the journal, the usage
log, the survey, the committed line and its run, the picks and decisions, the frozen curves), so
that a study resumed shows the same; state.json is rewritten from them each time, never edited
alone."""

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.runs import RunManifest, find_run, window_length
from sigpipe.masw.runs.writing import write_atomic

from paco.autonomous.host.budgets import EFFORTS, Effort, spent
from paco.autonomous.host.phases import NAMES, Phase, phase_of
from paco.autonomous.sensors.checks import line_checks, window_checks
from paco.autonomous.sensors.flags import flag_class
from paco.autonomous.sensors.scorecard import scorecard
from paco.autonomous.tools import decide, journal, models, pick
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.commit import Line, line_of
from paco.autonomous.tools.freeze import Frozen, frozen
from paco.autonomous.tools.given import Given, asks, given_of
from paco.autonomous.tools.invert import inversions_of, symptoms
from paco.autonomous.tools.survey import SAVED, Survey, shots_said
from paco.autonomous.tools.trials import NEAR_FIELD, PICKING, Study
from paco.qc.given import leaves
from paco.qc.log import latest, read_attempts
from paco.qc.models import GateResult
from paco.qc.report import stretches, xmid_of
from paco.settings import Settings

STATE = "state.json"  # in the study's folder
# The open issues and the experiments the card lists at most.
ISSUES = 5
EXPERIMENTS = 3


class State(BaseModel):
    """The study's state as state.json keeps it: its phase, its objective, the committed
    settings, each window's status and recipe, and the budgets spent."""

    model_config = ConfigDict(frozen=True)

    phase: Phase
    objective: dict[str, Any]
    settings: dict[str, Any]
    windows: dict[str, dict[str, Any]]
    spent: str


def objective_of(study: Study) -> journal.Entry | None:
    """The study's objective, the journal's entry."""
    return next((entry for entry in journal.entries(study) if entry.kind == "objective"), None)


def effort_of(study: Study) -> Effort:
    """The effort the user asked: normal unless their objective says another."""
    found = objective_of(study)
    effort = found.data.get("effort") if found is not None else None
    return effort if effort in ("quick", "thorough") else "normal"


def card(study: Study, settings: Settings) -> str:
    """The state card of `study`, PACo's folders in `settings`."""
    phase = phase_of(study)
    goal = objective_of(study)
    used = spent(study, effort_of(study))
    user = given_of(study)
    lines = [
        f"Phase: {phase}, the {NAMES[phase]} · objective: "
        f"{goal.text if goal is not None else 'not read yet'} · budget: {used.said()}",
        f"Kept as you gave: {'; '.join(user.said()) or 'none'}",
    ]
    if near := used.near():
        lines.append("Budgets: " + "; ".join(near) + ".")
    survey = study.folder / SAVED
    if survey.exists():
        lines.append(_survey(Survey.model_validate_json(survey.read_text())))
    elif phase == "P1":
        lines.append("Survey: not made yet.")
    made = frozen(study)
    if made is not None:
        depth = [curve.wavelength_m[1] / 2 for curve in made.curves]
        lines.append(
            f"Frozen (from L{made.commit}, run {made.run_id}): {len(made.curves)} curves accepted, "
            f"{len(made.rejected)} windows rejected; depth (λmax/2) median "
            f"{_median(depth):.3g} m."
        )
        lines += _models(study, made, settings)
        return "\n".join(lines + _held(find_run(made.run_id, settings), user))
    line = line_of(study)
    if line is None:
        if phase == "P2":
            lines.append("Line: none committed yet.")
        return "\n".join(lines)
    lines += _line(study, line, settings)
    return "\n".join(lines + _held(find_run(line.run_id, settings), user))


def outcome(study: Study, settings: Settings, tool: str) -> str | None:
    """What a call of `tool` did, in a few words for the user's list of steps: the survey, the
    search, an experiment, a commit, the windows picked again, the curves' or the models'
    decisions, the freeze, the models inverted; none for a tool that only reads."""
    found = journal.entries(study)
    if tool == "survey" and (study.folder / SAVED).exists():
        return _surveyed(Survey.model_validate_json((study.folder / SAVED).read_text()))
    if tool == "search" and (made := searching.searched(study)) is not None:
        return _searched(made, searching.knockout_of(study))
    if tool in ("try_settings", "commit"):
        kind = "experiment" if tool == "try_settings" else "commit"
        entry = next((one for one in reversed(found) if one.kind == kind), None)
        if entry is None:
            return None
        if kind == "commit":
            return ", ".join(_changes_said(entry.data.get("changes") or {}))
        variants = entry.data.get("variants", [])
        count = max((int(one.get("windows", 0)) for one in variants), default=0)
        tried = len(variants) - 1
        return f"{tried} variant{'s' if tried != 1 else ''} on {count} test windows"
    made, line = frozen(study), line_of(study)
    if tool == "freeze" and made is not None:
        return f"{len(made.curves)} curves accepted, {len(made.rejected)} windows rejected"
    if tool == "invert" and made is not None:
        inverted = {one.unit for one in inversions_of(study, made.run_id)}
        units = [curve.unit for curve in made.curves]
        return f"{sum(unit in inverted for unit in units)} of {len(units)} windows inverted"
    if tool not in ("pick", "decide") or line is None:
        return None
    folder = find_run(line.run_id, settings)
    if tool == "pick":
        return f"{len(pick.recipes(study, line.run_id))} windows picked again"
    if made is not None:
        return _decided(models.statuses(study, folder, line.run_id), len(made.curves), "models")
    manifest = RunManifest.model_validate_json((folder / "run.json").read_text())
    return _decided(decide.statuses(study, folder, line.run_id), len(manifest.windows), "windows")


def _surveyed(found: Survey) -> str:
    """The survey in a few words: the line, its records, its surface waves and usable band."""
    records = "noise records" if found.kind == "passive" else "shots"
    said = [f"{found.receivers} receivers", f"{len(found.shots)} {records}"]
    if found.cone_m_s is not None:
        said.append(f"surface waves {found.cone_m_s[0]:.0f}-{found.cone_m_s[2]:.0f} m/s")
    if found.usable_spread_hz is not None:
        low, high = found.usable_spread_hz
        said.append(f"usable {low:.3g}-{high:.3g} Hz")
    return ", ".join(said)


def _searched(made: searching.Search, knockout: searching.Knockout | None) -> str:
    """The search in a few words: its combinations, then its finalists, or the round being
    judged."""
    said = f"{len(made.rows)} combinations compared"
    if knockout is None:
        return said
    if knockout.finalists:
        return f"{said}; {len(knockout.finalists)} finalists"
    return (
        f"{said}; round {knockout.round}, {len(knockout.kept)} of {len(knockout.sheets)} sheets "
        "judged"
    )


def _changes_said(changes: Mapping[str, Any]) -> list[str]:
    """A commit's changes in the user's words, the main ones as the run's card says them: "21
    receivers", "farthest shot 31.6 m", "mute 102.5-1391 m/s"; the others as PACo names them."""
    said: list[str] = []
    rest: dict[str, Any] = {}
    for stage, values in changes.items():
        if stage == NEAR_FIELD:
            said.append(f"near field {values:g} m" if values else "no near field")
            continue
        if not isinstance(values, Mapping):
            rest[stage] = values
            continue
        inner = dict(cast(Mapping[str, Any], values))
        if stage == "masw":
            said += _masw_said(inner)
        elif stage == "muting" and "method" in inner:
            said.append(_mute_said(inner))
            inner = {}
        elif stage == "dispersion":
            said += _band_said(inner)
        if inner:
            rest[stage] = inner
    return said + leaves(rest)


def _masw_said(inner: dict[str, Any]) -> list[str]:
    """The windows' changes in words, taken out of `inner`."""
    said: list[str] = []
    if (length := inner.pop("length", None)) is not None:
        said.append(f"{length} receivers")
    if (far := inner.pop("distance_max", None)) is not None:
        said.append(f"farthest shot {far:g} m")
    if (near := inner.pop("distance_min", None)) is not None:
        said.append(f"nearest shot {near:g} m")
    return said


def _mute_said(muting: Mapping[str, Any]) -> str:
    method, low, high = muting.get("method"), muting.get("vmin"), muting.get("vmax")
    if method == "none":
        return "no mute"
    if isinstance(low, int | float) and isinstance(high, int | float):
        return f"mute {low:g}-{high:g} m/s"
    return f"mute {method}"


def _band_said(inner: dict[str, Any]) -> list[str]:
    """The images' band in words, taken out of `inner`; none when it does not change."""
    low, high = inner.pop("fmin", None), inner.pop("fmax", None)
    if low is None and high is None:
        return []
    if low is not None and high is not None:
        return [f"images {low:g}-{high:g} Hz"]
    return [f"images up to {high:g} Hz" if high is not None else f"images from {low:g} Hz"]


def _decided(statuses: Mapping[str, str], total: int, noun: str) -> str:
    """The decisions so far: accepted, rejected, and those left."""
    accepted = sum(1 for status in statuses.values() if status == "accepted")
    rejected = sum(1 for status in statuses.values() if status == "rejected")
    return f"{accepted} {noun} accepted, {rejected} rejected, {total - accepted - rejected} pending"


def state(study: Study, settings: Settings) -> State:
    """The study's state, rebuilt from its records, and written to state.json."""
    goal = objective_of(study)
    line = line_of(study)
    windows: dict[str, dict[str, Any]] = {}
    if line is not None:
        folder = find_run(line.run_id, settings)
        statuses = decide.statuses(study, folder, line.run_id)
        recipes = pick.recipes(study, line.run_id)
        manifest = RunManifest.model_validate_json((folder / "run.json").read_text())
        windows = {
            window.folder: {
                "status": statuses.get(window.folder, "pending"),
                "recipe": recipes[window.folder].model_dump(exclude_defaults=True)
                if window.folder in recipes
                else None,
            }
            for window in manifest.windows
        }
    made = State(
        phase=phase_of(study),
        objective=goal.data if goal is not None else {},
        settings=line.settings if line is not None else {},
        windows=windows,
        spent=spent(study, effort_of(study)).said(),
    )
    write_atomic(study.folder / STATE, made.model_dump_json(indent=2))
    return made


def _models(study: Study, made: Frozen, settings: Settings) -> list[str]:
    """The models of the frozen curves: G5's verdicts by symptom and G6's on the line, each
    window's retries, and the decisions on them."""
    done = inversions_of(study, made.run_id)
    if not done:
        return ["Models: none yet: invert the frozen curves."]
    folder = find_run(made.run_id, settings)
    units = [curve.unit for curve in made.curves]
    counts = Counter(one.unit for one in done)
    inverted = [unit for unit in units if unit in counts]
    jobs = [entry for entry in journal.entries(study) if entry.kind == "inversion"]
    lines = [
        f"Models ({jobs[-1].id if jobs else 'none'}): {len(inverted)} of {len(units)} windows "
        "inverted"
    ]
    lines += [f"  {line}" for line in symptoms(folder, inverted)]
    budget = EFFORTS[effort_of(study)].inversion_retries
    used: dict[int, list[float]] = {}
    for unit, count in counts.items():
        if count > 1 and (xmid := xmid_of(unit)) is not None:
            used.setdefault(count - 1, []).append(xmid)
    said = "; ".join(f"{count} at {stretches(xmids)}" for count, xmids in sorted(used.items()))
    lines.append(f"Retries: {budget} a window at most; used: {said or 'none'}")
    statuses = models.statuses(study, folder, made.run_id)
    accepted = sum(1 for status in statuses.values() if status == "accepted")
    rejected = sum(1 for status in statuses.values() if status == "rejected")
    lines.append(
        f"Models decided: {accepted} accepted · {rejected} rejected · "
        f"{len(units) - accepted - rejected} pending"
    )
    return lines


def _line(study: Study, line: Line, settings: Settings) -> list[str]:
    """The line now: its settings, its scorecard, its open issues, its checks, the last
    experiments and the windows' decisions."""
    folder = find_run(line.run_id, settings)
    manifest = RunManifest.model_validate_json((folder / "run.json").read_text())
    attempts = read_attempts(folder)
    g2: dict[float, GateResult | None] = {}
    g3: dict[float, GateResult | None] = {}
    g4: list[GateResult] = []
    for window in manifest.windows:
        imaged = latest(attempts, window.folder, "phase_shift")
        g2[window.xmid] = imaged.results.get("G2") if imaged is not None else None
        picked = latest(attempts, window.folder, "picking")
        g3[window.xmid] = picked.results.get("G3") if picked is not None else None
        if picked is not None and "G4" in picked.results:
            g4.append(picked.results["G4"])
    length_m = window_length(folder) or 0.0
    shots = str(manifest.preset.mode) != "passive"
    card = scorecard(g3, length_m, g4=g4, shots=shots, g2=g2)
    lines = [
        f"Line now (L{line.commits[-1].number}, run {line.run_id}): "
        + _settings(manifest, line.settings, length_m),
        f"  G3 {len(card.covered)} of {len(card.windows)} pass · depth (λmax/2) median "
        f"{_value(card.depth_m, ' m')}, P10 {_value(card.depth_p10_m, ' m')} · uncertainty "
        f"{_share(card.precision)} · G4 outliers {card.outliers or 0}"
        + (f" · near field {_share(card.near_field)}" if shots else ""),
    ]
    issues = _issues(g3)
    if issues:
        lines.append("Open issues:")
        lines += [f"  {issue}" for issue in issues]
    checks = [
        check
        for result in g3.values()
        if result is not None
        and result.verdict == "pass"
        and (check := window_checks(result, length_m, shots))
    ]
    if checks:
        summary = line_checks(checks)
        lines.append(
            f"Checks: λmax/L median {_value(summary.reach)} (beyond 3·L on "
            f"{summary.beyond_windows} windows); aliased on {summary.aliased_windows}"
            + (f"; nearest shot in the near field on {summary.near_field_windows}" if shots else "")
        )
    lines += _experiments(study)
    statuses = decide.statuses(study, folder, line.run_id)
    accepted = sum(1 for status in statuses.values() if status == "accepted")
    rejected = sum(1 for status in statuses.values() if status == "rejected")
    pending = len(manifest.windows) - accepted - rejected
    lines.append(f"Windows: {accepted} accepted · {rejected} rejected · {pending} pending")
    return lines


def _settings(manifest: RunManifest, settings: Mapping[str, Any], length_m: float) -> str:
    """The line's settings in words: its mode, its windows, then the stages its mode has."""
    preset = manifest.preset
    mode = str(preset.mode)
    values = preset.model_dump(mode="json")
    near = settings.get(NEAR_FIELD)
    far = preset.masw.distance_max
    dispersion = values.get("dispersion")
    said = [f"{mode} mode", f"{preset.masw.length} receivers ({length_m:g} m)"]
    if mode == "active":
        said.append(f"near field {near:g} m" if isinstance(near, int | float) else "no near field")
    if mode != "passive":
        said += [
            f"farthest shot {far:g} m" if far is not None else "shots at any distance",
            _method(values.get("muting"), "mute"),
        ]
    else:
        slicing, selection = values["slicing"], values["selection"]
        said += [
            f"segments {slicing['segment_duration']:g} s every {slicing['segment_step']:g} s",
            f"FK selection {selection.get('threshold', 0):g}"
            if selection.get("method") == "fk"
            else "no selection",
            _method(values.get("whitening"), "whitening"),
            _method(values.get("normalization"), "normalization"),
        ]
    said.append(_method(values.get("filtering"), "filter"))
    said.append(
        f"{dispersion['fmin']:g}-{dispersion['fmax']:g} Hz, "
        f"{dispersion['vmin']:g}-{dispersion['vmax']:g} m/s"
        if dispersion is not None
        else "no phase shift"
    )
    stacking = values.get("image_stacking") or values.get("stacking")
    if stacking is not None:
        said.append(f"{stacking.get('method', 'linear')} stack")
    picking = settings.get(PICKING)
    if picking:
        said.append(f"picking {json.dumps(picking, separators=(',', ':'))}")
    return ", ".join(said)


def _method(stage: Mapping[str, Any] | None, noun: str) -> str:
    method = (stage or {}).get("method")
    return f"no {noun}" if method in (None, "none") else f"{noun} {method}"


def _issues(g3: Mapping[float, GateResult | None]) -> list[str]:
    """The windows whose curve does not pass G3, grouped by verdict and flags, each group with the
    change its fixable flags suggest."""
    groups: dict[tuple[str, tuple[str, ...]], list[float]] = {}
    said: dict[tuple[str, tuple[str, ...]], str] = {}
    for xmid, result in g3.items():
        if result is not None and result.verdict == "pass":
            continue
        flags = (
            tuple(flag.name for flag in result.flags if flag_class(flag) != "note")
            if result
            else ()
        )
        key = (result.verdict if result is not None else "no curve", flags)
        groups.setdefault(key, []).append(xmid)
        if key not in said and result is not None:
            said[key] = ", ".join(
                pick.flag_said(flag) for flag in result.flags if flag_class(flag) != "note"
            )
    found = sorted(groups.items(), key=lambda item: -len(item[1]))
    return [
        f"{stretches(xmids)}: {verdict}" + (f": {said.get((verdict, flags))}" if flags else "")
        for (verdict, flags), xmids in found[:ISSUES]
    ]


def _experiments(study: Study) -> list[str]:
    """The last experiments, each with the conclusions drawn from it."""
    found = journal.entries(study)
    made = [entry for entry in found if entry.kind == "experiment"][-EXPERIMENTS:]
    if not made:
        return []
    lines = ["Last experiments:"]
    for entry in made:
        lines.append(f"  {journal.line(entry)}")
        for drawn in found:
            if drawn.kind in ("conclusion", "decision") and entry.id in drawn.of:
                lines.append(f"    -> {drawn.id} {drawn.text[:160]}")
    return lines


def _survey(found: Survey) -> str:
    lines = f"{found.receivers} receivers {found.spacing_m:g} m apart ({found.line_m:g} m)"
    if found.kind == "passive":
        parts = [
            lines,
            f"{len(found.shots)} records of ambient noise, segments of {found.segment_s or 0:g} s "
            "(the preset's)",
        ]
    else:
        parts = [lines, f"{len(found.shots)} shots: {shots_said(found.shots)}"]
    if found.reach_m is not None:
        parts.append(f"reach {found.reach_m:g} m")
    if found.cone_m_s is not None:
        parts.append(f"surface waves {found.cone_m_s[0]:.0f}-{found.cone_m_s[2]:.0f} m/s")
    if found.usable_spread_hz is not None:
        low, high = found.usable_spread_hz
        parts.append(f"usable {low:g}-{high:g} Hz")
    parts.append(f"windows of {found.lengths[0]} to {found.lengths[1]} receivers")
    return "Survey: " + "; ".join(parts)


def _held(run_folder: Path, user: Given) -> list[str]:
    """What the user's settings held back on the line, for the model to propose to them."""
    held = asks(run_folder, user)
    return (
        ["Held back by the user's settings (propose the change to them): " + "; ".join(held)]
        if held
        else []
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _value(value: float | None, unit: str = "") -> str:
    return f"{value:.3g}{unit}" if value is not None else "none"


def _share(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "none"
