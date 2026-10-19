"""commit (docs/autonomous_plan.md §6, §13 M4): the line made with the model's settings in PAC's layout,
from PACo's public steps of a line's processing (paco.qc.line), without its automatic choices (the
ladder, the mute trial, the line loop) and without the gates' retries. G1 judges every record (once
a study for the same preprocessing, band and cone: paco.autonomous.tools.trials.judge_records); every
window, one receiver apart, is imaged and judged by G2; each window whose image G2 did not reject is
picked with the line's picking and judged by G3; then G4 judges the line.

The first commit makes the run (<output>/<profile>/<run id>/). A later one changes the line's
settings in the same run, logged as PACo logs such a change (`line change`): the records
preprocessed again when their preprocessing changes, every window imaged again when the records,
the images or the near field change, every window picked again, those with a recipe of their own
as it says (paco.autonomous.tools.pick); what was there before goes to each unit's attempts/. A change of
the windows themselves (masw), or of the mode, makes a new run, without recipes: PACo refuses it
within a run (the windows would move; another mode is another preset).

The reason for each setting goes in a note of the line's phase-shift attempt, `<stage> <key>:
<reason>`, which PAC's run card shows as that setting's why; each commit's experiment goes in a
note of its own. The study keeps its line in line.json: the run, the settings, the commits."""

import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.profiles import Profile, load_profile
from sigpipe.masw.runs import (
    RecordOutcome,
    RunManifest,
    caching,
    find_run,
    load_manifest,
)
from sigpipe.masw.runs.processing import (
    new_run_folder,
    preprocess_records,
    process_windows,
    write_manifest,
)
from sigpipe.masw.runs.stopping import Stopped
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions

from paco import stopping
from paco.autonomous.tools import pick as picking
from paco.autonomous.tools.decide import DECISIONS
from paco.autonomous.tools.given import REASON, check_line, given_of, missing, without
from paco.autonomous.tools.pick import Picked, Recipe
from paco.autonomous.tools.trials import (
    EXPERIMENTS,
    MODE,
    NEAR_FIELD,
    PICKING,
    PREPROCESSING,
    LineState,
    Preset,
    Study,
    Trial,
    check_open,
    concrete,
    fast_cone,
    judge_records,
    line_windows,
)
from paco.autonomous.upstream.line import line_settings, log_attempt
from paco.qc import given as paco_given
from paco.qc.config import QCConfig, snapshot_qc_config
from paco.qc.curves import imaged_windows
from paco.qc.g4_profile import LINE
from paco.qc.judging import pick_windows
from paco.qc.line import (
    image_line,
    judge_images,
    keep_line_asks,
    line_reach,
    reprocess_records,
)
from paco.qc.log import LINE_CHANGE, append_attempt, latest, read_attempts
from paco.qc.loops import deep_merge
from paco.qc.models import Attempt, GateResult
from paco.qc.report import QCReport, build_report, summarize_report, write_report
from paco.runs import PACKAGES, image_cache
from paco.settings import Settings

LINE_FILE = "line.json"  # in the study's folder
# The near-field distance as the line's attempt logs it, PACo's way: paco-replay builds the
# windows from it, and a reason for it is said under that name.
NEAR_FIELD_LOGGED = "near_field distance_m"


class Commit(BaseModel):
    """One commit of the line: the run it made or changed, the changes, why, and whence."""

    model_config = ConfigDict(frozen=True)

    number: int  # L<n> in the study
    run_id: str
    changes: dict[str, Any]
    reasons: dict[str, str]
    experiment: str | None
    new_run: bool
    made_at: datetime


class Line(BaseModel):
    """The study's committed line: its run, its settings (the commits' changes merged), and its
    commits."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    settings: dict[str, Any]
    commits: tuple[Commit, ...]


@dataclass(frozen=True)
class Committed:
    """The line after a commit, its run's QC report, the windows picked again with their own
    recipes, and whether decisions were made before it (pending again now)."""

    line: Line
    report: QCReport
    reapplied: dict[str, Picked] = field(default_factory=dict[str, Picked])
    decided_before: bool = False


def line_of(study: Study) -> Line | None:
    """The study's committed line; None before its first commit."""
    path = study.folder / LINE_FILE
    return Line.model_validate_json(path.read_text()) if path.exists() else None


def commit(
    study: Study,
    changes: Mapping[str, Any],
    reasons: Mapping[str, str],
    experiment: str | None,
    settings: Settings,
    config: QCConfig,
) -> Committed:
    """The study's line made with `changes` on its committed settings (the first commit: on the
    preset's), each changed setting with its one-line reason in `reasons` ('<stage> <key>', or
    '<stage>' for all its keys), from `experiment` (E<n>) when one gave them."""
    check_open(study)
    profile = load_profile(study.profile, settings)
    line = line_of(study)
    changed = dict(changes)
    if not changed:
        raise ValueError("Give the settings to change: none given.")
    # The user's settings (§8, rule 1): refused changed; added where the line does not hold them
    # yet, with their reason; the model's reasons cover its own changes.
    user = given_of(study)
    check_line(changed, user)
    own = without(changed, user)
    reasons = {key: reason for key, reason in reasons.items() if key not in _named(user.changes())}
    # A setting given at the value it has is no change: none recorded, no reason asked.
    if MODE not in own:
        before = concrete(line.settings if line is not None else {}, profile, config).preset
        same = _unchanged(own, before)
        kept = _without(own, same)
        gone = {stage for stage in own if stage not in kept}
        reasons = {key: reason for key, reason in reasons.items() if key not in same | gone}
        own = kept
    _check_reasons(own, reasons)
    added = missing(line.settings if line is not None else {}, user)
    changed = deep_merge(own, added)
    reasons = {**reasons, **dict.fromkeys(_named(added), REASON)}
    if not changed:
        raise ValueError("The line holds these settings already: give one to change.")
    after = deep_merge(line.settings if line is not None else {}, changed)
    trial = concrete(after, profile, config)
    if experiment is not None and not (study.folder / EXPERIMENTS / f"{experiment}.json").is_file():
        raise ValueError(f"No experiment '{experiment}' in this study: E<n>, as try_settings says.")
    if not line_windows(profile, trial):
        raise ValueError(
            f"No window of {trial.preset.masw.length} receivers has a shot within "
            "masw.distance_min and distance_max: widen them."
        )
    made = Commit(
        number=len(line.commits) + 1 if line is not None else 1,
        run_id="",
        changes=changed,
        reasons=_by_key(changed, reasons),
        experiment=experiment,
        new_run=line is None or "masw" in changed or MODE in changed,
        made_at=datetime.now(UTC),
    )
    earlier = line.commits if line is not None else ()
    notes = _notes((*earlier, made), study)
    parameters = _parameters(after)
    decided_before = (study.folder / DECISIONS).exists()
    with caching.using(image_cache(settings)):
        if line is None or "masw" in changed or MODE in changed:
            run_id = _new_run(study, profile, trial, parameters, notes, settings, config)
            reapplied: dict[str, Picked] = {}
        else:
            run_id = line.run_id
            reapplied = _line_change(
                study, run_id, profile, trial, changed, parameters, notes, settings, config
            )
    made = made.model_copy(update={"run_id": run_id})
    committed = Line(run_id=run_id, settings=after, commits=(*earlier, made))
    write_atomic(study.folder / LINE_FILE, committed.model_dump_json(indent=2))
    run_folder = find_run(run_id, settings)
    report = build_report(
        run_id, run_folder, config.budgets, len(load_manifest(run_id, settings).windows)
    )
    write_report(report, run_folder)
    return Committed(committed, report, reapplied, decided_before)


def describe_commit(committed: Committed) -> str:
    """A commit as the model reads it: what it made, then the run's gates in brief."""
    made = committed.line.commits[-1]
    how = "made run" if made.new_run else "changed the line's settings in run"
    said = [f"L{made.number} {how} {made.run_id}."]
    picked = [one for one in committed.reapplied.values() if one.error is None]
    if picked:
        counts = Counter(one.recipe.how for one in picked)
        said.append(
            f"The windows' own recipes, picked again: {len(picked)} "
            f"({', '.join(f'{how} {count}' for how, count in counts.items())})."
        )
    if failed := [one.unit for one in committed.reapplied.values() if one.error is not None]:
        said.append(
            f"Recipes that no longer pick, the line's picking instead: {', '.join(failed)}."
        )
    if committed.decided_before:
        said.append("Every window is pending again: decide each once its curve suits you.")
    return " ".join(said) + f"\n{summarize_report(committed.report, retries=False)}"


def picking_of(study: Study, line: Line, settings: Settings, config: QCConfig) -> PickingParameters:
    """The committed line's picking: PACo's parameters with the line's changes."""
    return concrete(line.settings, load_profile(study.profile, settings), config).picking


def line_state(run_folder: Path) -> LineState:
    """What a trial set reads of the committed line in `run_folder`: each window's G3 verdict
    ("no curve" without a pick) and the flags of its latest image and pick (G2, G3, G4)."""
    attempts = read_attempts(run_folder)
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    verdicts: dict[float, str] = {}
    flags: dict[float, frozenset[str]] = {}
    for window in manifest.windows:
        imaged = latest(attempts, window.folder, "phase_shift")
        picked = latest(attempts, window.folder, "picking")
        results = [
            *(imaged.results.values() if imaged is not None else ()),
            *(picked.results.values() if picked is not None else ()),
        ]
        g3 = picked.results.get("G3") if picked is not None else None
        verdicts[window.xmid] = g3.verdict if g3 is not None else "no curve"
        flags[window.xmid] = frozenset(flag.name for result in results for flag in result.flags)
    return LineState(verdicts, flags)


def _new_run(
    study: Study,
    profile: Profile,
    trial: Trial,
    parameters: dict[str, Any],
    notes: tuple[str, ...],
    settings: Settings,
    config: QCConfig,
) -> str:
    """A new run of the line with `trial`'s settings, as PACo's run_processing makes one without
    its choosers, then picked: its id. Stopped, nothing of it is kept, as PACo's."""
    run_id, run_folder = new_run_folder(settings.output_dir / study.profile)
    try:
        _make(study, run_id, run_folder, profile, trial, parameters, notes, settings, config)
    except Stopped:
        shutil.rmtree(run_folder, ignore_errors=True)
        raise
    return run_id


def _make(
    study: Study,
    run_id: str,
    run_folder: Path,
    profile: Profile,
    trial: Trial,
    parameters: dict[str, Any],
    notes: tuple[str, ...],
    settings: Settings,
    config: QCConfig,
) -> None:
    """PACo's `_process` (paco.qc.line), its public steps without the choosers and the line loop:
    the records preprocessed and judged by G1, the windows imaged and judged by G2, the line's
    settings logged with the model's reasons; then every window picked."""
    preset = trial.preset
    snapshot_qc_config(config, run_folder)
    # The user's settings logged as PACo logs them: G2 then rejects a window it could fix only by
    # changing one, saying the change it asks (keep_line_asks).
    paco_given.give(run_folder, "processing", given_of(study).line)
    started_at = datetime.now(UTC)
    records = preprocess_records(
        preset, profile, run_folder, settings.workers, stop=stopping.current()
    )
    for record in records:
        log_attempt(run_folder, record.name, "preprocessing", 1, {}, "initial", started_at, record)
    reach = line_reach(run_folder, profile, records, config, preset)
    cone = fast_cone(profile, preset, config)
    exclusions, usable = judge_records(
        run_folder, profile, preset, records, config, reach, cone, study
    )
    imaged_at = datetime.now(UTC)
    outcomes = process_windows(
        preset,
        line_windows(profile, trial, exclusions.records),
        records,
        run_folder,
        settings.workers,
        exclusions=exclusions,
        stop=stopping.current(),
    )
    manifest = write_manifest(
        run_id,
        run_folder,
        profile,
        preset,
        started_at,
        records,
        outcomes,
        exclusions,
        packages=PACKAGES,
    )
    for outcome in outcomes:
        append_attempt(
            run_folder,
            Attempt(
                unit=outcome.folder,
                stage="phase_shift",
                attempt=1,
                parameters={},
                triggered_by="initial",
                started_at=imaged_at,
                finished_at=manifest.finished_at,
                status=outcome.status,
                error=outcome.error,
            ),
        )
    line_settings(run_folder, parameters, notes, started_at)
    judge_images(run_folder, manifest, config, settings, usable)
    keep_line_asks(run_folder)
    _pick_line(study, run_folder, manifest, trial, config, settings, "initial", {})


def _line_change(
    study: Study,
    run_id: str,
    profile: Profile,
    trial: Trial,
    changes: Mapping[str, Any],
    parameters: dict[str, Any],
    notes: tuple[str, ...],
    settings: Settings,
    config: QCConfig,
) -> dict[str, Picked]:
    """`changes` made on run `run_id`'s line, as PACo's redo of the line makes them (`line
    change`): the records preprocessed again and judged by G1 afresh when their preprocessing
    changes; the line's settings logged with the reasons (the windows are built from its near
    field); every window imaged again and judged by G2 when the records, the images or the near
    field change; every window picked again, those with a recipe as it says. The recipes' picks."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    preset = trial.preset
    paco_given.give(run_folder, "processing", given_of(study).line)
    stages = {key: value for key, value in changes.items() if key not in (PICKING, NEAR_FIELD)}
    records, exclusions = tuple(manifest.records), manifest.exclusions
    usable = _usable(run_folder, records)
    started_at = datetime.now(UTC)
    if any(stage in PREPROCESSING for stage in stages):
        records = reprocess_records(
            run_folder,
            profile,
            preset,
            records,
            Exclusions(),
            settings.workers,
            LINE_CHANGE,
            stages,
        )
        reach = line_reach(run_folder, profile, records, config, preset)
        cone = fast_cone(profile, preset, config)
        exclusions, usable = judge_records(
            run_folder, profile, preset, records, config, reach, cone, study
        )
    before = latest(read_attempts(run_folder), LINE, "phase_shift")
    append_attempt(
        run_folder,
        Attempt(
            unit=LINE,
            stage="phase_shift",
            attempt=before.attempt + 1 if before is not None else 1,
            parameters=parameters,
            triggered_by=LINE_CHANGE,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="succeeded",
            notes=notes,
        ),
    )
    if stages or NEAR_FIELD in changes:
        image_line(
            run_id,
            run_folder,
            profile,
            preset,
            records,
            exclusions,
            settings,
            LINE_CHANGE,
            stages,
        )
        judge_images(run_folder, load_manifest(run_id, settings), config, settings, usable)
        keep_line_asks(run_folder)
    return _pick_line(
        study,
        run_folder,
        load_manifest(run_id, settings),
        trial,
        config,
        settings,
        LINE_CHANGE,
        picking.recipes(study, run_id),
    )


def _pick_line(
    study: Study,
    run_folder: Path,
    manifest: RunManifest,
    trial: Trial,
    config: QCConfig,
    settings: Settings,
    triggered_by: str,
    recipes: Mapping[str, Recipe],
) -> dict[str, Picked]:
    """Every window whose latest image G2 did not reject, picked with the line's picking and
    judged by G3 against G2's band, but those with a recipe of their own, picked as it says once
    the others are (a recipe that no longer picks: the line's picking); then G4 over the line. No
    gate's retry: the model's to ask. Every pick kept in the study; the recipes' returned."""
    ready = imaged_windows(run_folder, manifest)
    own = {unit: recipe for unit, recipe in recipes.items() if unit in ready}
    jobs = {unit: (trial.picking, _band(g2)) for unit, g2 in ready.items() if unit not in own}
    results = pick_windows(run_folder, jobs, config, triggered_by, settings.workers) if jobs else {}
    reapplied = (
        picking.apply(
            study, run_folder, manifest, own, trial.picking, config, settings.workers, True
        )
        if own
        else {}
    )
    failed = {
        unit: (trial.picking, _band(ready[unit])) for unit, one in reapplied.items() if one.error
    }
    if failed:
        results |= pick_windows(run_folder, failed, config, triggered_by, settings.workers)
    picking.keep(study, picking.line_picks(manifest.run_id, results, trial.picking))
    picking.finish(run_folder, manifest, config)
    return reapplied


def _band(g2: GateResult | None) -> tuple[float, float] | None:
    return g2.kept.band_hz if g2 is not None else None


def _usable(run_folder: Path, records: Sequence[RecordOutcome]) -> dict[str, Any]:
    """Each record's usable band, from its latest G1 result."""
    attempts = read_attempts(run_folder)
    found: dict[str, Any] = {}
    for record in records:
        attempt = latest(attempts, record.name, "preprocessing")
        result = attempt.results.get("G1") if attempt is not None else None
        if result is not None:
            found[record.name] = result.kept.band_hz
    return found


def _parameters(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The line's settings as its phase-shift attempt logs them: the preset's stages the commits
    set, and the near-field distance as PACo logs it."""
    logged = {key: value for key, value in settings.items() if key not in (PICKING, NEAR_FIELD)}
    near = settings.get(NEAR_FIELD)
    if isinstance(near, int | float):
        logged["near_field"] = {"distance_m": float(near)}
    return logged


def _named(changes: Mapping[str, Any]) -> list[str]:
    """The settings `changes` sets, as reasons name them: '<stage> <key>', or a top-level key
    alone (near_field_m)."""
    named: list[str] = []
    for stage, value in changes.items():
        if isinstance(value, Mapping) and value:
            named.extend(f"{stage} {key}" for key in cast(Mapping[str, Any], value))
        else:
            named.append(stage)
    return named


def _unchanged(changes: Mapping[str, Any], before: Preset) -> set[str]:
    """The settings of `changes` ('<stage> <key>') at the value the line has, its preset's where
    it set none (`before`, a preset). A stage's method stays a change while another of its
    settings is one: the stage's override names its method."""
    same: set[str] = set()
    for stage, values in changes.items():
        current = getattr(before, stage, None)
        if not isinstance(values, Mapping) or current is None:
            continue
        given = cast(Mapping[str, Any], values)
        kept = {
            f"{stage} {key}"
            for key, value in given.items()
            if key in type(current).model_fields and _same(getattr(current, key), value)
        }
        if any(f"{stage} {key}" not in kept for key in given if key != "method"):
            kept.discard(f"{stage} method")
        same |= kept
    return same


def _same(had: object, value: object) -> bool:
    """Whether `value` is `had`, a number as a number (100 is 100.0)."""
    numbers = (int, float)
    if isinstance(had, numbers) and isinstance(value, numbers) and not isinstance(value, bool):
        return float(had) == float(value)
    return had == value


def _without(changes: Mapping[str, Any], settings: set[str]) -> dict[str, Any]:
    """`changes` without the `settings` named ('<stage> <key>'); a stage left empty, gone."""
    kept: dict[str, Any] = {}
    for stage, values in changes.items():
        if not isinstance(values, Mapping):
            kept[stage] = values
            continue
        left = {
            key: value
            for key, value in cast(Mapping[str, Any], values).items()
            if f"{stage} {key}" not in settings
        }
        if left:
            kept[stage] = left
    return kept


def _reason_of(key: str, changes: Mapping[str, Any], reasons: Mapping[str, str]) -> str | None:
    """The reason for setting `key`: its own, its stage's, or, for a stage whose method changes,
    those given for its other settings (a method and its parameters are one choice)."""
    stage = key.split(" ")[0]
    if key in reasons:
        return reasons[key]
    if stage in reasons:
        return reasons[stage]
    values = changes.get(stage)
    if isinstance(values, Mapping) and "method" in values:
        given = [
            reason.strip() for other, reason in reasons.items() if other.split(" ")[0] == stage
        ]
        if given:
            return "; ".join(dict.fromkeys(given))
    return None


def _check_reasons(changes: Mapping[str, Any], reasons: Mapping[str, str]) -> None:
    """Refused unless every setting changed has a one-line reason, and every reason a setting
    changed (§10)."""
    named = _named(changes)
    missing = [key for key in named if _reason_of(key, changes, reasons) is None]
    if missing:
        raise ValueError(
            "Give a one-line reason for each setting changed, by '<stage> <key>' as the changes "
            f"name it: missing {', '.join(missing)}."
        )
    if unknown := [key for key in reasons if key not in named and key not in changes]:
        raise ValueError(
            f"Reasons for settings this commit does not change: {', '.join(unknown)}. It changes "
            f"{', '.join(named)}."
        )
    if bad := [key for key, reason in reasons.items() if not reason.strip() or "\n" in reason]:
        raise ValueError(f"A reason is one line of text: {', '.join(bad)}.")


def _by_key(changes: Mapping[str, Any], reasons: Mapping[str, str]) -> dict[str, str]:
    """Each setting `changes` sets with its reason (_reason_of)."""
    return {key: (_reason_of(key, changes, reasons) or "").strip() for key in _named(changes)}


def _notes(commits: Sequence[Commit], study: Study) -> tuple[str, ...]:
    """The line's notes: each setting's latest reason, `<stage> <key>: <reason> (L<n>, E<k>)`,
    with the commit and the experiment it came from; then where the experiments are."""
    said: dict[str, str] = {}
    for one in commits:
        source = f"L{one.number}" + (f", {one.experiment}" if one.experiment else "")
        for key, reason in one.reasons.items():
            name = NEAR_FIELD_LOGGED if key == NEAR_FIELD else key
            said[name] = f"{reason.rstrip('.')} ({source})"
    return (
        *(f"{name}: {reason}" for name, reason in said.items()),
        f"Committed by PACo's autonomous agent (paco {version('paco')}) in study {study.id}, "
        "with its experiments.",
    )
