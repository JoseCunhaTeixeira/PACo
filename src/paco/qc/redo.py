"""Going back to a stage for some windows (docs/qc_workflow.md: "Retries target a subset"): the
agent's backtracking across stages, with the changes a gate suggested of an earlier stage. The
stage runs again with the changes, then every stage after it up to G4, each with its gate's own
retries; the inversions of those windows are archived, for invert to do again. The inversion
itself is done again as a job (paco.qc.inverting.run_inversion_job)."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from paco.pipelines import record_folder
from paco.presets import apply_overrides, resolve_preset
from paco.profiles import load_profile
from paco.qc.attempts import invalidate_record
from paco.qc.budgets import run_budget
from paco.qc.config import QCConfig, read_qc_config
from paco.qc.curves import pick_line
from paco.qc.line import settle_images, settle_records
from paco.qc.log import append_attempt, latest, read_attempts, retries_in_run
from paco.qc.loops import deep_merge
from paco.qc.models import Attempt
from paco.qc.report import QCReport, build_report, read_report, write_report
from paco.qc.rerun import rerun_phase_shift
from paco.runs import RunError, RunManifest, find_run, load_manifest
from paco.runs.processing import RECORDS_FOLDER, preprocess_records, write_manifest
from paco.settings import Settings
from paco.windows import MASWWindow

type RedoStage = Literal["preprocessing", "phase_shift", "picking"]


def select_windows(
    run_id: str,
    settings: Settings,
    xmids: Sequence[float] | None = None,
    flag: str | None = None,
) -> list[str]:
    """The windows of run `run_id` to go back for: the ones at `xmids`, or the ones whose
    latest results carry `flag`, or every window when neither is given."""
    manifest = load_manifest(run_id, settings)
    folders = {window.folder for window in manifest.windows}
    if xmids is not None:
        wanted = [f"xmid_{xmid:.2f}" for xmid in xmids]
        if unknown := [unit for unit in wanted if unit not in folders]:
            raise RunError(
                f"Run '{run_id}' has no window at {', '.join(u[5:] for u in unknown)}. Its "
                f"xmids: {', '.join(w.folder[5:] for w in manifest.windows)}."
            )
        return wanted
    if flag is None:
        return [window.folder for window in manifest.windows]
    report = read_report(find_run(run_id, settings))
    flagged = [
        unit.unit
        for unit in report.units
        if unit.unit in folders
        and any(one.name == flag for flags in unit.flags.values() for one in flags)
    ]
    if not flagged:
        raise RunError(f"No window of run '{run_id}' carries the flag '{flag}'.")
    return flagged


def redo_stage(
    run_id: str,
    stage: RedoStage,
    units: Sequence[str],
    changes: Mapping[str, Any] | None,
    settings: Settings,
) -> QCReport:
    """`stage` again for the windows `units` of run `run_id`, with `changes` over their latest
    parameters, then the stages after it up to G4, each with its gate's retries. Going back to
    the preprocessing does again every record those windows use, and every window using them."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    config = read_qc_config(run_folder)
    check_budget(run_id, run_folder, config, len(manifest.windows))
    changes = dict(changes or {})
    if "masw" in changes:
        raise RunError(
            "masw cannot change within a run: the windows would move. Call run_processing again "
            "with the new window length."
        )
    windows = list(units)
    if stage == "preprocessing":
        windows = _redo_records(run_id, manifest, windows, changes, settings)
        manifest = load_manifest(run_id, settings)
        _redo_images(run_id, manifest, windows, {}, settings)
    elif stage == "phase_shift":
        _redo_images(run_id, manifest, windows, changes, settings)
    if stage == "picking":
        pick_line(run_id, settings, windows, changes, "backtrack")
    else:
        # The images changed: their windows are picked again from the run's first parameters.
        pick_line(run_id, settings, windows, None, "backtrack", fresh=True)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def check_budget(run_id: str, run_folder: Path, config: QCConfig, n_windows: int) -> None:
    """Refuse to go back once the run's retry budget is spent (rule 4): the agent's backtracks
    count against it, as the gates' retries do."""
    spent = retries_in_run(read_attempts(run_folder))
    total = run_budget(config.budgets, max(1, n_windows))
    if spent >= total:
        raise RunError(
            f"The retry budget of run '{run_id}' is spent ({spent} of {total}): you are stuck. "
            "Tell the user what the run has, and ask whether to start a new run with other "
            "settings, or to stop here."
        )


def _redo_images(
    run_id: str,
    manifest: RunManifest,
    windows: Sequence[str],
    changes: Mapping[str, Any],
    settings: Settings,
) -> None:
    """The phase shift again for `windows`, each from its latest phase-shift changes with
    `changes` over them (grouped by the changes they end with), then G2's retries."""
    run_folder = find_run(run_id, settings)
    attempts = read_attempts(run_folder)
    groups: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for unit in windows:
        attempt = latest(attempts, unit, "phase_shift")
        parameters = deep_merge(attempt.parameters if attempt is not None else {}, changes)
        key = json.dumps(parameters, sort_keys=True)
        groups.setdefault(key, (parameters, []))[1].append(unit)
    for parameters, units in groups.values():
        rerun_phase_shift(run_id, units, parameters, settings, "backtrack")
    usable = _usable_bands(run_folder)
    settle_images(run_id, run_folder, manifest, read_qc_config(run_folder), settings, usable)


def _redo_records(
    run_id: str,
    manifest: RunManifest,
    windows: Sequence[str],
    changes: Mapping[str, Any],
    settings: Settings,
) -> list[str]:
    """The records `windows` use preprocessed again with `changes` over their latest
    parameters, then G1's fixes; returns every window that uses one of those records."""
    run_folder = find_run(run_id, settings)
    config = read_qc_config(run_folder)
    profile = load_profile(manifest.profile.name, settings)
    names = {
        path.name
        for unit in windows
        for path in MASWWindow.model_validate_json(
            (run_folder / unit / "window.json").read_text()
        ).selected_files
    }
    attempts = read_attempts(run_folder)
    by_name = {record.path.name: record for record in profile.records}
    parameters: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        attempt = latest(attempts, name, "preprocessing")
        number = attempt.attempt if attempt is not None else 0
        parameters[name] = deep_merge(attempt.parameters if attempt is not None else {}, changes)
        invalidate_record(record_folder(run_folder / RECORDS_FOLDER, by_name[name]), number)
    started_at = datetime.now(UTC)
    redone = preprocess_records(
        manifest.preset,
        profile,
        run_folder,
        settings.workers,
        presets={
            name: resolve_preset(apply_overrides(manifest.preset, values), profile)
            for name, values in parameters.items()
        },
    )
    for outcome in redone:
        previous = latest(attempts, outcome.name, "preprocessing")
        append_attempt(
            run_folder,
            Attempt(
                unit=outcome.name,
                stage="preprocessing",
                attempt=(previous.attempt if previous is not None else 0) + 1,
                parameters=parameters[outcome.name],
                triggered_by="backtrack",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status=outcome.status,
                error=outcome.error,
            ),
        )
    records = tuple(
        next((one for one in redone if one.name == record.name), record)
        for record in manifest.records
    )
    records, exclusions, _ = settle_records(
        run_folder, profile, manifest.preset, records, config, settings.workers
    )
    write_manifest(
        run_id,
        run_folder,
        profile,
        manifest.preset,
        manifest.started_at,
        records,
        manifest.windows,
        exclusions,
    )
    return [
        window.folder
        for window in manifest.windows
        if (run_folder / window.folder / "window.json").exists()
        and names
        & {
            path.name
            for path in MASWWindow.model_validate_json(
                (run_folder / window.folder / "window.json").read_text()
            ).selected_files
        }
    ]


def _usable_bands(run_folder: Path) -> dict[str, tuple[float, float] | None]:
    """Each record's usable band, from its latest G1 result."""
    bands: dict[str, tuple[float, float] | None] = {}
    for attempt in read_attempts(run_folder):
        if attempt.stage == "preprocessing" and "G1" in attempt.results:
            bands[attempt.unit] = attempt.results["G1"].kept.band_hz
    return bands
