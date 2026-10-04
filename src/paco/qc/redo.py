"""Going back to a stage (docs/qc_workflow.md: "Retries target a subset"): the agent's
backtracking across stages, with the changes a gate suggested or the user asked. The records and
the images are the line's, the same for every window: their stage is done again for the whole
line with the line's settings changed, then G2, and every window picked again. The picking is
each window's: done again for some windows. Then every stage after it up to G4, each with its
gate's own retries; the inversions of those windows are archived, for invert to do again. The
inversion itself is done again as a job (paco.qc.inverting.run_inversion_job)."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from sigpipe.masw.presets import apply_overrides, resolve_preset
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.runs import RunError, RunManifest, find_run, load_manifest
from sigpipe.masw.windows import Exclusions

from paco.qc.budgets import run_budget
from paco.qc.config import QCConfig, read_qc_config
from paco.qc.curves import imaged_windows, pick_line
from paco.qc.g4_profile import LINE
from paco.qc.line import (
    image_line,
    judge_images,
    keep_line_asks,
    line_reach,
    reprocess_records,
    settle_records,
)
from paco.qc.log import (
    LINE_CHANGE,
    append_attempt,
    latest,
    read_attempts,
    retries_in_run,
)
from paco.qc.origin import run_work
from paco.qc.report import QCReport, build_report, write_report
from paco.qc.state import read_report
from paco.qc.stuck import Stuck
from paco.settings import Settings

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
    trigger: str = "backtrack",
    replace_hand: bool = False,
) -> QCReport:
    """`stage` again with `changes`, then the stages after it up to G4. The preprocessing and the
    phase shift are the line's: done again for every window (`units` all of them; some only,
    refused), the line's settings changed (_redo_line). The picking, for the windows `units`, from
    their latest parameters with `changes`, with G3's retries. `trigger`: who asked (the agent's
    backtrack, or a gate's flag, "<gate>:<flag>"). A window holding a person's work
    (paco.qc.origin) keeps it (never imaged again, its M0 by hand never picked again), unless
    they chose to have it done again (`replace_hand`); refused when no window is left to redo."""
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
    work = run_work(run_folder, manifest)
    if stage in ("preprocessing", "phase_shift"):
        if set(units) != {window.folder for window in manifest.windows}:
            said = stage.replace("_", " ")
            raise RunError(
                f"The {said} is the line's, the same for every window: redo it for the whole line "
                "(no xmids, no flag), or leave it as it is."
            )
        if not replace_hand and all(work[unit].frozen for unit in units):
            raise RunError(
                "Every window holds work made by hand in PAC (verified by the user): left as it "
                "is, nothing to redo."
            )
        return _redo_line(run_id, manifest, stage, changes, settings, trigger, replace_hand)
    kept = [] if replace_hand else [unit for unit in units if work[unit].m0 == "user"]
    windows = [unit for unit in units if unit not in kept]
    if not windows:
        raise RunError(
            f"{', '.join(kept)} hold work made by hand in PAC (verified by the user): left as "
            "they are, nothing to redo."
        )
    pick_line(run_id, settings, windows, changes, trigger, replace_hand=replace_hand)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def _redo_line(
    run_id: str,
    manifest: RunManifest,
    stage: RedoStage,
    changes: Mapping[str, Any],
    settings: Settings,
    trigger: str,
    replace_hand: bool,
) -> QCReport:
    """The line's `stage` done again with `changes` on the line's settings, the same for every
    record and window: for the preprocessing, every record preprocessed again and judged by G1
    afresh (what it leaves out decided again); every window imaged again and judged by G2, a
    person's work kept unless `replace_hand`; then every window picked again from the run's first
    parameters, those whose new image G2 rejected left out (nothing to pick). The records' and
    images' attempts are the line's change (LINE_CHANGE); the line's attempt says it."""
    run_folder = find_run(run_id, settings)
    config = read_qc_config(run_folder)
    profile = load_profile(manifest.profile.name, settings)
    preset = resolve_preset(apply_overrides(manifest.preset, changes), profile)
    records = tuple(manifest.records)
    exclusions = manifest.exclusions
    usable = _usable_bands(run_folder)
    if stage == "preprocessing":
        records = reprocess_records(
            run_folder, profile, preset, records, Exclusions(), settings.workers, LINE_CHANGE
        )
        reach = line_reach(run_folder, profile, records, config, preset)
        records, exclusions, usable = settle_records(
            run_folder, profile, preset, records, config, reach
        )
    imaged = image_line(
        run_id,
        run_folder,
        profile,
        preset,
        records,
        exclusions,
        settings,
        LINE_CHANGE,
        replace_hand,
    )
    judge_images(run_folder, load_manifest(run_id, settings), config, settings, usable)
    keep_line_asks(run_folder)
    _line_redone(run_folder, stage, changes)
    ready = imaged_windows(run_folder, load_manifest(run_id, settings))
    pick_line(
        run_id,
        settings,
        [unit for unit in imaged if unit in ready],
        None,
        trigger,
        fresh=True,
        replace_hand=replace_hand,
    )
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def _line_redone(run_folder: Path, stage: RedoStage, changes: Mapping[str, Any]) -> None:
    """The line's attempt, logged again with the line redone: the settings the processing chose
    that `changes` replaced taken out of it (theirs now, not the checks'), and a note saying the
    redo."""
    line = latest(read_attempts(run_folder), LINE, "phase_shift")
    if line is None:
        return
    said = stage.replace("_", " ")
    shown = "; ".join(
        f"{name} {json.dumps(value, sort_keys=True)}" for name, value in changes.items()
    )
    note = f"The {said} done again for the whole line" + (f", with {shown}." if shown else ".")
    append_attempt(
        run_folder,
        line.model_copy(
            update={
                "parameters": _without(line.parameters, changes),
                "notes": (*line.notes, note),
            }
        ),
    )


def _without(values: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """`values` without the leaves `changes` gives."""
    kept: dict[str, Any] = {}
    for key, value in values.items():
        change = changes.get(key)
        if key not in changes:
            kept[key] = value
        elif isinstance(value, Mapping) and isinstance(change, Mapping):
            inner = _without(cast(Mapping[str, Any], value), cast(Mapping[str, Any], change))
            if inner:
                kept[key] = inner
    return kept


class BudgetSpent(Stuck):
    """The run's retry budget is spent: no stage may be done again."""


def check_budget(run_id: str, run_folder: Path, config: QCConfig, n_windows: int) -> None:
    """Refuse to go back once the run's retry budget is spent (rule 4): the agent's backtracks
    count against it, as the gates' retries do."""
    spent = retries_in_run(read_attempts(run_folder))
    total = run_budget(config.budgets, max(1, n_windows))
    if spent >= total:
        raise BudgetSpent(
            f"The retry budget of run '{run_id}' is spent ({spent} of {total}): you are stuck. "
            "Tell the user what the run has, and ask whether to start a new run with other "
            "settings, or to stop here."
        )


def _usable_bands(run_folder: Path) -> dict[str, tuple[float, float] | None]:
    """Each record's usable band, from its latest G1 result."""
    bands: dict[str, tuple[float, float] | None] = {}
    for attempt in read_attempts(run_folder):
        if attempt.stage == "preprocessing" and "G1" in attempt.results:
            bands[attempt.unit] = attempt.results["G1"].kept.band_hz
    return bands
