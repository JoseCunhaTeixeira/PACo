"""The picking stage the QC way (docs/qc_workflow.md, option B): S3 and G3 on every window whose
image passed G2, the picking done again for the windows G3 asks it of, then G4 over the line
and the outliers picked again along their neighbours' curve, each gate within its budgets.
What pick runs; its verdict (G4's, on the line) is what invert reads. A person's work, made in
PAC's pages (paco.qc.origin), is taken as it is: an image made there is picked, a curve picked
by hand is never picked again, and G4 compares the others with it. judge_curves judges the
automatic curves the assistant did not check, PAC's own automatic picks, picking nothing."""

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.picks import CURVES_FILE, save_picks_figures
from sigpipe.masw.quality.curve import pick_of
from sigpipe.masw.runs import (
    RunError,
    RunManifest,
    find_run,
    load_image,
    load_manifest,
    window_folders,
)
from sigpipe.masw.runs.finding import IMAGE_FILE
from sigpipe.masw.runs.origin import JUDGED

from paco.qc.attempts import set_aside
from paco.qc.budgets import budget_spent
from paco.qc.coherence import nearest_offset
from paco.qc.config import QCConfig, run_qc_config
from paco.qc.g3_curve import judge_curve
from paco.qc.g4_profile import LINE
from paco.qc.given import locked
from paco.qc.judging import judge_line, mutable_window, pick_windows, saved_m0
from paco.qc.log import append_attempt, attempts_of, latest, read_attempts, record_result
from paco.qc.loops import RetryBudget, deep_merge, next_try, refusal, spent, stage_changes
from paco.qc.models import Attempt, GateResult
from paco.qc.origin import assistant_run, run_work
from paco.qc.report import QCReport, build_report, write_report
from paco.settings import Settings

logger = logging.getLogger(__name__)


def pick_line(
    run_id: str,
    settings: Settings,
    units: Sequence[str] | None = None,
    changes: Mapping[str, Any] | None = None,
    triggered_by: str = "initial",
    fresh: bool = False,
    replace_hand: bool = False,
) -> QCReport:
    """Pick the windows `units` of run `run_id` (every window whose image passed G2 when None;
    in a run processed in PAC's pages, every window with an image), from their latest picking
    parameters (the run's first ones for a window never picked, or with `fresh`: its image
    changed) with `changes` on them, and settle G3 and G4 over the line. A window whose M0 a
    person picked is left as it is, unless the user chose to have it picked again
    (`replace_hand`)."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    config = run_qc_config(run_folder, settings.qc_config)
    attempts = read_attempts(run_folder)
    ready = imaged_windows(run_folder, manifest)
    if units is not None and (unknown := [unit for unit in units if unit not in ready]):
        raise RunError(
            f"Run '{run_id}' has no window {', '.join(unknown)} with an image G2 did not reject. "
            f"Those it has: {', '.join(ready) or 'none'}."
        )
    work = run_work(run_folder, manifest, attempts)
    given = locked(run_folder, "picking")
    jobs: dict[str, tuple[PickingParameters, tuple[float, float] | None]] = {}
    for unit in units if units is not None else list(ready):
        if work[unit].m0 == "user":
            if not replace_hand:
                continue  # a person's curve, kept as it is
            set_aside(run_folder / unit, "picking")
        picked = None if fresh else latest(attempts, unit, "picking")
        base = picked.parameters if picked is not None else config.picking.model_dump()
        g2 = ready[unit]
        if picked is None and g2 is not None and (wanted := stage_changes(g2, "picking", given)):
            # G2's advice for the picking, which an older log may hold: two modes (G2 keeps
            # competing ridges, the fundamental mode picked as the slowest ridge).
            base = deep_merge(base, wanted[0])
        parameters = PickingParameters.model_validate(deep_merge(base, changes or {}))
        jobs[unit] = (parameters, g2.kept.band_hz if g2 is not None else None)
    if jobs:
        pick_windows(run_folder, jobs, config, triggered_by, settings.workers)
    settle_curves(run_folder, manifest, config, settings.workers)
    if triggered_by == "initial":
        # The earlier stages G2 and G3 blame, done again once (the redo tool picks again
        # through here, as another trigger: no loop); then the line again.
        from paco.qc.redo import settle_earlier  # redo imports pick_line

        if settle_earlier(run_id, settings):
            settle_curves(run_folder, load_manifest(run_id, settings), config, settings.workers)
    # The picks along the line as PAC shows them, before any inversion: best effort.
    try:
        save_picks_figures(run_folder, window_folders(run_folder))
    except Exception:
        logger.exception("Could not draw the picks' figures of %s", run_folder)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def settle_curves(
    run_folder: Path, manifest: RunManifest, config: QCConfig, workers: int = 1
) -> None:
    """G3's re-picks until none is asked or the budgets are spent, then G4 over the line, and
    its outliers picked again (then G3 on them) until G4 asks for none; in up to `workers`
    processes."""
    while True:
        _settle_g3(run_folder, manifest, config, workers)
        results = judge_line(run_folder, manifest, config)
        again = _retries(
            [result for result in results if result.unit != LINE], run_folder, config, manifest
        )
        if not again:
            return
        _repick(run_folder, again, config, workers)


def _settle_g3(run_folder: Path, manifest: RunManifest, config: QCConfig, workers: int) -> None:
    while True:
        attempts = read_attempts(run_folder)
        work = run_work(run_folder, manifest, attempts)
        # G3's results on the curves the assistant's checks are of: not a person's, nor one
        # picked again in PAC since.
        results = [
            attempt.results["G3"]
            for window in manifest.windows
            if work[window.folder].m0 in ("judged", None)
            and (attempt := latest(attempts, window.folder, "picking")) is not None
            and "G3" in attempt.results
        ]
        again = _retries(results, run_folder, config, manifest)
        if not again:
            return
        _repick(run_folder, again, config, workers)


def _retries(
    results: Sequence[GateResult], run_folder: Path, config: QCConfig, manifest: RunManifest
) -> dict[str, tuple[dict[str, Any], str]]:
    """The windows of `results` to pick again, with their parameters and trigger; the ones
    that ask for it without budget left are rejected ("budget spent")."""
    attempts = read_attempts(run_folder)
    budget = RetryBudget(attempts, config.budgets, max(1, len(manifest.windows)))
    given = locked(run_folder, "picking")
    again: dict[str, tuple[dict[str, Any], str]] = {}
    for result in results:
        attempt = latest(attempts, result.unit, "picking")
        if attempt is None:
            continue
        wanted = next_try(result, "picking", budget, attempt.parameters, given)
        if wanted is not None:
            again[result.unit] = wanted
        elif spent(result, "picking"):
            record_result(
                run_folder,
                result.unit,
                "picking",
                attempt.attempt,
                budget_spent(result, *refusal(result, "picking", attempt.parameters, given)),
            )
    return again


def _repick(
    run_folder: Path,
    again: Mapping[str, tuple[dict[str, Any], str]],
    config: QCConfig,
    workers: int,
) -> None:
    """The windows of `again` picked again, with their parameters, batch by trigger, each
    against its latest image's band (G2's)."""
    attempts = read_attempts(run_folder)
    by_trigger: dict[str, dict[str, tuple[PickingParameters, tuple[float, float] | None]]] = {}
    for unit, (parameters, trigger) in again.items():
        imaged = latest(attempts, unit, "phase_shift")
        g2 = imaged.results.get("G2") if imaged is not None else None
        band = g2.kept.band_hz if g2 is not None else None
        by_trigger.setdefault(trigger, {})[unit] = (
            PickingParameters.model_validate(parameters),
            band,
        )
    for trigger, jobs in by_trigger.items():
        pick_windows(run_folder, jobs, config, trigger, workers)


def imaged_windows(run_folder: Path, manifest: RunManifest) -> dict[str, GateResult | None]:
    """The windows with an image to pick, with G2's result: those whose latest image G2 judged
    and did not reject; in a run processed in PAC's pages, every window with an image, a
    person's, which no gate judged (None)."""
    attempts = read_attempts(run_folder)
    if not assistant_run(attempts):
        return {
            window.folder: None
            for window in manifest.windows
            if (run_folder / window.folder / IMAGE_FILE).exists()
        }
    ready: dict[str, GateResult | None] = {}
    for window in manifest.windows:
        attempt = latest(attempts, window.folder, "phase_shift")
        if attempt is None or attempt.status != "succeeded":
            continue
        g2 = attempt.results.get("G2")
        if g2 is not None and g2.verdict != "reject":
            ready[window.folder] = g2
    return ready


def judge_curves(run_id: str, settings: Settings, units: Sequence[str] | None = None) -> QCReport:
    """G3 on the automatic M0 curves of run `run_id` its gates did not judge as they are now
    (PAC's own automatic picks), or on those of `units`, each judged as it is, nothing picked:
    an attempt of the window's picking triggered by "judge"; then G4 over the line. A person's
    curve is left out: verified by them."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    config = run_qc_config(run_folder, settings.qc_config)
    attempts = read_attempts(run_folder)
    work = run_work(run_folder, manifest, attempts)
    ready = imaged_windows(run_folder, manifest)
    wanted = (
        [unit for unit in units if work[unit].m0 in ("judged", "unjudged")]
        if units is not None
        else [unit for unit, one in work.items() if one.m0 == "unjudged"]
    )
    for unit in wanted:
        if unit not in ready:
            continue  # its image rejected, or none: nothing to judge the curve on
        folder = run_folder / unit
        image = load_image(folder)
        curve = saved_m0(folder / CURVES_FILE)
        previous = latest(attempts, unit, "picking")
        picking = (
            PickingParameters.model_validate(previous.parameters)
            if previous is not None and previous.parameters
            else config.picking
        )
        g2 = ready[unit]
        started_at = datetime.now(UTC)
        g3 = judge_curve(
            unit,
            image,
            pick_of(curve, image) if curve is not None else None,
            config.curve,
            g2.kept.band_hz if g2 is not None else None,
            picking,
            nearest_offset(folder),
            mutable_window(run_folder, unit),
        )
        append_attempt(
            run_folder,
            Attempt(
                unit=unit,
                stage="picking",
                attempt=len(attempts_of(attempts, unit, "picking")) + 1,
                parameters=picking.model_dump(),
                triggered_by=JUDGED,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status="succeeded",
                results={g3.gate: g3},
            ),
        )
    judge_line(run_folder, manifest, config)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report
