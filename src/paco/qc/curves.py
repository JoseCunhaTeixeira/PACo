"""The picking stage the QC way (docs/qc_workflow.md, option B): S3 and G3 on every window whose
image passed G2, the picking done again for the windows G3 asks it of, then G4 over the line
and the outliers picked again along their neighbours' curve, each gate within its budgets.
What pick runs; its verdict (G4's, on the line) is what invert reads."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.runs import RunError, RunManifest, find_run, load_image, load_manifest

from paco.qc.budgets import budget_spent
from paco.qc.config import QCConfig, read_qc_config
from paco.qc.g4_profile import LINE
from paco.qc.judging import judge_line, judge_picking
from paco.qc.log import latest, read_attempts, record_result
from paco.qc.loops import RetryBudget, deep_merge, next_try, spent, stage_changes
from paco.qc.models import GateResult
from paco.qc.report import QCReport, build_report, write_report
from paco.settings import Settings


def pick_line(
    run_id: str,
    settings: Settings,
    units: Sequence[str] | None = None,
    changes: Mapping[str, Any] | None = None,
    triggered_by: str = "initial",
    fresh: bool = False,
) -> QCReport:
    """Pick the windows `units` of run `run_id` (every window whose image passed G2 when None),
    from their latest picking parameters (the run's first ones for a window never picked, or
    with `fresh`: its image changed) with `changes` on them, and settle G3 and G4 over the
    line. The run must have been processed the QC way (run_processing)."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    config = read_qc_config(run_folder)
    attempts = read_attempts(run_folder)
    ready = _imaged(run_folder, manifest)
    if units is not None and (unknown := [unit for unit in units if unit not in ready]):
        raise RunError(
            f"Run '{run_id}' has no window {', '.join(unknown)} with an image G2 did not reject. "
            f"Those it has: {', '.join(ready) or 'none'}."
        )
    for unit in units if units is not None else list(ready):
        picked = None if fresh else latest(attempts, unit, "picking")
        base = picked.parameters if picked is not None else config.picking.model_dump()
        g2 = ready[unit]
        if picked is None and (wanted := stage_changes(g2, "picking")) is not None:
            base = deep_merge(base, wanted[0])  # G2's advice for the picking: two modes
        parameters = PickingParameters.model_validate(deep_merge(base, changes or {}))
        folder = run_folder / unit
        judge_picking(
            run_folder, unit, load_image(folder), parameters, config, g2.kept.band_hz, triggered_by
        )
    settle_curves(run_folder, manifest, config)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def settle_curves(run_folder: Path, manifest: RunManifest, config: QCConfig) -> None:
    """G3's re-picks until none is asked or the budgets are spent, then G4 over the line, and
    its outliers picked again (then G3 on them) until G4 asks for none."""
    while True:
        _settle_g3(run_folder, manifest, config)
        results = judge_line(run_folder, manifest, config)
        attempts = read_attempts(run_folder)
        again = _retries(
            [result for result in results if result.unit != LINE], run_folder, config, manifest
        )
        if not again:
            return
        for unit, (parameters, trigger) in again.items():
            _repick(run_folder, unit, parameters, trigger, config, attempts)


def _settle_g3(run_folder: Path, manifest: RunManifest, config: QCConfig) -> None:
    while True:
        attempts = read_attempts(run_folder)
        results = [
            attempt.results["G3"]
            for window in manifest.windows
            if (attempt := latest(attempts, window.folder, "picking")) is not None
            and "G3" in attempt.results
        ]
        again = _retries(results, run_folder, config, manifest)
        if not again:
            return
        for unit, (parameters, trigger) in again.items():
            _repick(run_folder, unit, parameters, trigger, config, attempts)


def _retries(
    results: Sequence[GateResult], run_folder: Path, config: QCConfig, manifest: RunManifest
) -> dict[str, tuple[dict[str, Any], str]]:
    """The windows of `results` to pick again, with their parameters and trigger; the ones
    that ask for it without budget left are rejected ("budget spent")."""
    attempts = read_attempts(run_folder)
    budget = RetryBudget(attempts, config.budgets, max(1, len(manifest.windows)))
    again: dict[str, tuple[dict[str, Any], str]] = {}
    for result in results:
        attempt = latest(attempts, result.unit, "picking")
        if attempt is None:
            continue
        wanted = next_try(result, "picking", budget, attempt.parameters)
        if wanted is not None:
            again[result.unit] = wanted
        elif spent(result, "picking"):
            record_result(run_folder, result.unit, "picking", attempt.attempt, budget_spent(result))
    return again


def _repick(
    run_folder: Path,
    unit: str,
    parameters: dict[str, Any],
    trigger: str,
    config: QCConfig,
    attempts: Sequence[Any],
) -> None:
    imaged = latest(attempts, unit, "phase_shift")
    g2 = imaged.results.get("G2") if imaged is not None else None
    folder = run_folder / unit
    judge_picking(
        run_folder,
        unit,
        load_image(folder),
        PickingParameters.model_validate(parameters),
        config,
        g2.kept.band_hz if g2 is not None else None,
        trigger,
    )


def _imaged(run_folder: Path, manifest: RunManifest) -> dict[str, GateResult]:
    """The windows whose latest image G2 judged and did not reject, with G2's result."""
    attempts = read_attempts(run_folder)
    if latest(attempts, LINE, "phase_shift") is None and not any(
        "G2" in attempt.results for attempt in attempts
    ):
        raise RunError(
            f"Run '{manifest.run_id}' was not processed the QC way: call run_processing again."
        )
    ready: dict[str, GateResult] = {}
    for window in manifest.windows:
        attempt = latest(attempts, window.folder, "phase_shift")
        if attempt is None or attempt.status != "succeeded":
            continue
        g2 = attempt.results.get("G2")
        if g2 is not None and g2.verdict != "reject":
            ready[window.folder] = g2
    return ready
