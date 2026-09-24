"""Doing a stage again for some windows of a run (rule 3, and the decision on attempts): the
window's results from that stage on move to attempts/, the stage runs with the run's parameters
plus the overrides given, and the QC log records the attempt. The other windows keep their
results."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from pydantic import ValidationError

from paco.picking import PickingParameters
from paco.presets import apply_overrides, resolve_preset
from paco.profiles import load_profile
from paco.qc.attempts import invalidate
from paco.qc.config import read_qc_config
from paco.qc.judging import judge_picking
from paco.qc.log import append_attempt, attempts_of, ensure_initial_attempts, latest
from paco.qc.models import Attempt, GateResult
from paco.runs import RunError, WindowOutcome, find_run, load_image, load_manifest
from paco.runs.processing import process_windows
from paco.settings import Settings
from paco.windows import build_windows


def rerun_phase_shift(
    run_id: str,
    units: Sequence[str],
    overrides: Mapping[str, object],
    settings: Settings,
    triggered_by: str = "backtrack",
) -> tuple[WindowOutcome, ...]:
    """The phase shift again for the windows `units` (their folders, xmid_<x>) of run `run_id`,
    with `overrides` on the run's preset; the picking and inversion of those windows go to
    attempts/ with the images they came from. Windows cannot move: `masw` cannot change."""
    if "masw" in overrides:
        raise RunError(
            "masw cannot change when the phase shift is done again: the windows would move. "
            "Start a new run for other windows."
        )
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    profile = load_profile(manifest.profile.name, settings)
    preset = resolve_preset(apply_overrides(manifest.preset, overrides), profile)
    by_folder = {
        f"xmid_{window.xmid:.2f}": window for window in build_windows(profile, preset.masw)
    }
    if unknown := [unit for unit in units if unit not in by_folder]:
        raise RunError(
            f"Run '{run_id}' has no window {', '.join(unknown)}. Its windows: "
            f"{', '.join(window.folder for window in manifest.windows)}."
        )

    attempts = ensure_initial_attempts(run_folder, manifest)
    numbers = {unit: len(attempts_of(attempts, unit, "phase_shift")) + 1 for unit in units}
    for unit in units:
        invalidate(run_folder / unit, "phase_shift", numbers[unit] - 1)
    started_at = datetime.now(UTC)
    outcomes = process_windows(
        preset,
        [by_folder[unit] for unit in units],
        manifest.records,
        run_folder,
        settings.workers,
        exclusions=manifest.exclusions,
    )
    finished_at = datetime.now(UTC)
    for outcome in outcomes:
        append_attempt(
            run_folder,
            Attempt(
                unit=outcome.folder,
                stage="phase_shift",
                attempt=numbers[outcome.folder],
                parameters=dict(overrides),
                triggered_by=triggered_by,
                started_at=started_at,
                finished_at=finished_at,
                status=outcome.status,
                error=outcome.error,
            ),
        )
    return outcomes


def rerun_picking(
    run_id: str,
    units: Sequence[str],
    overrides: Mapping[str, object],
    settings: Settings,
    triggered_by: str = "backtrack",
) -> tuple[GateResult, ...]:
    """The picking again for the windows `units` of run `run_id`, with `overrides` on the
    parameters of each window's latest pick (G4's guide differs from one window to the next),
    then G3 on each new curve; the previous curve and the inversion of those windows go to
    attempts/. The run must have been judged: its QC configuration gives the thresholds."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    config = read_qc_config(run_folder)
    known = {window.folder for window in manifest.windows if window.status == "succeeded"}
    if unknown := [unit for unit in units if unit not in known]:
        raise RunError(
            f"Run '{run_id}' has no processed window {', '.join(unknown)}. Its windows: "
            f"{', '.join(sorted(known))}."
        )
    if unexpected := sorted(set(overrides) - set(PickingParameters.model_fields)):
        raise RunError(
            f"Unknown picking parameter(s) {', '.join(unexpected)}. The picking's are "
            f"{', '.join(PickingParameters.model_fields)}."
        )

    attempts = ensure_initial_attempts(run_folder, manifest)
    results: list[GateResult] = []
    for unit in units:
        previous = latest(attempts, unit, "picking")
        base = previous.parameters if previous is not None else config.picking.model_dump()
        try:
            picking = PickingParameters.model_validate({**base, **overrides})
        except ValidationError as error:
            raise RunError(f"picking: {error.errors()[0]['msg']}") from error
        image_attempt = latest(attempts, unit, "phase_shift")
        g2 = image_attempt.results.get("G2") if image_attempt is not None else None
        band = g2.kept.band_hz if g2 is not None else None
        image = load_image(run_folder / unit)
        results.append(judge_picking(run_folder, unit, image, picking, config, band, triggered_by))
    return tuple(results)
