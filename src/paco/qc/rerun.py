"""Doing a stage again for some windows of a run (rule 3): the window's results from that stage
on move to attempts/, the stage runs with the run's parameters plus the overrides given, and the
QC log records the attempt. The other windows keep their results."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.presets import ActivePreset, PassivePreset, apply_overrides, resolve_preset
from sigpipe.masw.profiles import Profile, load_profile
from sigpipe.masw.runs import RunError, WindowOutcome, find_run, load_image, load_manifest
from sigpipe.masw.runs.processing import process_windows
from sigpipe.masw.runs.stopping import Stopped
from sigpipe.masw.windows import MASWWindow, build_windows

from paco import stopping
from paco.qc.attempts import invalidate, restore
from paco.qc.coherence import near_field_windows
from paco.qc.config import read_qc_config
from paco.qc.g4_profile import LINE
from paco.qc.judging import judge_picking
from paco.qc.log import (
    afresh,
    append_attempt,
    attempts_of,
    ensure_initial_attempts,
    latest,
    read_attempts,
    starts_afresh,
)
from paco.qc.models import Attempt, GateResult
from paco.settings import Settings


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
    by_folder = line_windows(run_folder, profile, preset)
    if unknown := [unit for unit in units if unit not in by_folder]:
        raise RunError(
            f"Run '{run_id}' has no window {', '.join(unknown)}. Its windows: "
            f"{', '.join(window.folder for window in manifest.windows)}."
        )

    attempts = ensure_initial_attempts(run_folder, manifest)
    # The attempt each window's files are archived as, to give them back on a stop.
    archived = {unit: len(attempts_of(attempts, unit, "phase_shift")) for unit in units}
    for unit in units:
        invalidate(run_folder / unit, "phase_shift", archived[unit])
    started_at = datetime.now(UTC)
    # Asked of the agent: each window's image, and all that was made of it, started afresh; a
    # gate's retry goes on from its attempts.
    fresh = starts_afresh(triggered_by)

    def log(outcomes: tuple[WindowOutcome, ...]) -> None:
        finished_at = datetime.now(UTC)
        for outcome in outcomes:
            unit = outcome.folder
            attempt = Attempt(
                unit=unit,
                stage="phase_shift",
                attempt=archived[unit] + 1,
                parameters=dict(overrides),
                triggered_by=triggered_by,
                started_at=started_at,
                finished_at=finished_at,
                status=outcome.status,
                error=outcome.error,
            )
            append_attempt(run_folder, afresh(run_folder, attempt) if fresh else attempt)

    try:
        outcomes = process_windows(
            preset,
            [by_folder[unit] for unit in units],
            manifest.records,
            run_folder,
            settings.workers,
            exclusions=manifest.exclusions,
            stop=stopping.current(),
        )
    except Stopped as stopped:
        # The windows that finished logged; the others given back their previous attempt.
        done = cast(tuple[WindowOutcome, ...], stopped.kept or ())
        log(done)
        for unit in set(units) - {outcome.folder for outcome in done}:
            restore(run_folder / unit, "phase_shift", archived[unit])
        raise
    log(outcomes)
    return outcomes


def line_windows(
    run_folder: Path, profile: Profile, preset: ActivePreset | PassivePreset
) -> dict[str, MASWWindow]:
    """The run's windows as `preset` builds them, by folder (xmid_<x>), the shots out of the near
    field as the line kept them: what the phase shift stacks, before the traces and records G1
    left out."""
    windows = build_windows(profile, preset.masw)
    line = latest(read_attempts(run_folder), LINE, "phase_shift")
    near = line.parameters.get("near_field", {}) if line is not None else {}
    if isinstance(near, Mapping) and isinstance(distance := near.get("distance_m"), int | float):
        windows, _ = near_field_windows(windows, float(distance))
    return {f"xmid_{window.xmid:.2f}": window for window in windows}


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
