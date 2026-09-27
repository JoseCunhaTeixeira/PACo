"""A line from its records to G2, the way the QC workflow runs it (docs/qc_workflow.md, option B):
S1 on every record, G1 with its fixes (a record preprocessed again with the changes G1 asks
for, traces and records it excludes left out of the windows), G1 over the line (the receivers
off the amplitude decay in most records left out of every window), the coherence rules for S2
(the band capped by G1's usable band, the window length from the ladder, the shots stacked
between the near field and the line's reach), S2 on the whole line, and G2 with its own retries
(the phase shift done again for the windows it flags). What run_processing runs; the picking
and G3, G4 are pick's."""

import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from sigpipe.masw.pipelines import PREPROCESSED, record_folder
from sigpipe.masw.presets import (
    ActivePreset,
    PassivePreset,
    apply_overrides,
    make_preset,
    resolve_preset,
)
from sigpipe.masw.profiles import Profile, load_profile
from sigpipe.masw.quality.signal import snr_reach, trace_snrs
from sigpipe.masw.runs import RecordOutcome, RunError, RunManifest, load_image
from sigpipe.masw.runs.processing import (
    RECORDS_FOLDER,
    ProgressCallback,
    new_run_folder,
    preprocess_records,
    process_windows,
    write_manifest,
)
from sigpipe.masw.runs.stopping import Stopped
from sigpipe.masw.windows import Exclusions, build_windows

from paco import stopping
from paco.qc.attempts import invalidate_record, restore_record
from paco.qc.budgets import budget_spent
from paco.qc.coherence import (
    TrialJudge,
    cap_band,
    choose_length,
    given_length,
    near_field,
    near_field_windows,
    near_note,
)
from paco.qc.config import QCConfig, snapshot_qc_config
from paco.qc.g1_signal import decay_outliers, judge_receivers, judge_signal
from paco.qc.g2_image import judge_image
from paco.qc.g4_profile import LINE
from paco.qc.judging import shared_band, stream_of
from paco.qc.log import append_attempt, latest, read_attempts, record_notes, record_result
from paco.qc.loops import RetryBudget, deep_merge, next_try, spent
from paco.qc.models import Attempt, ExcludeRecord, ExcludeTraces, GateResult, Stage
from paco.qc.report import QCReport, build_report, write_report
from paco.qc.rerun import rerun_phase_shift
from paco.runs import PACKAGES
from paco.settings import Settings

type Bands = dict[str, tuple[float, float] | None]  # each record's usable band, from G1


def process_line(
    profile: str,
    overrides: Mapping[str, object] | None,
    settings: Settings,
    config: QCConfig,
    on_progress: ProgressCallback | None = None,
) -> QCReport:
    """Process `profile` with its preset and `overrides` (the user's settings, where the rules
    and the gates start from), and judge it up to G2, each gate retrying what it can. The rules'
    changes are logged as notes of the line's phase shift, the ladder's trials kept in
    coherence.json."""
    loaded = load_profile(profile, settings)
    given, n = given_length(overrides), len(loaded.receivers)
    if given is not None and given > n:
        # A request the data do not allow: the agent must ask (the host then leaves its
        # question as it is).
        raise RunError(
            f"length ({given}) exceeds the {n} receivers of profile '{profile}': no window that "
            f"long fits the line, you are stuck. Ask the user which length to use, with options: "
            f"the whole line ({n}), half of it ({n // 2}), or the ladder's proposal (no length)."
        )
    # The profile's own mode unless the settings name another (passive-active on an active
    # profile): PAC's three modes.
    mode = (overrides or {}).get("mode", loaded.kind)
    preset = resolve_preset(make_preset(str(mode), overrides), loaded)
    run_id, run_folder = new_run_folder(settings.output_dir / profile)
    try:
        return _process(
            run_id, run_folder, profile, overrides, settings, config, on_progress, loaded, preset
        )
    except Stopped:
        # A run stopped before its end is not kept: nothing of it half-made.
        shutil.rmtree(run_folder, ignore_errors=True)
        raise


def _process(
    run_id: str,
    run_folder: Path,
    profile: str,
    overrides: Mapping[str, object] | None,
    settings: Settings,
    config: QCConfig,
    on_progress: ProgressCallback | None,
    loaded: Profile,
    preset: ActivePreset | PassivePreset,
) -> QCReport:
    """process_line's work, in the new run `run_folder`."""
    mode = (overrides or {}).get("mode", loaded.kind)
    snapshot_qc_config(config, run_folder)
    started_at = datetime.now(UTC)
    records = preprocess_records(
        preset, loaded, run_folder, settings.workers, stop=stopping.current()
    )
    for record in records:
        _log(run_folder, record.name, "preprocessing", 1, {}, "initial", started_at, record)
    reach = line_reach(run_folder, loaded, records, config)
    records, exclusions, usable = settle_records(
        run_folder, loaded, preset, records, config, settings.workers, reach
    )
    # The band the records G1 kept share: a rejected record constrains nothing.
    kept = [band for name, band in usable.items() if name not in exclusions.records]
    band, band_notes = cap_band(preset, kept, loaded.nyquist_hz)
    far, far_notes = far_limit(overrides, reach, config.signal.reach_snr_db)
    band = deep_merge(band, far)
    choice = choose_length(
        loaded,
        resolve_preset(apply_overrides(preset, band), loaded),
        records,
        run_folder,
        TrialJudge(config.coherence, config.curve, config.picking, config.image),
        settings.workers,
        given_length(overrides),
        exclusions,
    )
    changes: dict[str, Any] = deep_merge(band, {"masw": {"length": choice.length}})
    preset = resolve_preset(apply_overrides(preset, changes), loaded)
    windows = build_windows(loaded, preset.masw)
    near, longest = near_field(overrides, str(mode), choice)
    near_notes: tuple[str, ...] = ()
    logged = dict(changes)
    if near is not None and longest is not None:
        windows, near_only = near_field_windows(windows, near)
        near_notes = (near_note(near, longest, near_only),)
        logged["near_field"] = {"distance_m": near}
    if not windows:
        raise RunError(
            f"No window of {choice.length} receivers of profile '{profile}' has a valid shot: "
            "widen masw.distance_min and masw.distance_max."
        )
    imaged_at = datetime.now(UTC)
    outcomes = process_windows(
        preset,
        windows,
        records,
        run_folder,
        settings.workers,
        on_progress,
        exclusions=exclusions,
        stop=stopping.current(),
    )
    manifest = write_manifest(
        run_id,
        run_folder,
        loaded,
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
    append_attempt(
        run_folder,
        Attempt(
            unit=LINE,
            stage="phase_shift",
            attempt=1,
            parameters=logged,
            triggered_by="initial",
            started_at=started_at,
            finished_at=manifest.finished_at,
            status="succeeded",
            notes=band_notes + far_notes + choice.notes + near_notes,
        ),
    )
    settle_images(run_id, run_folder, manifest, config, settings, usable)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def line_reach(
    run_folder: Path, profile: Profile, records: tuple[RecordOutcome, ...], config: QCConfig
) -> float | None:
    """How far from the shots the line's traces still carry the wave (G1's per-trace SNR over
    every active record, `snr_reach`, in bins of a fifteenth of the line); None on a passive
    line, or when no distance falls below G1's SNR limit."""
    if profile.kind != "active":
        return None
    measured = [
        found
        for record in records
        if record.status == "succeeded"
        and (
            found := trace_snrs(
                stream_of(run_folder / record.folder / PREPROCESSED),
                config.signal.vg_min,
                config.signal.vg_max,
                config.signal.pad_s,
            )
        )
        is not None
    ]
    positions = [receiver.x for receiver in profile.receivers]
    span = max(positions) - min(positions)
    return snr_reach(measured, config.signal.reach_snr_db, span / 15) if span > 0 else None


def far_limit(
    overrides: Mapping[str, object] | None, reach: float | None, min_db: float
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """masw.distance_max at the line's reach, unless the user gave one: a window stacks no shot
    whose traces there are mostly noise. Overrides and notes."""
    masw = (overrides or {}).get("masw")
    if reach is None or (isinstance(masw, Mapping) and "distance_max" in masw):
        return {}, ()
    note = (
        f"masw distance_max {reach:g} m: beyond it from the shot, the traces' median SNR falls "
        f"under {min_db:g} dB (G1), so the windows stack no farther shot."
    )
    return {"masw": {"distance_max": reach}}, (note,)


def settle_records(
    run_folder: Path,
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    records: tuple[RecordOutcome, ...],
    config: QCConfig,
    workers: int,
    reach_m: float | None = None,
) -> tuple[tuple[RecordOutcome, ...], Exclusions, Bands]:
    """G1 on every preprocessed record, with its fixes, until nothing is left to fix or the
    budgets are spent: the traces and records it excludes are recorded (for S2 to leave out, and
    for G1 to judge the record without them), a record G1 asks to preprocess differently is
    preprocessed again. Returns the records, the exclusions and each record's usable band."""
    outcomes = {record.name: record for record in records}
    exclusions = Exclusions()
    results: dict[str, GateResult] = {}
    n_units = max(1, len(records))
    active = profile.kind == "active"
    while True:
        attempts = read_attempts(run_folder)
        results = {}
        for name, outcome in outcomes.items():
            if outcome.status != "succeeded" or name in exclusions.records:
                continue
            stream = stream_of(run_folder / outcome.folder / PREPROCESSED)
            result = judge_signal(
                name,
                stream,
                config.signal,
                active=active,
                excluded=exclusions.traces.get(name, ()),
                reach_m=reach_m,
            )
            results[name] = result
            attempt = latest(attempts, name, "preprocessing")
            if attempt is not None:
                record_result(run_folder, name, "preprocessing", attempt.attempt, result)
        before = exclusions
        for name, result in results.items():
            for flag in result.flags:
                if isinstance(flag.action, ExcludeTraces):
                    exclusions = exclusions.with_traces(name, flag.action.traces)
                elif isinstance(flag.action, ExcludeRecord):
                    exclusions = exclusions.with_record(name)
        attempts = read_attempts(run_folder)
        budget = RetryBudget(attempts, config.budgets, n_units)
        again: dict[str, tuple[dict[str, Any], str]] = {}
        for name, result in results.items():
            attempt = latest(attempts, name, "preprocessing")
            previous = attempt.parameters if attempt is not None else {}
            wanted = next_try(result, "preprocessing", budget, previous)
            if wanted is not None:
                parameters, trigger = wanted
                # G1 measures the trigger's delay on the record as preprocessed: it adds to the
                # correction already made.
                shift = parameters.get("trigger", {})
                if "t0" in shift and "trigger" in previous:
                    shift["t0"] = round(shift["t0"] + previous["trigger"].get("t0", 0.0), 4)
                again[name] = (parameters, trigger)
            elif spent(result, "preprocessing"):
                if attempt is not None:
                    record_result(
                        run_folder, name, "preprocessing", attempt.attempt, budget_spent(result)
                    )
                # Rejected, the record goes into no window.
                exclusions = exclusions.with_record(name)
        if not again and exclusions == before:
            break
        if again:
            by_name = {record.path.name: record for record in profile.records}
            numbers: dict[str, int] = {}
            for name in again:
                attempt = latest(attempts, name, "preprocessing")
                numbers[name] = (attempt.attempt if attempt is not None else 0) + 1
                invalidate_record(
                    record_folder(run_folder / RECORDS_FOLDER, by_name[name]), numbers[name] - 1
                )
            started_at = datetime.now(UTC)
            stopped: Stopped | None = None
            try:
                redone = preprocess_records(
                    preset,
                    profile,
                    run_folder,
                    workers,
                    presets={
                        name: resolve_preset(apply_overrides(preset, parameters), profile)
                        for name, (parameters, _) in again.items()
                    },
                    stop=stopping.current(),
                )
            except Stopped as error:
                # The records that finished logged below; the others given back their previous
                # stream (a redo's run keeps them; a new run is removed whole).
                stopped = error
                redone = cast(tuple[RecordOutcome, ...], error.kept or ())
                for name in set(again) - {outcome.name for outcome in redone}:
                    restore_record(
                        record_folder(run_folder / RECORDS_FOLDER, by_name[name]),
                        numbers[name] - 1,
                    )
            for outcome in redone:
                outcomes[outcome.name] = outcome
                parameters, trigger = again[outcome.name]
                _log(
                    run_folder,
                    outcome.name,
                    "preprocessing",
                    numbers[outcome.name],
                    parameters,
                    trigger,
                    started_at,
                    outcome,
                )
            if stopped is not None:
                raise stopped
    attempts = read_attempts(run_folder)
    for name in sorted({*exclusions.traces, *exclusions.records}):
        attempt = latest(attempts, name, "preprocessing")
        if attempt is None:
            continue
        note = (
            "left out of every window"
            if name in exclusions.records
            else f"traces {list(exclusions.traces[name])} left out of the windows"
        )
        record_notes(run_folder, name, "preprocessing", attempt.attempt, (note,))
    if active:
        exclusions = settle_receivers(run_folder, profile, outcomes, exclusions, config, reach_m)
    usable: Bands = {name: result.kept.band_hz for name, result in results.items()}
    return tuple(outcomes.values()), exclusions, usable


def settle_receivers(
    run_folder: Path,
    profile: Profile,
    outcomes: Mapping[str, RecordOutcome],
    exclusions: Exclusions,
    config: QCConfig,
    reach_m: float | None,
) -> Exclusions:
    """G1 over the line, once each record is settled: each receiver judged over every record
    that reaches it, those off the amplitude decay in most of them left out of every window.
    A trace off it in a few records stays (the one nearest each shot, where the fitted decay
    overshoots; a burst of noise): a record's own is no bad geophone. Logged as G1's result on
    the line."""
    started_at = datetime.now(UTC)
    off: dict[int, int] = {}
    reached: dict[int, int] = {}
    kept = [
        name
        for name, outcome in outcomes.items()
        if outcome.status == "succeeded" and name not in exclusions.records
    ]
    for name in kept:
        stream = stream_of(run_folder / outcomes[name].folder / PREPROCESSED)
        outliers, judged = decay_outliers(
            stream, config.signal, exclusions.traces.get(name, ()), reach_m
        )
        for receiver in np.flatnonzero(judged):
            reached[int(receiver)] = reached.get(int(receiver), 0) + 1
        for receiver in np.flatnonzero(outliers):
            off[int(receiver)] = off.get(int(receiver), 0) + 1
    result = judge_receivers(
        off, reached, [receiver.x for receiver in profile.receivers], config.signal
    )
    append_attempt(
        run_folder,
        Attempt(
            unit=LINE,
            stage="preprocessing",
            attempt=1,
            parameters={},
            triggered_by="initial",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="succeeded",
            results={"G1": result},
        ),
    )
    for flag in result.flags:
        if isinstance(flag.action, ExcludeTraces):
            for name in kept:
                exclusions = exclusions.with_traces(name, flag.action.traces)
    return exclusions


def settle_images(
    run_id: str,
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    settings: Settings,
    usable: Bands,
) -> None:
    """G2 on every window's latest image, and the phase shift done again, in groups sharing
    the same changes, for the windows G2 asks it of, until none is left or the budgets are
    spent."""
    n_units = max(1, len(manifest.windows))
    while True:
        attempts = read_attempts(run_folder)
        budget = RetryBudget(attempts, config.budgets, n_units)
        groups: dict[str, tuple[dict[str, Any], str, list[str]]] = {}
        for window in manifest.windows:
            attempt = latest(attempts, window.folder, "phase_shift")
            if attempt is None or attempt.status != "succeeded":
                continue
            result = attempt.results.get("G2")
            if result is None:
                folder = run_folder / window.folder
                result = judge_image(
                    window.folder, load_image(folder), config.image, shared_band(folder, usable)
                )
                attempt = record_result(
                    run_folder, window.folder, "phase_shift", attempt.attempt, result
                )
            wanted = next_try(result, "phase_shift", budget, attempt.parameters)
            if wanted is not None:
                parameters, trigger = wanted
                key = json.dumps(parameters, sort_keys=True) + trigger
                groups.setdefault(key, (parameters, trigger, []))[2].append(window.folder)
            elif spent(result, "phase_shift"):
                record_result(
                    run_folder, window.folder, "phase_shift", attempt.attempt, budget_spent(result)
                )
        if not groups:
            return
        for parameters, trigger, units in groups.values():
            rerun_phase_shift(run_id, units, parameters, settings, trigger)


def _log(
    run_folder: Path,
    unit: str,
    stage: Stage,
    number: int,
    parameters: dict[str, Any],
    trigger: str,
    started_at: datetime,
    outcome: RecordOutcome,
) -> None:
    append_attempt(
        run_folder,
        Attempt(
            unit=unit,
            stage=stage,
            attempt=number,
            parameters=parameters,
            triggered_by=trigger,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status=outcome.status,
            error=outcome.error,
        ),
    )
