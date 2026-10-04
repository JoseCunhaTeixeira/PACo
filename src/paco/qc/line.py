"""A line from its records to G2, the way the QC workflow runs it (docs/qc_workflow.md): one set
of settings for the whole line, every record and every window alike, as a person sets them in
PAC's pages. S1 on every record, G1 judging each (the traces and records it excludes left out of
the windows), G1 over the line (the receivers off the amplitude decay in most records left out
of every window), the coherence rules for S2 (the band capped by G1's usable band, the window
length from the ladder, the shots stacked between the near field and the line's reach), the mute
trial, S2 on the whole line, G2 on every window, and the line loop (qc.line_loop): the changes
of the records or the images the gates ask, tried for the line and kept when more windows get a
curve G3 passes, the whole line made again with each. What run_processing runs; the picking and
G3, G4 are pick's, each window's own."""

import shutil
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from sigpipe.base import Stream
from sigpipe.masw.pipelines import PREPROCESSED, record_folder
from sigpipe.masw.presets import (
    ActivePreset,
    PassivePreset,
    apply_overrides,
    make_preset,
    resolve_preset,
)
from sigpipe.masw.profiles import Profile, ProfileError, load_profile
from sigpipe.masw.quality.measures import decay_outliers, spectral_outliers
from sigpipe.masw.quality.measures import line_reach as sigpipe_line_reach
from sigpipe.masw.runs import RecordOutcome, RunError, RunManifest, load_image, load_manifest
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
    receiver_spacing,
)
from paco.qc.config import QCConfig, snapshot_qc_config
from paco.qc.g1_signal import (
    SPECTRA_MODES,
    judge_fast_arrivals,
    judge_receivers,
    judge_signal,
    judge_spectra,
)
from paco.qc.g2_image import judge_image
from paco.qc.g4_profile import LINE
from paco.qc.given import give, locked, said
from paco.qc.judging import (
    RecordsBeforeMuting,
    before_muting,
    correlations_of,
    draw_spectra,
    image_band,
    shared_band,
    stream_of,
)
from paco.qc.line_loop import (
    Candidate,
    LineTrial,
    candidates,
    kept_on_line,
    line_change,
    line_picks,
    try_candidate,
    write_line_loop,
)
from paco.qc.log import (
    LINE_CHANGE,
    MUTE_TRIAL,
    append_attempt,
    latest,
    read_attempts,
    record_notes,
    record_result,
)
from paco.qc.loops import deep_merge, held_changes
from paco.qc.models import Attempt, ExcludeRecord, ExcludeTraces, GateResult, Stage
from paco.qc.muting import (
    changed_muting,
    choose_mute,
    estimate_cone,
    given_muting,
    mutable,
    mute_candidates,
)
from paco.qc.origin import run_work
from paco.qc.report import QCReport, build_report, write_report
from paco.qc.rerun import line_windows, rerun_phase_shift
from paco.qc.segments import SEGMENT_STAGES, choose_segments
from paco.qc.shots import (
    file_triggers,
    muted_records,
    preprocessing_values,
    pulse_widths,
    trigger_context,
    window_muted,
)
from paco.qc.stuck import Stuck
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
        raise Stuck(
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
    # The user's settings, locked before any gate judges (qc.given).
    give(run_folder, "processing", overrides)
    started_at = datetime.now(UTC)
    # The surface waves' cone from the gathers (qc.muting), for the mute trial and G1's fast
    # arrivals; none over a muting the user gave, nor on a passive line.
    cone = (
        estimate_cone(loaded, preset, config.mute)
        if mutable(preset) and not given_muting(overrides)
        else None
    )
    fast = mute_candidates(cone, config.mute, loaded)["cone"] if cone is not None else None
    records = preprocess_records(
        preset, loaded, run_folder, settings.workers, stop=stopping.current()
    )
    for record in records:
        _log(run_folder, record.name, "preprocessing", 1, {}, "initial", started_at, record)
    reach = line_reach(run_folder, loaded, records, config, preset)
    records, exclusions, usable = settle_records(
        run_folder, loaded, preset, records, config, reach, fast
    )
    # The band the records G1 kept share: a rejected record constrains nothing.
    kept = [band for name, band in usable.items() if name not in exclusions.records]
    dispersion = (overrides or {}).get("dispersion")
    band, band_notes = cap_band(
        preset, kept, loaded.nyquist_hz, dispersion if isinstance(dispersion, Mapping) else None
    )
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
    # The mute trial at the line's length: a mute kept changes the images, never the windows;
    # the records then preprocessed again with it, and judged again by G1.
    mute_notes: tuple[str, ...] = ()
    if cone is not None:
        trial = choose_mute(
            loaded,
            preset,
            run_folder,
            TrialJudge(config.coherence, config.curve, config.picking, config.image),
            config.mute,
            settings.workers,
            cone,
            exclusions=exclusions,
        )
        muted = changed_muting(trial)
        if muted is not None:
            # A change of the line's: the note says it (no mute keeps the line as it is, and
            # run_processing says the trial's candidates apart).
            mute_notes = trial.notes if trial is not None else ()
            changes = deep_merge(changes, muted)
            preset = resolve_preset(apply_overrides(preset, muted), loaded)
            records = reprocess_records(
                run_folder, loaded, preset, records, exclusions, settings.workers, MUTE_TRIAL
            )
            records, exclusions, usable = settle_records(
                run_folder, loaded, preset, records, config, reach
            )
    # A passive line's segments, their length and FK selection tried on a few windows, those the
    # user set left as they are: PACo optimizes the parameters the passive workflow has.
    segment_notes: tuple[str, ...] = ()
    given = frozenset(overrides or {}) & SEGMENT_STAGES
    if str(mode) == "passive" and given != SEGMENT_STAGES:
        picked, segment_notes = choose_segments(
            loaded, preset, run_folder, config.segments, config.signal, config.picking,
            exclusions, given,
        )  # fmt: skip
        if picked is not None:
            changes = deep_merge(changes, picked)
            preset = resolve_preset(apply_overrides(preset, picked), loaded)
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
    notes = band_notes + far_notes + choice.notes + mute_notes + segment_notes + near_notes
    _line_settings(run_folder, logged, notes, started_at)
    judge_images(run_folder, manifest, config, settings, usable)
    settled = settle_line(
        run_id, run_folder, loaded, preset, records, exclusions, usable, config, settings, reach
    )
    if settled.notes:
        # The line's settings with the changes the loop kept, and what it tried.
        _line_settings(
            run_folder, deep_merge(logged, settled.changes), notes + settled.notes, started_at
        )
    keep_line_asks(run_folder)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def reprocess_records(
    run_folder: Path,
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    records: tuple[RecordOutcome, ...],
    exclusions: Exclusions,
    workers: int,
    trigger: str,
) -> tuple[RecordOutcome, ...]:
    """The line's records preprocessed again with the line's settings (`preset`), the same for
    every record, their previous results archived; logged as `trigger`'s attempts (the mute
    trial's, the line loop's, a redo's), each with no setting of its own. Those G1 left out stay
    as they were. Stopped, the records that finished are logged and the others given back their
    previous stream."""
    attempts = read_attempts(run_folder)
    by_name = {record.path.name: record for record in profile.records}
    outcomes = {record.name: record for record in records}
    numbers: dict[str, int] = {}
    for name, outcome in outcomes.items():
        if outcome.status != "succeeded" or name in exclusions.records:
            continue
        attempt = latest(attempts, name, "preprocessing")
        number = attempt.attempt if attempt is not None else 0
        invalidate_record(record_folder(run_folder / RECORDS_FOLDER, by_name[name]), number)
        numbers[name] = number + 1
    started_at = datetime.now(UTC)
    stopped: Stopped | None = None
    try:
        redone = preprocess_records(
            preset,
            profile,
            run_folder,
            workers,
            presets=dict.fromkeys(numbers, preset),
            stop=stopping.current(),
        )
    except Stopped as error:
        stopped = error
        redone = cast(tuple[RecordOutcome, ...], error.kept or ())
        for name in set(numbers) - {outcome.name for outcome in redone}:
            restore_record(
                record_folder(run_folder / RECORDS_FOLDER, by_name[name]), numbers[name] - 1
            )
    for outcome in redone:
        outcomes[outcome.name] = outcome
        _log(
            run_folder,
            outcome.name,
            "preprocessing",
            numbers[outcome.name],
            {},
            trigger,
            started_at,
            outcome,
        )
    if stopped is not None:
        raise stopped
    return tuple(outcomes.values())


def line_reach(
    run_folder: Path,
    profile: Profile,
    records: tuple[RecordOutcome, ...],
    config: QCConfig,
    preset: ActivePreset | PassivePreset,
) -> float | None:
    """How far from the shots the line's traces still carry the wave (G1's per-trace SNR over
    every active record, its noise before its muting as G1 measures it, `snr_reach`, in bins of
    a fifteenth of the line, each record's times from its shot where its latest preprocessing,
    over the run's `preset`, and its file's trigger put it); None on a passive line, or when no
    distance falls below G1's SNR limit."""
    if profile.kind != "active":
        return None
    triggers = file_triggers(profile)
    attempts = read_attempts(run_folder)

    def unmuted(record: RecordOutcome) -> tuple[Stream, float]:
        # Its noise as G1 measures it, before its muting; its times from its shot.
        attempt = latest(attempts, record.name, "preprocessing")
        values = preprocessing_values(preset, attempt)
        shot_s = trigger_context(values, triggers.get(record.name))[0]
        return before_muting(preset, profile, record.name, attempt, shot_s) or (
            stream_of(run_folder / record.folder / PREPROCESSED),
            shot_s,
        )

    positions = [receiver.x for receiver in profile.receivers]
    return sigpipe_line_reach(
        (unmuted(record) for record in records if record.status == "succeeded"),
        config.signal,
        max(positions) - min(positions),
    )


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
    reach_m: float | None = None,
    cone: Mapping[str, Any] | None = None,
) -> tuple[tuple[RecordOutcome, ...], Exclusions, Bands]:
    """G1 on every preprocessed record, the line's settings the same for each: the traces and
    records it excludes recorded (for S2 to leave out, and for G1 to judge the record again
    without them) until it excludes nothing more. A change of a record's preprocessing it asks
    (its trigger, a mute) is the line's to make (the line loop), never made for one record. With
    `cone` (the mute trial's, on a line it left unmuted), a record not muted is checked for
    arrivals faster than it. Returns the records, the exclusions and each record's usable
    band."""
    outcomes = {record.name: record for record in records}
    exclusions = Exclusions()
    results: dict[str, GateResult] = {}
    active = profile.kind == "active"
    # The traces' spectra against their neighbours': on every line (SPECTRA_MODES).
    spectra = preset.mode in SPECTRA_MODES
    triggers = file_triggers(profile)
    # The traces the cone leaves out, near the shot: the fast arrivals' too.
    nearest_m = config.mute.min_offset_spacings * receiver_spacing(profile)
    # Each record's spectra as drawn: its attempt, and the band G1 found.
    drawn: dict[str, tuple[object, ...]] = {}
    while True:
        attempts = read_attempts(run_folder)
        results = {}
        for name, outcome in outcomes.items():
            if outcome.status != "succeeded" or name in exclusions.records:
                continue
            latest_attempt = latest(attempts, name, "preprocessing")
            folder = run_folder / outcome.folder
            stream = stream_of(folder / PREPROCESSED)
            # Its shot where its muting and its file's trigger put it; its noise measured
            # before its muting.
            values = preprocessing_values(preset, latest_attempt)
            shot_s, applied_s = trigger_context(values, triggers.get(name))
            unmuted = before_muting(preset, profile, name, latest_attempt, shot_s)
            result = judge_signal(
                name,
                stream,
                config.signal,
                active=active,
                excluded=exclusions.traces.get(name, ()),
                reach_m=reach_m,
                shot_s=shot_s,
                applied_s=applied_s,
                spectra=spectra,
                before_muting=unmuted,
                image_band=image_band(values),
            )
            if cone is not None and values.get("muting", {}).get("method") in (None, "none"):
                gather, shot = unmuted or (stream, shot_s)
                result = judge_fast_arrivals(result, gather, shot, cone, config.signal, nearest_m)
            results[name] = result
            if latest_attempt is not None:
                record_result(run_folder, name, "preprocessing", latest_attempt.attempt, result)
            # Its spectra beside it, drawn again when the record or its usable band changed.
            shown = (
                latest_attempt.attempt if latest_attempt is not None else 0,
                result.kept.band_hz,
            )
            if drawn.get(name) != shown:
                draw_spectra(stream, folder, result.kept.band_hz)
                drawn[name] = shown
        before = exclusions
        for name, result in results.items():
            for flag in result.flags:
                if isinstance(flag.action, ExcludeTraces):
                    exclusions = exclusions.with_traces(name, flag.action.traces)
                elif isinstance(flag.action, ExcludeRecord):
                    exclusions = exclusions.with_record(name)
        if exclusions == before:
            break
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
        exclusions = settle_receivers(
            run_folder, profile, outcomes, exclusions, config, reach_m, spectra
        )
    elif spectra:
        _log_line(run_folder, _spectra_result(run_folder, profile, outcomes, exclusions, config))
    usable: Bands = {name: result.kept.band_hz for name, result in results.items()}
    return tuple(outcomes.values()), exclusions, usable


def settle_receivers(
    run_folder: Path,
    profile: Profile,
    outcomes: Mapping[str, RecordOutcome],
    exclusions: Exclusions,
    config: QCConfig,
    reach_m: float | None,
    spectra: bool = False,
) -> Exclusions:
    """G1 over the line, once each record is settled: each receiver judged over every record
    that reaches it, those off the amplitude decay in most of them left out of every window.
    A trace off it in a few records stays (the one nearest each shot, where the fitted decay
    overshoots; a burst of noise): a record's own is no bad geophone. With `spectra` (every
    line), the receivers off their neighbours' spectra flagged too. Logged as G1's result on the
    line."""
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
    if spectra:
        own = _spectra_result(run_folder, profile, outcomes, exclusions, config)
        result = result.model_copy(
            update={
                "metrics": result.metrics + own.metrics,
                "flags": result.flags + own.flags,
            }
        )
    _log_line(run_folder, result, started_at)
    for flag in result.flags:
        if isinstance(flag.action, ExcludeTraces):
            for name in kept:
                exclusions = exclusions.with_traces(name, flag.action.traces)
    return exclusions


def _spectra_result(
    run_folder: Path,
    profile: Profile,
    outcomes: Mapping[str, RecordOutcome],
    exclusions: Exclusions,
    config: QCConfig,
) -> GateResult:
    """G1 over a line: the receivers off their neighbours' spectra in most of the records judging
    them, flagged (judge_spectra)."""
    shots = profile.kind == "active"
    off: dict[int, int] = {}
    reached: dict[int, int] = {}
    for name, outcome in outcomes.items():
        if outcome.status != "succeeded" or name in exclusions.records:
            continue
        stream = stream_of(run_folder / outcome.folder / PREPROCESSED)
        bad, judged = spectral_outliers(
            stream, config.signal, exclusions.traces.get(name, ()), shots=shots
        )
        for receiver in np.flatnonzero(judged):
            reached[int(receiver)] = reached.get(int(receiver), 0) + 1
        for receiver in np.flatnonzero(bad):
            off[int(receiver)] = off.get(int(receiver), 0) + 1
    metric, flag = judge_spectra(
        off, reached, [receiver.x for receiver in profile.receivers], config.signal
    )
    return GateResult(
        gate="G1",
        unit=LINE,
        verdict="pass",
        metrics=(metric,),
        flags=(flag,) if flag is not None else (),
    )


def _log_line(run_folder: Path, result: GateResult, started_at: datetime | None = None) -> None:
    """G1's result on the line, logged as the line's preprocessing."""
    append_attempt(
        run_folder,
        Attempt(
            unit=LINE,
            stage="preprocessing",
            attempt=1,
            parameters={},
            triggered_by="initial",
            started_at=started_at or datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="succeeded",
            results={"G1": result},
        ),
    )


def judge_images(
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    settings: Settings,
    usable: Bands,
) -> None:
    """G2 on every window's latest image it has not judged yet, a window holding a person's work
    (paco.qc.origin) left as it is. A change of the images or the records G2 asks is the line's to
    make (the line loop), never one window's."""
    # A muted passive-active line's correlations measured on its records before their muting,
    # from their inputs.
    try:
        profile: Profile | None = load_profile(manifest.profile.name, settings)
    except ProfileError:
        profile = None
    attempts = read_attempts(run_folder)
    records = RecordsBeforeMuting(manifest, profile, attempts)
    work = run_work(run_folder, manifest, attempts)
    muted = muted_records(manifest.preset, attempts, latest)
    for window in manifest.windows:
        attempt = latest(attempts, window.folder, "phase_shift")
        if attempt is None or attempt.status != "succeeded" or work[window.folder].frozen:
            continue
        if attempt.results.get("G2") is not None:
            continue
        folder = run_folder / window.folder
        on_muted = window_muted(folder, muted)
        result = judge_image(
            window.folder,
            load_image(folder),
            config.image,
            shared_band(folder, usable),
            mode=manifest.preset.mode,
            muted=on_muted,
            **correlations_of(folder, manifest, config, on_muted, attempt, records),
        )
        record_result(run_folder, window.folder, "phase_shift", attempt.attempt, result)


@dataclass(frozen=True)
class SettledLine:
    """The line after its loop: its settings, records, exclusions and usable bands, the changes
    it kept (over the settings before it), and the changes it tried, in words."""

    preset: ActivePreset | PassivePreset
    records: tuple[RecordOutcome, ...]
    exclusions: Exclusions
    usable: Bands
    changes: dict[str, Any]
    notes: tuple[str, ...]


def settle_line(
    run_id: str,
    run_folder: Path,
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    records: tuple[RecordOutcome, ...],
    exclusions: Exclusions,
    usable: Bands,
    config: QCConfig,
    settings: Settings,
    reach_m: float | None,
) -> SettledLine:
    """The line loop (qc.line_loop) on run `run_id`, its windows imaged and judged by G2: each
    window's image picked on trial; the changes of the line's settings the gates ask tried, the
    most failing windows' first, until one is kept; the one kept made on the whole line (its
    records preprocessed again with it and judged by G1, for a change of the records; every
    window imaged again and judged by G2), the asks then read again; up to `max_changes` kept.
    The changes tried go to line_loop.json."""
    rules = config.line
    judge = TrialJudge(config.coherence, config.curve, config.picking, config.image)
    given = {**locked(run_folder, "preprocessing"), **locked(run_folder, "phase_shift")}
    # Whether a mute of the line is there to try: a passive line has no muting.
    can_mute = "muting" in type(preset).model_fields
    tried: set[str] = set()
    trials: list[LineTrial] = []
    changes: dict[str, Any] = {}
    while sum(trial.kept for trial in trials) < rules.max_changes:
        stopping.check()
        manifest = load_manifest(run_id, settings)
        attempts = read_attempts(run_folder)
        work = run_work(run_folder, manifest, attempts)
        built = line_windows(run_folder, profile, preset)
        windows = {
            window.folder: built[window.folder]
            for window in manifest.windows
            if not work[window.folder].frozen and window.folder in built
        }
        imaged = [unit for unit in windows if _imaged(attempts, unit)]
        picks = line_picks(run_folder, imaged, judge, can_mute, settings.workers)
        kept = [record.name for record in records if record.name not in exclusions.records]
        found = candidates(
            _latest_results(attempts, kept, "preprocessing", "G1"),
            _latest_results(attempts, list(windows), "phase_shift", "G2"),
            picks,
            {
                unit: frozenset(path.name for path in window.selected_files)
                for unit, window in windows.items()
            },
            preset.model_dump(mode="json"),
            given,
            frozenset(tried),
        )
        adopted: Candidate | None = None
        for candidate in found:
            tried.add(candidate.key)
            asked = _with_line_pulse(candidate, attempts, kept, config)
            trial = try_candidate(
                asked,
                preset,
                profile,
                run_folder,
                windows,
                picks,
                records,
                exclusions,
                judge,
                rules,
                settings.workers,
            )
            trials.append(trial)
            if trial.kept:
                adopted = asked
                break
        if trials:
            write_line_loop(run_folder, trials)
        if adopted is None:
            break
        changes = deep_merge(changes, adopted.overrides)
        preset = resolve_preset(apply_overrides(preset, adopted.overrides), profile)
        if adopted.stage == "preprocessing":
            records = reprocess_records(
                run_folder, profile, preset, records, exclusions, settings.workers, LINE_CHANGE
            )
            records, exclusions, usable = settle_records(
                run_folder, profile, preset, records, config, reach_m
            )
        image_line(run_id, run_folder, profile, preset, records, exclusions, settings, LINE_CHANGE)
        judge_images(run_folder, load_manifest(run_id, settings), config, settings, usable)
    return SettledLine(
        preset, records, exclusions, usable, changes, tuple(trial.note for trial in trials)
    )


def image_line(
    run_id: str,
    run_folder: Path,
    profile: Profile,
    preset: ActivePreset | PassivePreset,
    records: tuple[RecordOutcome, ...],
    exclusions: Exclusions,
    settings: Settings,
    trigger: str,
    replace_hand: bool = False,
) -> list[str]:
    """Every window of run `run_id` imaged again with the line's settings (`preset`), the same
    for each, from the records as they are: the manifest written with them first, the phase
    shift's attempts logged as `trigger`'s. A window holding a person's work keeps it, unless
    they chose to have it done again (`replace_hand`: their image archived). Returns the windows
    imaged."""
    manifest = load_manifest(run_id, settings)
    write_manifest(
        run_id,
        run_folder,
        profile,
        preset,
        manifest.started_at,
        records,
        manifest.windows,
        exclusions,
        packages=PACKAGES,
        inputs=manifest.inputs,
    )
    work = run_work(run_folder, manifest)
    units = [
        window.folder
        for window in manifest.windows
        if replace_hand or not work[window.folder].frozen
    ]
    if units:
        rerun_phase_shift(run_id, units, {}, settings, trigger)
    return units


def keep_line_asks(run_folder: Path) -> None:
    """Once the line is settled, the records' G1 and the windows' G2 still asking a change of the
    line's settings: their asks kept as notes (kept_on_line), the line's settings staying the
    same for every record and window; an ask of a setting the user gave rejects its unit, the
    change it asks said (L6: the gate fails and suggests)."""
    attempts = read_attempts(run_folder)
    given = {**locked(run_folder, "preprocessing"), **locked(run_folder, "phase_shift")}
    for stage, gate in (("preprocessing", "G1"), ("phase_shift", "G2")):
        units = {attempt.unit for attempt in attempts if attempt.stage == stage}
        for unit in sorted(units - {LINE}):
            attempt = latest(attempts, unit, cast(Stage, stage))
            result = attempt.results.get(gate) if attempt is not None else None
            if attempt is None or result is None or result.verdict != "retry":
                continue
            held = deep_merge(
                held_changes(result, "preprocessing", given),
                held_changes(result, "phase_shift", given),
            )
            if held:
                settled = budget_spent(result, "locked", said(held, given))
            elif line_change(result):
                settled = kept_on_line(result)
            else:
                continue
            record_result(run_folder, unit, cast(Stage, stage), attempt.attempt, settled)


def _imaged(attempts: tuple[Attempt, ...], unit: str) -> bool:
    attempt = latest(attempts, unit, "phase_shift")
    return attempt is not None and attempt.status == "succeeded"


def _latest_results(
    attempts: tuple[Attempt, ...], units: list[str], stage: Stage, gate: str
) -> dict[str, GateResult]:
    """`gate`'s result on each of `units` at its latest attempt of `stage`, where it has one."""
    found: dict[str, GateResult] = {}
    for unit in units:
        attempt = latest(attempts, unit, stage)
        if attempt is not None and (result := attempt.results.get(gate)) is not None:
            found[unit] = result
    return found


def _with_line_pulse(
    candidate: Candidate, attempts: tuple[Attempt, ...], names: list[str], config: QCConfig
) -> Candidate:
    """`candidate`, a mute without its width given, with the line's pulse as its width (the
    records' median pulse, as G1 measured them): one width for every record."""
    muting = candidate.overrides.get("muting")
    if not isinstance(muting, Mapping):
        return candidate
    values = cast(Mapping[str, Any], muting)
    if values.get("method") != "mute" or "width" in values:
        return candidate
    widths = pulse_widths(attempts, names, config.signal.mute_width_s)
    pulse = round(statistics.median(widths.values()), 4) if widths else config.signal.mute_width_s
    overrides = {**candidate.overrides, "muting": {**values, "width": pulse}}
    return replace(candidate, overrides=overrides)


def _line_settings(
    run_folder: Path, parameters: dict[str, Any], notes: tuple[str, ...], started_at: datetime
) -> None:
    """The line's settings the processing chose, and why (`notes`), as the line's phase-shift
    attempt; logged again when the line loop changes them: an attempt's last line is its
    current one."""
    append_attempt(
        run_folder,
        Attempt(
            unit=LINE,
            stage="phase_shift",
            attempt=1,
            parameters=parameters,
            triggered_by="initial",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="succeeded",
            notes=notes,
        ),
    )


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
