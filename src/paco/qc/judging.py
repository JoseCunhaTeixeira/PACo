"""The gates on a whole run: G1 on its preprocessed records, G2 on its images, then the picking
as an attempt of its own (the curve saved in PAC's layout) and G3 on its curve, G4 over the
line at the end, each verdict recorded in the QC log, and the report written."""

import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.base import DispersionCurve, DispersionImage, Stream
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.selection_plotting import load_selection
from sigpipe.masw.picks import CURVES_FILE, save_pick
from sigpipe.masw.pipelines import (
    PREPROCESSED,
    trigger_shift_s,
    unmuted_record,
    window_correlations,
)
from sigpipe.masw.presets import (
    ActivePreset,
    PassiveActivePreset,
    PassivePreset,
    apply_overrides,
    resolve_preset,
)
from sigpipe.masw.profiles import Profile, ProfileError, load_profile
from sigpipe.masw.quality.line import Series
from sigpipe.masw.quality.measures import measure_signal, selection_measures
from sigpipe.masw.quality.spectra import save_record_spectra
from sigpipe.masw.runs import RunManifest, find_run, load_image, load_manifest, start_worker
from sigpipe.masw.runs.stopping import finished
from sigpipe.masw.windows import MASWWindow
from sigpipe.transformers import Load
from sigpipe.workers import one_thread_each

from paco import stopping
from paco.qc.attempts import invalidate
from paco.qc.coherence import nearest_offset
from paco.qc.config import QCConfig, snapshot_qc_config
from paco.qc.g1_signal import SPECTRA_MODES, judge_signal
from paco.qc.g2_image import judge_image, more_data
from paco.qc.g3_curve import CurveThresholds, judge_curve
from paco.qc.g4_profile import LINE, judge_profile
from paco.qc.log import (
    afresh,
    append_attempt,
    attempts_of,
    ensure_initial_attempts,
    latest,
    read_attempts,
    record_result,
    starts_afresh,
)
from paco.qc.loops import deep_merge
from paco.qc.models import Attempt, GateResult, Stage
from paco.qc.origin import run_work
from paco.qc.report import QCReport, build_report, write_report
from paco.qc.shots import (
    file_triggers,
    muted_records,
    preprocessing_values,
    trigger_context,
    window_muted,
)
from paco.settings import Settings

logger = logging.getLogger(__name__)


def judge_run(
    run_id: str,
    settings: Settings,
    config: QCConfig,
    picking: PickingParameters | None = None,
) -> QCReport:
    """G1, G2 and G3 on every record and window of run `run_id`, with the picking in between
    (`picking`, or the configuration's), then G4 over the line; the results go to the QC log,
    the configuration used next to them, and the report is written and returned."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    snapshot_qc_config(config, run_folder)
    ensure_initial_attempts(run_folder, manifest)

    # The records' triggers, from their files, and their inputs: none when the profile is gone.
    try:
        profile: Profile | None = load_profile(manifest.profile.name, settings)
    except ProfileError:
        profile = None
    triggers = file_triggers(profile) if profile is not None else {}
    usable = judge_records(run_folder, manifest, config, triggers, profile)
    judge_windows(run_folder, manifest, config, usable, picking or config.picking, profile)
    judge_line(run_folder, manifest, config)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def judge_records(
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    triggers: Mapping[str, float | None] | None = None,
    profile: Profile | None = None,
) -> dict[str, tuple[float, float] | None]:
    """G1 on each preprocessed record of the run, against its latest preprocessing attempt (its
    noise on the record before its muting, from its input in `profile`: before_muting), each
    record's shot where its muting and file's trigger (`triggers`) put it; its spectra drawn
    beside it. Returns each record's usable band."""
    active = manifest.profile.kind == "active"
    usable: dict[str, tuple[float, float] | None] = {}
    attempts = read_attempts(run_folder)
    for record in manifest.records:
        if record.status != "succeeded":
            continue
        attempt = latest(attempts, record.name, "preprocessing")
        folder = run_folder / record.folder
        values = preprocessing_values(manifest.preset, attempt)
        shot_s, applied_s = trigger_context(values, (triggers or {}).get(record.name))
        stream = stream_of(folder / PREPROCESSED)
        unmuted = before_muting(manifest.preset, profile, record.name, attempt, shot_s)
        result = judge_signal(
            record.name,
            stream,
            config.signal,
            active=active,
            shot_s=shot_s,
            applied_s=applied_s,
            spectra=manifest.preset.mode in SPECTRA_MODES,
            before_muting=unmuted,
            image_band=image_band(values),
        )
        _judge_latest(run_folder, record.name, "preprocessing", result)
        draw_spectra(stream, folder, result.kept.band_hz)
        usable[record.name] = result.kept.band_hz
    return usable


def before_muting(
    preset: ActivePreset | PassivePreset,
    profile: Profile | None,
    name: str,
    attempt: Attempt | None,
    shot_s: float = 0.0,
) -> tuple[Stream, float] | None:
    """Record `name` before its muting, and where its shot is on it (s; `shot_s` on its saved
    record): what G1 measures its noise on (a muting zeroes the noise window after the slowest
    arrival; a trigger's shift, part of the muting, drops the one before the trigger).
    Preprocessed as its latest `attempt` had it (the run's `preset` with its changes), from its
    input file in `profile` (sigpipe's unmuted_record: neither its trigger shifted nor muted, its
    shot later by the shift). None when it is not muted (its saved record is the same), or its
    input is not at hand."""
    muting = preprocessing_values(preset, attempt).get("muting") or {}
    records = profile.records if profile is not None else ()
    record = next((one for one in records if one.path.name == name), None)
    if (
        profile is None
        or record is None
        or muting.get("method", "none") == "none"
        or not record.path.exists()
    ):
        return None
    if attempt is not None and attempt.parameters:
        preset = resolve_preset(apply_overrides(preset, attempt.parameters), profile)
    return unmuted_record(preset, record, profile), shot_s + trigger_shift_s(preset, record)


def image_band(values: Mapping[str, Any]) -> tuple[float, float] | None:
    """The band the dispersion images use (a preset's values, or a stage's: its dispersion's
    fmin and fmax): what G1 and G2 measure a signal's SNR and coherence in; None when either
    is not set."""
    dispersion: Mapping[str, Any] = values.get("dispersion") or {}
    fmin, fmax = dispersion.get("fmin"), dispersion.get("fmax")
    if isinstance(fmin, int | float) and isinstance(fmax, int | float):
        return float(fmin), float(fmax)
    return None


def draw_spectra(stream: Stream, folder: Path, band: tuple[float, float] | None) -> None:
    """Preprocessed record `stream`'s spectra beside it in `folder`, G1's usable band (`band`)
    dashed (sigpipe's save_record_spectra). Best effort."""
    try:
        save_record_spectra(stream, folder, band)
    except Exception:
        logger.exception("Could not draw the spectra of %s", folder)


def judge_windows(
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    usable: dict[str, tuple[float, float] | None],
    picking: PickingParameters,
    profile: Profile | None = None,
) -> None:
    """G2 on each window's image, against its latest phase-shift attempt; then the picking, an
    attempt of its own, and G3 on its M0 curve. The band every record of the window keeps
    usable tells G2 what the data allow, and whether they were muted what to try first; a muted
    passive-active window's correlations are measured on its records before their muting, from
    their inputs in `profile`."""
    attempts = read_attempts(run_folder)
    records = RecordsBeforeMuting(manifest, profile, attempts)
    muted = muted_records(manifest.preset, attempts, latest)
    for window in manifest.windows:
        if window.status != "succeeded":
            continue
        folder = run_folder / window.folder
        image = load_image(folder)
        attempt = latest(attempts, window.folder, "phase_shift")
        muting = window_muted(folder, muted)
        g2 = judge_image(
            window.folder,
            image,
            config.image,
            shared_band(folder, usable),
            mode=manifest.preset.mode,
            muted=muting,
            **correlations_of(folder, manifest, config, muting, attempt, records),
        )
        _judge_latest(run_folder, window.folder, "phase_shift", g2)
        judge_picking(run_folder, window.folder, image, picking, config, g2.kept.band_hz)


def judge_picking(
    run_folder: Path,
    unit: str,
    image: DispersionImage,
    picking: PickingParameters,
    config: QCConfig,
    band: tuple[float, float] | None = None,
    triggered_by: str = "initial",
) -> GateResult:
    """S3 on one window, an attempt of its own: pick M0 with `picking`, save the curve in PAC's
    layout (the previous pick and its inversion go to attempts/), judge it with G3 against
    `band` (G2's coherent band), and log the attempt with the verdict."""
    started_at = datetime.now(UTC)
    previous = len(attempts_of(read_attempts(run_folder), unit, "picking"))
    g3 = _pick(
        run_folder,
        unit,
        picking,
        config.curve,
        band,
        previous,
        image,
    )
    _log_pick(run_folder, unit, previous, picking, triggered_by, started_at, g3)
    return g3


def pick_windows(
    run_folder: Path,
    jobs: Mapping[str, tuple[PickingParameters, tuple[float, float] | None]],
    config: QCConfig,
    triggered_by: str,
    workers: int,
) -> dict[str, GateResult]:
    """S3 on the windows of `jobs`, each with its picking parameters and G2's band, each an
    attempt of its own, in up to `workers` processes (a single window here): the picks and G3
    run in the workers, each on its own window's folder, and the attempts are logged here as
    they end. G3's results by window."""
    attempts = read_attempts(run_folder)
    previous = {unit: len(attempts_of(attempts, unit, "picking")) for unit in jobs}
    started_at = datetime.now(UTC)
    results: dict[str, GateResult] = {}

    def logged(unit: str, g3: GateResult) -> None:
        picking = jobs[unit][0]
        _log_pick(run_folder, unit, previous[unit], picking, triggered_by, started_at, g3)
        results[unit] = g3

    # Stopped between windows: a pick archives the window's previous one before it saves its
    # own, so the running picks end (seconds) and are logged; the others never start.
    if workers <= 1 or len(jobs) <= 1:
        for unit, (picking, band) in jobs.items():
            stopping.check()
            logged(
                unit,
                _pick(run_folder, unit, picking, config.curve, band, previous[unit]),
            )
        return results
    one_thread_each()  # the workers are the cores the picks take
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), initializer=start_worker, initargs=(run_folder,)
    ) as executor:
        futures = {
            executor.submit(
                _pick, run_folder, unit, picking, config.curve, band, previous[unit]
            ): unit
            for unit, (picking, band) in jobs.items()
        }
        for future in finished(executor, futures, stopping.current(), kill=False):
            logged(futures[future], future.result())
    return results


def _pick(
    run_folder: Path,
    unit: str,
    picking: PickingParameters,
    thresholds: CurveThresholds,
    band: tuple[float, float] | None,
    previous: int,
    image: DispersionImage | None = None,
) -> GateResult:
    """One window's pick, in a worker or here: the `previous` attempts' files archived, M0
    picked and saved in PAC's layout, G3 on it. A mute is the line's, never one window's: a flag
    only a mute fixes rejects the pick."""
    folder = run_folder / unit
    if previous:
        invalidate(folder, "picking", previous)
    image = image if image is not None else load_image(folder)
    modes = pick_modes(image, picking)
    m0 = modes[0] if modes else None
    if m0 is not None and m0.curve is not None:
        save_pick(folder, image, m0.curve)
    return judge_curve(
        unit, image, m0, thresholds, band, picking, nearest_offset(folder), mutable=False
    )


def _log_pick(
    run_folder: Path,
    unit: str,
    previous: int,
    picking: PickingParameters,
    triggered_by: str,
    started_at: datetime,
    g3: GateResult,
) -> None:
    """The window's pick logged, after its `previous` attempts: going on from them for a retry a
    gate asked for; else afresh, the window's earlier picks and what was made of them (their
    inversions, soil columns) forgotten."""
    attempt = Attempt(
        unit=unit,
        stage="picking",
        attempt=previous + 1,
        parameters=picking.model_dump(),
        triggered_by=triggered_by,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        status="succeeded",
        results={g3.gate: g3},
    )
    append_attempt(
        run_folder, afresh(run_folder, attempt) if starts_afresh(triggered_by) else attempt
    )


def judge_line(run_folder: Path, manifest: RunManifest, config: QCConfig) -> tuple[GateResult, ...]:
    """G4 over the line: the saved M0 curve of every window whose latest pick passed G3, each
    against its neighbours, a person's curves among them as trusted references (never judged).
    A window's verdict goes to its latest picking attempt, the line's own (the coverage) to an
    attempt of the unit "line". No automatic curve on the line (a run picked by hand): no G4."""
    attempts = read_attempts(run_folder)
    work = run_work(run_folder, manifest, attempts)
    curves: list[Series] = []
    trusted: set[str] = set()
    without: list[float] = []
    for window in manifest.windows:
        state = work[window.folder].m0
        path = run_folder / window.folder / CURVES_FILE
        picked = latest(attempts, window.folder, "picking")
        g3 = picked.results.get("G3") if picked else None
        curve = (
            saved_m0(path)
            if state == "user" or (state == "judged" and g3 is not None and g3.verdict == "pass")
            else None
        )
        if curve is None:
            without.append(window.xmid)
            continue
        curves.append(Series.from_curve(window.folder, window.xmid, curve))
        if state == "user":
            trusted.add(window.folder)
    if not any(one.m0 in ("judged", "unjudged") for one in work.values()):
        return ()
    started_at = datetime.now(UTC)
    results = judge_profile(curves, config.profile, without, frozenset(trusted))
    for result in results:
        if result.unit != LINE:
            _judge_latest(run_folder, result.unit, "picking", result)
            continue
        append_attempt(
            run_folder,
            Attempt(
                unit=LINE,
                stage="picking",
                attempt=len(attempts_of(attempts, LINE, "picking")) + 1,
                parameters={},
                triggered_by="initial",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status="succeeded",
                results={result.gate: result},
            ),
        )
    return results


def _judge_latest(run_folder: Path, unit: str, stage: Stage, result: GateResult) -> None:
    attempt = latest(read_attempts(run_folder), unit, stage)
    if attempt is not None:
        record_result(run_folder, unit, stage, attempt.attempt, result)


def shared_band(
    window_folder: Path, usable: dict[str, tuple[float, float] | None]
) -> tuple[float, float] | None:
    """The band every record of the window keeps usable: the highest low edge, the lowest high
    edge; None when no record has one."""
    window = MASWWindow.model_validate_json((window_folder / "window.json").read_text())
    known = [band for path in window.selected_files if (band := usable.get(path.name))]
    if not known:
        return None
    low, high = max(band[0] for band in known), min(band[1] for band in known)
    return (low, high) if low < high else None


def saved_m0(path: Path) -> DispersionCurve | None:
    """The M0 curve saved in `path`, if any."""
    if not path.exists():
        return None
    curves = load_dispersion_curves([path])[0]
    return next((curve for curve in curves if curve.mode.number == 0), None)


class RecordsBeforeMuting:
    """A run's records before their muting, each preprocessed once, as its latest attempt had
    it, from its input in `profile` (before_muting): what a muted passive-active window's
    correlations are measured on (a muting zeroes a record's noise, and its correlations'
    after the slowest arrival). None for a record whose input is not at hand."""

    def __init__(
        self, manifest: RunManifest, profile: Profile | None, attempts: Sequence[Attempt]
    ) -> None:
        self._manifest = manifest
        self._profile = profile
        self._attempts = attempts
        self._streams: dict[str, Stream | None] = {}

    def stream(self, name: str) -> Stream | None:
        if name not in self._streams:
            attempt = latest(self._attempts, name, "preprocessing")
            found = before_muting(self._manifest.preset, self._profile, name, attempt)
            self._streams[name] = found[0] if found is not None else None
        return self._streams[name]


# The modes whose windows' images are made of stacked correlations, a virtual shot's.
CORRELATION_MODES = frozenset({"passive", "passive-active"})


def correlations_of(
    folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    muted: bool,
    attempt: Attempt | None,
    records: RecordsBeforeMuting | None = None,
) -> dict[str, Any]:
    """G2's arguments on a passive or passive-active window (none for an active one): its
    stacked correlations, saved beside its image, measured as a record is (sigpipe's
    measure_signal from the virtual source; `muted`, its records muted before correlating: its
    SNR and band measured on the correlations of its `records` before their muting, else not),
    its fk selection's measures (a passive window's, as its job saved it), and the phase shift
    again with more of the data (more_data, from the values it last ran with: the run's preset,
    its latest `attempt`'s changes on top)."""
    mode = str(manifest.preset.mode)
    if mode not in CORRELATION_MODES:
        return {}
    values = deep_merge(
        manifest.preset.model_dump(mode="json"), attempt.parameters if attempt is not None else {}
    )
    path = folder / PREPROCESSED
    unmuted = (
        _correlations_before_muting(folder, manifest, attempt, records)
        if muted and records is not None
        else None
    )
    correlations = (
        measure_signal(
            stream_of(path),
            config.signal,
            source="virtual",
            spectra=True,
            records_muted=muted,
            before_muting=None if unmuted is None else (unmuted, 0.0),
            image_band=image_band(values),
        )
        if path.exists()
        else None
    )
    selection = load_selection(folder)
    return {
        "correlations": correlations,
        "selection": selection_measures(selection, config.signal) if selection else (),
        "more": more_data(mode, values),
    }


def _correlations_before_muting(
    folder: Path,
    manifest: RunManifest,
    attempt: Attempt | None,
    records: RecordsBeforeMuting,
) -> Stream | None:
    """A passive-active window's stacked correlations made again from its records before their
    muting, as its latest `attempt` stacked them; None without them all."""
    preset = manifest.preset
    if not isinstance(preset, PassiveActivePreset):
        return None
    window = MASWWindow.model_validate_json((folder / "window.json").read_text())
    streams: dict[str, Stream] = {}
    for path in window.selected_files:
        found = records.stream(path.name)
        if found is None:
            return None
        streams[path.name] = found
    if attempt is not None and attempt.parameters:
        preset = apply_overrides(preset, attempt.parameters)
    try:
        return window_correlations(preset, window, streams)
    except Exception:
        logger.exception("Could not correlate the records of %s before their muting", folder)
        return None


def stream_of(path: Path) -> Stream:
    (stream,) = Load(file_paths=[path], data_type="stream").transform([])
    if not isinstance(stream, Stream):
        raise TypeError(f"{path} did not load as a stream")
    return stream
