"""The gates on a whole run: G1 on its preprocessed records, G2 on its images, then the picking
as an attempt of its own (the curve saved in PAC's layout) and G3 on its curve, G4 over the
line at the end, each verdict recorded in the QC log, and the report written. What milestone
14's tools call."""

from datetime import UTC, datetime
from pathlib import Path

from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.base import DispersionCurve, DispersionImage, Stream
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.masw.picks import CURVES_FILE, save_pick
from sigpipe.masw.pipelines import PREPROCESSED
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import RunManifest, find_run, load_image, load_manifest
from sigpipe.masw.windows import MASWWindow
from sigpipe.transformers import Load

from paco.qc.attempts import invalidate
from paco.qc.coherence import nearest_offset
from paco.qc.config import QCConfig, snapshot_qc_config
from paco.qc.g1_signal import judge_signal
from paco.qc.g2_image import judge_image
from paco.qc.g3_curve import judge_curve
from paco.qc.g4_profile import LINE, judge_profile
from paco.qc.log import (
    append_attempt,
    attempts_of,
    ensure_initial_attempts,
    latest,
    read_attempts,
    record_result,
)
from paco.qc.models import Attempt, GateResult, Stage
from paco.qc.report import QCReport, build_report, write_report
from paco.settings import Settings


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

    usable = judge_records(run_folder, manifest, config)
    judge_windows(run_folder, manifest, config, usable, picking or config.picking)
    judge_line(run_folder, manifest, config)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report


def judge_records(
    run_folder: Path, manifest: RunManifest, config: QCConfig
) -> dict[str, tuple[float, float] | None]:
    """G1 on each preprocessed record of the run, against its latest preprocessing attempt;
    returns each record's usable band."""
    active = manifest.profile.kind == "active"
    usable: dict[str, tuple[float, float] | None] = {}
    for record in manifest.records:
        if record.status != "succeeded":
            continue
        stream = stream_of(run_folder / record.folder / PREPROCESSED)
        result = judge_signal(record.name, stream, config.signal, active=active)
        _judge_latest(run_folder, record.name, "preprocessing", result)
        usable[record.name] = result.kept.band_hz
    return usable


def judge_windows(
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    usable: dict[str, tuple[float, float] | None],
    picking: PickingParameters,
) -> None:
    """G2 on each window's image, against its latest phase-shift attempt; then the picking, an
    attempt of its own, and G3 on its M0 curve. The band every record of the window keeps
    usable tells G2 what the data allow."""
    for window in manifest.windows:
        if window.status != "succeeded":
            continue
        folder = run_folder / window.folder
        image = load_image(folder)
        g2 = judge_image(window.folder, image, config.image, shared_band(folder, usable))
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
    previous = attempts_of(read_attempts(run_folder), unit, "picking")
    if previous:
        invalidate(run_folder / unit, "picking", len(previous))
    modes = pick_modes(image, picking)
    m0 = modes[0] if modes else None
    if m0 is not None and m0.curve is not None:
        save_pick(run_folder / unit, image, m0.curve)
    offset = nearest_offset(run_folder / unit)
    g3 = judge_curve(unit, image, m0, config.curve, band, picking, offset)
    append_attempt(
        run_folder,
        Attempt(
            unit=unit,
            stage="picking",
            attempt=len(previous) + 1,
            parameters=picking.model_dump(),
            triggered_by=triggered_by,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="succeeded",
            results={g3.gate: g3},
        ),
    )
    return g3


def judge_line(run_folder: Path, manifest: RunManifest, config: QCConfig) -> tuple[GateResult, ...]:
    """G4 over the line: the saved M0 curve of every window whose latest pick passed G3, each
    against its neighbours. A window's verdict goes to its latest picking attempt, the line's
    own (the coverage) to an attempt of the unit "line"."""
    attempts = read_attempts(run_folder)
    curves: list[Series] = []
    without: list[float] = []
    for window in manifest.windows:
        picked = latest(attempts, window.folder, "picking")
        g3 = picked.results.get("G3") if picked else None
        curve = (
            saved_m0(run_folder / window.folder / CURVES_FILE)
            if g3 is not None and g3.verdict == "pass"
            else None
        )
        if curve is None:
            without.append(window.xmid)
        else:
            curves.append(Series.from_curve(window.folder, window.xmid, curve))
    started_at = datetime.now(UTC)
    results = judge_profile(curves, config.profile, without)
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


def stream_of(path: Path) -> Stream:
    (stream,) = Load(file_paths=[path], data_type="stream").transform([])
    if not isinstance(stream, Stream):
        raise TypeError(f"{path} did not load as a stream")
    return stream
