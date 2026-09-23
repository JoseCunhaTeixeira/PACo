"""Inverting a run: the M0 curve of every picked window, one worker process per window.

A job goes through inversion.json in the run folder: submit_inversion writes it queued,
invert_run marks it running and records every window as it finishes, then marks it succeeded or
failed. summarize_inversion reads it back, from any thread, while the job runs.
"""

import re
import secrets
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from paco.inversion.models import (
    InversionError,
    InversionParameters,
    InversionRecord,
    WindowInversion,
)
from paco.inversion.window import invert_window
from paco.picks import RunPicks
from paco.runs import find_run, start_worker
from paco.settings import Settings

INVERSION_FILE = "inversion.json"
PICKS_FILE = "pick.json"
ERROR_FILE = "inversion_error.log"
_JOB_ID = re.compile(r"inv-\d{8}-\d{6}-[0-9a-f]{4}")

# Called with (windows done, windows in the job).
type ProgressCallback = Callable[[int, int], None]


def submit_inversion(
    run_id: str, parameters: InversionParameters, settings: Settings
) -> InversionRecord:
    """Record a new inversion job of run `run_id`, queued, and return it.

    Refuses a run without picks, or whose inversion is already queued or running.
    """
    folder, picks = check_inversion(run_id, settings)
    record = InversionRecord(
        job_id=f"inv-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}",
        run_id=run_id,
        parameters=parameters,
        state="queued",
        submitted_at=datetime.now(UTC),
        total=len(picks.windows),
    )
    write_record(folder, record)
    return record


def check_inversion(run_id: str, settings: Settings) -> tuple[Path, RunPicks]:
    """The folder and picks of run `run_id`, if an inversion of it can start now."""
    folder = find_run(run_id, settings)
    picks = _picks(folder, run_id)
    if (previous := read_record(folder)) is not None and previous.state in ("queued", "running"):
        raise InversionError(
            f"Run '{run_id}' is already being inverted: job '{previous.job_id}' is "
            f"{previous.state}. Follow it with job_status."
        )
    return folder, picks


def invert_run(
    record: InversionRecord, settings: Settings, on_progress: ProgressCallback | None = None
) -> InversionRecord:
    """Run queued job `record` to its end, recording every window as it finishes.

    A window that fails does not stop the job: its error goes to inversion.json and to its
    folder's inversion_error.log. The job fails when every window failed, or on any other
    failure.
    """
    folder = find_run(record.run_id, settings)
    record = record.model_copy(update={"state": "running", "started_at": datetime.now(UTC)})
    write_record(folder, record)
    try:
        picks = _picks(folder, record.run_id)
        record = _invert_windows(record, picks, folder, settings.workers, on_progress)
        if record.windows and all(window.status == "failed" for window in record.windows):
            # Not "succeeded": Qwen3-4B read that as a job done, with a few errors.
            update = {"state": "failed", "error": "Every window failed: see errors."}
        else:
            update = {"state": "succeeded"}
        record = record.model_copy(update=update)
    except Exception as exc:
        record = record.model_copy(
            update={"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
    record = record.model_copy(update={"finished_at": datetime.now(UTC)})
    write_record(folder, record)
    return record


def find_job(job_id: str, settings: Settings) -> tuple[Path, InversionRecord]:
    """The run folder and record of inversion job `job_id`."""
    if _JOB_ID.fullmatch(job_id):
        for path in sorted(settings.output_dir.glob(f"*/*/{INVERSION_FILE}")):
            record = InversionRecord.model_validate_json(path.read_text())
            if record.job_id == job_id:
                return path.parent, record
    raise InversionError(
        f"Unknown job '{job_id}'. A job ID looks like inv-20260923-142501-a3f9, as invert returns "
        "it; a run's latest inversion replaces the earlier ones."
    )


def read_record(folder: Path) -> InversionRecord | None:
    path = folder / INVERSION_FILE
    return InversionRecord.model_validate_json(path.read_text()) if path.exists() else None


def write_record(folder: Path, record: InversionRecord) -> None:
    # Written whole, then renamed: a reader never sees half a file.
    path = folder / INVERSION_FILE
    partial = path.with_suffix(".partial")
    partial.write_text(record.model_dump_json(indent=2))
    partial.replace(path)


def _picks(folder: Path, run_id: str) -> RunPicks:
    path = folder / PICKS_FILE
    if not path.exists():
        raise InversionError(f"Run '{run_id}' has no picks yet: call pick first.")
    return RunPicks.model_validate_json(path.read_text())


def _invert_windows(
    record: InversionRecord,
    picks: RunPicks,
    run_folder: Path,
    workers: int,
    on_progress: ProgressCallback | None,
) -> InversionRecord:
    finished: list[WindowInversion] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=start_worker, initargs=(run_folder,)
    ) as executor:
        futures: dict[Future[tuple[float, WindowInversion]], tuple[float, str]] = {
            executor.submit(_invert_window, run_folder / window.folder, record.parameters): (
                window.xmid,
                window.folder,
            )
            for window in picks.windows
        }
        if on_progress is not None:
            on_progress(0, len(futures))
        for done, future in enumerate(as_completed(futures), start=1):
            xmid, folder = futures[future]
            try:
                duration_s, window = future.result()
                window = window.model_copy(update={"xmid": xmid, "duration_s": duration_s})
            except Exception as exc:
                (run_folder / folder / ERROR_FILE).write_text(
                    "".join(traceback.format_exception(exc))
                )
                window = WindowInversion(
                    xmid=xmid,
                    folder=folder,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finished.append(window)
            ordered = tuple(sorted(finished, key=lambda window: window.xmid))
            record = record.model_copy(update={"windows": ordered})
            write_record(run_folder, record)
            if on_progress is not None:
                on_progress(done, len(futures))
    return record


def _invert_window(folder: Path, parameters: InversionParameters) -> tuple[float, WindowInversion]:
    """Runs in a worker: one window's inversion, its duration in seconds and its median model."""
    start = time.perf_counter()
    median = invert_window(folder, parameters).median
    window = WindowInversion(
        xmid=0.0,  # set by the caller, which knows it
        folder=folder.name,
        status="succeeded",
        vs_m_s=tuple(float(vs) for vs in median.vs_s),
        # The last thickness belongs to the half-space.
        thicknesses_m=tuple(float(thickness) for thickness in median.thicknesses[:-1]),
    )
    return time.perf_counter() - start, window
