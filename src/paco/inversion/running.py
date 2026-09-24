"""Inversion jobs on disk: a run's inversion.json, written whole then renamed so that a reader
never sees half of it, and found back by its job ID. The QC way of running one (the windows G4
passed, with G5 and G6) is paco.qc.inverting's."""

import re
import secrets
from datetime import UTC, datetime
from pathlib import Path

from paco.inversion.measuring import InversionMeasures
from paco.inversion.models import InversionError, InversionRecord, WindowInversion
from paco.settings import Settings

INVERSION_FILE = "inversion.json"
_JOB_ID = re.compile(r"inv-\d{8}-\d{6}-[0-9a-f]{4}")


def new_job_id() -> str:
    """A job ID, e.g. inv-20260923-142501-a3f9 (UTC time and a random suffix)."""
    return f"inv-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


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


def window_result(
    xmid: float,
    folder: str,
    measures: InversionMeasures,
    verdict: str | None,
    duration_s: float | None = None,
) -> WindowInversion:
    """What a job reports of one window's inversion: the layered median, and the smooth median
    it monitors (Vs at the job's depths, the useful depth, the misfit)."""
    return WindowInversion(
        xmid=xmid,
        folder=folder,
        status="succeeded",
        verdict=verdict,
        duration_s=duration_s,
        vs_m_s=measures.vs_layers,
        thicknesses_m=tuple(
            round(b - a, 2)
            for a, b in zip((0.0, *measures.interfaces_m), measures.interfaces_m, strict=False)
        ),
        vs_at_depths_m_s=tuple(vs for _, vs in measures.vs_at_depths),
        # None from the measures: the data inform the model down to its bottom.
        useful_depth_m=measures.useful_depth_m or measures.depth_max_m,
        misfit=measures.fit("smooth_median").misfit,
    )
