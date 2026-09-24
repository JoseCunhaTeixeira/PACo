"""Inversion of a run's picked M0 curves into layered shear-wave velocity models.

A port of PAC's seismic inversion (sigpipe's MCMC, PAC's form defaults and output files), and the
records of inversion jobs (the run's inversion.json): paco.qc.inverting runs them the QC way, and
summarize_inversion tells the agent where one stands.
"""

from .models import (
    InversionError,
    InversionParameters,
    InversionRecord,
    InversionStatus,
    JobState,
    ThicknessLayer,
    VsLayer,
    WindowInversion,
)
from .running import (
    INVERSION_FILE,
    find_job,
    new_job_id,
    read_record,
    window_result,
    write_record,
)
from .summary import summarize_inversion
from .window import build_inversion_pipeline, invert_window

__all__ = [
    "INVERSION_FILE",
    "InversionError",
    "InversionParameters",
    "InversionRecord",
    "InversionStatus",
    "JobState",
    "ThicknessLayer",
    "VsLayer",
    "WindowInversion",
    "build_inversion_pipeline",
    "find_job",
    "invert_window",
    "new_job_id",
    "read_record",
    "summarize_inversion",
    "window_result",
    "write_record",
]
