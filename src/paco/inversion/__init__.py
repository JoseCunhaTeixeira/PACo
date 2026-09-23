"""Inversion of a run's picked M0 curves into layered shear-wave velocity models.

A port of PAC's seismic inversion (sigpipe's MCMC, PAC's form defaults and output files), run as a
job: submit_inversion records it in the run's inversion.json, invert_run runs it window by window
in worker processes, and summarize_inversion tells the agent where it stands.
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
    check_inversion,
    find_job,
    invert_run,
    read_record,
    submit_inversion,
    write_record,
)
from .summary import summarize_inversion
from .window import build_inversion_pipeline, invert_window

__all__ = [
    "InversionError",
    "InversionParameters",
    "InversionRecord",
    "InversionStatus",
    "JobState",
    "ThicknessLayer",
    "VsLayer",
    "WindowInversion",
    "build_inversion_pipeline",
    "check_inversion",
    "find_job",
    "invert_run",
    "invert_window",
    "read_record",
    "submit_inversion",
    "summarize_inversion",
    "write_record",
]
