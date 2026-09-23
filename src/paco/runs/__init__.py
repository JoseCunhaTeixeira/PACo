"""Processing runs: a profile processed with a preset, one sigpipe pipeline per MASW window.

A run writes PAC's layout under <output_dir>/<profile>/<run_id>/: one xmid_<x>/ folder per
window, as PAC's UI expects, plus run.json with everything needed to understand or reproduce it.
"""

from .models import RunError, RunManifest, RunSummary, WindowOutcome
from .processing import run_processing
from .summary import summarize_run

__all__ = [
    "RunError",
    "RunManifest",
    "RunSummary",
    "WindowOutcome",
    "run_processing",
    "summarize_run",
]
