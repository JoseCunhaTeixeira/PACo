"""Processing runs: a profile processed with a preset, each record preprocessed once, then one
sigpipe pipeline per MASW window.

A run writes PAC's layout under <output_dir>/<profile>/<run_id>/: one xmid_<x>/ folder per
window, as PAC's UI expects, plus records/ with the preprocessed records, and run.json with
everything needed to understand or reproduce it.
"""

from .finding import find_run, load_image, load_manifest
from .models import RecordOutcome, RunError, RunManifest, RunSummary, WindowOutcome
from .processing import run_processing, start_worker
from .summary import summarize_run

__all__ = [
    "RecordOutcome",
    "RunError",
    "RunManifest",
    "RunSummary",
    "WindowOutcome",
    "find_run",
    "load_image",
    "load_manifest",
    "run_processing",
    "start_worker",
    "summarize_run",
]
