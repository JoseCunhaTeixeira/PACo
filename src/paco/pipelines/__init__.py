"""sigpipe pipelines for one MASW window, built from a resolved preset.

Port of PAC's adapters: the same steps in the same order, with the same fixed values. The
preset's stages give the tunable ones.
"""

from .active import build_active_pipeline
from .passive import build_passive_pipeline
from .registry import PIPELINE_BUILDERS, build_pipeline

__all__ = [
    "PIPELINE_BUILDERS",
    "build_active_pipeline",
    "build_passive_pipeline",
    "build_pipeline",
]
