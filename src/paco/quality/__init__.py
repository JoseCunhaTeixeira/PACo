"""The quality of a dispersion image: can its M0 pick be trusted, and if not, why.

Four measurements on the kept points of the automatic M0 pick (sharpness against the window's
resolution, prominence above the noise floor, agreement with the data, constant-wavelength
artifact), each with a threshold that raises a named flag. The verdict counts the flags.
"""

from .assessing import (
    ADVICE,
    dispersion_quality,
    good_stretches,
    load_quality,
    summarize_quality,
)
from .measuring import measure_quality
from .models import (
    Flag,
    ImageQuality,
    QualityParameters,
    QualitySummary,
    RunQuality,
    Verdict,
    WindowQuality,
)

__all__ = [
    "ADVICE",
    "Flag",
    "ImageQuality",
    "QualityParameters",
    "QualitySummary",
    "RunQuality",
    "Verdict",
    "WindowQuality",
    "dispersion_quality",
    "good_stretches",
    "load_quality",
    "measure_quality",
    "summarize_quality",
]
