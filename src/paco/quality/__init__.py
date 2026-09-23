"""The quality of a dispersion image: can its M0 pick be trusted, and if not, why.

Four measurements on the kept points of the automatic M0 pick (sharpness against the window's
resolution, prominence above the noise floor, agreement with the data, constant-wavelength
artifact), each with a threshold that raises a named flag. The verdict counts the flags.
"""

from .measuring import measure_quality
from .models import Flag, ImageQuality, QualityParameters, Verdict

__all__ = ["Flag", "ImageQuality", "QualityParameters", "Verdict", "measure_quality"]
