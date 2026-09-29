"""The quality of a dispersion image's M0 pick, as the curve QC (G3) reads it: sigpipe's four
measures (sharpness against the window's resolution, prominence above the noise floor, agreement
with the data, constant-wavelength artifact; sigpipe.masw.quality.pick), each with a threshold
that raises a named flag. The verdict counts the flags."""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from sigpipe.algorithms.picking.dispersion.tracking import PickedMode
from sigpipe.base import DispersionImage
from sigpipe.masw.quality.curve import PickLimits
from sigpipe.masw.quality.pick import PickMeasures, measure_pick


class QualityParameters(PickLimits):
    """Limits past which a measurement raises a flag (sigpipe's PickLimits, PAC's alike)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


type Flag = Literal["no_ridge", "sharpness", "prominence", "on_data", "constant_wavelength"]
type Verdict = Literal["good", "doubtful", "bad"]


class ImageQuality(BaseModel):
    """How much the M0 pick of one dispersion image can be trusted, and why.

    Every measurement is a median over the pick's kept points:
    - sharpness: peak width at half its height above the noise floor, over the same width for a
      perfect plane wave at that frequency and velocity, through the window's receivers and on
      the same grid. A propagating wave cannot be much narrower: well below 1, the peak is a
      fringe or an edge, not a ridge.
    - prominence: peak height above the floor over the column's median height above the floor,
      over the same ratio for the perfect plane wave: 1 is as prominent as the window allows.
    - on_data: share of points on their column's brightest value (within 10 %), so the data and not
      the tracker's smoothing drew the curve.
    - constant_wavelength: share of points where velocity grows like frequency, the signature of the
      edge of what the window resolves.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict  # good: no flag; doubtful: one; bad: two or more, or no ridge at all
    flags: tuple[Flag, ...]
    n_points: int
    band_hz: tuple[float, float] | None
    sharpness: float | None = None
    prominence: float | None = None
    on_data: float | None = None
    constant_wavelength: float | None = None


def measure_quality(
    image: DispersionImage, m0: PickedMode | None, parameters: QualityParameters | None = None
) -> ImageQuality:
    """The quality of `image`, from the measures of its M0 pick against `parameters`."""
    return quality_of(measure_pick(image, m0), parameters or QualityParameters())


def quality_of(measures: PickMeasures, parameters: PickLimits) -> ImageQuality:
    """The quality of an M0 pick from its `measures` (sigpipe's measure_pick) against
    `parameters`."""
    if (
        measures.sharpness is None
        or measures.prominence is None
        or measures.on_data is None
        or measures.constant_wavelength is None
    ):
        return ImageQuality(
            verdict="bad", flags=("no_ridge",), n_points=measures.n_points, band_hz=None
        )

    flags: list[Flag] = []
    if measures.sharpness < parameters.min_sharpness:
        flags.append("sharpness")
    if measures.prominence < parameters.min_prominence:
        flags.append("prominence")
    if measures.on_data < parameters.min_on_data:
        flags.append("on_data")
    if measures.constant_wavelength > parameters.max_constant_wavelength:
        flags.append("constant_wavelength")

    return ImageQuality(
        verdict="good" if not flags else "doubtful" if len(flags) == 1 else "bad",
        flags=tuple(flags),
        n_points=measures.n_points,
        band_hz=measures.band_hz,
        sharpness=measures.sharpness,
        prominence=measures.prominence,
        on_data=measures.on_data,
        constant_wavelength=measures.constant_wavelength,
    )
