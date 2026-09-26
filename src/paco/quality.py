"""The quality of a dispersion image's M0 pick, as the curve QC (G3) reads it: sigpipe's four
measures (sharpness against the window's resolution, prominence above the noise floor, agreement
with the data, constant-wavelength artifact; sigpipe.masw.quality.pick), each with a threshold
that raises a named flag. The verdict counts the flags."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sigpipe.algorithms.picking.dispersion.tracking import PickedMode
from sigpipe.base import DispersionImage
from sigpipe.masw.quality.pick import measure_pick


class QualityParameters(BaseModel):
    """Limits past which a measurement raises a flag.

    Tuned on the demo windows only, so recalibrate them on a reference set of windows judged by
    hand before trusting them. Sharpness and prominence are measured against a perfect plane
    wave for the same window, frequency and velocity (the user's decision of 2026-09-25): the
    demo's windows score 1.00 on both at every length from 5 to 24 receivers, where fixed
    limits measured the array, not the data (a 5-receiver window cannot be as prominent as a
    24-receiver one).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_sharpness: float = Field(
        default=0.8,
        gt=0,
        description="Flag when the median peak width is below this multiple of a perfect plane "
        "wave's width for the same window (a plane wave scores 1).",
    )
    min_prominence: float = Field(
        default=0.5,
        gt=0,
        description="Flag when the median peak prominence (height above the noise floor over "
        "its column's median height) is below this multiple of a perfect plane wave's for the "
        "same window (a plane wave scores 1).",
    )
    min_on_data: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Flag when a smaller share of the points sit on their column's brightest "
        "value (within 10 %).",
    )
    max_constant_wavelength: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Flag when a larger share of the points have a velocity growing like "
        "frequency: the edge of what the window resolves, not a dispersion curve.",
    )


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
    parameters = parameters or QualityParameters()
    measures = measure_pick(image, m0)
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
