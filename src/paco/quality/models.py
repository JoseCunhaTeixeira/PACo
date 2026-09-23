"""The quality thresholds, and the quality of one dispersion image."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QualityParameters(BaseModel):
    """Limits past which a measurement raises a flag.

    Tuned on the 14 demo windows only (active_p1 and passive_p1, 24-receiver windows), so they
    separate those perfectly by construction: recalibrate them on a reference set of windows judged
    by hand before trusting them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_sharpness: float = Field(default=1.0, gt=0)
    min_prominence: float = Field(default=2.0, gt=0)
    min_on_data: float = Field(default=0.6, ge=0, le=1)
    max_constant_wavelength: float = Field(default=0.4, ge=0, le=1)


type Flag = Literal["no_ridge", "sharpness", "prominence", "on_data", "constant_wavelength"]
type Verdict = Literal["good", "doubtful", "bad"]


class ImageQuality(BaseModel):
    """How much the M0 pick of one dispersion image can be trusted, and why.

    Every measurement is a median over the pick's kept points:
    - sharpness: peak width above the noise floor, over the width the window can resolve at that
      wavelength (v * wavelength / window length). A propagating wave cannot be narrower: below 1,
      the peak is a fringe or an edge, not a ridge.
    - prominence: peak height above the floor, over the column's median height above the floor.
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
