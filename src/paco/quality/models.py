"""The quality thresholds, the quality of one dispersion image, and of a whole run."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from paco.picking import PickingParameters


class QualityParameters(BaseModel):
    """Limits past which a measurement raises a flag.

    Tuned on the 14 demo windows only (active_p1 and passive_p1, 24-receiver windows), so they
    separate those perfectly by construction: recalibrate them on a reference set of windows judged
    by hand before trusting them. min_sharpness sits below the score of a perfect plane wave,
    0.98 to 1.08 depending on the array, to leave room for noise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The descriptions are what the agent reads (see paco.server's quality_settings).
    min_sharpness: float = Field(
        default=0.8,
        gt=0,
        description="Flag when the median peak width is below this multiple of the width the "
        "window can resolve (a perfect plane wave scores about 1).",
    )
    min_prominence: float = Field(
        default=2.0,
        gt=0,
        description="Flag when the median peak height above the noise floor is below this "
        "multiple of its column's median height.",
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
    - sharpness: peak width above the noise floor, over the width the window can resolve at that
      wavelength (v * wavelength / window length). A perfect plane wave scores about 1, and a
      propagating wave cannot be much narrower: well below 1, the peak is a fringe or an edge, not
      a ridge.
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


class WindowQuality(BaseModel):
    model_config = ConfigDict(frozen=True)

    xmid: float
    folder: str  # xmid_<x>, inside the run folder
    quality: ImageQuality


class RunQuality(BaseModel):
    """The quality of every window of a run, and how it was measured: written as quality.json."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    picking: PickingParameters
    thresholds: QualityParameters
    windows: tuple[WindowQuality, ...]  # sorted by xmid


class QualitySummary(BaseModel):
    """Short description of a run's quality for the agent."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    profile: str
    n_windows: int  # windows with a dispersion image
    good: int
    doubtful: int
    bad: int
    good_xmids: tuple[str, ...]  # runs of consecutive good windows: "2.88-20.88 m (7)"
    flags: dict[str, int]  # windows raising each flag, most frequent first
    advice: tuple[str, ...]  # what to change, for the most frequent flags
