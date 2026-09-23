"""What pick records on disk (pick.json) and what the agent reads back (PickSummary)."""

from pydantic import BaseModel, ConfigDict

from paco.picking import PickingParameters


class PickedWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    xmid: float
    folder: str  # xmid_<x>, inside the run folder
    n_points: int  # of the saved M0 curve, one per metre of wavelength
    band_hz: tuple[float, float]
    replaced: bool  # the window already had an M0 curve, which this one replaced


class RunPicks(BaseModel):
    """The windows pick saved an M0 curve for, and how: written as pick.json."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    picking: PickingParameters  # from the run's quality.json
    windows: tuple[PickedWindow, ...]  # the good windows, sorted by xmid


class PickSummary(BaseModel):
    """Short description of a run's picks for the agent."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    profile: str
    n_picked: int
    picked_xmids: tuple[str, ...]  # runs of consecutive picked windows: "2.88-20.88 m (7)"
    n_skipped: int  # doubtful or bad windows: no curve saved
    n_replaced: int  # picked windows whose previous M0 curve was replaced
