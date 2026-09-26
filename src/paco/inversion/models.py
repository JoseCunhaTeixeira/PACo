"""What a run records of its inversion job (inversion.json), and what the agent reads back
(InversionStatus). The inversion itself is sigpipe's (sigpipe.masw.inversion)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

type JobState = Literal["queued", "running", "succeeded", "failed", "interrupted"]


class WindowInversion(BaseModel):
    """One window's inversion: its median model, or why it failed."""

    model_config = ConfigDict(frozen=True)

    xmid: float
    folder: str  # xmid_<x>, inside the run folder
    status: Literal["succeeded", "failed"]
    verdict: str | None = None  # G5's latest on the window's model
    duration_s: float | None = None
    error: str | None = None  # "<type>: <message>"; the traceback is in <folder>/error.log
    vs_m_s: tuple[float, ...] | None = None  # median model, per layer, top down
    thicknesses_m: tuple[float, ...] | None = None  # median model, layers above the half-space
    # The smooth median, PAC's default view and the model monitored: its Vs at the job's depths,
    # the depth the data inform it down to (half its curve's longest wavelength, MASW's depth of
    # investigation), and its fit to the curve (RMS of the residuals over the uncertainties).
    vs_at_depths_m_s: tuple[float, ...] | None = None
    useful_depth_m: float | None = None
    misfit: float | None = None


class InversionRecord(BaseModel):
    """A run's inversion job and its results so far, written as inversion.json in the run folder."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    run_id: str
    # The values the user gave (InversionParameters' fields); every window's bounds come from its
    # own curve, through the checks before S4.
    given: dict[str, Any] = {}
    state: JobState
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total: int  # windows to invert: the run's picked windows
    depths_m: tuple[float, ...] = ()  # where the smooth medians are reported
    windows: tuple[WindowInversion, ...] = ()  # finished so far, sorted by xmid
    error: str | None = None  # why the whole job failed, or was interrupted
    summary: str | None = None  # what the gates found, once the job has ended
    changed: tuple[str, ...] = ()  # the settings the gates and the checks changed, in words


class InversionStatus(BaseModel):
    """Short description of an inversion job for the agent."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    run_id: str
    state: JobState
    done: int
    total: int
    n_failed: int
    elapsed_s: float | None  # since the job started
    # The smooth median models so far (PAC's default view): per depth of depths_m, the range of
    # their Vs; the range of the depths the data inform them down to (half each curve's longest
    # wavelength); the range of their fits to the curves (RMS of the residuals over the
    # uncertainties: about 1 fits within them).
    depths_m: tuple[float, ...]
    vs_m_s: tuple[tuple[float, float], ...]
    useful_depth_m: tuple[float, float] | None
    misfit: tuple[float, float] | None
    errors: tuple[str, ...]  # a few distinct failures: "xmid 12.50: <type>: <message>"
    error: str | None  # why the whole job failed, or was interrupted
    summary: str | None = None  # the gates' summary (G5, G6), once the job has ended
    changed: tuple[str, ...] = ()  # the settings the gates and the checks changed: report them
