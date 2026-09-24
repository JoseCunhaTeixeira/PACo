"""The inversion's parameters, what a run records of its inversion (inversion.json), and what the
agent reads back (InversionStatus)."""

from datetime import datetime
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The descriptions are what the agent reads (see paco.server's inversion_settings). Defaults are
# PAC's form values.

# sigpipe's sampler keeps one model every 150 iterations after the burn-in (bayesbay's
# save_every): with fewer left, a chain keeps none, and every window fails with a KeyError. PAC
# does not check it.
SAMPLE_EVERY = 150


class VsLayer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vs_min: float = Field(default=100.0, gt=0, description="m/s")
    vs_max: float = Field(default=1_000.0, gt=0, description="m/s")
    vs_perturb_std: float = Field(
        default=20.0, gt=0, description="m/s, size of the random steps the sampler takes"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.vs_max <= self.vs_min:
            raise ValueError("vs_max must be greater than vs_min")
        return self


class ThicknessLayer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thickness_min: float = Field(default=1.0, gt=0, description="m")
    thickness_max: float = Field(default=10.0, gt=0, description="m")
    thickness_perturb_std: float = Field(
        default=1.0, gt=0, description="m, size of the random steps the sampler takes"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.thickness_max <= self.thickness_min:
            raise ValueError("thickness_max must be greater than thickness_min")
        return self


class InversionParameters(BaseModel):
    """The layered model sought, and the sampler's effort: PAC's inversion form."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_layers: int = Field(default=2, ge=2, description="Layers, the half-space included.")
    vs_layers: tuple[VsLayer, ...] = Field(
        default=(VsLayer(), VsLayer()),
        description="Shear-wave velocity bounds of each layer, top down: n_layers of them.",
    )
    thickness_layers: tuple[ThicknessLayer, ...] = Field(
        default=(ThicknessLayer(),),
        description="Thickness bounds of each layer above the half-space: n_layers - 1 of them.",
    )
    n_iterations: int = Field(
        default=100_000,
        gt=0,
        description=f"Iterations of each chain; one model is kept every {SAMPLE_EVERY} after the "
        "burn-in.",
    )
    n_burnin_iterations: int = Field(
        default=10_000,
        gt=0,
        description="First iterations of each chain, discarded. Left out: a tenth of n_iterations.",
    )
    n_chains: int = Field(default=5, gt=0, description="Independent chains per window.")

    @model_validator(mode="before")
    @classmethod
    def _burnin_follows_iterations(cls, data: Any) -> Any:  # noqa: ANN401
        """A tenth of n_iterations, PAC's ratio, when only the iterations are given: asking for
        2,000 iterations kept PAC's 10,000 of burn-in, and left nothing to sample."""
        if not isinstance(data, dict):
            return data
        values = cast(dict[str, Any], data)
        iterations = values.get("n_iterations")
        if "n_burnin_iterations" in values or not isinstance(iterations, int | float):
            return values
        return {**values, "n_burnin_iterations": max(1, int(iterations) // 10)}

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.vs_layers) != self.n_layers:
            raise ValueError(f"vs_layers must have length n_layers ({self.n_layers})")
        if len(self.thickness_layers) != self.n_layers - 1:
            raise ValueError(
                f"thickness_layers must have length n_layers - 1 ({self.n_layers - 1})"
            )
        if self.n_iterations - self.n_burnin_iterations < SAMPLE_EVERY:
            raise ValueError(
                f"n_iterations ({self.n_iterations}) must exceed n_burnin_iterations "
                f"({self.n_burnin_iterations}) by at least {SAMPLE_EVERY}: each chain keeps one "
                f"model every {SAMPLE_EVERY} iterations after the burn-in"
            )
        return self


class InversionError(ValueError):
    """An inversion cannot start. Messages are written to be read by the agent."""


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
    # the depth the data inform it down to, and its fit to the curve (RMS of the residuals over
    # the uncertainties).
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
    # their Vs; the range of the depths the data inform them down to; the range of their fits
    # to the curves (RMS of the residuals over the uncertainties: about 1 fits within them).
    depths_m: tuple[float, ...]
    vs_m_s: tuple[tuple[float, float], ...]
    useful_depth_m: tuple[float, float] | None
    misfit: tuple[float, float] | None
    errors: tuple[str, ...]  # a few distinct failures: "xmid 12.50: <type>: <message>"
    error: str | None  # why the whole job failed, or was interrupted
    summary: str | None = None  # the gates' summary (G5, G6), once the job has ended
    changed: tuple[str, ...] = ()  # the settings the gates and the checks changed: report them
