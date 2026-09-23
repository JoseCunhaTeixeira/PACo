"""The inversion's parameters, what a run records of its inversion (inversion.json), and what the
agent reads back (InversionStatus)."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The descriptions are what the agent reads (see paco.server's inversion_settings). Defaults are
# PAC's form values.


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
    n_iterations: int = Field(default=100_000, gt=0, description="Iterations of each chain.")
    n_burnin_iterations: int = Field(
        default=10_000, gt=0, description="First iterations of each chain, discarded."
    )
    n_chains: int = Field(default=5, gt=0, description="Independent chains per window.")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.vs_layers) != self.n_layers:
            raise ValueError(f"vs_layers must have length n_layers ({self.n_layers})")
        if len(self.thickness_layers) != self.n_layers - 1:
            raise ValueError(
                f"thickness_layers must have length n_layers - 1 ({self.n_layers - 1})"
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
    duration_s: float | None = None
    error: str | None = None  # "<type>: <message>"; the traceback is in <folder>/error.log
    vs_m_s: tuple[float, ...] | None = None  # median model, per layer, top down
    thicknesses_m: tuple[float, ...] | None = None  # median model, layers above the half-space


class InversionRecord(BaseModel):
    """A run's inversion job and its results so far, written as inversion.json in the run folder."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    run_id: str
    parameters: InversionParameters
    state: JobState
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total: int  # windows to invert: the run's picked windows
    windows: tuple[WindowInversion, ...] = ()  # finished so far, sorted by xmid
    error: str | None = None  # why the whole job failed, or was interrupted


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
    vs_m_s: tuple[tuple[float, float], ...]  # per layer, top down: range of the median models
    depths_m: tuple[tuple[float, float], ...]  # per interface, top down: range of its depth
    errors: tuple[str, ...]  # a few distinct failures: "xmid 12.50: <type>: <message>"
    error: str | None  # why the whole job failed, or was interrupted
