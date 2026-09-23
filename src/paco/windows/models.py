"""MASW window parameters and windows, with PAC's field names."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sigpipe.base import LinearAcquisition


class MASWParameters(BaseModel):
    """Defaults are PAC's form defaults, except distance_max (1000 m), so that a partial override
    keeps the other values."""

    # Unknown keys are errors: presets expose these parameters to the agent's overrides.
    model_config = ConfigDict(frozen=True, extra="forbid")

    length: int = Field(default=3, ge=3)  # receivers per window
    step: int = Field(default=1, gt=0)  # receivers between two window starts
    distance_min: float = Field(default=0.0, ge=0)  # m, from the source to the window middle
    distance_max: float = Field(default=1_000.0, gt=0)  # m; both bounds exclusive

    @model_validator(mode="after")
    def _check_distances(self) -> Self:
        if self.distance_max <= self.distance_min:
            raise ValueError(
                f"distance_max ({self.distance_max:g}) must be greater than "
                f"distance_min ({self.distance_min:g})"
            )
        return self


class MASWWindow(BaseModel):
    xmid: float
    selected_files: list[Path]
    receiver_indices: list[int]
    acquisitions: list[LinearAcquisition]
