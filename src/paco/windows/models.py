"""MASW window parameters and windows, with PAC's field names."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sigpipe.base import LinearAcquisition


class MASWParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    length: int = Field(ge=3)  # receivers per window
    step: int = Field(gt=0)  # receivers between two window starts
    distance_min: float = Field(ge=0)  # m, from the source to the window middle, exclusive
    distance_max: float = Field(gt=0)  # m, exclusive

    @model_validator(mode="after")
    def _check_distances(self) -> Self:
        if self.distance_max <= self.distance_min:
            raise ValueError("distance_max must be greater than distance_min")
        return self


class MASWWindow(BaseModel):
    xmid: float
    selected_files: list[Path]
    receiver_indices: list[int]
    acquisitions: list[LinearAcquisition]
