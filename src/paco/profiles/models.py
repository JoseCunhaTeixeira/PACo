"""Data shapes of a profile: the validated Profile used by processing, and the summary the agent
reads."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from sigpipe.base import Coordinate


class ProfileKind(StrEnum):
    ACTIVE = "active"
    PASSIVE = "passive"


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    n_traces: int
    sampling_rate_hz: float
    duration_s: float
    source: Coordinate | None


class Profile(BaseModel):
    """Everything processing needs to know about a profile, validated."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ProfileKind
    folder: Path
    records: tuple[Record, ...]
    receivers: tuple[Coordinate, ...]

    @property
    def sampling_rate_hz(self) -> float:
        return self.records[0].sampling_rate_hz

    @property
    def nyquist_hz(self) -> float:
        return self.sampling_rate_hz / 2


class ProfileSummary(BaseModel):
    """Short description of a profile for the agent, with no per-record or per-trace lists."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ProfileKind
    n_records: int
    n_receivers: int
    receiver_x_range_m: tuple[float, float]
    receiver_spacing_m: float
    sampling_rate_hz: float
    nyquist_hz: float
    record_duration_range_s: tuple[float, float]
    source_x_range_m: tuple[float, float] | None
