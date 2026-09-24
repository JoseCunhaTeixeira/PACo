"""What a run records on disk (run.json) and what the agent reads back (RunSummary)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from paco.presets import Preset
from paco.profiles import ProfileSummary
from paco.windows import Exclusions


class RunError(ValueError):
    """A run cannot start. Messages are written to be read by the agent."""


class RecordOutcome(BaseModel):
    """One record's preprocessing, done once for every window that uses it."""

    model_config = ConfigDict(frozen=True)

    name: str  # the record's file name, e.g. 1.dat
    folder: str  # records/<stem>, inside the run folder: the preprocessed stream and its figure
    status: Literal["succeeded", "failed"]
    duration_s: float | None = None
    error: str | None = None  # "<type>: <message>"; the traceback is in <folder>/error.log


class WindowOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    xmid: float
    folder: str  # xmid_<x>, inside the run folder, as PAC names it
    status: Literal["succeeded", "failed"]
    duration_s: float | None = None
    error: str | None = None  # "<type>: <message>"; the traceback is in <folder>/error.log


class RunManifest(BaseModel):
    """Everything needed to understand or reproduce a run, written as run.json."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    profile: ProfileSummary
    preset: Preset  # resolved: every value the pipelines received
    versions: dict[str, str]
    started_at: datetime
    finished_at: datetime
    n_positions: int  # windows the line allows, before shot selection
    # In file order; empty in the runs made before the preprocessing split (2026-09-24).
    records: tuple[RecordOutcome, ...] = ()
    windows: tuple[WindowOutcome, ...]  # sorted by xmid
    # What G1 took out of the windows: the phase shift done again leaves them out too.
    exclusions: Exclusions = Exclusions()


class RunSummary(BaseModel):
    """Short description of a run for the agent."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    profile: str
    preset: str
    path: str  # run folder, relative to the output directory
    n_windows: int
    # Windows with a dispersion image: not judged yet. Qwen3-4B read "n_succeeded" as good
    # windows, and answered without calling dispersion_quality.
    n_processed: int
    n_failed: int
    n_skipped: int  # positions without any valid shot (active profiles only)
    duration_s: float
    errors: tuple[str, ...]  # a few distinct failures: "xmid 12.50: <type>: <message>"
