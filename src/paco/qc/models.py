"""What every gate says, in one language (docs/qc_workflow.md, rules 1 to 9): a verdict per unit,
each metric with its threshold, the flags raised, and for each flag the stage at fault and a
change that can be applied as it is; what the result keeps of the data; and one line of the QC
log per attempt."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The stages, in pipeline order: S1 to S4 of the spec, then the petrophysical inversion, which
# reads the picks as S4 does. Going back to one invalidates the later ones that use it, for the
# affected xmids only (attempts.downstream).
type Stage = Literal["preprocessing", "phase_shift", "picking", "inversion", "petro_inversion"]
STAGES: tuple[Stage, ...] = (
    "preprocessing",
    "phase_shift",
    "picking",
    "inversion",
    "petro_inversion",
)

type Verdict = Literal["pass", "retry", "reject"]


def stage_index(stage: Stage) -> int:
    return STAGES.index(stage)


class Metric(BaseModel):
    """One measurement against its threshold (rule 1)."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float | None  # None: could not be measured
    threshold: float | None = None
    bound: Literal["min", "max"] | None = None  # the value must stay above (min) or below (max)
    passed: bool
    unit: str = ""


class Override(BaseModel):
    """Change the parameters of a stage: `overrides` are the stage's own, ready to apply."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["override"] = "override"
    stage: Stage
    overrides: dict[str, Any]


class ExcludeTraces(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["exclude_traces"] = "exclude_traces"
    record: str  # the record's file name
    traces: tuple[int, ...]  # receiver indices, from 0


class ExcludeRecord(BaseModel):
    """Leave the record out of the windows that use it."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["exclude_record"] = "exclude_record"
    record: str


class Reject(BaseModel):
    """A legitimate end state, with its reason (rule 2)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["reject"] = "reject"
    reason: str


class Keep(BaseModel):
    """Nothing to change: the flag is information (an inverse trend is geology, not a fault)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["keep"] = "keep"
    note: str


# What a gate can advise (rule 2).
type Action = Annotated[
    Override | ExcludeTraces | ExcludeRecord | Reject | Keep, Field(discriminator="kind")
]


class Flag(BaseModel):
    """A problem a gate found, with the stage most likely at fault and what to do."""

    model_config = ConfigDict(frozen=True)

    name: str  # e.g. "ridge_at_vmax"
    message: str  # one sentence, for the agent and the report
    stage: Stage  # most likely at fault: this gate's stage, or an earlier one
    action: Action
    fixable: bool = True  # False: no parameter improves it (a dead geophone, a shifted trigger)


class Kept(BaseModel):
    """What a result keeps of the data (rule 6): a fix must not win by throwing data away."""

    model_config = ConfigDict(frozen=True)

    band_hz: tuple[float, float] | None = None
    wavelength_m: tuple[float, float] | None = None
    n_points: int | None = None
    n_traces: int | None = None
    n_records: int | None = None


class GateResult(BaseModel):
    """A gate's verdict on one unit: a record (G1) or a window (G2 to G6)."""

    model_config = ConfigDict(frozen=True)

    gate: str  # G1 to G6
    unit: str  # the record's file name, or the window's folder xmid_<x>
    verdict: Verdict
    metrics: tuple[Metric, ...] = ()
    flags: tuple[Flag, ...] = ()
    kept: Kept = Kept()
    shrank: bool = False  # kept much less than the attempt before, or than the line (rule 6)


class Attempt(BaseModel):
    """One line of the QC log (rule 8): a stage run once on one unit, and what its gate said."""

    model_config = ConfigDict(frozen=True)

    unit: str
    stage: Stage
    attempt: int  # 1 for the first
    parameters: dict[str, Any]  # the stage's overrides on the run's preset, for this attempt
    triggered_by: str  # "initial", "<gate>:<flag>" of the attempt before, or "backtrack"
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal["succeeded", "failed"]
    error: str | None = None
    # What the checks before the stage changed in the values given, and why (the inversion's).
    notes: tuple[str, ...] = ()
    # What the gates said of this attempt's output, by gate: the stage's own (G3 on a picking),
    # and a line-level one judging the same output (G4).
    results: dict[str, GateResult] = {}


class Budgets(BaseModel):
    """Retries, not attempts (rule 4): the first attempt is free."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_gate_and_unit: int = Field(default=2, ge=0, description="Retries of a unit at one gate.")
    per_xmid_of_the_run: int = Field(
        default=2,
        ge=0,
        description="Retries per xmid over the whole run, shared: a few hard xmids may take more.",
    )
