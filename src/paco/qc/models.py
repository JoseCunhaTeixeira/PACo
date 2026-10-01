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
    of: str = ""  # the object it describes: signal, spectrum, image, ... ("": said by its name)
    over: str = ""  # what it covers: "52 of 96 traces, within 63.35 m of the source"


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
    """A gate's verdict on one unit: a record (G1), a window (G2 to G8), or the line (G4, G6
    and G8)."""

    model_config = ConfigDict(frozen=True)

    gate: str  # G1 to G8
    unit: str  # the record's file name, or the window's folder xmid_<x>
    verdict: Verdict
    metrics: tuple[Metric, ...] = ()
    flags: tuple[Flag, ...] = ()
    kept: Kept = Kept()
    shrank: bool = False  # kept much less than the attempt before, or than the line (rule 6)


# Who made an attempt, or logged an event: the agent (its call), a gate (its retry, its
# verdict), or the user (a person's work in PAC, judged).
type Actor = Literal["agent", "gate", "user"]


class MadeBy(BaseModel):
    """What made an attempt (S4): who, the code's versions, and for a call of the agent its
    model, the prompts' version, its conversation and the turn."""

    model_config = ConfigDict(frozen=True)

    actor: Actor
    code: dict[str, str] = {}
    model: str | None = None
    prompts: str | None = None
    conversation: str | None = None
    turn: int | None = None


class Attempt(BaseModel):
    """One line of the QC log (rule 8): a stage run once on one unit, and what its gate said.
    Each state of the attempt is a line of its own (S2): its stage run (`event` "stage"), a
    gate's verdict on it ("verdict"), notes added after it ran ("notes")."""

    model_config = ConfigDict(frozen=True)

    # The line's format (sigpipe's masw.runs.history.LOG_VERSION; a line without one, 1), its
    # kind, and who logged it.
    version: int = 1
    event: Literal["stage", "verdict", "notes"] = "stage"
    actor: Actor | None = None  # None: a version 1 line

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
    # The retries on the run's budget that the attempts this one replaced had spent: a stage
    # started afresh forgets its window's earlier attempts, not what they cost (rule 4).
    forgotten: int = 0
    # The parameters of the attempt this one replaced, forgotten: what it changed is read
    # against them.
    replaced: dict[str, Any] = {}
    made_by: MadeBy | None = None  # None: a version 1 line


class Budgets(BaseModel):
    """Retries, not attempts (rule 4): the first attempt is free."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_gate_and_unit: int = Field(default=2, ge=0, description="Retries of a unit at one gate.")
    per_xmid_of_the_run: int = Field(
        default=2,
        ge=0,
        description="Retries per xmid over the whole run, shared: a few hard xmids may take more.",
    )
    inversion_per_window: int = Field(
        default=6,
        ge=0,
        description="Retries of a window's inversion the gates ask (G5, G6, or a failed run), "
        "whichever asks: each window's own, outside the run's.",
    )
