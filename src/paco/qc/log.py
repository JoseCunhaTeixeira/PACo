"""The QC log of a run: every attempt, appended as one JSON line to qc_log.jsonl in the run
folder, safe against a crash and readable while the run goes on. The state of a unit (its
attempts at each stage, the retries it spent) is read back from the log, never kept elsewhere."""

import functools
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, NamedTuple

from sigpipe.masw.runs.history import (
    LOG_FILE,
    LOG_VERSION,
    downstream,
    forget,
    log_entries,
    log_lock,
)
from sigpipe.masw.runs.models import RunManifest
from sigpipe.masw.runs.origin import JUDGED
from sigpipe.masw.runs.processing import package_versions

from paco import logs
from paco.qc.models import Actor, Attempt, GateResult, MadeBy, Stage

# The gates whose retries of a window's inversion the window's own budget bounds: G5, G6, and
# S4's own when an inversion failed.
INVERSION_GATES = ("G5", "G6", "S4")


def append_attempt(
    run_folder: Path, attempt: Attempt, event: Literal["stage", "verdict", "notes"] = "stage"
) -> None:
    """`attempt` appended to the run's log as one of its states (`event`), in the log's current
    version: who logged it, and what made the attempt (S2, S4)."""
    made_by = attempt.made_by or made_by_now(_maker(attempt.triggered_by))
    line = attempt.model_copy(
        update={
            "version": LOG_VERSION,
            "event": event,
            "actor": "gate" if event == "verdict" else made_by.actor,
            "made_by": made_by,
        }
    )
    with log_lock(run_folder), (run_folder / LOG_FILE).open("a") as file:
        file.write(line.model_dump_json() + "\n")


def made_by_now(actor: Actor) -> MadeBy:
    """What makes something now, by `actor`: the code's versions, and the call's model, prompts'
    version, conversation and turn when an agent's call runs (paco.logs)."""
    return MadeBy(
        actor=actor,
        code=_code(),
        model=logs.MODEL.get(),
        prompts=logs.PROMPTS.get(),
        conversation=logs.CONVERSATION.get(),
        turn=logs.TURN.get(),
    )


@functools.cache
def _code() -> dict[str, str]:
    return package_versions(("paco",))


def _maker(triggered_by: str) -> Actor:
    """Who made an attempt triggered so: a gate's retry, a person's work judged, or the agent's
    call (the first attempt, one asked for, a step back)."""
    if triggered_by == JUDGED:
        return "user"
    return "agent" if starts_afresh(triggered_by) or triggered_by in RULES else "gate"


# The trigger of a stage the user asked for, for some windows: done afresh, outside the run's
# retry budget (the gates' retries within it still on it).
ASKED = "asked"
# The trigger of the records preprocessed again with the mute the mute trial kept: a rule of the
# processing (qc.muting), outside the retry budgets.
MUTE_TRIAL = "mute trial"
# The trigger of the records and images made again with a change of the line's settings (one the
# line loop kept, qc.line_loop, or a redo of the line's): every record and window alike, the
# change logged on the line; outside the retry budgets.
LINE_CHANGE = "line change"
# The processing's own rules: a choice of the line's settings, no unit's retry.
RULES = (MUTE_TRIAL, LINE_CHANGE)


def starts_afresh(triggered_by: str) -> bool:
    """Whether an attempt triggered so starts its unit's stage afresh: the first one, or one the
    agent went back for (a backtrack) or the user asked for; a retry a gate asked for goes on
    from the attempts before it."""
    return triggered_by in ("initial", "backtrack", ASKED)


class Forgotten(NamedTuple):
    """What an attempt starting afresh carries of those it replaced."""

    spent: int  # the retries they had spent on the run's budget
    parameters: dict[str, Any]  # the last one's at the stage: what the new one changes


def forget_history(
    run_folder: Path, unit: str, stage: Stage, folder: Path | None = None
) -> Forgotten:
    """`unit`'s history at `stage` and the stages after it that use its results forgotten, as its
    stage starts afresh: its earlier attempts' lines in the log and their archived results
    (sigpipe's `forget`; the results now in place kept). Returns what the attempt starting
    afresh carries of them."""
    stages = set(downstream(stage))
    gone = [a for a in read_attempts(run_folder) if a.unit == unit and a.stage in stages]
    spent = sum(_on_run_budget(a) + a.forgotten for a in gone)
    last = latest(gone, unit, stage)
    forget(run_folder, unit, stage, results=False, folder=folder, actor="agent")
    return Forgotten(spent, dict(last.parameters) if last is not None else {})


def afresh(run_folder: Path, attempt: Attempt, folder: Path | None = None) -> Attempt:
    """`attempt`, starting its unit's stage afresh: the unit's earlier attempts there forgotten
    (`forget_history`), what they cost and the last one's parameters carried."""
    gone = forget_history(run_folder, attempt.unit, attempt.stage, folder)
    return attempt.model_copy(
        update={"attempt": 1, "forgotten": gone.spent, "replaced": gone.parameters}
    )


def read_attempts(run_folder: Path) -> tuple[Attempt, ...]:
    """Every attempt of the run, in the order they were first logged; none before the first. A
    gate that judges an attempt after its stage ran appends the attempt again with its result:
    the last line of an attempt is its current one. Those a reset left behind are not read
    (sigpipe's `log_entries`), nor the log's other events."""
    current: dict[tuple[str, str, int], Attempt] = {}
    for entry in log_entries(run_folder):
        if "attempt" in entry:
            attempt = Attempt.model_validate(entry)
            current[attempt.unit, attempt.stage, attempt.attempt] = attempt
    return tuple(current.values())


def record_result(
    run_folder: Path, unit: str, stage: Stage, attempt: int, result: GateResult
) -> Attempt:
    """Log the gate's result on an attempt: the attempt appended again, with the result added
    to its others (by gate)."""
    match = [
        a
        for a in read_attempts(run_folder)
        if (a.unit, a.stage, a.attempt) == (unit, stage, attempt)
    ]
    if not match:
        raise ValueError(f"No attempt {attempt} of {stage} for {unit} in the log of {run_folder}")
    judged = match[0].model_copy(update={"results": {**match[0].results, result.gate: result}})
    append_attempt(run_folder, judged, "verdict")
    return judged


def record_notes(
    run_folder: Path, unit: str, stage: Stage, attempt: int, notes: tuple[str, ...]
) -> Attempt:
    """Log what was changed around an attempt after it ran (the traces G1 had the windows leave
    out): the attempt appended again, with `notes` added to its own."""
    match = [
        a
        for a in read_attempts(run_folder)
        if (a.unit, a.stage, a.attempt) == (unit, stage, attempt)
    ]
    if not match:
        raise ValueError(f"No attempt {attempt} of {stage} for {unit} in the log of {run_folder}")
    noted = match[0].model_copy(update={"notes": (*match[0].notes, *notes)})
    append_attempt(run_folder, noted, "notes")
    return noted


def attempts_of(attempts: Iterable[Attempt], unit: str, stage: Stage) -> tuple[Attempt, ...]:
    return tuple(a for a in attempts if a.unit == unit and a.stage == stage)


def latest(attempts: Iterable[Attempt], unit: str, stage: Stage) -> Attempt | None:
    """The unit's last attempt at `stage`, or None."""
    own = attempts_of(attempts, unit, stage)
    return own[-1] if own else None


def retries_at_gate(attempts: Iterable[Attempt], unit: str, gate: str) -> int:
    """Retries the unit spent at `gate`: attempts that gate triggered."""
    return sum(1 for a in attempts if a.unit == unit and a.triggered_by.startswith(f"{gate}:"))


def retries_in_run(attempts: Iterable[Attempt]) -> int:
    """Retries spent on the run's budget, the windows': every gate's and the agent's, but G1's,
    which each record's own budget bounds, and the inversion's gates', which each window's
    does; with those of the attempts a stage started afresh forgot."""
    return sum(_on_run_budget(a) + a.forgotten for a in attempts)


def _on_run_budget(attempt: Attempt) -> int:
    """1 when `attempt` is a retry the run's budget pays for, else 0: not the first, nor one the
    user asked for, nor a judgement of a curve as it is (JUDGED)."""
    return int(
        attempt.triggered_by not in ("initial", ASKED, JUDGED, *RULES)
        and not attempt.triggered_by.startswith("G1:")
        and not _by_inversion_gate(attempt)
    )


def retries_of_inversion(attempts: Iterable[Attempt], unit: str | None = None) -> int:
    """Retries of the inversion the gates asked, on the windows' own budgets: `unit`'s, or
    every window's."""
    return sum(1 for a in attempts if (unit is None or a.unit == unit) and _by_inversion_gate(a))


def _by_inversion_gate(attempt: Attempt) -> bool:
    return attempt.stage == "inversion" and attempt.triggered_by.split(":")[0] in INVERSION_GATES


def retries_by_unit(attempts: Iterable[Attempt]) -> Counter[str]:
    return Counter(a.unit for a in attempts if a.triggered_by not in ("initial", *RULES))


def ensure_initial_attempts(run_folder: Path, manifest: RunManifest) -> tuple[Attempt, ...]:
    """The log with the run's first attempts in it: one per record (preprocessing) and per
    window (phase shift), from run.json, for the units the log does not hold yet. Returns
    every attempt."""
    attempts = read_attempts(run_folder)
    logged = {(attempt.unit, attempt.stage) for attempt in attempts}
    new: list[Attempt] = []
    for record in manifest.records:
        if (record.name, "preprocessing") not in logged:
            new.append(
                Attempt(
                    unit=record.name,
                    stage="preprocessing",
                    attempt=1,
                    parameters={},
                    triggered_by="initial",
                    started_at=manifest.started_at,
                    finished_at=manifest.finished_at,
                    status=record.status,
                    error=record.error,
                )
            )
    for window in manifest.windows:
        if (window.folder, "phase_shift") not in logged:
            new.append(
                Attempt(
                    unit=window.folder,
                    stage="phase_shift",
                    attempt=1,
                    parameters={},
                    triggered_by="initial",
                    started_at=manifest.started_at,
                    finished_at=manifest.finished_at,
                    status=window.status,
                    error=window.error,
                )
            )
    for attempt in new:
        append_attempt(run_folder, attempt)
    return (*attempts, *new)
