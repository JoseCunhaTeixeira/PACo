"""The QC log of a run: every attempt, appended as one JSON line to qc_log.jsonl in the run
folder, safe against a crash and readable while the run goes on. The state of a unit (its
attempts at each stage, the retries it spent) is read back from the log, never kept elsewhere."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from sigpipe.masw.runs.history import LOG_FILE, downstream, forget, log_lock
from sigpipe.masw.runs.models import RunManifest
from sigpipe.masw.runs.origin import JUDGED

from paco.qc.models import Attempt, GateResult, Stage

# The gates whose retries of a window's inversion the window's own budget bounds: G5, G6, and
# S4's own when an inversion failed.
INVERSION_GATES = ("G5", "G6", "S4")


def append_attempt(run_folder: Path, attempt: Attempt) -> None:
    with log_lock(run_folder), (run_folder / LOG_FILE).open("a") as file:
        file.write(attempt.model_dump_json() + "\n")


# The trigger of a stage the user asked for, for some windows: done afresh, outside the run's
# retry budget (the gates' retries within it still on it).
ASKED = "asked"


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
    forget(run_folder, unit, stage, results=False, folder=folder)
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
    the last line of an attempt is its current one."""
    path = run_folder / LOG_FILE
    if not path.exists():
        return ()
    current: dict[tuple[str, str, int], Attempt] = {}
    for line in path.read_text().splitlines():
        if line:
            attempt = Attempt.model_validate_json(line)
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
    append_attempt(run_folder, judged)
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
    append_attempt(run_folder, noted)
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
        attempt.triggered_by not in ("initial", ASKED, JUDGED)
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
    return Counter(a.unit for a in attempts if a.triggered_by != "initial")


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
