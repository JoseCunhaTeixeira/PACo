"""The QC log of a run: every attempt, appended as one JSON line to qc_log.jsonl in the run
folder, safe against a crash and readable while the run goes on. The state of a unit (its
attempts at each stage, the retries it spent) is read back from the log, never kept elsewhere."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from paco.qc.models import Attempt, GateResult, Stage
from paco.runs.models import RunManifest

LOG_FILE = "qc_log.jsonl"


def append_attempt(run_folder: Path, attempt: Attempt) -> None:
    with (run_folder / LOG_FILE).open("a") as file:
        file.write(attempt.model_dump_json() + "\n")


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
    """Retries spent on the whole run, at every gate."""
    return sum(1 for a in attempts if a.triggered_by != "initial")


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
