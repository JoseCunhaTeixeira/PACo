"""decide on models: after the freeze, the model's decision on each
window's Vs model, accepted, rejected or pending, with its reason, as decide does on the curves
before it. Code refuses: accepting a model G5 did not pass, or one with a blocking flag (G5's or
G6's); accepting a model whose notes (G5's and G6's, and G6's suggestions, which a model may keep:
a jump the curves do not show is non-uniqueness) have no reason; rejecting a window before its
model's fixable flags were tried (a later inversion of it, or the reason naming the flag), unless
its retries are spent or G5 rejected its model. A decision holds for the model it was made on: a
later inversion makes the window pending again. The study keeps every decision
(model_decisions.jsonl), a window's last its current one."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from paco.autonomous.sensors.flags import flag_class
from paco.autonomous.tools.decide import Decided, Status
from paco.autonomous.tools.invert import Inverted, inversions_of
from paco.autonomous.tools.pick import manifest_of
from paco.autonomous.tools.trials import CURVES, Study
from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt, Flag, GateResult

MODEL_DECISIONS = "model_decisions.jsonl"  # in the study's folder


class ModelDecision(BaseModel):
    """A window's decision on its model: its status and reason, and the model it was made on (its
    inversion attempt's end)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    unit: str
    status: Status
    reason: str | None
    inverted_at: datetime | None
    at: datetime


def current(study: Study, run_folder: Path, run_id: str) -> dict[str, ModelDecision]:
    """The decisions that hold on run `run_id`'s models: each window's last, while its model is
    the one it was made on; a pending one is no decision."""
    attempts = read_attempts(run_folder)
    last: dict[str, ModelDecision] = {}
    path = study.folder / MODEL_DECISIONS
    lines = path.read_text().splitlines() if path.exists() else []
    for one in (ModelDecision.model_validate_json(line) for line in lines if line):
        if one.run_id == run_id:
            last[one.unit] = one
    return {
        unit: one
        for unit, one in last.items()
        if one.status != "pending" and one.inverted_at == _inverted_at(attempts, unit)
    }


def statuses(study: Study, run_folder: Path, run_id: str) -> dict[str, Status]:
    """Each decided window's status on run `run_id`'s models."""
    return {unit: one.status for unit, one in current(study, run_folder, run_id).items()}


def decide(
    study: Study,
    run_folder: Path,
    units: Sequence[str],
    status: Status,
    reason: str | None,
    most_retries: int | None = None,
) -> Decided:
    """`units`' models decided `status`, with `reason`; refused whole when one of them breaks a
    rule, each window's problem said. `most_retries`: each window's budget of inversion retries,
    past which a model may be rejected without more."""
    if not (study.folder / CURVES).exists():
        raise ValueError("No model before the freeze: decide the windows' curves first.")
    reason = reason.strip() if reason is not None and reason.strip() else None
    if reason is not None and "\n" in reason:
        raise ValueError("A reason is one line.")
    manifest = manifest_of(run_folder)
    attempts = read_attempts(run_folder)
    done = inversions_of(study, manifest.run_id)
    problems: dict[str, str] = {}
    notes: dict[str, list[str]] = {}
    tried: dict[str, list[str]] = {}
    if status == "accepted":
        for unit in units:
            problem, carried = acceptable(attempts, unit)
            if problem is not None:
                problems[unit] = problem
            for name in carried:
                notes.setdefault(name, []).append(unit)
        if notes and reason is None and not problems:
            said = "; ".join(f"{name} ({len(found)})" for name, found in notes.items())
            raise ValueError(
                f"Nothing decided: accepting models with notes needs a reason, one line that "
                f"covers them: {said}."
            )
    elif status == "rejected":
        if reason is None:
            raise ValueError("Nothing decided: a rejection needs its reason.")
        for unit in units:
            own = [one for one in done if one.unit == unit]
            tried[unit] = [_tried(one) for one in own]
            problem = _not_tried(attempts, unit, own, reason, most_retries)
            if problem is not None:
                problems[unit] = problem
    if problems:
        rows = "\n".join(
            f"- {unit.removeprefix('xmid_')}: {said}" for unit, said in problems.items()
        )
        raise ValueError(
            f"Nothing decided, {len(problems)} of the models cannot be {status}:\n{rows}"
        )
    now = datetime.now(UTC)
    with (study.folder / MODEL_DECISIONS).open("a") as file:
        for unit in units:
            decision = ModelDecision(
                run_id=manifest.run_id,
                unit=unit,
                status=status,
                reason=reason,
                inverted_at=_inverted_at(attempts, unit),
                at=now,
            )
            file.write(decision.model_dump_json() + "\n")
    return Decided(
        tuple(units),
        status,
        reason,
        notes,
        tried,
        statuses(study, run_folder, manifest.run_id),
    )


def acceptable(attempts: tuple[Attempt, ...], unit: str) -> tuple[str | None, list[str]]:
    """Why the window's model cannot be accepted (none: it can), and what accepting it needs a
    reason for: G5's and G6's notes, and G6's suggestions."""
    inverted = latest(attempts, unit, "inversion")
    g5 = inverted.results.get("G5") if inverted is not None else None
    g6 = inverted.results.get("G6") if inverted is not None else None
    if inverted is None or inverted.status != "succeeded" or g5 is None:
        return "it has no model: invert it, or reject it", []
    blocking = [flag.name for flag in _flags(g5, g6) if flag_class(flag) == "blocking"]
    if blocking:
        return f"blocking flag {', '.join(blocking)}: it cannot be accepted", []
    if g5.verdict != "pass":
        fixable = [flag.name for flag in g5.flags if flag_class(flag) == "fixable"]
        return (
            f"G5 did not pass its model ({g5.verdict}: {', '.join(fixable)}): invert it again "
            "within its budget, or reject it",
            [],
        )
    kept = [
        flag.name
        for flag in _flags(g5, g6)
        if flag_class(flag) == "note" or (g6 is not None and flag in g6.flags)
    ]
    return None, kept


def _not_tried(
    attempts: tuple[Attempt, ...],
    unit: str,
    own: Sequence[Inverted],
    reason: str,
    most_retries: int | None,
) -> str | None:
    """What the window still owes before its model's rejection: a later inversion after each
    fixable flag, or the reason naming it; none when tried, when G5 rejected its model or found
    none, or when its retries are spent."""
    inverted = latest(attempts, unit, "inversion")
    g5 = inverted.results.get("G5") if inverted is not None else None
    if g5 is None or g5.verdict == "reject" or not own:
        return None
    if most_retries is not None and len(own) >= most_retries + 1:
        return None
    untried: list[str] = []
    for index, one in enumerate(own):
        retried = len(own) > index + 1
        for name, kind in one.flags.items():
            if kind == "fixable" and not retried and name not in untried and name not in reason:
                untried.append(name)
    if not untried:
        return None
    return (
        f"not tried yet: owes an inversion after its fixable {', '.join(untried)} (invert it "
        f"again, a group of its own), or the reason naming "
        f"{'it' if len(untried) == 1 else 'them'}: why none applies"
    )


def _inverted_at(attempts: tuple[Attempt, ...], unit: str) -> datetime | None:
    inverted = latest(attempts, unit, "inversion")
    return inverted.finished_at if inverted is not None else None


def _flags(*results: GateResult | None) -> list[Flag]:
    return [flag for result in results if result is not None for flag in result.flags]


def _tried(one: Inverted) -> str:
    """An inversion in a few words: its number, its effort and what G5 said."""
    iterations = one.parameters.get("n_iterations")
    effort = f"{iterations:,} iterations" if isinstance(iterations, int) else "not run"
    return f"#{one.number} {effort} ({one.verdict or one.error or 'no verdict'})"
