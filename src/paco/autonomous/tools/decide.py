"""decide: the model's decision on each window of the committed
line, accepted, rejected or pending, with its reason. Code refuses: accepting a window
whose curve did not pass G3, carries a blocking flag, or is off its neighbours (G4); accepting a
curve with notes without a reason; and rejecting a window before it was tried hard. Tried
hard means the line's automatic pick, a pick of the model's after it when it carried fixable flags
(or a reason naming those flags, saying why none applies), a pick guided by its neighbours when one
side has its two passing G3, and a lasso when the model sees the images. The model's own picks owe
nothing more: G3 judges each afresh, and their new flags would make the debt endless. A window whose
image G2 rejected has nothing to pick: its rejection needs its reason alone.

A decision holds for the curve it was made on: a later pick of the window, or a commit of the line,
makes the window pending again. The study keeps every decision (decisions.jsonl), a window's last
its current one. A call is decided whole, or refused whole, saying each window's problem."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import RunManifest

from paco.autonomous.sensors.checks import shot_note
from paco.autonomous.sensors.flags import flag_class
from paco.autonomous.tools.pick import Picked, manifest_of, neighbours, passing_curves, picks_of
from paco.autonomous.tools.trials import Study, check_open
from paco.qc.config import QCConfig
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt, Flag, GateResult
from paco.qc.report import stretches, xmid_of

DECISIONS = "decisions.jsonl"  # in the study's folder
type Status = Literal["accepted", "rejected", "pending"]


class Decision(BaseModel):
    """A window's decision: its status and reason, and the image and curve it was made on (their
    attempts' ends)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    unit: str
    status: Status
    reason: str | None
    imaged_at: datetime | None
    picked_at: datetime | None
    at: datetime


@dataclass(frozen=True)
class Decided:
    """A decide call: its windows, status and reason, the notes the reason covers (by flag, the
    windows carrying it), what each rejected window had tried, and every window's status now."""

    units: tuple[str, ...]
    status: Status
    reason: str | None
    notes: dict[str, list[str]]
    tried: dict[str, list[str]]
    statuses: dict[str, Status]


def current(study: Study, run_folder: Path, run_id: str) -> dict[str, Decision]:
    """The decisions that hold on run `run_id`: each window's last, while its image and curve are
    those it was made on; a pending one is no decision."""
    attempts = read_attempts(run_folder)
    last: dict[str, Decision] = {}
    path = study.folder / DECISIONS
    lines = path.read_text().splitlines() if path.exists() else []
    for one in (Decision.model_validate_json(line) for line in lines if line):
        if one.run_id == run_id:
            last[one.unit] = one
    return {
        unit: one
        for unit, one in last.items()
        if one.status != "pending" and (one.imaged_at, one.picked_at) == _made_on(attempts, unit)
    }


def statuses(study: Study, run_folder: Path, run_id: str) -> dict[str, Status]:
    """Each decided window's status on run `run_id`."""
    return {unit: one.status for unit, one in current(study, run_folder, run_id).items()}


def decide(
    study: Study,
    run_folder: Path,
    units: Sequence[str],
    status: Status,
    reason: str | None,
    vision: bool,
    config: QCConfig,
) -> Decided:
    """`units` of the committed line in `run_folder` decided `status`, with `reason`; refused
    whole when one of them breaks a rule, each window's problem said. `vision`: whether the model
    sees the images (a lasso is then owed before a rejection)."""
    check_open(study)
    reason = reason.strip() if reason is not None and reason.strip() else None
    if reason is not None and "\n" in reason:
        raise ValueError("A reason is one line.")
    manifest = manifest_of(run_folder)
    attempts = read_attempts(run_folder)
    problems: dict[str, str] = {}
    notes: dict[str, list[str]] = {}
    tried: dict[str, list[str]] = {}
    if status == "accepted":
        for unit in units:
            problem, carried = acceptable(run_folder, attempts, unit)
            if problem is not None:
                problems[unit] = problem
            for name in carried:
                notes.setdefault(name, []).append(unit)
        if notes and reason is None and not problems:
            said = "; ".join(f"{name} at {_where(found)}" for name, found in notes.items())
            raise ValueError(
                f"Nothing decided: accepting curves with notes needs a reason, one line that "
                f"covers them: {said}."
            )
    elif status == "rejected":
        if reason is None:
            raise ValueError("Nothing decided: a rejection needs its reason.")
        picks = picks_of(study, manifest.run_id)
        passing = passing_curves(run_folder, manifest, attempts)
        for unit in units:
            own = [one for one in picks if one.unit == unit]
            tried[unit] = [_tried(one) for one in own]
            problem = _not_tried_hard(
                unit, attempts, own, reason, vision, manifest, passing, config
            )
            if problem is not None:
                problems[unit] = problem
    if problems:
        rows = "\n".join(
            f"- {unit.removeprefix('xmid_')}: {said}" for unit, said in problems.items()
        )
        raise ValueError(
            f"Nothing decided, {len(problems)} of the windows cannot be {status}:\n{rows}"
        )
    now = datetime.now(UTC)
    with (study.folder / DECISIONS).open("a") as file:
        for unit in units:
            imaged_at, picked_at = _made_on(attempts, unit)
            decision = Decision(
                run_id=manifest.run_id,
                unit=unit,
                status=status,
                reason=reason,
                imaged_at=imaged_at,
                picked_at=picked_at,
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


def acceptable(
    run_folder: Path, attempts: tuple[Attempt, ...], unit: str
) -> tuple[str | None, list[str]]:
    """Why the window's curve cannot be accepted (none: it can), and the notes it carries, G3's
    and G4's, which accepting it needs a reason for."""
    imaged = latest(attempts, unit, "phase_shift")
    picked = latest(attempts, unit, "picking")
    g2 = imaged.results.get("G2") if imaged is not None else None
    g3 = picked.results.get("G3") if picked is not None else None
    g4 = picked.results.get("G4") if picked is not None else None
    if g3 is None or saved_m0(run_folder / unit / CURVES_FILE) is None:
        return "it has no curve: pick it, or reject it", []
    blocking = [flag.name for flag in _flags(g2, g3, g4) if flag_class(flag) == "blocking"]
    if blocking:
        return (
            f"blocking flag {', '.join(blocking)}: it cannot be accepted until a later pick "
            "clears it",
            [],
        )
    if g3.verdict != "pass":
        fixable = [flag.name for flag in g3.flags if flag_class(flag) == "fixable"]
        return (
            f"G3 did not pass its curve ({g3.verdict}: {', '.join(fixable)}): pick it again, or "
            "reject it",
            [],
        )
    if g4 is not None and g4.verdict != "pass":
        return (
            f"G4 finds its curve off its neighbours ({', '.join(f.name for f in g4.flags)}): "
            "pick it along them (how=guide, guide=neighbours)",
            [],
        )
    mode = str(manifest_of(run_folder).preset.mode)
    return None, [
        flag.name
        for flag in _flags(g3, g4)
        if flag_class(flag) == "note" and not shot_note(flag.name, mode)
    ]


def describe_decided(decided: Decided) -> str:
    """A decide call as the model reads it: what was decided, the notes the reason covers, what
    each rejected window had tried, and every window's status now."""
    count = len(decided.units)
    head = (
        f"{decided.status.capitalize()} {count} window{'s' if count > 1 else ''} at "
        f"{_where(decided.units)}"
    )
    lines = [head + (f": {decided.reason}" if decided.reason else ".")]
    if decided.notes:
        said = ", ".join(f"{name} ({len(found)})" for name, found in decided.notes.items())
        lines.append(f"The reason covers their notes: {said}.")
    for unit, tried in decided.tried.items():
        lines.append(f"- {unit.removeprefix('xmid_')} had: {', '.join(tried) or 'no pick'}.")
    counted = {
        status: sum(1 for one in decided.statuses.values() if one == status)
        for status in ("accepted", "rejected")
    }
    lines.append(
        f"Now {counted['accepted']} accepted, {counted['rejected']} rejected; the others pending."
    )
    return "\n".join(lines)


def _not_tried_hard(
    unit: str,
    attempts: tuple[Attempt, ...],
    own: Sequence[Picked],
    reason: str,
    vision: bool,
    manifest: RunManifest,
    passing: Sequence[Series],
    config: QCConfig,
) -> str | None:
    """What the window still owes before its rejection; none when it was tried hard, or
    has nothing to pick (no image, or one G2 rejected)."""
    imaged = latest(attempts, unit, "phase_shift")
    g2 = imaged.results.get("G2") if imaged is not None else None
    if (
        imaged is None
        or imaged.status != "succeeded"
        or (g2 is not None and g2.verdict == "reject")
    ):
        return None
    owed: list[str] = []
    if not any(one.recipe.how == "line" for one in own):
        owed.append("the line's automatic pick (commit)")
    untried = owed_flags(own, reason)
    if untried:
        owed.append(
            f"a pick after the line's, which carried the fixable {', '.join(untried)}, or the "
            f"reason naming {'it' if len(untried) == 1 else 'them'}: why none applies"
        )
    left, right = neighbours(unit, manifest, passing, config.profile)
    full = max(1, config.profile.neighbours // 2)
    if (len(left) >= full or len(right) >= full) and not any(o.recipe.how == "guide" for o in own):
        owed.append(
            "a pick guided by its neighbours (how=guide, guide=neighbours): "
            f"{full} on one side pass G3"
        )
    if vision and not any(one.recipe.how == "lasso" for one in own):
        owed.append("a lasso around its ridge (how=lasso), since you see the images")
    return f"not tried hard yet: owes {'; '.join(owed)}" if owed else None


def owed_flags(own: Sequence[Picked], reason: str) -> list[str]:
    """The fixable flags a window's rejection still owes a pick for: those the line's latest pick
    carried, until a pick of the model's follows it (not a deletion), less those `reason` names.
    The model's own picks owe nothing more: G3 judges each afresh, and their new flags would make
    the debt endless."""
    lines = [index for index, one in enumerate(own) if one.recipe.how == "line"]
    if not lines or any(o.recipe.how not in ("line", "delete") for o in own[lines[-1] + 1 :]):
        return []
    return [
        name
        for name, kind in own[lines[-1]].flags.items()
        if kind == "fixable" and not _named(name, reason)
    ]


def _named(flag: str, reason: str) -> bool:
    """Whether `reason` names `flag`, by its name or in words (no_ridge, no ridge)."""
    return flag in reason or flag.replace("_", " ") in reason.lower()


def _made_on(attempts: tuple[Attempt, ...], unit: str) -> tuple[datetime | None, datetime | None]:
    """The window's image and curve as they are now: their latest attempts' ends."""
    imaged = latest(attempts, unit, "phase_shift")
    picked = latest(attempts, unit, "picking")
    return (
        imaged.finished_at if imaged is not None else None,
        picked.finished_at if picked is not None else None,
    )


def _flags(*results: GateResult | None) -> list[Flag]:
    return [flag for result in results if result is not None for flag in result.flags]


def _tried(one: Picked) -> str:
    """A pick in a few words: how, and what G3 said."""
    how = one.recipe.how
    if how == "guide" and one.recipe.guide == "neighbours":
        how = "guide (neighbours)"
    if one.error is not None:
        return f"{how} (not picked)"
    if how == "delete":
        return how
    return f"{how} ({one.verdict})"


def _where(units: Sequence[str] | Mapping[str, object]) -> str:
    xmids = [x for unit in units if (x := xmid_of(unit)) is not None]
    return stretches(xmids)
