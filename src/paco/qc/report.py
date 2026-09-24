"""The QC report of a run (rule 8), built from the QC log: per unit, the final parameters, the
attempts, the verdict of each gate, the flags and reasons, and what the curve that went into
the inversion kept. Written as qc_report.json; the agent reads a short summary, grouped by
stretches of xmids, never one line per xmid."""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from paco.qc.log import read_attempts
from paco.qc.models import Action, Attempt, Budgets, Flag, GateResult, Kept, Stage, Verdict

REPORT_FILE = "qc_report.json"
_STRETCH_MARGIN = 1.5  # two xmids further apart than 1.5 steps are not consecutive
_ERROR_LENGTH = 200  # of an error in the summary
_NUMBER = re.compile(r"\d+(\.\d+)?")


class UnitReport(BaseModel):
    """One record or window, at the end of the run."""

    model_config = ConfigDict(frozen=True)

    unit: str
    xmid: float | None  # None for a record
    verdicts: dict[str, Verdict]  # per gate, the latest
    flags: dict[str, tuple[Flag, ...]]  # per gate, the latest
    attempts: int  # over every stage
    parameters: dict[Stage, dict[str, Any]]  # the final overrides of each stage run
    rejected_for: tuple[str, ...]  # the messages of the flags of a reject
    curve: Kept | None  # what the latest curve kept: band, wavelength range, points
    notes: dict[Stage, tuple[str, ...]] = {}  # what the checks before each stage changed
    failed: dict[Stage, str] = {}  # the error of each stage whose latest attempt failed


class StageResult(BaseModel):
    """What a stage tool returns to the agent: the run, what the gates found (their summary:
    counts, what they fixed or changed, flags with their suggested change), and what to do
    next."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    # The settings the gates and the checks changed, in words: to report, every one.
    changed: tuple[str, ...] = ()
    summary: str
    next: str
    job_id: str | None = None  # an inversion done again runs as a job


class QCReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    budgets: Budgets
    n_xmids: int
    retries: int  # spent over the run
    # Which units each trigger ("<gate>:<flag>", "backtrack") sent back, in the order logged,
    # and what it changed: the first unit's change over its attempt before, and whether every
    # unit had the same.
    retried: dict[str, tuple[str, ...]] = {}
    changed: dict[str, tuple[Stage, dict[str, Any], bool]] = {}
    # For the same first change: the values it replaced, at the same keys.
    replaced: dict[str, dict[str, Any]] = {}
    units: tuple[UnitReport, ...]  # in the order of their first attempt
    counts: dict[str, dict[Verdict, int]]  # per gate

    @property
    def rejected(self) -> tuple[UnitReport, ...]:
        return tuple(unit for unit in self.units if "reject" in unit.verdicts.values())


def build_report(run_id: str, run_folder: Path, budgets: Budgets, n_xmids: int) -> QCReport:
    attempts = read_attempts(run_folder)
    units = tuple(_unit_report(unit, own) for unit, own in _by_unit(attempts).items())
    counts: dict[str, Counter[Verdict]] = defaultdict(Counter)
    for unit in units:
        for gate, verdict in unit.verdicts.items():
            counts[gate][verdict] += 1
    retried: dict[str, list[str]] = defaultdict(list)
    changes: dict[str, list[tuple[Stage, dict[str, Any]]]] = defaultdict(list)
    replaced: dict[str, dict[str, Any]] = {}
    before: dict[tuple[str, Stage], dict[str, Any]] = {}
    for attempt in attempts:
        key = (attempt.unit, attempt.stage)
        if attempt.triggered_by != "initial":
            if attempt.unit not in retried[attempt.triggered_by]:
                retried[attempt.triggered_by].append(attempt.unit)
            change = _changes(before.get(key, {}), attempt.parameters)
            changes[attempt.triggered_by].append((attempt.stage, change))
            replaced.setdefault(attempt.triggered_by, _at(before.get(key, {}), change))
        before[key] = attempt.parameters
    return QCReport(
        run_id=run_id,
        budgets=budgets,
        n_xmids=n_xmids,
        retries=sum(1 for attempt in attempts if attempt.triggered_by != "initial"),
        retried={trigger: tuple(units) for trigger, units in retried.items()},
        changed={
            trigger: (made[0][0], made[0][1], all(change == made[0][1] for _, change in made))
            for trigger, made in changes.items()
        },
        replaced=replaced,
        units=units,
        counts={gate: dict(count) for gate, count in counts.items()},
    )


def write_report(report: QCReport, run_folder: Path) -> Path:
    path = run_folder / REPORT_FILE
    path.write_text(report.model_dump_json(indent=2))
    return path


def read_report(run_folder: Path) -> QCReport:
    return QCReport.model_validate_json((run_folder / REPORT_FILE).read_text())


def summarize_report(report: QCReport, gates: Sequence[str] | None = None) -> str:
    """What the agent reads: counts per gate, the retries, then each flag with the units raising
    it, as stretches of xmids, and the change it suggests; only `gates`' when given."""
    if gates is not None:
        report = _only(report, set(gates))
    lines = [
        f"{gate}: " + ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
        for gate, counts in report.counts.items()
    ]
    budget = report.budgets.per_xmid_of_the_run * report.n_xmids
    lines.append(f"Retries: {report.retries} of {budget}.")
    step = line_step(unit.xmid for unit in report.units if unit.xmid is not None)
    by_unit = {unit.unit: unit for unit in report.units}
    for trigger, names in report.retried.items():
        gate = trigger.split(":")[0]
        now = Counter(
            by_unit[name].verdicts.get(gate, "no verdict") for name in names if name in by_unit
        )
        how = ""
        if trigger in report.changed:
            stage, change, alike = report.changed[trigger]
            if change:
                shown = json.dumps(change, separators=(",", ":"))
                how = f" with {stage} {shown}" + ("" if alike else " (each its own)")
        lines.append(
            f"Retried {trigger}, {_where(names, step)}{how}: now "
            + ", ".join(f"{count} {verdict}" for verdict, count in sorted(now.items()))
            + "."
        )
    # Notes alike but for their numbers (each window's own velocities) go together, with the
    # first one's as an example; different notes each get a line.
    noted: dict[tuple[Stage, str], list[tuple[str, str]]] = defaultdict(list)
    for unit in report.units:
        for stage, notes in unit.notes.items():
            text = " ".join(notes)
            noted[stage, _NUMBER.sub("#", text)].append((unit.unit, text))
    for (stage, _), members in noted.items():
        where = _where([name for name, _ in members], step)
        suffix = " (each its own values)" if len({text for _, text in members}) > 1 else ""
        lines.append(f"Changes at {stage}, {where}: {members[0][1]}{suffix}")
    broken: dict[Stage, list[UnitReport]] = defaultdict(list)
    for unit in report.units:
        for stage in unit.failed:
            broken[stage].append(unit)
    for stage, units in broken.items():
        xmids = [unit.xmid for unit in units if unit.xmid is not None]
        names = [unit.unit for unit in units if unit.xmid is None]
        where = ", ".join(
            part for part in (stretches(xmids, step) if xmids else "", *names) if part
        )
        lines.append(f"Failed {stage}, {where}: {units[0].failed[stage][:_ERROR_LENGTH]}")
    windows: dict[tuple[str, str], list[float]] = defaultdict(list)
    records: dict[tuple[str, str], list[str]] = defaultdict(list)
    examples: dict[tuple[str, str], Flag] = {}
    for unit in report.units:
        for gate, flags in unit.flags.items():
            for flag in flags:
                key = (gate, flag.name)
                examples.setdefault(key, flag)
                if unit.xmid is None:
                    records[key].append(unit.unit)
                else:
                    windows[key].append(unit.xmid)
    for key, flag in examples.items():
        where = ", ".join(
            part
            for part in (
                stretches(windows[key], step) if windows[key] else "",
                ", ".join(records[key]),
            )
            if part
        )
        lines.append(f"{key[0]} {flag.name}, {where}: {flag.message} -> {describe(flag.action)}")
    return "\n".join(lines)


def describe(action: Action) -> str:
    """An action in a few words, with its overrides as the tool takes them."""
    match action.kind:
        case "override":
            return f"{action.stage} {json.dumps(action.overrides, separators=(',', ':'))}"
        case "exclude_traces":
            return f"exclude traces {list(action.traces)} of {action.record}"
        case "exclude_record":
            return f"exclude record {action.record}"
        case "reject":
            return f"reject: {action.reason}"
        case "keep":
            return f"keep: {action.note}"


def stretches(xmids: Iterable[float], step: float | None = None) -> str:
    """Runs of consecutive xmids, as 'xmid 12.00-18.00 (7), 21.00 (1)'. `step` is the line's
    spacing of windows (by default, the smallest gap between `xmids`): xmids named to 2
    decimals can sit 0.24 and 0.26 m apart on a line of 0.25 m, hence the margin."""
    values = sorted(set(xmids))
    if not values:
        return "no xmid"
    if len(values) == 1:
        return f"xmid {values[0]:.2f} (1)"
    step = step or line_step(values)
    groups: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= _STRETCH_MARGIN * step:
            groups[-1].append(value)
        else:
            groups.append([value])
    return "xmid " + ", ".join(
        f"{group[0]:.2f}-{group[-1]:.2f} ({len(group)})"
        if len(group) > 1
        else f"{group[0]:.2f} (1)"
        for group in groups
    )


def _only(report: QCReport, gates: set[str]) -> QCReport:
    """`report` with the verdicts, flags and retries of `gates` only; the notes and failures of
    the stages those gates judge stay."""
    judged = {"G1": "preprocessing", "G2": "phase_shift", "G3": "picking", "G4": "picking"}
    judged |= {"G5": "inversion", "G6": "inversion"}
    stages = {judged[gate] for gate in gates if gate in judged}
    units = tuple(
        unit.model_copy(
            update={
                "verdicts": {g: v for g, v in unit.verdicts.items() if g in gates},
                "flags": {g: f for g, f in unit.flags.items() if g in gates},
                "notes": {s: n for s, n in unit.notes.items() if s in stages},
                "failed": {s: e for s, e in unit.failed.items() if s in stages},
            }
        )
        for unit in report.units
    )
    return report.model_copy(
        update={
            "units": units,
            "counts": {g: c for g, c in report.counts.items() if g in gates},
            "retried": {
                t: u
                for t, u in report.retried.items()
                if t.split(":")[0] in gates or t == "backtrack"
            },
            "changed": {
                t: c
                for t, c in report.changed.items()
                if t.split(":")[0] in gates or t == "backtrack"
            },
            "replaced": {
                t: c
                for t, c in report.replaced.items()
                if t.split(":")[0] in gates or t == "backtrack"
            },
        }
    )


def changed_settings(report: QCReport, stages: Sequence[Stage] | None = None) -> tuple[str, ...]:
    """The settings the gates and the checks changed, in words, for the agent to report: each
    gate's change as "from -> to", where, and why (the agent's own backtracks left out), then the
    checks' changes (the window length, the band, the inversion's bounds, excluded traces)."""
    step = line_step(unit.xmid for unit in report.units if unit.xmid is not None)
    said: list[str] = []
    for trigger, names in report.retried.items():
        if trigger == "backtrack" or trigger not in report.changed:
            continue
        stage, change, alike = report.changed[trigger]
        if stages is not None and stage not in stages:
            continue
        before = report.replaced.get(trigger, {})
        items = "; ".join(
            f"{' '.join(path)} {_shown(_get(before, path))} -> {_shown(value)}"
            for path, value in _leaves(change)
        )
        if items:
            each = "" if alike else " (each window its own)"
            said.append(f"{items} at {_where(names, step)}{each}, by {trigger}")
    noted: dict[tuple[Stage, str], list[tuple[str, str]]] = defaultdict(list)
    for unit in report.units:
        for stage, notes in unit.notes.items():
            if stages is None or stage in stages:
                text = " ".join(notes)
                noted[stage, _NUMBER.sub("#", text)].append((unit.unit, text))
    for members in noted.values():
        where = _where([name for name, _ in members], step)
        suffix = " (each its own values)" if len({text for _, text in members}) > 1 else ""
        said.append(f"{where}: {members[0][1]}{suffix}")
    return tuple(said)


def _leaves(
    values: dict[str, Any], path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    found: list[tuple[tuple[str, ...], Any]] = []
    for key, value in values.items():
        if isinstance(value, dict):
            found += _leaves(cast(dict[str, Any], value), (*path, key))
        else:
            found.append(((*path, key), value))
    return found


def _get(values: dict[str, Any], path: tuple[str, ...]) -> Any:  # noqa: ANN401
    found: Any = values
    for key in path:
        found = cast(dict[str, Any], found).get(key) if isinstance(found, dict) else None
    return found


def _at(values: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """`values` at the keys of `change` (None where they had none)."""
    found: dict[str, Any] = {}
    for key, value in change.items():
        old = values.get(key)
        if isinstance(value, dict):
            found[key] = _at(
                cast(dict[str, Any], old) if isinstance(old, dict) else {},
                cast(dict[str, Any], value),
            )
        else:
            found[key] = old
    return found


def _shown(value: Any) -> str:  # noqa: ANN401
    if value is None:
        return "the default"
    if isinstance(value, float):
        return f"{value:g}"
    return (
        json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else str(value)
    )


def _changes(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """What `current` parameters change of `previous`, leaf by leaf (a nested stage kept
    nested)."""
    changed: dict[str, Any] = {}
    for key, value in current.items():
        old = previous.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            inner = _changes(cast(dict[str, Any], old), cast(dict[str, Any], value))
            if inner:
                changed[key] = inner
        elif value != old:
            changed[key] = value
    return changed


def _where(units: Iterable[str], step: float) -> str:
    """Units named for the agent: windows as stretches of xmids, records and the line by name."""
    names = list(units)
    xmids = [xmid for name in names if (xmid := xmid_of(name)) is not None]
    others = [name for name in names if xmid_of(name) is None]
    return ", ".join(part for part in (stretches(xmids, step) if xmids else "", *others) if part)


def line_step(xmids: Iterable[float]) -> float:
    """The smallest gap between distinct `xmids`: the line's spacing of windows (0 for fewer
    than two)."""
    values = sorted(set(xmids))
    return min((b - a for a, b in pairwise(values)), default=0.0)


def _by_unit(attempts: Iterable[Attempt]) -> dict[str, list[Attempt]]:
    by_unit: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_unit[attempt.unit].append(attempt)
    return by_unit


def _unit_report(unit: str, attempts: list[Attempt]) -> UnitReport:
    parameters: dict[Stage, dict[str, Any]] = {}
    notes: dict[Stage, tuple[str, ...]] = {}
    failed: dict[Stage, str] = {}
    latest_result: dict[str, GateResult] = {}
    for attempt in attempts:
        parameters[attempt.stage] = attempt.parameters
        notes[attempt.stage] = attempt.notes
        if attempt.status == "failed":
            failed[attempt.stage] = attempt.error or "failed"
        else:
            failed.pop(attempt.stage, None)
        for gate, result in attempt.results.items():
            latest_result[gate] = result
    rejected_for = tuple(
        flag.message
        for result in latest_result.values()
        if result.verdict == "reject"
        for flag in result.flags
    )
    curve = latest_result["G3"].kept if "G3" in latest_result else None
    return UnitReport(
        unit=unit,
        xmid=xmid_of(unit),
        verdicts={gate: result.verdict for gate, result in latest_result.items()},
        flags={gate: result.flags for gate, result in latest_result.items()},
        attempts=len(attempts),
        parameters=parameters,
        rejected_for=rejected_for,
        curve=curve,
        notes={stage: stage_notes for stage, stage_notes in notes.items() if stage_notes},
        failed=failed,
    )


def xmid_of(unit: str) -> float | None:
    """The xmid of a window's folder name, xmid_<x>; None for a record's name."""
    prefix = "xmid_"
    if unit.startswith(prefix):
        try:
            return float(unit[len(prefix) :])
        except ValueError:
            return None
    return None
