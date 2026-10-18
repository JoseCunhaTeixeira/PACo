"""invert and job_status (docs/geo_plan.md §4.8, §6, §10, §13 M7): the frozen curves inverted into
layered Vs models, in groups the model names, as a background job.

Each group names windows (frozen ones, as a spec names them) and the parameters they share
(sigpipe's InversionParameters' fields). Each window's set goes through sigpipe's checks before
S4, from its own curve (derive_inversion): the values given are kept where they pass, and the
checks' notes come back. The settings the user gave (paco.geo.tools.given: the iterations, the Vs
bounds) are in every window's, kept where the checks fail, the note saying what they ask; a group
changing one is refused. A window inverted before starts from its latest parameters, the group's
changes on them. Each inversion is seeded, from a hash of the run, the window and its inversion's
number in the study, through the copy of invert_window that passes the seed; the seed goes into
the attempt's notes. G5 judges each model, then G6 the line. No gate retries anything: the model
reads G5's symptoms (job_status, grouped) and inverts a group again, within each window's budget
of retries, which the host gives in the call's meta. The job's record is PACo's (inversion.json in
the run folder), which job_status reads; the study keeps each window's inversions
(inversions.jsonl), and the journal each job's results."""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sigpipe.base import DispersionCurve
from sigpipe.masw.inversion import InversionError, InversionParameters
from sigpipe.masw.inversion.priors import Derived, derive_inversion
from sigpipe.masw.inversion.section import save_comparison, save_section, save_sections_file
from sigpipe.masw.inversion.summary import save_line_summary
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import RunError, RunManifest, window_length
from sigpipe.masw.runs.stopping import Stopped
from sigpipe.masw.runs.writing import run_lock

from paco.geo.sensors.flags import FlagClass, flag_class
from paco.geo.tools import journal
from paco.geo.tools.freeze import frozen
from paco.geo.tools.given import check_inversion, given_of, with_inversion
from paco.geo.tools.pick import flag_said, manifest_of
from paco.geo.tools.trials import Study
from paco.geo.tools.windows import select
from paco.geo.upstream import inverting
from paco.inversion import (
    InversionRecord,
    JobProgress,
    WindowInversion,
    new_job_id,
    summarize_inversion,
    write_record,
)
from paco.qc import given as paco_given
from paco.qc.config import QCConfig
from paco.qc.g6_models import LINE
from paco.qc.inverting import QUICK_SAMPLER, given_again, judge_model_line, line_depths
from paco.qc.judging import saved_m0
from paco.qc.log import afresh, append_attempt, latest, read_attempts
from paco.qc.models import Attempt, GateResult
from paco.qc.report import build_report, stretches, write_report, xmid_of
from paco.settings import Settings

INVERSIONS = "inversions.jsonl"  # in the study's folder
# How long a job waits for its run, held by another writer, before it fails.
JOB_WAIT_S = 60.0
# The quick effort's sampler (PACo's), and the full effort: PAC's own, as derived.
type Effort = Literal["quick", "full"]
EFFORTS = (
    "quick: PACo's short sampler, 3,000 iterations and 2 chains, a first look; full: PAC's own, "
    "200,000 iterations and 5 chains, the final models' (some 40 minutes a line on 8 cores)"
)

logger = logging.getLogger(__name__)


class Group(BaseModel):
    """Windows inverted alike: those a spec names, and the parameters they share."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    windows: str = "all"
    parameters: dict[str, Any] = Field(default_factory=dict[str, Any])


class Inverted(BaseModel):
    """A window's inversion, as the study keeps it: the job, the window, its inversion's number
    in the study and its seed, the parameters it ran with and the checks' notes, G5's verdict and
    flags (each with its class), or why it failed."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    run_id: str
    unit: str
    number: int
    seed: int
    parameters: dict[str, Any] = Field(default_factory=dict[str, Any])
    notes: tuple[str, ...] = ()
    verdict: str | None = None
    flags: dict[str, FlagClass] = Field(default_factory=dict[str, FlagClass])
    error: str | None = None
    at: datetime


def inversions_of(study: Study, run_id: str | None = None) -> list[Inverted]:
    """The study's inversions, in their order; with `run_id`, that run's alone."""
    path = study.folder / INVERSIONS
    if not path.exists():
        return []
    found = [Inverted.model_validate_json(line) for line in path.read_text().splitlines() if line]
    return [one for one in found if run_id is None or one.run_id == run_id]


def seed_of(run_id: str, unit: str, number: int) -> int:
    """A window's seed: a hash of the run, the window and its inversion's number in the study."""
    digest = hashlib.sha256(f"{run_id}/{unit}/{number}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def frozen_curves(study: Study, run_folder: Path) -> dict[str, DispersionCurve]:
    """The curves the freeze kept, by window; refused before the freeze."""
    made = frozen(study)
    if made is None:
        raise RunError("No curve frozen yet: decide every window, then freeze, before inverting.")
    found: dict[str, DispersionCurve] = {}
    for curve in made.curves:
        saved = saved_m0(run_folder / curve.unit / CURVES_FILE)
        if saved is not None:
            found[curve.unit] = saved
    return found


def submit(
    study: Study,
    run_folder: Path,
    groups: Sequence[Group],
    effort: Effort,
    settings: Settings,
    config: QCConfig,
    most_retries: int | None = None,
    statuses: Mapping[str, str] | None = None,
) -> tuple[InversionRecord, Callable[[], InversionRecord]]:
    """The job inverting `groups` of the frozen curves of the run in `run_folder`, at `effort`,
    recorded queued, and the work it does; refused, saying why, when a group names no frozen
    window, a window twice, a parameter the inversion has not, or a window past its
    `most_retries` (none: no limit)."""
    curves = frozen_curves(study, run_folder)
    manifest = manifest_of(run_folder)
    if not groups:
        raise ValueError("Give a group at least: {windows, parameters}.")
    user = given_of(study)
    taken: dict[str, int] = {}  # each window's group, by its index
    for index, group in enumerate(groups):
        if unknown := sorted(set(group.parameters) - set(InversionParameters.model_fields)):
            raise ValueError(
                f"Unknown inversion parameter(s) {', '.join(unknown)}: see inversion_settings."
            )
        check_inversion(group.parameters, user)
        named = [
            unit for unit in select(run_folder, group.windows, statuses or {}) if unit in curves
        ]
        if not named:
            raise ValueError(f"No frozen curve where {group.windows}.")
        if twice := [unit for unit in named if unit in taken]:
            raise ValueError(f"Each window in one group: {_where(twice)} are in two.")
        taken |= dict.fromkeys(named, index)
    done = inversions_of(study, manifest.run_id)
    if most_retries is not None:
        spent = [
            unit for unit in taken if sum(1 for one in done if one.unit == unit) >= most_retries + 1
        ]
        if spent:
            raise ValueError(
                f"The windows at {_where(spent)} have spent their {most_retries} retries: "
                "decide their models."
            )
    if user.inversion:
        # The user's settings logged as PACo logs them, as the default layering takes them.
        paco_given.give(run_folder, "inversion", with_inversion({}, user, config.priors)[0])
    record = InversionRecord(
        job_id=new_job_id(),
        run_id=manifest.run_id,
        given={"effort": effort, "groups": [group.model_dump() for group in groups]},
        state="queued",
        submitted_at=datetime.now(UTC),
        total=len(taken),
    )
    write_record(run_folder, record)

    def work() -> InversionRecord:
        return run_job(
            study, run_folder, record, list(groups), taken, curves, effort, settings, config
        )

    return record, work


def run_job(
    study: Study,
    run_folder: Path,
    record: InversionRecord,
    groups: Sequence[Group],
    assigned: Mapping[str, int],
    curves: Mapping[str, DispersionCurve],
    effort: Effort,
    settings: Settings,
    config: QCConfig,
) -> InversionRecord:
    """Job `record` run to its end, holding the run while it writes it: each window
    `assigned` to one of `groups` (by its index) inverted with its group's parameters, G5 on
    each, then G6 over the line, the run's report written, the job's results in the journal."""
    try:
        with run_lock(
            run_folder, f"PACo's geophysicist (inversion job {record.job_id})", wait_s=JOB_WAIT_S
        ):
            return _run(
                study, run_folder, record, groups, assigned, curves, effort, settings, config
            )
    except Exception as error:
        failed = record.model_copy(
            update={"state": "failed", "error": str(error), "finished_at": datetime.now(UTC)}
        )
        write_record(run_folder, failed)
        return failed


def describe_job(record: InversionRecord, live: bool, run_folder: Path) -> str:
    """A job as the model reads it: where it stands; once ended, G5's verdicts grouped by
    symptom, G6 on the line, the models' depths and fits."""
    status = summarize_inversion(record, live)
    head = f"Job {record.job_id}: {status.state}, {status.done} of {status.total} windows"
    if status.elapsed_s is not None:
        head += f", {status.elapsed_s:g} s"
    if record.progress is not None and status.state in ("queued", "running"):
        head += f" ({record.progress.message})"
    lines = [head + "."]
    if status.state in ("queued", "running"):
        return "\n".join([*lines, "Call job_status again to follow it."])
    if status.error:
        lines.append(f"Error: {status.error}")
    lines += symptoms(run_folder, [window.folder for window in record.windows])
    if status.depth_informed_m is not None and status.misfit is not None:
        low, high = status.depth_informed_m
        lines.append(
            f"Models: informed down to {low:g}-{high:g} m; misfit {status.misfit[0]:g}-"
            f"{status.misfit[1]:g} (about 1 fits within the curves' uncertainties)."
        )
    if status.errors:
        lines.append("Failed: " + "; ".join(status.errors))
    return "\n".join(lines)


def symptoms(run_folder: Path, units: Sequence[str]) -> list[str]:
    """G5's verdicts on `units`' latest models grouped by symptom, each group with the change
    its fixable flags suggest and the windows spec that names it (invert's, decide's); the notes
    apart; then G6 on the line."""
    attempts = read_attempts(run_folder)
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    said: dict[tuple[str, tuple[str, ...]], str] = {}
    noted: dict[str, list[str]] = {}
    g6: list[GateResult] = []
    for unit in units:
        attempt = latest(attempts, unit, "inversion")
        g5 = attempt.results.get("G5") if attempt is not None else None
        if attempt is not None and "G6" in attempt.results:
            g6.append(attempt.results["G6"])
        if g5 is None:
            groups.setdefault(("failed", ()), []).append(unit)
            continue
        acting = tuple(flag.name for flag in g5.flags if flag_class(flag) != "note")
        key = (g5.verdict, acting)
        groups.setdefault(key, []).append(unit)
        said.setdefault(
            key, ", ".join(flag_said(flag) for flag in g5.flags if flag_class(flag) != "note")
        )
        for flag in g5.flags:
            if flag_class(flag) == "note":
                noted.setdefault(flag.name, []).append(unit)
    lines = ["G5:"]
    models = _verdicts(attempts)
    for (verdict, acting), found in sorted(groups.items(), key=lambda item: -len(item[1])):
        flags = f": {said.get((verdict, acting))}" if acting else ""
        named = _named(verdict, acting, found, models)
        lines.append(f"- {_where(found)}: {verdict}{flags} [windows: {named}]")
    if noted:
        lines.append(
            "Notes: " + "; ".join(f"{name} at {_where(found)}" for name, found in noted.items())
        )
    line = latest(attempts, LINE, "inversion")
    result = line.results.get("G6") if line is not None else None
    if result is not None:
        flags = ", ".join(flag_said(flag) for flag in result.flags) or "no flag"
        lines.append(f"G6 on the line: {result.verdict}; {flags}.")
    off = [one.unit for one in g6 if one.verdict != "pass"]
    if off:
        lines.append(f"G6 off their neighbours: {_where(off)}.")
    return lines


def _run(
    study: Study,
    run_folder: Path,
    record: InversionRecord,
    groups: Sequence[Group],
    assigned: Mapping[str, int],
    curves: Mapping[str, DispersionCurve],
    effort: Effort,
    settings: Settings,
    config: QCConfig,
) -> InversionRecord:
    record = record.model_copy(update={"state": "running", "started_at": datetime.now(UTC)})
    write_record(run_folder, record)
    depths = line_depths(curves)
    record = record.model_copy(update={"depths_m": depths})
    windows: dict[str, WindowInversion] = {}

    def on_window(window: WindowInversion) -> None:
        nonlocal record
        windows[window.folder] = window
        ordered = tuple(sorted(windows.values(), key=lambda one: one.xmid))
        record = record.model_copy(update={"windows": ordered})
        write_record(run_folder, record)

    def progressed(done: int, total: int, doing: str) -> None:
        nonlocal record
        record = record.model_copy(
            update={"progress": JobProgress(done=done, total=total, doing=doing)}
        )
        write_record(run_folder, record)

    try:
        before = inversions_of(study, record.run_id)
        for index, group in enumerate(groups):
            units = [unit for unit, one in assigned.items() if one == index]
            _invert_group(
                study, run_folder, record, group, units, curves, before, depths, effort,
                settings, config, progressed, on_window,
            )  # fmt: skip
        manifest = manifest_of(run_folder)
        judge_model_line(run_folder, manifest, config)
        write_report(
            build_report(record.run_id, run_folder, config.budgets, len(manifest.windows)),
            run_folder,
        )
        _sections(run_folder, manifest, config)
        units = list(assigned)
        summary = "\n".join(symptoms(run_folder, units))
        failed = record.windows and all(window.status == "failed" for window in record.windows)
        record = record.model_copy(
            update={
                "state": "failed" if failed else "succeeded",
                "error": "Every window failed." if failed else None,
                "summary": summary,
                "progress": None,
            }
        )
        _journal(study, record, units, run_folder, effort)
    except Stopped:
        record = record.model_copy(update={"state": "stopped", "progress": None})
    record = record.model_copy(update={"finished_at": datetime.now(UTC)})
    write_record(run_folder, record)
    return record


def _invert_group(
    study: Study,
    run_folder: Path,
    record: InversionRecord,
    group: Group,
    units: Sequence[str],
    curves: Mapping[str, DispersionCurve],
    before: Sequence[Inverted],
    depths: tuple[float, ...],
    effort: Effort,
    settings: Settings,
    config: QCConfig,
    progressed: Callable[[int, int, str], None],
    on_window: Callable[[WindowInversion], None],
) -> None:
    """One group's windows inverted, first ones (initial) and those again (backtrack) apart,
    each kept in the study."""
    jobs: dict[str, dict[str, Derived]] = {"initial": {}, "backtrack": {}}
    seeds: dict[str, int] = {}
    numbers: dict[str, int] = {}
    refused: dict[str, str] = {}
    user = given_of(study)
    for unit in units:
        own = [one for one in before if one.unit == unit]
        previous = own[-1] if own else None
        given = (
            given_again(previous.parameters, group.parameters)
            if previous is not None and previous.parameters
            else dict(group.parameters)
        )
        # The user's settings in every window's, kept by the checks where they fail (§8, rule 1).
        given, locked = with_inversion(given, user, config.priors)
        numbers[unit] = len(own) + 1
        seeds[unit] = seed_of(record.run_id, unit, numbers[unit])
        try:
            derived = derive_inversion(curves[unit], config.priors, given or None, locked or None)
        except InversionError as error:
            refused[unit] = f"{type(error).__name__}: {error}"
            continue
        trigger = "backtrack" if own else "initial"
        jobs[trigger][unit] = _at_effort(derived, given, effort)
    for unit, error in refused.items():
        _not_invertible(run_folder, unit, error, on_window)
    results: dict[str, WindowInversion] = {}
    for trigger, chosen in jobs.items():
        if chosen:
            results |= inverting.invert(
                run_folder, chosen, seeds, depths, config, trigger, settings.workers,
                progressed, on_window,
            )  # fmt: skip
    attempts = read_attempts(run_folder)
    now = datetime.now(UTC)
    with (study.folder / INVERSIONS).open("a") as file:
        for unit in units:
            attempt = latest(attempts, unit, "inversion")
            g5 = attempt.results.get("G5") if attempt is not None else None
            window = results.get(unit)
            derived = next((chosen[unit] for chosen in jobs.values() if unit in chosen), None)
            kept = Inverted(
                job_id=record.job_id,
                run_id=record.run_id,
                unit=unit,
                number=numbers[unit],
                seed=seeds[unit],
                parameters=derived.parameters.model_dump(mode="json") if derived else {},
                notes=derived.notes if derived else (),
                verdict=g5.verdict if g5 is not None else None,
                flags={flag.name: flag_class(flag) for flag in g5.flags} if g5 else {},
                error=refused.get(unit) or (window.error if window is not None else None),
                at=now,
            )
            file.write(kept.model_dump_json() + "\n")


def _at_effort(derived: Derived, given: Mapping[str, Any], effort: Effort) -> Derived:
    """`derived` at `effort`: quick, the sampler's fields not given short (PACo's QUICK_SAMPLER);
    full, as derived (PAC's effort)."""
    if effort != "quick":
        return derived
    quick = {key: value for key, value in QUICK_SAMPLER.items() if key not in given}
    return replace(derived, parameters=derived.parameters.model_copy(update=quick))


def _not_invertible(
    run_folder: Path, unit: str, error: str, on_window: Callable[[WindowInversion], None]
) -> None:
    """A window whose curve gives no inversion parameters, logged as a failed attempt."""
    now = datetime.now(UTC)
    append_attempt(
        run_folder,
        afresh(
            run_folder,
            Attempt(
                unit=unit,
                stage="inversion",
                attempt=1,
                parameters={},
                triggered_by="initial",
                started_at=now,
                finished_at=now,
                status="failed",
                error=error,
            ),
        ),
    )
    on_window(WindowInversion(xmid=xmid_of(unit) or 0.0, folder=unit, status="failed", error=error))


def _sections(run_folder: Path, manifest: RunManifest, config: QCConfig) -> None:
    """PAC's section of the line over the models G5 passed, as PACo's job saves it: best effort,
    never the job's failure."""
    attempts = read_attempts(run_folder)
    passed = [
        window.folder
        for window in manifest.windows
        if (attempt := latest(attempts, window.folder, "inversion")) is not None
        and (g5 := attempt.results.get("G5")) is not None
        and g5.verdict == "pass"
    ]
    try:
        save_section(run_folder, passed, window_m=window_length(run_folder))
        save_sections_file(run_folder, passed)
        save_comparison(run_folder, passed)
        save_line_summary(run_folder, passed, config.model.max_misfit, config.model.max_rhat)
    except Exception:
        logger.exception("Could not save the velocity section of %s", run_folder)


def _journal(
    study: Study, record: InversionRecord, units: Sequence[str], run_folder: Path, effort: Effort
) -> None:
    """The job's results in the journal: G5's verdicts and flags, G6's on the line."""
    attempts = read_attempts(run_folder)
    verdicts: dict[str, str] = {}
    flags: dict[str, list[str]] = {}
    for unit in units:
        attempt = latest(attempts, unit, "inversion")
        g5 = attempt.results.get("G5") if attempt is not None else None
        verdicts[unit] = g5.verdict if g5 is not None else "failed"
        for flag in g5.flags if g5 is not None else ():
            flags.setdefault(flag.name, []).append(unit)
    line = latest(attempts, LINE, "inversion")
    g6 = line.results.get("G6") if line is not None else None
    counts = {verdict: list(verdicts.values()).count(verdict) for verdict in set(verdicts.values())}
    said = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
    named = "; ".join(f"{name} {len(found)}" for name, found in flags.items())
    text = f"{effort}, {len(units)} windows: G5 {said}" + (f" ({named})" if named else "")
    if g6 is not None:
        text += f"; G6 {g6.verdict}" + (
            f" ({', '.join(flag.name for flag in g6.flags)})" if g6.flags else ""
        )
    journal.append(
        study,
        journal.Entry(
            id=journal.next_id(study, "inversion"),
            kind="inversion",
            at=datetime.now(UTC),
            text=text,
            data={
                "job_id": record.job_id,
                "groups": record.given.get("groups", []),
                "g5": verdicts,
                "flags": flags,
                "g6": g6.verdict if g6 is not None else None,
            },
        ),
    )


def _where(units: Sequence[str]) -> str:
    xmids = [x for unit in units if (x := xmid_of(unit)) is not None]
    return stretches(xmids)


def _verdicts(attempts: Sequence[Attempt]) -> dict[str, tuple[str, frozenset[str]]]:
    """Each inverted window's G5 verdict on its latest model, and that verdict's flags."""
    found: dict[str, tuple[str, frozenset[str]]] = {}
    for unit in {attempt.unit for attempt in attempts if attempt.stage == "inversion"} - {LINE}:
        inverted = latest(attempts, unit, "inversion")
        g5 = inverted.results.get("G5") if inverted is not None else None
        if g5 is not None:
            found[unit] = (g5.verdict, frozenset(flag.name for flag in g5.flags))
    return found


def _named(
    verdict: str,
    acting: Sequence[str],
    units: Sequence[str],
    models: Mapping[str, tuple[str, frozenset[str]]],
) -> str:
    """The windows spec that names `units`, a symptom's group: by G5's verdict and flags where
    those name them alone on the line, else by their positions."""
    if verdict != "failed":
        alike = {
            unit
            for unit, (seen, flags) in models.items()
            if seen == verdict and flags.issuperset(acting)
        }
        if alike == set(units):
            return "+".join([f"model:{verdict}", *(f"flag:{name}" for name in acting)])
    xmids = sorted(x for unit in units if (x := xmid_of(unit)) is not None)
    return "xmids:[" + ", ".join(f"{x:g}" for x in xmids) + "]"


def groups_said(groups: Sequence[Group]) -> str:
    """The groups in a line, for the model and the user."""
    return "; ".join(
        f"{group.windows} {json.dumps(group.parameters, separators=(',', ':'))}" for group in groups
    )
