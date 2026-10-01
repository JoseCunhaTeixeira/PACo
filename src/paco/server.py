"""PACo's MCP server: the tools the agent calls, each an adapter around a plain function.

The only module that imports mcp. Settings come from the server's environment (PACO_* variables or
.env), never from the model. Run it with `paco-server`: it serves Streamable HTTP at
http://<PACO_HOST>:<PACO_PORT>/mcp, by default http://127.0.0.1:8000/mcp, this machine only.
"""

import functools
import json
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal

import anyio.from_thread
import matplotlib
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw import presets, profiles, runs
from sigpipe.masw.inversion import InversionParameters, priors
from sigpipe.workers import one_thread_each

from paco import choices, inspection, inversion, prompts, qc, stopping
from paco.jobs import JobManager
from paco.qc import StageResult
from paco.settings import Settings, get_settings

# pick draws figures in this process, from the SDK's worker threads: never start a GUI backend.
matplotlib.use("Agg")

# For the host's system prompt: the workflow the tools make together (prompts/instructions.md).
INSTRUCTIONS = prompts.prompt("instructions")

server = MCPServer("paco", instructions=INSTRUCTIONS)
# How long job_status waits for a job to end: fewer calls for the model, which polls at once.
JOB_WAIT_S = 120.0
# How often job_status reports the job's progress while it waits.
JOB_PROGRESS_S = 3.0
# Inversions run in this process's background, one at a time.
JOBS = JobManager()
# What a tool says the model when the user stopped its work (see paco.stopping).
STOPPED = "Stopped on the user's request: what had finished is kept, the rest as it was."

ProfileName = Annotated[str, Field(description="A profile name from inspect(what=profiles).")]
RunId = Annotated[str, Field(description="A run_id returned by run_processing.")]
JobId = Annotated[str, Field(description="A job_id returned by invert or redo.")]
Positions = Annotated[
    list[float] | None,
    Field(description="Only the windows nearest these positions (m). Left out, the whole line."),
]
Mode = Annotated[
    Literal["active", "passive", "passive-active"] | None,
    Field(description="One of the profile's modes (inspect); left out, its own."),
]
Windows = Annotated[
    Literal["all", "missing"] | None,
    Field(description="Only if the user said which: every window, or those without one."),
]
Hand = Annotated[
    Literal["keep", "replace"] | None,
    Field(description="Only the user's reply to the keep-or-replace question."),
]


def _agent_errors[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """Send PACo's errors to the model: they are ValueErrors, written for the agent; and a stop
    (the host's: see paco.stopping), said as such, the call's work undone where it had not
    finished.

    Any other exception is a bug: the SDK hides its message from the model, and logs its
    traceback in the server's terminal.
    """

    @functools.wraps(tool)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        stopping.pin()  # the stop of the answer this call runs in, whatever answer comes next
        try:
            return tool(*args, **kwargs)
        except stopping.Stopped:
            raise ToolError(STOPPED) from None
        except ValueError as error:
            raise ToolError(str(error)) from error

    return wrapped


@server.tool()
@_agent_errors
def inspect(
    what: Annotated[
        Literal["profiles", "profile", "runs", "run", "window"],
        Field(description="What to read."),
    ],
    profile: Annotated[str | None, Field(description="For what=profile.")] = None,
    run_id: Annotated[str | None, Field(description="For what=run or window.")] = None,
    position: Annotated[float | None, Field(description="For what=window, m.")] = None,
) -> str:
    """Read what exists, changing nothing: the profiles, a profile's acquisition, the runs, a
    run (its settings, then its windows: who made each image, curve and model, the gates'
    verdicts), or a window in detail. Work made in PAC's pages is said by hand."""
    settings = get_settings()
    match what:
        case "profiles":
            return inspection.profiles_text(settings)
        case "profile":
            return inspection.profile_text(_needed(profile, "profile", what), settings)
        case "runs":
            return inspection.runs_text(settings)
        case "run":
            return inspection.run_text(_needed(run_id, "run_id", what), settings)
        case "window":
            return inspection.window_text(
                _needed(run_id, "run_id", what), _needed(position, "position", what), settings
            )


@server.tool()
@_agent_errors
def preset_settings(profile: ProfileName, mode: Mode = None) -> str:
    """The settings run_processing and redo can change for this profile, as a JSON Schema:
    meaning, unit, allowed values and default of each; a null default is derived from the
    profile. Call it only before changing settings."""
    return json.dumps(presets.override_schema(_preset(profile, mode)), separators=(",", ":"))


@server.tool()
@_agent_errors
def run_processing(
    profile: ProfileName,
    ctx: Context,
    overrides: Annotated[
        dict[str, Any] | None,
        Field(
            # Placeholders, not values: a model copies the values of examples.
            description="Only the settings the user gave, by stage, e.g. "
            '{"masw": {"length": <receivers>, "step": <receivers>}, "dispersion": {"vmax": '
            "<m/s>}}; see preset_settings. Left out, they come from the data."
        ),
    ] = None,
    mode: Mode = None,
    again: Annotated[
        bool, Field(description="Only if the user asked to process again: a new run.")
    ] = False,
) -> StageResult:
    """Process a profile into one dispersion image per window, checked stage by stage: records
    (G1, its fixes applied: trigger delays corrected, bad traces left out), window length and
    band from the data unless given, images (G2, retried when a change can fix them). Takes
    seconds to minutes. Returns the run_id and the gates' summary; a profile with a run already
    gives options first."""

    def report(done: int, total: int) -> None:
        # The SDK runs this tool in a worker thread: progress goes out through the event loop.
        anyio.from_thread.run(ctx.report_progress, done, total, f"{done} of {total} windows")

    settings = get_settings()
    _preset(profile, mode)  # a mode the profile cannot take refused first
    conversation = _conversation(ctx)
    given = {"overrides": overrides} if overrides else {}
    given |= {"mode": mode} if mode else {}
    again = again and conversation.allows("run_processing", "again=true")
    if not again and (asked := choices.processing(profile, conversation, settings, given)):
        return _chosen(asked)
    # The mode as an argument, as preset_settings takes it: a model that asked preset_settings
    # for a mode may leave it out of the overrides.
    if mode is not None:
        overrides = {**(overrides or {}), "mode": mode}
    result = qc.process_line(profile, overrides, settings, _qc_config(settings), report)
    choices.worked(conversation, result.run_id)
    choice = qc.read_length_choice(runs.find_run(result.run_id, settings))
    given = qc.given_length(overrides) is not None
    processed = _stage_result(
        result,
        ("G1", "G2"),
        ("preprocessing", "phase_shift"),
        _after_processing(result, given, choice),
    )
    lengths = qc.describe_lengths(choice) if choice is not None else ()
    manifest = runs.load_manifest(result.run_id, settings)
    used = qc.processing_used(manifest, overrides, _line_notes(result))
    return processed.model_copy(update={"lengths": lengths, "used": used})


@server.tool()
@_agent_errors
def pick(
    run_id: RunId,
    ctx: Context,
    positions: Positions = None,
    changes: Annotated[
        dict[str, Any] | None,
        Field(description="Only picking settings the user typed; left out otherwise."),
    ] = None,
    windows: Windows = None,
    hand: Hand = None,
) -> StageResult:
    """Pick each window's M0, checked by G3 (picked again when a change can fix it) and by G4
    over the line (outliers picked again along their neighbours), saved in PAC's layout. G4's
    verdict on the line decides whether invert can run. Curves already there, or picked by
    hand, give options first."""
    settings = get_settings()
    conversation = _conversation(ctx)
    run_folder = runs.find_run(run_id, settings)
    manifest = runs.load_manifest(run_id, settings)
    positions = _asked_positions(conversation, positions)
    changes = checked_picking(changes)
    units, read = _positions(run_id, positions, settings)
    given = _given(positions=positions, changes=changes)
    planned = choices.picking(run_folder, manifest, conversation, units, windows, hand, given)
    if isinstance(planned, choices.Choice):
        return _chosen(planned)
    trigger = qc.ASKED if planned.units is not None else "initial"
    report = qc.pick_line(
        run_id, settings, planned.units, changes, trigger, replace_hand=planned.replace_hand
    )
    choices.worked(conversation, run_id)
    picked = _stage_result(report, ("G3", "G4"), ("picking",), _after_picking(report), read)
    return picked.model_copy(update={"used": _picking_used(run_id, settings, changes)})


@server.tool()
@_agent_errors
def judge(run_id: RunId, ctx: Context, positions: Positions = None) -> StageResult:
    """Judge curves as they are, picking nothing: G3 on automatic M0s no gate judged (PAC's
    Auto-pick) or at positions, then G4 over the line. Hand-picked curves are never judged."""
    settings = get_settings()
    conversation = _conversation(ctx)
    units, read = _positions(run_id, _asked_positions(conversation, positions), settings)
    report = qc.judge_curves(run_id, settings, units)
    choices.worked(conversation, run_id)
    return _stage_result(report, ("G3", "G4"), ("picking",), _after_picking(report), read)


@server.tool()
@_agent_errors
def inversion_settings() -> str:
    """The inversion parameters invert can take, as a JSON Schema with PAC's defaults. Left out,
    the bounds come from each curve."""
    schema = InversionParameters.model_json_schema()
    return json.dumps(presets.without_titles(schema), separators=(",", ":"))


@server.tool()
@_agent_errors
def invert(
    run_id: RunId,
    ctx: Context,
    parameters: Annotated[
        dict[str, Any] | None,
        Field(
            description="Only the inversion parameters the user typed (see inversion_settings); "
            "left out otherwise."
        ),
    ] = None,
    positions: Positions = None,
    windows: Windows = None,
    hand: Hand = None,
) -> inversion.InversionStatus | StageResult:
    """Invert curves into layered Vs models as a background job, every mode of each window:
    those G3 and G4 passed, and those picked by hand (taken as they are); unjudged automatic
    curves are judged first. By default the windows without a model. Bounds from each curve
    (values given are checked), G5 and G6 retrying what they can. Follow it with job_status.
    Models already there, or made by hand, give options first."""
    settings = get_settings()
    checked = priors.checkable(parameters, priors.PriorRules().n_layers) if parameters else None
    _parse(InversionParameters, checked, "parameters", "inversion_settings")
    conversation = _conversation(ctx)
    run_folder = runs.find_run(run_id, settings)
    manifest = runs.load_manifest(run_id, settings)
    positions = _asked_positions(conversation, positions)
    units, read = _positions(run_id, positions, settings)
    given = _given(parameters=parameters, positions=positions)
    planned = choices.inverting(run_folder, manifest, conversation, units, windows, hand, given)
    if isinstance(planned, choices.Choice):
        return _chosen(planned)
    notes = _judged_first(run_id, settings)
    if planned.units is not None:
        units, left_out = _taken(run_id, planned.units, settings)
        said = (*notes, f"Positions: {read}.") if read else notes
        if left_out:
            where = qc.stretches([_xmid(unit) for unit in left_out])
            said = (*said, f"Left out, without a curve the inversion takes: {where}.")
        record = qc.submit_inversion(run_id, parameters, settings, units, said)
        job = functools.partial(
            qc.run_inversion_job,
            record,
            settings,
            None,
            units,
            parameters or {},
            qc.ASKED,
            planned.replace_hand,
        )
    else:
        record = qc.submit_inversion(run_id, parameters, settings, notes=notes)
        job = functools.partial(qc.run_inversion_job, record, settings)
    choices.worked(conversation, run_id)
    JOBS.submit(record.job_id, stopping.bound(job))
    return _job_status(record.job_id, settings)


@server.tool()
@_agent_errors
def job_status(job_id: JobId, ctx: Context) -> inversion.InversionStatus:
    """Where an inversion job stands, after waiting up to 2 minutes for it to end: windows done,
    the models so far (Vs at a few depths, depth informed, misfit: about 1 is within the
    uncertainties), failures; once ended, the gates' summary and the changed settings."""
    settings = get_settings()
    inversion.find_job(job_id, settings)  # an unknown job fails at once
    deadline = time.monotonic() + JOB_WAIT_S
    reported: inversion.JobProgress | None = None
    while JOBS.is_live(job_id) and (left := deadline - time.monotonic()) > 0:
        JOBS.wait(job_id, min(JOB_PROGRESS_S, left))
        progress = inversion.find_job(job_id, settings)[1].progress
        if progress is not None and progress != reported and JOBS.is_live(job_id):
            # The SDK runs this tool in a worker thread: progress goes out through the event loop.
            anyio.from_thread.run(
                ctx.report_progress, progress.done, progress.total, progress.message
            )
            reported = progress
    return _job_status(job_id, settings)


@server.tool()
@_agent_errors
def petro_models(run_id: RunId) -> qc.PetroChoice:
    """Only if the user asks for soils or the water table: the Silex models of the
    petrophysical inversion, what each was trained on, and how many of the run's curves G4
    passed it covers (why not the others). Choose the one covering the most."""
    return qc.petro_models(run_id, get_settings())


@server.tool()
@_agent_errors
def invert_petro(
    run_id: RunId,
    model: Annotated[str, Field(description="A Silex model name from petro_models.")],
    ctx: Context,
    windows: Windows = None,
    hand: Hand = None,
) -> StageResult:
    """Only if the user asks: invert the run's curves G4 passed that `model` covers into soils,
    N values and the water table, checked by G7 per window and G8 along the line; writes PAC's
    petrophysical sections. Returns the gates' summary and what the models say. Soil columns
    already there, or made by hand, give options first."""

    def report(done: int, total: int) -> None:
        anyio.from_thread.run(ctx.report_progress, done, total, f"{done} of {total} windows")

    settings = get_settings()
    conversation = _conversation(ctx)
    run_folder = runs.find_run(run_id, settings)
    manifest = runs.load_manifest(run_id, settings)
    planned = choices.soils(run_folder, manifest, conversation, model, windows, hand)
    if isinstance(planned, choices.Choice):
        return _chosen(planned)
    result, described = qc.invert_petro_line(
        run_id, model, settings, report, planned.units, planned.replace_hand
    )
    choices.worked(conversation, run_id)
    judged = _stage_result(result, ("G7", "G8"), ("petro_inversion",), _after_petro(result))
    return judged.model_copy(
        update={"summary": f"{described}\n{judged.summary}", "used": (f"Silex model {model}",)}
    )


@server.tool()
@_agent_errors
def redo(
    run_id: RunId,
    ctx: Context,
    stage: Annotated[
        Literal["preprocessing", "phase_shift", "picking", "inversion"],
        Field(description="The stage to go back to."),
    ],
    xmids: Annotated[list[float] | None, Field(description="The windows to redo, by xmid.")] = None,
    flag: Annotated[
        str | None, Field(description="Or the windows carrying this flag of the summary.")
    ] = None,
    changes: Annotated[
        dict[str, Any] | None,
        Field(description="The settings to change, as the flag suggests them."),
    ] = None,
    hand: Hand = None,
) -> StageResult:
    """Go back to a stage for some windows (all, without xmids or flag) with changes, and redo
    what follows up to G4: preprocessing (their records), phase_shift or picking; inversion runs
    as a job to follow with job_status. For a change a gate asked of an earlier stage. Work made
    by hand there gives options first."""
    settings = get_settings()
    units = qc.select_windows(run_id, settings, xmids, flag)
    run_folder = runs.find_run(run_id, settings)
    work = qc.run_work(run_folder, runs.load_manifest(run_id, settings))
    conversation = _conversation(ctx)
    if asked := choices.redoing(conversation, run_id, stage, units, work, hand, changes):
        return _chosen(asked)
    choices.worked(conversation, run_id)
    replace_hand = hand == "replace"
    if stage == "inversion":
        if hand == "keep":
            units = [unit for unit in units if work[unit].model != "user"]
        run_folder = runs.find_run(run_id, settings)
        n_windows = len(runs.load_manifest(run_id, settings).windows)
        qc.check_budget(run_id, run_folder, qc.read_qc_config(run_folder), n_windows)
        record = qc.submit_inversion(run_id, None, settings)
        JOBS.submit(
            record.job_id,
            stopping.bound(
                functools.partial(
                    qc.run_inversion_job,
                    record,
                    settings,
                    None,
                    units,
                    changes or {},
                    "backtrack",
                    replace_hand,
                )
            ),
        )
        return StageResult(
            run_id=run_id,
            summary=f"Inversion of {len(units)} window(s) started again as job {record.job_id}.",
            next=f"Follow it with job_status: job_id {record.job_id}.",
            job_id=record.job_id,
        )
    report = qc.redo_stage(run_id, stage, units, changes, settings, replace_hand=replace_hand)
    picking = _picking_used(run_id, settings)
    if stage == "picking":
        redone = _stage_result(report, ("G3", "G4"), ("picking",), _after_picking(report))
        return redone.model_copy(update={"used": picking})
    redone = _stage_result(
        report,
        ("G1", "G2", "G3", "G4"),
        ("preprocessing", "phase_shift", "picking"),
        _after_picking(report),
    )
    processing = qc.processing_used(runs.load_manifest(run_id, settings), None, _line_notes(report))
    return redone.model_copy(update={"used": processing + picking})


def _job_status(job_id: str, settings: Settings) -> inversion.InversionStatus:
    """The status of job `job_id`, with the parameters its windows were inverted with once it
    ended."""
    _, record = inversion.find_job(job_id, settings)
    live = JOBS.is_live(job_id)
    status = inversion.summarize_inversion(record, live=live)
    if live:
        return status
    attempts = qc.read_attempts(runs.find_run(record.run_id, settings))
    used = qc.inversion_used(attempts, [window.folder for window in record.windows], record.given)
    return status.model_copy(update={"used": used})


def _line_notes(report: qc.QCReport) -> tuple[str, ...]:
    """What the rules on the whole line decided and why: the window length, the shots' reach,
    the band."""
    line = next((unit for unit in report.units if unit.unit == qc.LINE), None)
    return tuple(note for notes in line.notes.values() for note in notes) if line else ()


def _picking_used(
    run_id: str, settings: Settings, given: dict[str, Any] | None = None
) -> tuple[str, ...]:
    return qc.picking_used(qc.read_qc_config(runs.find_run(run_id, settings)).picking, given)


def _qc_config(settings: Settings) -> qc.QCConfig:
    return qc.load_qc_config(settings.qc_config)


def _stage_result(
    report: qc.QCReport,
    gates: tuple[str, ...],
    stages: tuple[qc.Stage, ...],
    next_step: str,
    read: str = "",
) -> StageResult:
    """A stage tool's result: what its gates found, after how the positions asked were read."""
    summary = qc.summarize_report(report, gates)
    return StageResult(
        run_id=report.run_id,
        changed=qc.changed_settings(report, stages),
        summary=f"Positions: {read}.\n{summary}" if read else summary,
        next=next_step,
    )


def _positions(
    run_id: str, positions: list[float] | None, settings: Settings
) -> tuple[list[str] | None, str]:
    """The windows at `positions` (None: the whole line) and how each position was read."""
    if not positions:
        return None, ""
    return qc.at_positions(runs.load_manifest(run_id, settings), positions)


def _conversation(ctx: Context) -> choices.Conversation:
    """The conversation a call belongs to, its turn and its message's scope, as PACo's host
    sends them (paco.agent); unknown from another client."""
    meta = ctx.request_context.meta
    if not isinstance(meta, dict):
        return choices.Conversation()
    conversation, turn = meta.get("conversation"), meta.get("turn")
    return choices.Conversation(
        conversation if isinstance(conversation, str) else None,
        turn if isinstance(turn, int) and not isinstance(turn, bool) else None,
        _asked(meta.get("scope")),
    )


def _asked(scope: object) -> choices.Asked | None:
    """The scope a call carries (paco.agent.scope's `for_server`), or None; a malformed one is
    refused, never guessed."""
    if scope is None:
        return None
    try:
        parsed = _Scope.model_validate(scope)
    except ValidationError as error:
        raise ValueError(f"The call's scope is not valid: {error}") from None
    return choices.Asked(
        frozenset(parsed.asked),
        parsed.redo,
        parsed.hand_work,
        parsed.chosen,
        tuple(parsed.positions_m),
    )


class _Scope(BaseModel):
    """The scope as PACo's host sends it with each call."""

    model_config = ConfigDict(extra="forbid")

    asked: list[Literal["process", "pick", "invert", "soils"]]
    redo: bool
    hand_work: Literal["replace", "unsaid"]
    positions_m: list[float] = []
    chosen: str | None = None


def _asked_positions(
    conversation: choices.Conversation, positions: list[float] | None
) -> list[float] | None:
    """The positions a call works at: with a scope, those the message named (none: the whole
    line), whatever the model wrote; else the call's own."""
    if conversation.asked is None:
        return positions
    return list(conversation.asked.positions) or None


def _taken(run_id: str, units: list[str], settings: Settings) -> tuple[list[str], list[str]]:
    """Of `units`, the windows whose curve the inversion takes, and those left out; refused
    when none is left."""
    run_folder = runs.find_run(run_id, settings)
    ready = qc.invertible(run_folder, runs.load_manifest(run_id, settings))
    taken = [unit for unit in units if unit in ready]
    if not taken:
        where = qc.stretches([_xmid(unit) for unit in units])
        raise ValueError(
            f"Run '{run_id}' has no curve the inversion takes at {where}: none there, or G3 or "
            "G4 rejected it."
        )
    return taken, [unit for unit in units if unit not in ready]


def checked_picking(changes: dict[str, Any] | None) -> dict[str, Any] | None:
    """`changes` if every one is a picking setting; refused, with the settings there are,
    otherwise."""
    if not changes:
        return changes
    known = PickingParameters.model_fields
    if unknown := [name for name in changes if name not in known]:
        raise ValueError(
            f"Not picking settings: {', '.join(unknown)}. The picking settings: "
            f"{', '.join(known)}. Call pick again without them: it picks M0. PACo picks M0 "
            "alone: a higher mode is picked by hand in PAC's Dispersion picking page, then "
            "invert takes it with M0."
        )
    return changes


def _xmid(unit: str) -> float:
    return float(unit.removeprefix("xmid_"))


def _given(**arguments: Any) -> dict[str, Any]:  # noqa: ANN401
    """A call's arguments that were given, for the options a choice writes."""
    return {name: value for name, value in arguments.items() if value}


def _chosen(asked: choices.Choice) -> StageResult:
    """A stage tool's result when the user must choose first, or when the run's work goes on:
    what is there, and the options, each with its call."""
    return StageResult(
        run_id=asked.run_id,
        summary=asked.summary,
        next=asked.next,
        options=tuple(qc.Option(label=label, call=made) for label, made in asked.options),
    )


def _judged_first(run_id: str, settings: Settings) -> tuple[str, ...]:
    """The run's automatic curves no gate judged (PAC's own automatic picks), judged before
    the inversion takes them: what G3 and G4 said, for the job's notes; none to judge, none."""
    run_folder = runs.find_run(run_id, settings)
    work = qc.run_work(run_folder, runs.load_manifest(run_id, settings))
    if not any(one.m0 == "unjudged" for one in work.values()):
        return ()
    report = qc.judge_curves(run_id, settings)
    return (f"Judged first, as they are: {qc.summarize_report(report, ('G3', 'G4'))}",)


def _needed[T](value: T | None, name: str, what: str) -> T:
    """`value`, which inspect needs for `what`; refused when missing."""
    if value is None:
        raise ValueError(f"inspect(what={what}) needs {name}.")
    return value


def _after_processing(
    report: qc.QCReport, length_given: bool = False, choice: qc.LengthChoice | None = None
) -> str:
    imaged = [
        unit
        for unit in report.units
        if unit.xmid is not None and unit.verdicts.get("G2") not in (None, "reject")
    ]
    if not imaged:
        return (
            "No image passed G2: you are stuck. Ask the user which to try, with options: redo "
            "with a change the flags suggest, a new run with other settings, or stopping here."
        )
    step = f"pick comes next for run_id {report.run_id}, if the user asked for curves or models."
    if length_given or choice is None:
        return step
    return qc.length_hint(choice, step)


def _after_picking(report: qc.QCReport) -> str:
    line = next((unit for unit in report.units if unit.unit == qc.LINE), None)
    if line is None or line.verdicts.get("G4") == "reject":
        return (
            "G4 rejected the line: no curve to invert, you are stuck. Ask the user which to try, "
            "with options: redo with a change the flags suggest, a new run with other settings, "
            "or stopping here."
        )
    curves = [
        unit
        for unit in report.units
        if unit.xmid is not None
        and unit.verdicts.get("G3") == "pass"
        and unit.verdicts.get("G4") == "pass"
    ]
    return (
        f"{len(curves)} curves passed G3 and G4. invert can run on run_id {report.run_id}, if the "
        "user asked for models; otherwise answer. Higher modes (M1, M2) are picked by hand in "
        "PAC's Dispersion picking page, then invert takes them with M0."
    )


def _after_petro(report: qc.QCReport) -> str:
    # The agent may leave out why the Silex model covered only some of the curves.
    covered = (
        "the soils and water table the summary gives, how many curves the model covered and why "
        "it left the others out"
    )
    line = next((unit for unit in report.units if unit.unit == qc.LINE), None)
    if line is None or line.verdicts.get("G8") != "pass":
        return (
            f"Fewer than two petrophysical models passed G7 and G8: no section. Answer with "
            f"{covered}, and the flags."
        )
    return f"Answer with {covered}, and the sections."


def _preset(profile: str, mode: str | None = None) -> str:
    """The preset for `profile` in `mode`, by default the one named after its kind; refused
    when the profile cannot be processed that way."""
    kind = profiles.load_profile(profile, get_settings()).kind
    if mode is None:
        return kind.value
    if mode not in profiles.MODES[kind]:
        modes = ", ".join(profiles.MODES[kind])
        raise ValueError(f"Profile '{profile}' is {kind}: its modes are {modes}.")
    return mode


def _parse[M: BaseModel](
    model: type[M], values: dict[str, Any] | None, argument: str, described_by: str
) -> M | None:
    """`values` as a `model`, or None to keep the defaults; one line per problem otherwise, naming
    what to send instead."""
    if values is None:
        return None
    try:
        return model.model_validate(values)
    except ValidationError as error:
        lines = presets.explain_parameters(error, model, argument)
        problems = "\n".join(f"- {line}" for line in lines)
        raise ValueError(f"Invalid {argument} (see {described_by}):\n{problems}") from error


def transport_security(settings: Settings) -> TransportSecuritySettings | None:
    """Host checks for the settings' allowed hosts; None leaves the SDK's own rule (checks on
    127.0.0.1 only)."""
    if not settings.allowed_hosts:
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(settings.allowed_hosts),
        allowed_origins=[f"http://{host}" for host in settings.allowed_hosts],
    )


def main() -> None:
    """paco-server: serve PACo's tools over Streamable HTTP, where the settings say."""
    # A wrong setting stops the server here, instead of failing every tool call.
    settings = get_settings()
    # One thread in each process of its jobs: the workers are the cores a job takes.
    one_thread_each()
    server.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
        transport_security=transport_security(settings),
    )


if __name__ == "__main__":
    main()
