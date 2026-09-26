"""PACo's MCP server: the tools the agent calls, each an adapter around a plain function.

The only module that imports mcp. Settings come from the server's environment (PACO_* variables or
.env), never from the model. Run it with `paco-server`: it serves Streamable HTTP at
http://<PACO_HOST>:<PACO_PORT>/mcp, by default http://127.0.0.1:8000/mcp, this machine only.
"""

import functools
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

import anyio.from_thread
import matplotlib
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, ValidationError
from sigpipe.masw import presets, profiles, runs
from sigpipe.masw.inversion import InversionParameters, priors

from paco import inversion, qc
from paco.jobs import JobManager
from paco.qc import StageResult
from paco.settings import Settings, get_settings

# pick draws figures in this process, from the SDK's worker threads: never start a GUI backend.
matplotlib.use("Agg")

# For the host's system prompt: the workflow the tools make together.
INSTRUCTIONS = (
    "PACo turns MASW profiles into shear-wave velocity models, checking each stage with a "
    "quality gate that retries what it can. Order: list_profiles, inspect_profile, "
    "run_processing, pick, invert, then job_status until the job ends. Each returns a summary: "
    "verdicts, what was fixed or changed, flags with a suggested change; a change asked of an "
    "earlier stage is yours to make with redo. preset_settings and inversion_settings describe "
    "the settings. Say which settings the gates changed. Ask the user only when stuck, with 2 "
    "or 3 options. For soils or the water table, only if asked: petro_models, then invert_petro."
)

server = MCPServer("paco", instructions=INSTRUCTIONS)
# How long job_status waits for a job to end: fewer calls for the model, which polls at once.
JOB_WAIT_S = 120.0
# Inversions run in this process's background, one at a time.
JOBS = JobManager()

ProfileName = Annotated[str, Field(description="A profile name from list_profiles.")]
RunId = Annotated[str, Field(description="A run_id returned by run_processing.")]
JobId = Annotated[str, Field(description="A job_id returned by invert or redo.")]
Mode = Annotated[
    Literal["active", "passive", "passive-active"] | None,
    Field(description="One of the profile's modes (inspect_profile); left out, its own."),
]


def _agent_errors[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """Send PACo's errors to the model: they are ValueErrors, written for the agent.

    Any other exception is a bug: the SDK hides its message from the model, and logs its
    traceback in the server's terminal.
    """

    @functools.wraps(tool)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return tool(*args, **kwargs)
        except ValueError as error:
            raise ToolError(str(error)) from error

    return wrapped


@server.tool()
@_agent_errors
def list_profiles() -> list[str]:
    """Names of the seismic profiles available for processing. Call this first: tools that work on
    a profile take one of these names."""
    return profiles.list_profiles(get_settings())


@server.tool()
@_agent_errors
def inspect_profile(profile: ProfileName) -> profiles.ProfileSummary:
    """Summary of one profile: active or passive, records, receivers and their spacing, sampling
    rate and Nyquist frequency, record durations."""
    return profiles.inspect_profile(profile, get_settings())


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
            # Qwen3-4B copied the values of examples: placeholders only.
            description="Only the settings the user gave, by stage, e.g. "
            '{"masw": {"length": <receivers>, "step": <receivers>}, "dispersion": {"vmax": '
            "<m/s>}}; see preset_settings. Left out, they come from the data."
        ),
    ] = None,
    mode: Mode = None,
) -> StageResult:
    """Process a profile into one dispersion image per window, checked stage by stage: records
    (G1, its fixes applied: trigger delays corrected, bad traces left out), window length and
    band from the data unless given, images (G2, retried when a change can fix them). Takes
    seconds to minutes. Returns the run_id and the gates' summary."""

    def report(done: int, total: int) -> None:
        # The SDK runs this tool in a worker thread: progress goes out through the event loop.
        anyio.from_thread.run(ctx.report_progress, done, total, f"{done} of {total} windows")

    settings = get_settings()
    # The mode as an argument, as preset_settings takes it: Qwen3-8B asked preset_settings for
    # passive-active, then left the mode out of the overrides (2 of 3 plays, 2026-09-26).
    if mode is not None:
        overrides = {**(overrides or {}), "mode": mode}
    result = qc.process_line(profile, overrides, settings, _qc_config(settings), report)
    choice = qc.read_length_choice(runs.find_run(result.run_id, settings))
    given = qc.given_length(overrides) is not None
    processed = _stage_result(
        result,
        ("G1", "G2"),
        ("preprocessing", "phase_shift"),
        _after_processing(result, given, choice),
    )
    lengths = qc.describe_lengths(choice) if choice is not None else ()
    return processed.model_copy(update={"lengths": lengths})


@server.tool()
@_agent_errors
def pick(run_id: RunId) -> StageResult:
    """Pick each window's fundamental mode (M0), checked by G3 (picked again when a change can
    fix it) and over the whole line by G4 (outliers picked again along their neighbours), and
    save the curves in PAC's layout. G4's verdict on the line decides whether invert can run."""
    report = qc.pick_line(run_id, get_settings())
    return _stage_result(report, ("G3", "G4"), ("picking",), _after_picking(report))


@server.tool()
@_agent_errors
def inversion_settings() -> str:
    """The inversion parameters invert can take, as a JSON Schema with PAC's defaults: layers and
    their bounds, and the sampler's effort. Left out, the bounds come from each curve."""
    schema = InversionParameters.model_json_schema()
    return json.dumps(presets.without_titles(schema), separators=(",", ":"))


@server.tool()
@_agent_errors
def invert(
    run_id: RunId,
    parameters: Annotated[
        dict[str, Any] | None,
        Field(
            description="Only the inversion parameters the user gave, e.g. "
            '{"n_iterations": <n>, "vs_layers": [{"vs_min": <m/s>, "vs_max": <m/s>}]} (one range: '
            "every layer); see inversion_settings."
        ),
    ] = None,
) -> inversion.InversionStatus:
    """Invert the curves G4 passed into layered Vs models, as a background job: bounds from each
    curve (values given are checked), G5 on each model and G6 on the line, retrying what they
    can. Returns a job_id: follow it with job_status."""
    settings = get_settings()
    checked = priors.checkable(parameters, priors.PriorRules().n_layers) if parameters else None
    _parse(InversionParameters, checked, "parameters", "inversion_settings")
    record = qc.submit_inversion(run_id, parameters, settings)
    JOBS.submit(record.job_id, functools.partial(qc.run_inversion_job, record, settings))
    return inversion.summarize_inversion(record, live=True)


@server.tool()
@_agent_errors
def job_status(job_id: JobId) -> inversion.InversionStatus:
    """Where an inversion job stands, after waiting up to 2 minutes for it to end: windows done,
    the smooth median models so far (Vs at a few depths, useful depth, misfit: about 1 is within
    the uncertainties), failures; once ended, the gates' summary and the changed settings."""
    settings = get_settings()
    inversion.find_job(job_id, settings)  # an unknown job fails at once
    JOBS.wait(job_id, JOB_WAIT_S)
    _, record = inversion.find_job(job_id, settings)
    return inversion.summarize_inversion(record, live=JOBS.is_live(job_id))


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
) -> StageResult:
    """Only if the user asks: invert the run's curves G4 passed that `model` covers into soils,
    N values and the water table, checked by G7 per window and G8 along the line; writes PAC's
    petrophysical sections. Returns the gates' summary and what the models say."""

    def report(done: int, total: int) -> None:
        anyio.from_thread.run(ctx.report_progress, done, total, f"{done} of {total} windows")

    result, described = qc.invert_petro_line(run_id, model, get_settings(), report)
    judged = _stage_result(result, ("G7", "G8"), ("petro_inversion",), _after_petro(result))
    return judged.model_copy(update={"summary": f"{described}\n{judged.summary}"})


@server.tool()
@_agent_errors
def redo(
    run_id: RunId,
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
) -> StageResult:
    """Go back to a stage for some windows (all, without xmids or flag) with changes, and redo
    what follows up to G4: preprocessing (their records), phase_shift or picking; inversion runs
    as a job to follow with job_status. For a change a gate asked of an earlier stage."""
    settings = get_settings()
    units = qc.select_windows(run_id, settings, xmids, flag)
    if stage == "inversion":
        run_folder = runs.find_run(run_id, settings)
        n_windows = len(runs.load_manifest(run_id, settings).windows)
        qc.check_budget(run_id, run_folder, qc.read_qc_config(run_folder), n_windows)
        record = qc.submit_inversion(run_id, None, settings)
        JOBS.submit(
            record.job_id,
            functools.partial(qc.run_inversion_job, record, settings, None, units, changes or {}),
        )
        return StageResult(
            run_id=run_id,
            summary=f"Inversion of {len(units)} window(s) started again as job {record.job_id}.",
            next=f"Follow it with job_status: job_id {record.job_id}.",
            job_id=record.job_id,
        )
    report = qc.redo_stage(run_id, stage, units, changes, settings)
    if stage == "picking":
        return _stage_result(report, ("G3", "G4"), ("picking",), _after_picking(report))
    return _stage_result(
        report,
        ("G1", "G2", "G3", "G4"),
        ("preprocessing", "phase_shift", "picking"),
        _after_picking(report),
    )


def _qc_config(settings: Settings) -> qc.QCConfig:
    return qc.load_qc_config(settings.qc_config)


def _stage_result(
    report: qc.QCReport, gates: tuple[str, ...], stages: tuple[qc.Stage, ...], next_step: str
) -> StageResult:
    return StageResult(
        run_id=report.run_id,
        changed=qc.changed_settings(report, stages),
        summary=qc.summarize_report(report, gates),
        next=next_step,
    )


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
        "user asked for models; otherwise answer."
    )


def _after_petro(report: qc.QCReport) -> str:
    # Qwen3-8B left out why the model covered 1 curve of 6 in 1 play of 3 (2026-09-26).
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
    server.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
        transport_security=transport_security(settings),
    )


if __name__ == "__main__":
    main()
