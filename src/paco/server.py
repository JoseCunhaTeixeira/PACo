"""PACo's MCP server: the tools the agent calls, each an adapter around a plain function.

The only module that imports mcp. Settings come from the server's environment (PACO_* variables or
.env), never from the model. Run it with `paco-server`: it serves Streamable HTTP at
http://<PACO_HOST>:<PACO_PORT>/mcp, by default http://127.0.0.1:8000/mcp, this machine only.
"""

import functools
import json
from collections.abc import Callable
from typing import Annotated, Any

import anyio.from_thread
import matplotlib
from mcp.server import MCPServer
from mcp.server.mcpserver import (
    AcceptedElicitation,
    Context,
    Elicit,
    ElicitationResult,
    Resolve,
)
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, ValidationError

from paco import inversion, picks, presets, profiles, quality, runs
from paco.inversion import InversionParameters
from paco.jobs import JobManager
from paco.picking import PickingParameters
from paco.quality import QualityParameters
from paco.settings import Settings, get_settings

# pick draws figures in this process, from the SDK's worker threads: never start a GUI backend.
matplotlib.use("Agg")

# For the host's system prompt: the workflow the tools make together.
INSTRUCTIONS = (
    "PACo turns MASW seismic profiles into dispersion curves. Usual order: list_profiles, "
    "inspect_profile, run_processing, then dispersion_quality. If few windows are good, follow "
    "its advice with run_processing's overrides (listed by preset_settings), except on settings "
    "the user chose: ask before changing those. Change picking parameters or thresholds "
    "(quality_settings) only with a reason. Then pick saves the curves of the good windows. "
    "invert asks the user to approve them, then runs in the background (inversion_settings lists "
    "its parameters): follow it with job_status."
)

server = MCPServer("paco", instructions=INSTRUCTIONS)
# Inversions run in this process's background, one at a time.
JOBS = JobManager()

ProfileName = Annotated[str, Field(description="A profile name from list_profiles.")]
RunId = Annotated[str, Field(description="A run_id returned by run_processing.")]
JobId = Annotated[str, Field(description="A job_id returned by invert.")]


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
    rate and Nyquist frequency, record durations. Call it before run_processing, to choose
    settings that fit the profile."""
    return profiles.inspect_profile(profile, get_settings())


@server.tool()
@_agent_errors
def preset_settings(profile: ProfileName) -> str:
    """The settings run_processing can change for this profile, as a JSON Schema: meaning, unit,
    allowed values and default of each; a null default is derived from the profile. Call it only
    before changing settings: the defaults suit a first run."""
    return json.dumps(presets.override_schema(_preset(profile)), separators=(",", ":"))


@server.tool()
@_agent_errors
def run_processing(
    profile: ProfileName,
    ctx: Context,
    overrides: Annotated[
        dict[str, Any] | None,
        Field(
            # Qwen3-4B copied the values of examples: {"masw": {"length": 24}} dropped the step
            # the user asked for, {"masw": {"length": 48, "step": 12}} added one.
            description="Only the settings to change, by stage, e.g. "
            '{"masw": {"length": <receivers>, "step": <receivers>}}; see preset_settings. Leave '
            "it out to use the defaults."
        ),
    ] = None,
) -> runs.RunSummary:
    """Process a profile into dispersion images, one per MASW window, with the preset that fits
    it (active or passive). Takes seconds to minutes. Returns a summary with the run_id that the
    tools working on a run take. A processed window has an image, not yet a judged one:
    dispersion_quality tells which are good."""

    def report(done: int, total: int) -> None:
        # The SDK runs this tool in a worker thread: progress goes out through the event loop.
        anyio.from_thread.run(ctx.report_progress, done, total, f"{done} of {total} windows")

    return runs.run_processing(profile, _preset(profile), overrides, get_settings(), report)


@server.tool()
@_agent_errors
def quality_settings() -> str:
    """The picking parameters and quality thresholds dispersion_quality can change, as JSON
    Schemas with their defaults. The defaults were tuned on demo data: change them only with a
    reason, and tell the user, since thresholds decide which windows count as good."""
    schemas = {
        "picking": PickingParameters.model_json_schema(),
        "thresholds": QualityParameters.model_json_schema(),
    }
    return json.dumps(presets.without_titles(schemas), separators=(",", ":"))


@server.tool()
@_agent_errors
def dispersion_quality(
    run_id: RunId,
    picking: Annotated[
        dict[str, Any] | None,
        Field(description="Changes to the picking parameters; see quality_settings."),
    ] = None,
    thresholds: Annotated[
        dict[str, Any] | None,
        Field(description="Changes to the quality thresholds; see quality_settings."),
    ] = None,
) -> quality.QualitySummary:
    """Judge every window of a processed run: pick its fundamental mode (M0) and tell whether the
    pick can be trusted (good, doubtful or bad), with the flags raised and advice on the
    processing settings to change. Call it after run_processing; then call pick, or process the
    profile again with the advice."""
    return quality.dispersion_quality(
        run_id,
        get_settings(),
        _parse(PickingParameters, picking, "picking", "quality_settings"),
        _parse(QualityParameters, thresholds, "thresholds", "quality_settings"),
    )


@server.tool()
@_agent_errors
def pick(run_id: RunId) -> picks.PickSummary:
    """Save the M0 curve of every good window of a judged run, in PAC's layout, replacing each
    window's earlier M0 curve, for a human to review before any inversion. Call it after
    dispersion_quality."""
    return picks.pick(run_id, get_settings())


@server.tool()
@_agent_errors
def inversion_settings() -> str:
    """The inversion parameters invert can change, as a JSON Schema with PAC's defaults: layers and
    their bounds, and the sampler's effort. The defaults suit a first inversion."""
    schema = InversionParameters.model_json_schema()
    return json.dumps(presets.without_titles(schema), separators=(",", ":"))


class Approval(BaseModel):
    approve: bool = Field(description="Start the inversion of these curves.")


@_agent_errors
def _ask_approval(run_id: str, parameters: dict[str, Any] | None = None) -> Elicit[Approval]:
    """The user's approval of a run's curves, asked through the host: the agent cannot answer.

    Runs again when the host sends the answer back: it only reads, and checks everything first,
    so that the user is never asked about an inversion that cannot start.
    """
    settings = get_settings()
    chosen = _parse(InversionParameters, parameters, "parameters", "inversion_settings")
    chosen = chosen or InversionParameters()
    folder, run_picks = inversion.check_inversion(run_id, settings)
    xmids = [window.xmid for window in run_picks.windows]
    return Elicit(
        f"The agent asks to invert run {run_id}: the M0 curves of {len(xmids)} windows, xmid "
        f"{min(xmids):.2f} to {max(xmids):.2f} m, with {chosen.n_layers} layers and "
        f"{chosen.n_iterations} iterations x {chosen.n_chains} chains per window. Review the "
        f"curves first: DispersionImage_0000.png in each window folder of {folder}, which PAC's "
        "UI also opens, to cut or correct them. Approve?",
        Approval,
    )


@server.tool()
@_agent_errors
def invert(
    run_id: RunId,
    approval: Annotated[ElicitationResult[Approval], Resolve(_ask_approval)],
    parameters: Annotated[
        dict[str, Any] | None,
        Field(description="Changes to the inversion parameters; see inversion_settings."),
    ] = None,
) -> inversion.InversionStatus:
    """Invert the picked M0 curves of a run into layered shear-wave velocity models, as a
    background job; the user is first asked to approve the curves. Returns a job_id at once:
    follow the job with job_status. About two minutes per window with the defaults."""
    if not (isinstance(approval, AcceptedElicitation) and approval.data.approve):
        raise ToolError(
            f"The user did not approve the inversion of run {run_id}. Ask them what to change "
            "before trying again."
        )
    settings = get_settings()
    chosen = _parse(InversionParameters, parameters, "parameters", "inversion_settings")
    record = inversion.submit_inversion(run_id, chosen or InversionParameters(), settings)
    JOBS.submit(record.job_id, functools.partial(inversion.invert_run, record, settings))
    return inversion.summarize_inversion(record, live=True)


@server.tool()
@_agent_errors
def job_status(job_id: JobId) -> inversion.InversionStatus:
    """Where an inversion job stands: windows done, the range of layer velocities and interface
    depths found so far, failures. When it has succeeded, PAC's UI shows the models."""
    _, record = inversion.find_job(job_id, get_settings())
    return inversion.summarize_inversion(record, live=JOBS.is_live(job_id))


def _preset(profile: str) -> str:
    """The preset that fits `profile`: the one named after its kind, active or passive."""
    return profiles.load_profile(profile, get_settings()).kind.value


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
