"""The autonomous agent's server (paco-server --mode autonomous): the MCP server of its tools
(docs/autonomous_plan.md §6), each an adapter around a plain function (paco.autonomous.tools).

Settings come from the server's environment, never from the model, as PACo's own server takes them
(paco.settings: the folders, the workers, where it listens). It serves Streamable HTTP, by default
at http://127.0.0.1:8000/mcp, this machine only.

Each tool works on a profile's study (<output>/<profile>/studies/<id>/): the one the host names in
the call's meta ("study"), else the profile's newest, else a new one. Whether figures may reach
the model is the host's to say too (meta "vision"; none by default): the server never decides it.
PACo's read-only tools are served as PACo serves them, its own functions and cards.
"""

import argparse
import base64
import functools
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import ImageContent, TextContent
from pydantic import Field
from sigpipe.masw.runs import RunError, find_run
from sigpipe.workers import one_thread_each

from paco import logs, stopping
from paco import server as paco_server
from paco.autonomous.tools import advise as advising
from paco.autonomous.tools import commit as committing
from paco.autonomous.tools import decide as deciding
from paco.autonomous.tools import freeze as freezing
from paco.autonomous.tools import invert as inverter
from paco.autonomous.tools import journal as journaling
from paco.autonomous.tools import look as looking
from paco.autonomous.tools import models as modelling
from paco.autonomous.tools import observe as observing
from paco.autonomous.tools import pick as picker
from paco.autonomous.tools import survey as surveying
from paco.autonomous.tools import trials
from paco.autonomous.tools import windows as selecting
from paco.autonomous.tools.commit import Line
from paco.autonomous.tools.trials import LineState, Study
from paco.inversion import find_job
from paco.jobs import JobManager
from paco.qc.config import QCConfig, load_qc_config
from paco.settings import get_settings

# PACo's read-only tools, "as today" (§6), in the plan's order.
PACO_TOOLS = (paco_server.inspect, paco_server.preset_settings, paco_server.inversion_settings)
# preset_settings' card, its tools named: PACo's names run_processing and redo, which this server
# has not.
PRESET_SETTINGS = (
    "The settings try_settings and commit can change for this profile, as a JSON Schema: meaning, "
    "unit, allowed values and default of each; a null default is derived from the profile. With "
    "mode, another mode's (passive-active on an active line). They also take mode, picking (the "
    "automatic pick's parameters) and near_field_m (the active mode's). Call it only before "
    "changing settings."
)
# What a tool says the model when the user stopped its work (paco.stopping).
STOPPED = "Stopped on the user's request: what had finished is kept, the rest as it was."
# Inversions run in this process's background, one at a time (PACo's job manager); job_status
# waits up to JOB_WAIT_S for one to end.
JOBS = JobManager()
JOB_WAIT_S = 120.0

server = MCPServer("paco-autonomous")
for tool in PACO_TOOLS:
    description = PRESET_SETTINGS if tool is paco_server.preset_settings else None
    server.add_tool(tool, annotations=paco_server.READS, description=description)
logger = logging.getLogger(__name__)

ProfileName = Annotated[str, Field(description="A profile name from inspect(what=profiles).")]
LineSettings = Annotated[
    dict[str, Any],
    Field(
        description="The line's settings: mode (the profile's own, or passive-active on an active "
        "line), preset_settings' stages for that mode, picking and near_field_m (m, or null; the "
        "active mode's). masw.step stays 1."
    ),
]


def _errors[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """The package's errors to the model, each tagged with its kind as PACo tags them:
    `[bad argument]` (a value the tool does not take), `[precondition]` (a run or a stage
    missing), `[retry]` (the machine's: a file busy or the disk full); a stop said as such. Any
    other exception is a bug: its traceback in the server's terminal, and its reason to the
    model, tagged `[error]` (the SDK would show it only "Error executing tool <name>")."""

    @functools.wraps(tool)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        stopping.pin()
        try:
            return tool(*args, **kwargs)
        except stopping.Stopped:
            raise ToolError(STOPPED) from None
        except RunError as error:
            raise ToolError(f"[precondition] {error}") from error
        except ValueError as error:
            raise ToolError(f"[bad argument] {error}") from error
        except OSError as error:
            raise ToolError(f"[retry] {type(error).__name__}: {error}") from error
        except MCPError:
            raise
        except Exception as error:
            logger.exception("Tool %s failed", tool.__name__)
            raise ToolError(f"[error] {type(error).__name__}: {error}") from error

    return wrapped


@server.tool(annotations=paco_server.ADDS)
@_errors
def survey(profile: ProfileName, ctx: Context) -> str:
    """The line before any setting: its geometry and the limits it sets the windows and the
    images; its raw records as G1 measures them, the reach, the usable band, and the surface
    waves' cone (a line with shots; a passive line's records are noise, their segments its
    images' step); the modes it takes. Call it first: what it says is where the line's settings
    start from."""
    settings = get_settings()
    return surveying.describe_survey(surveying.survey(_study(profile, ctx), settings, _config()))


@server.tool(annotations=paco_server.READS)
@_errors
def observe(
    profile: ProfileName,
    level: Annotated[
        Literal["records", "images", "curves", "line", "models", "models_line"],
        Field(description="Which gate: records G1, images G2, curves G3, line G4, models G5/G6."),
    ],
    ctx: Context,
    where: Annotated[
        str | None,
        Field(description=f"Rows of these units only: {observing.WHERE}. Left out, a summary."),
    ] = None,
    page: Annotated[int, Field(ge=1, description="The page of rows.")] = 1,
) -> str:
    """The gates' results on the committed line, as text: a summary of the flags and their
    windows, or rows of the units `where` names; on the line, a pseudo-section too."""
    return observing.observe(_run_folder(_study(profile, ctx)), level, _config(), where, page)


@server.tool(annotations=paco_server.READS)
@_errors
def look(
    profile: ProfileName,
    what: Annotated[
        looking.What,
        Field(
            description="A record's gather or spectra, a window's image, a passive or "
            "passive-active window's correlations, a passive one's FK selection, a sheet of up to "
            "12, or a window's model: its profile, fit, marginals or chains."
        ),
    ],
    where: Annotated[
        str,
        Field(description="A record's file name; a window's xmid, m; a sheet's span, a-b in m."),
    ],
    ctx: Context,
) -> list[TextContent | ImageContent]:
    """Figures of the committed line; with no vision, a window's image as a ridge table, and a
    record's gate measures."""
    seen = looking.look(_run_folder(_study(profile, ctx)), what, where, _vision(ctx))
    return [
        TextContent(type="text", text=seen.text),
        *(
            ImageContent(type="image", data=base64.b64encode(png).decode(), mime_type="image/png")
            for png in seen.figures
        ),
    ]


@server.tool(annotations=paco_server.ADDS)
@_errors
def advise(
    profile: ProfileName,
    topic: Annotated[
        Literal["window_length", "mute", "segments", "line_changes", "picking", "inversion"],
        Field(
            description="Whose choice: PACo's ladder, mute trial (a line with shots), segment "
            "trial (a passive line), line loop, G3's or G5's."
        ),
    ],
    ctx: Context,
) -> str:
    """PACo's own automatic choice on a topic, as a second opinion with its evidence, never
    applied."""
    study = _study(profile, ctx)
    line = committing.line_of(study)
    folder = find_run(line.run_id, get_settings()) if line is not None else None
    current = line.settings if line is not None else {}
    return advising.advise(study, topic, current, get_settings(), _config(), folder)


@server.tool(annotations=paco_server.ADDS)
@_errors
def try_settings(
    profile: ProfileName,
    variants: Annotated[
        list[LineSettings],
        Field(description=f"Changes of the line's settings, {trials.MOST_VARIANTS} at most."),
    ],
    ctx: Context,
    windows: Annotated[str, Field(description=f"The trial set: {trials.SPEC}.")] = trials.WINDOWS,
    hypothesis: Annotated[
        str | None, Field(description="The journal's hypothesis it tests, by its id (H<n>).")
    ] = None,
) -> str:
    """Each variant tried with the current settings on the same trial windows, judged by G1 to
    G3, scored, the windows gained and lost against the current settings; changes no run. A
    trial made before is not made again. The experiment goes to the journal."""
    study = _study(profile, ctx)
    line = committing.line_of(study)
    state = committing.line_state(_run_folder(study)) if line is not None else LineState()
    experiment = trials.try_settings(
        study,
        variants,
        windows,
        current=line.settings if line is not None else None,
        line=state,
        settings=get_settings(),
        config=_config(),
    )
    journaling.experiment(study, experiment, hypothesis)
    return trials.describe(experiment)


@server.tool(annotations=paco_server.REPLACES)
@_errors
def commit(
    profile: ProfileName,
    changes: Annotated[
        dict[str, Any],
        Field(
            description="Changes of the committed line's settings, as a try_settings variant "
            "gives them (the first commit: of the preset's)."
        ),
    ],
    reasons: Annotated[
        dict[str, str],
        Field(
            description="Why each setting changed: '<stage> <key>', as the changes name it, to one "
            "line, from the experiments."
        ),
    ],
    ctx: Context,
    experiment: Annotated[
        str | None, Field(description="The experiment the changes come from, by its id (E<n>).")
    ] = None,
) -> str:
    """The line made with these changes of its settings: every window one receiver apart, its
    records judged by G1, its images by G2, each picked (as its recipe says, if it has one) and
    judged by G3, then G4 along the line. The first commit makes the run, in PAC's layout; a later
    one changes the run's settings, what it replaces kept in each window's attempts/; a change of
    masw makes a new run, without recipes."""
    study = _study(profile, ctx)
    committed = committing.commit(study, changes, reasons, experiment, get_settings(), _config())
    journaling.commit(study, committed)
    return committing.describe_commit(committed)


Windows = Annotated[str, Field(description=f"The windows: {selecting.SPEC}.")]
Points = list[tuple[float, float]]


@server.tool(annotations=paco_server.REPLACES)
@_errors
def pick(
    profile: ProfileName,
    windows: Windows,
    how: Annotated[
        Literal["auto", "guide", "lasso", "delete"],
        Field(
            description="The tracker (auto), its corridor on a guide (guide), on the maxima "
            "inside a polygon (lasso), or the curve deleted (delete)."
        ),
    ],
    ctx: Context,
    picking: Annotated[
        dict[str, Any] | None,
        Field(description="Changes of the line's picking parameters, for auto, guide and lasso."),
    ] = None,
    guide: Annotated[
        Points | Literal["neighbours"] | None,
        Field(
            description="For guide: points [f Hz, v m/s] the corridor follows, or 'neighbours', "
            "their median curve as G4 gives it."
        ),
    ] = None,
    polygon: Annotated[
        Points | None,
        Field(
            description="For lasso: vertices [f Hz, v m/s] around the ridge, read off its image."
        ),
    ] = None,
) -> str:
    """Each window's M0 picked as `how` says, from the line's picking, then judged by G3; then G4
    along the line. A window keeps its recipe: a later commit picks it the same way."""
    study = _study(profile, ctx)
    line = _line(study)
    folder = find_run(line.run_id, get_settings())
    units = selecting.select(folder, windows, deciding.statuses(study, folder, line.run_id))
    recipe = picker.Recipe.model_validate(
        {"how": how, "picking": picking or {}, "guide": guide, "polygon": polygon}
    )
    base = committing.picking_of(study, line, get_settings(), _config())
    picks = picker.pick(study, folder, units, recipe, base, _config(), get_settings().workers)
    return picker.describe_picks(picks)


@server.tool(annotations=paco_server.ADDS)
@_errors
def decide(
    profile: ProfileName,
    windows: Windows,
    status: Annotated[
        Literal["accepted", "rejected", "pending"],
        Field(
            description="The windows' curves (before the freeze) or models (after it) accepted, "
            "rejected, or pending again."
        ),
    ],
    ctx: Context,
    reason: Annotated[
        str | None,
        Field(description="One line: why. Needed to reject, and to accept with notes."),
    ] = None,
) -> str:
    """The windows decided. Before the freeze, their curves: accepted (passed G3 with no blocking
    flag, and G4), rejected (once tried hard: the line's pick, a pick after each fixable flag, a
    guide from its neighbours, a lasso if you see the images), or pending. After it, their
    models: accepted (passed G5, no blocking flag), rejected (once each fixable flag was tried,
    or its retries spent), or pending. Decided whole, or refused whole with each window's
    problem."""
    study = _study(profile, ctx)
    line = _line(study)
    folder = find_run(line.run_id, get_settings())
    if (made := freezing.frozen(study)) is not None:
        # The frozen curves' windows, as invert takes them: one rejected before has no model.
        curves = {curve.unit for curve in made.curves}
        named = selecting.select(folder, windows, modelling.statuses(study, folder, line.run_id))
        units = [unit for unit in named if unit in curves]
        if not units:
            raise ValueError(f"No frozen curve where {windows}: no model to decide.")
        decided = modelling.decide(study, folder, units, status, reason, _retries(ctx))
    else:
        units = selecting.select(folder, windows, deciding.statuses(study, folder, line.run_id))
        decided = deciding.decide(study, folder, units, status, reason, _vision(ctx), _config())
    return deciding.describe_decided(decided)


@server.tool(annotations=paco_server.REPLACES)
@_errors
def freeze(profile: ProfileName, ctx: Context) -> str:
    """The curves frozen for the inversion once every window is decided: the one way from the
    curves to the models. Refused before, saying what is missing; no processing or picking
    after."""
    study = _study(profile, ctx)
    line = _line(study)
    made = freezing.freeze(study, find_run(line.run_id, get_settings()), line)
    journaling.freeze(study, made)
    return freezing.describe_frozen(made)


@server.tool(annotations=paco_server.ADDS)
@_errors
def invert(
    profile: ProfileName,
    groups: Annotated[
        list[inverter.Group],
        Field(
            description="Windows inverted alike: each its windows (a spec, as pick's) and the "
            "inversion parameters they share (inversion_settings' fields), a window in one group."
        ),
    ],
    effort: Annotated[inverter.Effort, Field(description=inverter.EFFORTS)],
    ctx: Context,
) -> str:
    """The frozen curves inverted into layered Vs models, group by group, as a background job:
    each window's parameters from its own curve through the checks before S4 (the group's values
    kept where they pass), each inversion seeded; G5 on each model, then G6 on the line; no gate
    retries anything. Follow it with job_status. PACo's PACO_INVERSION_EFFORT set to quick (an
    evaluation's) makes every job quick, whatever the effort asked, as it does PACo's."""
    settings = get_settings()
    study = _study(profile, ctx)
    line = _line(study)
    folder = find_run(line.run_id, settings)
    ran: inverter.Effort = "quick" if settings.inversion_effort == "quick" else effort
    record, work = inverter.submit(
        study,
        folder,
        groups,
        ran,
        settings,
        _config(),
        _retries(ctx),
        modelling.statuses(study, folder, line.run_id),
    )
    JOBS.submit(record.job_id, stopping.bound(work))
    capped = "" if ran == effort else f", PACO_INVERSION_EFFORT's ({effort} was asked)"
    return (
        f"Job {record.job_id} queued: {record.total} windows, {inverter.groups_said(groups)} at "
        f"the {ran} effort{capped}. Follow it with job_status."
    )


@server.tool(annotations=paco_server.READS)
@_errors
def job_status(
    job_id: Annotated[str, Field(description="An inversion job's id, as invert gives it.")],
) -> str:
    """Where an inversion job stands, after waiting up to 2 minutes for it to end: the windows
    done; once ended, G5's verdicts grouped by symptom, each with the change its fixable flags
    suggest, and G6 on the line."""
    settings = get_settings()
    find_job(job_id, settings)  # an unknown job refused at once
    deadline = time.monotonic() + JOB_WAIT_S
    while JOBS.is_live(job_id) and (left := deadline - time.monotonic()) > 0:
        JOBS.wait(job_id, left)
    folder, record = find_job(job_id, settings)
    return inverter.describe_job(record, JOBS.is_live(job_id), folder)


# PACo's petrophysical inversion, as its agent runs it, on the line's run once its models are
# decided (P4): only when the user asks for soils, N values or the water table.
server.add_tool(paco_server.petro_models, annotations=paco_server.READS)
server.add_tool(paco_server.invert_petro, annotations=paco_server.REPLACES)


@server.tool(annotations=paco_server.ADDS)
@_errors
def journal(
    profile: ProfileName,
    action: Annotated[Literal["write", "read"], Field(description="Write an entry, or read.")],
    ctx: Context,
    kind: Annotated[
        Literal["hypothesis", "conclusion", "decision"] | None,
        Field(description="To write: the entry's kind."),
    ] = None,
    text: Annotated[
        str | None,
        Field(description=f"To write: what it says, {journaling.MOST_TEXT} characters at most."),
    ] = None,
    of: Annotated[
        list[str] | None,
        Field(description="To write: the ids of the entries it rests on (E<n>, H<n>...)."),
    ] = None,
    expect: Annotated[
        str | None, Field(description="To write a hypothesis: what the experiment should show.")
    ] = None,
    entry: Annotated[str | None, Field(description="To read: one entry whole, by its id.")] = None,
    last: Annotated[
        int | None, Field(ge=1, description="To read: the last entries, 5 by default.")
    ] = None,
    setting: Annotated[
        str | None,
        Field(
            description=f"To read: the experiments and commits that changed {journaling.SETTING}."
        ),
    ] = None,
) -> str:
    """The study's journal: write a hypothesis, a conclusion or a decision, resting on the entries
    it names; or read one entry, the last ones, or those that changed a setting. Code writes the
    objective, the experiments, the commits and the freeze."""
    study = _study(profile, ctx)
    if action == "read":
        return journaling.read(study, entry, last, setting)
    if kind is None or text is None:
        raise ValueError("To write: give kind and text.")
    written = journaling.write(study, kind, text, of or (), expect)
    return f"Written: {journaling.line(written)}"


def _study(profile: str, ctx: Context) -> Study:
    """The study the host names in the call's meta, else the profile's newest, else a new one."""
    meta = ctx.request_context.meta if isinstance(ctx, Context) else None
    named = meta.get("study") if isinstance(meta, dict) else None
    return trials.study_of(profile, get_settings(), named if isinstance(named, str) else None)


def _retries(ctx: Context) -> int | None:
    """Each window's budget of inversion retries, as the host gives it in the call's meta
    ("inversion_retries"); none: no limit."""
    meta = ctx.request_context.meta if isinstance(ctx, Context) else None
    found = meta.get("inversion_retries") if isinstance(meta, dict) else None
    return found if isinstance(found, int) else None


def _vision(ctx: Context) -> bool:
    meta = ctx.request_context.meta if isinstance(ctx, Context) else None
    return isinstance(meta, dict) and meta.get("vision") is True


def _line(study: Study) -> Line:
    """The study's committed line; refused before any commit."""
    line = committing.line_of(study)
    if line is None:
        raise RunError("No line committed yet in this study: commit one first.")
    return line


def _run_folder(study: Study) -> Path:
    """The study's committed run's folder; refused before any commit."""
    return find_run(_line(study).run_id, get_settings())


def _config() -> QCConfig:
    """The gates' thresholds: PACo's file (PACO_QC_CONFIG), else its defaults; locked for the
    study."""
    return load_qc_config(get_settings().qc_config)


def listing() -> str:
    """The tools' names, one a line, in the order the model is given them."""
    return "\n".join(tool.name for tool in anyio.run(server.list_tools))


def main(argv: Sequence[str] | None = None) -> None:
    """paco-server --mode autonomous: serve the tools over Streamable HTTP where the settings say,
    after a log line naming them; with --list, print their names instead."""
    parser = argparse.ArgumentParser(
        prog="paco-server --mode autonomous",
        description="Serve the autonomous agent's tools over MCP (Streamable HTTP).",
    )
    parser.add_argument("--list", action="store_true", help="print the tools' names, and stop")
    if parser.parse_args(argv).list:
        print(listing())
        return
    # A wrong setting stops the server here, instead of failing every tool call.
    settings = get_settings()
    logs.setup()
    # One thread in each process of its jobs: the workers are the cores a job takes.
    one_thread_each()
    names = ", ".join(listing().splitlines())
    logger.info("Serving %s at http://%s:%d/mcp", names, settings.host, settings.port)
    server.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
        transport_security=paco_server.transport_security(settings),
    )
