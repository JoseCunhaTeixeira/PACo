"""The autonomous agent's server (paco-server --mode autonomous): the MCP server of its tools,
each an adapter around a plain function (paco.autonomous.tools).

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
import re
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
from sigpipe.masw.runs.writing import run_lock
from sigpipe.workers import one_thread_each

from paco import inspection, logs, qc, stopping
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
from paco.autonomous.tools import search as searching
from paco.autonomous.tools import survey as surveying
from paco.autonomous.tools import tidy as tidying
from paco.autonomous.tools import trials
from paco.autonomous.tools import windows as selecting
from paco.autonomous.tools.commit import Line
from paco.autonomous.tools.trials import LineState, Study
from paco.inversion import find_job
from paco.jobs import JobManager
from paco.qc.config import QCConfig, load_qc_config
from paco.qc.loops import deep_merge
from paco.settings import get_settings


def _no_retries[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """`tool`, its runs saying no retries of the gates' budget: the agent's inversions again
    would fill it, under a budget of its own (the card's)."""

    @functools.wraps(tool)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        token = inspection.RETRIES_SAID.set(False)
        try:
            return tool(*args, **kwargs)
        finally:
            inspection.RETRIES_SAID.reset(token)

    return wrapped


# PACo's read-only tools, first.
PACO_TOOLS = (
    _no_retries(paco_server.inspect),
    paco_server.preset_settings,
    paco_server.inversion_settings,
)
# preset_settings' card, its tools named: PACo's names run_processing and redo, which this server
# has not.
PRESET_SETTINGS = (
    "The settings try_settings and commit can change for this profile, as a JSON Schema: meaning, "
    "unit, allowed values and default of each; a null default is derived from the profile. With "
    "mode, another mode's (passive-active on an active line). They also take mode, picking (the "
    "automatic pick's parameters) and near_field_m (the modes' with shots). Call it only before "
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
        "modes' with shots, active and passive-active). masw.step stays 1."
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
            "12, an experiment's variants side by side, or a window's model: its profile, fit, "
            "marginals or chains."
        ),
    ],
    where: Annotated[
        str,
        Field(
            description="A record's file name; a window's xmid, m; a sheet's span, a-b in m; an "
            "experiment, E<n>, or E<n>:<labels> for up to 4 of its variants (current among them)."
        ),
    ],
    ctx: Context,
    mute: Annotated[
        looking.Mute | None,
        Field(
            description="For a gather: a mute drawn on it, its window from x/vmax to x/vmin "
            "plus its width (x the offset from the shot)."
        ),
    ] = None,
    velocities: Annotated[
        list[float] | None,
        Field(
            description="For a gather: velocities, m/s, drawn as lines from the shot (t = offset "
            "/ v), to read the arrivals' speeds."
        ),
    ] = None,
) -> list[TextContent | ImageContent]:
    """Figures of the committed line, and of an experiment's variants side by side; before the
    line's first commit, the survey's raw records (a gather, spectra). A mute's window or
    velocities from the shot drawn on a raw gather, to set a mute by eye. With no vision, a
    window's image as a ridge table, a record's gate measures, an experiment's measures."""
    study = _study(profile, ctx)
    line = committing.line_of(study)
    lines = mute is not None or bool(velocities)
    if what == "experiment":
        seen = looking.look_experiment(study, where, _vision(ctx))
    elif line is None or (what == "gather" and lines):
        seen = looking.look_survey(study, what, where, _vision(ctx), mute, velocities or ())
    else:
        seen = looking.look(
            find_run(line.run_id, get_settings()),
            what,
            where,
            _vision(ctx),
            mute,
            velocities or (),
        )
    return _content(seen.text, seen.figures)


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
        Field(
            description=f"Changes of the line's settings, {trials.MOST_VARIANTS} at most; "
            '{"grid": "<label>"}: a row of the search\'s table.'
        ),
    ],
    ctx: Context,
    windows: Annotated[str, Field(description=f"The trial set: {trials.SPEC}.")] = trials.WINDOWS,
    hypothesis: Annotated[
        str | None,
        Field(
            description="The hypothesis it tests: the journal's, by its id (H<n>), or a new one "
            "in words, which goes to the journal with expect."
        ),
    ] = None,
    expect: Annotated[
        str | None, Field(description="With a new hypothesis: what the experiment should show.")
    ] = None,
) -> list[TextContent | ImageContent]:
    """Each variant tried with the current settings on the same trial windows, judged by G1 to
    G3, the windows gained and lost against the current settings; changes no run. A trial made
    before is not made again. The experiment goes to the journal. With vision, every variant side
    by side, sheets of the current settings and three (look experiment). Once every sheet of the
    search is judged: to refine around your choice."""
    study = _study(profile, ctx)
    if (left := searching.unjudged(study)) is not None:
        raise RunError(left)
    new = hypothesis is not None and _HYPOTHESIS.fullmatch(hypothesis) is None
    if new:
        journaling.check(study, "hypothesis", str(hypothesis), expect=expect)
    line = committing.line_of(study)
    state = committing.line_state(_run_folder(study)) if line is not None else LineState()
    # A row of the search's table, by its label: its changes, with the variant's own over them.
    variants = [
        deep_merge(searching.row_changes(study, str(one["grid"])), _without_grid(one))
        if "grid" in one
        else one
        for one in variants
    ]
    experiment = trials.try_settings(
        study,
        variants,
        windows,
        current=line.settings if line is not None else None,
        line=state,
        settings=get_settings(),
        config=_config(),
    )
    written = ""
    if new:
        entry = journaling.write(study, "hypothesis", str(hypothesis), expect=expect)
        hypothesis, written = entry.id, f"Written: {journaling.line(entry)}\n"
    journaling.experiment(study, experiment, hypothesis)
    said = written + trials.describe(experiment)
    if not _vision(ctx):
        return _content(said)
    seen = looking.look_experiment(study, experiment.id, vision=True)
    return _content(f"{said}\n\n{seen.text}", seen.figures)


def _without_grid(variant: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in variant.items() if key != "grid"}


@server.tool(annotations=paco_server.ADDS)
@_errors
def search(
    profile: ProfileName,
    ctx: Context,
    windows: Annotated[str, Field(description=f"The trial set: {trials.SPEC}.")] = (
        searching.WINDOWS
    ),
    grid: Annotated[
        dict[str, Any] | None,
        Field(
            description="Changes of PACo's grid (grid.yaml) for this study only, as the file "
            'holds them: {"group_velocities": {"slowest": {"value": <m/s>}}}, or '
            '{"<method>": {"<setting>": {"values": [<values>]}}}; where the user\'s words '
            "call for them (their velocities, the values they want). The file stays as it is."
        ),
    ] = None,
) -> list[TextContent | ImageContent]:
    """The full grid of the line's settings, once a study (minutes): every combination of the
    values the physics sets for each setting of PAC's page for the line's method, each variant
    on the same windows, judged by G1 to G3 as try_settings judges it. The really bad dropped (no
    image, or too few windows passing for a curve along the line); every other variant judged by
    eye, no score, in sheets by window length, each at the same places along the line, in short
    calls of their own, round after round, until the finalists: tried side by side at full size
    (an experiment). Choose among them, refine with try_settings, then commit. Called again,
    the sheets left are judged."""
    study = _study(profile, ctx)
    if (made := searching.searched(study)) is not None:
        knockout = searching.knockout_of(study)
        return _sheets(study, made, knockout, ctx, searching.describe(made, knockout))
    line = committing.line_of(study)
    state = committing.line_state(_run_folder(study)) if line is not None else LineState()
    made, knockout = searching.search(
        study,
        line=state,
        settings=get_settings(),
        config=_config(),
        windows=windows,
        grid_changes=grid,
    )
    return _sheets(study, made, knockout, ctx, searching.describe(made, knockout))


@server.tool(annotations=paco_server.ADDS)
@_errors
def judge(
    profile: ProfileName,
    kept: Annotated[
        dict[int, list[str]],
        Field(
            description=f"By sheet number, the variants you keep of it by eye: its cleanest, "
            f"{searching.KEEP} at most ({searching.LAST_KEEP} of the last sheet)."
        ),
    ],
    why: Annotated[
        str, Field(description="In a sentence, what the kept show that the others do not.")
    ],
    ctx: Context,
) -> list[TextContent | ImageContent]:
    """The search's sheets judged by eye, the host's (its short calls judge them): each one's
    cleanest kept, the judgment recorded; then the next sheets. The kept meet again, round after
    round, until one sheet is left; once it is judged, its kept are tried side by side at full
    size (an experiment)."""
    study = _study(profile, ctx)
    made = searching.searched(study)
    knockout = searching.judge(study, kept, why)
    assert made is not None
    if not knockout.finalists:
        return _sheets(study, made, knockout, ctx)
    line = committing.line_of(study)
    state = committing.line_state(_run_folder(study)) if line is not None else LineState()
    experiment = trials.try_settings(
        study,
        [searching.row_changes(study, label) for label in knockout.finalists],
        made.windows,
        current=line.settings if line is not None else None,
        line=state,
        settings=get_settings(),
        config=_config(),
    )
    journaling.experiment(study, experiment, None)
    searching.tried(study, experiment.id)
    named = ", ".join(
        f"{one.label} = {label}"
        for one, label in zip(experiment.tried[1:], knockout.finalists, strict=True)
    )
    said = (
        f"Every sheet is judged. The finalists, tried side by side at full size ({named}):\n"
        + trials.describe(experiment)
    )
    if not _vision(ctx):
        return _content(said)
    seen = looking.look_experiment(study, experiment.id, vision=True)
    return _content(f"{said}\n\n{seen.text}", seen.figures)


def _sheets(
    study: Study,
    made: searching.Search,
    knockout: searching.Knockout | None,
    ctx: Context,
    said: str = "",
) -> list[TextContent | ImageContent]:
    """The knockout's next sheets, searching.A_CALL at most, in words and figures, after `said`
    in a text of its own (the host judges the sheets alone); or where it stands."""
    if knockout is None:
        return _content(said or searching.describe(made, None))
    if knockout.finalists:
        return _content(
            f"Every sheet is judged: the finalists {', '.join(knockout.finalists)}, tried side by "
            f"side as {knockout.experiment} (look experiment shows them again). Refine with "
            "try_settings, then commit."
        )
    head = [TextContent(type="text", text=said)] if said else []
    shown = knockout.left[: searching.A_CALL]
    text = "\n\n".join(searching.sheet_text(made, knockout, sheet) for sheet in shown)
    if not _vision(ctx):
        return [*head, *_content(text)]
    folder = study.folder / searching.SEARCHES / searching.IMAGES
    lengths = {row.label: row.length for row in made.rows}
    figures = [looking.draw_sheet(folder, sheet.labels, lengths) for sheet in shown]
    return [*head, *_content(text, figures)]


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
            description="Why each setting changed, one line each, from the experiments: by "
            "'<stage> <key>' as the changes name it, or '<stage>' for all its keys; a method and "
            "its parameters take one. A setting given at the value it has needs none."
        ),
    ],
    ctx: Context,
    experiment: Annotated[
        str | None, Field(description="The experiment the changes come from, by its id (E<n>).")
    ] = None,
    conclusion: Annotated[
        str | None,
        Field(
            description="What the experiment showed, in words: the journal's conclusion, "
            "resting on it, written with the commit."
        ),
    ] = None,
) -> list[TextContent | ImageContent]:
    """The line made with these changes of its settings: every window one receiver apart, its
    records judged by G1, its images by G2, each picked (as its recipe says, if it has one) and
    judged by G3, then G4 along the line. The first commit makes the run, in PAC's layout; a later
    one changes the run's settings, what it replaces kept in each window's attempts/; a change of
    masw makes a new run, without recipes. With vision, a sheet of its images and picks along the
    line. Once every sheet of the search is judged."""
    study = _study(profile, ctx)
    if (left := searching.unjudged(study)) is not None:
        raise RunError(left)
    of = (experiment,) if experiment else ()
    if conclusion is not None:
        journaling.check(study, "conclusion", conclusion, of)
    committed = committing.commit(study, changes, reasons, experiment, get_settings(), _config())
    written = ""
    if conclusion is not None:
        entry = journaling.write(study, "conclusion", conclusion, of)
        written = f"Written: {journaling.line(entry)}\n"
    journaling.commit(study, committed)
    said = written + committing.describe_commit(committed)
    if not _vision(ctx):
        return _content(said)
    seen = looking.look_line(find_run(committed.line.run_id, get_settings()))
    return _content(f"{said}\n\nIts images and picks: {seen.text}", seen.figures)


Windows = Annotated[str, Field(description=f"The windows: {selecting.SPEC}.")]
# A hypothesis of the journal, by its id.
_HYPOTHESIS = re.compile(r"H\d+")


@server.tool(annotations=paco_server.REPLACES)
@_errors
def pick(
    profile: ProfileName,
    ways: Annotated[
        list[picker.Way],
        Field(
            min_length=1,
            description=f"Each way of picking and its windows ({selecting.SPEC}), a window in one "
            "of them: every fix you have, in one call.",
        ),
    ],
    ctx: Context,
) -> str:
    """Each window's M0 picked as its way says, from the line's picking, then judged by G3; then
    G4 along the line, once. A window keeps its recipe: a later commit picks it the same way."""
    study = _study(profile, ctx)
    line = _line(study)
    folder = find_run(line.run_id, get_settings())
    statuses = deciding.statuses(study, folder, line.run_id)
    recipes: dict[str, picker.Recipe] = {}
    for way in ways:
        for unit in selecting.select(folder, way.windows, statuses):
            if unit in recipes:
                raise ValueError(
                    f"xmid {unit.removeprefix('xmid_')} is in two ways: a window in one of them."
                )
            recipes[unit] = way.recipe
    base = committing.picking_of(study, line, get_settings(), _config())
    picks = picker.pick(study, folder, recipes, base, _config(), get_settings().workers)
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
    flag, and G4), rejected (once tried hard: the line's pick, a pick after it when it carried
    fixable flags, a guide from its neighbours, a lasso if you see the images), or pending. After it, their
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
    after, and the study's working files (its trials, their records) removed."""
    study = _study(profile, ctx)
    line = _line(study)
    made = freezing.freeze(study, find_run(line.run_id, get_settings()), line)
    journaling.freeze(study, made)
    freed = tidying.tidy(study)
    return (
        f"{freezing.describe_frozen(made)}\nThe study's working files removed ({freed / 1e9:.2f} "
        "GB): its trials and their records; the line's run, the curves and the journal stay."
    )


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


@server.tool(annotations=paco_server.READS)
@_errors
def petro_models(profile: ProfileName, ctx: Context) -> qc.PetroChoice:
    """Only if the user asks for soils, N values or the water table, once the curves are frozen:
    the Silex models of the petrophysical inversion, what each was trained on, and how many of
    the frozen curves each covers (why not the others). Choose the one covering the most."""
    made = _frozen(_study(profile, ctx))
    return qc.petro_models(made.run_id, get_settings(), [curve.unit for curve in made.curves])


@server.tool(annotations=paco_server.REPLACES)
@_errors
def invert_petro(
    profile: ProfileName,
    model: Annotated[str, Field(description="A Silex model name from petro_models.")],
    ctx: Context,
) -> str:
    """Only if the user asks: the frozen curves `model` covers inverted into soils, N values and
    the water table, checked by G7 per window and G8 along the line; PAC's petrophysical sections
    written. Says what the models give, and the gates' summary."""

    def report(done: int, total: int) -> None:
        anyio.from_thread.run(ctx.report_progress, done, total, f"{done} of {total} windows")

    settings = get_settings()
    made = _frozen(_study(profile, ctx))
    folder = find_run(made.run_id, settings)
    holder = "PACo's autonomous agent (invert_petro)"
    with run_lock(folder, holder, wait_s=paco_server.WRITER_WAIT_S):
        result, described = qc.invert_petro_line(
            made.run_id, model, settings, report, [curve.unit for curve in made.curves]
        )
    summary, _ = qc.capped(qc.summarize_report(result, ("G7", "G8"), retries=False))
    return f"Soils of run {made.run_id}, from its frozen curves. {described}\n{summary}"


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
        str | None,
        Field(
            description="Required to write a hypothesis, apart from its text: what the "
            "experiment should show."
        ),
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


def _content(text: str, figures: Sequence[bytes] = ()) -> list[TextContent | ImageContent]:
    """A tool's text and its figures, PNG files, as MCP carries them."""
    return [
        TextContent(type="text", text=text),
        *(
            ImageContent(type="image", data=base64.b64encode(png).decode(), mime_type="image/png")
            for png in figures
        ),
    ]


def _vision(ctx: Context) -> bool:
    meta = ctx.request_context.meta if isinstance(ctx, Context) else None
    return isinstance(meta, dict) and meta.get("vision") is True


def _frozen(study: Study) -> freezing.Frozen:
    """The study's frozen curves; refused before the freeze."""
    made = freezing.frozen(study)
    if made is None:
        raise RunError("The curves are not frozen yet: the soils come from the frozen curves.")
    return made


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
