"""Which run a message's work is on, settled by the host before the model acts: the run the
message names (checked to exist), the conversation's own, a new one, or the user's choice
between the runs there. The model is then told the run and the plan; the scope guard keeps its
calls on that run (paco.agent.scope)."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from paco.agent.scope import STAGES, Offer, Scope, Stage

# The runs offered to choose from, the newest first: with a new run, the role's 2 to 4 options.
OFFERED = 3


@dataclass(frozen=True)
class RunInfo:
    """A run as the server lists it."""

    run_id: str
    profile: str
    label: str  # how it was made and how far it went: "active, windows of 7 receivers, curves"
    went: str = "images"  # how far it went: images, curves or models


def to_do(stages: tuple[Stage, ...], run: RunInfo | None) -> tuple[Stage, ...]:
    """The stages to do for `stages` on `run` (None: a new run, processed first): those asked,
    and the curves the models and the soils need when the run has none."""
    todo = set(stages) | ({"process"} if run is None else set())
    has_curves = run is not None and run.went in ("curves", "models")
    if todo & {"invert", "soils"} and not has_curves:
        todo.add("pick")
    return tuple(stage for stage in STAGES if stage in todo)


@dataclass(frozen=True)
class Plan:
    """The turn's work: the stages asked, in their order, on run `run_id`, or on a new run of
    `profile` (`new`); no run for a look, or when the message names no profile."""

    stages: tuple[Stage, ...]
    profile: str | None = None
    run_id: str | None = None
    new: bool = False


@dataclass(frozen=True)
class Ask:
    """The user's choice before anything runs: what is there, the question, and its options."""

    said: str
    question: str
    offers: tuple[Offer, ...]


# A profile's runs with images, the newest first; a run by its id, None when it does not exist;
# the newest runs of every profile.
type RunsOf = Callable[[str], Awaitable[list[RunInfo]]]
type RunOf = Callable[[str], Awaitable[RunInfo | None]]
type Newest = Callable[[], Awaitable[list[RunInfo]]]


@dataclass(frozen=True)
class Lister:
    """How the host reads the runs: from the server's resources."""

    runs_of: RunsOf
    run_of: RunOf
    newest: Newest


async def resolve(
    scope: Scope, current: RunInfo | None, made: frozenset[str], runs: Lister
) -> Plan | Ask:
    """The run `scope`'s work is on, or the question that settles it first. `current` is the
    run the conversation is on, `made` the runs it made."""
    stages: tuple[Stage, ...] = tuple(stage for stage in STAGES if stage in scope.asked)
    if not stages:
        return Plan(stages=())
    if scope.run_id is not None:
        named = await runs.run_of(scope.run_id)
        if named is None:
            return await _not_there(scope, current, runs)
        if "process" not in stages:
            return Plan(to_do(stages, named), named.profile, named.run_id)
        if scope.redo:
            return Plan(to_do(stages, None), named.profile, new=True)
        return Ask(
            f"Run {named.run_id} of {named.profile} is there ({named.label}).",
            "A new run, or this one?",
            (new_run(named.profile, scope), work_on(named)),
        )
    profile = scope.profile or (current.profile if current is not None else None)
    if profile is None:
        return Plan(stages)
    on_current = current is not None and current.profile == profile
    if "process" in stages:
        if scope.redo or (on_current and current is not None and current.run_id in made):
            return Plan(to_do(stages, None), profile, new=True)
        there = await runs.runs_of(profile)
        if not there:
            return Plan(to_do(stages, None), profile, new=True)
        return Ask(
            f"{profile} has {_runs(len(there))}.",
            "A new run, or a run to work on?",
            (new_run(profile, scope), *(work_on(run) for run in there[:OFFERED])),
        )
    if on_current and current is not None:
        return Plan(to_do(stages, current), profile, current.run_id)
    there = await runs.runs_of(profile)
    if not there:
        return Plan(to_do(stages, None), profile, new=True)
    if len(there) == 1:
        return Plan(to_do(stages, there[0]), profile, there[0].run_id)
    return Ask(
        f"{profile} has {_runs(len(there))}.",
        "Which run?",
        tuple(work_on(run) for run in there[:OFFERED]),
    )


async def _not_there(scope: Scope, current: RunInfo | None, runs: Lister) -> Ask:
    """A run named that does not exist: said so, with the runs there to choose from (the
    profile's, else the newest of every profile)."""
    said = f"Run {scope.run_id} does not exist."
    profile = scope.profile or (current.profile if current is not None else None)
    there = await runs.runs_of(profile) if profile is not None else await runs.newest()
    if there:
        whose = f"{profile} has {_runs(len(there))}" if profile else "The newest runs are these"
        return Ask(f"{said} {whose}.", "Which run?", tuple(work_on(run) for run in there[:OFFERED]))
    if profile is not None:
        return Ask(f"{said} {profile} has no run yet.", "A new run?", (new_run(profile, scope),))
    return Ask(f"{said} There is no run yet.", "Which profile should PACo process?", ())


# Each stage after processing, as the plan says it.
_STEP: dict[Stage, str] = {
    "pick": "pick",
    "invert": "invert",
    "soils": "petro_models, then invert_petro",
}


def plan_note(
    scope: Scope, plan: Plan, chosen: Offer | None, host_makes: bool, request: str | None = None
) -> str:
    """What the model reads after the message: its scope, the option the user chose, and the
    plan, the run named once: work on it, and on no other. An option the host asked before
    the model read `request` (the message that got the options): the request, whose settings
    go into the plan's calls."""
    note = scope.for_model(None)
    if chosen is not None:
        made = ": PACo makes it below" if host_makes else ""
        note += f" The user chose ({scope.option}) {chosen.label}{made}."
        if request is not None:
            note += f' Their request: "{request}"; every setting it gives goes into the calls.'
    steps = ", then ".join(_STEP[stage] for stage in plan.stages if stage != "process")
    if plan.run_id is not None:
        of = f" of {plan.profile}" if plan.profile else ""
        note += f" Work on run {plan.run_id}{of}, and on no other run"
        return note + (f": {steps}." if steps else ".")
    if plan.new and plan.profile is not None:
        arguments: dict[str, object] = {"profile": plan.profile}
        if scope.mode is not None:
            arguments["mode"] = scope.mode
        if chosen is not None and chosen.with_request:
            first = f"{chosen.call}, with the request's settings"
        elif host_makes:
            first = "PACo makes it below"
        else:
            first = _call("run_processing", arguments)
        note += f" A new run of {plan.profile} first ({first})"
        return note + (f"; then, on it: {steps}." if steps else ".")
    return note


def new_run(profile: str, scope: Scope) -> Offer:
    """The option of a new run of `profile`, in the mode the message gives: the model makes
    it, with the settings the message gives (a stacking, a mute), which only it reads."""
    arguments: dict[str, object] = {"profile": profile}
    if scope.mode is not None:
        arguments["mode"] = scope.mode
    arguments["again"] = True
    return Offer("a new run", _call("run_processing", arguments), with_request=True)


def work_on(run: RunInfo) -> Offer:
    """The option of working on `run`: the host settles it, no tool to call."""
    return Offer(
        f"work on run {run.run_id} ({run.label})", f'work_on(run_id="{run.run_id}")', run.run_id
    )


def parse_call(call: str) -> tuple[str, dict[str, object]] | None:
    """An offered call (name(key=json, ...), as the options write it) as a tool name and its
    arguments; None when it does not read, or holds a placeholder for the user to fill
    ("<m>")."""
    name, _, rest = call.partition("(")
    if not name.isidentifier() or not rest.endswith(")") or "<" in rest:
        return None
    text = rest[:-1].strip()
    decoder = json.JSONDecoder()
    arguments: dict[str, object] = {}
    while text:
        key, equals, text = text.partition("=")
        if not equals or not key.strip().isidentifier():
            return None
        try:
            value, end = decoder.raw_decode(text)
        except json.JSONDecodeError:
            return None
        arguments[key.strip()] = value
        text = text[end:].lstrip().removeprefix(",").lstrip()
    return name, arguments


def _call(name: str, arguments: dict[str, object]) -> str:
    shown = ", ".join(f"{key}={json.dumps(value)}" for key, value in arguments.items())
    return f"{name}({shown})"


def _runs(count: int) -> str:
    return f"{count} run" + ("s" if count != 1 else "")
