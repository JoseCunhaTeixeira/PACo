"""What the user chooses before a stage redoes work already there, or changes work they made by
hand: the stage tools meet it, say what is there, and give the options, each with the call it
makes, doing nothing. The agent reads the request: when it says which option, the agent calls
it; when it asks only for later stages, the agent goes on from the run; otherwise it asks.

Records, images, curves, models and soil columns: asked to process a profile that has a run, to
pick a run that has curves, to invert one that has models (seismic or soil columns), the user
chooses whether to start again, complete what is missing, go on from what is there, or work on
some windows. A step that would change work made by hand in PAC's pages (paco.qc.origin) asks
whether to keep it or replace it (theirs set aside in the window's by_hand/ folder), and
replaces it only once the user could reply: the question shown in an earlier turn of the
conversation, over those windows. Work the assistant did in the same conversation is its own to
go on with, never asked about. The host sends the conversation's id and turn with each call.

With the scope of the user's message (paco.agent.scope), which PACo's host sends too, the rules
are the user's words, read by the model and applied here: a stage the message does not ask
goes on from the run's work without a question; one it asks whose work is there is asked
about, unless the message asks to redo it; `again`, `windows="all"` and `hand="replace"` hold
only when the message asked for them, in its words or by the option it chose."""

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError
from sigpipe.masw.runs import RunError, RunManifest, find_run, list_runs, load_manifest

from paco.qc.curves import imaged_windows
from paco.qc.inverting import invertible
from paco.qc.log import read_attempts
from paco.qc.origin import WindowWork, assistant_run, run_work
from paco.qc.report import stretches
from paco.settings import Settings

type Windows = Literal["all", "missing"] | None
type Hand = Literal["keep", "replace"] | None
# The stage of the user's message each tool that changes work serves (paco.agent.scope).
_STAGE_OF_TOOL = {
    "run_processing": "process",
    "pick": "pick",
    "invert": "invert",
    "invert_petro": "soils",
}
# The stage of the user's message each stage redo goes back to belongs to.
STAGE_OF_REDO = {
    "preprocessing": "process",
    "phase_shift": "process",
    "picking": "pick",
    "inversion": "invert",
}


@dataclass(frozen=True)
class Asked:
    """The scope of the message a call serves, as PACo's host sends it: the stages it asks,
    whether it asks to do again work already there, what it says of the work made by hand, the
    offered call it chose, and the positions it named."""

    stages: frozenset[str]
    redo: bool = False
    hand_work: Literal["replace", "unsaid"] = "unsaid"
    chosen: str | None = None
    positions: tuple[float, ...] = ()  # the positions the message named (m)
    # The windows the message gave, as masw overrides (length, step; in metres: length_m, step_m).
    window: Mapping[str, float] = field(default_factory=dict)
    # The window lengths the message compares, in receivers ("length") or metres ("length_m").
    compared: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def chose(self, tool: str, argument: str) -> bool:
        """Whether the message asked `tool` for `argument`, as the options write it
        (`again=true`, `windows="all"`): by asking to redo a stage it asks, that tool's, or by
        choosing an offered call of that tool with it."""
        return (self.redo and _STAGE_OF_TOOL[tool] in self.stages) or self._offered(tool, argument)

    def replaces(self, tool: str, stage: str) -> bool:
        """Whether the message asked to replace the work made by hand that `tool` would change
        at `stage`: in its words, for a stage it asks, or by choosing that tool's offered call
        that replaces it."""
        said = self.hand_work == "replace" and stage in self.stages
        return said or self._offered(tool, 'hand="replace"')

    def keeps(self, tool: str) -> bool:
        """Whether the message chose `tool`'s offered call that keeps the work made by hand."""
        return self._offered(tool, 'hand="keep"')

    def _offered(self, tool: str, argument: str) -> bool:
        chosen = self.chosen
        return chosen is not None and chosen.startswith(f"{tool}(") and argument in chosen


@dataclass(frozen=True)
class Conversation:
    """The conversation a call belongs to, its turn (the user's messages so far) and the scope
    of the turn's message, as the host sends them; None from a client that sends none."""

    id: str | None = None
    turn: int | None = None
    asked: Asked | None = None

    def allows(self, tool: str, argument: str) -> bool:
        """Whether `tool` may take `argument`, which makes work there again: always from a
        client that sends no scope, else as the message asked."""
        return self.asked is None or self.asked.chose(tool, argument)

    def hand(self, hand: Hand, tool: str, stage: str) -> Hand:
        """`tool`'s `hand` at `stage` as the message allows it: replace only when the message
        asked it (its words, or the option it chose), keep only when it chose the option that
        keeps it, else the question comes (keep or replace is the user's to say, not the
        model's); replace when it asked it, whatever the call said (the user first)."""
        asked = self.asked
        if asked is None:
            return hand
        if asked.replaces(tool, stage):
            return "replace"
        if hand == "keep" and asked.keeps(tool):
            return "keep"
        return None


# The runs the assistant worked on in each conversation, by the id its host sends.
_WORKED: dict[str, set[str]] = defaultdict(set)
# Where the question over work made by hand was shown, by conversation, run and step: its turn
# and windows.
_SHOWN: dict[tuple[str | None, str, str], tuple[int | None, frozenset[str]]] = {}


def worked(conversation: Conversation, run_id: str) -> None:
    """Record that the assistant worked on run `run_id` in `conversation`."""
    if conversation.id is not None:
        _WORKED[conversation.id].add(run_id)


def earlier(conversation: Conversation, run_id: str) -> bool:
    """Whether run `run_id`'s work is from before `conversation` (an unknown conversation: all
    of it)."""
    return conversation.id is None or run_id not in _WORKED[conversation.id]


@dataclass(frozen=True)
class Choice:
    """What a stage tool returns instead of acting: what is there, and the user's options, each
    with its call (none: the way to go on, without a question)."""

    run_id: str
    summary: str
    next: str
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Plan:
    """What a stage does, the user's choice made: its windows (None: its own default) and
    whether a person's work in them is replaced."""

    units: list[str] | None
    replace_hand: bool = False


def call(name: str, **arguments: Any) -> str:  # noqa: ANN401
    """A tool call as the options write it: name(argument=value, ...)."""
    shown = ", ".join(f"{key}={json.dumps(value)}" for key, value in arguments.items())
    return f"{name}({shown})"


def choose(
    run_id: str,
    what: str,
    asked: str,
    options: Sequence[tuple[str, str]],
    later: tuple[str, str] | None = None,
) -> Choice:
    """The choice a request `asked` (for a stage whose work is there) leaves to the user: `what`
    is there, each option and its call, and `later`, the stages after it and the call that goes
    on with them."""
    listed = "; ".join(f"({i}) {label}: {made}" for i, (label, made) in enumerate(options, 1))
    goes_on = f" Asked only {later[0]}: go on without asking, {later[1]}." if later else ""
    return Choice(
        run_id=run_id,
        summary=what,
        next=f"Nothing was done. Asked {asked}, the user chooses how, unless their request says "
        f"which option: ask them with these options, then call as they choose: {listed}."
        f"{goes_on}",
        options=tuple(options),
    )


def go_on(run_id: str, what: str, why: str, then: str) -> Choice:
    """The way on when the message does not ask to make the run's work again: `why`, and the
    call `then` that goes on from it, without a question."""
    return Choice(
        run_id=run_id,
        summary=what,
        next=f"Nothing was done: {why}. Go on from this run without asking: {then}.",
    )


def held(run_folder: Path, manifest: RunManifest, work: dict[str, WindowWork]) -> str:
    """What a run holds, in one sentence: who made it, its images, curves, models, soils."""
    maker = "the assistant's" if assistant_run(read_attempts(run_folder)) else "made in PAC"
    count = _Counts(work)
    parts = [_of(count.images, "images")]
    parts.append(_of(count.curves, "M0 curves", count.hand_curves))
    if count.higher:
        parts.append(_of(count.higher, "higher-mode curves", count.higher))
    parts.append(_of(count.models, "models", count.hand_models))
    if count.soils:
        parts.append(_of(count.soils, "soil columns", count.hand_soils))
    return (
        f"Run {manifest.run_id} ({maker}, {manifest.started_at:%Y-%m-%d %H:%M}) holds "
        + ", ".join(parts)
        + "."
    )


def processing(
    profile: str,
    conversation: Conversation,
    settings: Settings,
    given: dict[str, Any],
) -> Choice | None:
    """Asked to process `profile` (with `given`, the call's other arguments): its latest run
    from before this conversation, when it has one, and the user's options (start again, or go
    on from it); None to process, as when the conversation made a run of it already."""
    found = [one.split("/", 1) for one in list_runs(settings)]
    ids = [run_id for name, run_id in found if name == profile]
    asked = conversation.asked
    if asked is not None and "process" not in asked.stages:
        return _processed(profile, ids, settings, asked.stages)
    if any(not earlier(conversation, run_id) for run_id in ids) or (asked and asked.redo):
        return None
    for run_id, run_folder, manifest in _readable(ids, settings):
        work = run_work(run_folder, manifest)
        if not any(one.image is not None for one in work.values()):
            continue
        count = _Counts(work)
        options = _goes_on(run_id, count)
        again = call("run_processing", profile=profile, **given, again=True)
        options.append(("process again, a new run (this one stays)", again))
        what = held(run_folder, manifest, work)
        later = _later(run_id, count) if asked is None else None
        return choose(run_id, what, "to process", options, later)
    return None


def _processed(
    profile: str, ids: Sequence[str], settings: Settings, stages: frozenset[str]
) -> Choice | None:
    """Processing asked only for the later `stages`: the way on from the profile's latest run
    with images, to the first stage asked (whose tool says what is there); None to process, the
    profile having none."""
    for run_id, run_folder, manifest in _readable(ids, settings):
        work = run_work(run_folder, manifest)
        if any(one.image is not None for one in work.values()):
            tools = (("pick", "pick"), ("invert", "invert"), ("soils", "petro_models"))
            then = next(
                (call(tool, run_id=run_id) for stage, tool in tools if stage in stages),
                call("inspect", what="run", run_id=run_id),
            )
            why = f"this message does not ask to process {profile}, whose run {run_id} is there"
            return go_on(run_id, held(run_folder, manifest, work), why, then)
    return None


def _readable(ids: Sequence[str], settings: Settings) -> Iterator[tuple[str, Path, RunManifest]]:
    """The runs among `ids` whose manifest reads, with their folder: one still being written,
    or broken, is not gone on from."""
    for run_id in ids:
        try:
            yield run_id, find_run(run_id, settings), load_manifest(run_id, settings)
        except RunError, ValidationError:
            continue


def picking(
    run_folder: Path,
    manifest: RunManifest,
    conversation: Conversation,
    units: list[str] | None,
    windows: Windows,
    hand: Hand,
    given: dict[str, Any] | None = None,
) -> Choice | Plan:
    """Asked to pick run `manifest.run_id` (`units`, the windows at the positions asked, or the
    whole line; `windows` narrows them), with `given`, the call's other arguments: the user's
    choice when it has curves from before this conversation and they did not say which
    windows, or when a curve picked by hand would be picked again; else the windows to pick."""
    run_id = manifest.run_id
    given = given or {}
    hand = conversation.hand(hand, "pick", "pick")
    if windows == "all" and not conversation.allows("pick", 'windows="all"'):
        windows = None  # the message did not ask to pick its curves again
    work = run_work(run_folder, manifest)
    count = _Counts(work)
    asked = conversation.asked
    if asked is not None and "pick" not in asked.stages:
        # Picking serves a later stage: the windows without a curve alone, which it needs.
        ready = imaged_windows(run_folder, manifest)
        among = units if units is not None else list(ready)
        lacking = [unit for unit in among if unit in ready and work[unit].m0 is None]
        if not lacking:
            then = (
                call("invert", run_id=run_id)
                if "invert" in asked.stages
                else call("petro_models", run_id=run_id)
            )
            why = "this message does not ask to pick, and the run's curves are there"
            return go_on(run_id, held(run_folder, manifest, work), why, then)
        first = units is None and len(lacking) == len(ready)  # the line's first pick
        units, windows = (None if first else lacking), None
    if units is None and windows is None and count.curves and earlier(conversation, run_id):
        if asked is not None and asked.redo:
            return picking(run_folder, manifest, conversation, None, "all", hand, given)
        options = [
            ("pick every window again", call("pick", run_id=run_id, **given, windows="all")),
            ("invert the curves as they are", call("invert", run_id=run_id)),
            ("some windows only", call("pick", run_id=run_id, **given, positions=["<m>"])),
        ]
        if count.missing_curves:
            done = f"complete the {len(count.missing_curves)} windows without a curve"
            options.insert(0, (done, call("pick", run_id=run_id, **given, windows="missing")))
        what = held(run_folder, manifest, work)
        later = ("to invert", call("invert", run_id=run_id)) if asked is None else None
        return choose(run_id, what, "to pick", options, later)
    asked_at = units
    if windows == "missing":
        ready = imaged_windows(run_folder, manifest)
        among = asked_at if asked_at is not None else ready
        units = [unit for unit in among if unit in ready and work[unit].m0 is None]
        if not units:
            where = "window asked" if asked_at is not None else "window"
            raise RunError(f"Run '{run_id}' has a curve in every {where} with an image.")
    elif windows == "all" and units is None and count.curves:
        units = list(imaged_windows(run_folder, manifest))
    targets = units if units is not None else list(work)
    by_hand = [unit for unit in targets if work[unit].m0 == "user"]
    if asked_at is None and windows is None:
        by_hand = []  # the whole line's default pick keeps them, asking nothing
    step = _Step(conversation, run_id, "pick", by_hand)
    if not step.decided(hand):
        asked = {"run_id": run_id, **given, **({"windows": windows} if windows else {})}
        return step.question("curves picked by hand", asked)
    return Plan(units=units, replace_hand=bool(by_hand) and hand == "replace")


def inverting(
    run_folder: Path,
    manifest: RunManifest,
    conversation: Conversation,
    units: list[str] | None,
    windows: Windows,
    hand: Hand,
    given: dict[str, Any] | None = None,
) -> Choice | Plan:
    """Asked to invert run `manifest.run_id` (`units`, the windows at the positions asked, or
    the whole line; `windows` narrows them), with `given`, the call's other arguments: the
    user's choice when it has models from before this conversation and they did not say which
    windows, or when a model made by hand would be made again; else the windows to invert
    (None: those without a model)."""
    run_id = manifest.run_id
    given = given or {}
    hand = conversation.hand(hand, "invert", "invert")
    if windows == "all" and not conversation.allows("invert", 'windows="all"'):
        windows = None  # the message did not ask to invert its windows again
    work = run_work(run_folder, manifest)
    count = _Counts(work)
    asked = conversation.asked
    if units is None and windows is None and count.models and earlier(conversation, run_id):
        if asked is not None and asked.redo:
            return inverting(run_folder, manifest, conversation, None, "all", hand, given)
        options = [
            ("invert every window again", call("invert", run_id=run_id, **given, windows="all")),
            ("some windows only", call("invert", run_id=run_id, **given, positions=["<m>"])),
        ]
        missing = _without_model(run_folder, manifest, work)
        if missing:
            done = f"complete the {len(missing)} windows without a model"
            options.insert(0, (done, call("invert", run_id=run_id, **given, windows="missing")))
        what = held(run_folder, manifest, work)
        soils = ("for soils", call("petro_models", run_id=run_id)) if asked is None else None
        return choose(run_id, what, "to invert", options, soils)
    if windows == "missing" and units is not None:
        units = [unit for unit in units if work[unit].model is None]
        if not units:
            raise RunError(f"Run '{run_id}' has a model in every window asked.")
    elif windows == "missing":
        units = None  # the job's own: the windows without a model
    elif windows == "all" and units is None:
        units = list(invertible(run_folder, manifest))
    targets = units if units is not None else []
    by_hand = [unit for unit in targets if work[unit].model == "user"]
    step = _Step(conversation, run_id, "invert", by_hand)
    if not step.decided(hand):
        asked = {"run_id": run_id, **given, **({"windows": windows} if windows else {})}
        return step.question("models made by hand", asked)
    if by_hand and hand == "keep" and units is not None:
        units = [unit for unit in units if unit not in by_hand]
    return Plan(units=units, replace_hand=bool(by_hand) and hand == "replace")


def soils(
    run_folder: Path,
    manifest: RunManifest,
    conversation: Conversation,
    model: str,
    windows: Windows,
    hand: Hand,
) -> Choice | Plan:
    """Asked for run `manifest.run_id`'s soils: the user's choice when it has soil columns from
    before this conversation and they did not say which windows, or when a soil column made by
    hand would be made again; else the windows to invert (None: the whole line, a new inversion
    replacing the last one)."""
    run_id = manifest.run_id
    hand = conversation.hand(hand, "invert_petro", "soils")
    if windows == "all" and not conversation.allows("invert_petro", 'windows="all"'):
        windows = None  # the message did not ask for the soils again
    work = run_work(run_folder, manifest)
    count = _Counts(work)
    if windows is None and count.soils and earlier(conversation, run_id):
        if conversation.asked is not None and conversation.asked.redo:
            return soils(run_folder, manifest, conversation, model, "all", hand)
        options = [
            (
                "a new inversion of the whole line",
                call("invert_petro", run_id=run_id, model=model, windows="all"),
            )
        ]
        curves = invertible(run_folder, manifest)
        missing = [unit for unit in curves if work[unit].soils is None]
        if missing:
            done = f"complete the {len(missing)} windows without a soil column"
            options.insert(
                0, (done, call("invert_petro", run_id=run_id, model=model, windows="missing"))
            )
        return choose(run_id, held(run_folder, manifest, work), "for soils", options)
    units: list[str] | None = None
    if windows == "missing":
        units = [unit for unit in invertible(run_folder, manifest) if work[unit].soils is None]
    by_hand = [
        unit for unit in (units if units is not None else list(work)) if work[unit].soils == "user"
    ]
    step = _Step(conversation, run_id, "invert_petro", by_hand)
    if not step.decided(hand):
        asked = {"run_id": run_id, "model": model, "windows": windows or "all"}
        return step.question("soil columns made by hand", asked)
    return Plan(units=units, replace_hand=bool(by_hand) and hand == "replace")


def redoing(
    conversation: Conversation,
    run_id: str,
    stage: str,
    units: list[str],
    work: dict[str, WindowWork],
    hand: Hand,
    changes: dict[str, Any] | None,
) -> Choice | None:
    """Asked to redo `stage` for `units` with `changes`: the user's choice when it would change
    work made by hand there (a new image, a curve picked again, a model made again); None to
    redo."""
    hand = conversation.hand(hand, "redo", STAGE_OF_REDO[stage])
    if stage == "picking":
        by_hand = [unit for unit in units if work[unit].m0 == "user"]
    elif stage == "inversion":
        by_hand = [unit for unit in units if work[unit].model == "user"]
    else:
        by_hand = [unit for unit in units if work[unit].frozen]
    step = _Step(conversation, run_id, f"redo {stage}", by_hand)
    if step.decided(hand):
        return None
    asked: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "xmids": [_xmid(unit) for unit in units],
        **({"changes": changes} if changes else {}),
    }
    return step.question("work made by hand", asked, "redo")


class _Counts:
    """A run's work counted: its images, curves, models and soil columns, and who made them."""

    def __init__(self, work: dict[str, WindowWork]) -> None:
        ones = list(work.values())
        self.images = sum(one.image is not None for one in ones)
        self.curves = sum(one.m0 is not None for one in ones)
        self.hand_curves = sum(one.m0 == "user" for one in ones)
        self.higher = sum(len(one.modes) - (one.m0 is not None) for one in ones)
        self.models = sum(one.model is not None for one in ones)
        self.hand_models = sum(one.model == "user" for one in ones)
        self.soils = sum(one.soils is not None for one in ones)
        self.hand_soils = sum(one.soils == "user" for one in ones)
        self.missing_curves = [
            unit for unit, one in work.items() if one.image is not None and one.m0 is None
        ]


class _Step:
    """A step over windows holding work made by hand (`by_hand`): whether the user's choice is
    made, and the question that asks it."""

    def __init__(
        self, conversation: Conversation, run_id: str, name: str, by_hand: list[str]
    ) -> None:
        self.conversation = conversation
        self.run_id = run_id
        self.by_hand = by_hand
        self.key = (conversation.id, run_id, name)

    def decided(self, hand: Hand) -> bool:
        """Whether the step may run with `hand`: no work made by hand in its windows, keep, or
        replace once the user could reply to the question over these windows (shown in an
        earlier turn). The choice serves one step."""
        if not self.by_hand:
            return True
        shown = _SHOWN.get(self.key)
        if hand == "keep":
            _SHOWN.pop(self.key, None)
            return True
        if hand == "replace" and self.conversation.asked is not None:
            return True  # the message asked it (Conversation.hand)
        if hand != "replace" or shown is None:
            return False
        turn, windows = shown
        now = self.conversation.turn
        if not set(self.by_hand) <= windows or (
            turn is not None and now is not None and now <= turn
        ):
            return False
        del _SHOWN[self.key]
        return True

    def question(self, what: str, asked: dict[str, Any], tool: str | None = None) -> Choice:
        """The question over the work made by hand: keep it, or replace it, each with its call
        (`tool`, by default the step's, with `asked`); the turn it is shown in kept."""
        shown = _SHOWN.get(self.key)
        if shown is None or not set(self.by_hand) <= shown[1]:
            _SHOWN[self.key] = (self.conversation.turn, frozenset(self.by_hand))
        tool = tool or self.key[2]
        where = stretches([_xmid(unit) for unit in self.by_hand])
        keep = call(tool, **asked, hand="keep")
        replace = call(tool, **asked, hand="replace")
        return Choice(
            run_id=self.run_id,
            options=(
                ("keep it as it is", keep),
                ("replace it, theirs set aside in the window's by_hand folder", replace),
            ),
            summary=f"{what[0].upper()}{what[1:]} at {where}, which {tool} would change.",
            next="Nothing was done: this work is the user's, made in PAC and verified by them. "
            "Ask them whether to keep it or replace it, then call as they choose; it is "
            "replaced only once they have replied: (1) keep it as it is: "
            f"{keep}; (2) replace it, theirs set aside in the window's by_hand folder: "
            f"{replace}.",
        )


def _later(run_id: str, count: _Counts) -> tuple[str, str] | None:
    """The stages after a run's work, and the call that goes on with them: its images picked,
    its curves inverted, its models' soils; none after its soil columns."""
    if not count.curves:
        return "to pick or invert", call("pick", run_id=run_id)
    if not count.models:
        return "to invert", call("invert", run_id=run_id)
    if not count.soils:
        return "for soils", call("petro_models", run_id=run_id)
    return None


def _goes_on(run_id: str, count: _Counts) -> list[tuple[str, str]]:
    """The ways to go on from a run: its curves inverted, its picking redone or completed, its
    images picked, or some windows only."""
    if not count.curves:
        return [("go on from these images", call("pick", run_id=run_id))]
    options: list[tuple[str, str]] = []
    if count.models:
        options.append(("invert every window again", call("invert", run_id=run_id, windows="all")))
    else:
        options.append(("invert the curves as they are", call("invert", run_id=run_id)))
    options.append(("pick every window again", call("pick", run_id=run_id, windows="all")))
    if count.missing_curves:
        done = f"complete the {len(count.missing_curves)} windows without a curve"
        options.append((done, call("pick", run_id=run_id, windows="missing")))
    options.append(("some windows only", call("pick", run_id=run_id, positions=["<m>"])))
    return options


def _without_model(
    run_folder: Path, manifest: RunManifest, work: dict[str, WindowWork]
) -> list[str]:
    """The windows whose curve the inversion takes, without a model."""
    try:
        curves = invertible(run_folder, manifest)
    except RunError:
        return []
    return [unit for unit in curves if work[unit].model is None]


def _xmid(unit: str) -> float:
    return float(unit.removeprefix("xmid_"))


def _of(count: int, what: str, hand: int = 0) -> str:
    """`count` `what` (a plural, its singular for one), with how many of them a person made."""
    said = what if count != 1 else what.removesuffix("s")
    return f"{count} {said}" + (f" ({hand} by hand)" if hand else "")
