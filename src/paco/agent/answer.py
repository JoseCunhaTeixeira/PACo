"""The answer the user reads, written by code around the model's text (R2): the scope line, what
the model says (an answer form it fills from its draft, constrained to the form's schema), what
was done and left out (from the tools' results), the stages the message asked that no tool did
(U6), the question and the options when the user must choose, else what they can do next; then
the parameters used and the settings the gates changed. Numbers in the model's text that no
result of the turn holds are flagged (R1).

Whether a question is shown comes from the turn, never from its words: a tool offered options,
a tool is stuck, or no work was done yet (a clarification). After work done, the model's
question is left out: what it would have offered, the "Next" line says."""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paco import prompts
from paco.agent.model import ChatModel, Filled
from paco.agent.scope import STAGES, Offer, Scope, offers_in

# The tools that do the request's work, and the stage each serves (job_status: the inversion).
STAGE_OF_TOOL = {
    "run_processing": "images",
    "pick": "curves",
    "judge": "curves",
    "redo": "curves",
    "invert": "models",
    "job_status": "models",
    "invert_petro": "soils",
}
# The stage a redo goes back to, by its `stage` argument.
_REDO_STAGE = {
    "preprocessing": "images",
    "phase_shift": "images",
    "picking": "curves",
    "inversion": "models",
    "petro_inversion": "soils",
}
# What each stage a message asks makes.
_MADE = {"process": "images", "pick": "curves", "invert": "models", "soils": "soils"}
# What the user can ask next, after each stage's work.
_NEXT_ASK = {
    "images": "the curves (ask to pick them)",
    "curves": "the Vs models (ask to invert them)",
    "models": "the soils and the water table (ask for them)",
}
# A tool's error that leaves the request stuck, as the SDK words it with the server's tag.
_STUCK_ERROR = re.compile(r"Error executing tool \w+: \[stuck\]")
# Numbers of a count or an order (3 windows, option 2): not checked against the results.
_SMALL = 10
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
# A run's id, or a job's (a run's with its kind before): a name, its digits no numbers.
_ID = re.compile(r"(?:\w+-)?\d{8}-\d{6}-[0-9a-f]{4}")


class AnswerForm(BaseModel):
    """What the model says, for the user: statements (no question mark is let through), and the
    one question the user must answer, if any."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Neither question mark, the ASCII one nor the full-width one (U+FF1F).
    said: str = Field(pattern=r"^[^?\uff1f]*$", max_length=2500)
    question: str | None = Field(pattern=r"^[^\n]*$", max_length=300)


# The form's JSON schema, as the model's output is constrained to it.
SCHEMA = AnswerForm.model_json_schema()


@dataclass
class Turn:
    """What the tools did in the turn, as their results say it: for the answer's fixed parts."""

    done: list[str] = field(default_factory=list)  # each stage's work, in one line
    left: list[str] = field(default_factory=list)  # the windows left out, with why
    offers: tuple[Offer, ...] = ()  # the options of the last result, if it offered any
    # The options an earlier answer gave, which still wait for the user's choice.
    pending: tuple[Offer, ...] = ()
    stuck: bool = False  # a tool cannot go on without the user
    stages: list[str] = field(default_factory=list)  # the stages done, in order
    used: list[str] = field(default_factory=list)  # the parameters the stages ran with
    changed: list[str] = field(default_factory=list)  # the settings the gates changed
    kept: list[str] = field(default_factory=list)  # the user's settings, kept as given
    # The settings and arguments the calls gave that their tools do not have: left out, said.
    ignored: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)  # every result's text, for R1
    # A comparison of settings made: what a message asking to compare them asks of processing.
    compared: bool = False

    def read(self, name: str, result: str, failed: bool, arguments: str = "") -> None:
        """Take in a tool's result: the call `name`, its text, whether it failed, and the call's
        arguments (a redo's stage)."""
        self.results.append(result)
        if failed:
            self.stuck = self.stuck or _STUCK_ERROR.match(result) is not None
            return
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return
        if not isinstance(parsed, dict):
            return
        # The options a tool offered stand for the turn: a later result offering none (a
        # listing) leaves them.
        if offered := offers_in(result):
            self.offers = offered
        status = parsed.get("status")
        self.stuck = self.stuck or status == "stuck"
        did = parsed.get("did")
        if isinstance(did, str) and did and did not in self.done:
            self.done.append(did)
            self.compared = self.compared or name == "compare"
            if (stage := _stage_of(name, arguments)) is not None:
                self.stages.append(stage)
        for key, into in (
            ("left", self.left),
            ("used", self.used),
            ("changed", self.changed),
            ("kept", self.kept),
            ("ignored", self.ignored),
        ):
            items = parsed.get(key)
            if isinstance(items, list):
                into.extend(str(item) for item in items if str(item) not in into)

    def ignore(self, items: Iterable[str]) -> None:
        """Take in what the host left out of a call: arguments its tool does not declare."""
        self.ignored.extend(item for item in items if item not in self.ignored)

    @property
    def worked(self) -> bool:
        """Whether a stage tool did work in the turn."""
        return bool(self.done)

    def undone(self, asked: Iterable[str]) -> list[str]:
        """The stages `asked` (the scope's) that no tool did in the turn, in their order."""
        made = {*self.stages, *(("images",) if self.compared else ())}
        return [stage for stage in STAGES if stage in asked and _MADE[stage] not in made]


def _stage_of(name: str, arguments: str) -> str | None:
    """The stage a call did: a redo's, the one it went back to."""
    if name == "redo":
        try:
            called = json.loads(arguments)
        except json.JSONDecodeError:
            called = None
        stage = called.get("stage") if isinstance(called, dict) else None
        if isinstance(stage, str) and stage in _REDO_STAGE:
            return _REDO_STAGE[stage]
    return STAGE_OF_TOOL.get(name)


class AnswerError(ValueError):
    """The model filled no valid answer form, twice."""


@dataclass(frozen=True)
class Written:
    """The answer form read, with the model's fills (one, or two when the first did not parse)."""

    form: AnswerForm
    fills: tuple[Filled, ...]


async def write_answer(model: ChatModel, message: str, draft: str) -> Written:
    """The answer form for the model's `draft` to the user's `message`: strictly parsed, a form
    that does not parse sent back once with its error. Raises AnswerError when the second does
    not parse either."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompts.prompt("answer")},
    ]
    for example_message, example_draft, form in prompts.answer_examples():
        messages.append(
            {"role": "user", "content": f"Message: {example_message}\nAnswer: {example_draft}"}
        )
        messages.append({"role": "assistant", "content": json.dumps(form)})
    messages.append({"role": "user", "content": f"Message: {message}\nAnswer: {draft}"})
    fills: list[Filled] = []
    said = ""
    for _ in range(2):
        filled = await model.fill(messages, SCHEMA)
        fills.append(filled)
        try:
            return Written(form=AnswerForm.model_validate_json(filled.content), fills=tuple(fills))
        except ValidationError as error:
            said = "; ".join(item["msg"] for item in error.errors())
            messages.append({"role": "assistant", "content": filled.content})
            messages.append({"role": "user", "content": f"Not valid: {said}. Fill it again."})
    raise AnswerError(said)


def render(
    scope: Scope, said: str, question: str | None, turn: Turn, sources: Sequence[str] = ()
) -> str:
    """The answer the user reads: the scope line, the model's text (`said`), its numbers that
    neither the turn's results nor `sources` (what the conversation holds) nor the scope hold
    flagged, what was done and left out, the question with the options (the turn's, else
    those still waiting when it did no work) or what to ask next, then the parameters used and
    the settings the gates changed (last: PAC's chat folds them)."""
    blocks = [scope.line(), said.strip()]
    if unfound := unfound_numbers(said, [*turn.results, *sources, scope.line()]):
        blocks.append(f"Numbers not found in the tools' results: {', '.join(unfound)}.")
    if turn.done:
        blocks.append("Done:\n" + "\n".join(f"- {line}" for line in turn.done))
    if turn.left:
        blocks.append("Left out:\n" + "\n".join(f"- {line}" for line in turn.left))
    if turn.ignored:
        # What the calls gave that their tools do not have: the calls ran without it (U6).
        blocks.append("Ignored:\n" + "\n".join(f"- {line}" for line in turn.ignored))
    offers = turn.offers or (() if turn.worked else turn.pending)
    asks = bool(offers) or bool(question and (turn.stuck or (not turn.worked and scope.asked)))
    # The stages asked that no tool did, said (U6); when the user must choose, the question
    # says why.
    if not asks and (undone := turn.undone(scope.asked)):
        blocks.append(f"Asked but not done: {', '.join(undone)}.")
    if offers:
        options = "\n".join(f"({i}) {offer.label}" for i, offer in enumerate(offers, 1))
        blocks.append(f"{question or 'Which do you choose?'}\n{options}")
    elif question and (turn.stuck or (not turn.worked and scope.asked)):
        # A question when the user must answer it: stuck, or a request for work left unclear.
        # A message asking for no work (a look, something no tool makes) gets an answer.
        blocks.append(question)
    elif turn.worked and (next_line := _next(turn, scope)):
        blocks.append(next_line)
    for title, items in (
        ("Parameters used", turn.used),
        ("Settings the gates changed", turn.changed),
        ("Settings kept as you gave them", turn.kept),
    ):
        if items:
            blocks.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(block for block in blocks if block)


def unfound_numbers(text: str, sources: Iterable[str]) -> list[str]:
    """The numbers of `text` (10 and over, or with decimals; a run's or a job's id is a name)
    that none of `sources` holds, to the text's precision (12.5 is found in 12.46)."""
    found = [_value(number) for source in sources for number in _NUMBER.findall(source)]
    unfound: list[str] = []
    for number in _NUMBER.findall(_ID.sub(" ", text)):
        value = _value(number)
        decimals = len(number.replace(",", ".").partition(".")[2])
        if "." not in number.replace(",", ".") and value < _SMALL:
            continue
        tolerance = 0.5 * 10.0**-decimals
        known = any(abs(value - other) <= tolerance + 1e-9 for other in found)
        if not known and number not in unfound:
            unfound.append(number)
    return unfound


# A window's line when its gate asked to change a setting the user gave (qc.budgets).
LOCKED = "locked, asks"


def _next(turn: Turn, scope: Scope) -> str:
    """What the user can do next, after the turn's work: the windows a setting they gave left
    out, to ask again with the change a gate asked; the stage after the last one done, when the
    message did not ask it; and where PAC shows the results."""
    last = turn.stages[-1] if turn.stages else None
    asked = {"curves": "pick", "models": "invert", "soils": "soils"}
    after = {"images": "curves", "curves": "models", "models": "soils"}.get(last or "")
    parts: list[str] = []
    if any(LOCKED in line for line in turn.left):
        parts.append("for the windows left out over a setting you gave, ask again with the change")
    if last in _NEXT_ASK and (after is None or asked[after] not in scope.asked):
        parts.append(_NEXT_ASK[last])
    parts.append("the results in PAC's Visualization page")
    return "Next: " + "; ".join(parts) + "."


def _value(number: str) -> float:
    return float(number.replace(",", "."))
