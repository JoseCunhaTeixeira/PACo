"""The scope of each of the user's messages: what it asks, read by the model into a form (its
JSON schema constrains the model's output) and checked in code. The host keeps the model's
calls within it, and the server applies the user's rules with it (paco.choices): go on from
the work already there, ask before making it again, replace work made by hand only when asked.
Nothing here reads the user's words: the model does."""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paco import prompts
from paco.agent.model import ChatModel, Filled

# The stages a message can ask for, in the pipeline's order.
type Stage = Literal["process", "pick", "invert", "soils"]
STAGES: tuple[Stage, ...] = ("process", "pick", "invert", "soils")
# The tools that change nothing: within every scope.
READ_ONLY = frozenset(
    {"inspect", "preset_settings", "inversion_settings", "job_status", "petro_models"}
)
# The stages a changing tool serves: it is within a scope that asks one of them. An earlier
# stage serves the later ones, which need its work when the run lacks it (the server decides,
# with the run's state).
_SERVES: dict[str, frozenset[Stage]] = {
    "run_processing": frozenset(STAGES),
    "pick": frozenset({"pick", "invert", "soils"}),
    "judge": frozenset({"pick", "invert", "soils"}),
    "invert": frozenset({"invert"}),
    "invert_petro": frozenset({"soils"}),
}
_REDO_SERVES: dict[str, frozenset[Stage]] = {
    "preprocessing": frozenset(STAGES),
    "phase_shift": frozenset(STAGES),
    "picking": frozenset({"pick", "invert", "soils"}),
    "inversion": frozenset({"invert"}),
}
# A run id as PACo gives them: when it started, and a short random suffix.
_RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")


class Scope(BaseModel):
    """What one message asks, as the model filled the form (every field required: the schema
    makes the model write each)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    process: bool = Field(description="The message asks to process records into images.")
    pick: bool = Field(description="It asks for dispersion curves.")
    invert: bool = Field(description="It asks for Vs models.")
    soils: bool = Field(description="It asks for soils or the water table.")
    profile: str | None = Field(description="The profile's name as written.")
    run_id: str | None = Field(description="A run id as written.")
    positions_m: list[float] = Field(
        max_length=10, description="Positions along the line the message names, in metres."
    )
    redo: bool = Field(description="It asks to do again work already done.")
    replace_hand_work: bool = Field(description="It asks to replace work made by hand.")
    option: int | None = Field(description="The option it chooses among those offered last.")

    @property
    def asked(self) -> frozenset[Stage]:
        """The stages the message asks for (none: it only looks at what exists)."""
        return frozenset(stage for stage in STAGES if getattr(self, stage))

    def line(self) -> str:
        """The scope as the answer starts with it, for the user to check."""
        stages = _said(self.asked)
        parts = [stages or "look only"]
        if self.run_id or self.profile:
            parts.append(f"run {self.run_id}" if self.run_id else str(self.profile))
        if self.positions_m:
            parts.append("at " + ", ".join(f"{position:g}" for position in self.positions_m) + " m")
        if self.redo:
            parts.append("again")
        if self.replace_hand_work:
            parts.append("your hand work replaced")
        if self.option is not None:
            parts.append(f"option {self.option}")
        return "Scope: " + " · ".join(parts) + "."

    def for_model(self, chosen: Offer | None) -> str:
        """What the model reads after the message: its scope, and the option chosen with its
        call."""
        stages = _said(self.asked)
        said = [f"asks {stages}" if stages else "asks to look at what exists, running nothing"]
        if self.run_id or self.profile:
            said.append(f"run {self.run_id}" if self.run_id else f"profile {self.profile}")
        if self.positions_m:
            said.append("positions " + ", ".join(f"{p:g}" for p in self.positions_m) + " m")
        said.append("to do again" if self.redo else "not to do again")
        said.append(
            "to replace the work made by hand"
            if self.replace_hand_work
            else "not to replace the work made by hand"
        )
        note = "[PACo] This message " + "; ".join(said) + "."
        if chosen is not None:
            note += f" The user chose ({self.option}) {chosen.label}: {chosen.call}."
        return note

    def for_server(self) -> dict[str, Any]:
        """The scope as each call carries it, for the server's rules (paco.choices)."""
        return {
            "asked": [stage for stage in STAGES if stage in self.asked],
            "redo": self.redo,
            "hand_work": "replace" if self.replace_hand_work else "unsaid",
            "positions_m": list(self.positions_m),
        }


# The form's JSON schema, as the model's output is constrained to it.
SCHEMA: dict[str, Any] = Scope.model_json_schema()
# A message asking nothing: the worked examples say what differs from it.
_BLANK: dict[str, Any] = {
    "process": False,
    "pick": False,
    "invert": False,
    "soils": False,
    "profile": None,
    "run_id": None,
    "positions_m": [],
    "redo": False,
    "replace_hand_work": False,
    "option": None,
}


@dataclass(frozen=True)
class Offer:
    """An option a tool offered the user: its words, and the call that makes it."""

    label: str
    call: str


@dataclass(frozen=True)
class Context:
    """What the host knows when a message comes: the options offered last, and the profile and
    run the conversation is on."""

    offers: tuple[Offer, ...] = ()
    profile: str | None = None
    run_id: str | None = None


class ScopeError(ValueError):
    """The model filled no valid form, twice."""


@dataclass(frozen=True)
class Read:
    """A scope read, with the model's fills (one, or two when the first was not valid)."""

    scope: Scope
    fills: tuple[Filled, ...]


async def read_scope(model: ChatModel, message: str, context: Context) -> Read:
    """The scope of `message`: the model fills the form, strictly parsed; a form that does not
    parse is sent back once with its error. Raises ScopeError when the second does not parse
    either."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompts.prompt("scope")},
        *_examples(),
        {"role": "user", "content": _with_context(message, context)},
    ]
    fills: list[Filled] = []
    said = ""
    for _ in range(2):
        filled = await model.fill(messages, SCHEMA)
        fills.append(filled)
        try:
            scope = Scope.model_validate_json(filled.content)
        except ValidationError as error:
            said = _errors(error)
            messages.append({"role": "assistant", "content": filled.content})
            messages.append({"role": "user", "content": f"Not valid: {said}. Fill the form again."})
            continue
        return Read(scope=checked(scope, context), fills=tuple(fills))
    raise ScopeError(said)


def checked(scope: Scope, context: Context) -> Scope:
    """`scope` with what code can tell is not so taken out: a run id PACo never gives, an
    option never offered, a negative position."""
    run_id = scope.run_id if scope.run_id and _RUN_ID.fullmatch(scope.run_id) else None
    option = scope.option if scope.option and 1 <= scope.option <= len(context.offers) else None
    positions = [position for position in scope.positions_m if position >= 0]
    return scope.model_copy(update={"run_id": run_id, "option": option, "positions_m": positions})


def chosen(scope: Scope, context: Context) -> Offer | None:
    """The offer the message chose, if any."""
    return context.offers[scope.option - 1] if scope.option is not None else None


def refusal(scope: Scope, name: str, arguments: str) -> str | None:
    """Why the call `name(arguments)` is outside `scope`, said to the model; None when it is
    within it."""
    if name in READ_ONLY:
        return None
    serves = _SERVES.get(name)
    if name == "redo":
        serves = _REDO_SERVES.get(_argument(arguments, "stage") or "", frozenset(STAGES))
    if serves is None or serves & scope.asked:
        return None  # an unknown tool: the server says so
    asked = _said(scope.asked)
    what = f"asks {asked}" if asked else "asks to look at what exists, running nothing"
    return (
        f"Not called: this message {what}, and {name} is outside it. Do what it asks, or ask "
        "the user whether they want more."
    )


def offers_in(result: str) -> tuple[Offer, ...]:
    """The options a tool's result offers the user (none: it offers nothing)."""
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return ()
    options = parsed.get("options") if isinstance(parsed, dict) else None
    if not isinstance(options, list):
        return ()
    return tuple(
        Offer(label=str(option["label"]), call=str(option["call"]))
        for option in options
        if isinstance(option, dict) and "label" in option and "call" in option
    )


def _examples() -> list[ChatCompletionMessageParam]:
    """The form's worked examples, as earlier messages and the forms filled for them."""
    turns: list[ChatCompletionMessageParam] = []
    for message, form in prompts.scope_examples():
        turns.append({"role": "user", "content": f"Message: {message}"})
        turns.append({"role": "assistant", "content": json.dumps({**_BLANK, **form})})
    return turns


def _with_context(message: str, context: Context) -> str:
    """The message, after what the form needs to read it: the options offered last, and the
    current profile and run."""
    lines: list[str] = []
    if context.offers:
        offered = "; ".join(
            f"({i}) {offer.label}: {offer.call}" for i, offer in enumerate(context.offers, 1)
        )
        lines.append(f"Offered last: {offered}.")
    if context.profile or context.run_id:
        lines.append(f"Current profile: {context.profile}; current run: {context.run_id}.")
    lines.append(f"Message: {message}")
    return "\n".join(lines)


def _errors(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'form'}: {item['msg']}"
        for item in error.errors()
    )


def _argument(arguments: str, name: str) -> str | None:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return None
    value = parsed.get(name) if isinstance(parsed, dict) else None
    return value if isinstance(value, str) else None


def _said(stages: frozenset[Stage]) -> str:
    """Stages in the pipeline's order, as words."""
    return ", ".join(stage for stage in STAGES if stage in stages)
