"""The model client's smoke test (docs/autonomous_plan.md §13, M1): calls to the real model, each kind of
call the client makes in turn, with the usage of each. `paco-ab --smoke` plays it with the model of
.env (PACO_LLM_*)."""

import base64
import io
import time
from dataclasses import dataclass
from typing import Any, Literal

from matplotlib.figure import Figure
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.agent.conversion import tools_for_model
from paco.autonomous import server
from paco.llm.chat import Chat
from paco.llm.model import Filled, Reply

type Kind = Literal["thinking", "no thinking", "parallel", "form", "figure"]

# The calls the test makes, and their kinds, in turn: a figure only when vision is on.
CALLS = 20
KINDS: tuple[Kind, ...] = ("thinking", "no thinking", "parallel", "form", "figure")
SYSTEM = "You process MASW seismic profiles with the tools you are given. Answer in one sentence."
ASKED: dict[Kind, str] = {
    "thinking": "Which profiles are there? Use a tool if you need one.",
    "no thinking": "Which processing settings can active_p2 take? Use a tool if you need one.",
    "parallel": "Read the processing settings of active_p2 and the inversion's parameters.",
    "form": "Process the profile active_p2, then pick its curves.",
    "figure": "What does this figure show?",
}
# The form the form calls fill: a scope, much smaller than PACo's.
FORM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "profile": {"type": "string"},
        "stages": {
            "type": "array",
            "items": {"type": "string", "enum": ["process", "pick", "invert"]},
        },
    },
    "required": ["profile", "stages"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Call:
    """One call of the smoke test: its kind, how long it took, the usage the server gave back
    (None for what it did not say), and its error, if it failed."""

    number: int
    kind: Kind
    seconds: float
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    tool_calls: int = 0
    error: str | None = None


async def smoke(model: Chat, calls: int = CALLS) -> list[Call]:
    """`calls` calls to `model`, each kind in turn, offered the autonomous agent's tools; a call that
    fails is kept with its error, and the next one goes on."""
    tools = tools_for_model(await server.server.list_tools())
    kinds: tuple[Kind, ...] = (
        KINDS if model.vision else tuple(kind for kind in KINDS if kind != "figure")
    )
    made: list[Call] = []
    for number in range(1, calls + 1):
        kind = kinds[(number - 1) % len(kinds)]
        start = time.perf_counter()
        try:
            got = await _call(model, kind, tools)
        except Exception as error:
            seconds = round(time.perf_counter() - start, 2)
            made.append(Call(number, kind, seconds, error=f"{type(error).__name__}: {error}"))
            continue
        made.append(
            Call(
                number,
                kind,
                round(time.perf_counter() - start, 2),
                got.prompt_tokens,
                got.cached_tokens,
                got.completion_tokens,
                len(got.tool_calls) if isinstance(got, Reply) else 0,
            )
        )
    return made


def format_calls(calls: list[Call]) -> str:
    """The calls one a line, then how many failed: the tokens each read (those the server took
    from its cache, when it says), wrote, and the tool calls of a reply."""
    lines = [
        f"{call.number:3} {call.kind:12} {call.seconds:6.2f} s  {_usage(call)}" for call in calls
    ]
    failed = sum(1 for call in calls if call.error)
    lines.append(f"{len(calls)} calls, {failed} failed.")
    return "\n".join(lines)


def _usage(call: Call) -> str:
    if call.error:
        return f"error: {call.error}"
    cached = f" ({call.cached_tokens} cached)" if call.cached_tokens is not None else ""
    said = f"read {call.prompt_tokens}{cached}, wrote {call.completion_tokens}"
    return said if call.kind == "form" else f"{said}, {call.tool_calls} tool call(s)"


async def _call(
    model: Chat, kind: Kind, tools: list[ChatCompletionFunctionToolParam]
) -> Reply | Filled:
    """One call of `kind`, in a conversation of its own."""
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": SYSTEM}]
    match kind:
        case "form":
            messages.append({"role": "user", "content": ASKED[kind]})
            return await model.fill(messages, FORM)
        case "figure":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ASKED[kind]},
                        {"type": "image_url", "image_url": {"url": figure()}},
                    ],
                }
            )
            return await model(messages, tools, thinking=False)
        case _:
            messages.append({"role": "user", "content": ASKED[kind]})
            return await model(
                messages, tools, thinking=kind == "thinking", parallel=kind == "parallel"
            )


def figure() -> str:
    """A small figure, as a PNG data URL: a dispersion curve, phase velocity against frequency."""
    drawn = Figure(figsize=(3, 2), dpi=80)
    axes = drawn.add_subplot()
    frequencies = [5 + 2.5 * step for step in range(15)]
    axes.plot(frequencies, [150 + 1200 / frequency for frequency in frequencies])
    axes.set_xlabel("Frequency (Hz)")
    axes.set_ylabel("Phase velocity (m/s)")
    png = io.BytesIO()
    drawn.savefig(png, format="png")
    return f"data:image/png;base64,{base64.b64encode(png.getvalue()).decode()}"
