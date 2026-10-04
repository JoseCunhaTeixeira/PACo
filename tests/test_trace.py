"""A saved conversation printed turn by turn, and played again through the current loop with
its recorded replies and results (no tool runs)."""

import json
from typing import Any, cast

import anyio
from mcp import Client
from mcp.types import CallToolResult, TextContent
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.agent import Agent, Filled, Reply, ToolCall, Transcript
from paco.agent.answer import SCHEMA as ANSWER_SCHEMA
from paco.agent.trace import answers, differences, questions, replay, show

LOOK: dict[str, Any] = {
    "process": False,
    "pick": False,
    "invert": False,
    "soils": False,
    "profile": None,
    "run_id": None,
    "positions_m": [],
    "length_receivers": None,
    "length_m": None,
    "step_receivers": None,
    "step_m": None,
    "compare_lengths_receivers": [],
    "compare_lengths_m": [],
    "redo": False,
    "replace_hand_work": False,
    "option": None,
    "workers": None,
    "mode": None,
}


class Model:
    """Looks at the runs, then answers; fills its forms plainly."""

    def __init__(self) -> None:
        self._replies = [
            Reply(content="", tool_calls=(ToolCall("call_0", "inspect", '{"what": "runs"}'),)),
            Reply(content="One run: r.", tool_calls=()),
        ]

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        return self._replies.pop(0)

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        schema: dict[str, Any],
    ) -> Filled:
        form = {"said": "One run: r.", "question": None} if schema == ANSWER_SCHEMA else LOOK
        return Filled(content=json.dumps(form))


class Server:
    """Stands in for PACo's server: one run."""

    async def call_tool(self, name: str, *_: object, **__: object) -> CallToolResult:
        assert name == "inspect"
        return CallToolResult(content=[TextContent(type="text", text="Run r: active_p1.")])


def _recorded() -> Transcript:
    async def conversation() -> Transcript:
        agent = Agent(cast(Client, Server()), Model(), [], None, on_event=lambda _: None)
        await agent.answer("Which runs are there?")
        return agent.transcript("qwen")

    return anyio.run(conversation)


def test_a_conversation_is_printed_turn_by_turn() -> None:
    transcript = _recorded()

    shown = show(transcript)

    assert questions(transcript) == ["Which runs are there?"]
    assert "you> Which runs are there?" in shown
    assert '-> inspect({"what": "runs"})' in shown
    assert "   Run r: active_p1." in shown
    assert shown.endswith("Scope: look only.\n\nOne run: r.")
    assert transcript.conversation is not None and transcript.prompt_version is not None


def test_a_replay_gives_the_recorded_answers_and_says_a_difference() -> None:
    transcript = _recorded()

    replayed = anyio.run(replay, transcript)

    assert differences(answers(transcript), replayed) == ""
    changed = transcript.model_copy(
        update={
            "messages": [*transcript.messages[:-1], {"role": "assistant", "content": "Two runs."}]
        }
    )
    said = differences(answers(changed), replayed)
    assert said.startswith("Turn 1:") and "-Two runs." in said
