"""paco-trace show FILE | paco-trace replay FILE: a saved conversation (the chat's agent_logs, an
evaluation's transcript.json) printed turn by turn; or played again through the current loop
with the recorded replies, forms and tool results, no tool running, to see where its answers
now differ (O2)."""

import argparse
import difflib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.agent.answer import SCHEMA as ANSWER_SCHEMA
from paco.agent.loop import Agent, from_data
from paco.agent.model import Filled, Reply, ToolCall
from paco.agent.record import AnswerStep, ScopeStep, ToolStep, Transcript

# The model's note after each message (paco.agent.scope): the user's own words come before it.
_NOTE = "\n\n[PACo] "
# The JSON-RPC code of a resource that does not exist (MCP's).
RESOURCE_NOT_FOUND = -32002
# How much of a tool's result a printed turn shows.
_SHOWN = 160


def questions(transcript: Transcript) -> list[str]:
    """The user's messages, as they typed them."""
    return [
        str(message.get("content", "")).split(_NOTE, 1)[0]
        for message in transcript.messages
        if message.get("role") == "user"
    ]


def answers(transcript: Transcript) -> list[str]:
    """The answers the user read, one per message."""
    return [
        str(message.get("content") or "")
        for message in transcript.messages
        if message.get("role") == "assistant" and not message.get("tool_calls")
    ]


def show(transcript: Transcript) -> str:
    """The conversation turn by turn: the message, its scope, the calls and what each
    returned (shortened), the answer."""
    scopes = iter(step for step in transcript.steps if isinstance(step, ScopeStep))
    lines = [
        f"{transcript.model}, from {transcript.started_at:%Y-%m-%d %H:%M}, conversation "
        f"{transcript.conversation}, {transcript.prompt_version}"
    ]
    for message in transcript.messages:
        role, content = message.get("role"), str(message.get("content") or "")
        if role == "user":
            scope = next(scopes, None)
            lines += ["", f"you> {content.split(_NOTE, 1)[0]}"]
            if scope is not None:
                asked = scope.scope or {"unread": scope.error}
                lines.append(f"   scope: {json.dumps(asked, ensure_ascii=False)}")
        elif role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                lines.append(f"-> {call['function']['name']}({call['function']['arguments']})")
        elif role == "tool":
            shown = " ".join(from_data(content).split())
            lines.append(f"   {shown[:_SHOWN]}{'...' if len(shown) > _SHOWN else ''}")
        elif role == "assistant":
            lines += ["paco>", content]
    return "\n".join(lines)


class _Recorded:
    """Stands in for the model: the recorded replies, scope forms and answer forms, in order."""

    def __init__(self, transcript: Transcript) -> None:
        self._replies = [
            _reply(message) for message in transcript.messages if message.get("role") == "assistant"
        ]
        steps = transcript.steps
        self._scopes = [step.scope for step in steps if isinstance(step, ScopeStep)]
        self._answers = [step.form for step in steps if isinstance(step, AnswerStep)]

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        if not self._replies:
            return Reply(content="(the recording ends here)", tool_calls=())
        return self._replies.pop(0)

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        schema: dict[str, Any],
    ) -> Filled:
        recorded = self._answers if schema == ANSWER_SCHEMA else self._scopes
        form = recorded.pop(0) if recorded else None
        return Filled(content=json.dumps(form) if form is not None else "{}")


class _RecordedServer:
    """Stands in for PACo's server: the recorded results of the calls, in order."""

    def __init__(self, transcript: Transcript) -> None:
        self._results = [
            step for step in transcript.steps if isinstance(step, ToolStep) and step.called
        ]
        self.instructions = None

    async def call_tool(self, name: str, *_: object, **__: object) -> CallToolResult:
        if not self._results or self._results[0].name != name:
            return CallToolResult(
                content=[TextContent(type="text", text="(the replay leaves the recording here)")],
                is_error=True,
            )
        step = self._results.pop(0)
        text = TextContent(type="text", text=step.result)
        return CallToolResult(content=[text], is_error=step.is_error)

    async def read_resource(self, uri: str) -> object:
        # The latest run: the recorded messages said it; the replay leaves it out.
        raise MCPError(RESOURCE_NOT_FOUND, f"{uri}: not recorded")


async def replay(transcript: Transcript) -> list[str]:
    """The answers the current loop gives the recorded conversation."""
    model = _Recorded(transcript)
    server: Any = _RecordedServer(transcript)
    agent = Agent(server, model, [], None, on_event=lambda _: None)
    return [await agent.answer(question) for question in questions(transcript)]


def differences(recorded: Sequence[str], replayed: Sequence[str]) -> str:
    """Where the replayed answers differ from the recorded ones, turn by turn ("" when none)."""
    out: list[str] = []
    for turn, (before, now) in enumerate(zip(recorded, replayed, strict=False), 1):
        if before != now:
            diff = difflib.unified_diff(
                before.splitlines(), now.splitlines(), "recorded", "replayed", lineterm=""
            )
            out += [f"Turn {turn}:", *diff]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(prog="paco-trace", description=__doc__)
    parser.add_argument("action", choices=("show", "replay"))
    parser.add_argument("file", type=Path, help="a transcript: agent_logs/*.json, transcript.json")
    arguments = parser.parse_args()
    transcript = Transcript.model_validate_json(arguments.file.read_text())
    if arguments.action == "show":
        print(show(transcript))
        return
    replayed = anyio.run(replay, transcript)
    said = differences(answers(transcript), replayed)
    print(said or f"The current loop gives the {len(replayed)} answer(s) recorded.")


def _reply(message: dict[str, Any]) -> Reply:
    calls = tuple(
        ToolCall(
            id=call["id"], name=call["function"]["name"], arguments=call["function"]["arguments"]
        )
        for call in message.get("tool_calls") or ()
    )
    return Reply(content=str(message.get("content") or ""), tool_calls=calls)


if __name__ == "__main__":
    main()
