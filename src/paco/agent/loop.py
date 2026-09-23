"""The agent loop: the model asks for tools, the host calls PACo's server, until the model
answers the user."""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from mcp import Client
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.agent.conversion import result_for_model, tools_for_model
from paco.agent.model import ChatModel, Reply, ToolCall
from paco.agent.record import ModelStep, Step, ToolStep, Transcript

ROLE = (
    "You are PACo's assistant. You help a geophysicist turn MASW seismic profiles into "
    "dispersion curves and velocity models, with the tools you have. Answer briefly. Report only "
    "what the tools return: never invent a result."
)

# What the loop does, for the user to follow: tool calls, progress.
type OnEvent = Callable[[str], None]


class Agent:
    """One conversation between the user, the model and PACo's server."""

    def __init__(
        self,
        client: Client,
        model: ChatModel,
        tools: list[ChatCompletionFunctionToolParam],
        instructions: str | None,
        max_tool_calls: int = 15,
        on_event: OnEvent = print,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._max_tool_calls = max_tool_calls
        self._on_event = on_event
        system = f"{ROLE}\n\n{instructions}" if instructions else ROLE
        self.messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
        self.steps: list[Step] = []
        self.started_at = datetime.now(UTC)

    @classmethod
    async def start(
        cls, client: Client, model: ChatModel, max_tool_calls: int = 15, on_event: OnEvent = print
    ) -> Agent:
        """An agent with the tools and instructions of the server `client` is connected to."""
        tools = tools_for_model((await client.list_tools()).tools)
        return cls(client, model, tools, client.instructions, max_tool_calls, on_event)

    async def answer(self, question: str) -> str:
        """The model's answer to `question`, after every tool call it asked for."""
        self.messages.append({"role": "user", "content": question})
        calls = 0
        while True:
            start = time.perf_counter()
            reply = await self._model(self.messages, self._tools)
            self.steps.append(
                ModelStep(
                    duration_s=round(time.perf_counter() - start, 3),
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    tool_calls=len(reply.tool_calls),
                )
            )
            self.messages.append(_assistant_message(reply))
            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:
                calls += 1
                step = await self._call(call, over_budget=calls > self._max_tool_calls)
                self.steps.append(step)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": step.result}
                )

    def transcript(self, model: str) -> Transcript:
        """The conversation so far, and every step with its cost."""
        return Transcript(
            started_at=self.started_at,
            model=model,
            messages=[dict(message) for message in self.messages],
            steps=list(self.steps),
        )

    async def _call(self, call: ToolCall, over_budget: bool) -> ToolStep:
        start = time.perf_counter()

        def refused(result: str) -> ToolStep:
            return ToolStep(
                name=call.name,
                arguments=call.arguments,
                called=False,
                is_error=True,
                duration_s=0.0,
                result=result,
            )

        if over_budget:
            return refused(
                f"Not called: this answer already made {self._max_tool_calls} tool calls. "
                "Answer the user with what you have, and say what is left to do."
            )
        try:
            parsed: Any = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            return refused(
                f"Not called: the arguments of {call.name} are not valid JSON ({error})."
            )
        if not isinstance(parsed, dict):
            return refused(f"Not called: the arguments of {call.name} must be a JSON object.")

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            self._on_event(f"   {call.name}: {message or f'{progress:g} of {total:g}'}")

        self._on_event(f"-> {call.name}({call.arguments})")
        result = await self._client.call_tool(call.name, parsed, progress_callback=on_progress)
        return ToolStep(
            name=call.name,
            arguments=call.arguments,
            called=True,
            is_error=bool(result.is_error),
            duration_s=round(time.perf_counter() - start, 3),
            result=result_for_model(result),
        )


def _assistant_message(reply: Reply) -> ChatCompletionMessageParam:
    if not reply.tool_calls:
        return {"role": "assistant", "content": reply.content}
    return {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in reply.tool_calls
        ],
    }
