"""The model the agent talks to, behind a small interface: the real one is served by vLLM, the
tests use a scripted one."""

import re
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

# Qwen3 thinks aloud between these tags, unless vLLM's reasoning parser removes them.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str  # the model's name for this call, which the tool's result must quote
    name: str
    arguments: str  # JSON text, as the model wrote it: not necessarily valid


@dataclass(frozen=True, slots=True)
class Reply:
    content: str  # what the model says, without its thinking
    tool_calls: tuple[ToolCall, ...]  # empty when the model answers the user
    # When the API says: the context the model read, and what it wrote.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(Protocol):
    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply: ...


class OpenAIChat:
    """A model behind an OpenAI-compatible chat API, such as vLLM's."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            # One call per reply: in a batch, the model makes up what a call needs from the one
            # before it (a run_id). vLLM then keeps the first call only.
            parallel_tool_calls=False,
        )
        message = response.choices[0].message
        calls = tuple(
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in message.tool_calls or ()
            if call.type == "function"
        )
        usage = response.usage
        return Reply(
            content=without_thinking(message.content or ""),
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )


def without_thinking(text: str) -> str:
    """`text` without the model's thinking: it would fill the context at every turn."""
    return _THINKING.sub("", text).strip()
