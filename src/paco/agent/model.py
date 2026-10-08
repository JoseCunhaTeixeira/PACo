"""The model the agent talks to, behind a small interface: the real one is served by vLLM, the
tests use a scripted one."""

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import anyio
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)

logger = logging.getLogger(__name__)

# Qwen3 thinks aloud between these tags, unless vLLM's reasoning parser removes them.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# The tokens a form may take: a filled scope takes about 100; the cap ends a model repeating
# itself inside a list, which constrained decoding allows.
FORM_TOKENS = 400
# How long a model server that refuses or drops the connection is asked again, in seconds,
# the waits doubling from 1 s: the tunnel to Cloud Run (gcloud run services proxy) restarts
# every 55 min for a fresh token, refusing connections for a moment, or for about 6 s when a
# request outlives its shutdown (gcloud exits, systemd starts it again 5 s later).
RECONNECT_S = 30.0


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


@dataclass(frozen=True, slots=True)
class Filled:
    content: str  # the form as the model filled it: JSON text, not necessarily valid
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(Protocol):
    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply: ...

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        """A form filled under `schema`: the model's output is constrained to it."""
        ...


class OpenAIChat:
    """A model behind an OpenAI-compatible chat API, such as vLLM's. `temperature` and `seed`
    are the conversation's sampling (None: the server's own); a form is filled at temperature
    0, without thinking. A server that refuses or drops the connection is asked again for up to
    `reconnect_s` (RECONNECT_S), waiting with `sleep`."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
        reconnect_s: float = RECONNECT_S,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self.name = model
        self._reconnect_s = reconnect_s
        self._sleep = sleep
        self._sampling: dict[str, Any] = {
            key: value
            for key, value in (("temperature", temperature), ("seed", seed))
            if value is not None
        }

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply:
        async def create() -> ChatCompletion:
            return await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                # One call per reply: in a batch, the model makes up what a call needs from the
                # one before it (a run_id). vLLM then keeps the first call only.
                parallel_tool_calls=False,
                **self._sampling,
            )

        response = await self._reconnecting(create)
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

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        async def create() -> ChatCompletion:
            return await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
                max_tokens=FORM_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "form", "strict": True, "schema": schema},
                },
                # Qwen3's thinking takes 5 to 20 s before a form and fills it no better; a chat
                # template without this switch ignores it.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        response = await self._reconnecting(create)
        usage = response.usage
        return Filled(
            content=without_thinking(response.choices[0].message.content or ""),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    async def _reconnecting(
        self, create: Callable[[], Awaitable[ChatCompletion]]
    ) -> ChatCompletion:
        """`create`'s completion, asked again while the server refuses or drops the connection,
        the waits doubling from 1 s, for `reconnect_s` at most; then its error. A timeout is
        not asked again: the server was there."""
        waited, wait = 0.0, 1.0
        while True:
            try:
                return await create()
            except APITimeoutError:
                raise
            except APIConnectionError as error:
                left = self._reconnect_s - waited
                if left <= 0:
                    raise
                pause = min(wait, left)
                logger.warning(
                    "The model server at %s did not answer (%s): asked again in %g s",
                    self._client.base_url,
                    error,
                    pause,
                )
                await self._sleep(pause)
                waited, wait = waited + pause, wait * 2


def without_thinking(text: str) -> str:
    """`text` without the model's thinking: it would fill the context at every turn."""
    return _THINKING.sub("", text).strip()
