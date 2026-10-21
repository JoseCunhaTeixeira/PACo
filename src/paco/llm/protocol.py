"""PACo's ChatModel: the interface both agents talk to their model through, the replies and forms it
gives back, and what the clients share (a form's tokens, the reconnection budget, the thinking left
out of a reply). The real models are paco.llm's clients; the tests use scripted ones."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

logger = logging.getLogger(__name__)

# Qwen3 thinks aloud between these tags, unless vLLM's reasoning parser removes them.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# The tokens a form may take: a filled scope takes about 100; the cap ends a model repeating
# itself inside a list, which constrained decoding allows.
FORM_TOKENS = 400
# How long a model server that refuses or drops the connection is asked again, in seconds,
# the waits doubling from 1 s: a tunnel or a proxy in front of it may restart, refusing
# connections for a few seconds.
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
    # When the API says: the context the model read, and what it wrote; the prompt tokens the
    # server took from its cache.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Filled:
    content: str  # the form as the model filled it: JSON text, not necessarily valid
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None


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


def without_thinking(text: str) -> str:
    """`text` without the model's thinking: it would fill the context at every turn."""
    return _THINKING.sub("", text).strip()
