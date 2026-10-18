"""The model client (docs/geo_plan.md §3): PACo's ChatModel protocol, which the host calls and
PACo's own agent too, for a model behind OpenAI's chat API, as OpenAI, Mistral, vLLM or another
server serves it; with what the geophysicist agent adds: images in the conversation, thinking
switched on or off per call, the prompt tokens the server took from its cache, and several calls a
reply when they only read. Anthropic's API has its own client (paco.llm.claude); paco.llm.chat makes
the one a provider needs."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

import anyio
from openai import AsyncOpenAI, BadRequestError, omit
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)
from openai.types.shared import ReasoningEffort

from paco.llm.protocol import (
    FORM_TOKENS,
    RECONNECT_S,
    Filled,
    Reply,
    ToolCall,
    without_thinking,
)
from paco.llm.reconnecting import reconnecting
from paco.llm.settings import Effort

# How a server takes what the client sends beside OpenAI's chat API: vLLM's chat-template switch
# of thinking; OpenAI's reasoning effort and its cap on the completion's tokens, reasoning
# included (Azure's too); Gemini's reasoning effort, its levels, and the thought signatures its
# calls carry; nothing more for the others (Mistral, other servers), which refuse what they do not
# know.
type Dialect = Literal["vllm", "openai", "gemini", "plain"]
# The dialects whose models reason within the completion's tokens: a form leaves them room.
REASONING: tuple[Dialect, ...] = ("openai", "gemini")
# Gemini's reasoning efforts (its compatibility layer's): the settings' higher ones are its high.
GEMINI_EFFORTS: dict[Effort, ReasoningEffort] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
# The signature Google allows on a call whose own is not known (a resumed study's): Gemini 3 and
# later refuse a call given back without one.
SKIP_SIGNATURE = "skip_thought_signature_validator"
# A form's tokens on OpenAI's API, its reasoning included: PACo's FORM_TOKENS are the form's own.
OPENAI_FORM_TOKENS = 4_000
# A request asking only whether the model takes images: an image of one pixel, a short question,
# and its tokens (a reasoning model's included).
DOT = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
PROBE = "Reply with OK."
PROBE_TOKENS = 256


class OpenAIChat:
    """A model behind OpenAI's chat API, as PACo's ChatModel: OpenAI's, Mistral's, a vLLM server's
    or another's, as `dialect` says.

    `vision`: whether images may reach the model (PACO_LLM_VISION); a conversation holding one
    is refused without it, before anything is sent. `thinking`: whether the model thinks before
    its replies (PACO_LLM_THINKING) unless a call says otherwise: on vLLM through its
    chat-template switch, which Qwen3's template reads and a template without it ignores; on
    OpenAI, `effort` sent as its reasoning effort when set (a model that does not reason refuses
    it); elsewhere, as the model does. A form is filled under its schema without thinking: at
    temperature 0, as in PACo, but on OpenAI, whose reasoning models refuse a temperature. A
    server that refuses or drops the connection is asked again for up to `reconnect_s`, waiting
    with `sleep` (paco.llm.reconnecting).
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
        *,
        vision: bool = False,
        thinking: bool = True,
        dialect: Dialect = "vllm",
        effort: Effort | None = None,
        reconnect_s: float = RECONNECT_S,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        self._client = client
        self.name = model
        self.vision = vision
        self.thinking = thinking
        self.dialect: Dialect = dialect
        # The conversation's sampling (PACO_LLM_TEMPERATURE, PACO_LLM_SEED), on a server of the
        # user's: OpenAI's reasoning models refuse a temperature, Gemini's ignore it.
        self._sampling: dict[str, Any] = {
            key: value
            for key, value in (("temperature", temperature), ("seed", seed))
            if value is not None and dialect in ("vllm", "plain")
        }
        # Gemini's thought signatures, by the call they came with: given back with it.
        self._signatures: dict[str, Any] = {}
        self.effort: Effort | None = effort
        self._reconnect_s = reconnect_s
        self._sleep = sleep

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
        *,
        thinking: bool | None = None,
        parallel: bool = False,
    ) -> Reply:
        """The model's reply to `messages`, offered `tools`. It thinks first if `thinking` says so
        (None: the client's own). Several tool calls come in one reply only when `parallel`, which
        the host asks for calls that only read; otherwise one call a reply, as in PACo: in a batch,
        the model makes up what a call needs from the one before it."""
        self._refuse_images(messages)
        think = self.thinking if thinking is None else thinking
        effort = self._reasoning() if think else None
        sent = self._signed(messages) if self.dialect == "gemini" else messages

        async def create() -> ChatCompletion:
            return await self._client.chat.completions.create(
                model=self.name,
                messages=sent,
                tools=tools or omit,
                parallel_tool_calls=parallel if tools else omit,
                reasoning_effort=effort if effort is not None else omit,
                extra_body=self._switch(think),
                **self._sampling,
            )

        response = await self._reconnecting(create)
        message = response.choices[0].message
        for call in message.tool_calls or ():
            extra = (call.model_extra or {}).get("extra_content")
            if extra is not None:
                self._signatures[call.id] = extra
        calls = tuple(
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in message.tool_calls or ()
            if call.type == "function"
        )
        prompt, cached, completion = _tokens(response.usage)
        return Reply(
            content=without_thinking(message.content or ""),
            tool_calls=calls,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
        )

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        """A form filled under `schema`, as PACo's client fills one: the output constrained to the
        schema, at temperature 0, without thinking, within PACo's FORM_TOKENS; on OpenAI, at the
        model's temperature, reasoning little when it reasons, within OPENAI_FORM_TOKENS."""
        self._refuse_images(messages)
        openai, reasons = self.dialect == "openai", self.dialect in REASONING
        # Gemini's 3 and later always think, its 2.5 at a budget: a low one.
        low = (openai and self.effort is not None) or self.dialect == "gemini"

        async def create() -> ChatCompletion:
            return await self._client.chat.completions.create(
                model=self.name,
                messages=self._signed(messages) if self.dialect == "gemini" else messages,
                temperature=omit if reasons else 0,
                max_tokens=omit if openai else (OPENAI_FORM_TOKENS if reasons else FORM_TOKENS),
                max_completion_tokens=OPENAI_FORM_TOKENS if openai else omit,
                reasoning_effort="low" if low else omit,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "form", "strict": True, "schema": schema},
                },
                extra_body=self._switch(False),
            )

        response = await self._reconnecting(create)
        prompt, cached, completion = _tokens(response.usage)
        return Filled(
            content=without_thinking(response.choices[0].message.content or ""),
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
        )

    async def takes_images(self) -> bool:
        """Whether the model takes images: a short question about an image of one pixel not
        refused (a text-only model's server refuses the image, 400). Other errors are raised: a
        key refused, a model not there."""
        openai = self.dialect == "openai"

        async def create() -> ChatCompletion:
            return await self._client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": DOT}},
                            {"type": "text", "text": PROBE},
                        ],
                    }
                ],
                max_tokens=omit if openai else PROBE_TOKENS,
                max_completion_tokens=PROBE_TOKENS if openai else omit,
                extra_body=self._switch(False),
            )

        try:
            await self._reconnecting(create)
        except BadRequestError:
            return False
        return True

    def _reasoning(self) -> ReasoningEffort | None:
        """The reasoning effort sent, where the dialect takes one and the settings give it."""
        if self.effort is None:
            return None
        if self.dialect == "openai":
            return self.effort
        return GEMINI_EFFORTS[self.effort] if self.dialect == "gemini" else None

    def _signed(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """`messages` with Gemini's thought signatures given back on the calls that carried them;
        a reply whose calls' signatures are not known (a resumed study's) signed on its first call
        with the placeholder Google allows."""
        signed: list[ChatCompletionMessageParam] = []
        for message in messages:
            calls = cast(list[dict[str, Any]], message.get("tool_calls") or [])
            if message["role"] != "assistant" or not calls:
                signed.append(message)
                continue
            given: list[dict[str, Any]] = []
            for index, call in enumerate(calls):
                extra = self._signatures.get(str(call.get("id")))
                if extra is None and index == 0:
                    extra = {"google": {"thought_signature": SKIP_SIGNATURE}}
                given.append({**call, "extra_content": extra} if extra is not None else call)
            signed.append(cast(ChatCompletionMessageParam, {**message, "tool_calls": given}))
        return signed

    async def _reconnecting(
        self, create: Callable[[], Awaitable[ChatCompletion]]
    ) -> ChatCompletion:
        return await reconnecting(create, self._client.base_url, self._reconnect_s, self._sleep)

    def _switch(self, think: bool) -> dict[str, Any] | None:
        """vLLM's chat-template switch of thinking; nothing for the other servers."""
        if self.dialect != "vllm":
            return None
        return {"chat_template_kwargs": {"enable_thinking": think}}

    def _refuse_images(self, messages: list[ChatCompletionMessageParam]) -> None:
        """Refuse `messages` when they hold an image and vision is off: nothing is sent."""
        if not self.vision and (count := images(messages)):
            raise ValueError(
                f"The conversation holds {count} image(s) but vision is off "
                "(PACO_LLM_VISION): no image may reach the model."
            )


def images(messages: list[ChatCompletionMessageParam]) -> int:
    """How many images `messages` hold, in the content parts of their lists."""
    found = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            found += sum(1 for part in content if part.get("type") == "image_url")
    return found


def _tokens(usage: CompletionUsage | None) -> tuple[int | None, int | None, int | None]:
    """The prompt tokens the model read, those of them the server took from its cache, and the
    tokens the model wrote; None for what the server does not say (vLLM says the cached ones when
    started with --enable-prompt-tokens-details)."""
    if usage is None:
        return None, None, None
    details = usage.prompt_tokens_details
    cached = details.cached_tokens if details is not None else None
    return usage.prompt_tokens, cached, usage.completion_tokens
