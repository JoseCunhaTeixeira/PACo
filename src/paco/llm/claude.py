"""The Anthropic client: PACo's ChatModel protocol over Anthropic's own
API, for Claude, beside the client of OpenAI's chat API (paco.llm.model). The host and PACo's agent
write the conversation in OpenAI's format; this client reads it into Anthropic's: the system prompt
apart, each call a tool_use block, each result a tool_result block in the user's next message, the
figures that follow a result inside it (Anthropic's API takes images in a tool's result, OpenAI's
in user messages only), the state card after the results.

What Claude's API adds, used here:

- Prompt caching: the tools, the system prompt, the conversation's first turn (the request, and
  the atlas it carries) and the conversation up to the model's last reply are marked for the
  cache, so that each call reads them back; the host keeps the prompt's beginning the same from
  call to call. On Anthropic's own API the parts fixed for a phase (the tools, the system prompt,
  the first turn) are kept an hour: a tool that runs longer than the cache's 5 minutes (the
  search's judging, an inversion followed to its end) leaves them read back.
- Thinking as the model takes it, which the models API says: adaptive where it can, else a budget
  of tokens; its blocks given back unchanged with the calls they led to, as the API asks within a
  tool loop.
- A form is the input of a tool the model must call, under the form's schema, without thinking;
  for a model that cannot leave thinking out, its output constrained to the schema; where it
  takes neither, the form's tool offered and its call asked.

Its errors are raised as the OpenAI SDK's of the same kind, which the host, PACo's runner and the
evaluation tell apart: a server out of reach loses a play, a refusal fails it."""

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

import anthropic
import anyio
import openai
from anthropic import AsyncAnthropic, Omit, omit, transform_schema
from anthropic.types import (
    Message,
    MessageParam,
    ModelCapabilities,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolParam,
    Usage,
)
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.llm.model import images
from paco.llm.protocol import RECONNECT_S, Filled, Reply, ToolCall
from paco.llm.reconnecting import reconnecting

logger = logging.getLogger(__name__)

# The tokens of a reply, its thinking included: past some 21 000 the SDK asks for a stream.
REPLY_TOKENS = 16_000
# Thinking's budget, where the model takes a budget rather than adaptive thinking.
THINKING_TOKENS = 8_000
# A form's tokens: PACo's 400 are a small model's; Claude's forms come back whole within these.
FORM_TOKENS = 2_000
# The tool whose input is a form.
FORM_TOOL = "form"
# The fields of a value or nothing (anyOf with null) Claude's API compiles a form's schema with,
# at most: past them, it refuses the request.
UNIONS = 16
# What a form's tool, offered without forcing it, is asked with.
FILL = "Fill the form now: call the tool form, whose fields are your answer."
CACHED: dict[str, Any] = {"type": "ephemeral"}
# The parts of the prompt fixed for a phase, kept an hour where the API takes it.
FIXED: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}
_DATA_URL = re.compile(r"data:(?P<type>[\w/+.-]+);base64,(?P<data>.+)", re.DOTALL)
# The OpenAI SDK's errors, by the status Anthropic's API answered.
_STATUS: dict[int, type[openai.APIStatusError]] = {
    400: openai.BadRequestError,
    401: openai.AuthenticationError,
    403: openai.PermissionDeniedError,
    404: openai.NotFoundError,
    409: openai.ConflictError,
    422: openai.UnprocessableEntityError,
    429: openai.RateLimitError,
}


class AnthropicChat:
    """Claude through Anthropic's API, as PACo's ChatModel: `vision` as its maker gives it (PACo's
    own connect it on), `thinking` unless a call says otherwise, at the model's own effort. A
    conversation holding an image is refused without vision, before anything is sent. A server
    that refuses or drops the connection is asked again for up to `reconnect_s`, waiting with
    `sleep`, as the other client does. `hour`: the prompt's fixed parts cached an hour (Anthropic's
    own API), else 5 minutes as every part."""

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        *,
        vision: bool = False,
        thinking: bool = True,
        reconnect_s: float = RECONNECT_S,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
        hour: bool = False,
    ) -> None:
        self._client = client
        self._fixed = FIXED if hour else CACHED
        self.name = model
        self.vision = vision
        self.thinking = thinking
        self._reconnect_s = reconnect_s
        self._sleep = sleep
        # What the models API says of the model, asked once: None when it did not answer.
        self._asked = False
        self._capabilities: ModelCapabilities | None = None
        self._max_tokens: int | None = None
        # Each reply that made calls, its content blocks as they came, by each call's id.
        self._replies: dict[str, list[dict[str, Any]]] = {}

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
        *,
        thinking: bool | None = None,
        parallel: bool = False,
    ) -> Reply:
        """The model's reply to `messages`, offered `tools`. It thinks first if `thinking` says so
        (None: the client's own). Several tool calls come in one reply only when `parallel`."""
        self._refuse_images(messages)
        await self._described()
        think = self.thinking if thinking is None else thinking
        system, turns = self._converted(messages)
        offered = _tools(tools, self._fixed)
        choice: ToolChoiceParam = {"type": "auto", "disable_parallel_tool_use": not parallel}
        config = self._thinking(think)
        response = await self._create(
            lambda: self._client.messages.create(
                model=self.name,
                max_tokens=self._tokens(REPLY_TOKENS),
                system=system,
                messages=turns,
                tools=offered or omit,
                tool_choice=choice if offered else omit,
                thinking=config if config is not None else omit,
            )
        )
        calls = tuple(
            ToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        if calls:
            kept = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
            self._replies |= dict.fromkeys((call.id for call in calls), kept)
        prompt, cached, completion = _tokens(response.usage)
        return Reply(
            content=_text(response),
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
        """A form filled under `schema`: the input of the tool the model must call, without
        thinking (the API takes no temperature); or, for a model that cannot leave thinking out,
        its output constrained to the schema."""
        self._refuse_images(messages)
        await self._described()
        system, turns = self._converted(messages)
        capabilities = self._capabilities
        silent = capabilities is None or capabilities.thinking.types.disabled.supported
        constrained = capabilities is not None and capabilities.structured_outputs.supported
        schema, relaxed = _within_unions(schema)
        form: ToolParam = {
            "name": FORM_TOOL,
            "description": "Fill the form: its fields are your answer.",
            "input_schema": cast(Any, schema),
        }
        tools: list[ToolParam] | Omit = omit
        choice: ToolChoiceParam | Omit = omit
        output: OutputConfigParam | Omit = omit
        thinking: ThinkingConfigParam | Omit = omit
        if silent:
            tools, choice, thinking = (
                [form],
                {"type": "tool", "name": FORM_TOOL},
                {"type": "disabled"},
            )
        elif constrained:
            # The schema in the subset structured outputs take: what they do not (a list's
            # longest) goes into its description; the form is checked after, as every form.
            output = {"format": {"type": "json_schema", "schema": transform_schema(schema)}}
        else:
            # Neither (a model that always thinks, without structured outputs): the tool offered,
            # its call asked; a form not called comes back as text, checked as any.
            tools, choice = [form], {"type": "auto", "disable_parallel_tool_use": True}
            asked: TextBlockParam = {"type": "text", "text": FILL}
            system = [*system, asked]
        response = await self._create(
            lambda: self._client.messages.create(
                model=self.name,
                max_tokens=self._tokens(FORM_TOKENS),
                system=system,
                messages=turns,
                tools=tools,
                tool_choice=choice,
                output_config=output,
                thinking=thinking,
            )
        )
        filled = next((block.input for block in response.content if block.type == "tool_use"), None)
        prompt, cached, completion = _tokens(response.usage)
        return Filled(
            content=_with_nulls(filled if filled is not None else _text(response), relaxed),
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
        )

    async def takes_images(self) -> bool:
        """Whether the model takes images: as the models API says; a proxy that says nothing: as
        Claude's models do."""
        await self._described()
        return self._capabilities is None or self._capabilities.image_input.supported

    async def _described(self) -> None:
        """The model's capabilities and its most tokens, from the models API, asked once; its
        defaults where the API does not say (a proxy that does not serve it)."""
        if self._asked:
            return
        self._asked = True
        try:
            info = await self._client.models.retrieve(self.name)
        except anthropic.APIError as error:
            logger.warning(
                "Anthropic's models API did not describe %s (%s): its defaults are used",
                self.name,
                error,
            )
            return
        self._capabilities = info.capabilities
        self._max_tokens = info.max_tokens

    def _thinking(self, think: bool) -> ThinkingConfigParam | None:
        """Thinking as the model takes it: adaptive where it can, else a budget; off where it
        can be turned off; None (the model's default) where the models API did not say."""
        capabilities = self._capabilities
        if capabilities is None:
            return None
        kinds = capabilities.thinking.types
        if think and capabilities.thinking.supported:
            if kinds.adaptive.supported:
                return {"type": "adaptive"}
            if kinds.enabled.supported:
                budget = max(1024, min(THINKING_TOKENS, self._tokens(REPLY_TOKENS) // 2))
                return {"type": "enabled", "budget_tokens": budget}
            return None
        if not think and kinds.disabled.supported:
            return {"type": "disabled"}
        return None

    def _tokens(self, wanted: int) -> int:
        return min(wanted, self._max_tokens) if self._max_tokens else wanted

    async def _create(self, request: Callable[[], Awaitable[Message]]) -> Message:
        """`request`'s message; its errors raised as the OpenAI SDK's, asked again while the
        server refuses or drops the connection."""

        async def create() -> Message:
            try:
                return await request()
            except anthropic.APIError as error:
                raise as_openai(error) from error

        return await reconnecting(create, self._client.base_url, self._reconnect_s, self._sleep)

    def _refuse_images(self, messages: list[ChatCompletionMessageParam]) -> None:
        """Refuse `messages` when they hold an image and vision is off: nothing is sent."""
        if not self.vision and (count := images(messages)):
            raise ValueError(
                f"The conversation holds {count} image(s) but vision is off for this model: no "
                "image may reach it."
            )

    def _converted(
        self, messages: list[ChatCompletionMessageParam]
    ) -> tuple[list[TextBlockParam], list[MessageParam]]:
        """`messages`, in OpenAI's format, as Anthropic's API takes them: the system prompt, and
        the turns, a role's messages in a row merged into one, each result first in its turn,
        the cache marked at the end of the system prompt and of the first turn (fixed for the
        phase), and at the end of the model's last reply: the results after it shrink once read,
        the conversation up to it does not."""
        system: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        previous = ""
        for message in cast(list[dict[str, Any]], messages):
            role = str(message["role"])
            content = message.get("content")
            if role in ("system", "developer"):
                if text := _plain(content):
                    system.append({"type": "text", "text": text})
            elif role == "assistant":
                _join(turns, "assistant", self._assistant(message))
            elif role == "tool":
                result = {
                    "type": "tool_result",
                    "tool_use_id": str(message["tool_call_id"]),
                    "content": _plain(content) or "(empty)",
                }
                _join(turns, "user", [result])
            elif previous == "tool" and isinstance(content, list) and turns:
                # The figures that follow a result: inside it.
                last = cast(dict[str, Any], turns[-1]["content"][-1])
                if last.get("type") == "tool_result":
                    said = last["content"]
                    blocks = [{"type": "text", "text": said}] if isinstance(said, str) else said
                    last["content"] = [*blocks, *_user(content)]
                else:
                    _join(turns, "user", _user(content))
            else:
                _join(turns, "user", _user(content))
            previous = role
        if system:
            system[-1]["cache_control"] = self._fixed
        if len(turns) > 1 and turns[0]["role"] == "user":
            turns[0]["content"][-1]["cache_control"] = self._fixed
        replies = [index for index, turn in enumerate(turns) if turn["role"] == "assistant"]
        if replies:
            turns[replies[-1]]["content"][-1]["cache_control"] = CACHED
        return cast(list[TextBlockParam], system), cast(list[MessageParam], turns)

    def _assistant(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """An assistant message as content blocks: the reply as it came when this client made it
        (its thinking given back with its calls), else its text and calls."""
        calls = cast(list[dict[str, Any]], message.get("tool_calls") or [])
        if calls and (kept := self._replies.get(str(calls[0]["id"]))) is not None:
            return [dict(block) for block in kept]
        blocks: list[dict[str, Any]] = []
        if text := _plain(message.get("content")).strip():
            blocks.append({"type": "text", "text": text})
        for call in calls:
            function = cast(dict[str, Any], call["function"])
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(call["id"]),
                    "name": str(function["name"]),
                    "input": _input(function.get("arguments")),
                }
            )
        return blocks


def as_openai(error: anthropic.APIError) -> openai.APIError:
    """`error` as the OpenAI SDK's error of the same kind."""
    if isinstance(error, anthropic.APITimeoutError):
        return openai.APITimeoutError(request=error.request)
    if isinstance(error, anthropic.APIConnectionError):
        return openai.APIConnectionError(message=error.message, request=error.request)
    if isinstance(error, anthropic.APIStatusError):
        status = error.status_code
        default = openai.InternalServerError if status >= 500 else openai.APIStatusError
        return _STATUS.get(status, default)(error.message, response=error.response, body=error.body)
    return openai.APIError(error.message, error.request, body=error.body)


def _tools(
    tools: Sequence[ChatCompletionFunctionToolParam], cached: dict[str, Any] = CACHED
) -> list[ToolParam]:
    """The tools as Anthropic's API takes them, the last marked for the cache (`cached`)."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = cast(dict[str, Any], tool["function"])
        converted.append(
            {
                "name": function["name"],
                "description": function.get("description") or "",
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    if converted:
        converted[-1]["cache_control"] = cached
    return cast(list[ToolParam], converted)


def _join(turns: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    """`blocks` added to the turns: into the last one when it is `role`'s, the results first."""
    if not blocks:
        return
    if not turns or turns[-1]["role"] != role:
        turns.append({"role": role, "content": blocks})
        return
    merged = [*turns[-1]["content"], *blocks]
    results = [block for block in merged if block.get("type") == "tool_result"]
    turns[-1]["content"] = results + [
        block for block in merged if block.get("type") != "tool_result"
    ]


def _user(content: object) -> list[dict[str, Any]]:
    """A user message's content as blocks: its text, and its images from their data URLs."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    blocks: list[dict[str, Any]] = []
    for part in cast(list[dict[str, Any]], content or []):
        if part.get("type") == "text" and str(part.get("text") or "").strip():
            blocks.append({"type": "text", "text": str(part["text"])})
        elif part.get("type") == "image_url":
            url = str(cast(dict[str, Any], part["image_url"])["url"])
            if (found := _DATA_URL.fullmatch(url)) is not None:
                source = {"type": "base64", "media_type": found["type"], "data": found["data"]}
            else:
                source = {"type": "url", "url": url}
            blocks.append({"type": "image", "source": source})
    return blocks


def _plain(content: object) -> str:
    """A message's text: its string, or its text parts."""
    if isinstance(content, str):
        return content
    parts = cast(list[dict[str, Any]], content or [])
    return "\n".join(str(part.get("text") or "") for part in parts if part.get("type") == "text")


def _input(arguments: object) -> dict[str, Any]:
    """A call's arguments as a tool_use's input: an object, empty when they did not parse."""
    try:
        parsed = json.loads(str(arguments or "{}"))
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}


def _within_unions(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """`schema` within the unions Claude's API compiles (UNIONS): past them, its last fields of a
    value or nothing sent as their value alone, and no longer required. Those fields too, which
    the form gives back as null where the model leaves them out: the form PACo reads is the same."""
    properties: dict[str, Any] = schema.get("properties") or {}
    unions = [name for name, field in properties.items() if _nullable(field) is not None]
    relaxed = unions[UNIONS:] if len(unions) > UNIONS else []
    if not relaxed:
        return schema, []
    sent = dict(properties)
    for name in relaxed:
        field = properties[name]
        value = _nullable(field) or {}
        sent[name] = {
            **{key: kept for key, kept in field.items() if key not in ("anyOf", "default")},
            **value,
        }
    required = [name for name in schema.get("required", []) if name not in relaxed]
    return {**schema, "properties": sent, "required": required}, relaxed


def _nullable(field: object) -> dict[str, Any] | None:
    """The value a field of a value or nothing takes (`anyOf` of it and null); None for another."""
    if not isinstance(field, dict):
        return None
    options = cast(dict[str, Any], field).get("anyOf")
    if not isinstance(options, list) or len(options) != 2 or {"type": "null"} not in options:
        return None
    return next(option for option in options if option != {"type": "null"})


def _with_nulls(filled: object, relaxed: list[str]) -> str:
    """The form `filled` (the tool's input, or the text of structured outputs) as JSON, the fields
    `relaxed` the model left out, or wrote as the text null (as some models do), given back as
    null."""
    if isinstance(filled, str):
        if not relaxed:
            return filled
        try:
            filled = json.loads(filled)
        except json.JSONDecodeError:
            return filled
    if isinstance(filled, dict):
        found = cast(dict[str, Any], filled)
        gone = [name for name in relaxed if name not in found or found[name] == "null"]
        filled = {**found, **dict.fromkeys(gone)}
    return json.dumps(filled)


def _text(response: Message) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


def _tokens(usage: Usage) -> tuple[int, int, int]:
    """The prompt tokens the model read (those read from the cache and written to it included),
    those read from the cache, and the tokens it wrote."""
    read = usage.cache_read_input_tokens or 0
    written = usage.cache_creation_input_tokens or 0
    return usage.input_tokens + read + written, read, usage.output_tokens
