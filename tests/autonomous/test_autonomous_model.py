"""The model client: PACo's ChatModel protocol, with images, thinking per call, the cached tokens
and several calls a reply, against a scripted OpenAI-compatible server."""

from typing import Any

import anyio
import openai
import pytest
from autonomous_scripted import ANSWERED, MODEL, PNG, Answer, ScriptedServer, usage
from mcp.types import CallToolResult, ImageContent, TextContent
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from pydantic import SecretStr

from paco.agent import model as paco_model
from paco.agent.model import FORM_TOKENS, ToolCall
from paco.autonomous.conversion import tool_messages
from paco.llm.chat import chat_model, tried
from paco.llm.claude import AnthropicChat
from paco.llm.model import OPENAI_FORM_TOKENS, PROBE, Filled, OpenAIChat, Reply, images
from paco.llm.settings import ModelSettings, Provider

TOOLS: list[ChatCompletionFunctionToolParam] = [
    {"type": "function", "function": {"name": "look", "parameters": {}}},
    {"type": "function", "function": {"name": "observe", "parameters": {}}},
]
QUESTION: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Is the image of xmid 2.88 clean?"}
]
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"profile": {"type": "string"}},
    "required": ["profile"],
    "additionalProperties": False,
}


def test_a_figure_after_a_tool_result_reaches_the_model_and_the_usage_comes_back() -> None:
    server = ScriptedServer(
        ({"role": "assistant", "content": "One clean ridge."}, usage(1200, 40, cached=1024))
    )
    result = CallToolResult(
        content=[
            TextContent(type="text", text="xmid 2.88: its dispersion image"),
            ImageContent(type="image", data=PNG, mime_type="image/png"),
        ]
    )
    messages: list[ChatCompletionMessageParam] = [
        *QUESTION,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "look", "arguments": "{}"},
                }
            ],
        },
        *tool_messages("call_1", "look", result, vision=True),
    ]

    async def ask() -> Reply:
        return await OpenAIChat(server.client, MODEL, vision=True)(messages, TOOLS)

    reply = anyio.run(ask)

    assert reply == Reply(
        content="One clean ridge.",
        tool_calls=(),
        prompt_tokens=1200,
        completion_tokens=40,
        cached_tokens=1024,
    )
    tool, figure = server.bodies[0]["messages"][2:]
    assert tool["role"] == "tool"
    assert tool["content"].endswith("The figure follows, in the next message.")
    assert figure["role"] == "user"
    assert figure["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{PNG}"},
    }


def test_without_vision_no_image_leaves_the_machine() -> None:
    server = ScriptedServer()
    messages: list[ChatCompletionMessageParam] = [
        *QUESTION,
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
            ],
        },
    ]
    model = OpenAIChat(server.client, MODEL)

    async def ask() -> Reply:
        return await model(messages, TOOLS)

    async def fill() -> Filled:
        return await model.fill(messages, SCHEMA)

    with pytest.raises(ValueError, match="vision is off"):
        anyio.run(ask)
    with pytest.raises(ValueError, match="vision is off"):
        anyio.run(fill)
    assert images(messages) == 1
    assert server.bodies == []


def test_thinking_is_the_clients_unless_a_call_says_otherwise_and_never_in_a_form() -> None:
    server = ScriptedServer(
        ({"role": "assistant", "content": '{"profile": "active_test_profile"}'}, None)
    )

    async def calls() -> None:
        thinking = OpenAIChat(server.client, MODEL)
        await thinking(QUESTION, TOOLS)
        await thinking(QUESTION, TOOLS, thinking=False)
        await thinking.fill(QUESTION, SCHEMA)
        quiet = OpenAIChat(server.client, MODEL, thinking=False)
        await quiet(QUESTION, TOOLS)
        await quiet(QUESTION, TOOLS, thinking=True)

    anyio.run(calls)

    switches = [body["chat_template_kwargs"]["enable_thinking"] for body in server.bodies]
    assert switches == [True, False, False, False, True]


def test_several_calls_come_in_one_reply_only_when_asked() -> None:
    both = [
        {"id": f"call_{number}", "type": "function", "function": {"name": name, "arguments": "{}"}}
        for number, name in enumerate(("look", "observe"))
    ]
    server = ScriptedServer(({"role": "assistant", "content": None, "tool_calls": both}, None))

    async def calls() -> list[Reply]:
        model = OpenAIChat(server.client, MODEL)
        return [
            await model(QUESTION, TOOLS),
            await model(QUESTION, TOOLS, parallel=True),
            await model(QUESTION, []),
        ]

    _, parallel, _ = anyio.run(calls)

    assert parallel.tool_calls == (
        ToolCall("call_0", "look", "{}"),
        ToolCall("call_1", "observe", "{}"),
    )
    assert [body.get("parallel_tool_calls") for body in server.bodies] == [False, True, None]
    # Offered no tool, the request names none.
    assert "tools" not in server.bodies[2]


def test_a_form_is_filled_under_its_schema_as_paco_fills_one() -> None:
    server = ScriptedServer(
        (
            {"role": "assistant", "content": '{"profile": "active_test_profile"}'},
            usage(400, 20, cached=384),
        )
    )

    async def fill() -> Filled:
        return await OpenAIChat(server.client, MODEL).fill(QUESTION, SCHEMA)

    filled = anyio.run(fill)

    assert filled == Filled(
        content='{"profile": "active_test_profile"}',
        prompt_tokens=400,
        completion_tokens=20,
        cached_tokens=384,
    )
    body = server.bodies[0]
    assert body["response_format"]["json_schema"] == {
        "name": "form",
        "strict": True,
        "schema": SCHEMA,
    }
    assert (body["temperature"], body["max_tokens"]) == (0, FORM_TOKENS)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("said", "tokens"),
    [
        (None, (None, None, None)),
        (usage(900, 30), (900, None, 30)),
        (usage(900, 30, cached=0), (900, 0, 30)),
    ],
)
def test_the_usage_is_what_the_server_says(
    said: dict[str, Any] | None, tokens: tuple[int | None, int | None, int | None]
) -> None:
    # The thinking Qwen3 writes when no reasoning parser removes it is left out, as in PACo.
    server = ScriptedServer(
        ({"role": "assistant", "content": "<think>\nplan\n</think>\n\nTwo profiles."}, said)
    )

    async def ask() -> Reply:
        return await OpenAIChat(server.client, MODEL)(QUESTION, TOOLS)

    reply = anyio.run(ask)

    assert reply.content == "Two profiles."
    assert (reply.prompt_tokens, reply.cached_tokens, reply.completion_tokens) == tokens


def test_a_server_that_refuses_the_connection_is_asked_again_for_a_while() -> None:
    # The tunnel to the model's server restarting: refused, then there again.
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    def reply(*answers: Answer, reconnect_s: float = 30.0) -> Reply:
        server = ScriptedServer(*answers)
        model = OpenAIChat(server.client, MODEL, reconnect_s=reconnect_s, sleep=sleep)

        async def ask() -> Reply:
            return await model(QUESTION, TOOLS)

        return anyio.run(ask)

    assert reply("refused", "refused", ANSWERED).content == "Two profiles."
    assert waits == [1.0, 2.0]
    # Refused past its budget: the error, the waits doubling up to it.
    waits.clear()
    with pytest.raises(openai.APIConnectionError):
        reply("refused", reconnect_s=10.0)
    assert waits == [1.0, 2.0, 4.0, 3.0]
    # A timeout is not asked again: the server was there.
    waits.clear()
    with pytest.raises(openai.APITimeoutError):
        reply("slow", ANSWERED)
    assert waits == []


def test_the_client_is_pacos_chat_model() -> None:
    # PACo's agent calls its model through PACo's protocol: the same client serves both agents.
    server = ScriptedServer()
    model: paco_model.ChatModel = OpenAIChat(server.client, MODEL)

    async def ask() -> paco_model.Reply:
        return await model(QUESTION, TOOLS)

    assert anyio.run(ask).content == "Two profiles."
    # One call a reply, as PACo's own client asks.
    assert server.bodies[0]["parallel_tool_calls"] is False


def test_the_client_is_made_from_its_settings() -> None:
    settings = ModelSettings(llm_base_url="http://gpu.test:8001/v1", llm_model=MODEL)

    model = chat_model(settings)

    assert (model.name, model.vision, model.thinking) == (MODEL, True, True)
    assert isinstance(model, OpenAIChat)
    # A server unless the settings say otherwise, as PACo's agent had it.
    assert model.dialect == "server"


@pytest.mark.parametrize(
    ("provider", "address", "dialect"),
    [
        ("openai", "https://api.openai.com/v1/", "openai"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini"),
        ("mistral", "https://api.mistral.ai/v1/", "plain"),
        ("server", "http://gpu.test:8001/v1/", "server"),
    ],
)
def test_each_provider_has_its_dialect_and_address(
    provider: Provider, address: str, dialect: str
) -> None:
    served = "http://gpu.test:8001/v1" if provider == "server" else None
    settings = ModelSettings(
        llm_provider=provider, llm_base_url=served, llm_model="m", llm_api_key=SecretStr("sk-user")
    )

    model = chat_model(settings)

    assert isinstance(model, OpenAIChat)
    assert (model.dialect, str(model._client.base_url)) == (dialect, address)  # pyright: ignore[reportPrivateUsage]


def test_claude_has_its_own_client() -> None:
    settings = ModelSettings(llm_provider="anthropic", llm_model="claude-sonnet-5-5")

    model = chat_model(settings)

    assert isinstance(model, AnthropicChat)
    assert (model.name, model.thinking) == ("claude-sonnet-5-5", True)


def test_each_dialect_sends_what_its_server_takes() -> None:
    server = ScriptedServer(
        ({"role": "assistant", "content": '{"profile": "active_test_profile"}'}, None)
    )

    async def calls() -> None:
        reasoning = OpenAIChat(server.client, MODEL, dialect="openai")
        await reasoning(QUESTION, TOOLS)
        await reasoning.fill(QUESTION, SCHEMA)
        plain = OpenAIChat(server.client, MODEL, dialect="plain")
        await plain(QUESTION, TOOLS)
        await plain.fill(QUESTION, SCHEMA)

    anyio.run(calls)

    bodies = server.bodies
    # OpenAI: no vLLM switch, the model's own reasoning effort; its forms at the model's
    # temperature, the reasoning within their tokens.
    assert not any("chat_template_kwargs" in body or "reasoning_effort" in body for body in bodies)
    assert "temperature" not in bodies[1] and "max_tokens" not in bodies[1]
    assert bodies[1]["max_completion_tokens"] == OPENAI_FORM_TOKENS
    # Mistral's: nothing beside OpenAI's chat API.
    assert (bodies[3]["temperature"], bodies[3]["max_tokens"]) == (0, FORM_TOKENS)


def test_a_server_that_refuses_vllms_switch_is_asked_without_it() -> None:
    # A gateway in front of another API refuses what it does not know: asked again without the
    # switch, and never sent it again.
    server = ScriptedServer(400, ANSWERED)

    async def calls() -> None:
        model = OpenAIChat(server.client, MODEL)
        await model(QUESTION, TOOLS)
        await model.fill(QUESTION, SCHEMA)

    anyio.run(calls)

    assert ["chat_template_kwargs" in body for body in server.bodies] == [True, False, False]


def test_a_request_refused_for_another_reason_keeps_the_switch() -> None:
    server = ScriptedServer(400, 400, ANSWERED)
    model = OpenAIChat(server.client, MODEL)

    with pytest.raises(openai.BadRequestError):
        anyio.run(model, QUESTION, TOOLS)
    anyio.run(model, QUESTION, TOOLS)

    assert ["chat_template_kwargs" in body for body in server.bodies] == [True, False, True]


def test_a_connection_is_tried_with_one_short_call() -> None:
    server = ScriptedServer(({"role": "assistant", "content": "OK"}, usage(12, 2)))
    model = OpenAIChat(server.client, MODEL)

    assert anyio.run(tried, model) == "OK"

    (body,) = server.bodies
    assert body["messages"] == [{"role": "user", "content": PROBE}]
    # No tool, no thinking: a few tokens.
    assert "tools" not in body and body["chat_template_kwargs"] == {"enable_thinking": False}
