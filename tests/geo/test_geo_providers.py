"""The providers (docs/geo_plan.md §3, §16): Claude on Vertex AI, Bedrock and Foundry, described by
what the model answers where no models API says it, its forms filled the way it takes; Azure's and
Gemini's APIs in their dialects, Gemini's thought signatures given back; each provider's settings;
and vision switched off for a model that takes no images. The APIs are scripted (tests/scripted.py):
no cloud is called."""

import importlib.util
import json
from typing import Any

import anyio
import pytest
from anthropic import AsyncAnthropicFoundry
from anthropic.lib.bedrock import AsyncAnthropicBedrockMantle
from geo_scripted import Accepts, ScriptedAnthropic, ScriptedServer, usage
from openai.types.chat import ChatCompletionMessageParam
from pydantic import SecretStr, ValidationError

from paco.llm.chat import connect, seeing
from paco.llm.claude import FILL, PROBE_SCHEMA, AnthropicChat
from paco.llm.model import SKIP_SIGNATURE, OpenAIChat
from paco.llm.settings import ModelSettings, missing

MESSAGES: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Give me the Vs."}]
# A form's schema as pydantic writes one (an optional field: anyOf).
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"profile": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    "required": ["profile"],
}
FORM = ({"role": "assistant", "content": '{"profile": "active_p2"}'}, usage(400, 20))
KEY = SecretStr("key")


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


# ---------------------------------------------------------------- the settings and the clients


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"llm_provider": "anthropic-vertex"}, "PACO_LLM_REGION, PACO_LLM_PROJECT"),
        ({"llm_provider": "gemini-vertex", "llm_region": "global"}, "PACO_LLM_PROJECT"),
        ({"llm_provider": "anthropic-bedrock"}, "PACO_LLM_REGION"),
        ({"llm_provider": "azure-openai"}, "PACO_LLM_BASE_URL"),
        ({"llm_provider": "anthropic-foundry"}, "PACO_LLM_BASE_URL"),
    ],
)
def test_each_cloud_asks_what_reaches_it(fields: dict[str, str], named: str) -> None:
    with pytest.raises(ValidationError) as error:
        ModelSettings(llm_model="a-model", **fields)  # pyright: ignore[reportArgumentType]

    assert missing(error.value) == named


def test_each_provider_gets_its_client_and_address() -> None:
    bedrock = connect("anthropic-bedrock", "anthropic.claude-sonnet-5-5", KEY, region="us-east-1")
    foundry = connect(
        "anthropic-foundry", "claude-sonnet-5-5", KEY, "https://paco.services.ai.azure.com"
    )
    azure = connect("azure-openai", "gpt-5-deployment", KEY, "https://paco.openai.azure.com/")
    gemini = connect("gemini", "gemini-3.6-flash", KEY)

    assert isinstance(bedrock, AnthropicChat) and isinstance(foundry, AnthropicChat)
    assert isinstance(bedrock._client, AsyncAnthropicBedrockMantle)  # pyright: ignore[reportPrivateUsage]
    assert str(bedrock._client.base_url) == "https://bedrock-mantle.us-east-1.api.aws/anthropic/"  # pyright: ignore[reportPrivateUsage]
    assert isinstance(foundry._client, AsyncAnthropicFoundry)  # pyright: ignore[reportPrivateUsage]
    assert str(foundry._client.base_url) == "https://paco.services.ai.azure.com/anthropic/"  # pyright: ignore[reportPrivateUsage]
    # Azure's v1 API, the deployment's name as the model; Gemini's compatible API.
    assert isinstance(azure, OpenAIChat) and isinstance(gemini, OpenAIChat)
    assert (azure.dialect, str(azure._client.base_url)) == (  # pyright: ignore[reportPrivateUsage]
        "openai",
        "https://paco.openai.azure.com/openai/v1/",
    )
    assert (gemini.dialect, str(gemini._client.base_url)) == (  # pyright: ignore[reportPrivateUsage]
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )


@pytest.mark.skipif(_has("google.auth"), reason="google-auth is there: Vertex AI connects")
def test_vertex_ai_without_its_library_says_what_to_install() -> None:
    for provider in ("anthropic-vertex", "gemini-vertex"):
        with pytest.raises(ValueError, match="need google-auth: `uv sync --extra vertex`"):
            connect(provider, "a-model", SecretStr("EMPTY"), region="global", project="p")  # pyright: ignore[reportArgumentType]


@pytest.mark.skipif(_has("boto3"), reason="boto3 is there: Bedrock connects with AWS's credentials")
def test_bedrock_without_a_key_needs_aws_library() -> None:
    with pytest.raises(ValueError, match="need boto3: `uv sync --extra bedrock`"):
        connect("anthropic-bedrock", "anthropic.claude-sonnet-5-5", SecretStr("EMPTY"), region="r")


# ---------------------------------------------------------------- Claude on the clouds


def test_claude_on_vertex_is_asked_what_it_takes_and_fills_forms_under_the_schema() -> None:
    # As Sonnet 5.5 on Vertex AI: it always thinks, takes the effort and structured outputs.
    server = ScriptedAnthropic(
        ({"role": "assistant", "content": "Two profiles."}, None),
        FORM,
        platform="vertex",
        accepts=Accepts(disabled=False),
    )
    chat = AnthropicChat(server.client, "claude-sonnet-5-5", effort="high")

    async def talk() -> tuple[str, str]:
        reply = await chat(MESSAGES, [])
        filled = await chat.fill(MESSAGES, SCHEMA)
        return reply.content, filled.content

    reply, filled = anyio.run(talk)

    assert (reply, json.loads(filled)) == ("Two profiles.", {"profile": "active_p2"})
    # Asked once, by requests of a few tokens (thinking off refused): adaptive thinking, the
    # effort, the schema.
    assert [(probe.get("thinking"), probe.get("output_config")) for probe in server.probes] == [
        ({"type": "adaptive"}, None),
        (None, {"effort": "high"}),
        (None, {"format": {"type": "json_schema", "schema": PROBE_SCHEMA}}),
    ]
    talked, form = server.bodies
    assert talked["thinking"] == {"type": "adaptive"}
    assert talked["output_config"] == {"effort": "high"}
    assert "format" in form["output_config"] and "tool_choice" not in form
    # Vertex AI's address, the model in it; Google's token.
    assert server.urls[-1] == (
        "https://us-east5-aiplatform.googleapis.com/v1/projects/paco-project/locations/us-east5/"
        "publishers/anthropic/models/claude-sonnet-5-5:rawPredict"
    )
    assert set(server.credentials) == {"Bearer ya29.token"}


def test_claude_on_bedrock_fills_a_form_through_its_offered_tool() -> None:
    # As Sonnet 5.5 on Bedrock's Messages endpoint: no thinking off, no forced tool, no schema.
    server = ScriptedAnthropic(
        FORM,
        platform="bedrock",
        accepts=Accepts(disabled=False, forced=False, structured=False),
    )
    chat = AnthropicChat(server.client, "anthropic.claude-sonnet-5-5")

    filled = anyio.run(lambda: chat.fill(MESSAGES, SCHEMA))

    assert json.loads(filled.content) == {"profile": "active_p2"}
    (form,) = server.bodies
    assert form["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert [tool["name"] for tool in form["tools"]] == ["form"]
    assert form["system"][-1]["text"] == FILL
    assert server.urls[-1] == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
    assert set(server.credentials) == {"Bearer bedrock-key"}


def test_claude_on_foundry_forces_its_form_where_thinking_can_be_off() -> None:
    server = ScriptedAnthropic(FORM, platform="foundry", accepts=Accepts(adaptive=False))
    chat = AnthropicChat(server.client, "claude-haiku-4-5")

    filled = anyio.run(lambda: chat.fill(MESSAGES, SCHEMA))

    assert json.loads(filled.content) == {"profile": "active_p2"}
    (form,) = server.bodies
    assert (form["tool_choice"], form["thinking"]) == (
        {"type": "tool", "name": "form"},
        {"type": "disabled"},
    )
    assert server.urls[-1] == "https://paco.services.ai.azure.com/anthropic/v1/messages"
    assert set(server.credentials) == {"foundry-key"}


# ---------------------------------------------------------------- vision


def test_vision_goes_off_for_a_model_that_takes_no_images() -> None:
    blind = ScriptedAnthropic(platform="vertex", accepts=Accepts(images=False))
    seeing_model = ScriptedAnthropic(platform="vertex")
    chats = [
        AnthropicChat(server.client, "claude-sonnet-5-5", vision=True)
        for server in (blind, seeing_model)
    ]

    said = [anyio.run(seeing, chat) for chat in chats]

    assert said[0] == (
        "claude-sonnet-5-5 takes no images: vision is off for this run, the figures come as "
        "ridge tables and measures."
    )
    assert said[1] is None
    assert [chat.vision for chat in chats] == [False, True]
    # Asked with an image of one pixel, which the first refused.
    assert not any(_imaged(probe) for probe in blind.probes)
    assert _imaged(seeing_model.probes[-1])


def test_a_server_refusing_the_image_takes_none() -> None:
    text_only = ScriptedServer(400)
    multimodal = ScriptedServer(({"role": "assistant", "content": "OK"}, None))

    takes = [
        anyio.run(OpenAIChat(server.client, "a-model", dialect="plain").takes_images)
        for server in (text_only, multimodal)
    ]

    assert takes == [False, True]
    (probe,) = multimodal.bodies
    assert probe["messages"][0]["content"][0]["type"] == "image_url"


# ---------------------------------------------------------------- Gemini's dialect


def test_gemini_gets_its_thought_signatures_back_and_its_efforts() -> None:
    signed = {"google": {"thought_signature": "sig-1"}}
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "survey", "arguments": "{}"},
        "extra_content": signed,
    }
    server = ScriptedServer(
        ({"role": "assistant", "content": "", "tool_calls": [call]}, None),
        ({"role": "assistant", "content": "Done."}, None),
        FORM,
    )
    chat = OpenAIChat(server.client, "gemini-3.6-flash", dialect="gemini", effort="max")
    unsigned = {"id": "old", "type": "function", "function": {"name": "observe", "arguments": "{}"}}

    async def talk() -> None:
        reply = await chat(MESSAGES, [])
        (made,) = reply.tool_calls
        history: list[Any] = [
            *MESSAGES,
            {"role": "assistant", "content": "", "tool_calls": [unsigned]},
            {"role": "tool", "tool_call_id": "old", "content": "G3: 2 pass."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": made.id,
                        "type": "function",
                        "function": {"name": made.name, "arguments": made.arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": made.id, "content": "96 receivers."},
        ]
        await chat(history, [])
        await chat.fill(MESSAGES, SCHEMA)

    anyio.run(talk)

    first, second, form = server.bodies
    # Its levels: the settings' max is its high.
    assert first["reasoning_effort"] == second["reasoning_effort"] == "high"
    calls = [
        message["tool_calls"][0] for message in second["messages"] if message.get("tool_calls")
    ]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": SKIP_SIGNATURE}}
    assert calls[1]["extra_content"] == signed
    # A form: room for the thinking Gemini cannot leave out, little of it.
    assert (form["max_tokens"], form["reasoning_effort"]) == (4000, "low")
    assert "temperature" not in form


def _imaged(body: dict[str, Any]) -> bool:
    """Whether a request to Claude holds an image."""
    return any(part.get("type") == "image" for part in body["messages"][0]["content"])
