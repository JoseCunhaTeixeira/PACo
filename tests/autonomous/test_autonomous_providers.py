"""The providers: a model that takes no images refused, as
Anthropic's models API or a server refusing an image says it; Gemini's dialect, its
thought signatures given back. The APIs are scripted
(tests/autonomous/autonomous_scripted.py): nothing is called."""

from typing import Any

import anyio
from autonomous_scripted import ScriptedAnthropic, ScriptedServer, capabilities, usage
from openai.types.chat import ChatCompletionMessageParam

from paco.llm.chat import blind
from paco.llm.claude import AnthropicChat
from paco.llm.model import SKIP_SIGNATURE, OpenAIChat

MESSAGES: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Give me the Vs."}]
# A form's schema as pydantic writes one (an optional field: anyOf).
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"profile": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    "required": ["profile"],
}
FORM = ({"role": "assistant", "content": '{"profile": "active_test_profile"}'}, usage(400, 20))


# ---------------------------------------------------------------- vision


def test_a_model_that_takes_no_images_is_refused() -> None:
    text_only = ScriptedAnthropic(described=capabilities(images=False))
    multimodal = ScriptedAnthropic()
    chats = [
        AnthropicChat(server.client, "claude-sonnet-5-5", vision=True)
        for server in (text_only, multimodal)
    ]

    said = [anyio.run(blind, chat) for chat in chats]

    assert said[0] == (
        "claude-sonnet-5-5 takes no images: the autonomous mode judges every image by eye, so it "
        "needs a model that sees them. Choose one, or the workflow mode (PACO_AGENT_MODE=workflow)."
    )
    assert said[1] is None
    # The models API said it: no request was made of the model.
    assert text_only.bodies == multimodal.bodies == []


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


def test_gemini_gets_its_thought_signatures_back() -> None:
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
    chat = OpenAIChat(server.client, "gemini-3.6-flash", dialect="gemini")
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
    # No reasoning effort in a reply: the model's own.
    assert "reasoning_effort" not in first and "reasoning_effort" not in second
    calls = [
        message["tool_calls"][0] for message in second["messages"] if message.get("tool_calls")
    ]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": SKIP_SIGNATURE}}
    assert calls[1]["extra_content"] == signed
    # A form: room for the thinking Gemini cannot leave out, little of it.
    assert (form["max_tokens"], form["reasoning_effort"]) == (4000, "low")
    assert "temperature" not in form
