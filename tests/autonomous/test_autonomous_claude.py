"""The Anthropic client: PACo's ChatModel over Anthropic's API, against a
scripted one (tests/scripted.py). The host's conversation read into Anthropic's turns, the figures
inside their results, the cache marked; thinking as the model takes it, given back within a tool
loop; forms by a forced tool; the usage with the cache; the errors as OpenAI's. Then a whole
conversation of the host, P0 to P5 on active_test_profile, through it."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import openai
import pytest
from autonomous_scripted import ANSWERED, PNG, Answer, ScriptedAnthropic, capabilities, usage
from mcp import Client
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from test_autonomous_host import PLAY, REQUEST

from paco.autonomous import server as autonomous_server
from paco.autonomous.host.loop import CARD, Host
from paco.autonomous.host.phases import phase_of
from paco.llm.claude import CACHED, FIXED, FORM_TOKENS, AnthropicChat
from paco.llm.model import Filled, Reply
from paco.settings import get_settings

MODEL = "claude-sonnet-5-5"
TOOLS: list[ChatCompletionFunctionToolParam] = [
    {"type": "function", "function": {"name": "look", "parameters": {}}},
    {"type": "function", "function": {"name": "observe", "description": "The gates' results."}},
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
FORM: Answer = (
    {"role": "assistant", "content": '{"profile": "active_test_profile"}'},
    usage(400, 20),
)


def _calls(*names: str) -> list[dict[str, Any]]:
    return [
        {"id": f"toolu_{name}", "type": "function", "function": {"name": name, "arguments": "{}"}}
        for name in names
    ]


def test_the_hosts_conversation_reads_as_anthropics_turns() -> None:
    server = ScriptedAnthropic(
        ({"role": "assistant", "content": "Clean."}, usage(1200, 40, cached=1000))
    )
    figure = f"data:image/png;base64,{PNG}"
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are the autonomous agent of an MASW line."},
        {"role": "user", "content": "Give me the Vs of active_test_profile."},
        {"role": "assistant", "content": "", "tool_calls": _calls("look", "observe")},  # pyright: ignore[reportAssignmentType]
        {"role": "tool", "tool_call_id": "toolu_look", "content": "The figure follows."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The figure of look, as data:"},
                {"type": "image_url", "image_url": {"url": figure}},
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_observe", "content": "G3: 7 of 7 pass."},
        {"role": "user", "content": f"{CARD}\nPhase: P2"},
    ]

    async def ask() -> Reply:
        return await AnthropicChat(server.client, MODEL, vision=True)(messages, TOOLS)

    reply = anyio.run(ask)

    assert (reply.content, reply.prompt_tokens, reply.cached_tokens) == ("Clean.", 1200, 1000)
    body = server.bodies[0]
    # The system prompt apart, cached; the tools, the last cached.
    assert body["system"] == [
        {
            "type": "text",
            "text": "You are the autonomous agent of an MASW line.",
            "cache_control": CACHED,
        }
    ]
    assert body["tools"] == [
        {"name": "look", "description": "", "input_schema": {"type": "object", "properties": {}}},
        {
            "name": "observe",
            "description": "The gates' results.",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": CACHED,
        },
    ]
    request, reply_turn, results = body["messages"]
    # The first turn cached, and the conversation up to the model's last reply.
    assert request == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Give me the Vs of active_test_profile.",
                "cache_control": CACHED,
            }
        ],
    }
    assert reply_turn["content"] == [
        {"type": "tool_use", "id": "toolu_look", "name": "look", "input": {}},
        {
            "type": "tool_use",
            "id": "toolu_observe",
            "name": "observe",
            "input": {},
            "cache_control": CACHED,
        },
    ]
    # The results first, the figure inside its own, the state card after them.
    assert results["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_look",
            "content": [
                {"type": "text", "text": "The figure follows."},
                {"type": "text", "text": "The figure of look, as data:"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": PNG},
                },
            ],
        },
        {"type": "tool_result", "tool_use_id": "toolu_observe", "content": "G3: 7 of 7 pass."},
        {"type": "text", "text": f"{CARD}\nPhase: P2"},
    ]
    assert body["thinking"] == {"type": "adaptive"}
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_the_thinking_goes_back_with_the_calls_it_led_to() -> None:
    thought = {
        "role": "assistant",
        "content": "Looking first.",
        "thinking": "The image before the picks.",
        "tool_calls": _calls("look"),
    }
    server = ScriptedAnthropic((thought, None), ANSWERED)

    async def loop() -> Reply:
        model = AnthropicChat(server.client, MODEL)
        reply = await model(QUESTION, TOOLS)
        calls = [
            {"id": one.id, "type": "function", "function": {"name": one.name, "arguments": "{}"}}
            for one in reply.tool_calls
        ]
        history: list[Any] = [
            *QUESTION,
            {"role": "assistant", "content": reply.content, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "toolu_look", "content": "Clean."},
        ]
        await model(history, TOOLS)
        return reply

    reply = anyio.run(loop)

    assert reply.content == "Looking first."
    _, given_back, _ = server.bodies[1]["messages"]
    assert given_back["content"] == [
        {"type": "thinking", "thinking": "The image before the picks.", "signature": "s"},
        {"type": "text", "text": "Looking first."},
        {
            "type": "tool_use",
            "id": "toolu_look",
            "name": "look",
            "input": {},
            "cache_control": CACHED,
        },
    ]


def test_on_anthropics_own_api_the_fixed_parts_are_cached_an_hour() -> None:
    server = ScriptedAnthropic(ANSWERED)
    history: list[Any] = [
        *QUESTION,
        {"role": "assistant", "content": "", "tool_calls": _calls("look")},
        {"role": "tool", "tool_call_id": "toolu_look", "content": "Clean."},
        {"role": "user", "content": f"{CARD}\nPhase: P2"},
    ]

    async def ask() -> Reply:
        model = AnthropicChat(server.client, MODEL, hour=True)
        return await model([{"role": "system", "content": "The role."}, *history], TOOLS)

    anyio.run(ask)

    body = server.bodies[0]
    # The tools, the system prompt and the first turn, fixed for the phase: an hour; the
    # conversation up to the model's last reply, which the results after it leave the same: 5
    # minutes, after them as the API asks.
    assert body["tools"][-1]["cache_control"] == FIXED
    assert body["system"][-1]["cache_control"] == FIXED
    request, reply_turn, results = body["messages"]
    assert request["content"][-1]["cache_control"] == FIXED
    assert reply_turn["content"][-1]["cache_control"] == CACHED
    assert all("cache_control" not in block for block in results["content"])


@pytest.mark.parametrize(
    ("described", "on", "off"),
    [
        (capabilities(), {"type": "adaptive"}, {"type": "disabled"}),
        (
            capabilities(adaptive=False),
            {"type": "enabled", "budget_tokens": 8_000},
            {"type": "disabled"},
        ),
        # The models API does not describe it: the model's defaults.
        (False, None, None),
    ],
)
def test_thinking_is_as_the_model_takes_it(
    described: Any,  # noqa: ANN401
    on: dict[str, Any] | None,
    off: dict[str, Any] | None,
) -> None:
    server = ScriptedAnthropic(ANSWERED, described=described)

    async def calls() -> None:
        model = AnthropicChat(server.client, MODEL)
        await model(QUESTION, TOOLS)
        await model(QUESTION, TOOLS, thinking=False)

    anyio.run(calls)

    assert [body.get("thinking") for body in server.bodies] == [on, off]
    # The model's own effort.
    assert "output_config" not in server.bodies[0]


def test_a_form_is_the_input_of_the_tool_the_model_must_call() -> None:
    server = ScriptedAnthropic(FORM)

    async def fill() -> Filled:
        return await AnthropicChat(server.client, MODEL).fill(QUESTION, SCHEMA)

    filled = anyio.run(fill)

    assert filled == Filled(
        content='{"profile": "active_test_profile"}',
        prompt_tokens=400,
        completion_tokens=20,
        cached_tokens=0,
    )
    body = server.bodies[0]
    assert body["tool_choice"] == {"type": "tool", "name": "form"}
    assert body["tools"][0]["input_schema"] == SCHEMA
    assert (body["thinking"], body["max_tokens"]) == ({"type": "disabled"}, FORM_TOKENS)


def test_a_form_past_claudes_unions_is_sent_within_them() -> None:
    # As the autonomous mode's scope: more fields of a value or nothing than Claude's API compiles
    # (16; past them it refuses the request).
    nullable = {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}
    fields = {f"f{index:02}": nullable for index in range(20)}
    wide = {
        "type": "object",
        "properties": {"profile": {"type": "string"}, **fields},
        "required": ["profile", *fields],
        "additionalProperties": False,
    }
    server = ScriptedAnthropic(
        (
            {
                "role": "assistant",
                "content": '{"profile": "active_test_profile", "f00": 3, "f17": "null"}',
            },
            None,
        )
    )

    filled = anyio.run(AnthropicChat(server.client, MODEL).fill, QUESTION, wide)

    sent = server.bodies[0]["tools"][0]["input_schema"]
    unions = [name for name, field in sent["properties"].items() if "anyOf" in field]
    # The last ones sent as their value alone, no longer required; the others as they were.
    assert unions == [f"f{index:02}" for index in range(16)]
    assert sent["properties"]["f19"] == {"type": "integer"}
    assert "f19" not in sent["required"] and "f15" in sent["required"]
    # Left out by the model, or written as the text null (as some models write them), given
    # back as null: the form PACo reads is the same.
    answer = json.loads(filled.content)
    assert answer == {
        "profile": "active_test_profile",
        "f00": 3,
        **dict.fromkeys(("f16", "f17", "f18", "f19")),
    }


def test_a_model_that_always_thinks_fills_its_form_under_the_schema() -> None:
    server = ScriptedAnthropic(FORM, described=capabilities(disabled=False))
    listed = {**SCHEMA, "properties": {"positions_m": {"type": "array", "maxItems": 10}}}

    async def fill() -> Filled:
        return await AnthropicChat(server.client, MODEL).fill(QUESTION, listed)

    assert anyio.run(fill).content == '{"profile": "active_test_profile"}'
    body = server.bodies[0]
    assert "tools" not in body and "thinking" not in body
    # The schema as structured outputs take it: a list's longest in its description.
    positions = body["output_config"]["format"]["schema"]["properties"]["positions_m"]
    assert positions == {"type": "array", "description": "{maxItems: 10}"}


@pytest.mark.parametrize(
    ("status", "raised"),
    [
        (400, openai.BadRequestError),
        (404, openai.NotFoundError),
        (429, openai.RateLimitError),
        (500, openai.InternalServerError),
        (529, openai.InternalServerError),  # overloaded: not the model's doing
    ],
)
def test_its_errors_are_raised_as_openais(status: int, raised: type[Exception]) -> None:
    server = ScriptedAnthropic(status)

    async def ask() -> Reply:
        return await AnthropicChat(server.client, MODEL)(QUESTION, TOOLS)

    with pytest.raises(raised):
        anyio.run(ask)


def test_a_server_that_refuses_the_connection_is_asked_again_for_a_while() -> None:
    waits: list[float] = []

    async def wait(seconds: float) -> None:
        waits.append(seconds)

    def reply(*answers: Answer, reconnect_s: float) -> Reply:
        server = ScriptedAnthropic(*answers)
        model = AnthropicChat(server.client, MODEL, reconnect_s=reconnect_s, sleep=wait)

        async def ask() -> Reply:
            return await model(QUESTION, TOOLS)

        return anyio.run(ask)

    assert reply("refused", ANSWERED, reconnect_s=30).content == "Two profiles."
    assert waits == [1.0]
    with pytest.raises(openai.APIConnectionError):
        reply("refused", reconnect_s=0)


def test_without_vision_no_image_leaves_the_machine() -> None:
    server = ScriptedAnthropic()
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
            ],
        }
    ]

    async def ask() -> Reply:
        return await AnthropicChat(server.client, MODEL)(messages, TOOLS)

    with pytest.raises(ValueError, match="vision is off"):
        anyio.run(ask)
    assert server.bodies == []


@pytest.fixture
def folders(input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PACO_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("PACO_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("PACO_WORKERS", "4")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.slow
@pytest.mark.usefixtures("folders", "searched")
@pytest.mark.parametrize(
    "described",
    [capabilities(), capabilities(disabled=False)],
    ids=["forms by a forced tool", "forms under the schema"],
)
def test_a_whole_conversation_runs_through_claude(described: Any) -> None:  # noqa: ANN401
    server = ScriptedAnthropic(*PLAY, described=described)

    async def run() -> tuple[Host, str]:
        async with Client(autonomous_server.server) as client:
            model = AnthropicChat(server.client, MODEL)
            host = await Host.start(client, model, get_settings(), on_event=lambda _: None)
            return host, await host.answer(REQUEST)

    host, answer = anyio.run(run)

    assert host.study is not None and phase_of(host.study) == "P5"
    assert answer.startswith("Scope: invert · active_test_profile")
    forms = [
        body
        for body in server.bodies
        if body.get("tool_choice", {}).get("type") == "tool"
        or "format" in body.get("output_config", {})
    ]
    calls = [body for body in server.bodies if body not in forms]
    # The objective's form and the report's, by the forced tool; the model's calls between.
    assert len(forms) == 2
    assert len(calls) == len(PLAY) - 2
    for body in calls:
        assert body["system"][-1]["cache_control"] == CACHED
        assert body["tools"][-1]["cache_control"] == CACHED
        assert body["thinking"] == {"type": "adaptive"}
        last = body["messages"][-1]
        assert last["role"] == "user"
        assert last["content"][-1]["text"].startswith(CARD)
    # Every call's result came back before the next call.
    results = [
        block["tool_use_id"]
        for block in calls[-1]["messages"][-1]["content"]
        if block["type"] == "tool_result"
    ]
    assert results == ["call_13"]
