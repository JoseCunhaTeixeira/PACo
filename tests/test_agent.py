import builtins
import json
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.client.session import ClientRequestContext, ElicitationFnT
from mcp.types import ElicitRequestFormParams, ElicitRequestParams, ElicitResult
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco import profiles, server
from paco.agent import (
    Agent,
    OpenAIChat,
    Reply,
    ToolCall,
    ask_the_user,
    chat,
    result_for_model,
    without_thinking,
)
from paco.agent.loop import ROLE
from paco.settings import Settings

# Four 24-receiver windows along the active demo line, as in test_runs.py.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window.
SHORT = {"n_iterations": 500, "n_burnin_iterations": 50, "n_chains": 1}


class ScriptedModel:
    """Stands in for Qwen: gives its replies in order, and keeps what it was sent."""

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.seen: list[list[ChatCompletionMessageParam]] = []
        self.tools: list[ChatCompletionFunctionToolParam] = []

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply:
        self.seen.append(list(messages))
        self.tools = tools
        return self._replies.pop(0)


def _calls(*calls: tuple[str, dict[str, Any] | str]) -> Reply:
    """A reply asking for tool calls, with their arguments as a dict, or as raw text."""
    return Reply(
        content="",
        tool_calls=tuple(
            ToolCall(
                id=f"call_{index}",
                name=name,
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
            )
            for index, (name, arguments) in enumerate(calls)
        ),
    )


def _says(text: str) -> Reply:
    return Reply(content=text, tool_calls=())


def _tool_results(messages: list[ChatCompletionMessageParam]) -> list[str]:
    return [str(message["content"]) for message in messages if message["role"] == "tool"]


def _converse(
    model: ScriptedModel, question: str, max_tool_calls: int = 15
) -> tuple[str, list[str], list[ChatCompletionMessageParam]]:
    """One question to an agent on PACo's server, in memory: the answer, what the user saw of the
    tool calls, and the conversation."""
    events: list[str] = []

    async def conversation() -> tuple[str, list[ChatCompletionMessageParam]]:
        async with Client(server.server) as client:
            agent = await Agent.start(client, model, max_tool_calls, on_event=events.append)
            return await agent.answer(question), agent.messages

    answer, messages = anyio.run(conversation)
    return answer, events, messages


# ---------------------------------------------------------------- what the model gets


def test_the_model_gets_every_tool_card_and_the_workflow() -> None:
    model = ScriptedModel(_says("Hello."))

    _, _, messages = _converse(model, "Hello?")

    assert all(tool["type"] == "function" for tool in model.tools)
    assert [tool["function"]["name"] for tool in model.tools] == [
        "list_profiles",
        "inspect_profile",
        "preset_settings",
        "run_processing",
        "quality_settings",
        "dispersion_quality",
        "pick",
        "inversion_settings",
        "invert",
        "job_status",
    ]
    assert messages[0] == {"role": "system", "content": f"{ROLE}\n\n{server.INSTRUCTIONS}"}


def test_the_agent_calls_tools_until_the_model_answers(paco_env: Settings) -> None:
    model = ScriptedModel(
        _calls(("list_profiles", {})),
        _calls(("inspect_profile", {"profile": "active_p1"})),
        _says("Two profiles; active_p1 is an active line of 96 receivers."),
    )

    answer, events, messages = _converse(model, "What can I process?")

    assert answer == "Two profiles; active_p1 is an active line of 96 receivers."
    assert events == ["-> list_profiles({})", '-> inspect_profile({"profile": "active_p1"})']
    summary = profiles.inspect_profile("active_p1", paco_env).model_dump(mode="json")
    # Compact JSON, not the server's indented text.
    assert _tool_results(messages) == [
        '{"result":["active_p1","passive_p1"]}',
        json.dumps(summary, separators=(",", ":")),
    ]
    # Each reply saw the results before it.
    assert model.seen[2][-1] == messages[-2]


@pytest.mark.usefixtures("paco_env")
def test_settings_tools_give_their_schema_as_it_is() -> None:
    async def call() -> str:
        async with Client(server.server) as client:
            return result_for_model(await client.call_tool("inversion_settings", {}))

    text = anyio.run(call)

    assert json.loads(text)["properties"]["n_layers"]["default"] == 2
    assert '\\"' not in text  # not a JSON string inside JSON


# ---------------------------------------------------------------- mistakes


@pytest.mark.usefixtures("paco_env")
def test_mistakes_go_back_to_the_model() -> None:
    model = ScriptedModel(
        _calls(
            ("inspect_profile", "{profile: active_p1"),
            ("inspect_profile", "[1]"),
            ("invent_curve", {}),
            ("inspect_profile", {"profile": "active_p2"}),
        ),
        _says("Sorry, I will check the profile's name."),
    )

    _, events, messages = _converse(model, "Inspect active_p2.")

    # The user sees the calls made, and why they failed; calls refused unmade stay silent.
    assert events == [
        "-> invent_curve({})",
        "   failed: Unknown tool: invent_curve",
        '-> inspect_profile({"profile": "active_p2"})',
        "   failed: Unknown profile 'active_p2'. Available profiles: active_p1, passive_p1.",
    ]
    results = _tool_results(messages)
    assert results[0].startswith("Not called: the arguments of inspect_profile are not valid JSON")
    assert results[1] == "Not called: the arguments of inspect_profile must be a JSON object."
    assert results[2] == "Unknown tool: invent_curve"
    assert results[3] == (
        "Error executing tool inspect_profile: Unknown profile 'active_p2'. Available profiles: "
        "active_p1, passive_p1."
    )


@pytest.mark.usefixtures("paco_env")
def test_an_answer_has_a_tool_call_budget() -> None:
    model = ScriptedModel(
        _calls(("list_profiles", {}), ("list_profiles", {}), ("list_profiles", {})),
        _says("Two profiles."),
    )

    _, events, messages = _converse(model, "List the profiles.", max_tool_calls=2)

    assert len(events) == 2
    assert _tool_results(messages)[2] == (
        "Not called: this answer already made 2 tool calls. Answer the user with what you have, "
        "and say what is left to do."
    )


# ---------------------------------------------------------------- the user's approval


@pytest.mark.usefixtures("paco_env")
def test_the_model_never_sees_the_approval_question() -> None:
    asked: list[str] = []

    async def user(
        context: ClientRequestContext,  # noqa: ARG001
        params: ElicitRequestParams,
    ) -> ElicitResult:
        asked.append(params.message)
        return ElicitResult(action="accept", content={"approve": True})

    async def conversation() -> tuple[str, list[ChatCompletionMessageParam]]:
        async with Client(server.server, elicitation_callback=user) as client:
            processed = await client.call_tool(
                "run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}
            )
            run_id = (processed.structured_content or {})["run_id"]
            await client.call_tool("dispersion_quality", {"run_id": run_id})
            await client.call_tool("pick", {"run_id": run_id})

            model = ScriptedModel(
                _calls(("invert", {"run_id": run_id, "parameters": SHORT})),
                _says("The inversion has started."),
            )
            agent = await Agent.start(client, model, on_event=lambda _: None)
            await agent.answer("Invert the run.")
            job = json.loads(_tool_results(agent.messages)[0])
            # Let the job end before its folder is deleted.
            while True:
                status = await client.call_tool("job_status", {"job_id": job["job_id"]})
                if (status.structured_content or {})["state"] not in ("queued", "running"):
                    break
                await anyio.sleep(0.2)
            return job["state"], agent.messages

    state, messages = anyio.run(conversation)

    assert state == "queued"
    (question,) = asked
    assert question.startswith("The agent asks to invert run")
    assert all(question not in str(message.get("content")) for message in messages)


def _question() -> ElicitRequestFormParams:
    return ElicitRequestFormParams(
        message="Approve?",
        requested_schema={"type": "object", "properties": {"approve": {"type": "boolean"}}},
    )


@pytest.mark.parametrize(
    ("typed", "result"),
    [
        ("y", ElicitResult(action="accept", content={"approve": True})),
        ("yes", ElicitResult(action="accept", content={"approve": True})),
        ("n", ElicitResult(action="decline")),
        ("", ElicitResult(action="decline")),
    ],
)
def test_the_terminal_asks_the_user(
    monkeypatch: pytest.MonkeyPatch, typed: str, result: ElicitResult
) -> None:
    def type_it(_prompt: str) -> str:
        return typed

    monkeypatch.setattr(builtins, "input", type_it)
    callback: ElicitationFnT = ask_the_user

    async def ask() -> ElicitResult:
        return await callback(cast(ClientRequestContext, None), _question())  # type: ignore[return-value]

    assert anyio.run(ask) == result


# ---------------------------------------------------------------- the model behind vLLM's API


def test_openai_chat_sends_the_conversation_and_reads_tool_calls() -> None:
    requests: list[httpx2.Request] = []

    def vllm(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        message = {
            "role": "assistant",
            "content": "<think>\nThe user wants the profiles.\n</think>\n\n",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_profiles", "arguments": "{}"},
                }
            ],
        }
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen/Qwen3-8B",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
            },
        )

    client = AsyncOpenAI(
        base_url="http://vllm.test/v1",
        api_key="secret",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(vllm)),
    )
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Profiles?"}]
    tools: list[ChatCompletionFunctionToolParam] = [
        {"type": "function", "function": {"name": "list_profiles", "parameters": {}}}
    ]

    async def ask() -> Reply:
        return await OpenAIChat(client, "Qwen/Qwen3-8B")(messages, tools)

    reply = anyio.run(ask)

    assert reply == Reply(content="", tool_calls=(ToolCall("call_1", "list_profiles", "{}"),))
    (request,) = requests
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert (body["model"], body["messages"], body["tools"]) == ("Qwen/Qwen3-8B", messages, tools)
    # Each call is chosen after the result of the one before.
    assert body["parallel_tool_calls"] is False


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("<think>\nplan\n</think>\n\nTwo profiles.", "Two profiles."),
        ("Two profiles.", "Two profiles."),
    ],
)
def test_thinking_is_not_kept(text: str, kept: str) -> None:
    assert without_thinking(text) == kept


def test_chat_says_which_settings_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env
    for name in ("PACO_LLM_BASE_URL", "PACO_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    anyio.run(chat)

    assert capsys.readouterr().out == (
        "Set PACO_LLM_BASE_URL, PACO_LLM_MODEL in .env (vLLM's address and model name).\n"
    )
