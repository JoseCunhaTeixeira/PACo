import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from mcp import Client
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from sigpipe.masw import profiles

from paco import server
from paco.agent import (
    Agent,
    OpenAIChat,
    Reply,
    ToolCall,
    chat,
    host,
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
        "pick",
        "inversion_settings",
        "invert",
        "job_status",
        "petro_models",
        "invert_petro",
        "redo",
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


# ---------------------------------------------------------------- a call that just failed


@pytest.mark.usefixtures("paco_env")
def test_a_call_that_just_failed_is_not_made_again() -> None:
    model = ScriptedModel(
        _calls(("inspect_profile", {"profile": "active_p2"})),
        # The same call again, its JSON written otherwise: refused, unmade.
        _calls(("inspect_profile", '{ "profile" :"active_p2" }')),
        _calls(("inspect_profile", {"profile": "active_p1"})),
        _says("active_p2 does not exist; active_p1 has 96 receivers."),
    )

    _, events, messages = _converse(model, "Inspect active_p2.")

    results = _tool_results(messages)
    assert results[1].startswith(
        "Not called: this exact call just failed, and would fail again: Error executing tool "
        "inspect_profile: Unknown profile 'active_p2'."
    )
    assert results[1].endswith(
        "Change the arguments, call another tool, or answer the user with what you have."
    )
    assert (
        events[2]
        == '-> inspect_profile({ "profile" :"active_p2" }) refused: the same call just failed'
    )
    # Another call goes through.
    assert json.loads(results[2])["name"] == "active_p1"


# ---------------------------------------------------------------- no question for the user


def test_the_agent_asks_only_when_stuck() -> None:
    # The go or no-go before an inversion is G4's verdict (docs/qc_workflow.md): no tool asks
    # the user anything; the model asks, in its answer, only when the data cannot decide.
    assert "Ask the user only when the request cannot be finished" in ROLE
    assert "2 or 3 concrete options, your choice first" in ROLE
    # Rejected windows are gaps to report; the answer ends without an offer.
    assert "Rejected windows are gaps to report, not a reason to ask" in ROLE
    assert "no offer, no question" in ROLE
    assert "report every item of the results' changed lists" in ROLE
    assert "Ask the user only when stuck" in server.INSTRUCTIONS


# ---------------------------------------------------------------- what the host guarantees


def test_the_hosts_rules() -> None:
    assert host.asks_for_models("Process active_p1 and give me the Vs models.")
    assert host.asks_for_models("pick the curves and invert them quickly")
    assert not host.asks_for_models("Process active_p1 with velocities up to 250 m/s.")
    assert not host.asks_for_models("pick the curves, reaching as deep as this line allows")
    assert host.starts_inversion("invert", "{}")
    assert host.starts_inversion("redo", '{"run_id": "r", "stage": "inversion"}')
    assert not host.starts_inversion("redo", '{"run_id": "r", "stage": "picking"}')
    assert host.asks_for_soils("pick the curves and give me the soil types and the water table")
    assert host.asks_for_soils("Quelle est la profondeur de la nappe ?")
    assert not host.asks_for_soils("Process active_p1 and give me the Vs models.")
    assert host.unasked("invert_petro", "{}", "invert the curves") is not None
    assert host.unasked("invert_petro", "{}", "what are the soils?") is None
    assert host.unasked("invert", "{}", "what are the soils?") is not None
    for asking in (
        "Would you like me to invert them",
        "Let me know if you want more.",
        "1. Redo. 2. Stop. Choose one to proceed.",
        "Which one?",
    ):
        assert host.asks_or_offers(asking)
    assert not host.asks_or_offers("4 curves passed G3 and G4. No further action required.")
    assert host.changed_items('{"run_id": "r", "changed": ["a", "b"]}') == ["a", "b"]
    assert host.changed_items("Error executing tool pick: Unknown run.") == []
    assert host.with_changes("4 curves passed.", ["a"]) == (
        "4 curves passed.\n\nSettings the gates changed:\n- a"
    )
    assert host.with_changes("4 curves passed.", []) == "4 curves passed."


class PolicyModel(ScriptedModel):
    """Stands in for Qwen with a policy: its reply read from the conversation so far."""

    def __init__(self, policy: Callable[[list[ChatCompletionMessageParam]], Reply]) -> None:
        super().__init__()
        self._policy = policy

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
    ) -> Reply:
        self.seen.append(list(messages))
        self.tools = tools
        return self._policy(messages)


@pytest.mark.usefixtures("paco_env")
def test_the_host_lists_the_settings_the_gates_changed() -> None:
    def picks(messages: list[ChatCompletionMessageParam]) -> Reply:
        results = _tool_results(messages)
        if not results:
            return _calls(("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}))
        if len(results) == 1:
            return _calls(("pick", {"run_id": json.loads(results[0])["run_id"]}))
        return _says("3 curves passed G3 and G4.")

    answer, _, messages = _converse(PolicyModel(picks), "Process active_p1 and pick the curves.")

    head, listed = answer.split("\n\nSettings the gates changed:\n")
    assert head == "3 curves passed G3 and G4."
    assert listed.splitlines()[:2] == [
        "- trigger t0 the default -> 0.0188 at 1.dat, 2.dat (each window its own), by "
        "G1:shifted_trigger",
        "- 2.dat: traces [1, 13, 89, 90] left out of the windows",
    ]
    # What the user saw is what the conversation keeps.
    assert messages[-1] == {"role": "assistant", "content": answer}


@pytest.mark.usefixtures("paco_env")
def test_an_answer_that_asks_after_the_work_is_asked_again_once() -> None:
    def offers(messages: list[ChatCompletionMessageParam]) -> Reply:
        if not _tool_results(messages):
            return _calls(("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}))
        if messages[-1]["role"] == "tool":
            return _says("The images are ready. Shall I pick the curves?")
        return _says("The images are ready. Shall I pick the curves now?")

    answer, events, messages = _converse(PolicyModel(offers), "Process active_p1.")

    # Asked again once, then the answer is the model's, question or not.
    assert answer.startswith("The images are ready. Shall I pick the curves now?")
    assert {"role": "user", "content": host.ANSWER_AGAIN} in messages
    assert "   (asked to answer again, without a question or an offer)" in events


@pytest.mark.usefixtures("paco_env")
def test_a_question_before_any_work_reaches_the_user() -> None:
    # 120 receivers asked on a 96-receiver line: the model asks before running anything. Asked
    # again, it may run 96 receivers unasked.
    question = (
        "active_p1 has only 96 receivers, so windows of 120 cannot fit. Which should I use: 96 "
        "(the whole line), 48, or 24?"
    )
    model = ScriptedModel(_calls(("inspect_profile", {"profile": "active_p1"})), _says(question))

    answer, events, _ = _converse(model, "Process active_p1 with windows of 120 receivers.")

    assert answer == question
    assert not [event for event in events if "asked to answer again" in event]


@pytest.mark.usefixtures("paco_env")
def test_no_inversion_starts_unless_the_user_asked_for_models() -> None:
    model = ScriptedModel(
        _calls(("invert", {"run_id": "20260925-100000-abcd"})),
        _says("The curves are picked."),
    )

    _, events, messages = _converse(model, "Pick the curves of run 20260925-100000-abcd.")

    (result,) = _tool_results(messages)
    assert result == (
        "Not called: the user asked for no velocity model, so no inversion starts. Answer with "
        "what the request asked for."
    )
    assert events[0] == '-> invert({"run_id": "20260925-100000-abcd"}) refused: not asked for'
    # Asked for, the call goes to the server.
    model = ScriptedModel(_calls(("invert", {"run_id": "20260925-100000-abcd"})), _says("No run."))
    _, _, messages = _converse(model, "Invert run 20260925-100000-abcd.")
    assert "Unknown run" in _tool_results(messages)[0]


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
