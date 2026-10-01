import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.shared.dispatcher import ProgressFnT
from mcp.types import CallToolResult, TextContent
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco import inspection, server
from paco.agent import (
    Agent,
    Filled,
    OpenAIChat,
    Reply,
    ScopeStep,
    ToolCall,
    chat,
    host,
    result_for_model,
    without_thinking,
)
from paco.agent.loop import ROLE, UNREAD
from paco.agent.record import ToolStep
from paco.agent.scope import SCHEMA, Scope
from paco.settings import Settings

# Four 24-receiver windows along the active demo line, as in test_runs.py.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window.
SHORT = {"n_iterations": 500, "n_burnin_iterations": 50, "n_chains": 2}  # two at least


# A message asking every stage: the tools behave as without a scope.
EVERYTHING = Scope(
    process=True,
    pick=True,
    invert=True,
    soils=True,
    profile=None,
    run_id=None,
    positions_m=[],
    redo=False,
    replace_hand_work=False,
    option=None,
)


def _scoped(answer: str, scope: Scope = EVERYTHING) -> str:
    """`answer` as the user reads it: after the scope's line."""
    return f"{scope.line()}\n\n{answer}"


class ScriptedModel:
    """Stands in for Qwen: gives its replies in order, fills every form with its scopes (the
    last one again once they run out), and keeps what it was sent."""

    def __init__(self, *replies: Reply, scopes: tuple[Scope | str, ...] = (EVERYTHING,)) -> None:
        self._replies = list(replies)
        self._scopes = list(scopes)
        self.seen: list[list[ChatCompletionMessageParam]] = []
        self.forms: list[list[ChatCompletionMessageParam]] = []
        self.tools: list[ChatCompletionFunctionToolParam] = []

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        assert schema == SCHEMA
        self.forms.append(list(messages))
        scope = self._scopes.pop(0) if len(self._scopes) > 1 else self._scopes[0]
        return Filled(content=scope if isinstance(scope, str) else scope.model_dump_json())

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
        "inspect",
        "preset_settings",
        "run_processing",
        "pick",
        "judge",
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
        _calls(("inspect", {"what": "profiles"})),
        _calls(("inspect", {"what": "profile", "profile": "active_p1"})),
        _says("Two profiles; active_p1 is an active line of 96 receivers."),
    )

    answer, events, messages = _converse(model, "What can I process?")

    assert answer == _scoped("Two profiles; active_p1 is an active line of 96 receivers.")
    assert events == [
        '-> inspect({"what": "profiles"})',
        '-> inspect({"what": "profile", "profile": "active_p1"})',
    ]
    # The inspection's own text, as the tool writes it.
    assert _tool_results(messages) == [
        inspection.profiles_text(paco_env),
        inspection.profile_text("active_p1", paco_env),
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
            ("inspect", "{what: profile"),
            ("inspect", "[1]"),
            ("invent_curve", {}),
            ("inspect", {"what": "profile", "profile": "active_p2"}),
        ),
        _says("Sorry, I will check the profile's name."),
    )

    _, events, messages = _converse(model, "Inspect active_p2.")

    # The user sees the calls made, and why they failed; calls refused unmade stay silent.
    assert events == [
        "-> invent_curve({})",
        "   failed: Unknown tool: invent_curve",
        '-> inspect({"what": "profile", "profile": "active_p2"})',
        "   failed: Unknown profile 'active_p2'. Available profiles: active_p1, passive_p1.",
    ]
    results = _tool_results(messages)
    assert results[0].startswith("Not called: the arguments of inspect are not valid JSON")
    assert results[1] == "Not called: the arguments of inspect must be a JSON object."
    assert results[2] == "Unknown tool: invent_curve"
    assert results[3] == (
        "Error executing tool inspect: Unknown profile 'active_p2'. Available profiles: "
        "active_p1, passive_p1."
    )


@pytest.mark.usefixtures("paco_env")
def test_an_answer_has_a_tool_call_budget() -> None:
    model = ScriptedModel(
        _calls(*(("inspect", {"what": "profiles"}),) * 3),
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
        _calls(("inspect", {"what": "profile", "profile": "active_p2"})),
        # The same call again, its JSON written otherwise: refused, unmade.
        _calls(("inspect", '{ "profile" :"active_p2", "what": "profile" }')),
        _calls(("inspect", {"what": "profile", "profile": "active_p1"})),
        _says("active_p2 does not exist; active_p1 has 96 receivers."),
    )

    _, events, messages = _converse(model, "Inspect active_p2.")

    results = _tool_results(messages)
    assert results[1].startswith(
        "Not called: this exact call just failed, and would fail again: Error executing tool "
        "inspect: Unknown profile 'active_p2'."
    )
    assert results[1].endswith(
        "Change the arguments, call another tool, or answer the user with what you have."
    )
    assert events[2] == (
        '-> inspect({ "profile" :"active_p2", "what": "profile" }) refused: the same call just '
        "failed"
    )
    # Another call goes through.
    assert results[2].startswith("active_p1: active, 2 records, 96 receivers")


# ---------------------------------------------------------------- no question for the user


def test_the_agent_asks_what_the_request_leaves_open_or_when_stuck() -> None:
    # The go or no-go before an inversion is G4's verdict (docs/qc_workflow.md). The model asks,
    # in its answer, what the request leaves to the user (work already there, work made by
    # hand) or what the data cannot decide; the settings are its own.
    assert "Ask the user when the request leaves open what they want" in ROLE
    assert "for a stage whose work is there already, the tool gives the user's options" in ROLE
    assert "the work they made by hand in PAC's pages, verified by them" in ROLE
    assert "2 to 4 concrete options, your choice first" in ROLE
    # Rejected windows are gaps to report; the answer ends without an offer.
    assert "rejected windows (gaps to report), are never a reason to ask" in ROLE
    assert "no offer, no question" in ROLE
    assert "report every item of the results' changed lists" in ROLE
    assert "Answer in the user's language" in ROLE
    assert "Ask when stuck, or for a tool's choice the request leaves open" in server.INSTRUCTIONS


def test_the_role_has_a_fixed_structure() -> None:
    # The role, what the agent cannot do and where the user does it, who decides, the answer,
    # and when to ask and stop.
    headings = [line for line in ROLE.splitlines() if line.startswith("## ")]
    assert headings == [
        "## How you work",
        "## What you cannot do, and where the user does it",
        "## Who decides",
        "## Your answer",
        "## When you ask, and when you stop",
    ]


# ---------------------------------------------------------------- what the host guarantees


def test_the_host_reads_the_tools_results() -> None:
    assert host.starts_inversion("invert", "{}")
    assert host.starts_inversion("redo", '{"run_id": "r", "stage": "inversion"}')
    assert not host.starts_inversion("redo", '{"run_id": "r", "stage": "picking"}')
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
def test_the_host_lists_the_parameters_used_and_the_settings_the_gates_changed() -> None:
    def picks(messages: list[ChatCompletionMessageParam]) -> Reply:
        results = _tool_results(messages)
        if not results:
            return _calls(("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}))
        if len(results) == 1:
            return _calls(("pick", {"run_id": json.loads(results[0])["run_id"]}))
        return _says("3 curves passed G3 and G4.")

    answer, _, messages = _converse(PolicyModel(picks), "Process active_p1 and pick the curves.")

    head, listed = answer.split("\n\nSettings the gates changed:\n")
    said, used = head.split("\n\nParameters used:\n")
    assert said == _scoped("3 curves passed G3 and G4.")
    # The window length and the band first, then the picker's settings.
    # Each with why: the window length as given, with what its trial windows gave.
    assert used.splitlines()[1].startswith(
        "- MASW windows of 24 receivers (5.75 m), 4 windows, xmid 2.875 to 20.875 m: given (in "
        "your request, or chosen by the agent)"
    )
    assert used.splitlines()[-1].startswith("- picking: M0 tracked along its ridge")
    # No trigger changed: the demo's files say the shot comes 20 ms in, where the first breaks
    # put it, and the muting is off (the trigger is part of it). First the line's rules: the
    # shots' reach, the length as given, the near field. No trace left out for its amplitude in
    # one record: the line judges its receivers over them all.
    first = listed.splitlines()[0]
    assert first.startswith("- line: masw distance_max ")
    assert "near_field distance_m " in first
    assert "trigger" not in listed
    assert "left out of the windows" not in listed
    # What the user saw is what the conversation keeps.
    assert messages[-1] == {"role": "assistant", "content": answer}


@pytest.mark.usefixtures("paco_env")
def test_the_models_answer_reaches_the_user_as_it_is() -> None:
    # The host watches no words: a question or an offer, before the work or after it, is the
    # model's to make.
    def offers(messages: list[ChatCompletionMessageParam]) -> Reply:
        if not _tool_results(messages):
            return _calls(("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}))
        return _says("The images are ready. Shall I pick the curves?")

    answer, events, messages = _converse(PolicyModel(offers), "Process active_p1.")

    assert answer.startswith(_scoped("The images are ready. Shall I pick the curves?"))
    assert [message["role"] for message in messages].count("user") == 1
    assert events[0].startswith("-> run_processing(")
    assert len(events) == len([event for event in events if not event.startswith("   (")])


class MetaServer:
    """Stands in for PACo's server: keeps each call and what it carried, and answers {}."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.metas: list[dict[str, Any]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        progress_callback: ProgressFnT,  # noqa: ARG002
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        assert meta is not None
        self.calls.append((name, arguments))
        self.metas.append(dict(meta))
        return _structured({})


def test_every_call_carries_the_conversation_and_its_turn() -> None:
    # The server replaces work made by hand only once the user could reply to its question: a
    # later turn of the same conversation (paco.choices).
    model = ScriptedModel(
        _calls(("inspect", {"what": "runs"}), ("inspect", {"what": "profiles"})),
        _says("One run."),
        _calls(("inspect", {"what": "runs"})),
        _says("Still one run."),
    )
    server_ = MetaServer()

    async def conversation() -> None:
        agent = Agent(server_, model, [], None, on_event=lambda _: None)  # pyright: ignore[reportArgumentType]
        await agent.answer("Which runs are there?")
        await agent.answer("And now?")

    anyio.run(conversation)

    assert [name for name, _ in server_.calls] == ["inspect", "inspect", "inspect"]
    first, second, third = server_.metas
    assert first["conversation"] == second["conversation"] == third["conversation"]
    assert (first["turn"], second["turn"], third["turn"]) == (1, 1, 2)
    # With the scope of the turn's message, for the server's rules.
    assert first["scope"] == {**EVERYTHING.for_server(), "chosen": None}


class OfferServer(MetaServer):
    """Stands in for PACo's server: pick without windows offers two options, doing nothing;
    every other call answers {}."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        progress_callback: ProgressFnT,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        await super().call_tool(name, arguments, progress_callback, meta)
        if name != "pick" or "windows" in arguments:
            return _structured({})
        options = [
            {
                "label": "complete the 1 windows without a curve",
                "call": 'pick(run_id="r", windows="missing")',
            },
            {"label": "pick every window again", "call": 'pick(run_id="r", windows="all")'},
        ]
        return _structured({"run_id": "r", "summary": "Run r holds 3 curves.", "options": options})


def test_calls_outside_the_messages_scope_are_refused_unmade() -> None:
    look = EVERYTHING.model_copy(
        update={"process": False, "pick": False, "invert": False, "soils": False}
    )
    model = ScriptedModel(
        _calls(("invert", {"run_id": "r"})),
        _calls(("inspect", {"what": "runs"})),
        _says("One run."),
        scopes=(look,),
    )
    server_ = MetaServer()
    events: list[str] = []

    async def conversation() -> Agent:
        agent = Agent(server_, model, [], None, on_event=events.append)  # pyright: ignore[reportArgumentType]
        await agent.answer("Which runs are there?")
        return agent

    agent = anyio.run(conversation)

    assert [name for name, _ in server_.calls] == ["inspect"]  # the refused call never ran
    assert events[0] == '-> invert({"run_id": "r"}) refused: outside the scope'
    (refused, _) = _tool_results(agent.messages)
    assert refused.startswith("Not called: this message asks to look at what exists")
    # The model read the scope after the message; the user reads it before the answer.
    assert agent.messages[1] == {
        "role": "user",
        "content": f"Which runs are there?\n\n{look.for_model(None)}",
    }
    assert agent.messages[-1].get("content") == _scoped("One run.", look)
    (scope_step,) = [step for step in agent.steps if isinstance(step, ScopeStep)]
    assert scope_step.scope == look.model_dump() and scope_step.tries == 1
    assert scope_step.prompt_version.startswith("prompts-")


def test_an_option_chosen_is_the_one_the_tool_offered() -> None:
    first = EVERYTHING.model_copy(update={"process": False, "soils": False, "invert": False})
    second = first.model_copy(update={"option": 2})
    model = ScriptedModel(
        _calls(("pick", {"run_id": "r"})),
        _says("Run r has 3 curves: complete the missing one, or pick every window again?"),
        _calls(("pick", {"run_id": "r", "windows": "all"})),
        _says("Picked again."),
        scopes=(first, second),
    )
    server_ = OfferServer()

    pending: list[int] = []

    async def conversation() -> Agent:
        agent = Agent(server_, model, [], None, on_event=lambda _: None)  # pyright: ignore[reportArgumentType]
        await agent.answer("Pick run r.")
        pending.append(len(agent.offers))  # the answer asks the user to choose
        await agent.answer("The second.")
        pending.append(len(agent.offers))
        return agent

    agent = anyio.run(conversation)

    assert pending == [2, 0]

    # The form read "the second" among the options offered; the model and the server were
    # told the call it makes.
    offered = model.forms[1][-1].get("content")
    assert isinstance(offered, str) and offered.startswith(
        'Offered last: (1) complete the 1 windows without a curve: pick(run_id="r", '
        'windows="missing"); (2) pick every window again: pick(run_id="r", windows="all").'
    )
    assert agent.messages[-4].get("content") == (
        f"The second.\n\n{second.for_model(None)} The user chose (2) pick every window again: "
        'pick(run_id="r", windows="all").'
    )
    assert server_.metas[-1]["scope"]["chosen"] == 'pick(run_id="r", windows="all")'


def test_a_message_whose_scope_cannot_be_read_runs_nothing() -> None:
    model = ScriptedModel(scopes=('{"pick": true}',))
    server_ = MetaServer()

    async def conversation() -> tuple[str, Agent]:
        agent = Agent(server_, model, [], None, on_event=lambda _: None)  # pyright: ignore[reportArgumentType]
        return await agent.answer("Pick active_p1."), agent

    answer, agent = anyio.run(conversation)

    assert answer.startswith(UNREAD.split("(")[0]) and "Field required" in answer
    assert server_.calls == [] and model.seen == []
    (scope_step,) = agent.steps
    assert isinstance(scope_step, ScopeStep) and scope_step.scope is None
    assert scope_step.tries == 2


class JobServer:
    """Stands in for PACo's server: invert starts a job, which job_status finds running twice,
    reporting its progress, then finished."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._running = 2

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        progress_callback: ProgressFnT,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        # Every call carries the conversation, the job's follow-up too.
        assert meta is not None and "conversation" in meta
        self.calls.append(name)
        if name == "invert":
            return _structured({"job_id": "job-1", "state": "queued", "done": 0, "total": 4})
        assert arguments == {"job_id": "job-1"}
        if self._running:
            self._running -= 1
            await progress_callback(
                2 - self._running, 4, f"inverted: {2 - self._running} of 4 windows"
            )
            return _structured({"job_id": "job-1", "state": "running", "done": 1, "total": 4})
        return _structured(
            {
                "job_id": "job-1",
                "state": "succeeded",
                "done": 4,
                "total": 4,
                "changed": ["n_layers 4 -> 5"],
            }
        )


def _structured(value: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(value))], structured_content=value
    )


def test_an_inversion_is_followed_to_its_end_before_the_model_reads_on() -> None:
    model = ScriptedModel(
        _calls(("invert", {"run_id": "20260925-100000-abcd"})), _says("Four models.")
    )
    server_ = JobServer()
    events: list[str] = []

    async def conversation() -> tuple[str, Agent]:
        agent = Agent(server_, model, [], None, on_event=events.append)  # pyright: ignore[reportArgumentType]
        return await agent.answer("Invert run 20260925-100000-abcd."), agent

    answer, agent = anyio.run(conversation)

    # The host called job_status until the job ended; the model saw only its end, as invert's
    # result, and answered once.
    assert server_.calls == ["invert", "job_status", "job_status", "job_status"]
    (result,) = _tool_results(agent.messages)
    assert json.loads(result)["state"] == "succeeded"
    assert len(model.seen) == 2
    assert events == [
        '-> invert({"run_id": "20260925-100000-abcd"})',
        '-> job_status({"job_id": "job-1"})',
        "   job_status: inverted: 1 of 4 windows",
        "   job_status: inverted: 2 of 4 windows",
    ]
    # The settings the job changed are listed after the answer.
    assert answer == _scoped("Four models.\n\nSettings the gates changed:\n- n_layers 4 -> 5")
    followed = agent.steps[-2]
    assert isinstance(followed, ToolStep)
    assert (followed.name, followed.by_host) == ("job_status", True)


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


def test_openai_chat_fills_a_form_under_its_schema() -> None:
    requests: list[httpx2.Request] = []

    def vllm(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen/Qwen3-8B",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"pick": true}'},
                    }
                ],
                "usage": {"prompt_tokens": 400, "completion_tokens": 90, "total_tokens": 490},
            },
        )

    client = AsyncOpenAI(
        base_url="http://vllm.test/v1",
        api_key="secret",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(vllm)),
    )
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Pick it."}]

    async def fill() -> Filled:
        return await OpenAIChat(client, "Qwen/Qwen3-8B", 0.6, 7).fill(messages, SCHEMA)

    filled = anyio.run(fill)

    assert filled == Filled(content='{"pick": true}', prompt_tokens=400, completion_tokens=90)
    body = json.loads(requests[0].content)
    # Constrained to the schema, greedy, without thinking, and capped.
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["temperature"] == 0 and "seed" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] > 0


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
