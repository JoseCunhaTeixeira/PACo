import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.shared.dispatcher import ProgressFnT
from mcp.shared.exceptions import MCPError
from mcp.types import (
    REQUEST_TIMEOUT,
    CallToolResult,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
)
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco import inspection, server
from paco.agent import (
    Agent,
    Filled,
    Limits,
    OpenAIChat,
    Reply,
    ScopeStep,
    ToolCall,
    chat,
    host,
    result_for_model,
    without_thinking,
)
from paco.agent.answer import SCHEMA as ANSWER_SCHEMA
from paco.agent.loop import ROLE, UNREAD, as_data, from_data
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
    length_receivers=None,
    length_m=None,
    step_receivers=None,
    step_m=None,
    compare_lengths_receivers=[],
    compare_lengths_m=[],
    redo=False,
    replace_hand_work=False,
    option=None,
)


def _scoped(answer: str, scope: Scope = EVERYTHING) -> str:
    """`answer` as the user reads it: after the scope's line."""
    return f"{scope.line()}\n\n{answer}"


def _answer_form(messages: list[ChatCompletionMessageParam]) -> str:
    """The answer form a stand-in fills from the draft it was sent: its statements, and its
    last question apart."""
    draft = str(messages[-1].get("content")).split("\nAnswer: ", 1)[1]
    sentences = re.split(r"(?<=[.?!])\s+", draft)
    asked = [sentence for sentence in sentences if sentence.endswith("?")]
    said = " ".join(sentence for sentence in sentences if not sentence.endswith("?"))
    return json.dumps({"said": said, "question": asked[-1] if asked else None})


class ScriptedModel:
    """Stands in for Qwen: gives its replies in order, fills every form with its scopes (the
    last one again once they run out), and keeps what it was sent."""

    def __init__(self, *replies: Reply, scopes: tuple[Scope | str, ...] = (EVERYTHING,)) -> None:
        self._replies = list(replies)
        self._scopes = list(scopes)
        self.name: str | None = None  # a model behind a server names itself
        self.seen: list[list[ChatCompletionMessageParam]] = []
        self.forms: list[list[ChatCompletionMessageParam]] = []
        self.tools: list[ChatCompletionFunctionToolParam] = []

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any],
    ) -> Filled:
        if schema == ANSWER_SCHEMA:
            return Filled(content=_answer_form(messages))
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
    """The tool messages' results, out of their data blocks."""
    return [from_data(str(message["content"])) for message in messages if message["role"] == "tool"]


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
        "compare",
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
        "   failed: [bad argument] Unknown profile 'active_p2'. Available profiles: active_p1, "
        "passive_p1.",
    ]
    results = _tool_results(messages)
    assert results[0].startswith("Not called: the arguments of inspect are not valid JSON")
    assert results[1] == "Not called: the arguments of inspect must be a JSON object."
    assert results[2] == "Unknown tool: invent_curve"
    assert results[3] == (
        "Error executing tool inspect: [bad argument] Unknown profile 'active_p2'. Available "
        "profiles: active_p1, passive_p1."
    )
    # What the server said comes as data (T10, X3); what the host says, as it is.
    raw = [str(message["content"]) for message in messages if message["role"] == "tool"]
    assert raw[1] == results[1]
    assert raw[3] == as_data("inspect", results[3])
    assert raw[3].startswith('<data from="inspect">\n') and raw[3].endswith("\n</data>")


@pytest.mark.usefixtures("paco_env")
def test_an_answer_has_a_tool_call_budget() -> None:
    model = ScriptedModel(
        _calls(
            ("inspect", {"what": "profiles"}),
            ("inspect", {"what": "runs"}),
            ("inspect", {"what": "profile", "profile": "active_p1"}),
        ),
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
        "inspect: [bad argument] Unknown profile 'active_p2'."
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
    # Rejected windows are gaps to report; PACo writes what was done and what next.
    assert "rejected windows (gaps to report), are never a reason to ask" in ROLE
    assert "do not repeat them, and offer nothing" in ROLE
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
    assert host.job_running('{"state": "running"}')
    assert not host.job_running('{"state": "succeeded"}')


class PolicyModel(ScriptedModel):
    """Stands in for Qwen with a policy: its reply read from the conversation so far."""

    def __init__(
        self,
        policy: Callable[[list[ChatCompletionMessageParam]], Reply],
        scopes: tuple[Scope | str, ...] = (EVERYTHING,),
    ) -> None:
        super().__init__(scopes=scopes)
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
    # The model's text, then what the tools did, as their results say it.
    assert said.startswith(_scoped("3 curves passed G3 and G4.\n\nDone:\n- Processed run "))
    assert "\n- Picked run " in said
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
def test_after_the_work_an_offer_gives_way_to_what_the_user_can_do_next() -> None:
    # No word is read: the turn did the work it was asked, no tool left the user a choice, so
    # the model's question is not shown; code says what the user can ask next.
    def offers(messages: list[ChatCompletionMessageParam]) -> Reply:
        if not _tool_results(messages):
            return _calls(("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}))
        return _says("The images are ready. Shall I pick the curves?")

    process = EVERYTHING.model_copy(update={"pick": False, "invert": False, "soils": False})
    answer, events, messages = _converse(PolicyModel(offers, (process,)), "Process active_p1.")

    said, rest = answer.split("\n\nDone:\n", 1)
    assert said == _scoped("The images are ready.", process)
    assert "Shall I" not in answer
    assert (
        "\n\nNext: the curves (ask to pick them); the results in PAC's Visualization page." in rest
    )
    assert [message["role"] for message in messages].count("user") == 1
    assert events[0].startswith("-> run_processing(")


@pytest.mark.usefixtures("paco_env")
def test_a_question_before_any_work_reaches_the_user() -> None:
    # The request's clarification: no stage tool ran, so the question is the model's to ask.
    model = ScriptedModel(
        _calls(("inspect", {"what": "profiles"})),
        _says("Two profiles hold an active line. Which one, active_p1 or passive_p1?"),
    )

    answer, _, _ = _converse(model, "Process my active line.")

    assert answer == _scoped(
        "Two profiles hold an active line.\n\nWhich one, active_p1 or passive_p1?"
    )


class MetaServer:
    """Stands in for PACo's server: keeps each call and what it carried, and answers {}."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.metas: list[dict[str, Any]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        read_timeout_seconds: float | None = None,  # noqa: ARG002
        progress_callback: ProgressFnT | None = None,  # noqa: ARG002
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
    # A model that names itself is named, for what its calls make (S4).
    model.name = "Qwen/Qwen3-14B-FP8"
    named = MetaServer()
    agent = Agent(named, model, [], None, on_event=lambda _: None)  # pyright: ignore[reportArgumentType]
    assert agent.meta["model"] == "Qwen/Qwen3-14B-FP8"


class OfferServer(MetaServer):
    """Stands in for PACo's server: pick without windows offers two options, doing nothing;
    every other call answers {}."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        read_timeout_seconds: float | None = None,  # noqa: ARG002
        progress_callback: ProgressFnT | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        await super().call_tool(name, arguments, None, progress_callback, meta)
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


def test_once_a_tool_offered_options_only_reading_runs_until_the_user_chooses() -> None:
    # The model goes round the options: it inverts the curves as they are, one of them, and
    # lists the runs; the choice is the user's.
    model = ScriptedModel(
        _calls(("pick", {"run_id": "r"})),
        _calls(("invert", {"run_id": "r"}), ("inspect", {"what": "runs"})),
        _says("Run r holds 3 curves."),
    )
    server_ = OfferServer()

    answer, agent, events = _played(model, server_, "Pick and invert run r.")

    assert [name for name, _ in server_.calls] == ["pick", "inspect"]
    assert '-> invert({"run_id": "r"}) refused: the user chooses first' in events
    (refused,) = [step for step in agent.steps if isinstance(step, ToolStep) and not step.called]
    assert refused.result.startswith("Not called: a tool offered the user options above")
    # The options stand in the answer, the listing after them notwithstanding.
    assert answer.endswith(
        "Which do you choose?\n(1) complete the 1 windows without a curve\n"
        "(2) pick every window again"
    )


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


def test_compare_serves_a_message_asking_to_process_and_nothing_else() -> None:
    # Comparing settings is processing a sample of the line: refused to a message that only
    # looks, made for one that asks to process (the scope guard knows its stage).
    look = EVERYTHING.model_copy(
        update={"process": False, "pick": False, "invert": False, "soils": False}
    )
    process = look.model_copy(update={"process": True})
    variants = {"profile": "active_p1", "variants": [{}, {}], "metric": "depth"}
    for scope, ran in ((look, []), (process, ["compare"])):
        model = ScriptedModel(_calls(("compare", variants)), _says("Compared."), scopes=(scope,))
        server_ = MetaServer()

        _, _, events = _played(model, server_, "Compare two lengths on active_p1.")

        assert [name for name, _ in server_.calls] == ran
        assert bool(ran) != events[0].endswith("refused: outside the scope")


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
        read_timeout_seconds: float | None = None,  # noqa: ARG002
        progress_callback: ProgressFnT | None = None,
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
            assert progress_callback is not None
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
    (followed,) = [step for step in agent.steps if isinstance(step, ToolStep) and step.by_host]
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
                "model": "Qwen/Qwen3-14B-FP8",
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
        return await OpenAIChat(client, "Qwen/Qwen3-14B-FP8")(messages, tools)

    reply = anyio.run(ask)

    assert reply == Reply(content="", tool_calls=(ToolCall("call_1", "list_profiles", "{}"),))
    (request,) = requests
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert (body["model"], body["messages"], body["tools"]) == (
        "Qwen/Qwen3-14B-FP8",
        messages,
        tools,
    )
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
                "model": "Qwen/Qwen3-14B-FP8",
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
        return await OpenAIChat(client, "Qwen/Qwen3-14B-FP8", 0.6, 7).fill(messages, SCHEMA)

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


# ---------------------------------------------------------------- the loop's caps and state


def _with_tokens(reply: Reply, prompt: int, written: int) -> Reply:
    return Reply(reply.content, reply.tool_calls, prompt_tokens=prompt, completion_tokens=written)


def _played(
    model: ScriptedModel,
    server_: object,
    question: str = "Which runs?",
    limits: Limits | None = None,
) -> tuple[str, Agent, list[str]]:
    """The answer of an agent on the stand-in server `server_`, the agent, and its events."""
    events: list[str] = []

    async def conversation() -> tuple[str, Agent]:
        client = cast(Client, server_)
        agent = Agent(client, model, [], None, on_event=events.append, limits=limits)
        return await agent.answer(question), agent

    answer, agent = anyio.run(conversation)
    return answer, agent, events


def test_a_call_made_already_is_refused_and_a_third_ends_the_answer() -> None:
    runs = _calls(("inspect", {"what": "runs"}))
    server_ = MetaServer()

    answer, agent, events = _played(ScriptedModel(runs, runs, runs, runs), server_)

    assert [name for name, _ in server_.calls] == ["inspect"]  # made once
    refused = _tool_results(agent.messages)[1:]
    assert len(refused) == 2 and all(
        text.startswith("Not called: this exact call") for text in refused
    )
    assert events[-1] == "   (stopped: inspect called again, the same way)"
    assert answer == _scoped(
        "PACo stopped this answer: inspect was called again with the same arguments, without "
        "progress."
    )


def test_an_answer_stops_at_its_caps() -> None:
    model = ScriptedModel(
        _with_tokens(_calls(("inspect", {"what": "runs"})), 100, 30),
        _with_tokens(_says("Never read."), 100, 30),
    )

    answer, _, events = _played(model, MetaServer(), limits=Limits(tokens=20))

    assert answer == _scoped(
        "PACo stopped this answer: it reached its cap of 20 tokens written by the model."
    )
    assert events[-1] == "   (stopped: cap of 20 tokens written by the model)"


class SlowServer(MetaServer):
    """Stands in for PACo's server: every call outlives its timeout."""

    async def call_tool(
        self,
        name: str,  # noqa: ARG002
        arguments: dict[str, Any],  # noqa: ARG002
        read_timeout_seconds: float | None = None,
        progress_callback: ProgressFnT | None = None,  # noqa: ARG002
        meta: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> CallToolResult:
        raise MCPError(REQUEST_TIMEOUT, f"no answer in {read_timeout_seconds} s")


def test_a_tool_call_has_a_timeout() -> None:
    model = ScriptedModel(_calls(("inspect", {"what": "runs"})), _says("The tool took too long."))

    _, agent, events = _played(model, SlowServer(), limits=Limits(tool_seconds=2))

    (result,) = _tool_results(agent.messages)
    assert result.startswith('{"error"') or "No result in 2 s" in result
    assert "   failed: no result in 2 s" in events


class RunsServer(MetaServer):
    """Stands in for PACo's server: its resource names the profile's latest run."""

    async def read_resource(self, uri: str) -> ReadResourceResult:
        assert uri == "paco://profiles/active_p1/latest-run"
        text = "Run 20260930-161253-89f5: active_p1 (active). Holds 4 images, 4 M0 curves."
        return ReadResourceResult(contents=[TextResourceContents(uri=uri, text=text)])


def test_the_model_is_told_the_profiles_latest_run() -> None:
    # The host supplies the run's id (M4): the model makes none up.
    on_active = EVERYTHING.model_copy(update={"profile": "active_p1"})
    model = ScriptedModel(_says("Run 20260930-161253-89f5 holds 4 curves."), scopes=(on_active,))

    _, agent, _ = _played(model, RunsServer(), "What does active_p1 hold?")

    told = str(agent.messages[1].get("content"))
    assert told.endswith(
        "active_p1's latest run: Run 20260930-161253-89f5: active_p1 (active). Holds 4 images, "
        "4 M0 curves."
    )


def test_earlier_results_are_kept_short_and_the_trace_whole() -> None:
    model = ScriptedModel(
        _with_tokens(_calls(("pick", {"run_id": "r"})), 95, 10),
        _says("Run r holds 3 curves."),
    )

    _, agent, events = _played(model, OfferServer(), "Pick run r.", Limits(context=100))

    (kept,) = _tool_results(agent.messages)
    assert json.loads(kept) == {
        "run_id": "r",
        "options": ["complete the 1 windows without a curve", "pick every window again"],
    }
    (step,) = [step for step in agent.steps if isinstance(step, ToolStep)]
    assert (
        '"summary":"Run r holds 3 curves."' in step.result.replace(" ", "")
        or "summary" in step.result
    )
    assert "   (the conversation fills 95% of the model's context)" in events
