"""The host (docs/autonomous_plan.md §13 M6): a scripted conversation runs P0 to P5 on active_test_profile through
the autonomous agent's tools; the context stays under the set size; the budgets stop the run as
designed; a run stopped in P2 resumes from the journal with the same phase, budgets and state card.
The model is scripted (tests/scripted.py); the line, 90-receiver windows on a coarse grid (7
windows, three of them picked again along their neighbours, one rejected), a commit in seconds."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import openai
import pytest
from autonomous_scripted import ScriptedServer, usage
from mcp import Client

from paco.autonomous import server
from paco.autonomous.host import budgets
from paco.autonomous.host.card import card
from paco.autonomous.host.loop import CARD, DROPPED, PENDING, Host, Limits
from paco.autonomous.host.phases import phase_of
from paco.autonomous.host.scope import BLANK
from paco.autonomous.tools import journal
from paco.autonomous.tools.given import given_of
from paco.autonomous.tools.trials import Study
from paco.llm.model import OpenAIChat
from paco.settings import Settings, get_settings

PROFILE = "active_test_profile"
LINE = {"masw": {"length": 90}, "dispersion": {"nv": 250}}
REASONS = {"masw length": "the deepest windows E1 tried", "dispersion nv": "a coarse grid"}
SCOPE = {**BLANK, "invert": True, "profile": PROFILE, "effort": "quick"}
REQUEST = "Give me the Vs of active_test_profile, quickly."
INTERPRETATION = "A stiffer layer below 10 m, if the picks are right."
# The models of one quick pass, rejected, naming each fixable flag G5 may raise on them: its
# chains' seeds change with the run's id.
REJECTED_MODELS = (
    "a quick pass only, a first look: not_converged, at_bound, thin_layer, alike_layers, "
    "too_deep or underfit left to a full one"
)
# The window whose curve fails G3 after a pick along its neighbours: its flag named.
NARROW = "narrow_span remains after a guide from the neighbours"
# The curves accepted, then their models inverted and rejected: all but 69.75's.
ACCEPTED = "xmid 66.75-68.25 (2), 71.25-75.75 (4)"
# A small context, for the oldest calls to drop out (§7.3): P2's tools, role and manual take
# about 6,500 tokens of it, past its 60 %.
CONTEXT = 9_000


def call(
    name: str, arguments: dict[str, Any], number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The model's reply asking for one tool call."""
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{number}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps({"profile": PROFILE, **arguments}),
                },
            }
        ],
    }
    return message, usage(1000, 50)


def said(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The model's reply without a call: its text."""
    return {"role": "assistant", "content": text}, usage(1000, 50)


def form(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A form the model fills."""
    return {"role": "assistant", "content": json.dumps(fields)}, usage(800, 30)


def answer_form(text: str, question: str | None = None, reading: str | None = None) -> Any:  # noqa: ANN401
    return form({"said": text, "question": question, "interpretation": reading})


# The play, P0 to P5: the objective's form, then the model's calls, then the report's form.
PLAY = [
    form(SCOPE),
    call("survey", {}, 1),
    call(
        "journal",
        {
            "action": "write",
            "kind": "hypothesis",
            "text": "Windows of 90 receivers reach deepest.",
            "expect": "more trial windows pass",
        },
        2,
    ),
    call("try_settings", {"variants": [LINE], "windows": "spread:3", "hypothesis": "H1"}, 3),
    call(
        "journal",
        {"action": "write", "kind": "conclusion", "text": "2 of 3 pass, none now.", "of": ["E1"]},
        4,
    ),
    call("commit", {"changes": LINE, "reasons": REASONS, "experiment": "E1"}, 5),
    call("observe", {"level": "curves"}, 6),
    call("pick", {"windows": "verdict:retry", "how": "guide", "guide": "neighbours"}, 7),
    call("decide", {"windows": "verdict:pass", "status": "accepted"}, 8),
    call(
        "decide",
        {"windows": "verdict:pass", "status": "accepted", "reason": "the near field noted"},
        9,
    ),
    call("decide", {"windows": "verdict:retry", "status": "rejected", "reason": NARROW}, 10),
    call("freeze", {}, 11),
    call("invert", {"groups": [{"windows": "all"}], "effort": "quick"}, 12),
    call(
        "decide",
        {"windows": "status:pending", "status": "rejected", "reason": REJECTED_MODELS},
        13,
    ),
    said("The 6 curves are frozen and inverted once. " + INTERPRETATION),
    answer_form("The 6 windows' curves are frozen, down to 20 m.", reading=INTERPRETATION),
]


@dataclass(frozen=True)
class Played:
    """A conversation played: the host, the requests the model received, the answer, the
    events the user read, and the study."""

    host: Host
    model: ScriptedServer
    answer: str
    events: list[str]
    study: Study
    settings: Settings


def play(
    answers: list[Any],
    messages: list[str],
    *,
    context: int = 131_072,
    raises: type[BaseException] | None = None,
) -> Played:
    """The scripted model `answers` the user's `messages`, the host between it and the server."""
    model = ScriptedServer(*answers)
    events: list[str] = []

    async def run() -> tuple[Host, str]:
        async with Client(server.server) as client:
            host = await Host.start(
                client,
                OpenAIChat(model.client, "scripted"),
                get_settings(),
                limits=Limits(context=context),
                on_event=events.append,
            )
            reply = ""
            for message in messages:
                reply = await host.answer(message)
            return host, reply

    if raises is not None:
        with pytest.raises(BaseException) as stopped:
            anyio.run(run)
        error = stopped.value
        # Under the MCP client's task group, the error comes in a group of its own.
        while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
            error = error.exceptions[0]
        assert isinstance(error, raises)
        study = _only_study()
        return Played(None, model, "", events, study, get_settings())  # pyright: ignore[reportArgumentType]
    host, reply = anyio.run(run)
    assert host.study is not None
    return Played(host, model, reply, events, host.study, get_settings())


def _only_study() -> Study:
    (folder,) = (get_settings().output_dir / PROFILE / "studies").iterdir()
    return Study(PROFILE, folder)


@pytest.fixture
def folders(input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PACO_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("PACO_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("PACO_WORKERS", "4")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def played(input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Played]:
    root = tmp_path_factory.mktemp("host")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.setenv("PACO_INPUT_DIR", str(input_dir))
        patch.setenv("PACO_OUTPUT_DIR", str(root / "output"))
        patch.setenv("PACO_WORKERS", "4")
        get_settings.cache_clear()
        yield play(PLAY, [REQUEST], context=CONTEXT)
    get_settings.cache_clear()


@pytest.mark.slow
def test_a_scripted_conversation_runs_p0_to_p5(played: Played) -> None:
    assert [entry.id for entry in journal.entries(played.study)] == [
        "O1",
        "H1",
        "E1",
        "C1",
        "L1",
        "F1",
        "I1",
        "R1",
    ]
    assert phase_of(played.study) == "P5"
    assert played.events[0] == (
        f"Objective: balanced, no target depth, quick effort. Study {played.study.id} of "
        f"{PROFILE}; first, the survey."
    )
    assert "Phase: P5, the report." in played.events
    # Each call offered its phase's tools, its manual, and the state card last.
    bodies = [body for body in played.model.bodies if "tools" in body]
    offered = [{tool["function"]["name"] for tool in body["tools"]} for body in bodies]
    assert "survey" in offered[0] and "commit" not in offered[0]
    assert {"commit", "pick", "decide", "freeze"} <= offered[1]
    assert "commit" not in offered[-1]
    assert {"invert", "job_status", "decide"} <= offered[-1]
    assert bodies[0]["messages"][0]["content"].endswith(
        "the usable band, and the first hypotheses to test. They are where the line's settings "
        "start from."
    )
    assert all(body["messages"][-1]["content"].startswith(CARD) for body in bodies)


@pytest.mark.slow
def test_the_report_adds_what_was_tried_and_each_windows_outcome(played: Played) -> None:
    report = played.answer

    assert report.startswith(
        "Scope: invert · active_test_profile · balanced, no target depth, quick effort."
    )
    assert "What I tried:\n- E1 on spread:3: current 0/3; 1 " in report
    assert f"Each window's outcome:\n- curves accepted: {ACCEPTED}" in report
    assert f"Interpretation, by the model (not a measurement): {INTERPRETATION}" in report
    assert (
        "- MASW windows of 90 receivers (133.5 m), 7 windows, xmid 66.75 to 75.75 m: model: the "
        "deepest windows E1 tried (L1, E1)"
    ) in report
    assert "- windows every receiver (1.5 m): the plan's rule" in report
    # P4: the models of one quick pass, decided and reported.
    assert f"- models rejected: {ACCEPTED}" in report
    assert f"- the models at {ACCEPTED}: {REJECTED_MODELS}" in report
    assert "- inversion of 6 windows, the layers chosen by the data" in report
    assert "- 3000 iterations (750 burn-in), 2 chains: the model's effort" in report


@pytest.mark.slow
def test_the_model_reads_the_refusals_and_a_fresh_start_after_the_freeze(played: Played) -> None:
    bodies = [body for body in played.model.bodies if "tools" in body]
    # The note refused without a reason (decide), the model read it next.
    refused = bodies[8]["messages"][-2]
    assert refused["role"] == "tool"
    assert "accepting curves with notes needs a reason" in refused["content"]
    # After the freeze: the request, the frozen curves and the processing, nothing else.
    after = bodies[11]["messages"]
    assert [message["role"] for message in after] == ["system", "user", "user"]
    assert after[1]["content"].startswith(f"{REQUEST}\n\n[PACo] The objective:")
    assert "[PACo] The curves are frozen: Frozen, from L1" in after[1]["content"]


@pytest.mark.slow
def test_the_context_stays_under_the_set_size(played: Played) -> None:
    assert max(played.host.sizes) < CONTEXT
    # Past 60 % of it, the oldest calls dropped out, the model told so.
    assert any(DROPPED in json.dumps(body["messages"]) for body in played.model.bodies)
    # A result once read shrinks to its short form: the experiment to its journal line.
    tool_messages = [one for one in played.host.transcript().messages if one["role"] == "tool"]
    assert all(len(str(one["content"])) < 1200 for one in tool_messages)


@pytest.mark.slow
def test_every_call_goes_to_the_usage_log_and_the_transcript_is_saved(played: Played) -> None:
    used = budgets.uses(played.study)

    assert sum(1 for one in used if one.kind == "model") == 14
    # The model's 13 calls, and the host's job_status following the inversion to its end.
    calls = [one.name for one in used if one.kind == "tool"]
    assert calls.count("job_status") >= 1
    assert len(calls) - calls.count("job_status") == 13
    assert sum(1 for one in used if one.kind == "form") == 2
    (saved,) = (played.study.folder / "transcripts").iterdir()
    assert json.loads(saved.read_text())["prompt_version"].startswith("autonomous-prompts-")


@pytest.mark.slow
@pytest.mark.usefixtures("folders")
def test_the_commits_budget_refuses_a_third_commit() -> None:
    picked = {"picking": {"fmax": 80.0}}
    played = play(
        [
            form(SCOPE),
            call("survey", {}, 1),
            call("commit", {"changes": LINE, "reasons": REASONS}, 2),
            call("commit", {"changes": picked, "reasons": {"picking": "the band"}}, 3),
            call(
                "commit", {"changes": {"picking": {"fmax": 70.0}}, "reasons": {"picking": "x"}}, 4
            ),
            said("Which band should I keep?"),
            answer_form("Two commits made.", question="Which band should I keep?"),
        ],
        [REQUEST],
    )

    refusal = played.model.bodies[5]["messages"][-2]["content"]
    assert refusal == (
        "Not called: the line commits' budget is spent (2 of 2). Pick and decide the windows of "
        "the line as it is."
    )
    assert played.answer.splitlines()[-1].startswith("Where the work stands: Phase: P2")
    assert "Which band should I keep?" in played.answer


@pytest.mark.usefixtures("folders")
def test_the_model_calls_budget_stops_the_run() -> None:
    first = play([form(SCOPE), said("Hello."), answer_form("Hello.", question="Go on?")], [REQUEST])
    for _ in range(60):
        budgets.record(
            first.study,
            budgets.Use(kind="model", phase="P1", at=journal.datetime.now(journal.UTC), seconds=0),
        )

    again = play([form(SCOPE)], ["Go on with active_test_profile."])

    assert again.study == first.study
    assert len(again.model.bodies) == 1  # the objective's form, no call: the budget is spent
    assert "The run's budget of model calls is spent (61 of 60): it stops here." in again.answer


@pytest.mark.slow
@pytest.mark.usefixtures("folders")
def test_a_run_stopped_in_p2_resumes_with_the_same_phase_budgets_and_card() -> None:
    stopped = play(
        [
            form(SCOPE),
            call("survey", {}, 1),
            call("commit", {"changes": LINE, "reasons": REASONS}, 2),
            500,
        ],
        [REQUEST],
        raises=openai.InternalServerError,
    )
    study, settings = stopped.study, stopped.settings
    before = (phase_of(study), budgets.spent(study, "quick"), card(study, settings))

    resumed = play(
        [form(SCOPE), said("Which windows first?"), answer_form("Resumed.", question="Which?")],
        ["Go on with active_test_profile."],
    )

    assert resumed.study == study
    assert before[0] == "P2"
    assert resumed.events[0] == f"Resuming study {study.id} at P2, the curves."
    first_call = next(body for body in resumed.model.bodies if "tools" in body)
    assert first_call["messages"][-1]["content"] == f"{CARD}\n{before[2]}"
    assert budgets.spent(study, "quick").commits == before[1].commits == 1
    assert "This conversation resumes study" in first_call["messages"][1]["content"]
    assert DROPPED not in json.dumps(first_call["messages"])


@pytest.mark.usefixtures("folders")
def test_the_settings_typed_are_kept_and_a_later_message_changes_them() -> None:
    typed = {**SCOPE, "length_receivers": 24, "vmax_m_s": 350.0}
    first = play([form(typed), said("Go on?"), answer_form("Asked.", question="Go on?")], [REQUEST])

    first_call = next(body for body in first.model.bodies if "tools" in body)
    assert (
        "Kept as you gave: masw length 24; dispersion vmax 350"
        in (first_call["messages"][-1]["content"])
    )
    # The user widens the grid: the latest message wins, the journal says so.
    again = play(
        [form({**typed, "vmax_m_s": 600.0}), said("Go on?"), answer_form("Ok.", question="Go?")],
        ["Use velocities up to 600 m/s on active_test_profile."],
    )
    assert again.study == first.study
    assert given_of(again.study).line == {"masw": {"length": 24}, "dispersion": {"vmax": 600.0}}
    (entry,) = (one for one in journal.entries(again.study) if one.kind == "settings")
    assert (entry.id, entry.text) == ("U1", "The user now gives dispersion vmax 600 (was 350).")
    assert "U1: The user now gives dispersion vmax 600 (was 350)." in again.events


@pytest.mark.slow
@pytest.mark.usefixtures("folders")
def test_a_report_with_models_not_decided_sends_the_model_back_once() -> None:
    # The play to the first inversion; then a report before any model is decided.
    early = [said("The 6 windows are inverted."), answer_form("The 6 windows are inverted.")]
    played = play([*PLAY[:13], *early, *PLAY[13:]], [REQUEST])

    sent_back = [
        message["content"]
        for body in played.model.bodies
        for message in body["messages"]
        if message["role"] == "user" and str(message["content"]).startswith("[PACo] The run is")
    ]
    assert sent_back[0] == PENDING.format(count=6, where=ACCEPTED)
    # Then the models decided, and the report.
    assert phase_of(played.study) == "P5"
    assert f"- models rejected: {ACCEPTED}" in played.answer
