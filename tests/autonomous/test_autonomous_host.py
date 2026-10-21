"""The host: a scripted conversation runs P0 to P5 on active_test_profile through
the autonomous agent's tools; the context stays under the set size; the budgets stop the run as
designed; a run stopped in P2 resumes from the journal with the same phase, budgets and state card.
The model is scripted (tests/scripted.py); the line, 78-receiver windows on a coarse grid (19
windows: the 8 whose curves pass accepted, the others rejected once those beside them were picked
again along their neighbours), a commit in seconds."""

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import openai
import pytest
from autonomous_scripted import ScriptedServer, usage
from mcp import Client

from paco.agent.answer import unfound_numbers
from paco.autonomous import prompts, server
from paco.autonomous.host import budgets
from paco.autonomous.host.answer import GeoAnswerForm, report
from paco.autonomous.host.card import card
from paco.autonomous.host.loop import CARD, DROPPED, PENDING, Host, Limits
from paco.autonomous.host.phases import phase_of, reopen, study_for, wanted
from paco.autonomous.host.scope import BLANK, GeoScope
from paco.autonomous.tools import grid, journal
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.commit import LINE_FILE, Line
from paco.autonomous.tools.given import Given, given_of
from paco.autonomous.tools.survey import SAVED
from paco.autonomous.tools.trials import CURVES, STUDIES, Study
from paco.llm.model import OpenAIChat
from paco.settings import Settings, get_settings

PROFILE = "active_test_profile"
LINE = {"masw": {"length": 78, "distance_max": 150.0}, "dispersion": {"nv": 250}}
REASONS = {
    "masw length": "the deepest windows E1 tried",
    "masw distance_max": "every shot of the line, for a test",
    "dispersion nv": "a coarse grid",
}
SCOPE = {**BLANK, "invert": True, "profile": PROFILE, "effort": "quick"}
REQUEST = "Give me the Vs of active_test_profile, quickly."
INTERPRETATION = "A stiffer layer below 10 m, if the picks are right."
# What the model tried, as it says it for a person, from the journal's experiments.
TRIED = (
    "I compared windows of 78 receivers with the current ones on 3 windows spread along the "
    "line: 1 of 3 curves passed either way, so I kept the deeper windows."
)
# The models of one quick pass, rejected, naming each fixable flag G5 may raise on them: its
# chains' seeds change with the run's id.
REJECTED_MODELS = (
    "a quick pass only, a first look: not_converged, at_bound, thin_layer, alike_layers, "
    "too_deep or underfit left to a full one"
)
# The windows whose curves fail G3, those beside the passing ones after a pick along their
# neighbours: their flags named.
REJECTED_CURVES = (
    "no_ridge, on_data, narrow_span and prominence remain after the line's pick and a guide from "
    "the neighbours"
)
# The curves accepted, then their models inverted and rejected: the 8 from 74.25 m on.
ACCEPTED = "xmid 74.25-84.75 (8)"
# A small context, for the oldest calls to drop out: P2's tools, role and manual take
# about 13,400 tokens of it, past its 60 %.
CONTEXT = 16_000


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


def answer_form(
    text: str, question: str | None = None, reading: str | None = None, tried: str | None = None
) -> Any:  # noqa: ANN401
    return form({"said": text, "question": question, "interpretation": reading, "tried": tried})


# The play, P0 to P5: the objective's form, then the model's calls, then the report's form.
PLAY = [
    form(SCOPE),
    call("survey", {}, 1),
    call("journal", {"action": "write", "kind": "conclusion", "text": "A fan, 120-330 m/s."}, 2),
    # The hypothesis given to the experiment testing it, the conclusion to the commit: each
    # written in the journal with them.
    call(
        "try_settings",
        {
            "variants": [LINE],
            "windows": "spread:3",
            "hypothesis": "Windows of 78 receivers reach deepest.",
            "expect": "more trial windows pass",
        },
        3,
    ),
    call("journal", {"action": "read", "last": 2}, 4),
    call(
        "commit",
        {
            "changes": LINE,
            "reasons": REASONS,
            "experiment": "E1",
            "conclusion": "1 of 3 pass, as now: the deeper windows kept.",
        },
        5,
    ),
    call("observe", {"level": "curves"}, 6),
    call(
        "pick", {"ways": [{"windows": "verdict:retry", "how": "guide", "guide": "neighbours"}]}, 7
    ),
    call("decide", {"windows": "verdict:pass", "status": "accepted"}, 8),
    call(
        "decide",
        {"windows": "verdict:pass", "status": "accepted", "reason": "the near field noted"},
        9,
    ),
    call(
        "decide",
        {"windows": "status:pending", "status": "rejected", "reason": REJECTED_CURVES},
        10,
    ),
    call("freeze", {}, 11),
    call("invert", {"groups": [{"windows": "all"}], "effort": "quick"}, 12),
    call(
        "decide",
        {"windows": "status:pending", "status": "rejected", "reason": REJECTED_MODELS},
        13,
    ),
    said("The 8 curves are frozen and inverted once. " + INTERPRETATION),
    answer_form(
        "The 8 windows' curves are frozen and inverted once.", reading=INTERPRETATION, tried=TRIED
    ),
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
    vision: bool = False,
) -> Played:
    """The scripted model `answers` the user's `messages`, the host between it and the server;
    the model sees with `vision`."""
    model = ScriptedServer(*answers)
    events: list[str] = []

    async def run() -> tuple[Host, str]:
        async with Client(server.server) as client:
            host = await Host.start(
                client,
                OpenAIChat(model.client, "scripted", vision=vision),
                get_settings(),
                limits=Limits(context=context),
                vision=vision,
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
def played(
    input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    searched: None,  # noqa: ARG001
) -> Iterator[Played]:
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
    found = journal.entries(played.study)
    assert [entry.id for entry in found] == ["O1", "C1", "H1", "E1", "C2", "L1", "F1", "I1", "R1"]
    # The experiment rests on the hypothesis it was given, the conclusion on the experiment.
    assert (found[2].expect, found[3].of, found[4].of) == (
        "more trial windows pass",
        ("H1",),
        ("E1",),
    )
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
    # The survey lasts until the model says it in the journal (C1), looking at its records.
    assert "look" in offered[1] and "commit" not in offered[1]
    assert {"commit", "pick", "decide", "freeze"} <= offered[2]
    assert "commit" not in offered[-1]
    assert {"invert", "job_status", "decide"} <= offered[-1]
    assert bodies[0]["messages"][0]["content"].endswith(
        "the first hypotheses to test, a mute on the surface waves' fan among them on a line with "
        "shots. They are where the line's settings start from."
    )
    assert all(body["messages"][-1]["content"].startswith(CARD) for body in bodies)


@pytest.mark.slow
def test_the_report_adds_what_was_tried_and_each_windows_outcome(played: Played) -> None:
    report = played.answer

    assert report.startswith(
        "Scope: invert · active_test_profile · balanced, no target depth, quick effort."
    )
    # What was tried, as the model says it, from the journal's experiments given to it as data.
    assert f"What I tried: {TRIED}" in report
    assert "E1 on spread:3" not in report
    asked = [
        str(message["content"])
        for body in played.model.bodies
        for message in body["messages"]
        if message["role"] == "user" and '<data from="journal">' in str(message["content"])
    ]
    assert asked and "Experiments:\n- E1 on spread:3: current 2/3; 1 " in asked[-1]
    assert f"Each window's outcome:\n- curves accepted: {ACCEPTED}" in report
    assert f"Interpretation, by the model (not a measurement): {INTERPRETATION}" in report
    assert (
        "- MASW windows of 78 receivers (115.5 m), 19 windows, xmid 57.75 to 84.75 m: model: the "
        "deepest windows E1 tried (L1, E1)"
    ) in report
    assert "- windows every receiver (1.5 m): PACo's rule" in report
    # P4: the models of one quick pass, decided and reported.
    assert f"- models rejected: {ACCEPTED}" in report
    assert f"- the models at {ACCEPTED}: {REJECTED_MODELS}" in report
    assert "- inversion of 8 windows, the layers chosen by the data" in report
    assert "- 3000 iterations (750 burn-in), 2 chains: the model's effort" in report


@pytest.mark.slow
def test_what_was_tried_falls_back_on_the_journal_and_its_numbers_are_checked(
    played: Played,
) -> None:
    def written(tried: str | None) -> str:
        form = GeoAnswerForm(said="Done.", question=None, interpretation=None, tried=tried)
        return report(played.study, played.settings, "Scope: invert.", form)

    # Nothing said of it: the journal's lines.
    assert "What I tried:\n- E1 on spread:3: current 2/3; 1 " in written(None)
    # Numbers the journal does not hold, flagged.
    said = written("Windows of 78 receivers passed 33 of 40.")
    assert "What I tried: Windows of 78 receivers passed 33 of 40." in said
    assert "Numbers not found in the tools' results: 33, 40." in said


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
    # A result once read shrinks to its short form: the experiment to its journal line, with
    # each variant's figures, the evidence of what the model concludes from it.
    tool_messages = [one for one in played.host.transcript().messages if one["role"] == "tool"]
    assert all(len(str(one["content"])) < 1200 for one in tool_messages)
    # The experiments' calls are gone after the freeze's fresh start: the model read them before.
    read = [message for body in played.model.bodies for message in body.get("messages", [])]
    tried = next(
        str(one["content"])
        for one in read
        if one.get("role") == "tool"
        and str(one.get("content")).startswith('<data from="try_settings">\nE1 on')
    )
    assert "Their figures:\n- current: " in tried and "uncertainty" in tried


@pytest.mark.slow
def test_the_state_cards_numbers_are_no_unfound_numbers(played: Played) -> None:
    # The card the model read last is stored state: its numbers, said in the answer, are found.
    read = played.host._card  # pyright: ignore[reportPrivateUsage]
    sources = played.host._sources()  # pyright: ignore[reportPrivateUsage]
    assert unfound_numbers(read, [one for one in sources if one != read])
    assert not unfound_numbers(read, sources)


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
            call(
                "journal",
                {"action": "write", "kind": "conclusion", "text": "A fan from 120 to 330 m/s."},
                9,
            ),
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

    refusal = played.model.bodies[6]["messages"][-2]["content"]
    assert refusal == (
        "Not called: the line commits' budget is spent (2 of 2). Pick and decide the windows of "
        "the line as it is."
    )
    assert played.answer.splitlines()[-1].startswith("Where the work stands: Phase: P2")
    assert "Which band should I keep?" in played.answer


@pytest.mark.usefixtures("folders")
def test_a_message_naming_no_profile_there_is_answered_with_the_profiles() -> None:
    model = ScriptedServer(form(BLANK), form({**BLANK, "profile": "no_such_line"}))

    async def run() -> list[str]:
        async with Client(server.server) as client:
            host = await Host.start(
                client, OpenAIChat(model.client, "scripted"), get_settings(), on_event=print
            )
            return [
                await host.answer("Which profiles can I study?"),
                await host.answer("Study no_such_line."),
            ]

    listed, unknown = anyio.run(run)

    assert listed.startswith(
        "Which profile should I study? The profiles:\nactive_test_profile: a profile to process"
    )
    assert unknown.startswith("There is no profile 'no_such_line'. Which profile should I study?")
    # The objective's forms alone: no call to the model, no study.
    assert len(model.bodies) == 2
    assert not (get_settings().output_dir / "no_such_line").exists()


def test_the_survey_lasts_until_its_conclusion_is_written(tmp_path: Path) -> None:
    study = Study(PROFILE, tmp_path)
    journal.objective(study, {}, "balanced, no target depth")
    phases = [phase_of(study)]
    (tmp_path / SAVED).write_text("{}")
    phases.append(phase_of(study))
    journal.write(study, "conclusion", "A clean fan from 120 to 330 m/s; a mute of 90 to 1,400.")
    phases.append(phase_of(study))

    # The curves' tools open once the model has said what the survey's records show.
    assert phases == ["P1", "P1", "P2"]


@pytest.mark.usefixtures("folders")
def test_a_model_with_vision_sees_the_atlas_after_the_request() -> None:
    seen = play(
        [form(SCOPE), call("survey", {}, 1), said("Surveyed."), answer_form("Surveyed.")],
        [REQUEST],
        vision=True,
    )

    first = next(body for body in seen.model.bodies if "tools" in body)
    request = first["messages"][1]["content"]
    # The request, then each sheet's caption as data and its figure; the conversation the host
    # keeps holds the request alone.
    assert request[0] == {"type": "text", "text": REQUEST}
    assert [part["type"] for part in request[1:]] == ["text", "image_url"] * 3
    assert request[1]["text"].startswith('<data from="atlas">\nPACo\'s atlas, shot gathers')
    assert seen.host.history[0] == {"role": "user", "content": REQUEST}


@pytest.mark.usefixtures("folders")
def test_the_model_calls_budget_stops_the_run_and_a_new_message_starts_afresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The quick effort's run: two model calls.
    quick = budgets.EFFORTS["quick"].model_copy(update={"model_calls": 2})
    monkeypatch.setitem(budgets.EFFORTS, "quick", quick)
    first = play([form(SCOPE), *SURVEYED, said("Surveyed.")], [REQUEST])

    again = play(
        [form(SCOPE), said("Going on."), answer_form("Going on.", question="Which mute?")],
        ["Go on with active_test_profile."],
    )

    # Stopped after its two calls; the next message a fresh run of the same study.
    assert sum(1 for body in first.model.bodies if "tools" in body) == 2
    assert "The run's budget of model calls is spent (2 of 2): it stops here." in first.answer
    assert again.study == first.study
    assert sum(1 for body in again.model.bodies if "tools" in body) == 1
    assert "Which mute?" in again.answer


@pytest.mark.usefixtures("folders")
def test_the_tokens_budget_counts_the_conversation_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    first = play([form(SCOPE), said("Hello."), answer_form("Hello.", question="Go on?")], [REQUEST])
    # Another conversation's judging, past the quick effort's tokens: not this run's.
    budgets.record(
        first.study,
        budgets.Use(
            kind="form",
            phase="P2",
            at=journal.datetime.now(journal.UTC),
            seconds=0,
            name="judge",
            prompt_tokens=1_900_000,
            completion_tokens=150_000,
            conversation="another",
        ),
    )
    quick = budgets.EFFORTS["quick"].model_copy(update={"tokens": 1_500})
    monkeypatch.setitem(budgets.EFFORTS, "quick", quick)

    again = play([form(SCOPE), *SURVEYED], ["Go on with active_test_profile."])

    # Its objective's form and one call (1,880 tokens): past its 1,500, it stops.
    assert sum(1 for body in again.model.bodies if "tools" in body) == 1
    assert "The run's budget of tokens is spent" in again.answer


@pytest.mark.slow
@pytest.mark.usefixtures("folders", "searched")
def test_a_run_stopped_in_p2_resumes_with_the_same_phase_budgets_and_card() -> None:
    stopped = play(
        [
            form(SCOPE),
            call("survey", {}, 1),
            call(
                "journal",
                {"action": "write", "kind": "conclusion", "text": "A fan from 120 to 330 m/s."},
                9,
            ),
            call("commit", {"changes": LINE, "reasons": REASONS}, 2),
            500,
        ],
        [REQUEST],
        raises=openai.InternalServerError,
    )
    study, settings = stopped.study, stopped.settings
    # A new conversation goes on with the study's budgets, its model calls and tokens its own:
    # none yet.
    before = (phase_of(study), budgets.spent(study, "quick"), card(study, settings, "new"))

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
    first = play(
        [form(typed), said("Go on?"), answer_form("Asked.", question="Go on?")],
        ["Process active_test_profile with windows of 24 receivers, velocities up to 350 m/s."],
    )

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
@pytest.mark.usefixtures("folders", "searched")
def test_a_report_with_models_not_decided_sends_the_model_back_once() -> None:
    # The play to the first inversion; then a report before any model is decided.
    early = [said("The 8 windows are inverted."), answer_form("The 8 windows are inverted.")]
    played = play([*PLAY[:13], *early, *PLAY[13:]], [REQUEST])

    sent_back = [
        message["content"]
        for body in played.model.bodies
        for message in body["messages"]
        if message["role"] == "user" and str(message["content"]).startswith("[PACo] The run is")
    ]
    assert sent_back[0] == PENDING.format(count=8, where=ACCEPTED)
    # Then the models decided, and the report.
    assert phase_of(played.study) == "P5"
    assert f"- models rejected: {ACCEPTED}" in played.answer


def test_a_study_whose_run_is_gone_is_not_resumed(paco_env: Settings) -> None:
    # Its run deleted outside PACo: nothing to go on from. A new study, and why, said once.
    folder = paco_env.output_dir / "active_test_profile" / STUDIES / "20200101-000000-0000"
    folder.mkdir(parents=True)
    line = Line(run_id="20200101-000001-0000", settings={}, commits=())
    (folder / LINE_FILE).write_text(line.model_dump_json())

    study, said = study_for("active_test_profile", paco_env)

    assert study.folder != folder
    assert said == (
        "Study 20200101-000000-0000 cannot go on: its run 20200101-000001-0000 is gone. A new "
        "study."
    )
    again, said_again = study_for("active_test_profile", paco_env)
    assert (again.folder, said_again) == (study.folder, None)


def test_a_new_study_when_the_user_asks_to_start_over(paco_env: Settings) -> None:
    folder = paco_env.output_dir / "active_test_profile" / STUDIES / "20200101-000000-0000"
    folder.mkdir(parents=True)
    study = Study("active_test_profile", folder)
    journal.objective(study, {}, "balanced, no target depth, normal effort")

    resumed, _ = study_for("active_test_profile", paco_env)
    fresh, said = study_for("active_test_profile", paco_env, fresh=True)

    # Asked to start over: a new study, the one there kept as it is; the newest the next one
    # a conversation resumes.
    assert resumed.folder == folder
    assert fresh.folder != folder and folder.is_dir()
    assert said == "A new study, as asked; study 20200101-000000-0000 kept as it is."
    assert study_for("active_test_profile", paco_env)[0].folder == fresh.folder


def test_a_follow_up_after_the_report_opens_the_study_again(paco_env: Settings) -> None:
    folder = paco_env.output_dir / "active_test_profile" / STUDIES / "20200101-000000-0000"
    folder.mkdir(parents=True)
    study = Study("active_test_profile", folder)
    journal.objective(study, {}, "balanced, no target depth, normal effort")
    (folder / SAVED).write_text("{}")
    (folder / CURVES).write_text("{}")
    journal.report(study, "The report.")
    assert phase_of(study) == "P5"

    # Where a message takes the study: the curves, the inversion, or nowhere (a question).
    def asked(**fields: Any) -> GeoScope:  # noqa: ANN401
        return GeoScope.model_validate({**BLANK, **fields})

    assert wanted(asked(process=True), Given()) == "P2"
    assert wanted(asked(), Given(mode="passive-active")) == "P2"
    assert wanted(asked(invert=True), Given()) == "P4"
    assert wanted(asked(), Given(inversion={"n_iterations": 50_000})) == "P4"
    assert wanted(asked(), Given()) is None
    # To the inversion: the frozen curves kept.
    assert (
        reopen(study, "P4", "now invert it").text == "Back to the inversion after R1: now invert it"
    )
    assert phase_of(study) == "P4" and (folder / CURVES).exists()
    # Reported again, then to the curves: the freeze undone, its curves set aside.
    journal.report(study, "The second report.")
    assert reopen(study, "P2", "reprocess using muting").id == "B2"
    assert phase_of(study) == "P2"
    assert not (folder / CURVES).exists() and (folder / "curves-B2.json").exists()


@pytest.mark.slow
def test_a_follow_up_asking_to_process_again_reopens_the_reported_study(played: Played) -> None:
    # The played study, copied, its report given: a message asks to process again. The study
    # goes back to its curves, its journal kept, and the model works on: here it asks which mute.
    folder = played.study.folder.parent / "20990101-000000-0000"
    shutil.copytree(played.study.folder, folder)
    study = Study(PROFILE, folder)
    model = ScriptedServer(
        form({**BLANK, "process": True, "redo": True}),
        said("Which mute?"),
        answer_form("Before processing again:", question="Which mute: by velocity, or by offset?"),
    )
    events: list[str] = []

    async def run() -> str:
        async with Client(server.server) as client:
            host = await Host.start(
                client,
                OpenAIChat(model.client, "scripted"),
                played.settings,
                on_event=events.append,
            )
            host.study = study
            return await host.answer("can you reprocess using muting")

    answered = anyio.run(run)

    assert [entry.id for entry in journal.entries(study)][-2:] == ["R1", "B1"]
    assert phase_of(study) == "P2" and (folder / "curves-B1.json").exists()
    assert "Which mute: by velocity, or by offset?" in answered
    assert "B1: Back to the curves after R1: can you reprocess using muting" in events
    # The model read the request, why the study opened again, then the state card.
    read = model.bodies[1]["messages"]
    assert read[1]["content"].startswith(
        "can you reprocess using muting\n\n[PACo] Study 20990101-000000-0000 opens again after "
        "its report (B1: Back to the curves after R1"
    )
    assert read[-1]["content"].startswith(CARD)


# The survey, said: the curves' tools open.
SURVEYED = [
    call("survey", {}, 1),
    call("journal", {"action": "write", "kind": "conclusion", "text": "A fan, 120-330 m/s."}, 2),
]
# 14 variants: 12 of 5 receivers (round 1's sheet 1), 2 of 9 (its sheet 2); each picks to its
# own fmax, a trial in seconds.
ROWS = [
    searching.Row(
        label=f"g{n}",
        values={"length": "5" if n <= 12 else "9", "muting": "ab"[n % 2]},
        changes={"picking": {"fmax": 60.0 + n}},
        trial=f"t{n}",
        length=5 if n <= 12 else 9,
        card={"image": 1.0, "covered": 3, "windows": 3},
    )
    for n in range(1, 15)
]


def canned(windows: str, *, thumbs: bool = False) -> Callable[..., Any]:
    """A search that runs no grid: ROWS its variants on `windows`, sorted into sheets, with
    `thumbs` each variant's thumbnails at the 4 places."""

    def search(study: Study, **_: Any) -> tuple[searching.Search, searching.Knockout]:  # noqa: ANN401
        made = searching.Search(
            id=searching.SEARCH,
            windows=windows,
            axes={"length": ["5", "9"], "muting": ["a", "b"]},
            units={"length": "receivers", "muting": "name"},
            changes={},
            rows=tuple(ROWS),
            refused=0,
            refusal=None,
            places=(15.0, 50.0, 90.0, 125.0),
            seconds=60.0,
            made_at=datetime.now(UTC),
        )
        folder = study.folder / searching.SEARCHES
        (folder / searching.IMAGES).mkdir(parents=True)
        (folder / f"{made.id}.json").write_text(made.model_dump_json())
        knockout = searching.Knockout(
            round=1, sheets=searching.first_sheets(ROWS, ["length", "muting"])
        )
        (folder / searching.KNOCKOUT).write_text(knockout.model_dump_json())
        image = np.tile(np.linspace(0, 255, grid.THUMB, dtype=np.uint8)[:, None], grid.THUMB)
        pick = (np.array([10.0, 20.0], np.float32), np.array([300.0, 250.0], np.float32))
        for row in ROWS if thumbs else []:
            grid.write_thumbs(
                folder / searching.IMAGES / f"{row.label}.npz",
                [
                    grid.Thumb(x, image, (0.0, 100.0, 1.0, 1000.0), 0.33, pick, (3.0, 54.0))
                    for x in made.places
                ],
            )
        return made, knockout

    return search


def kept(*sheets: tuple[int, list[str]]) -> Any:  # noqa: ANN401
    """The judge's form: the variants kept of each sheet."""
    return form(
        {"why": "One ridge at every place.", "sheets": [{"sheet": n, "kept": k} for n, k in sheets]}
    )


def judged(played: Played) -> list[dict[str, Any]]:
    """The judge's requests."""
    judge = prompts.prompt("judge")
    return [body for body in played.model.bodies if body["messages"][0]["content"] == judge]


@pytest.mark.usefixtures("folders")
def test_the_sheets_are_judged_in_calls_of_their_own_until_refused_three_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(searching, "search", canned("spread:3"))
    wrong = kept((2, ["g1"]))
    played = play(
        [
            form(SCOPE),
            *SURVEYED,
            call("search", {}, 3),
            # The judge's calls: a variant not on the sheet, then sheet 1 judged; then sheet 2
            # judged wrong three times.
            kept((1, ["g99"])),
            kept((1, ["g1", "g2"])),
            wrong,
            wrong,
            wrong,
            said("The judging stopped: shall I search again?"),
            answer_form("The judging stopped.", question="Shall I search again?"),
        ],
        [REQUEST],
    )

    asked = judged(played)
    first, again, second = (one["messages"][-1]["content"][0]["text"] for one in asked[:3])
    read = next(
        message["content"]
        for message in played.model.bodies[-2]["messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "call_3"
    )
    records = (played.study.folder / searching.SEARCHES / searching.JUDGED).read_text()
    used = budgets.uses(played.study)
    steps = [step for step in played.host.transcript().host_steps if step.name == "judge"]
    # Each judge's call, a form, reads its prompt, the line's facts and the sheets alone: no
    # conversation, no state card, no tools.
    assert len(asked) == 5 and all(len(one["messages"]) == 2 for one in asked)
    assert all("tools" not in one and "response_format" in one for one in asked)
    assert first.startswith(
        "Line active_test_profile, active: each window stacks its shots' images. The objective: "
    )
    assert "Round 1, sheet 1 of 2 (keep 1 or 2):" in first and "sheet 2 of 2 (keep" in first
    assert "S1: the full grid" not in first
    # A refused answer said again with the same sheets; the next call, the sheet left.
    assert "Your last answer was not recorded: " in again and "no variant g99 on it" in again
    assert "Round 1, sheet 2 of 2 (keep 1 or 2):" in second and "sheet 1 of 2" not in second
    # The model reads the search's words and why the judging stopped, never the sheets, and is
    # not offered the judge.
    assert read.startswith('<data from="search">\nS1: the full grid, 14 variants')
    assert "The sheets could not be judged (" in read and "search again judges those left" in read
    assert "Round 1, sheet" not in read
    assert "judge" not in {tool["function"]["name"] for tool in played.model.bodies[3]["tools"]}
    # Sheet 1's judgment recorded, once; the judge's forms out of the model calls' budget, their
    # tokens in the transcript.
    assert [json.loads(line)["kept"] for line in records.splitlines()] == [["g1", "g2"]]
    assert sum(1 for one in used if one.kind == "form" and one.name == "judge") == 5
    assert budgets.spent(played.study, "quick").model_calls == 4
    assert [step.is_error for step in steps] == [True, False, True, True, True]
    assert "   Round 1, judged by eye: sheet 1: g1, g2." in played.events


@pytest.mark.slow
@pytest.mark.usefixtures("folders")
def test_the_judged_finalists_come_to_the_model_tried_side_by_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(searching, "search", canned("spread:3", thumbs=True))
    played = play(
        [
            form(SCOPE),
            *SURVEYED,
            call("search", {}, 3),
            kept((1, ["g1", "g2"]), (2, ["g13"])),
            kept((1, ["g2", "g13"])),
            said("Which of the two?"),
            answer_form("The finalists are tried.", question="Which of the two?"),
        ],
        [REQUEST],
        vision=True,
    )

    first, last = (one["messages"] for one in judged(played))
    messages = played.model.bodies[-2]["messages"]
    at = next(i for i, one in enumerate(messages) if one.get("tool_call_id") == "call_3")
    knockout = searching.knockout_of(played.study)
    # Each call sees the atlas's dispersion images, in a turn of their own, then its sheets: round
    # 1's two, then round 2's last.
    assert first[1]["content"][0]["text"].startswith(
        '<data from="atlas">\nPACo\'s atlas, dispersion images'
    )
    assert [part["type"] for part in first[1]["content"]] == ["text", "image_url"]
    assert first[2]["role"] == "assistant" and first[1:3] == last[1:3]
    assert [part["type"] for part in first[3]["content"]] == ["text", "image_url", "image_url"]
    assert [part["type"] for part in last[3]["content"]] == ["text", "image_url"]
    assert "Round 2, sheet 1 of 1 (the last: keep 1 to 3)" in last[3]["content"][0]["text"]
    assert knockout is not None and knockout.finalists == ("g2", "g13")
    assert knockout.experiment == "E1"
    # The model reads the search's words, then the finalists tried side by side, with figures.
    read = messages[at]["content"]
    assert "S1: the full grid, 14 variants" in read
    assert "The sheets judged by eye in 2 short calls. Every sheet is judged. The finalists" in read
    assert messages[at + 1]["role"] == "user" and messages[at + 1]["content"][1]["type"] == (
        "image_url"
    )
    assert "   Round 1, judged by eye: sheet 1: g1, g2; sheet 2: g13." in played.events
    assert "   Round 2, judged by eye: sheet 1: g2, g13." in played.events
