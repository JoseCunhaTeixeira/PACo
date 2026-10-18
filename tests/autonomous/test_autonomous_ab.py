"""The A/B evaluation (docs/autonomous_plan.md §13 M9): the lines both agents play, the scenarios' pairs,
the known-truth set's curves and scores, and a known-truth scenario played by both agents through
the runner, its report comparing them. The models are stand-ins: the workflow agent's decides each reply
from the conversation, as PACo's tests' PolicyModel; the autonomous agent's is scripted
(tests/scripted.py), as in tests/test_host.py."""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anyio
import numpy as np
import obspy
import pytest
from autonomous_judged import JudgedLine
from autonomous_scripted import ScriptedServer, usage
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from sigpipe.masw.picks import CURVES_FILE

from paco.agent import Filled, Reply, ToolCall, Transcript
from paco.agent.answer import SCHEMA as ANSWER_SCHEMA
from paco.autonomous.evaluation import known_truth
from paco.autonomous.evaluation.inputs import build_inputs, offered
from paco.autonomous.evaluation.report import format_ab
from paco.autonomous.evaluation.running import VARIANTS, ABReport, Models, run_ab
from paco.autonomous.evaluation.scenarios import PAIRS, TRUTH_PAIRS, pairs_named
from paco.autonomous.host.scope import BLANK
from paco.autonomous.tools import models
from paco.autonomous.tools.trials import Study
from paco.evaluation.scenarios import SCENARIOS
from paco.llm.model import OpenAIChat
from paco.qc.judging import saved_m0
from paco.settings import Settings, get_settings

PROFILE = "active_test_profile"
TRUTH = next(pair for pair in TRUTH_PAIRS if pair.name == "truth_two_layers")
RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
JUDGED = '{"score": 4, "reason": "Sound, and faithful to the numbers."}'
# The autonomous agent's models, rejected where they did not pass: every fixable flag named.
REJECTED = (
    "a quick pass only, none retried: not_converged, at_bound, underfit, alike_layers, "
    "thin_layer and too_deep are left for a full pass"
)
INTERPRETATION = "A stiffer layer below some 5 m, if the curves are right."


# ---------------------------------------------------------------- the lines


def test_the_lines_are_active_test_profile_and_its_defect_copies(
    input_dir: Path, tmp_path: Path
) -> None:
    inputs = build_inputs(input_dir, tmp_path / "inputs")

    assert offered(input_dir) == {"active_test_profile", "active_dead", "active_refracted"}
    assert sorted(path.name for path in inputs.iterdir()) == [
        "active_dead",
        "active_refracted",
        "active_test_profile",
    ]
    # The real line linked, not copied; active_test_profile only where the input folder holds it.
    assert all(path.is_symlink() for path in (inputs / PROFILE).iterdir())
    # Its first record's trace 40 zeroed.
    dead = obspy.read(str(inputs / "active_dead" / "101.mseed"))
    assert not np.any(dead[40].data)
    assert np.any(dead[39].data)
    made = (inputs / "active_refracted" / "101.mseed").stat().st_mtime_ns
    build_inputs(input_dir, tmp_path / "inputs")
    assert (inputs / "active_refracted" / "101.mseed").stat().st_mtime_ns == made


# ---------------------------------------------------------------- the scenarios


def test_both_agents_play_the_same_messages_with_their_own_checks() -> None:
    pacos = {scenario.name: scenario for scenario in SCENARIOS}
    # PACo's own scenario, its message on active_test_profile: its checks name no line.
    zero = next(pair for pair in PAIRS if pair.name == "zero_settings")
    assert zero.workflow.checks == pacos["zero_settings"].checks
    assert zero.workflow.questions == ("Process active_test_profile and give me the Vs models.",)
    for pair in (*PAIRS, *TRUTH_PAIRS):
        (question,) = pair.questions
        # Every message asks for the models, and none a window step (§8, rule 3).
        assert "Vs models" in question or "nvert" in question
        assert "every" not in question
        assert pair.checks
        assert pair.workflow.checks
    # The known-truth set: PACo's curves_invert on active_test_profile, its models at PAC's effort.
    assert TRUTH.questions == TRUTH.workflow.questions == ("Invert active_test_profile.",)
    assert TRUTH.workflow.checks == pacos["curves_invert"].checks
    assert TRUTH.workflow.full_inversion


def test_the_scenarios_are_chosen_among_the_lines_there_are() -> None:
    lines = frozenset({PROFILE, "active_dead", "active_refracted"})

    every = pairs_named([], lines)
    assert len(every) == len(PAIRS) + len(TRUTH_PAIRS)
    # Without the defect copies, their scenarios left out.
    assert {pair.name for pair in pairs_named([], frozenset({PROFILE}))}.isdisjoint(
        {"dead_trace", "refractions"}
    )
    assert [pair.name for pair in pairs_named(["deeper", "dead_trace"], lines)] == [
        "dead_trace",
        "deeper",
    ]
    with pytest.raises(ValueError, match="Unknown scenario"):
        pairs_named(["shallower"], lines)


# ---------------------------------------------------------------- the known truth


@pytest.mark.slow
def test_a_truths_curve_stands_on_a_picked_window(judged_line: JudgedLine) -> None:
    # A window whose pick passed G3, as the known-truth set takes them: 17.25's, 22 to 29 Hz.
    xmid = min(x for x, g3 in judged_line.g3.items() if g3.verdict == "pass")
    picked = saved_m0(judged_line.folder / f"xmid_{xmid:.2f}" / CURVES_FILE)
    assert picked is not None
    truth = known_truth.TRUTHS[0]

    curve = truth.curve(picked, seed=7)

    assert np.array_equal(curve.fs, picked.fs)
    assert curve.mode == picked.mode
    assert curve.vs_err is not None
    assert np.all(curve.vs_err / curve.vs <= 0.41)
    # Its noise is seeded: the same window and truth, the same curve.
    assert np.array_equal(truth.curve(picked, seed=7).vs, curve.vs)
    assert not np.array_equal(truth.curve(picked, seed=8).vs, curve.vs)
    # The fundamental mode of 200 over 350 m/s: slow at the high frequencies, faster below.
    assert curve.vs[np.argmax(curve.fs)] < curve.vs[np.argmin(curve.fs)]
    assert truth.vs_at(np.array([0.5, 4.99, 5.0, 30.0])).tolist() == [200, 200, 350, 350]


# ---------------------------------------------------------------- a play of both agents


class WorkflowPolicy:
    """the workflow agent's model, a stand-in: inverts the run PACo's host settled (its id in the note
    after the message), then says so."""

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        called = [
            call["function"]["name"]
            for message in cast(list[dict[str, Any]], messages)
            if message["role"] == "assistant"
            for call in message.get("tool_calls") or ()
        ]
        if "invert" in called:
            return Reply(content="The 8 windows are inverted.", tool_calls=())
        note = next(str(m["content"]) for m in reversed(messages) if m["role"] == "user")
        found = RUN_ID.search(note)
        assert found is not None, note
        arguments = json.dumps({"run_id": found.group(0)})
        return Reply(content="", tool_calls=(ToolCall("call_1", "invert", arguments),))

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        schema: dict[str, Any],
    ) -> Filled:
        if schema == ANSWER_SCHEMA:
            return Filled(
                content=json.dumps({"said": "The 8 windows are inverted.", "question": None})
            )
        scope = {
            **dict.fromkeys(("process", "pick", "soils", "redo", "replace_hand_work"), False),
            **dict.fromkeys(("run_id", "length_receivers", "length_m", "step_receivers"), None),
            **dict.fromkeys(("step_m", "option", "workers", "mode"), None),
            "positions_m": [],
            "compare_lengths_receivers": [],
            "compare_lengths_m": [],
            "invert": True,
            "profile": PROFILE,
        }
        return Filled(content=json.dumps(scope))


class Judge:
    """The judge, a stand-in: the same grade for everything."""

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        return Reply(content=JUDGED, tool_calls=())

    async def fill(
        self,
        messages: list[ChatCompletionMessageParam],  # noqa: ARG002
        schema: dict[str, Any],  # noqa: ARG002
    ) -> Filled:
        raise AssertionError("the judge fills no form")


def _call(
    name: str, arguments: dict[str, Any], number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _said(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return {"role": "assistant", "content": text}, usage(1000, 50)


# The autonomous agent from the freeze on, without advise: the objective's form, advise refused,
# one quick pass, the models passing accepted and the others rejected, the report.
AUTONOMOUS_PLAY = [
    _said(json.dumps({**BLANK, "invert": True, "profile": PROFILE})),
    _call("advise", {"topic": "inversion"}, 1),
    _call("invert", {"groups": [{"windows": "all"}], "effort": "quick"}, 2),
    _call(
        "decide",
        {"windows": "model:pass", "status": "accepted", "reason": "fits its curve: notes kept"},
        3,
    ),
    _call("decide", {"windows": "status:pending", "status": "rejected", "reason": REJECTED}, 4),
    _said("The models are decided. " + INTERPRETATION),
    _said(
        json.dumps(
            {
                "said": "The 8 windows' models are decided.",
                "question": None,
                "interpretation": INTERPRETATION,
                "tried": None,
            }
        )
    ),
]


@pytest.fixture(scope="module")
def evaluated(
    input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[ABReport, Path]]:
    root = tmp_path_factory.mktemp("ab")
    autonomous = ScriptedServer(*AUTONOMOUS_PLAY)
    played = Models(
        autonomous=OpenAIChat(autonomous.client, "scripted"),
        autonomous_name="scripted",
        workflow=WorkflowPolicy(),
        workflow_name="policy",
        judge=Judge(),
        judge_name="judge",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.setenv("PACO_INPUT_DIR", str(input_dir))
        get_settings.cache_clear()
        report = anyio.run(
            lambda: run_ab(
                [TRUTH],
                played,
                root,
                input_dir,
                variants=[VARIANTS["no-advise"]],
                inversions="quick",
                on_event=lambda _: None,
            )
        )
    get_settings.cache_clear()
    yield report, root / report.eval_id


@pytest.mark.slow
def test_both_agents_play_a_known_truth_scenario(evaluated: tuple[ABReport, Path]) -> None:
    report, folder = evaluated

    assert ABReport.model_validate_json((folder / "report.json").read_text()) == report
    workflow, autonomous = report.plays
    assert (workflow.label, autonomous.label) == ("workflow", "autonomous no-advise")
    assert [(check.name, check.passed) for check in workflow.checks] == [
        ("called invert()", True),
        ("the inversion gave every window a model", True),
        ("invert with no parameters", True),
        ("nothing redone: picking", True),
        ("nothing redone: preprocessing, phase_shift", True),
        ("the agent asked nothing", True),
        ("at most 5 tool calls", True),
    ]
    assert [(check.name, check.passed) for check in autonomous.checks] == [
        ("the report given", True),
        ("every model decided", True),
        ("the thresholds stayed the configuration's", True),
        ("the agent asked nothing", True),
    ]
    assert workflow.judge is not None and workflow.judge.score == 4
    assert autonomous.judge is not None and autonomous.judge.score == 4
    assert autonomous.reasoning is not None and autonomous.reasoning.score == 4
    assert workflow.reasoning is None


@pytest.mark.slow
def test_each_play_is_measured_from_what_it_left(evaluated: tuple[ABReport, Path]) -> None:
    report, folder = evaluated
    workflow, autonomous = report.plays

    for play in (workflow, autonomous):
        # The line's 19 windows: the 8 whose pick passed G3 and G4 frozen with the truth's curves,
        # each inverted; the others rejected (known_truth.REJECTED).
        assert (play.measures.windows, play.measures.accepted, play.measures.models) == (19, 8, 8)
        # The workers' time counted: the inversion ran on 8 of them.
        assert play.measures.cpu_s > play.measures.wall_s > 0
        assert play.measures.depth_informed_m is not None
    # The autonomous agent's own work alone: the line was made before its play.
    assert (
        autonomous.measures.experiments,
        autonomous.measures.commits,
        autonomous.measures.refused,
    ) == (0, 0, 1)
    assert autonomous.measures.model_calls == 7
    assert (workflow.measures.experiments, workflow.measures.commits) == (None, 0)
    # Its transcript's answer is the report the user read.
    transcript = Transcript.model_validate_json(
        (folder / autonomous.folder / "transcript.json").read_text()
    )
    assert transcript.answer == autonomous.answer
    assert autonomous.answer.startswith("Scope: invert · active_test_profile")
    assert (
        f"Interpretation, by the model (not a measurement): {INTERPRETATION}" in autonomous.answer
    )


@pytest.mark.slow
def test_the_kept_models_are_scored_against_the_truth(evaluated: tuple[ABReport, Path]) -> None:
    report, folder = evaluated
    workflow, autonomous = report.plays

    outputs = folder / autonomous.folder / "outputs"
    (run_folder,) = (path.parent for path in outputs.glob("*/*/run.json"))
    (study_folder,) = (outputs / PROFILE / "studies").iterdir()
    study = Study(PROFILE, study_folder)
    accepted = [
        unit
        for unit, status in models.statuses(study, run_folder, run_folder.name).items()
        if status == "accepted"
    ]
    assert autonomous.truth is not None
    assert autonomous.truth.windows == len(accepted)
    assert workflow.truth is not None
    workflow_run = next((folder / workflow.folder / "outputs").glob("*/*/run.json")).parent
    assert workflow.truth.windows == len(known_truth.kept(workflow_run, None))
    # Every window inverted, scored: the median Vs's error and the truth's share within P10-P90.
    assert TRUTH.truth is not None
    every = known_truth.score(TRUTH.truth, run_folder, ["xmid_74.25", "xmid_75.75"])
    assert every.windows == 2
    assert every.error is not None and 0 <= every.error < 1
    assert every.in_band is not None and 0 <= every.in_band <= 1


@pytest.mark.slow
def test_the_report_compares_the_agents(evaluated: tuple[ABReport, Path]) -> None:
    report, _ = evaluated

    text = format_ab(report, prices=(0.5, 2.0, 0.05))

    assert text.startswith(
        f"A/B evaluation {report.eval_id}: the workflow agent (policy, {report.workflow_prompts}) against "
        f"the autonomous agent (scripted, {report.autonomous_prompts}), 1 play of each; judge: judge;"
    )
    lines = text.splitlines()
    assert 'truth_two_layers, on active_test_profile: "Invert active_test_profile."' in lines
    header = next(line for line in lines if line.strip().startswith("workflow"))
    assert header.split() == ["workflow", "autonomous", "no-advise"]
    rows = {line[:30].strip(): line[30:].split() for line in lines if line.startswith("  ")}
    assert rows["plays passed"] == ["1/1", "1/1"]
    assert rows["judge: the reasoning (1-5)"] == ["-", "4.0"]
    assert rows["windows accepted"] == ["8/19", "8/19"]
    # advise refused, the variant without it; the accept of the models G5 passed fails when the
    # quick pass passed none (its seeds come from the run's id).
    assert rows["calls refused/failed"][1] in ("1/0", "1/1")
    assert "money ($)" in rows
    assert "models scored" in rows
    assert any(line.startswith("At the same CPU budget") for line in lines)
    assert any(line.startswith("  truth_two_layers: PACo ") for line in lines)
