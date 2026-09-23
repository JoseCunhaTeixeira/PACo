import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.agent import Reply, Step, ToolCall, ToolStep, Transcript
from paco.evaluation import (
    SCENARIOS,
    CheckResult,
    EvaluationReport,
    JudgeScore,
    ScenarioResult,
    Trial,
    cli,
    format_report,
    read_score,
    render,
    run_evaluation,
    run_scenario,
)
from paco.evaluation.checks import (
    answer_mentions,
    asked_the_user,
    at_most_calls,
    called,
    good_windows,
    in_order,
    inversion_succeeded,
    never_called,
    no_inversion_started,
    not_succeeded,
    only_called,
    succeeded,
)
from paco.inversion import InversionParameters, InversionRecord, JobState, WindowInversion
from paco.picking import PickingParameters
from paco.quality import ImageQuality, QualityParameters, RunQuality, WindowQuality
from paco.settings import Settings

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}

type Policy = Callable[[list[ChatCompletionMessageParam]], Reply]
type WindowStatus = Literal["succeeded", "failed"]


class PolicyModel:
    """Stands in for Qwen: decides each reply from the conversation so far."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],  # noqa: ARG002
    ) -> Reply:
        return self._policy(messages)


def _calls(name: str, arguments: dict[str, Any]) -> Reply:
    return Reply(content="", tool_calls=(ToolCall("call_0", name, json.dumps(arguments)),))


def _says(text: str) -> Reply:
    return Reply(content=text, tool_calls=())


def _results(messages: list[ChatCompletionMessageParam]) -> list[str]:
    return [str(message["content"]) for message in messages if message["role"] == "tool"]


def _step(
    name: str, arguments: dict[str, Any], is_error: bool = False, called: bool = True
) -> ToolStep:
    return ToolStep(
        name=name,
        arguments=json.dumps(arguments),
        called=called,
        is_error=is_error,
        duration_s=0.1,
        result="{}",
    )


def _trial(
    steps: list[Step],
    answer: str = "",
    output_dir: Path = Path("/nowhere"),
    asked: tuple[str, ...] = (),
) -> Trial:
    transcript = Transcript(
        started_at=datetime(2026, 9, 23, tzinfo=UTC),
        model="qwen",
        messages=[{"role": "user", "content": "?"}, {"role": "assistant", "content": answer}],
        steps=steps,
    )
    return Trial(transcript, output_dir, asked)


# ---------------------------------------------------------------- rule checks


@pytest.mark.parametrize(
    ("steps", "passed", "detail"),
    [
        ([_step("inspect_profile", {"profile": "active_p1"})], True, ""),
        (
            [_step("inspect_profile", {"profile": "passive_p1"})],
            False,
            'called with {"profile": "passive_p1"}',
        ),
        ([_step("list_profiles", {})], False, "inspect_profile was never called"),
        # A call the host refused (invalid arguments) does not count.
        (
            [_step("inspect_profile", {"profile": "active_p1"}, called=False)],
            False,
            "inspect_profile was never called",
        ),
    ],
)
def test_called(steps: list[Step], passed: bool, detail: str) -> None:
    result = called("inspect_profile", profile="active_p1")(_trial(steps))

    assert result == CheckResult(
        name='called inspect_profile(profile="active_p1")', passed=passed, detail=detail
    )


def test_called_matches_nested_arguments_that_hold_more() -> None:
    step = _step(
        "run_processing",
        {
            "profile": "active_p1",
            "overrides": {"masw": {"length": 24, "step": 24, "distance_max": 30}},
        },
    )

    assert called("run_processing", overrides=SMALL_WINDOWS)(_trial([step])).passed
    assert not called("run_processing", overrides={"masw": {"length": 12}})(_trial([step])).passed


def test_called_reads_objects_sent_as_json_text() -> None:
    # Qwen3-4B sometimes sends an object as a string: the SDK decodes it for the server.
    step = _step("run_processing", {"profile": "active_p1", "overrides": json.dumps(SMALL_WINDOWS)})

    assert called("run_processing", overrides=SMALL_WINDOWS)(_trial([step])).passed
    assert not called("run_processing", overrides=SMALL_WINDOWS)(
        _trial([_step("run_processing", {"overrides": "{masw"})])
    ).passed


def test_only_called() -> None:
    kept = _step("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    refused = _step("run_processing", {"overrides": {"masw": {"lenght": 24}}}, is_error=True)
    # Qwen3-4B's second run, on the quality advice, instead of the windows the user asked for.
    changed = _step(
        "run_processing", {"profile": "active_p1", "overrides": {"masw": {"length": 48}}}
    )
    check = only_called("run_processing", overrides=SMALL_WINDOWS)

    # Only what the server ran counts.
    assert check(_trial([refused, kept])).passed
    assert check(_trial([kept, changed])) == CheckResult(
        name='only run_processing(overrides={"masw": {"length": 24, "step": 24}})',
        passed=False,
        detail='called with {"profile": "active_p1", "overrides": {"masw": {"length": 48}}}',
    )
    assert check(_trial([refused])).detail == "run_processing never succeeded"


def test_succeeded_and_not_succeeded() -> None:
    failed_then_passed = _trial([_step("pick", {}, is_error=True), _step("pick", {})])
    only_failed = _trial([_step("pick", {}, is_error=True)])

    assert succeeded("pick")(failed_then_passed).passed
    assert not succeeded("pick")(only_failed).passed
    assert not not_succeeded("pick")(failed_then_passed).passed
    assert not_succeeded("pick")(only_failed).passed


def test_never_called() -> None:
    # Qwen3-4B asked to invert a run nobody wanted inverted; the user declined.
    declined = _trial([_step("pick", {}), _step("invert", {}, is_error=True)])
    refused = _trial([_step("invert", {}, called=False)])  # never reached the server

    assert never_called("invert")(declined) == CheckResult(
        name="invert never called", passed=False, detail="called 1 time(s)"
    )
    assert never_called("invert")(refused).passed


def test_in_order() -> None:
    steps: list[Step] = [
        _step("dispersion_quality", {}, is_error=True),  # a failed call does not count
        _step("run_processing", {}),
        _step("dispersion_quality", {}),
    ]

    assert in_order("run_processing", "dispersion_quality")(_trial(steps)).passed
    assert not in_order("dispersion_quality", "run_processing")(_trial(steps)).passed
    assert not in_order("run_processing", "pick")(_trial(steps)).passed


def test_at_most_calls_counts_every_call() -> None:
    steps: list[Step] = [_step("list_profiles", {}), _step("list_profiles", {}, called=False)]

    assert at_most_calls(2)(_trial(steps)).passed
    assert at_most_calls(1)(_trial(steps)) == CheckResult(
        name="at most 1 tool calls", passed=False, detail="2 calls"
    )


@pytest.mark.parametrize(
    ("answer", "facts", "passed"),
    [
        ("There are 96 receivers, 0.25 m apart.", ("96", "0.25"), True),
        ("24 windows", ("4",), False),  # 4 is not in 24
        ("10.25 m", ("0.25",), False),
        ("4 windows are good.", ("4",), True),
        ("Profiles: Active_P1 and passive_p1.", ("active_p1", "passive_p1"), True),
    ],
)
def test_answer_mentions(answer: str, facts: tuple[str, ...], passed: bool) -> None:
    assert answer_mentions(*facts)(_trial([], answer)).passed == passed


def test_the_last_answer_counts() -> None:
    transcript = Transcript(
        started_at=datetime(2026, 9, 23, tzinfo=UTC),
        model="qwen",
        messages=[
            {"role": "user", "content": "?"},
            {"role": "assistant", "content": "First, 3.", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "{}"},
            {"role": "assistant", "content": "Finally, 4."},
        ],
        steps=[],
    )

    assert transcript.answer == "Finally, 4."


def test_facts_read_from_disk(tmp_path: Path) -> None:
    run = tmp_path / "active_p1" / "20260923-100000-abcd"
    run.mkdir(parents=True)

    def window(xmid: float, verdict: str) -> WindowQuality:
        quality = ImageQuality.model_validate(
            {
                "verdict": verdict,
                "flags": () if verdict == "good" else ("sharpness",),
                "n_points": 9,
                "band_hz": None,
            }
        )
        return WindowQuality(xmid=xmid, folder=f"xmid_{xmid:.2f}", quality=quality)

    record = RunQuality(
        run_id=run.name,
        picking=PickingParameters(),
        thresholds=QualityParameters(),
        windows=(window(1.0, "good"), window(2.0, "doubtful"), window(3.0, "good")),
    )
    (run / "quality.json").write_text(record.model_dump_json())
    trial = _trial([], "2 windows are good.", tmp_path)

    assert good_windows(trial) == "2"
    assert answer_mentions(good_windows)(trial).passed
    assert no_inversion_started()(trial).passed
    (run / "inversion.json").write_text("{}")
    assert not no_inversion_started()(trial).passed


def test_inversion_succeeded(tmp_path: Path) -> None:
    run = tmp_path / "active_p1" / "20260923-100000-abcd"
    run.mkdir(parents=True)
    trial = _trial([], "", tmp_path)

    def record(statuses: tuple[WindowStatus, ...], state: JobState) -> str:
        windows = tuple(
            WindowInversion(xmid=float(xmid), folder=f"xmid_{xmid}.00", status=status)
            for xmid, status in enumerate(statuses)
        )
        return InversionRecord(
            job_id="inv-20260923-100000-abcd",
            run_id=run.name,
            parameters=InversionParameters(),
            state=state,
            submitted_at=datetime(2026, 9, 23, tzinfo=UTC),
            total=2,
            windows=windows,
        ).model_dump_json()

    assert inversion_succeeded()(trial).detail == "no inversion on disk"
    (run / "inversion.json").write_text(record(("succeeded", "succeeded"), "succeeded"))
    assert inversion_succeeded()(trial).passed
    # Qwen3-4B's jobs before the burn-in fix: started, then failed in every window.
    (run / "inversion.json").write_text(record(("failed", "failed"), "failed"))
    assert inversion_succeeded()(trial).detail == (
        "inv-20260923-100000-abcd failed, 2 of 2 windows failed"
    )


def test_asked_the_user() -> None:
    assert asked_the_user()(_trial([], asked=("Approve?",))).passed
    assert not asked_the_user()(_trial([])).passed


# ---------------------------------------------------------------- the judge


@pytest.mark.parametrize(
    ("text", "score"),
    [
        (
            '{"score": 4, "reason": "Correct, but vague."}',
            JudgeScore(score=4, reason="Correct, but vague."),
        ),
        (
            'Here it is:\n```json\n{"score": 2, "reason": "Invented."}\n```',
            JudgeScore(score=2, reason="Invented."),
        ),
        (
            "I think it is good.",
            JudgeScore(score=None, reason="unreadable judgement: I think it is good."),
        ),
        ('{"score": 9}', JudgeScore(score=None, reason='unreadable judgement: {"score": 9}')),
    ],
)
def test_read_score(text: str, score: JudgeScore) -> None:
    assert read_score(text) == score


def test_render_shows_the_judge_the_whole_conversation() -> None:
    transcript = Transcript(
        started_at=datetime(2026, 9, 23, tzinfo=UTC),
        model="qwen",
        messages=[
            {"role": "system", "content": "You are PACo's assistant."},
            {"role": "user", "content": "Profiles?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "list_profiles", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "x" * 400},
            {"role": "assistant", "content": "Two profiles."},
        ],
        steps=[],
    )

    assert render(transcript) == (
        "USER: Profiles?\nTOOL CALL: list_profiles {}\nTOOL RESULT: " + "x" * 300 + "...\n"
        "ASSISTANT: Two profiles."
    )


# ---------------------------------------------------------------- playing scenarios


def _scenario(name: str) -> Any:  # noqa: ANN401
    (scenario,) = [scenario for scenario in SCENARIOS if scenario.name == name]
    return scenario


def _play(name: str, policy: Policy, folder: Path, judge_says: str | None = None) -> ScenarioResult:
    judge = PolicyModel(lambda _messages: _says(judge_says)) if judge_says else None

    async def play() -> ScenarioResult:
        return await run_scenario(
            _scenario(name), PolicyModel(policy), "scripted", folder, judge, on_event=lambda _: None
        )

    return anyio.run(play)


def lists_profiles(messages: list[ChatCompletionMessageParam]) -> Reply:
    results = _results(messages)
    if not results:
        return _calls("list_profiles", {})
    return _says("You can process " + " and ".join(json.loads(results[0])["result"]) + ".")


def invents_an_answer(_messages: list[ChatCompletionMessageParam]) -> Reply:
    return _says("active_p2 has 48 receivers, 0.5 m apart.")


def goes_up_to_the_inversion(messages: list[ChatCompletionMessageParam]) -> Reply:
    results = _results(messages)
    if not results:
        return _calls("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    run_id = json.loads(results[0])["run_id"]
    steps = ["dispersion_quality", "pick", "invert"]
    if len(results) <= len(steps):
        return _calls(steps[len(results) - 1], {"run_id": run_id})
    return _says("You declined the inversion, so none started. What should I change?")


@pytest.mark.usefixtures("paco_env")
def test_a_good_policy_passes_and_is_judged(tmp_path: Path) -> None:
    result = _play("list_profiles", lists_profiles, tmp_path, '{"score": 5, "reason": "Exact."}')

    assert result.passed
    assert result.answer == "You can process active_p1 and passive_p1."
    assert (result.tool_calls, result.failed_calls) == (1, 0)
    assert result.judge == JudgeScore(score=5, reason="Exact.")
    saved = Transcript.model_validate_json((tmp_path / "transcript.json").read_text())
    assert saved.answer == result.answer
    assert [step.kind for step in saved.steps] == ["model", "tool", "model"]


@pytest.mark.usefixtures("paco_env")
def test_an_invented_answer_fails(tmp_path: Path) -> None:
    result = _play("unknown_profile", invents_an_answer, tmp_path)

    assert not result.passed
    assert [check.passed for check in result.checks] == [False, False, True]
    assert result.judge is None


def test_the_simulated_user_declines_and_nothing_starts(paco_env: Settings, tmp_path: Path) -> None:
    result = _play("inversion_declined", goes_up_to_the_inversion, tmp_path)

    assert [(check.name, check.passed) for check in result.checks] == [
        (
            'only run_processing(profile="active_p1", overrides={"masw": {"length": 24, '
            '"step": 24}})',
            True,
        ),
        ("called invert()", True),
        ("the user was asked to approve", True),
        ("no inversion started", True),
    ]
    assert result.failed_calls == 1  # invert, declined
    # The server wrote into the scenario's folder, and got its settings back afterwards.
    assert list((tmp_path / "outputs" / "active_p1").iterdir())
    assert Settings().output_dir == paco_env.output_dir


@pytest.mark.usefixtures("paco_env")
def test_an_evaluation_writes_its_report_and_transcripts(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"  # does not exist yet

    async def evaluate() -> EvaluationReport:
        return await run_evaluation(
            [_scenario("list_profiles")],
            PolicyModel(lists_profiles),
            "scripted",
            root,
            on_event=lambda _: None,
        )

    report = anyio.run(evaluate)

    folder = root / report.eval_id
    assert EvaluationReport.model_validate_json((folder / "report.json").read_text()) == report
    assert (folder / "list_profiles" / "transcript.json").exists()
    assert report.judge_model is None


@pytest.mark.usefixtures("paco_env")
def test_an_evaluation_repeats_each_scenario(tmp_path: Path) -> None:
    async def evaluate() -> EvaluationReport:
        return await run_evaluation(
            [_scenario("list_profiles"), _scenario("unknown_profile")],
            PolicyModel(lists_profiles),
            "scripted",
            tmp_path,
            repeat=2,
            on_event=lambda _: None,
        )

    report = anyio.run(evaluate)

    assert report.repeat == 2
    assert [(result.name, result.attempt) for result in report.results] == [
        ("list_profiles", 1),
        ("list_profiles", 2),
        ("unknown_profile", 1),
        ("unknown_profile", 2),
    ]
    # Each play has its own folder.
    folder = tmp_path / report.eval_id
    assert (folder / "list_profiles" / "1" / "transcript.json").exists()
    assert (folder / "unknown_profile" / "2" / "transcript.json").exists()


def test_the_evaluation_refuses_zero_plays(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["paco-evaluate", "--repeat", "0"])

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 2
    assert "--repeat must be at least 1" in capsys.readouterr().err


# ---------------------------------------------------------------- the report


def test_the_report_table() -> None:
    def result(name: str, passed: bool, score: int | None) -> ScenarioResult:
        return ScenarioResult(
            name=name,
            kind="look around",
            checks=(
                CheckResult(
                    name="answer mentions 96", passed=passed, detail="" if passed else "missing 96"
                ),
            ),
            judge=JudgeScore(score=score, reason="") if score else None,
            tool_calls=2,
            failed_calls=0,
            max_prompt_tokens=1_612,
            duration_s=3.21,
            answer="",
        )

    report = EvaluationReport(
        eval_id="eval-20260923-100000-abcd",
        model="Qwen/Qwen3-8B",
        judge_model="judge",
        started_at=datetime(2026, 9, 23, tzinfo=UTC),
        results=(result("list_profiles", True, 5), result("describe_profile", False, 2)),
    )

    text = format_report(report)

    assert text.startswith("Evaluation eval-20260923-100000-abcd of Qwen/Qwen3-8B (judge: judge)")
    assert (
        "list_profiles        look around          1/1      5     2      0      1,612    3.2s"
        in text
    )
    assert text.endswith(
        "Passed 1 of 2 scenarios.\nMean judge score: 3.5 of 5.\nFailed checks:\n"
        "  describe_profile: answer mentions 96 (missing 96)"
    )


def test_the_report_table_with_repeats() -> None:
    def play(name: str, attempt: int, passed: bool) -> ScenarioResult:
        return ScenarioResult(
            name=name,
            kind="approval",
            attempt=attempt,
            checks=(CheckResult(name="invert succeeded", passed=passed, detail="-"),),
            judge=None,
            tool_calls=5,
            failed_calls=1,
            max_prompt_tokens=2_444,
            duration_s=30.4,
            answer="",
        )

    report = EvaluationReport(
        eval_id="eval-20260923-100000-abcd",
        model="Qwen/Qwen3-4B",
        judge_model=None,
        started_at=datetime(2026, 9, 23, tzinfo=UTC),
        repeat=2,
        results=(
            play("list_profiles", 1, True),
            play("list_profiles", 2, True),
            play("inversion_approved", 1, True),
            play("inversion_approved", 2, False),
        ),
    )

    text = format_report(report)

    assert text.startswith(
        "Evaluation eval-20260923-100000-abcd of Qwen/Qwen3-4B (judge: none), "
        "2 plays of each scenario"
    )
    # The column widens for the longest label.
    assert (
        "inversion_approved #2 approval             0/1      -     5      1      2,444   30.4s"
        in text
    )
    assert text.endswith(
        "Passed 3 of 4 plays:\n"
        "  list_profiles         2/2\n"
        "  inversion_approved    1/2\n"
        "Failed checks:\n"
        "  inversion_approved #2: invert succeeded (-)"
    )
