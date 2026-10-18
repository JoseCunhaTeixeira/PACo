"""paco-ab: both agents' servers loaded in one process, as the A/B runner loads them, each
one's tools and the scenarios listed; without a model's settings, the evaluation says which to set
(tests/test_ab.py plays it)."""

from datetime import UTC, datetime
from importlib.metadata import version

import anyio
import pytest

from paco import server as workflow_server
from paco.autonomous import pins
from paco.autonomous import server as autonomous_server
from paco.autonomous.evaluation import cli
from paco.autonomous.evaluation.running import ABReport, Models
from paco.autonomous.evaluation.scenarios import PAIRS, TRUTH_PAIRS
from paco.autonomous.host.loop import short_context
from paco.llm.claude import AnthropicChat
from paco.llm.settings import LOCAL_CONTEXT, NO_MODEL


def test_the_evaluation_lists_each_agents_tools_and_the_scenarios(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["--list"])

    sigpipe = pins.commit("sigpipe")
    assert sigpipe is not None
    upstream = f"PACo's quality control {pins.paco_code()[:7]}, sigpipe {sigpipe[:7]}"
    pacos = [f"  {tool.name}" for tool in anyio.run(workflow_server.server.list_tools)]
    ours = [f"  {tool.name}" for tool in anyio.run(autonomous_server.server.list_tools)]
    pairs = (*PAIRS, *TRUTH_PAIRS)
    assert capsys.readouterr().out.splitlines() == [
        f"The workflow agent as it is ({upstream}): {len(pacos)} tools",
        *pacos,
        f"The autonomous agent (paco {version('paco')}, on {upstream}): {len(ours)} tools",
        *ours,
        f"Scenarios, both agents on the same messages: {len(pairs)}",
        *(f"  {pair.name} ({pair.profile}): {pair.questions[0]}" for pair in pairs),
    ]


@pytest.mark.usefixtures("paco_env")
def test_without_the_models_settings_the_evaluation_says_which_to_set(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["zero_settings"])
    assert stopped.value.code == 1
    assert capsys.readouterr().out.splitlines() == [
        NO_MODEL.format(missing="PACO_LLM_BASE_URL, PACO_LLM_MODEL")
    ]

    # A model on a server of the user's, its context unset: a small local model's, too short.
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", "scripted")
    with pytest.raises(SystemExit):
        cli.main(["zero_settings"])
    assert capsys.readouterr().out.splitlines() == [short_context(LOCAL_CONTEXT)]


@pytest.mark.usefixtures("paco_env")
def test_an_unknown_scenario_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["shallower"])

    assert stopped.value.code == 2
    assert "Unknown scenario(s): shallower." in capsys.readouterr().err


@pytest.mark.usefixtures("paco_env")
def test_both_modes_and_the_judge_play_on_the_model_of_env(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACO_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("PACO_LLM_MODEL", "claude-sonnet-5-5")
    played: list[Models] = []

    async def run_ab(_pairs: object, models: Models, *_: object, **__: object) -> ABReport:
        played.append(models)
        return ABReport(
            eval_id="ab-20261017-120000-abcd",
            started_at=datetime(2026, 10, 17, tzinfo=UTC),
            workflow_model=models.workflow_name,
            autonomous_model=models.autonomous_name,
            judge_model=models.judge_name,
            workflow_prompts="workflow",
            autonomous_prompts="autonomous",
            repeat=1,
            inversions="scenario",
            questions={},
            profiles={},
            plays=(),
        )

    monkeypatch.setattr(cli, "run_ab", run_ab)

    cli.main(["zero_settings"])

    (models,) = played
    # Both modes and the judge on the same Claude, each its own client.
    assert isinstance(models.workflow, AnthropicChat) and isinstance(
        models.autonomous, AnthropicChat
    )
    assert models.workflow is not models.autonomous
    assert (models.workflow_name, models.autonomous_name) == (
        "claude-sonnet-5-5",
        "claude-sonnet-5-5",
    )
    assert isinstance(models.judge, AnthropicChat) and models.judge_name == "claude-sonnet-5-5"
    assert models.judge is not models.autonomous
    assert "A/B evaluation ab-20261017-120000-abcd" in capsys.readouterr().out
