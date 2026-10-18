"""paco-ab: both agents' servers loaded in one process, as the A/B runner loads them, each
one's tools and the scenarios listed; without a model's settings, the evaluation says which to set
(tests/test_ab.py plays it)."""

from datetime import UTC, datetime
from importlib.metadata import version

import anyio
import pytest

from paco import server as paco_server
from paco.geo import pins
from paco.geo import server as geo_server
from paco.geo.evaluation import cli
from paco.geo.evaluation.running import ABReport, Models
from paco.geo.evaluation.scenarios import PAIRS, TRUTH_PAIRS
from paco.geo.host.loop import short_context
from paco.llm.claude import AnthropicChat
from paco.llm.settings import LOCAL_CONTEXT, NO_MODEL


def test_the_evaluation_lists_each_agents_tools_and_the_scenarios(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["--list"])

    sigpipe = pins.commit("sigpipe")
    assert sigpipe is not None
    upstream = f"PACo's quality control {pins.paco_code()[:7]}, sigpipe {sigpipe[:7]}"
    pacos = [f"  {tool.name}" for tool in anyio.run(paco_server.server.list_tools)]
    ours = [f"  {tool.name}" for tool in anyio.run(geo_server.server.list_tools)]
    pairs = (*PAIRS, *TRUTH_PAIRS)
    assert capsys.readouterr().out.splitlines() == [
        f"PACo's guided agent as it is ({upstream}): {len(pacos)} tools",
        *pacos,
        f"The autonomous geophysicist (paco {version('paco')}, on {upstream}): {len(ours)} tools",
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
def test_both_agents_play_on_the_model_of_env(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PACO_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("PACO_LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("PACO_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("PACO_JUDGE_MODEL", "claude-opus-5-5")
    played: list[Models] = []

    async def run_ab(_pairs: object, models: Models, *_: object, **__: object) -> ABReport:
        played.append(models)
        return ABReport(
            eval_id="ab-20261017-120000-abcd",
            started_at=datetime(2026, 10, 17, tzinfo=UTC),
            paco_model=models.paco_name,
            geo_model=models.geo_name,
            judge_model=models.judge_name,
            paco_prompts="paco",
            geo_prompts="geo",
            repeat=1,
            inversions="scenario",
            questions={},
            profiles={},
            plays=(),
        )

    monkeypatch.setattr(cli, "run_ab", run_ab)

    cli.main(["zero_settings"])

    (models,) = played
    # Both agents on the same Claude, each its own client; the judge another.
    assert isinstance(models.paco, AnthropicChat) and isinstance(models.geo, AnthropicChat)
    assert models.paco is not models.geo
    assert (models.paco_name, models.geo_name) == ("claude-sonnet-5-5", "claude-sonnet-5-5")
    assert isinstance(models.judge, AnthropicChat) and models.judge_name == "claude-opus-5-5"
    assert "A/B evaluation ab-20261017-120000-abcd" in capsys.readouterr().out
