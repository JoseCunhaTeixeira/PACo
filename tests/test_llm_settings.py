"""One model setup for both agents (paco.llm.settings): the model, PACO_LLM_*, any provider, which
the guided agent's settings and the geophysicist's read alike; its context, set or assumed; the
evaluations' judge, PACO_JUDGE_*; and the agent a chat runs, PACO_AGENT_MODE."""

import pytest
from pydantic import ValidationError

from paco.agent.settings import AgentSettings
from paco.llm.settings import (
    FRONTIER_CONTEXT,
    LOCAL_CONTEXT,
    PROVIDERS,
    JudgeSettings,
    ModelSettings,
    Provider,
    missing,
    provider_info,
)
from paco.modes import ModeSettings


def test_both_agents_read_the_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", "Qwen/Qwen3-14B-FP8")

    model = ModelSettings()  # pyright: ignore[reportCallIssue]  # fields come from the environment
    guided = AgentSettings()  # pyright: ignore[reportCallIssue]

    assert model.model_dump() == guided.model_dump(include=set(ModelSettings.model_fields))
    # vLLM's by default, as the guided agent was first served, and a small model's context.
    assert (guided.llm_provider, guided.context()) == ("vllm", LOCAL_CONTEXT)


def test_the_model_settings_and_their_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://gpu.test:8001/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", "Qwen/Qwen3.8-27B")
    for name in ("API_KEY", "CONTEXT", "VISION", "THINKING"):
        monkeypatch.delenv(f"PACO_LLM_{name}", raising=False)

    settings = ModelSettings()  # pyright: ignore[reportCallIssue]  # fields come from the environment

    assert (settings.llm_base_url, settings.llm_model) == (
        "http://gpu.test:8001/v1",
        "Qwen/Qwen3.8-27B",
    )
    # Figures reach the model only when allowed; thinking on.
    assert (settings.llm_vision, settings.llm_thinking) == (False, True)
    assert settings.llm_api_key.get_secret_value() == "EMPTY"


@pytest.mark.parametrize(
    ("provider", "context"),
    [
        ("vllm", LOCAL_CONTEXT),
        ("openai-compatible", LOCAL_CONTEXT),
        ("anthropic", FRONTIER_CONTEXT),
    ],
)
def test_the_context_assumed_follows_the_provider(provider: Provider, context: int) -> None:
    settings = ModelSettings(llm_provider=provider, llm_model="a-model", llm_base_url="http://x/v1")

    assert settings.context() == context
    assert settings.model_copy(update={"llm_context": 65_536}).context() == 65_536


@pytest.mark.parametrize("provider", ["anthropic", "openai", "mistral"])
def test_an_api_with_its_own_address_needs_none(provider: Provider) -> None:
    settings = ModelSettings(llm_provider=provider, llm_model="a-model")

    assert settings.llm_base_url is None
    assert settings.llm_effort is None


@pytest.mark.parametrize("provider", ["vllm", "openai-compatible"])
def test_a_model_served_elsewhere_needs_its_address(provider: Provider) -> None:
    with pytest.raises(ValidationError) as refused:
        ModelSettings(llm_provider=provider, llm_model="a-model")

    assert missing(refused.value) == "PACO_LLM_BASE_URL"


def test_the_judge_is_none_unless_named(monkeypatch: pytest.MonkeyPatch) -> None:
    assert JudgeSettings().judge_model is None

    monkeypatch.setenv("PACO_JUDGE_MODEL", "claude-opus-5-5")
    with pytest.raises(ValidationError) as refused:
        JudgeSettings()
    assert missing(refused.value) == "PACO_JUDGE_BASE_URL"
    # The guided agent's settings check it too, beside the model's own address.
    monkeypatch.setenv("PACO_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("PACO_LLM_MODEL", "claude-sonnet-5-5")
    with pytest.raises(ValidationError) as refused:
        AgentSettings()  # pyright: ignore[reportCallIssue]
    assert missing(refused.value) == "PACO_JUDGE_BASE_URL"

    monkeypatch.setenv("PACO_JUDGE_PROVIDER", "anthropic")
    judge = JudgeSettings()
    assert (judge.judge_provider, judge.judge_model) == ("anthropic", "claude-opus-5-5")


def test_the_agent_mode_is_guided_unless_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PACO_AGENT_MODE", raising=False)
    assert ModeSettings().agent_mode == "guided"

    monkeypatch.setenv("PACO_AGENT_MODE", "autonomous")
    assert ModeSettings().agent_mode == "autonomous"
    monkeypatch.setenv("PACO_AGENT_MODE", "free")
    with pytest.raises(ValidationError):
        ModeSettings()


def test_each_provider_says_what_it_needs() -> None:
    assert len(PROVIDERS) == 11 and PROVIDERS[0].name == "anthropic"
    vertex, bedrock, served = (
        provider_info("anthropic-vertex"),
        provider_info("anthropic-bedrock"),
        provider_info("vllm"),
    )

    # Vertex AI: the machine's Google credentials, a region and a project; no key.
    assert (vertex.key, vertex.region, vertex.project) == ("none", True, True)
    # Bedrock: a Bedrock API key, or else AWS's credentials; a region.
    assert (bedrock.key, bedrock.region, bedrock.project) == ("optional", True, False)
    # A server of the user's: its address needed, a small local model's context assumed.
    assert (served.address, served.key, served.context) == ("needed", "optional", LOCAL_CONTEXT)
    assert provider_info("openai").address == "optional"
    assert provider_info("anthropic").context == FRONTIER_CONTEXT


def test_an_empty_setting_is_one_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # As compose passes a variable its .env leaves out, and a settings page saves one over .env.
    for name, value in (
        ("PROVIDER", "anthropic"),
        ("MODEL", "claude-opus-5-5"),
        ("BASE_URL", ""),
        ("CONTEXT", ""),
        ("API_KEY", ""),
        ("EFFORT", ""),
        ("VISION", ""),
        ("THINKING", ""),
    ):
        monkeypatch.setenv(f"PACO_LLM_{name}", value)
    for name in ("PROVIDER", "MODEL", "API_KEY"):
        monkeypatch.setenv(f"PACO_JUDGE_{name}", "")
    monkeypatch.setenv("PACO_AGENT_MODE", "")

    settings = AgentSettings()  # pyright: ignore[reportCallIssue]

    assert (settings.llm_base_url, settings.llm_context, settings.llm_effort) == (None, None, None)
    assert (settings.llm_vision, settings.llm_thinking) == (False, True)
    assert settings.context() == FRONTIER_CONTEXT
    assert settings.llm_api_key.get_secret_value() == "EMPTY"
    assert (settings.judge_provider, settings.judge_model) == ("vllm", None)
    assert ModeSettings().agent_mode == "guided"
    # An empty provider is vLLM's, which needs its address.
    monkeypatch.setenv("PACO_LLM_PROVIDER", "")
    with pytest.raises(ValidationError) as refused:
        ModelSettings()  # pyright: ignore[reportCallIssue]
    assert missing(refused.value) == "PACO_LLM_BASE_URL"
