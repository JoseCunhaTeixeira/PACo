"""One model for both of PACo's modes (paco.llm.settings): PACO_LLM_*, which the workflow's agent
settings and the autonomous mode read alike; its context, set or assumed; an empty setting its
default; and the mode a chat runs, PACO_AGENT_MODE."""

import pytest
from pydantic import ValidationError

from paco.agent.settings import AgentSettings
from paco.llm.settings import FRONTIER_CONTEXT, LOCAL_CONTEXT, ModelSettings, Provider, missing
from paco.modes import ModeSettings


def test_both_modes_read_the_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", "Qwen/Qwen3-14B-FP8")

    model = ModelSettings()  # pyright: ignore[reportCallIssue]  # fields come from the environment
    agent = AgentSettings()  # pyright: ignore[reportCallIssue]

    assert model.model_dump() == agent.model_dump(include=set(ModelSettings.model_fields))
    # A server by default, your GPU's, and the context compose serves vLLM with.
    assert (agent.llm_provider, agent.context()) == ("server", LOCAL_CONTEXT) == ("server", 16_384)
    assert (agent.llm_vision, agent.llm_api_key.get_secret_value()) == (False, "EMPTY")


@pytest.mark.parametrize(
    ("provider", "context"),
    [
        ("server", LOCAL_CONTEXT),
        ("anthropic", FRONTIER_CONTEXT),
        ("mistral", FRONTIER_CONTEXT),
    ],
)
def test_the_context_assumed_follows_the_provider(provider: Provider, context: int) -> None:
    settings = ModelSettings(llm_provider=provider, llm_model="a-model", llm_base_url="http://x/v1")

    assert settings.context() == context
    assert settings.model_copy(update={"llm_context": 65_536}).context() == 65_536


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini", "mistral"])
def test_an_api_with_its_own_address_needs_none(provider: Provider) -> None:
    assert ModelSettings(llm_provider=provider, llm_model="a-model").llm_base_url is None


def test_a_server_needs_its_address() -> None:
    with pytest.raises(ValidationError) as refused:
        ModelSettings(llm_provider="server", llm_model="a-model")

    assert missing(refused.value) == "PACO_LLM_BASE_URL"


def test_an_empty_setting_is_one_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # As compose passes a variable its .env leaves out, and a settings page saves one over .env.
    for name, value in (
        ("PROVIDER", "anthropic"),
        ("MODEL", "claude-opus-5-5"),
        ("BASE_URL", ""),
        ("CONTEXT", ""),
        ("API_KEY", ""),
        ("VISION", ""),
    ):
        monkeypatch.setenv(f"PACO_LLM_{name}", value)
    monkeypatch.setenv("PACO_AGENT_MODE", "")

    settings = AgentSettings()  # pyright: ignore[reportCallIssue]

    assert (settings.llm_base_url, settings.llm_context, settings.llm_vision) == (None, None, False)
    assert settings.context() == FRONTIER_CONTEXT
    assert settings.llm_api_key.get_secret_value() == "EMPTY"
    assert ModeSettings().agent_mode == "workflow"
    # An empty provider is a server, which needs its address.
    monkeypatch.setenv("PACO_LLM_PROVIDER", "")
    with pytest.raises(ValidationError) as refused:
        ModelSettings()  # pyright: ignore[reportCallIssue]
    assert missing(refused.value) == "PACO_LLM_BASE_URL"


def test_the_mode_is_the_workflow_unless_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PACO_AGENT_MODE", raising=False)
    assert ModeSettings().agent_mode == "workflow"

    monkeypatch.setenv("PACO_AGENT_MODE", "autonomous")
    assert ModeSettings().agent_mode == "autonomous"
    monkeypatch.setenv("PACO_AGENT_MODE", "guided")
    with pytest.raises(ValidationError):
        ModeSettings()
