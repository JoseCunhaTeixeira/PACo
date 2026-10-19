"""The model PACo's two modes talk to, PACO_LLM_*, from the environment or a .env file: one they
hold an API key for (Claude, OpenAI's, Gemini, Mistral), or one a server serves (theirs, with vLLM,
or any server or gateway speaking OpenAI's chat API, a company's cloud reached through one). The
evaluations' judge is the same model."""

from typing import Literal

from pydantic import Field, SecretStr, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Who serves the model, and so how the client talks to it (paco.llm.chat): Claude through
# Anthropic's API; the others through OpenAI's chat API, as OpenAI, Google (Gemini), Mistral or a
# server (vLLM, Ollama, LM Studio, OpenRouter, a gateway) serves it.
type Provider = Literal["anthropic", "openai", "gemini", "mistral", "server"]
# A server's address has no default, its key is optional, and its model is a small one. The
# context assumed when the settings give none: on a server, the one PAC's and PACo's compose files
# serve vLLM with; on a provider's API, a frontier model's.
LOCAL = ("server",)
LOCAL_CONTEXT = 16_384
FRONTIER_CONTEXT = 131_072
# The key of a server started without one, which takes any.
NO_KEY = "EMPTY"
# What a command says when the model's settings are missing.
NO_MODEL = "Set {missing} in .env (the model: its server and name)."


class ModelSettings(BaseSettings):
    """The model PACo's two modes talk to, PACO_LLM_*."""

    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        # .env also holds the server's and the agents' other settings (paco.settings...).
        extra="ignore",
        frozen=True,
    )

    # A server by default: the user's GPU (Qwen3 served by vLLM, as PACo's workflow was first served).
    llm_provider: Provider = "server"
    # The model's API: a provider's has its own address; a server needs it.
    llm_base_url: str | None = Field(
        default=None, validate_default=True, description="e.g. http://gpu-host:8001/v1"
    )
    # No default, so that a missing setting fails here, with its name.
    llm_model: str = Field(min_length=1, description="The name the API serves the model under.")
    # The user's API key; a server started without a key accepts any.
    llm_api_key: SecretStr = SecretStr(NO_KEY)
    # The model's context, in tokens: the one the server serves (vLLM's --max-model-len, set from
    # it in PAC's and PACo's compose files). Unset: LOCAL_CONTEXT or FRONTIER_CONTEXT (context()).
    # The autonomous mode needs 128k; each mode keeps its conversation within it.
    llm_context: int | None = Field(default=None, ge=1)
    # Whether figures may reach the model: on unless set off. An online model's provider receives
    # what the model sees (the data policy): PACO_LLM_VISION=false keeps them on the machine.
    llm_vision: bool = True

    @field_validator(
        "llm_provider",
        "llm_base_url",
        "llm_api_key",
        "llm_context",
        "llm_vision",
        mode="before",
    )
    @classmethod
    def _empty_is_default(cls, value: object, info: ValidationInfo) -> object:
        """An empty setting is one not set, its default: compose passes one for a variable its
        .env leaves out, and a settings page saves one over a value of .env."""
        if value != "" or info.field_name is None:
            return value
        return cls.model_fields[info.field_name].get_default(call_default_factory=True)

    @field_validator("llm_base_url")
    @classmethod
    def _served_somewhere(cls, value: str | None, info: ValidationInfo) -> str | None:
        provider = info.data.get("llm_provider")
        if value is None and provider in LOCAL:
            raise ValueError(f"the address of the model's API is needed with provider {provider}")
        return value

    def context(self) -> int:
        """The model's context: the one set, else a small local model's or a frontier one's."""
        if self.llm_context is not None:
            return self.llm_context
        return LOCAL_CONTEXT if self.llm_provider in LOCAL else FRONTIER_CONTEXT


def missing(error: ValidationError, prefix: str = "PACO_") -> str:
    """The settings a ValidationError names, as the environment spells them."""
    return ", ".join(f"{prefix}{problem['loc'][0]}".upper() for problem in error.errors())
