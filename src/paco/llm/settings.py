"""The model both of PACo's agents talk to, PACO_LLM_*, and the evaluations' judge, PACO_JUDGE_*,
from the environment or a .env file: any model the user holds an API key or a cloud account for.
Claude through Anthropic's API, Google Cloud's Vertex AI, Amazon Bedrock or Microsoft Foundry;
OpenAI's models through OpenAI or Azure; Gemini through Google's API or Vertex AI; Mistral's; a vLLM
server, or any other server speaking OpenAI's chat API."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Who serves the model, and so how the client talks to it (paco.llm.chat): Claude through
# Anthropic's API or a cloud's (Vertex AI, Bedrock, Foundry); the others through OpenAI's chat API,
# as OpenAI, Azure, Google (Gemini, on its API or Vertex AI), Mistral, vLLM or another server
# serves it.
type Provider = Literal[
    "anthropic",
    "anthropic-vertex",
    "anthropic-bedrock",
    "anthropic-foundry",
    "openai",
    "azure-openai",
    "gemini",
    "gemini-vertex",
    "mistral",
    "vllm",
    "openai-compatible",
]
# How hard the model thinks, where its API takes it: Anthropic's effort, OpenAI's reasoning effort.
type Effort = Literal["low", "medium", "high", "xhigh", "max"]
# The providers whose API is wherever the user serves it or their cloud resource is: their address
# has no default (Azure's and Foundry's: the resource's endpoint).
SERVED = ("vllm", "openai-compatible", "azure-openai", "anthropic-foundry")
# Those reached in a cloud region (Vertex AI's location, Bedrock's AWS region), and in a Google
# Cloud project.
REGIONAL = ("anthropic-vertex", "anthropic-bedrock", "gemini-vertex")
IN_PROJECT = ("anthropic-vertex", "gemini-vertex")


# The servers of the user's own, whose model is a small local one unless they say otherwise, and
# the contexts assumed when the settings give none.
LOCAL = ("vllm", "openai-compatible")
LOCAL_CONTEXT = 12_288
FRONTIER_CONTEXT = 131_072
# The key of a server started without one, which takes any.
NO_KEY = "EMPTY"
# What a command says when the model's settings are missing.
NO_MODEL = "Set {missing} in .env (the model: its server and name)."
# What each provider is called, as a settings page offers them (PAC's).
LABELS: dict[Provider, str] = {
    "anthropic": "Anthropic (Claude)",
    "anthropic-vertex": "Claude on Google Cloud Vertex AI",
    "anthropic-bedrock": "Claude on Amazon Bedrock",
    "anthropic-foundry": "Claude on Microsoft Foundry",
    "openai": "OpenAI",
    "azure-openai": "Azure OpenAI",
    "gemini": "Google Gemini",
    "gemini-vertex": "Gemini on Google Cloud Vertex AI",
    "mistral": "Mistral",
    "vllm": "vLLM server (your own GPU)",
    "openai-compatible": "Another OpenAI-compatible server",
}
# The providers reached with the machine's cloud credentials rather than an API key (Vertex AI:
# Google's Application Default Credentials), and those whose key may be left out (Bedrock: AWS's
# credentials then; a server of the user's started without one).
CLOUD_CREDENTIALS = ("anthropic-vertex", "gemini-vertex")
KEY_OPTIONAL = ("anthropic-bedrock", "vllm", "openai-compatible")


class ProviderInfo(BaseModel):
    """What a provider needs, as a settings page asks for it: its address (needed, or the
    provider's own unless given), its key (needed, optional, or none: the cloud's credentials), a
    cloud region and a Google Cloud project, and the context assumed when none is set."""

    model_config = ConfigDict(frozen=True)

    name: Provider
    label: str
    address: Literal["needed", "optional"]
    key: Literal["needed", "optional", "none"]
    region: bool
    project: bool
    context: int


def provider_info(provider: Provider) -> ProviderInfo:
    return ProviderInfo(
        name=provider,
        label=LABELS[provider],
        address="needed" if provider in SERVED else "optional",
        key=(
            "none"
            if provider in CLOUD_CREDENTIALS
            else "optional"
            if provider in KEY_OPTIONAL
            else "needed"
        ),
        region=provider in REGIONAL,
        project=provider in IN_PROJECT,
        context=LOCAL_CONTEXT if provider in LOCAL else FRONTIER_CONTEXT,
    )


PROVIDERS: tuple[ProviderInfo, ...] = tuple(provider_info(name) for name in LABELS)


class ModelSettings(BaseSettings):
    """The model both agents talk to, PACO_LLM_*."""

    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        # .env also holds the server's and the agents' other settings (paco.settings...).
        extra="ignore",
        frozen=True,
    )

    # vLLM's by default, as PACo's guided agent was first served (Qwen3 on a GPU of the user's).
    llm_provider: Provider = "vllm"
    # The model's API: Anthropic's, OpenAI's and Mistral's have their own address; a vLLM server
    # or another one needs it.
    llm_base_url: str | None = Field(
        default=None, validate_default=True, description="e.g. http://gpu-host:8001/v1"
    )
    # No default, so that a missing setting fails here, with its name.
    llm_model: str = Field(min_length=1, description="The name the API serves the model under.")
    # The user's API key; a server started without a key accepts any. Vertex AI takes Google's
    # Application Default Credentials instead (gcloud auth application-default login); Bedrock
    # takes a Bedrock API key here, or else AWS's credentials (AWS_PROFILE, AWS_ACCESS_KEY_ID...).
    llm_api_key: SecretStr = SecretStr(NO_KEY)
    # The cloud region: Vertex AI's location (global, us-east5...), Bedrock's AWS region.
    llm_region: str | None = Field(default=None, validate_default=True)
    # The Google Cloud project, on Vertex AI.
    llm_project: str | None = Field(default=None, validate_default=True)
    # The model's context, in tokens, as its server serves it (vLLM's --max-model-len); unset,
    # a small local model's on a server of the user's, a frontier model's on a provider's API
    # (context()). The autonomous agent needs 128k at least; each agent keeps its conversation
    # within it.
    llm_context: int | None = Field(default=None, ge=1)
    # Whether figures may reach the model. Off unless set: an online model's provider receives
    # what the model sees (§3, the data policy).
    llm_vision: bool = False
    # Whether the model thinks before its replies, unless a call says otherwise; forms are filled
    # without thinking.
    llm_thinking: bool = True
    # How hard it thinks, where the API takes it; the model's default when unset.
    llm_effort: Effort | None = None
    # The conversation's sampling on a server of the user's (vLLM, any other speaking OpenAI's
    # chat API); None keeps the server's own (Qwen3's: 0.6, as its makers advise when it thinks).
    # A form is filled at temperature 0.
    llm_temperature: float | None = None
    llm_seed: int | None = None

    @field_validator(
        "llm_base_url",
        "llm_region",
        "llm_project",
        "llm_context",
        "llm_effort",
        "llm_temperature",
        "llm_seed",
        mode="before",
    )
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return _unset(value)

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        return NO_KEY if value == "" else value

    def context(self) -> int:
        """The model's context: the one set, else a small local model's or a frontier one's."""
        if self.llm_context is not None:
            return self.llm_context
        return LOCAL_CONTEXT if self.llm_provider in LOCAL else FRONTIER_CONTEXT

    @field_validator("llm_base_url")
    @classmethod
    def _served_somewhere(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _address(value, info.data.get("llm_provider"))

    @field_validator("llm_region")
    @classmethod
    def _in_a_region(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _needed(value, info.data.get("llm_provider"), REGIONAL, "a cloud region")

    @field_validator("llm_project")
    @classmethod
    def _in_a_project(cls, value: str | None, info: ValidationInfo) -> str | None:
        return _needed(value, info.data.get("llm_provider"), IN_PROJECT, "a Google Cloud project")


class JudgeSettings(BaseSettings):
    """The evaluations' judge model, PACO_JUDGE_*: any provider, as the agents' model; none when
    PACO_JUDGE_MODEL is unset. Better another model than the one it judges."""

    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    judge_provider: Provider = "openai-compatible"
    judge_model: str | None = None
    judge_base_url: str | None = Field(default=None, validate_default=True)
    judge_api_key: SecretStr = SecretStr(NO_KEY)
    judge_region: str | None = None
    judge_project: str | None = None
    judge_effort: Effort | None = None

    # Named apart from the model's own validators: PACo's agent settings inherit both.
    @field_validator(
        "judge_model",
        "judge_base_url",
        "judge_region",
        "judge_project",
        "judge_effort",
        mode="before",
    )
    @classmethod
    def _judge_empty_is_unset(cls, value: object) -> object:
        return _unset(value)

    @field_validator("judge_base_url")
    @classmethod
    def _judge_served_somewhere(cls, value: str | None, info: ValidationInfo) -> str | None:
        if info.data.get("judge_model") is None:
            return value
        return _address(value, info.data.get("judge_provider"))


def _unset(value: object) -> object:
    """An empty setting is one not set: compose passes one for a variable its .env leaves out, and
    a settings page saves one over a value of .env."""
    return None if value == "" else value


def missing(error: ValidationError, prefix: str = "PACO_") -> str:
    """The settings a ValidationError names, as the environment spells them."""
    return ", ".join(f"{prefix}{problem['loc'][0]}".upper() for problem in error.errors())


def _address(value: str | None, provider: object) -> str | None:
    if value is None and provider in SERVED:
        raise ValueError(f"the address of the model's API is needed with provider {provider}")
    return value


def _needed(
    value: str | None, provider: object, providers: tuple[str, ...], what: str
) -> str | None:
    if value is None and provider in providers:
        raise ValueError(f"{what} is needed with provider {provider}")
    return value
