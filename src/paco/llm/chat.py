"""The model's client, made from the settings (docs/geo_plan.md §3, §16): any model the user holds
an API key or a cloud account for. Claude through Anthropic's own client (paco.llm.claude), on
Anthropic's API or a cloud's: Google Cloud's Vertex AI, Amazon Bedrock (its Messages endpoint),
Microsoft Foundry. The others through the client of OpenAI's chat API (paco.llm.model), in the
dialect each takes: OpenAI's and Azure's (its v1 API, the deployment's name as the model), Gemini's
on Google's API or Vertex AI, Mistral's, a vLLM server's or another server's. Both are PACo's
ChatModel, so that PACo's agent runs on either too.

A cloud's credentials need its library, an extra of the package: Vertex AI's Application Default
Credentials google-auth (`uv sync --extra vertex`), Bedrock's AWS credentials boto3 (`uv sync
--extra bedrock`; a Bedrock API key needs none)."""

import importlib
import importlib.util
from collections.abc import Awaitable, Callable

import anyio
from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncAnthropicFoundry
from anthropic.lib.bedrock import AsyncAnthropicBedrockMantle
from anthropic.lib.vertex import AsyncAnthropicVertex
from openai import AsyncOpenAI
from pydantic import SecretStr

from paco.llm.claude import AnthropicChat, Client
from paco.llm.model import PROBE, Dialect, OpenAIChat
from paco.llm.protocol import RECONNECT_S
from paco.llm.settings import NO_KEY, Effort, JudgeSettings, ModelSettings, Provider

type Chat = OpenAIChat | AnthropicChat
# The address of a provider's API when the settings give none: OpenAI's and Anthropic's are their
# SDK's own, Bedrock's the region's, Vertex AI's the project's (below).
ADDRESSES: dict[Provider, str] = {
    "mistral": "https://api.mistral.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
DIALECTS: dict[Provider, Dialect] = {
    "openai": "openai",
    "azure-openai": "openai",
    "gemini": "gemini",
    "gemini-vertex": "gemini",
    "vllm": "vllm",
    "mistral": "plain",
    "openai-compatible": "plain",
}
CLAUDE: tuple[Provider, ...] = (
    "anthropic",
    "anthropic-vertex",
    "anthropic-bedrock",
    "anthropic-foundry",
)
# Google Cloud's scope for Vertex AI's tokens.
CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"


def connect(
    provider: Provider,
    model: str,
    api_key: SecretStr,
    base_url: str | None = None,
    *,
    region: str | None = None,
    project: str | None = None,
    vision: bool = False,
    thinking: bool = True,
    effort: Effort | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    reconnect_s: float = RECONNECT_S,
) -> Chat:
    """The client of `model` at `provider`'s API (at `base_url` when given), with `api_key`, or
    the cloud's credentials in `region` and `project`; `temperature` and `seed` the sampling on a
    server of the user's; a server refusing the connection asked again for `reconnect_s`. Refused,
    saying what to install, when a cloud's credentials need a library that is not there."""
    key = api_key.get_secret_value()
    given = key if key != NO_KEY else None
    if provider in CLAUDE:
        client = _claude(provider, given, base_url, region, project)
        return AnthropicChat(
            client,
            model,
            vision=vision,
            thinking=thinking,
            effort=effort,
            reconnect_s=reconnect_s,
        )
    secret: str | Callable[[], Awaitable[str]] = key
    address = base_url or ADDRESSES.get(provider)
    if provider == "azure-openai" and base_url is not None:
        address = _azure_v1(base_url)
    if provider == "gemini-vertex":
        secret, address = _google_token(), base_url or _vertex_openai(region, project)
        model = model if "/" in model else f"google/{model}"
    return OpenAIChat(
        AsyncOpenAI(base_url=address, api_key=secret),
        model,
        temperature,
        seed,
        vision=vision,
        thinking=thinking,
        dialect=DIALECTS[provider],
        effort=effort,
        reconnect_s=reconnect_s,
    )


async def tried(chat: Chat) -> str:
    """The model's reply to a short question, without thinking: its provider, its name and the
    credentials given reach it. The client's error otherwise (a key refused, a model not there),
    which a settings page shows."""
    reply = await chat([{"role": "user", "content": PROBE}], [], thinking=False)
    return reply.content


async def seeing(chat: Chat) -> str | None:
    """Vision as the model allows it (PACO_LLM_VISION, off unless set): switched off for the
    run, saying why, when the model takes no images; None when it takes them, or vision is off."""
    if not chat.vision or await chat.takes_images():
        return None
    chat.vision = False
    return (
        f"{chat.name} takes no images: vision is off for this run, the figures come as ridge "
        "tables and measures."
    )


def chat_model(settings: ModelSettings, *, reconnect_s: float = RECONNECT_S) -> Chat:
    """The client of the model `settings` name (PACO_LLM_*); a server refusing the connection
    asked again for `reconnect_s`."""
    return connect(
        settings.llm_provider,
        settings.llm_model,
        settings.llm_api_key,
        settings.llm_base_url,
        region=settings.llm_region,
        project=settings.llm_project,
        vision=settings.llm_vision,
        thinking=settings.llm_thinking,
        effort=settings.llm_effort,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        reconnect_s=reconnect_s,
    )


def judge_model(settings: JudgeSettings) -> Chat | None:
    """The client of the evaluations' judge (PACO_JUDGE_*); None without one. It reads text
    alone."""
    if settings.judge_model is None:
        return None
    return connect(
        settings.judge_provider,
        settings.judge_model,
        settings.judge_api_key,
        settings.judge_base_url,
        region=settings.judge_region,
        project=settings.judge_project,
        effort=settings.judge_effort,
    )


def _claude(
    provider: Provider,
    key: str | None,
    base_url: str | None,
    region: str | None,
    project: str | None,
) -> Client:
    """Claude's client on `provider`'s API: Anthropic's with `key`; Vertex AI's in `region` and
    `project`, with Google's Application Default Credentials; Bedrock's Messages endpoint in
    `region`, with a Bedrock API key (`key`) or else AWS's credentials; Foundry's at `base_url`,
    the resource's endpoint (https://<resource>.services.ai.azure.com/anthropic/), with `key`."""
    if provider == "anthropic-vertex":
        _needs(
            "google.auth", "google-auth", "Vertex AI's Application Default Credentials", "vertex"
        )
        return AsyncAnthropicVertex(
            region=region or NOT_GIVEN, project_id=project or NOT_GIVEN, base_url=base_url
        )
    if provider == "anthropic-bedrock":
        if key is None:
            _needs("boto3", "boto3", "Bedrock's AWS credentials (or a Bedrock API key)", "bedrock")
        return AsyncAnthropicBedrockMantle(aws_region=region, api_key=key, base_url=base_url)
    if provider == "anthropic-foundry" and base_url is not None:
        return AsyncAnthropicFoundry(api_key=key, base_url=_foundry(base_url))
    return AsyncAnthropic(api_key=key or NO_KEY, base_url=base_url)


def _azure_v1(endpoint: str) -> str:
    """Azure's v1 API at a resource's endpoint (https://<resource>.openai.azure.com), as OpenAI's
    client speaks to it."""
    trimmed = endpoint.rstrip("/")
    return f"{trimmed}/" if trimmed.endswith("/openai/v1") else f"{trimmed}/openai/v1/"


def _foundry(endpoint: str) -> str:
    """Claude's API at a Foundry resource's endpoint (https://<resource>.services.ai.azure.com)."""
    trimmed = endpoint.rstrip("/")
    return f"{trimmed}/" if trimmed.endswith("/anthropic") else f"{trimmed}/anthropic/"


def _vertex_openai(region: str | None, project: str | None) -> str:
    """Vertex AI's OpenAI-compatible endpoint in `region` (global: Google's global host) of
    `project`."""
    host = (
        "aiplatform.googleapis.com"
        if region in (None, "global")
        else f"{region}-aiplatform.googleapis.com"
    )
    return f"https://{host}/v1/projects/{project}/locations/{region or 'global'}/endpoints/openapi"


def _google_token() -> Callable[[], Awaitable[str]]:
    """Google Cloud's Application Default Credentials (gcloud auth application-default login, or
    GOOGLE_APPLICATION_CREDENTIALS) as the bearer token OpenAI's client asks before each request,
    refreshed when it expires (an hour)."""
    _needs("google.auth", "google-auth", "Vertex AI's Application Default Credentials", "vertex")
    google_auth = importlib.import_module("google.auth")
    requests = importlib.import_module("google.auth.transport.requests")
    try:
        credentials, _ = google_auth.default(scopes=[CLOUD_PLATFORM])
    except Exception as error:  # google.auth.exceptions.DefaultCredentialsError
        raise ValueError(
            f"No Application Default Credentials for Vertex AI ({error}): run `gcloud auth "
            "application-default login`, or set GOOGLE_APPLICATION_CREDENTIALS."
        ) from error

    async def token() -> str:
        if not credentials.valid:
            await anyio.to_thread.run_sync(credentials.refresh, requests.Request())
        return str(credentials.token)

    return token


def _needs(module: str, package: str, what: str, extra: str) -> None:
    """Refused, saying what to install, when `module` (of `package`) is not there."""
    try:
        found = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # its parent package is not there either
        found = False
    if not found:
        raise ValueError(
            f"{what} need {package}: `uv sync --extra {extra}` (or `pip install 'paco[{extra}]'`)."
        )
