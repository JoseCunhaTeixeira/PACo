"""The model's client, made from the settings: Claude through
Anthropic's own client (paco.llm.claude); the others through the client of OpenAI's chat API
(paco.llm.model), in the dialect each takes: OpenAI's, Gemini's, Mistral's, or a server's (vLLM,
Ollama, LM Studio, OpenRouter, a company's gateway). Both are PACo's ChatModel, for both modes;
the evaluations' judge is another client of the same model."""

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr

from paco.llm.claude import AnthropicChat
from paco.llm.model import PROBE, Dialect, OpenAIChat
from paco.llm.protocol import RECONNECT_S
from paco.llm.settings import ModelSettings, Provider

type Chat = OpenAIChat | AnthropicChat
# The address of a provider's API when the settings give none: OpenAI's and Anthropic's are their
# SDK's own.
ADDRESSES: dict[Provider, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "mistral": "https://api.mistral.ai/v1",
}
DIALECTS: dict[Provider, Dialect] = {
    "openai": "openai",
    "gemini": "gemini",
    "mistral": "plain",
    "server": "server",
}


def connect(
    provider: Provider,
    model: str,
    api_key: SecretStr,
    base_url: str | None = None,
    *,
    vision: bool = False,
    reconnect_s: float = RECONNECT_S,
) -> Chat:
    """The client of `model` at `provider`'s API (at `base_url` when given), with `api_key`; a
    server refusing the connection asked again for `reconnect_s`."""
    key = api_key.get_secret_value()
    if provider == "anthropic":
        client = AsyncAnthropic(api_key=key, base_url=base_url)
        # The prompt's fixed parts cached an hour on Anthropic's own API, which takes it.
        return AnthropicChat(
            client, model, vision=vision, reconnect_s=reconnect_s, hour=base_url is None
        )
    return OpenAIChat(
        AsyncOpenAI(base_url=base_url or ADDRESSES.get(provider), api_key=key),
        model,
        vision=vision,
        dialect=DIALECTS[provider],
        reconnect_s=reconnect_s,
    )


async def tried(chat: Chat) -> str:
    """The model's reply to a short question, without thinking: its provider, its name and the
    key given reach it. The client's error otherwise (a key refused, a model not there), which a
    settings page shows."""
    reply = await chat([{"role": "user", "content": PROBE}], [], thinking=False)
    return reply.content


async def blind(chat: Chat) -> str | None:
    """Why the autonomous mode cannot run on the model: it takes no images, and the mode judges
    every image by eye; None when it takes them."""
    if await chat.takes_images():
        return None
    return (
        f"{chat.name} takes no images: the autonomous mode judges every image by eye, so it needs "
        "a model that sees them. Choose one, or the workflow mode (PACO_AGENT_MODE=workflow)."
    )


def chat_model(settings: ModelSettings, *, reconnect_s: float = RECONNECT_S) -> Chat:
    """A client of the model `settings` name (PACO_LLM_*); a server refusing the connection asked
    again for `reconnect_s`."""
    return connect(
        settings.llm_provider,
        settings.llm_model,
        settings.llm_api_key,
        settings.llm_base_url,
        vision=True,
        reconnect_s=reconnect_s,
    )
