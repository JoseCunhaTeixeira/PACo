"""The model servers of the tests: OpenAI-compatible, and Anthropic's API, scripted, reached through
httpx2's MockTransport as PACo's tests reach theirs."""

import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx2
from anthropic import AsyncAnthropic, AsyncAnthropicFoundry
from anthropic.lib.bedrock import AsyncAnthropicBedrockMantle
from anthropic.lib.vertex import AsyncAnthropicVertex
from openai import AsyncOpenAI

from paco.llm.claude import PROBE, Client

MODEL = "Qwen/Qwen3-14B-FP8"
# A PNG of one pixel, for a tool's figure.
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
# An answer: a chat message and its usage (None: the server says none); an HTTP error status;
# or the connection refused, or timed out.
type Answer = tuple[dict[str, Any], dict[str, Any] | None] | int | Literal["refused", "slow"]
ANSWERED: Answer = ({"role": "assistant", "content": "Two profiles."}, None)


class ScriptedServer:
    """An OpenAI-compatible model server, scripted: each request is answered with the next of
    `answers`, the last one again once none is left. The requests' bodies are kept, in order."""

    def __init__(self, *answers: Answer) -> None:
        self.answers: list[Answer] = list(answers) or [ANSWERED]
        self.bodies: list[dict[str, Any]] = []
        self.client = AsyncOpenAI(
            base_url="http://vllm.test/v1",
            api_key="secret",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._answer)),
        )

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if answer == "refused":
            raise httpx2.ConnectError("All connection attempts failed", request=request)
        if answer == "slow":
            raise httpx2.ReadTimeout("The model took too long", request=request)
        if isinstance(answer, int):
            return httpx2.Response(answer, json={"error": {"message": "scripted failure"}})
        message, usage = answer
        completion: dict[str, Any] = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
        if usage is not None:
            completion["usage"] = usage
        return httpx2.Response(200, json=completion)


def usage(prompt: int, completion: int, cached: int | None = None) -> dict[str, Any]:
    """A completion's usage, as an OpenAI-compatible server says it."""
    said: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached is not None:
        said["prompt_tokens_details"] = {"cached_tokens": cached}
    return said


def capabilities(*, adaptive: bool = True, enabled: bool = True, disabled: bool = True) -> Any:  # noqa: ANN401
    """What Anthropic's models API says a model takes: thinking (adaptive, a budget, off), every
    effort but xhigh, images and structured outputs."""
    yes, no = {"supported": True}, {"supported": False}
    return {
        "batch": yes,
        "citations": yes,
        "code_execution": no,
        "context_management": {"supported": False},
        "effort": {
            "supported": True,
            "low": yes,
            "medium": yes,
            "high": yes,
            "max": yes,
            "xhigh": no,
        },
        "image_input": yes,
        "pdf_input": yes,
        "server_tools": {"supported": False, "code_execution": no, "web_search": no},
        "structured_outputs": yes,
        "thinking": {
            "supported": adaptive or enabled,
            "types": {
                "adaptive": yes if adaptive else no,
                "between_tools": no,
                "disabled": yes if disabled else no,
                "enabled": yes if enabled else no,
            },
        },
    }


type Platform = Literal["anthropic", "vertex", "bedrock", "foundry"]


@dataclass(frozen=True)
class Accepts:
    """What a cloud platform's model takes, which no models API says there: thinking adaptive, with
    a budget, off; a forced tool; the effort; structured outputs; images."""

    adaptive: bool = True
    enabled: bool = False
    disabled: bool = True
    forced: bool = True
    effort: bool = True
    structured: bool = True
    images: bool = True

    def refuses(self, body: dict[str, Any]) -> bool:
        """Whether the platform refuses a request (400) for what it asks."""
        thinking = (body.get("thinking") or {}).get("type")
        output = body.get("output_config") or {}
        imaged = any(
            isinstance(part, dict) and part.get("type") == "image"
            for message in body.get("messages", [])
            for part in (message["content"] if isinstance(message["content"], list) else [])
        )
        return (
            (thinking == "adaptive" and not self.adaptive)
            or (thinking == "enabled" and not self.enabled)
            or (thinking == "disabled" and not self.disabled)
            or ((body.get("tool_choice") or {}).get("type") == "tool" and not self.forced)
            or ("effort" in output and not self.effort)
            or ("format" in output and not self.structured)
            or (imaged and not self.images)
        )


class ScriptedAnthropic:
    """Anthropic's API, scripted as ScriptedServer scripts OpenAI's: each messages request answered
    with the next of `answers` (an OpenAI chat message and its usage, an HTTP status, or the
    connection refused), read into Anthropic's response: a "thinking" key a thinking block, the
    text a text block, each tool call a tool_use block; when the request forces a tool, or offers
    the form's alone, the message's content (a form's JSON) its input. The models API describes
    the model with `described` (capabilities() by default; False: it answers 404). The requests'
    bodies are kept, in order, with their addresses and credentials.

    On a cloud `platform` (Vertex AI, Bedrock's Messages endpoint, Foundry), there is no models
    API: the client asks the model, which `accepts` answers, refusing (400) what it does not
    take, a probe taking no answer of `answers` (and kept apart, in `probes`)."""

    def __init__(
        self,
        *answers: Answer,
        described: Any = None,  # noqa: ANN401
        platform: Platform = "anthropic",
        accepts: Accepts | None = None,
    ) -> None:
        self.answers: list[Answer] = list(answers) or [ANSWERED]
        self.described = capabilities() if described is None else described
        # Anthropic's own API refuses nothing here: its models API says what the model takes.
        self.accepts = accepts if accepts is not None or platform == "anthropic" else Accepts()
        self.bodies: list[dict[str, Any]] = []
        self.probes: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self.credentials: list[str | None] = []
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self._answer))
        self.client: Client
        match platform:
            case "vertex":
                self.client = AsyncAnthropicVertex(
                    region="us-east5",
                    project_id="paco-project",
                    access_token="ya29.token",
                    max_retries=0,
                    http_client=http,
                )
            case "bedrock":
                self.client = AsyncAnthropicBedrockMantle(
                    aws_region="us-east-1", api_key="bedrock-key", max_retries=0, http_client=http
                )
            case "foundry":
                self.client = AsyncAnthropicFoundry(
                    api_key="foundry-key",
                    base_url="https://paco.services.ai.azure.com/anthropic/",
                    max_retries=0,
                    http_client=http,
                )
            case "anthropic":
                self.client = AsyncAnthropic(
                    base_url="http://claude.test",
                    api_key="sk-ant-secret",
                    max_retries=0,
                    http_client=http,
                )

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path.startswith("/v1/models/"):
            if self.described is False:
                return httpx2.Response(404, json=_error("not_found_error", "no such model"))
            return httpx2.Response(
                200,
                json={
                    "type": "model",
                    "id": request.url.path.rsplit("/", 1)[-1],
                    "display_name": "Claude",
                    "created_at": "2026-01-01T00:00:00Z",
                    "lifecycle": "active",
                    "max_input_tokens": 200_000,
                    "max_tokens": 64_000,
                    "capabilities": self.described,
                },
            )
        body = json.loads(request.content)
        model = body.get("model") or request.url.path.rsplit("/", 1)[-1].split(":")[0]
        self.urls.append(str(request.url))
        self.credentials.append(
            request.headers.get("authorization")
            or request.headers.get("api-key")
            or request.headers.get("x-api-key")
        )
        if self.accepts is not None and self.accepts.refuses(body):
            return httpx2.Response(400, json=_error("invalid_request_error", "not taken"))
        if _probe(body):
            self.probes.append(body)
            return _message(model, [{"type": "text", "text": "OK"}], None)
        self.bodies.append(body)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if answer == "refused":
            raise httpx2.ConnectError("All connection attempts failed", request=request)
        if answer == "slow":
            raise httpx2.ReadTimeout("The model took too long", request=request)
        if isinstance(answer, int):
            return httpx2.Response(answer, json=_error("api_error", "scripted failure"))
        message, said = answer
        content: list[dict[str, Any]] = []
        choice = body.get("tool_choice") or {}
        offered = [tool["name"] for tool in body.get("tools") or ()]
        if choice.get("type") == "tool" or offered == ["form"]:
            forced = {"type": "tool_use", "id": "toolu_form", "name": "form"}
            content.append({**forced, "input": json.loads(message.get("content") or "{}")})
        else:
            if message.get("thinking"):
                content.append(
                    {"type": "thinking", "thinking": message["thinking"], "signature": "s"}
                )
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls") or ():
                function = call["function"]
                arguments = json.loads(function["arguments"])
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                )
        return _message(model, content, said)


def _probe(body: dict[str, Any]) -> bool:
    """Whether a request only asks the model whether it takes a parameter."""
    (message,) = body["messages"][:1] or [{"content": ""}]
    parts = message["content"] if isinstance(message["content"], list) else [message["content"]]
    return any(isinstance(part, dict) and part.get("text") == PROBE for part in parts)


def _message(
    model: str, content: list[dict[str, Any]], said: dict[str, Any] | None
) -> httpx2.Response:
    """Anthropic's response holding `content`, with `said`'s usage."""
    uses = any(block["type"] == "tool_use" for block in content)
    tokens = said or {"prompt_tokens": 10, "completion_tokens": 5}
    cached = (tokens.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if uses else "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": tokens["prompt_tokens"] - cached,
                "output_tokens": tokens["completion_tokens"],
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0,
            },
        },
    )


def _error(kind: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}
