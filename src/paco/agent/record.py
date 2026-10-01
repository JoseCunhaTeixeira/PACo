"""What happened in a conversation, step by step: for the user to review, and for evaluation."""

import secrets
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class ModelStep(BaseModel):
    """One call to the model."""

    kind: Literal["model"] = "model"
    duration_s: float
    prompt_tokens: int | None  # the context it read: the number to watch in an 8k window
    completion_tokens: int | None
    tool_calls: int  # how many it asked for; 0 when it answered the user


class ToolStep(BaseModel):
    """One tool call the model asked for, or the host made for it (job_status following a job
    the model started)."""

    kind: Literal["tool"] = "tool"
    name: str
    arguments: str  # as the model wrote them
    # False when the host refused it: invalid arguments, over the budget, outside the message's
    # scope, or the repeat of a call that just failed.
    called: bool
    is_error: bool  # the call failed, or was refused
    duration_s: float
    result: str  # what the model read back
    by_host: bool = False  # the host followed a job with it; the model read its last result


class ScopeStep(BaseModel):
    """The scope the model read from one of the user's messages (paco.agent.scope), with the
    version of the prompts the turn used."""

    kind: Literal["scope"] = "scope"
    prompt_version: str
    scope: dict[str, Any] | None  # None when no form parsed
    error: str | None = None  # why, then
    tries: int  # 2 when the first form did not parse
    duration_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


class AnswerStep(BaseModel):
    """The answer form the model filled from its draft (paco.agent.answer)."""

    kind: Literal["answer"] = "answer"
    form: dict[str, Any] | None  # None when no form parsed: the draft is the answer
    error: str | None = None
    tries: int
    duration_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


type Step = Annotated[ModelStep | ToolStep | ScopeStep | AnswerStep, Field(discriminator="kind")]


class Transcript(BaseModel):
    """A whole conversation: the messages as the model saw them, and every step with its cost."""

    started_at: datetime
    model: str
    messages: list[dict[str, Any]]
    steps: list[Step]
    # The conversation's id, as each call carried it (the runs' agent_calls.jsonl name it),
    # and the prompts' version.
    conversation: str | None = None
    prompt_version: str | None = None

    @property
    def tool_steps(self) -> list[ToolStep]:
        """The tool calls the model asked for; the host's own are `host_steps`."""
        return [step for step in self.steps if isinstance(step, ToolStep) and not step.by_host]

    @property
    def host_steps(self) -> list[ToolStep]:
        """The calls the host made for the model: job_status following a job."""
        return [step for step in self.steps if isinstance(step, ToolStep) and step.by_host]

    @property
    def answer(self) -> str:
        """The model's last answer to the user."""
        for message in reversed(self.messages):
            if message["role"] == "assistant" and not message.get("tool_calls"):
                return str(message.get("content") or "")
        return ""


def save_transcript(transcript: Transcript, folder: Path) -> Path:
    """Write `transcript` in `folder`, named after when it started, and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{transcript.started_at:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}.json"
    path.write_text(transcript.model_dump_json(indent=2))
    return path
