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
    """One tool call the model asked for."""

    kind: Literal["tool"] = "tool"
    name: str
    arguments: str  # as the model wrote them
    called: bool  # False when the host refused it: invalid arguments, or over the budget
    is_error: bool  # the call failed, or was refused
    duration_s: float
    result: str  # what the model read back


type Step = Annotated[ModelStep | ToolStep, Field(discriminator="kind")]


class Transcript(BaseModel):
    """A whole conversation: the messages as the model saw them, and every step with its cost."""

    started_at: datetime
    model: str
    messages: list[dict[str, Any]]
    steps: list[Step]

    @property
    def tool_steps(self) -> list[ToolStep]:
        return [step for step in self.steps if isinstance(step, ToolStep)]

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
