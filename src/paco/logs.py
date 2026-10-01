"""Log lines as JSON (C6): when, the level, the logger, the message, and the conversation, the
turn and the run the line belongs to, as the server's tools and the agent's loop set them."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# What a log line belongs to: set by the server for each tool call, and by the agent for each
# answer; contextvars follow the call into its thread.
CONVERSATION: ContextVar[str | None] = ContextVar("conversation", default=None)
TURN: ContextVar[int | None] = ContextVar("turn", default=None)
RUN: ContextVar[str | None] = ContextVar("run", default=None)
# The model and the prompts' version of the call: what an attempt was made with (S4).
MODEL: ContextVar[str | None] = ContextVar("model", default=None)
PROMPTS: ContextVar[str | None] = ContextVar("prompts", default=None)


class JsonLines(logging.Formatter):
    """One JSON object a line, the fields with no value left out."""

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "at": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "conversation": CONVERSATION.get(),
            "turn": TURN.get(),
            "run": RUN.get(),
        }
        if record.exc_info:
            line["error"] = self.formatException(record.exc_info)
        return json.dumps({key: value for key, value in line.items() if value is not None})


def setup(level: int = logging.INFO) -> None:
    """Every log line of the process as JSON, on stderr (stdout stays the command's own)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLines())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
