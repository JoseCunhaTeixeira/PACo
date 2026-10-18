"""The model the agent talks to, behind a small interface (paco.llm.protocol): the real ones are
paco.llm's clients, made from the settings by paco.llm.chat; the tests use a scripted one."""

from paco.llm.model import OpenAIChat
from paco.llm.protocol import (
    FORM_TOKENS,
    RECONNECT_S,
    ChatModel,
    Filled,
    Reply,
    ToolCall,
    without_thinking,
)

__all__ = [
    "FORM_TOKENS",
    "RECONNECT_S",
    "ChatModel",
    "Filled",
    "OpenAIChat",
    "Reply",
    "ToolCall",
    "without_thinking",
]
