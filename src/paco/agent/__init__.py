"""The agent: the host between the model (Qwen, served by vLLM) and PACo's MCP server.

The only package that imports openai. The model sees the server's tool cards and instructions,
asks for tool calls, and reads compact results.
"""

from .conversion import result_for_model, tools_for_model
from .loop import Agent, Limits
from .model import ChatModel, Filled, OpenAIChat, Reply, ToolCall, without_thinking
from .record import (
    AnswerStep,
    ModelStep,
    ScopeStep,
    Step,
    ToolStep,
    Transcript,
    save_transcript,
)
from .settings import AgentSettings
from .terminal import chat

__all__ = [
    "Agent",
    "AgentSettings",
    "AnswerStep",
    "ChatModel",
    "Filled",
    "Limits",
    "ModelStep",
    "OpenAIChat",
    "Reply",
    "ScopeStep",
    "Step",
    "ToolCall",
    "ToolStep",
    "Transcript",
    "chat",
    "result_for_model",
    "save_transcript",
    "tools_for_model",
    "without_thinking",
]
