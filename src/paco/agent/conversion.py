"""Between MCP and the model: tool cards become OpenAI tools, results become the text the model
reads."""

import json

from mcp.types import CallToolResult, Tool
from openai.types.chat import ChatCompletionFunctionToolParam


def tools_for_model(tools: list[Tool]) -> list[ChatCompletionFunctionToolParam]:
    """The server's tool cards, in the format of the model's chat API."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def result_for_model(result: CallToolResult) -> str:
    """What the model reads of a tool's result: compact JSON, or the error's message.

    The server also sends its results as indented text, a third longer for nothing.
    """
    text = "\n".join(block.text for block in result.content if block.type == "text")
    structured = result.structured_content
    if result.is_error or structured is None:
        return text
    # A tool that returns a string (the settings tools) has it wrapped: {"result": "..."}.
    if isinstance(structured, dict) and structured.keys() == {"result"}:
        wrapped = structured["result"]
        if isinstance(wrapped, str):
            return wrapped
    return json.dumps(structured, separators=(",", ":"), ensure_ascii=False)
