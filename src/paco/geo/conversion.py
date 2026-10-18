"""Between MCP and the model's chat API, for what PACo's conversion does not carry: a tool's
figures (docs/geo_plan.md §3). An OpenAI-compatible server takes images in user messages only, so a
result with figures becomes its tool message, which says they follow, then a user message that
holds them as data URLs."""

from mcp.types import CallToolResult, ImageContent
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam

from paco.agent.conversion import result_for_model
from paco.agent.loop import as_data


def tool_messages(
    call_id: str, tool: str, result: CallToolResult, vision: bool
) -> list[ChatCompletionMessageParam]:
    """The messages that carry `result`, of tool `tool` for call `call_id`: its tool message, the
    text as data (PACo's as_data, as PACo's loop sends it); and when the result holds figures and
    `vision` is on, a user message right after it, holding them. Without vision, the tool message
    says the figures were left out."""
    content = as_data(tool, result_for_model(result))
    figures = [block for block in result.content if isinstance(block, ImageContent)]
    if not figures:
        return [{"role": "tool", "tool_call_id": call_id, "content": content}]
    named = "The figure" if len(figures) == 1 else f"The {len(figures)} figures"
    if not vision:
        left_out = "was" if len(figures) == 1 else "were"
        note = f"{named} {left_out} left out: vision is off (PACO_LLM_VISION)."
        return [{"role": "tool", "tool_call_id": call_id, "content": f"{content}\n\n{note}"}]
    follows = "follows" if len(figures) == 1 else "follow"
    parts: list[ChatCompletionContentPartParam] = [
        {"type": "text", "text": f"{named} of {tool} (call {call_id}), as data:"}
    ]
    parts.extend(
        {"type": "image_url", "image_url": {"url": f"data:{figure.mime_type};base64,{figure.data}"}}
        for figure in figures
    )
    return [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"{content}\n\n{named} {follows}, in the next message.",
        },
        {"role": "user", "content": parts},
    ]
