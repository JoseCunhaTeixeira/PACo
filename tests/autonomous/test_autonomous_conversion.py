"""Tool results for the model's chat API: their figures after them, in a user message, as an
OpenAI-compatible server takes images."""

from autonomous_scripted import PNG
from mcp.types import CallToolResult, ImageContent, TextContent

from paco.agent.conversion import result_for_model
from paco.agent.loop import as_data
from paco.autonomous.conversion import tool_messages
from paco.llm.model import images

TEXT = TextContent(type="text", text="xmid 2.88: its dispersion image")
FIGURE = ImageContent(type="image", data=PNG, mime_type="image/png")


def test_a_result_without_a_figure_is_the_tool_message_pacos_loop_sends() -> None:
    result = CallToolResult(content=[TEXT])

    assert tool_messages("call_1", "observe", result, vision=True) == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": as_data("observe", result_for_model(result)),
        }
    ]


def test_figures_follow_their_tool_message_in_a_user_message() -> None:
    result = CallToolResult(content=[TEXT, FIGURE, FIGURE])

    tool, user = tool_messages("call_1", "look", result, vision=True)

    assert tool == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": f"{as_data('look', TEXT.text)}\n\nThe 2 figures follow, in the next message.",
    }
    picture = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    assert user == {
        "role": "user",
        "content": [
            {"type": "text", "text": "The 2 figures of look (call call_1), as data:"},
            picture,
            picture,
        ],
    }


def test_without_vision_a_figure_is_left_out_and_said() -> None:
    result = CallToolResult(content=[TEXT, FIGURE])

    messages = tool_messages("call_1", "look", result, vision=False)

    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": f"{as_data('look', TEXT.text)}\n\nThe figure was left out: vision is off.",
        }
    ]
    assert images(messages) == 0
