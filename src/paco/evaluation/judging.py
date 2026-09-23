"""A judge model grades a conversation against the scenario's rubric, from 1 to 5."""

import json
import re
from typing import Any

from paco.agent.model import ChatModel
from paco.agent.record import Transcript
from paco.evaluation.models import JudgeScore

_RESULT_LENGTH = 300  # characters of each tool result shown to the judge
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

INSTRUCTIONS = (
    "You grade an assistant that helps a geophysicist process MASW seismic profiles with PACo's "
    "tools. Read the scenario's expectation and the conversation, then grade the assistant's "
    "final answer from 1 (wrong, invented or misleading) to 5 (correct, complete, and honest "
    'about what failed). Reply with JSON only: {"score": <1 to 5>, "reason": "<one sentence>"}.'
)


async def judge(model: ChatModel, rubric: str, transcript: Transcript) -> JudgeScore:
    """The judge model's grade of `transcript` against `rubric`."""
    prompt = f"Expectation: {rubric}\n\nConversation:\n{render(transcript)}"
    reply = await model(
        [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}], []
    )
    return read_score(reply.content)


def render(transcript: Transcript) -> str:
    """The conversation as the judge reads it: the user, the tool calls and their results (cut
    short), the assistant."""
    lines: list[str] = []
    for message in transcript.messages:
        role, content = message["role"], str(message.get("content") or "")
        if role == "user":
            lines.append(f"USER: {content}")
        elif role == "assistant":
            for call in message.get("tool_calls") or ():
                function = call["function"]
                lines.append(f"TOOL CALL: {function['name']} {function['arguments']}")
            if content:
                lines.append(f"ASSISTANT: {content}")
        elif role == "tool":
            cut = content if len(content) <= _RESULT_LENGTH else content[:_RESULT_LENGTH] + "..."
            lines.append(f"TOOL RESULT: {cut}")
    return "\n".join(lines)


def read_score(text: str) -> JudgeScore:
    """The score in the judge's reply; None when the reply holds no readable score."""
    match = _JSON_OBJECT.search(text)
    try:
        parsed: Any = json.loads(match.group()) if match else {}
        # A score outside 1 to 5 fails JudgeScore's validation, a ValueError.
        return JudgeScore(score=int(parsed["score"]), reason=str(parsed.get("reason", "")))
    except json.JSONDecodeError, KeyError, TypeError, ValueError:
        return JudgeScore(score=None, reason=f"unreadable judgement: {text[:200]}")
