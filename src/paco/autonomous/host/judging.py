"""The search's sheets judged by eye in short calls of their own, which cost far less than judging
them in the conversation. The host sends the model only the judging prompt, the atlas's dispersion
images, the line's facts and the sheets the server gives, without the conversation, its state
card, its manuals or its tools, and reads the variants it keeps in a form, without thinking. The
server's judge tool records them and gives the next sheets, until the finalists."""

import base64
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult, ImageContent, TextContent
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paco.agent.loop import as_data
from paco.agent.model import ChatModel
from paco.autonomous import prompts
from paco.llm.protocol import Filled

# How the line's method is said to the judge.
METHODS = {
    "active": "active: each window stacks its shots' images",
    "passive-active": "passive-active: each window stacks its shots' correlations",
    "passive": "passive: each window stacks its noise segments' correlations",
}
# The atlas's sheet the judge sees: its dispersion images, clean and bad.
ATLAS = "images"


class Kept(BaseModel):
    """A sheet's kept variants."""

    model_config = ConfigDict(extra="forbid")

    sheet: int = Field(description="The sheet's number.")
    kept: list[str] = Field(description="The labels of the variants kept, g<n>.")


class JudgeForm(BaseModel):
    """The judge's answer: why, written first (a form does not think), then the variants kept
    of each sheet."""

    model_config = ConfigDict(extra="forbid")

    why: str = Field(description="What the kept show that the others do not: a sentence a sheet.")
    sheets: list[Kept]


SCHEMA: dict[str, Any] = JudgeForm.model_json_schema()


class JudgeError(Exception):
    """The judge's form did not parse twice."""

    def __init__(self, said: str, fills: tuple[Filled, ...]) -> None:
        super().__init__(said)
        self.fills = fills


@dataclass(frozen=True)
class Judged:
    """What the judge kept, by sheet number, its reason, and the forms it filled."""

    kept: dict[int, list[str]]
    why: str
    fills: tuple[Filled, ...]


def facts(profile: str, mode: str, objective: str) -> str:
    """The line's facts the judge reads before the sheets."""
    return f"Line {profile}, {METHODS.get(mode, mode)}. The objective: {objective}."


def text_of(result: CallToolResult) -> str:
    """A tool result's text."""
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


async def judge(
    model: ChatModel,
    line: str,
    sheets: CallToolResult,
    *,
    vision: bool,
    refused: str | None = None,
) -> Judged:
    """The variants `model` keeps of the `sheets` the server shows (its text and figures), by eye;
    `line`: the line's facts; `refused`: why the server refused the last answer, said again. A
    form that does not parse is sent back once with its error; raises JudgeError after."""
    text = f"{line}\n\n{text_of(sheets)}"
    if refused is not None:
        text += f"\n\nYour last answer was not recorded: {refused}"
    parts: list[ChatCompletionContentPartParam] = [{"type": "text", "text": text}]
    if vision:
        parts.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
            }
            for block in sheets.content
            if isinstance(block, ImageContent)
        )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompts.prompt("judge")},
        *(_atlas() if vision else []),
        {"role": "user", "content": parts},
    ]
    fills: list[Filled] = []
    said = ""
    for _ in range(2):
        filled = await model.fill(messages, SCHEMA)
        fills.append(filled)
        try:
            form = JudgeForm.model_validate_json(filled.content)
        except ValidationError as error:
            said = "; ".join(item["msg"] for item in error.errors())
            messages.append({"role": "assistant", "content": filled.content})
            messages.append({"role": "user", "content": f"Not valid: {said}. Fill it again."})
            continue
        kept = {one.sheet: one.kept for one in form.sheets if one.kept}
        return Judged(kept=kept, why=form.why, fills=tuple(fills))
    raise JudgeError(said, tuple(fills))


def _atlas() -> list[ChatCompletionMessageParam]:
    """The atlas's dispersion images as the judge reads them, in a turn of their own, before the
    sheets: the same from call to call, read from the model server's cache."""
    caption, figure = dict(zip(prompts.ATLAS, prompts.atlas(), strict=True))[ATLAS]
    url = f"data:image/png;base64,{base64.b64encode(figure).decode()}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": as_data("atlas", caption)},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
        {"role": "assistant", "content": "The atlas, seen: now the sheets."},
    ]


def as_arguments(profile: str, judged: Judged) -> dict[str, Any]:
    """The judge tool's arguments for what the judge kept."""
    return {
        "profile": profile,
        "kept": {str(sheet): labels for sheet, labels in judged.kept.items()},
        "why": judged.why,
    }
