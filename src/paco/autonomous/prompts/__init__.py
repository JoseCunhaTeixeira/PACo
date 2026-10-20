"""The prompts the autonomous agent reads, in files (docs/autonomous_plan.md §4): its role,
each phase's manual (which loads only in its phase), what the objective form and the answer form
add to PACo's own (PACo's scope and answer prompts, imported with their worked examples), the
judge's, which the search's sheets are judged with in short calls of their own, and the atlas,
reference figures of good and bad data with their captions, which a model with vision sees until
the freeze. Their version, logged with every call, is a hash of every prompt as the model
reads it, PACo's and the atlas among them; tests/test_prompts.py pins it, so that every change to a
prompt is a reviewed one."""

import json
from functools import cache
from hashlib import sha256
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

from paco import prompts as paco_prompts

# Every prompt, by file name (without .md).
NAMES = (
    "role",
    "manual_survey",
    "manual_curves",
    "manual_inversion",
    "objective",
    "answer",
    "judge",
)
# The atlas's sheets, each a figure (paco.autonomous.atlas draws them again) and its caption.
ATLAS = ("gathers", "images", "correlations")
ATLAS_FOLDER = files(__package__) / "atlas"


@cache
def prompt(name: str) -> str:
    """The prompt `name` as the model reads it: the file's paragraphs and list items, each on one
    line (the file wraps them), as PACo reads its own."""
    return _read(files(__package__) / f"{name}.md")


@cache
def atlas() -> tuple[tuple[str, bytes], ...]:
    """The atlas as the model sees it: each sheet's caption, read as a prompt is, and its figure,
    a PNG file."""
    return tuple(
        (_read(ATLAS_FOLDER / f"{name}.md"), (ATLAS_FOLDER / f"{name}.png").read_bytes())
        for name in ATLAS
    )


@cache
def objective_examples() -> tuple[tuple[str, dict[str, Any]], ...]:
    """The objective form's worked examples: each message, and the fields of its form that differ
    from a message asking nothing."""
    text = (files(__package__) / "objective_examples.json").read_text(encoding="utf-8")
    return tuple((example["message"], example["form"]) for example in json.loads(text))


@cache
def version() -> str:
    """The prompts' version: the first hex digits of a hash of every prompt as the model reads
    it, the forms' examples, the atlas's captions and figures, and PACo's prompts version, whose
    scope and answer prompts the forms build on."""
    texts = [prompt(name) for name in NAMES]
    texts.append(json.dumps(objective_examples(), sort_keys=True))
    texts.extend(caption for caption, _ in atlas())
    texts.append(sha256(b"".join(figure for _, figure in atlas())).hexdigest())
    texts.append(paco_prompts.version())
    digest = sha256("\0".join(texts).encode()).hexdigest()
    return f"autonomous-prompts-{digest[:8]}"


def _read(path: Traversable) -> str:
    """A prompt's file as the model reads it: its paragraphs and list items, each on one line."""
    text = path.read_text(encoding="utf-8")
    blocks = [block for block in text.split("\n\n") if block.strip()]
    return "\n\n".join(_unwrapped(block) for block in blocks)


def _unwrapped(block: str) -> str:
    """A paragraph on one line, or a list with each item on one line."""
    lines: list[str] = []
    for line in block.strip().splitlines():
        if line.startswith(("- ", "## ")) or not lines:
            lines.append(line.strip())
        else:
            lines[-1] += " " + line.strip()
    return "\n".join(lines)
