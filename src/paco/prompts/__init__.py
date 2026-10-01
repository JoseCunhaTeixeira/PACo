"""The prompts the model reads, in files: the agent's role, the scope form's instructions and
worked examples, and the server's instructions. Their version, logged with every turn, is a
hash of their text; `tests/test_prompts.py` pins it, so that every change to a prompt is a
reviewed one."""

import json
from functools import cache
from hashlib import sha256
from importlib.resources import files
from typing import Any

# Every prompt, by file name (without .md).
NAMES = ("role", "scope", "instructions")
# The pages of PAC the role sends the user to: PAC's tests check each is one of its pages.
PAC_PAGES = (
    "Active",
    "Passive",
    "Passive-active",
    "Dispersion picking",
    "Seismic inversion",
    "Petrophysical inversion",
    "Visualization",
)


@cache
def prompt(name: str) -> str:
    """The prompt `name` as the model reads it: the file's paragraphs and list items, each on
    one line (the file wraps them)."""
    text = (files(__package__) / f"{name}.md").read_text(encoding="utf-8")
    blocks = [block for block in text.split("\n\n") if block.strip()]
    return "\n\n".join(_unwrapped(block) for block in blocks)


def _unwrapped(block: str) -> str:
    """A paragraph on one line, or a list with each item on one line."""
    lines: list[str] = []
    for line in block.strip().splitlines():
        if line.startswith("- ") or not lines:
            lines.append(line.strip())
        else:
            lines[-1] += " " + line.strip()
    return "\n".join(lines)


@cache
def scope_examples() -> tuple[tuple[str, dict[str, Any]], ...]:
    """The scope form's worked examples: each message, and the fields of its form that differ
    from a message asking nothing."""
    text = (files(__package__) / "scope_examples.json").read_text(encoding="utf-8")
    return tuple((example["message"], example["form"]) for example in json.loads(text))


@cache
def version() -> str:
    """The prompts' version: the first hex digits of a hash of every prompt as the model reads
    it, and of the scope form's examples."""
    texts = [prompt(name) for name in NAMES]
    texts.append(json.dumps(scope_examples(), sort_keys=True))
    digest = sha256("\0".join(texts).encode()).hexdigest()
    return f"prompts-{digest[:8]}"
