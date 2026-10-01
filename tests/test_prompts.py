"""The prompts the model reads, from their files: a change to one is a reviewed change, with a
new version the turns log and the evaluation reports."""

from paco import prompts, server
from paco.agent.loop import ROLE

# The prompts' version: update it with the prompt, once the scenarios ran on it.
VERSION = "prompts-f7fdc480"


def test_the_prompts_version_is_the_one_reviewed() -> None:
    assert prompts.version() == VERSION


def test_every_prompt_comes_from_its_file() -> None:
    assert prompts.prompt("role") == ROLE
    assert prompts.prompt("instructions") == server.INSTRUCTIONS
    # A paragraph or a list item on one line, as the model reads it.
    assert all(len(line) > 60 or not line or line.startswith("#") for line in ROLE.splitlines())


def test_the_role_sends_the_user_to_pacs_pages_by_their_names() -> None:
    # PAC's tests check the other side: each name is one of its pages.
    for page in prompts.PAC_PAGES:
        assert page in ROLE, page
