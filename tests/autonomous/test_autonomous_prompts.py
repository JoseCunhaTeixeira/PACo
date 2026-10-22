"""The prompts the autonomous agent reads, from their files:
a change to one is a reviewed change, with a new version the calls log. The role says who decides,
how an experiment goes, how to look, the objective and when to ask; each phase loads its manual
alone; the tools' descriptions hold placeholders, never values a model would copy."""

import json

import anyio

from paco import server as paco_server
from paco.agent.conversion import tools_for_model
from paco.autonomous import prompts, server
from paco.autonomous.host.phases import MANUALS, TOOLS, offered

# The prompts' version: update it with the prompt, once the change is reviewed.
VERSION = "autonomous-prompts-b13bb0a4"
# Each phase's fixed part, the role, its manual and its tools' cards, takes this many tokens at
# most of the 128k context the autonomous mode needs: room for the physics and the procedures.
PHASE_TOKENS = 16_384


def test_the_prompts_version_is_the_one_reviewed() -> None:
    assert prompts.version() == VERSION


def test_each_prompt_reads_a_paragraph_or_an_item_a_line() -> None:
    for name in prompts.NAMES:
        lines = prompts.prompt(name).splitlines()
        assert lines, name
        # A paragraph or a list item on one line, as the model reads it.
        assert all(len(line) > 60 or not line or line.startswith("#") for line in lines), name


def test_the_role_says_who_decides_how_to_experiment_the_objective_and_when_to_ask() -> None:
    headings = [line for line in prompts.prompt("role").splitlines() if line.startswith("## ")]

    assert headings == [
        "## How a run goes",
        "## Who decides",
        "## How you work: experiments",
        "## Looking",
        "## The objective",
        "## Budgets and stopping",
        "## When to ask the user",
        "## What you cannot do",
    ]


def test_each_phase_the_model_works_in_loads_its_manual_alone() -> None:
    assert MANUALS == {"P1": "manual_survey", "P2": "manual_curves", "P4": "manual_inversion"}
    curves = prompts.prompt("manual_curves")
    for part in ("## Windows", "## Shots", "## Preprocessing", "## Images", "## Picking"):
        assert part in curves


def test_the_tools_descriptions_hold_placeholders_not_values() -> None:
    # A model copies the values of examples (PACo's lesson): PACo's own cards are its.
    pacos = {tool.name for tool in anyio.run(paco_server.server.list_tools)}
    for tool in anyio.run(server.server.list_tools):
        if tool.name in pacos and tool.name != "preset_settings":
            continue
        texts = [tool.description or ""] + [
            str(one.get("description", ""))
            for one in tool.input_schema.get("properties", {}).values()
        ]
        assert not any("e.g." in text for text in texts), tool.name


def test_each_phases_fixed_part_fits_in_its_budget() -> None:
    tools = anyio.run(server.server.list_tools)
    for phase, manual in MANUALS.items():
        cards = json.dumps(tools_for_model(offered(phase, tools)))
        fixed = len(prompts.prompt("role")) + len(prompts.prompt(manual)) + len(cards)
        assert fixed // 4 < PHASE_TOKENS, phase
    assert set(MANUALS) <= set(TOOLS)
