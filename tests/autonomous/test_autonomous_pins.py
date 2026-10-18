"""sigpipe runs as a library at the commit uv.lock records, PACo's quality control keyed by its own
sources, and no agent framework is among the dependencies (the autonomous agent's plan, §14)."""

import tomllib
from importlib import import_module
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from paco.autonomous import pins

ROOT = Path(__file__).resolve().parents[2]
# Agent frameworks the package does without (§14): its own loop, MCP through mcp, the model
# through its SDK. Each name stands for its family too (langchain-core, langgraph-checkpoint...).
FRAMEWORKS = (
    "ag2",
    "agno",
    "autogen",
    "crewai",
    "dspy",
    "haystack-ai",
    "langchain",
    "langgraph",
    "llama-index",
    "openai-agents",
    "pydantic-ai",
    "semantic-kernel",
    "smolagents",
)


def _locked() -> dict[str, dict[str, Any]]:
    """uv.lock's packages, by name."""
    with (ROOT / "uv.lock").open("rb") as lock:
        return {package["name"]: package for package in tomllib.load(lock)["package"]}


def test_sigpipe_is_imported_at_the_commit_uv_lock_records() -> None:
    module = import_module("sigpipe")

    assert pins.commit("sigpipe") == _locked()["sigpipe"]["source"]["git"].rpartition("#")[2]
    # The package imported is that distribution, not a checkout found elsewhere on the path.
    assert module.__file__ is not None
    installed = Path(str(distribution("sigpipe").locate_file("sigpipe")))
    assert Path(module.__file__).parent.resolve() == installed.resolve()


def test_pacos_quality_control_is_keyed_by_its_sources() -> None:
    assert len(pins.paco_code()) == 64
    assert pins.paco_code() == pins.paco_code()


def test_no_agent_framework_is_a_dependency() -> None:
    frameworks = [
        name
        for name in _locked()
        if any(name == family or name.startswith(f"{family}-") for family in FRAMEWORKS)
    ]

    assert frameworks == []
