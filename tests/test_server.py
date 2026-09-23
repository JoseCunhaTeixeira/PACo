import json
import time
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest
from mcp import Client
from mcp.client.session import ClientRequestContext
from mcp.types import CallToolResult, ElicitRequestParams, ElicitResult, Tool

from paco import profiles, server
from paco.presets import override_schema
from paco.settings import Settings

# The workflow's order: tools/list should list them the same way, every time.
TOOLS = [
    "list_profiles",
    "inspect_profile",
    "preset_settings",
    "run_processing",
    "quality_settings",
    "dispersion_quality",
    "pick",
    "inversion_settings",
    "invert",
    "job_status",
]
# Every card (name, description, argument schema) travels with every request to the model.
# Raised from 4,500 in milestone 5 for the three inversion tools (5,221 measured).
CARDS_BUDGET = 5_500  # characters, all tools together
# The server's instructions go into the model's system prompt too.
INSTRUCTIONS_BUDGET = 600  # characters
# Four 24-receiver windows along the active demo line, as in test_runs.py.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window.
SHORT = {"n_iterations": 500, "n_burnin_iterations": 50, "n_chains": 1}

type Answer = Literal["accept", "decline", "cancel"]


def _tools() -> list[Tool]:
    async def list_tools() -> list[Tool]:
        async with Client(server.server) as client:
            return (await client.list_tools()).tools

    return anyio.run(list_tools)


def _call(
    name: str,
    arguments: dict[str, Any],
    updates: list[tuple[float, float | None]] | None = None,
    answer: Answer = "cancel",
    asked: list[str] | None = None,
) -> CallToolResult:
    """Call a tool as a host would, through an in-memory client.

    `updates` collects progress; a question to the user gets `answer`, and goes into `asked`.
    """

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:  # noqa: ARG001
        if updates is not None:
            updates.append((progress, total))

    async def on_question(
        context: ClientRequestContext,  # noqa: ARG001
        params: ElicitRequestParams,
    ) -> ElicitResult:
        if asked is not None:
            asked.append(params.message)
        return ElicitResult(
            action=answer, content={"approve": True} if answer == "accept" else None
        )

    async def call() -> CallToolResult:
        async with Client(server.server, elicitation_callback=on_question) as client:
            return await client.call_tool(name, arguments, progress_callback=on_progress)

    return anyio.run(call)


def _text(result: CallToolResult) -> str:
    (content,) = result.content
    assert content.type == "text"
    return content.text


def _card(tool: Tool) -> str:
    """The tool as a host sends it to the model."""
    return json.dumps(
        {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}
    )


# ---------------------------------------------------------------- what the model reads


def test_tools_are_listed_in_the_workflows_order() -> None:
    assert [tool.name for tool in _tools()] == TOOLS


def test_cards_stay_within_budget() -> None:
    assert sum(len(_card(tool)) for tool in _tools()) <= CARDS_BUDGET


def test_every_argument_is_described() -> None:
    for tool in _tools():
        for name, argument in tool.input_schema["properties"].items():
            assert argument.get("description"), f"{tool.name}.{name}"


def test_the_instructions_give_the_whole_workflow() -> None:
    async def instructions() -> str | None:
        async with Client(server.server) as client:
            return client.instructions

    text = anyio.run(instructions)

    assert text is not None
    assert len(text) <= INSTRUCTIONS_BUDGET
    for name in TOOLS:
        assert name in text


def test_host_checks_follow_the_allowed_hosts(demo_input_dir: Path) -> None:
    open_to_all = Settings(input_dir=demo_input_dir)
    compose = Settings(input_dir=demo_input_dir, allowed_hosts=("paco-server:*", "localhost:*"))

    assert server.transport_security(open_to_all) is None
    security = server.transport_security(compose)
    assert security is not None
    assert security.model_dump() == {
        "enable_dns_rebinding_protection": True,
        "allowed_hosts": ["paco-server:*", "localhost:*"],
        "allowed_origins": ["http://paco-server:*", "http://localhost:*"],
    }


def test_the_context_is_not_an_argument() -> None:
    (run_processing,) = [tool for tool in _tools() if tool.name == "run_processing"]

    assert list(run_processing.input_schema["properties"]) == ["profile", "overrides"]


# ---------------------------------------------------------------- results


def test_profiles_are_listed_and_inspected(paco_env: Settings) -> None:
    listed = _call("list_profiles", {})
    inspected = _call("inspect_profile", {"profile": "passive_p1"})

    assert listed.structured_content == {"result": ["active_p1", "passive_p1"]}
    assert inspected.structured_content == (
        profiles.inspect_profile("passive_p1", paco_env).model_dump(mode="json")
    )


@pytest.mark.parametrize(
    ("profile", "preset"), [("active_p1", "active"), ("passive_p1", "passive")]
)
@pytest.mark.usefixtures("paco_env")
def test_preset_settings_are_the_profiles_override_schema(profile: str, preset: str) -> None:
    result = _call("preset_settings", {"profile": profile})

    assert json.loads(_text(result)) == override_schema(preset)


def test_quality_settings_describe_every_parameter() -> None:
    schemas = json.loads(_text(_call("quality_settings", {})))

    assert set(schemas) == {"picking", "thresholds"}
    for schema in schemas.values():
        assert "title" not in schema
        for name, parameter in schema["properties"].items():
            assert parameter.get("description"), name


def test_inversion_settings_describe_every_parameter() -> None:
    schema = json.loads(_text(_call("inversion_settings", {})))

    assert "title" not in json.dumps(schema)
    for model in (schema, *schema["$defs"].values()):
        for name, parameter in model["properties"].items():
            assert parameter.get("description"), name


def test_the_workflow_process_judge_pick_invert(paco_env: Settings) -> None:
    progress: list[tuple[float, float | None]] = []
    asked: list[str] = []

    processed = _call(
        "run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}, progress
    )
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""
    judged = _call("dispersion_quality", {"run_id": run_id})
    picked = _call("pick", {"run_id": run_id})

    assert not (processed.is_error or judged.is_error or picked.is_error)
    # The host hears of every window as it finishes.
    assert progress == [(done, 4) for done in range(5)]
    assert judged.structured_content is not None
    assert judged.structured_content["good"] == 4
    assert picked.structured_content is not None
    assert picked.structured_content["n_picked"] == 4

    # The user declines: nothing starts.
    declined = _call("invert", {"run_id": run_id}, answer="decline", asked=asked)
    assert declined.is_error
    assert f"The user did not approve the inversion of run {run_id}." in _text(declined)
    assert not (paco_env.output_dir / "active_p1" / run_id / "inversion.json").exists()

    # The user approves: the job starts in the background, and job_status follows it.
    started = _call("invert", {"run_id": run_id, "parameters": SHORT}, answer="accept", asked=asked)
    assert started.structured_content is not None
    assert (started.structured_content["state"], started.structured_content["total"]) == (
        "queued",
        4,
    )
    assert len(asked) == 2
    assert asked[1].startswith(f"The agent asks to invert run {run_id}: the M0 curves of 4 ")
    status = _wait_for(started.structured_content["job_id"])
    assert (status["state"], status["done"], status["n_failed"]) == ("succeeded", 4, 0)
    assert len(status["vs_m_s"]) == 2


def test_invert_asks_nothing_before_the_run_is_picked(paco_env: Settings) -> None:
    asked: list[str] = []
    processed = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""

    result = _call("invert", {"run_id": run_id}, answer="accept", asked=asked)

    assert result.is_error
    assert f"Run '{run_id}' has no picks yet: call pick first." in _text(result)
    assert asked == []
    assert not (paco_env.output_dir / "active_p1" / run_id / "inversion.json").exists()


def _wait_for(job_id: str, timeout_s: float = 120) -> dict[str, Any]:
    """The job's status once it has stopped running."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = _call("job_status", {"job_id": job_id}).structured_content
        assert status is not None
        if status["state"] not in ("queued", "running"):
            return status
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} still running after {timeout_s} s")


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        (
            "inspect_profile",
            {"profile": "active_p2"},
            "Unknown profile 'active_p2'. Available profiles: active_p1, passive_p1.",
        ),
        (
            "run_processing",
            {"profile": "active_p1", "overrides": {"masw": {"lenght": 24}}},
            "- masw.lenght: unknown parameter. Allowed: length, step, distance_min, distance_max. "
            "Did you mean length?",
        ),
        (
            "run_processing",
            {"profile": "active_p1", "overrides": {"masw": {"length": 97}}},
            "length (97) exceeds the 96 receivers of profile 'active_p1'.",
        ),
        (
            "dispersion_quality",
            {"run_id": "20260923-000000-0000"},
            "Unknown run '20260923-000000-0000'. Latest runs: none.",
        ),
        (
            "dispersion_quality",
            {"run_id": "20260923-000000-0000", "thresholds": {"min_sharpnes": 1}},
            "Invalid thresholds (see quality_settings):\n"
            "- thresholds.min_sharpnes: Extra inputs are not permitted",
        ),
        (
            "pick",
            {"run_id": "20260923-000000-0000"},
            "Unknown run '20260923-000000-0000'. Latest runs: none.",
        ),
        (
            "invert",
            {"run_id": "20260923-000000-0000"},
            "Unknown run '20260923-000000-0000'. Latest runs: none.",
        ),
        (
            "invert",
            {"run_id": "20260923-000000-0000", "parameters": {"n_layers": 3}},
            "Invalid parameters (see inversion_settings):\n"
            "- parameters: Value error, vs_layers must have length n_layers (3)",
        ),
        (
            "job_status",
            {"job_id": "inv-20260923-000000-0000"},
            "Unknown job 'inv-20260923-000000-0000'.",
        ),
    ],
)
@pytest.mark.usefixtures("paco_env")
def test_pacos_errors_reach_the_model(name: str, arguments: dict[str, Any], message: str) -> None:
    result = _call(name, arguments)

    assert result.is_error
    assert message in _text(result)


@pytest.mark.usefixtures("paco_env")
def test_a_bug_stays_hidden_from_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_settings: Settings) -> list[str]:
        raise KeyError("internal detail")

    monkeypatch.setattr(profiles, "list_profiles", broken)

    result = _call("list_profiles", {})

    assert result.is_error
    assert "internal detail" not in _text(result)
