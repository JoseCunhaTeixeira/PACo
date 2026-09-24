import json
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.types import CallToolResult, Tool

from paco import profiles, server
from paco.presets import override_schema
from paco.settings import Settings

# The workflow's order: tools/list should list them the same way, every time.
TOOLS = [
    "list_profiles",
    "inspect_profile",
    "preset_settings",
    "run_processing",
    "pick",
    "inversion_settings",
    "invert",
    "job_status",
    "redo",
]
# Every card (name, description, argument schema) travels with every request to the model.
# Raised from 4,500 in milestone 5 for the three inversion tools (5,221 measured); 5,424 with
# milestone 14's stage tools.
CARDS_BUDGET = 5_500  # characters, all tools together
# The server's instructions go into the model's system prompt too.
INSTRUCTIONS_BUDGET = 600  # characters
# Four 24-receiver windows along the active demo line, as in test_runs.py.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window.
SHORT = {"n_iterations": 500, "n_burnin_iterations": 50, "n_chains": 1}


def _tools() -> list[Tool]:
    async def list_tools() -> list[Tool]:
        async with Client(server.server) as client:
            return (await client.list_tools()).tools

    return anyio.run(list_tools)


def _call(
    name: str, arguments: dict[str, Any], updates: list[tuple[float, float | None]] | None = None
) -> CallToolResult:
    """Call a tool as a host would, through an in-memory client; `updates` collects progress."""

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:  # noqa: ARG001
        if updates is not None:
            updates.append((progress, total))

    async def call() -> CallToolResult:
        async with Client(server.server) as client:
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


def test_inversion_settings_describe_every_parameter() -> None:
    schema = json.loads(_text(_call("inversion_settings", {})))

    assert "title" not in json.dumps(schema)
    for model in (schema, *schema["$defs"].values()):
        for name, parameter in model["properties"].items():
            assert parameter.get("description"), name


@pytest.mark.usefixtures("paco_env")
def test_the_workflow_process_pick_redo_invert() -> None:
    progress: list[tuple[float, float | None]] = []

    processed = _call(
        "run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS}, progress
    )
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""
    picked = _call("pick", {"run_id": run_id})

    assert not (processed.is_error or picked.is_error)
    # The host hears of every window as it finishes.
    assert progress == [(done, 4) for done in range(5)]
    # What the gates did, in a few lines, and what to call next.
    done = processed.structured_content
    assert done is not None
    assert done["summary"].splitlines()[:2] == ["G1: 2 pass", "G2: 4 pass"]
    # Each record's own trigger correction, the first one shown.
    assert (
        'Retried G1:shifted_trigger, 1.dat, 2.dat with preprocessing {"trigger":{"t0":0.0188}} '
        "(each its own): now 2 pass."
    ) in done["summary"]
    assert done["next"] == (
        f"pick comes next for run_id {run_id}, if the user asked for curves or models."
    )
    # The settings the gates changed, in words, for the agent to report.
    assert done["changed"] == [
        "trigger t0 the default -> 0.0188 at 1.dat, 2.dat (each window its own), by "
        "G1:shifted_trigger",
        "2.dat: traces [1, 89, 90] left out of the windows",
        "line: masw length 24 for the whole line: trial windows G3 passed, by length in "
        "receivers: 8/9 at 24.",
    ]
    assert picked.structured_content is not None
    assert picked.structured_content["summary"].splitlines()[:2] == [
        "G3: 3 pass, 1 reject",
        "G4: 4 pass",
    ]
    assert picked.structured_content["changed"] == [
        "corridor 0.2 -> 0.1 at xmid 14.88 (1) (each window its own), by G3:mode_jump"
    ]
    assert picked.structured_content["next"] == (
        f"3 curves passed G3 and G4. invert can run on run_id {run_id}, if the user asked for "
        "models; otherwise answer."
    )

    # Going back to the picking for the window G3 sent back, with other changes.
    redone = _call(
        "redo",
        {"run_id": run_id, "stage": "picking", "flag": "mode_jump", "changes": {"corridor": 0.1}},
    )
    assert redone.structured_content is not None
    assert "Retried backtrack, xmid 14.88 (1)" in redone.structured_content["summary"]

    # No question: the job starts in the background, and job_status follows it to the gates'
    # summary.
    started = _call("invert", {"run_id": run_id, "parameters": SHORT})
    assert started.structured_content is not None
    assert (started.structured_content["state"], started.structured_content["total"]) == (
        "queued",
        3,
    )
    status = _wait_for(started.structured_content["job_id"])
    assert (status["state"], status["done"], status["n_failed"]) == ("succeeded", 3, 0)
    # The smooth median models, at round depths.
    assert status["depths_m"] and len(status["vs_m_s"]) == len(status["depths_m"])
    assert status["summary"].startswith("G5: ")
    # G5's retries spent the run's budget: going back is refused, with what to do instead.
    refused = _call("redo", {"run_id": run_id, "stage": "inversion"})
    assert refused.is_error
    assert f"The retry budget of run '{run_id}' is spent" in _text(refused)


def test_invert_refuses_a_run_g4_has_not_judged(paco_env: Settings) -> None:
    processed = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""

    result = _call("invert", {"run_id": run_id})

    assert result.is_error
    assert f"Run '{run_id}' has not been judged up to G4: judge it before inverting." in (
        _text(result)
    )
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


@pytest.mark.usefixtures("paco_env")
def test_redo_needs_windows_that_exist() -> None:
    processed = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""

    unknown = _call("redo", {"run_id": run_id, "stage": "picking", "xmids": [3.0]})
    assert unknown.is_error
    assert "has no window at 3.00. Its xmids: 2.88, 8.88, 14.88, 20.88." in _text(unknown)
    moved = _call(
        "redo", {"run_id": run_id, "stage": "phase_shift", "changes": {"masw": {"length": 48}}}
    )
    assert moved.is_error
    assert "masw cannot change within a run" in _text(moved)


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
            "redo",
            {"run_id": "20260923-000000-0000", "stage": "picking"},
            "Unknown run '20260923-000000-0000'. Latest runs: none.",
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
            "- parameters: vs_layers must have length n_layers (3).",
        ),
        (
            # Qwen3-4B's guess, four times in a row, when the message only said "not permitted".
            "invert",
            {"run_id": "20260923-000000-0000", "parameters": {"iterations": 2000, "chains": 1}},
            "- parameters.iterations: unknown parameter. Allowed: n_layers, vs_layers, "
            "thickness_layers, n_iterations, n_burnin_iterations, n_chains. "
            "Did you mean n_iterations?\n"
            "- parameters.chains: unknown parameter. Allowed: n_layers, vs_layers, "
            "thickness_layers, n_iterations, n_burnin_iterations, n_chains. "
            "Did you mean n_chains?",
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
