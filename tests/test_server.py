import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.types import CallToolResult, RequestParamsMeta, Tool
from sigpipe.masw.presets import override_schema
from sigpipe.masw.runs import caching
from sigpipe.masw.runs.writing import run_lock

from paco import inspection, server
from paco.settings import Settings, get_settings

# The workflow's order: tools/list should list them the same way, every time.
TOOLS = [
    "inspect",
    "preset_settings",
    "run_processing",
    "compare",
    "pick",
    "judge",
    "inversion_settings",
    "invert",
    "job_status",
    "petro_models",
    "invert_petro",
    "redo",
]
# Every card (name, description, argument schema) travels with every request to the model: keep
# them small.
# The served model reads 12,288 tokens at most; the cards take about a quarter of it (the
# largest prompt measured, 8,107 tokens).
CARDS_BUDGET = 11_600  # characters, all tools together
# The tools' schemas as last reviewed (T12).
SCHEMAS_SNAPSHOT = Path(__file__).parent / "data" / "tool_schemas.json"
# The server's instructions go into the model's system prompt too.
INSTRUCTIONS_BUDGET = 700  # characters
# Four 24-receiver windows along the active demo line, as in test_runs.py.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window.
SHORT = {"n_iterations": 500, "n_burnin_iterations": 50, "n_chains": 2}  # two at least


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
            # One conversation, as the host sends it: what a test did, it goes on with.
            meta: RequestParamsMeta = {"conversation": "test"}
            return await client.call_tool(name, arguments, progress_callback=on_progress, meta=meta)

    return anyio.run(call)


def _run_id(result: CallToolResult) -> str:
    assert not result.is_error and result.structured_content is not None
    return str(result.structured_content["run_id"])


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


def _schemas() -> dict[str, Any]:
    """Every tool as the server declares it: its card, its annotations, its result's schema."""
    return {
        tool.name: {
            "description": tool.description,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "annotations": tool.annotations.model_dump(exclude_none=True)
            if tool.annotations
            else None,
        }
        for tool in _tools()
    }


def test_the_tools_schemas_match_their_reviewed_snapshot() -> None:
    # T12: a change of a tool's schema is a change of what the model and PAC read: reviewed,
    # then the snapshot written again (PACO_SNAPSHOT_UPDATE=1), its version with it.
    schemas = _schemas()
    text = json.dumps(schemas, indent=1, sort_keys=True)
    version = "tools-" + hashlib.sha256(text.encode()).hexdigest()[:8]
    if os.environ.get("PACO_SNAPSHOT_UPDATE"):
        SCHEMAS_SNAPSHOT.write_text(
            json.dumps({"version": version, "tools": schemas}, indent=1, sort_keys=True) + "\n"
        )
    snapshot = json.loads(SCHEMAS_SNAPSHOT.read_text())

    changed = sorted(
        name
        for name in {*schemas, *snapshot["tools"]}
        if schemas.get(name) != snapshot["tools"].get(name)
    )
    assert not changed, (
        f"Tool schemas changed ({', '.join(changed)}): review the change, then "
        "PACO_SNAPSHOT_UPDATE=1 uv run pytest tests/test_server.py -k snapshot"
    )
    assert snapshot["version"] == version


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

    assert list(run_processing.input_schema["properties"]) == [
        "profile",
        "overrides",
        "mode",
        "again",
    ]


# ---------------------------------------------------------------- results


def test_profiles_are_listed_and_inspected(paco_env: Settings) -> None:
    listed = _call("inspect", {"what": "profiles"})
    inspected = _call("inspect", {"what": "profile", "profile": "passive_p1"})
    unsaid = _call("inspect", {"what": "run"})

    assert _text(listed) == (
        "active_p1: a profile to process, no run yet\npassive_p1: a profile to process, no run yet"
    )
    assert _text(inspected) == inspection.profile_text("passive_p1", paco_env)
    assert unsaid.is_error and "inspect(what=run) needs run_id." in _text(unsaid)


@pytest.mark.parametrize(
    ("profile", "preset"), [("active_p1", "active"), ("passive_p1", "passive")]
)
@pytest.mark.usefixtures("paco_env")
def test_preset_settings_are_the_profiles_override_schema(profile: str, preset: str) -> None:
    result = _call("preset_settings", {"profile": profile})

    assert json.loads(_text(result)) == override_schema(preset)


@pytest.mark.usefixtures("paco_env")
def test_an_active_profile_has_a_passive_active_mode() -> None:
    summary = _text(_call("inspect", {"what": "profile", "profile": "active_p1"}))

    assert summary.endswith("modes: active, passive-active.")
    settings = json.loads(
        _text(_call("preset_settings", {"profile": "active_p1", "mode": "passive-active"}))
    )
    assert settings == override_schema("passive-active")
    refused = _call("preset_settings", {"profile": "active_p1", "mode": "passive"})
    assert refused.is_error
    assert "Profile 'active_p1' is active: its modes are active, passive-active." in (
        _text(refused)
    )


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
    assert done["summary"].splitlines()[:2] == ["G1: 3 pass", "G2: 4 pass"]
    # No trigger corrected: the demo's files say the shot comes 20 ms in, where the first breaks
    # put it, and the muting is off (the trigger is part of it).
    assert "shifted_trigger" not in done["summary"]
    assert done["next"] == (
        f"pick comes next for run_id {run_id}, if the user asked for curves or models."
    )
    # The settings the gates changed, in words, for the agent to report.
    changed = done["changed"]
    # The line's rules: the shots' reach, the length as given, then the near field.
    assert changed[0].startswith(
        "line: masw distance_max 24.29 m: beyond it from the shot, the traces' median SNR falls "
        "under 2 dB (G1), so the windows stack no farther shot. masw length 24 for the whole "
        "line, as given: trial windows G3 passed 26/27 at 24 (picks "
    )
    assert "%). near_field distance_m " in changed[0]
    # No trace left out for its amplitude in one record: the line judges its receivers.
    assert len(changed) == 1
    # The lengths the ladder tried, for the agent to choose from: the user's only, here.
    assert done["lengths"] == [
        "line: 96 receivers 0.25 m apart (23.75 m); windows of up to 48 receivers (half the line)",
        "24 receivers (5.75 m): 26/27 trial windows passed G3, wavelengths 5.0-30.0 m (models "
        "down to about 15.0 m), picks within 40%, 4 windows on the line (proposed)",
    ]
    assert picked.structured_content is not None
    # Every window keeps its traces (none is off the decay in most records) and stacks its
    # shots out of the near field where it has farther ones.
    # xmid 2.88's ridge spans a metre of wavelength: G3 resamples it finer and keeps more of the
    # ridge, and rejects it when its retries are spent; xmid 20.88 falls off its neighbours: G4
    # picks it again along their curve. Three curves pass.
    assert picked.structured_content["summary"].splitlines()[:2] == [
        "G4: 4 pass",
        "G3: 3 pass, 1 reject",
    ]
    changed = picked.structured_content["changed"]
    # Too few points, and too narrow a span: resampled finer and more of the ridge kept.
    assert changed[0] == (
        "min_relative_coherence 0.5 -> 0.3; wavelength_step 1 -> 0.2 at xmid 2.88 (1), by "
        "G3:too_few_points"
    )
    assert changed[1] == "min_relative_coherence 0.3 -> 0.18 at xmid 2.88 (1), by G3:narrow_span"
    assert changed[2].endswith("at xmid 20.88 (1), by G4:outlier")
    assert picked.structured_content["next"] == (
        f"3 curves passed G3 and G4. invert can run on run_id {run_id}, if the user asked for "
        "models; otherwise answer. Higher modes (M1, M2) are picked by hand in PAC's Dispersion "
        "picking page, then invert takes them with M0."
    )

    # Going back to the picking for a window, with a change: the verdict of the gate that
    # judges the stage redone.
    redone = _call(
        "redo",
        {"run_id": run_id, "stage": "picking", "xmids": [2.88], "changes": {"corridor": 0.1}},
    )
    assert redone.structured_content is not None
    assert (
        'Retried backtrack, xmid 2.88 (1) with picking {"corridor":0.1}: now 1 pass.'
        in redone.structured_content["summary"]
    )

    # No question: the job starts in the background, and job_status follows it to the gates'
    # summary.
    started = _call("invert", {"run_id": run_id, "parameters": SHORT})
    # invert returns a job or the user's choice: one of two models, wrapped.
    assert started.structured_content is not None
    job = started.structured_content["result"]
    # xmid 2.88's curve, picked again in a narrower corridor, passed: 4 windows to invert.
    assert job["state"] in ("queued", "running")
    assert job["total"] == 4
    status = _wait_for(job["job_id"])
    assert (status["state"], status["done"]) == ("succeeded", 4)
    # A window may fail twice (the sampler is not seeded): left out, the job still succeeds.
    assert status["n_failed"] <= 1
    # The models (the ensembles), at round depths.
    assert status["depths_m"] and len(status["vs_m_s"]) == len(status["depths_m"])
    # G6 first, when the line reached it.
    assert any(line.startswith("G5: ") for line in status["summary"].splitlines()[:2])


def test_run_processing_takes_the_mode_as_an_argument(paco_env: Settings) -> None:
    # As preset_settings takes it: a model that asked preset_settings for a mode may leave
    # "mode" out of the overrides.
    processed = _call(
        "run_processing",
        {"profile": "active_p1", "overrides": SMALL_WINDOWS, "mode": "passive-active"},
    )

    assert not processed.is_error and processed.structured_content is not None
    run_id = processed.structured_content["run_id"]
    manifest = json.loads((paco_env.output_dir / "active_p1" / run_id / "run.json").read_text())
    assert manifest["preset"]["mode"] == "passive-active"
    refused = _call("run_processing", {"profile": "active_p1", "mode": "passive"})
    assert refused.is_error
    assert "Profile 'active_p1' is active: its modes are active, passive-active." in (
        _text(refused)
    )


def test_a_run_made_again_takes_its_images_from_the_cache(paco_env: Settings) -> None:
    # The muting given: no mute trial, the test's runs short.
    overrides = {**SMALL_WINDOWS, "muting": {"method": "none"}}
    arguments = {"profile": "active_p1", "overrides": overrides}
    first, again = _call("run_processing", arguments), _call("run_processing", arguments)

    manifests = [
        json.loads(next(paco_env.output_dir.glob(f"*/{run_id}/run.json")).read_text())
        for run_id in (_run_id(first), _run_id(again))
    ]
    # The same code, records and settings: the images the first run made, byte for byte.
    assert all(window["cached"] for window in manifests[1]["windows"])
    assert manifests[0]["windows"] and paco_env.cache_folder.is_relative_to(paco_env.output_dir)


def test_each_call_takes_the_cache_its_settings_give(
    paco_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[caching.Cache | None] = []

    def listing(_settings: Settings) -> list[str]:
        seen.append(caching.CACHE.get())
        return []

    monkeypatch.setattr(inspection, "list_profiles", listing)
    _call("inspect", {"what": "profiles"})
    monkeypatch.setenv("PACO_CACHE_GB", "0")
    get_settings.cache_clear()
    _call("inspect", {"what": "profiles"})

    assert seen == [caching.Cache(paco_env.cache_folder, max_bytes=2_000_000_000), None]
    assert caching.CACHE.get() is None


@pytest.mark.usefixtures("paco_env")
def test_one_vs_range_stands_for_every_layer_at_invert() -> None:
    # The form invert's card offers: checked as the job reads it, not against PAC's 2 layers
    # (refused, a model makes up a range for each layer).
    processed = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""

    result = _call(
        "invert",
        {"run_id": run_id, "parameters": {"vs_layers": [{"vs_min": 100, "vs_max": 180}]}},
    )

    # Past the parameters' check: refused only because the run has no curve yet.
    assert result.is_error
    assert "has no curve: pick it first" in _text(result)


def test_invert_refuses_a_run_without_a_curve(paco_env: Settings) -> None:
    processed = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""

    result = _call("invert", {"run_id": run_id})

    assert result.is_error
    assert f"Run '{run_id}' has no curve: pick it first." in _text(result)
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


def test_a_run_another_application_writes_is_refused_saying_who(
    paco_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = _call("run_processing", {"profile": "active_p1", "overrides": SMALL_WINDOWS})
    run_id = processed.structured_content["run_id"] if processed.structured_content else ""
    run_folder = paco_env.output_dir / "active_p1" / run_id
    monkeypatch.setattr(server, "WRITER_WAIT_S", 0.0)
    held, release = threading.Event(), threading.Event()

    def page_of_pac() -> None:
        with run_lock(run_folder, "PAC", shared=True):
            held.set()
            release.wait(timeout=60)

    thread = threading.Thread(target=page_of_pac)
    thread.start()
    assert held.wait(timeout=10)
    try:
        result = _call("pick", {"run_id": run_id})
    finally:
        release.set()
        thread.join(timeout=10)

    # Nothing picked, the model told why (S5).
    assert result.is_error
    assert f"[precondition] Run {run_id} is being written by one of PAC's pages" in _text(result)
    assert not list(run_folder.glob("xmid_*/DispersionCurves_0000.csv"))


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
            "inspect",
            {"what": "profile", "profile": "active_p2"},
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
            "length (97) exceeds the 96 receivers of profile 'active_p1': no window that long "
            "fits the line, you are stuck. Ask the user which length to use, with options: the "
            "whole line (96), half of it (48), or the ladder's proposal (no length).",
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
            # Two different ranges for three layers (3 layers alone is a request the job reads:
            # bounds derived).
            "invert",
            {
                "run_id": "20260923-000000-0000",
                "parameters": {
                    "n_layers": 3,
                    "vs_layers": [
                        {"vs_min": 100, "vs_max": 200},
                        {"vs_min": 150, "vs_max": 300},
                    ],
                },
            },
            "Invalid parameters (see inversion_settings):\n"
            "- parameters: vs_layers must have length n_layers (3).",
        ),
        (
            # A model's guess: the message names the right ones, or the model guesses again.
            "invert",
            {"run_id": "20260923-000000-0000", "parameters": {"iterations": 2000, "chains": 1}},
            "- parameters.iterations: unknown parameter. Allowed: layering, free, n_layers, "
            "vs_layers, thickness_layers, max_vs_drop, n_iterations, n_burnin_iterations, "
            "n_chains. Did you mean n_iterations?\n"
            "- parameters.chains: unknown parameter. Allowed: layering, free, n_layers, "
            "vs_layers, thickness_layers, max_vs_drop, n_iterations, n_burnin_iterations, "
            "n_chains. Did you mean n_chains?",
        ),
        (
            "job_status",
            {"job_id": "inv-20260923-000000-0000"},
            "Unknown job 'inv-20260923-000000-0000'.",
        ),
        (
            "petro_models",
            {"run_id": "20260923-000000-0000"},
            "Unknown run '20260923-000000-0000'.",
        ),
        (
            "invert_petro",
            {"run_id": "20260923-000000-0000", "model": "grand_est_15-50hz_193-415mps"},
            "Unknown run '20260923-000000-0000'.",
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

    monkeypatch.setattr(inspection, "list_profiles", broken)

    result = _call("inspect", {"what": "profiles"})

    assert result.is_error
    assert "internal detail" not in _text(result)
