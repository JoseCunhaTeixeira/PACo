"""The autonomous agent's server: the tools it serves, their errors as the model reads them, and its
command."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.types import CallToolResult, ImageContent

from paco import inspection, qc
from paco import server as paco_server
from paco.autonomous import server
from paco.autonomous.host import terminal
from paco.autonomous.tools.trials import Study
from paco.call import shown

# The tools in the plan's order (docs/autonomous_plan.md §6), those built so far; PACo's first.
PACO_TOOLS = ["inspect", "preset_settings", "inversion_settings"]
# PACo's petrophysical inversion on the study's frozen curves, after the inversion's tools.
PETRO_TOOLS = ["petro_models", "invert_petro"]
TOOLS = [
    *PACO_TOOLS,
    *("survey", "observe", "look", "advise", "try_settings", "search", "commit"),
    *("pick", "decide", "freeze", "invert", "job_status", *PETRO_TOOLS, "journal"),
]
# How long the server's command may take to listen: it loads PACo and sigpipe first.
STARTUP_S = 60.0


def test_the_tools_are_listed_in_the_plans_order() -> None:
    assert [tool.name for tool in anyio.run(server.server.list_tools)] == TOOLS


def test_pacos_read_only_tools_are_served_as_paco_serves_them() -> None:
    ours = {tool.name: tool for tool in anyio.run(server.server.list_tools)}
    pacos = {tool.name: tool for tool in anyio.run(paco_server.server.list_tools)}

    for name in PACO_TOOLS:
        # The same card: name, description, schemas and annotations; preset_settings' card names
        # this server's tools.
        card = pacos[name]
        if name == "preset_settings":
            assert ours[name].description == server.PRESET_SETTINGS
            card = card.model_copy(update={"description": server.PRESET_SETTINGS})
        assert ours[name] == card
        annotations = ours[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint


def test_inspect_says_no_retries_of_the_gates_budget() -> None:
    # The agent's inversions again would fill it: its budget is its own.
    def said() -> bool:
        return inspection.RETRIES_SAID.get()

    assert server._no_retries(said)() is False  # pyright: ignore[reportPrivateUsage]
    assert inspection.RETRIES_SAID.get()


def test_the_petrophysical_tools_take_the_profile_and_read_its_frozen_curves() -> None:
    ours = {tool.name: tool for tool in anyio.run(server.server.list_tools)}

    # The profile, as the study's other tools take it, not a run: the frozen curves are the
    # study's, the windows its model rejected left out.
    assert ours["petro_models"].input_schema["required"] == ["profile"]
    assert ours["invert_petro"].input_schema["required"] == ["profile", "model"]
    assert all("frozen curves" in (ours[name].description or "") for name in PETRO_TOOLS)
    assert ours["petro_models"].annotations == paco_server.READS
    assert ours["invert_petro"].annotations == paco_server.REPLACES


@pytest.mark.usefixtures("paco_env")
def test_the_soils_wait_for_the_freeze_and_take_its_curves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def study(*_: object) -> Study:
        return Study("active_test_profile", tmp_path)

    monkeypatch.setattr(server, "_study", study)
    for name, arguments in (("petro_models", {}), ("invert_petro", {"model": "a_model"})):
        refused = _call(name, {"profile": "active_test_profile", **arguments})
        assert refused.is_error
        assert "[precondition] The curves are not frozen yet" in shown(refused)
    frozen = SimpleNamespace(
        run_id="20261018-120000-abcd",
        curves=(SimpleNamespace(unit="xmid_74.25"), SimpleNamespace(unit="xmid_75.75")),
    )

    def frozen_of(_: object) -> object:
        return frozen

    monkeypatch.setattr(server.freezing, "frozen", frozen_of)
    asked: list[tuple[str, object]] = []

    def models(run_id: str, _settings: object, units: object) -> qc.PetroChoice:
        asked.append((run_id, units))
        return qc.PetroChoice(models=(), next="No model.")

    monkeypatch.setattr(server.qc, "petro_models", models)

    assert not _call("petro_models", {"profile": "active_test_profile"}).is_error
    assert asked == [("20261018-120000-abcd", ["xmid_74.25", "xmid_75.75"])]


def test_each_tool_says_what_it_changes() -> None:
    annotations = {tool.name: tool.annotations for tool in anyio.run(server.server.list_tools)}

    assert annotations["observe"] == annotations["look"] == paco_server.READS
    # The study's own files: its survey, advice and trials.
    assert annotations["survey"] == annotations["advise"] == paco_server.ADDS
    assert annotations["try_settings"] == annotations["decide"] == paco_server.ADDS
    assert annotations["journal"] == annotations["invert"] == paco_server.ADDS
    assert annotations["job_status"] == paco_server.READS
    # A curve or a run replaced, the curves frozen for good.
    assert annotations["commit"] == annotations["pick"] == paco_server.REPLACES
    assert annotations["freeze"] == paco_server.REPLACES


def test_the_list_names_every_tool(capsys: pytest.CaptureFixture[str]) -> None:
    server.main(["--list"])

    assert capsys.readouterr().out.splitlines() == TOOLS


@pytest.mark.usefixtures("paco_env")
def test_the_tools_read_the_profile() -> None:
    async def calls() -> list[CallToolResult]:
        async with Client(server.server) as client:
            return [
                await client.call_tool("inspect", {"what": "profiles"}),
                await client.call_tool("preset_settings", {"profile": "active_test_profile"}),
                await client.call_tool("inversion_settings", {}),
            ]

    profiles, preset, inversion = (shown(result) for result in anyio.run(calls))

    assert profiles.startswith("active_test_profile: a profile to process, no run yet")
    assert "masw" in json.loads(preset)["properties"]
    assert "n_iterations" in json.loads(inversion)["properties"]


@pytest.mark.usefixtures("paco_env")
def test_an_unforeseen_failure_reaches_the_model_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_: object) -> object:
        raise RuntimeError("the disk is on fire")

    def busy(*_: object) -> object:
        raise BlockingIOError("the study is being written")

    async def call() -> CallToolResult:
        async with Client(server.server) as client:
            return await client.call_tool("survey", {"profile": "active_test_profile"})

    monkeypatch.setattr(server, "_study", broken)
    crashed = anyio.run(call)
    monkeypatch.setattr(server, "_study", busy)
    retried = anyio.run(call)

    # Not only "Error executing tool survey": the reason, tagged as PACo tags it.
    assert crashed.is_error and shown(crashed) == (
        "Error executing tool survey: [error] RuntimeError: the disk is on fire"
    )
    assert retried.is_error and shown(retried) == (
        "Error executing tool survey: [retry] BlockingIOError: the study is being written"
    )


@pytest.mark.slow
@pytest.mark.usefixtures("paco_env", "searched")
def test_the_tools_study_the_profile() -> None:
    # A study from its survey to a committed line, the tools' errors tagged as PACo tags them.
    first = {"masw": {"length": 90, "distance_max": 150.0}, "dispersion": {"nv": 250}}
    reasons = {
        "masw length": "for a test",
        "masw distance_max": "every shot of the line, for a test",
        "dispersion nv": "a coarse grid, for a test",
    }

    async def calls() -> list[CallToolResult]:
        async with Client(server.server) as client:
            return [
                await client.call_tool(
                    "observe", {"profile": "active_test_profile", "level": "curves"}
                ),
                await client.call_tool("survey", {"profile": "active_test_profile"}),
                await client.call_tool(
                    "look",
                    {
                        "profile": "active_test_profile",
                        "what": "gather",
                        "where": "105.dat",
                        "mute": {"vmin": 80, "vmax": 1500},
                    },
                ),
                await client.call_tool(
                    "commit", {"profile": "active_test_profile", "changes": first, "reasons": {}}
                ),
                await client.call_tool(
                    "commit",
                    {"profile": "active_test_profile", "changes": first, "reasons": reasons},
                    meta={"vision": True},
                ),
                await client.call_tool(
                    "observe", {"profile": "active_test_profile", "level": "curves"}
                ),
                await client.call_tool(
                    "look", {"profile": "active_test_profile", "what": "image", "where": "71"}
                ),
                await client.call_tool(
                    "advise", {"profile": "active_test_profile", "topic": "picking"}
                ),
                await client.call_tool(
                    "try_settings",
                    {
                        "profile": "active_test_profile",
                        "variants": [{"picking": {"fmax": 90.0}}],
                        "windows": "spread:2",
                    },
                    meta={"vision": True},
                ),
                await client.call_tool(
                    "look", {"profile": "active_test_profile", "what": "experiment", "where": "E1"}
                ),
            ]

    results = anyio.run(calls)

    assert [result.is_error for result in results] == [
        True, False, False, True, False, False, False, False, False, False,
    ]  # fmt: skip
    no_line, surveyed, raw, refused, committed, curves, image, advice, trial, sheet = map(
        shown, results
    )
    # The SDK names the tool before the error.
    assert no_line == (
        "Error executing tool observe: [precondition] No line committed yet in this study: commit "
        "one first."
    )
    assert surveyed.startswith("active_test_profile: 96 receivers 1.5 m apart")
    # Before the first commit, the survey's raw records: G1's measures, vision being off.
    assert raw.startswith("105.dat, G1's measures in its gather's stead (vision is off): 105.dat:")
    assert refused.startswith(
        "Error executing tool commit: [bad argument] Give a one-line reason for each setting"
    )
    assert committed.startswith("L1 made run ")
    assert curves.startswith("G3: ")
    # Vision is off unless the host says it is on: a ridge table.
    assert image.startswith("xmid_71.25, its image as a ridge table")
    # Three of the long windows fail G3: the changes it suggests, none applied.
    assert advice.startswith("G3's changes of the picking, by window (not applied):")
    assert trial.startswith("E1: 1 variant on spread:2, against the current settings:")
    # With vision, a commit shows the line's images and picks, an experiment its variants.
    figures = [
        [block for block in result.content if isinstance(block, ImageContent)]
        for result in (results[4], results[8])
    ]
    assert [len(one) for one in figures] == [1, 1]
    assert "Its images and picks: " in committed and "E1 side by side, by row: current" in trial
    assert sheet.startswith("E1 at ") and "the panels' measures (vision is off):" in sheet


@pytest.mark.usefixtures("paco_env")
def test_the_command_serves_the_tools_the_chat_lists(
    free_port: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The command as installed, started from a folder without a .env: PACo's folders come from
    # the environment paco_env set, the port from PACO_PORT.
    command = Path(sys.executable).with_name("paco-server")
    environment = {**os.environ, "PACO_PORT": str(free_port)}
    url = f"http://127.0.0.1:{free_port}/mcp"
    log = tmp_path / "server.log"

    with log.open("w") as stderr:
        process = subprocess.Popen(
            [command, "--mode", "autonomous"], cwd=tmp_path, env=environment, stderr=stderr
        )
        try:
            _wait_until_listening(free_port, process)
            anyio.run(terminal.chat, url)
        finally:
            process.terminate()
            process.wait(timeout=STARTUP_S)

    assert capsys.readouterr().out.splitlines()[1:-1] == [f"  {name}" for name in TOOLS]
    assert f"Serving {', '.join(TOOLS)} at {url}" in log.read_text()


def _call(name: str, arguments: dict[str, Any]) -> CallToolResult:
    async def call() -> CallToolResult:
        async with Client(server.server) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(call)


def _wait_until_listening(port: int, process: subprocess.Popen[bytes]) -> None:
    """Wait until something listens on `port` of this machine, while `process` runs."""
    deadline = time.monotonic() + STARTUP_S
    while time.monotonic() < deadline and process.poll() is None:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    pytest.fail(
        f"paco-server --mode autonomous did not listen on port {port} "
        f"(exit status {process.poll()})"
    )
