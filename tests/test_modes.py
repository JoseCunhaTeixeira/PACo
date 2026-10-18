"""PACo's two agents behind its commands: paco-server and paco-agent serve and run the guided
agent, or the autonomous geophysicist with --mode autonomous or PACO_AGENT_MODE."""

import socket

import anyio
import pytest
from mcp.server import MCPServer

from paco import server
from paco.agent import terminal
from paco.geo import server as geo_server
from paco.llm.settings import NO_MODEL


def _names(served: MCPServer) -> list[str]:
    return [tool.name for tool in anyio.run(served.list_tools)]


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """The commands' JSON log lines left out: the tests' own logging stays as it is."""

    def setup(level: int = 0) -> None: ...

    monkeypatch.setattr(terminal.logs, "setup", setup)


def test_the_server_serves_the_modes_tools(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PACO_AGENT_MODE", raising=False)
    guided, autonomous = _names(server.server), _names(geo_server.server)

    server.main(["--list"])
    assert capsys.readouterr().out.splitlines() == guided
    server.main(["--mode", "autonomous", "--list"])
    assert capsys.readouterr().out.splitlines() == autonomous
    # The mode of .env when the command is not told.
    monkeypatch.setenv("PACO_AGENT_MODE", "autonomous")
    server.main(["--list"])
    assert capsys.readouterr().out.splitlines() == autonomous
    assert {"try_settings", "commit", "freeze"} <= set(autonomous) - set(guided)


def test_the_chat_runs_the_modes_agent(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("PACO_AGENT_MODE", "PACO_LLM_BASE_URL", "PACO_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    monkeypatch.setenv("PACO_GEO_MCP_URL", url)

    # The guided agent needs its model first.
    terminal.main([])
    assert capsys.readouterr().out == (
        NO_MODEL.format(missing="PACO_LLM_BASE_URL, PACO_LLM_MODEL") + "\n"
    )
    # The geophysicist's chat reaches its own server first.
    with pytest.raises(SystemExit) as stopped:
        terminal.main(["--mode", "autonomous"])
    assert str(stopped.value.code).startswith(f"No tools from the geophysicist's server at {url}")
