"""The geophysicist's chat in a terminal (paco-agent --mode autonomous), which lists its server's
tools, then talks with the model of .env."""

import anyio
import pytest

from paco.geo import server
from paco.geo.host import terminal
from paco.llm.settings import NO_MODEL


def test_the_chat_lists_the_tools_the_server_offers(capsys: pytest.CaptureFixture[str]) -> None:
    anyio.run(terminal.chat, server.server)

    tools = [tool.name for tool in anyio.run(server.server.list_tools)]
    assert capsys.readouterr().out.splitlines() == [
        f"The geophysicist's server offers {len(tools)} tools:",
        *(f"  {name}" for name in tools),
        NO_MODEL.format(missing="PACO_LLM_BASE_URL, PACO_LLM_MODEL"),
    ]


def test_the_chat_stops_saying_why_when_no_server_answers(free_port: int) -> None:
    url = f"http://127.0.0.1:{free_port}/mcp"

    with pytest.raises(SystemExit) as stopped:
        anyio.run(terminal.chat, url)

    assert str(stopped.value.code).startswith(
        f"No tools from the geophysicist's server at {url} (ConnectError: "
    )
