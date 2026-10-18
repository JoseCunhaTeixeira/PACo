"""The model client's smoke test (M1): each kind of call in turn, the usage of each read back, a
failed call kept with its error; played here against a scripted server."""

import anyio
import pytest
from geo_scripted import ANSWERED, MODEL, ScriptedServer, usage

from paco.geo import server as geo_server
from paco.geo.evaluation import cli, smoke
from paco.llm.model import OpenAIChat
from paco.llm.settings import ModelSettings


def test_the_smoke_test_makes_each_kind_of_call_in_turn() -> None:
    server = ScriptedServer(
        ({"role": "assistant", "content": "Two profiles."}, usage(900, 30, cached=768))
    )

    calls = anyio.run(smoke.smoke, OpenAIChat(server.client, MODEL))

    assert len(calls) == len(server.bodies) == smoke.CALLS
    # Without vision, no figure.
    assert [call.kind for call in calls[:5]] == [
        "thinking",
        "no thinking",
        "parallel",
        "form",
        "thinking",
    ]
    assert all(call.error is None and call.cached_tokens == 768 for call in calls)
    printed = smoke.format_calls(calls).splitlines()
    assert printed[0].endswith("read 900 (768 cached), wrote 30, 0 tool call(s)")
    assert printed[3].endswith("read 900 (768 cached), wrote 30")
    thinking, quiet, parallel, form = server.bodies[:4]
    assert thinking["chat_template_kwargs"] == {"enable_thinking": True}
    assert quiet["chat_template_kwargs"] == {"enable_thinking": False}
    assert parallel["parallel_tool_calls"] is True
    assert form["response_format"]["json_schema"]["schema"] == smoke.FORM
    # The tools offered are the geophysicist's.
    assert [tool["function"]["name"] for tool in thinking["tools"]] == [
        tool.name for tool in anyio.run(geo_server.server.list_tools)
    ]


def test_with_vision_the_smoke_test_shows_the_model_a_figure() -> None:
    server = ScriptedServer()

    calls = anyio.run(smoke.smoke, OpenAIChat(server.client, MODEL, vision=True))

    assert calls[4].kind == "figure"
    question = server.bodies[4]["messages"][1]["content"]
    (picture,) = [part for part in question if part["type"] == "image_url"]
    assert picture["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_failed_call_is_kept_and_the_command_exits_with_status_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The server failing on the third call: the others go on.
    server = ScriptedServer(ANSWERED, ANSWERED, 500, ANSWERED)
    monkeypatch.setenv("PACO_LLM_BASE_URL", "http://vllm.test/v1")
    monkeypatch.setenv("PACO_LLM_MODEL", MODEL)

    def scripted(settings: ModelSettings) -> OpenAIChat:
        return OpenAIChat(server.client, settings.llm_model)

    monkeypatch.setattr(cli, "chat_model", scripted)

    with pytest.raises(SystemExit) as stopped:
        cli.main(["--smoke"])

    assert stopped.value.code == 1
    lines = capsys.readouterr().out.splitlines()
    assert "  3 parallel" in lines[3]
    assert "error: InternalServerError" in lines[3]
    assert lines[-1] == f"{smoke.CALLS} calls, 1 failed."


def test_the_command_says_which_settings_are_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("PACO_LLM_BASE_URL", "PACO_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit):
        cli.main(["--smoke"])

    assert capsys.readouterr().out.startswith("Set PACO_LLM_BASE_URL, PACO_LLM_MODEL in .env")
