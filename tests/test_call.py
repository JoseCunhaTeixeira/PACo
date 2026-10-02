"""paco-call: a tool called without a model, from the command line (A9)."""

import json

import pytest

from paco import call, server
from paco.examples import EXAMPLES
from paco.settings import Settings


@pytest.mark.usefixtures("paco_env")
def test_a_tool_is_called_and_what_it_returns_printed(capsys: pytest.CaptureFixture[str]) -> None:
    call.main(["inspect", '{"what": "profiles"}'])
    listed = capsys.readouterr().out

    call.main(["preset_settings", '{"profile": "active_p1"}'])
    schema = json.loads(capsys.readouterr().out)

    assert listed.startswith("active_p1: a profile to process, no run yet")
    assert "masw" in schema["properties"]


def test_a_run_changed_from_the_command_line_says_so_in_its_trace(
    paco_env: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    # The muting given: no mute trial, the test's run short.
    overrides = {"masw": {"length": 24, "step": 24}, "muting": {"method": "none"}}
    arguments = {"profile": "active_p1", "overrides": overrides}

    call.main(["run_processing", json.dumps(arguments)])

    result = json.loads(capsys.readouterr().out)
    (trace,) = paco_env.output_dir.glob(f"*/{result['run_id']}/agent_calls.jsonl")
    assert json.loads(trace.read_text())["conversation"] == call.CONVERSATION


@pytest.mark.usefixtures("paco_env")
def test_an_error_goes_to_stderr_with_status_1(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        call.main(["pick", '{"run_id": "20261001-120000-abcd"}'])

    assert stopped.value.code == (
        "Error executing tool pick: [precondition] Unknown run '20261001-120000-abcd'. Latest "
        "runs: none."
    )
    with pytest.raises(SystemExit):
        call.main(["inspect", "[1, 2]"])
    assert "ARGUMENTS is a JSON object" in capsys.readouterr().err


def test_the_list_names_every_tool_with_its_example(capsys: pytest.CaptureFixture[str]) -> None:
    call.main(["--list"])

    lines = capsys.readouterr().out.splitlines()
    tools = server.server._tool_manager.list_tools()  # pyright: ignore[reportPrivateUsage]
    assert [line.split(" ")[0] for line in lines] == [tool.name for tool in tools]
    assert f"inspect '{json.dumps(EXAMPLES['inspect'].arguments)}'" in lines
