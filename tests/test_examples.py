"""Each tool's example call and result (T11): one for every tool, the call valid as the server
reads its arguments, the result valid as the tool returns it."""

import typing

import pytest
from pydantic import TypeAdapter

from paco import server
from paco.examples import EXAMPLES

TOOLS = server.server._tool_manager.list_tools()  # pyright: ignore[reportPrivateUsage]


def test_every_tool_has_an_example() -> None:
    assert set(EXAMPLES) == {tool.name for tool in TOOLS}


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_an_examples_call_and_result_are_the_tools(tool: typing.Any) -> None:  # noqa: ANN401
    example = EXAMPLES[tool.name]

    # Its arguments the tool's own (the server's model lets unknown ones through), valid.
    assert set(example.arguments) <= set(tool.parameters["properties"])
    tool.fn_metadata.arg_model.model_validate(example.arguments)
    returns = typing.get_type_hints(tool.fn)["return"]
    TypeAdapter(returns).validate_python(example.result)
