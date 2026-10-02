"""paco-call TOOL [ARGUMENTS]: one of PACo's tools called without a model (A9), in this process,
as a host calls it; ARGUMENTS a JSON object, {} when left out. Prints what the tool returns: its
object as JSON, or its text; an error, on stderr, with exit status 1. The call is traced in the
run's agent_calls.jsonl as conversation "paco-call". `paco-call --list` names the tools, each
with an example call (paco.examples)."""

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

import anyio
from mcp import Client
from mcp.types import CallToolResult, RequestParamsMeta

from paco import server
from paco.examples import EXAMPLES

# The conversation the calls are traced under.
CONVERSATION = "paco-call"


async def call(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Tool `name` called with `arguments` through an in-memory client."""
    async with Client(server.server) as client:
        meta: RequestParamsMeta = {"conversation": CONVERSATION}
        return await client.call_tool(name, arguments, meta=meta)


def shown(result: CallToolResult) -> str:
    """What the tool returned: its object as JSON, else its text (which the SDK also sends
    wrapped as {"result": text})."""
    structured = result.structured_content
    if structured is not None and not isinstance(structured.get("result"), str):
        return json.dumps(structured, indent=2)
    return "\n".join(content.text for content in result.content if content.type == "text")


def listing() -> str:
    """Each tool by name, with its example call."""
    tools = anyio.run(server.server.list_tools)
    return "\n".join(
        f"{tool.name} '{json.dumps(EXAMPLES[tool.name].arguments)}'"
        if tool.name in EXAMPLES
        else tool.name
        for tool in tools
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="paco-call", description=__doc__)
    parser.add_argument("tool", nargs="?", help="the tool's name; --list names them")
    parser.add_argument("arguments", nargs="?", default="{}", help="a JSON object")
    parser.add_argument("--list", action="store_true", help="the tools, with an example each")
    parsed = parser.parse_args(argv)
    if parsed.list:
        print(listing())
        return
    if parsed.tool is None:
        parser.error("name a tool, or --list")
    try:
        arguments = json.loads(parsed.arguments)
    except json.JSONDecodeError as error:
        parser.error(f"ARGUMENTS is not JSON: {error}")
    if not isinstance(arguments, dict):
        parser.error('ARGUMENTS is a JSON object, e.g. \'{"what": "profiles"}\'')
    result = anyio.run(call, parsed.tool, arguments)
    if result.is_error:
        sys.exit(shown(result))
    print(shown(result))


if __name__ == "__main__":
    main()
