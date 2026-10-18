"""The geophysicist's chat in a terminal (paco-agent --mode autonomous). It reaches its server
(paco-server --mode autonomous), lists the tools the model will be given, then runs the
conversation with the model of .env (PACO_LLM_*) until the user types exit. The study keeps all
the work: stopped (Ctrl+C), the next chat on the same profile resumes it (docs/geo_plan.md §14)."""

import sys
from collections.abc import Callable

import anyio
import anyio.to_thread
from mcp import Client
from mcp.server import MCPServer
from mcp.types import Tool
from pydantic import ValidationError

from paco.agent.model import ChatModel
from paco.geo.host.loop import Host, Limits, short_context
from paco.llm.chat import chat_model, seeing
from paco.llm.settings import NO_MODEL, ModelSettings, missing
from paco.settings import get_settings


async def chat(
    server: str | MCPServer,
    model: ChatModel | None = None,
    read: Callable[[str], str] = input,
) -> None:
    """The chat with the geophysicist's server at `server` (or, in a test, the server itself): its
    tools, then the conversation with `model` (the one .env names when none is given), each line
    the user types read with `read`. Stops with exit status 1, saying why, when the server gives
    no tools."""
    try:
        tools = await listed(server)
    except Exception as error:
        sys.exit(
            f"No tools from the geophysicist's server at {server} ({_innermost(error)}): is it "
            "running? Start it with `uv run paco-server --mode autonomous`."
        )
    print(f"The geophysicist's server offers {len(tools)} tools:")
    for tool in tools:
        print(f"  {tool.name}")
    limits, vision = Limits(), False
    if model is None:
        try:
            settings = ModelSettings()  # pyright: ignore[reportCallIssue]  # fields come from .env
        except ValidationError as error:
            print(NO_MODEL.format(missing=missing(error)))
            return
        try:
            model = chat_model(settings)
        except ValueError as error:  # a cloud's library or credentials missing
            print(error)
            return
        if (short := short_context(settings.context())) is not None:
            print(short)
            return
        if (said := await seeing(model)) is not None:
            print(said)
        limits, vision = Limits.of(settings), model.vision
    async with Client(server) as client:
        host = await Host.start(client, model, get_settings(), limits=limits, vision=vision)
        print("The geophysicist agent. Say which profile, and what you want; exit to leave.")
        while True:
            try:
                message = (await anyio.to_thread.run_sync(read, "\nyou> ")).strip()
            except EOFError, KeyboardInterrupt:
                return
            if message in ("exit", "quit"):
                return
            if message:
                print(f"\npaco> {await host.answer(message)}")


async def listed(server: str | MCPServer) -> list[Tool]:
    """The tools `server` offers, as the model will be given them."""
    async with Client(server) as client:
        return (await client.list_tools()).tools


def _innermost(error: BaseException) -> str:
    """The error under the task groups that wrap it: its kind and its message."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"
