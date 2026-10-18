"""The user's side of the agent, in a terminal: the chat. No tool asks the user anything: the go
or no-go before an inversion is G4's verdict (docs/qc_workflow.md)."""

import argparse
import logging
from collections.abc import Sequence

import anyio
import anyio.to_thread
from mcp import Client
from pydantic import ValidationError

from paco import logs
from paco.agent.loop import Agent, Limits
from paco.agent.record import save_transcript
from paco.agent.settings import AgentSettings
from paco.llm.chat import chat_model
from paco.llm.settings import NO_MODEL, missing
from paco.modes import MODES, ModeSettings


def main(argv: Sequence[str] | None = None) -> None:
    """paco-agent [--mode guided|autonomous]: the chat with the mode's agent (PACO_AGENT_MODE when
    not told), in this terminal; its log lines as JSON on stderr, only warnings."""
    parser = argparse.ArgumentParser(
        prog="paco-agent", description="Chat with PACo's agent in this terminal."
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="guided: PACo's workflow, its server leading the model step by step; autonomous: "
        "the geophysicist, which chooses and tries its own settings, on a frontier model with a "
        "long context (default: PACO_AGENT_MODE, else guided)",
    )
    mode = parser.parse_args(argv).mode or ModeSettings().agent_mode
    logs.setup(logging.WARNING)
    if mode == "autonomous":
        # The geophysicist's chat (paco.geo), imported only when it runs: its own server's tools.
        from paco.geo.host import terminal
        from paco.geo.settings import GeoSettings

        anyio.run(terminal.chat, GeoSettings().mcp_url)
        return
    anyio.run(chat)


async def chat() -> None:
    """Chat with the agent until the user types exit. PACo's server must be running."""
    try:
        settings = AgentSettings()  # pyright: ignore[reportCallIssue]  # fields come from .env
    except ValidationError as error:
        print(NO_MODEL.format(missing=missing(error)))
        return
    try:
        model = chat_model(settings)
    except ValueError as error:  # a cloud's library or credentials missing
        print(error)
        return
    async with Client(settings.mcp_url) as server:
        agent = await Agent.start(server, model, limits=Limits.of(settings))
        print(f"PACo's agent, with {settings.llm_model}. Type exit to leave.")
        try:
            while True:
                try:
                    question = (await anyio.to_thread.run_sync(input, "\nyou> ")).strip()
                except EOFError, KeyboardInterrupt:
                    return
                if question in ("exit", "quit"):
                    return
                if question:
                    print(f"\npaco> {await agent.answer(question)}")
        finally:
            if agent.steps:
                path = save_transcript(agent.transcript(settings.llm_model), settings.logs)
                print(f"\nConversation saved in {path}")
