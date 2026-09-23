"""The user's side of the agent, in a terminal: questions from PACo's tools, and the chat."""

import anyio
import anyio.to_thread
from mcp import Client
from mcp.client.session import ClientRequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from openai import AsyncOpenAI
from pydantic import ValidationError

from paco.agent.loop import Agent
from paco.agent.model import OpenAIChat
from paco.agent.record import save_transcript
from paco.agent.settings import AgentSettings


async def ask_the_user(
    context: ClientRequestContext,  # noqa: ARG001
    params: ElicitRequestParams,
) -> ElicitResult:
    """A question a tool asks the user (an approval): it goes to the terminal, never to the model."""
    print(f"\n[PACo asks you] {params.message}")
    answer = await anyio.to_thread.run_sync(input, "Approve? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        return ElicitResult(action="decline")
    # Yes to every yes-or-no field the question has.
    schema = getattr(params, "requested_schema", {}) or {}
    properties = schema.get("properties", {})
    content: dict[str, str | int | float | bool | list[str] | None] = {
        name: True for name, field in properties.items() if field.get("type") == "boolean"
    }
    return ElicitResult(action="accept", content=content)


def main() -> None:
    """paco-agent: the chat, in this terminal."""
    anyio.run(chat)


async def chat() -> None:
    """Chat with the agent until the user types exit. PACo's server must be running."""
    try:
        settings = AgentSettings()  # pyright: ignore[reportCallIssue]  # fields come from .env
    except ValidationError as error:
        missing = ", ".join(f"PACO_{problem['loc'][0]}".upper() for problem in error.errors())
        print(f"Set {missing} in .env (vLLM's address and model name).")
        return
    client = AsyncOpenAI(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key.get_secret_value()
    )
    model = OpenAIChat(client, settings.llm_model)
    async with Client(settings.mcp_url, elicitation_callback=ask_the_user) as server:
        agent = await Agent.start(server, model, settings.max_tool_calls)
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
                path = save_transcript(agent.transcript(settings.llm_model), settings.log_dir)
                print(f"\nConversation saved in {path}")
