"""PACo's two ways to use the model, and which one runs: PACO_AGENT_MODE, in the environment or a
.env file, read by paco-agent, paco-server and PAC's chat when they are not told. Both modes talk
to the same model (PACO_LLM_*), through the same server address (PACO_MCP_URL)."""

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# workflow: PACo's workflow, its server leading the model one tool at a time; made for a small
# model on your GPU. autonomous: the model surveys the line, tries its own settings and judges
# them with the gates; made for a frontier model with a long context (128k tokens), and it uses
# many more tokens a line.
type Mode = Literal["workflow", "autonomous"]
MODES: tuple[Mode, ...] = ("workflow", "autonomous")


class ModeSettings(BaseSettings):
    """Which mode runs, PACO_AGENT_MODE, and where the chat finds its server, PACO_MCP_URL."""

    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    agent_mode: Mode = "workflow"
    # paco-server, whichever mode it serves.
    mcp_url: str = "http://127.0.0.1:8000/mcp"

    @field_validator("agent_mode", mode="before")
    @classmethod
    def _empty_is_workflow(cls, value: object) -> object:
        # An empty setting is one not set: compose passes one for a variable .env leaves out.
        return "workflow" if value == "" else value
