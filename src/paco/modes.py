"""PACo's two agents, and which one a chat runs: PACO_AGENT_MODE in the environment or a .env file,
read by paco-agent, paco-server and PAC's chat when they are not told."""

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The two agents a chat runs (PACO_AGENT_MODE). guided: PACo's workflow, which its server's tools
# and instructions lead step by step, made for a small model with a short context (Qwen3 on the
# user's GPU). autonomous: the geophysicist (paco.geo), which chooses the settings, runs its own
# experiments and judges them by the gates, made for a frontier model with a long context (128k
# tokens), at many more tokens a line.
type Mode = Literal["guided", "autonomous"]
MODES: tuple[Mode, ...] = ("guided", "autonomous")


class ModeSettings(BaseSettings):
    """Which agent the chat runs, PACO_AGENT_MODE."""

    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    agent_mode: Mode = "guided"

    @field_validator("agent_mode", mode="before")
    @classmethod
    def _empty_is_guided(cls, value: object) -> object:
        # An empty setting is one not set: compose passes one for a variable .env leaves out.
        return "guided" if value == "" else value
