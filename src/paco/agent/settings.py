"""What PACo's agents read from the environment or .env: the model (PACO_LLM_*, paco.llm.settings),
the mode and its server (paco.modes), and the output folder, where they write."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from paco.llm.settings import ModelSettings
from paco.modes import ModeSettings


class OutputSettings(BaseSettings):
    """PACo's output folder (PACO_OUTPUT_DIR, the server's own): the chat saves its conversations
    in agent_logs, the evaluations write in evaluations."""

    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    output_dir: Path = Path("/data/output")

    @property
    def logs(self) -> Path:
        return self.output_dir / "agent_logs"

    @property
    def evaluations(self) -> Path:
        return self.output_dir / "evaluations"


class AgentSettings(ModelSettings, ModeSettings, OutputSettings):
    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        # .env also holds the server's settings (paco.settings), which are not the agent's.
        extra="ignore",
        frozen=True,
    )
