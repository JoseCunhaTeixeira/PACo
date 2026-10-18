"""How PACo's agents reach their model and their server: the model, PACO_LLM_*, and the
evaluations' judge, PACO_JUDGE_* (paco.llm.settings: any provider), PACo's server, PACO_MCP_URL,
the caps on an answer, and where the agents write, from the environment or .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from paco.llm.settings import JudgeSettings, ModelSettings


class OutputSettings(BaseSettings):
    """Where the agents write: PACo's output folder (PACO_OUTPUT_DIR, the server's own), and the
    folders set apart from it."""

    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    output_dir: Path = Path("/data/output")
    # Where the chat saves its conversations, one JSON file each: <output>/agent_logs unless set.
    log_dir: Path | None = None
    # Where evaluations write their report, transcripts and runs: <output>/evaluations unless set.
    evaluation_dir: Path | None = None

    @property
    def logs(self) -> Path:
        return self.log_dir or self.output_dir / "agent_logs"

    @property
    def evaluations(self) -> Path:
        return self.evaluation_dir or self.output_dir / "evaluations"


class AgentSettings(ModelSettings, JudgeSettings, OutputSettings):
    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        # .env also holds the server's settings (paco.settings), which are not the agent's.
        extra="ignore",
        frozen=True,
    )

    mcp_url: str = "http://127.0.0.1:8000/mcp"  # PACo's server (python -m paco.server)
    # Tool calls the model may make to answer one message, before the loop stops it.
    max_tool_calls: int = Field(default=15, ge=1)
    # The caps on one answer (the model's tokens include its thinking) and one tool call's
    # timeout: a whole line inverted at PAC's effort takes about 40 min on 8 cores.
    max_turn_s: float = Field(default=7200.0, gt=0)
    max_turn_tokens: int = Field(default=40_000, ge=1)
    tool_timeout_s: float = Field(default=3600.0, gt=0)
