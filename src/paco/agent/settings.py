"""How the agent reaches the model and PACo's server: PACO_LLM_* and PACO_MCP_URL, from the
environment or .env."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        # .env also holds the server's settings (paco.settings), which are not the agent's.
        extra="ignore",
        frozen=True,
    )

    # The model, behind an OpenAI-compatible API such as vLLM's: no default, so that a missing
    # setting fails here, with its name.
    llm_base_url: str = Field(description="e.g. http://gpu-host:8001/v1")
    llm_model: str = Field(description="The name vLLM serves the model under.")
    # vLLM started without --api-key accepts any key.
    llm_api_key: SecretStr = SecretStr("EMPTY")

    mcp_url: str = "http://127.0.0.1:8000/mcp"  # PACo's server (python -m paco.server)
    # Tool calls the model may make to answer one message, before the loop stops it.
    max_tool_calls: int = Field(default=15, ge=1)
    # Where the chat saves its conversations, one JSON file each.
    log_dir: Path = Path("data/output/agent_logs")

    # The evaluation's judge model (python -m paco.evaluation): optional; without it, only the
    # rule checks score the scenarios. Better another model than the one judged.
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: SecretStr = SecretStr("EMPTY")
    # Where evaluations write their report, transcripts and runs.
    evaluation_dir: Path = Path("data/output/evaluations")
