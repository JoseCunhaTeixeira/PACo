"""The geophysicist agent's own settings, PACO_GEO_* in the environment or a .env file: where its
server listens, where the chat finds it, and where its evaluations go. The model is both agents'
(PACO_LLM_*, paco.llm.settings), the evaluations' judge too (PACO_JUDGE_*); the folders and the
workers stay PACo's settings (PACO_INPUT_DIR, PACO_OUTPUT_DIR, PACO_WORKERS: paco.settings), so
that both agents work on the same data from one .env (docs/geo_plan.md §3)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeoSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PACO_GEO_",
        env_file=".env",
        # .env also holds PACo's settings (PACO_*) and the model's (PACO_LLM_*).
        extra="ignore",
        frozen=True,
    )

    # Where the geophysicist's server (paco-server --mode autonomous) listens: this machine only,
    # beside PACo's guided server (8000), a local vLLM (8001) and the tunnel to a model on Cloud
    # Run (8002), so that both agents run side by side.
    host: str = "127.0.0.1"
    port: int = Field(default=8010, ge=1, le=65_535)
    # Where the chat (paco-agent --mode autonomous) finds it.
    mcp_url: str = "http://127.0.0.1:8010/mcp"
    # Where paco-ab writes its evaluations (ab-<time>): by default, the evaluations folder of
    # PACo's output folder (PACO_OUTPUT_DIR), beside PACo's own (eval-<time>).
    evaluation_dir: Path | None = None
