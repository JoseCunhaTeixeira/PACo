"""Runtime settings, read from environment variables prefixed with ``PACO_`` or from a ``.env`` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import DirectoryPath, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The images' cache, in the output folder by default.
CACHE_FOLDER = ".cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        # .env also holds the agent's PACO_LLM_* settings (paco.agent), which are not the server's.
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    # Profiles are read where PAC keeps them, so PAC and PACo process the same inputs.
    input_dir: DirectoryPath = Path("/data/input")

    # PACo writes its own results, never into PAC's data folder.
    output_dir: Path = Path("data/output")

    # Windows a run processes in parallel, one worker process each.
    workers: int = Field(default=1, ge=1)

    # Where the MCP server listens (paco-server). 127.0.0.1 keeps it on this machine; in a
    # container it listens on 0.0.0.0, and the container's port is published on 127.0.0.1 only.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    # Host headers the server accepts, e.g. ["paco-server:*", "localhost:*"]: protection against
    # DNS rebinding, which the SDK only turns on by itself when the host is 127.0.0.1.
    allowed_hosts: tuple[str, ...] = ()
    # The QC thresholds and retry budgets, a JSON file (paco.qc.QCConfig); PACo's defaults if
    # none. Changed between runs only: a run records the values it used.
    qc_config: Path | None = None
    # The windows' images kept, to be taken again when a call makes the same image (the same
    # code, records and settings: sigpipe.masw.runs.caching): where (None: <output_dir>/.cache),
    # and how many GB at most, the images used longest ago removed first; 0 keeps none.
    cache_dir: Path | None = None
    cache_gb: float = Field(default=2.0, ge=0)
    # The largest difference paco-replay allows between a run's value and the replay's, relative
    # to the run's largest value: records and images are float32, whose resolution is 1.2e-7;
    # the margin is for the sums of a phase shift done in another order, on another machine.
    replay_tolerance: float = Field(default=1e-5, gt=0)

    @field_validator("input_dir", "output_dir")
    @classmethod
    def _resolve(cls, path: Path) -> Path:
        # Relative paths are resolved once, against the directory PACo is started from.
        return path.resolve()

    @property
    def cache_folder(self) -> Path:
        return (self.cache_dir or self.output_dir / CACHE_FOLDER).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
