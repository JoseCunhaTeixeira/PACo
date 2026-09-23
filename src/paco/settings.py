"""Runtime settings, read from environment variables prefixed with ``PACO_`` or from a ``.env`` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import DirectoryPath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PACO_",
        env_file=".env",
        frozen=True,
        validate_default=True,
    )

    # Profiles are read where PAC keeps them, so PAC and PACo process the same inputs.
    input_dir: DirectoryPath = Path("/data/input")

    # PACo writes its own results, never into PAC's data folder.
    output_dir: Path = Path("data/output")

    @field_validator("input_dir", "output_dir")
    @classmethod
    def _resolve(cls, path: Path) -> Path:
        # Relative paths are resolved once, against the directory PACo is started from.
        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
