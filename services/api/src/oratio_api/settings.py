"""Runtime configuration, read once from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORATIO_", extra="ignore")

    model_dir: Path = Path("/opt/oratio/models")
    cors_origins: list[str] = Field(default_factory=list)
    cors_origin_regex: str | None = None
    enable_spike: bool = False
    image_digest: str = "unknown"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
