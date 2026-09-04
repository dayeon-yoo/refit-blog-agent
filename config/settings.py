from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    openai_api_key: str = Field(default="")
    model_name: str = Field(default="gpt-4o-mini")
    environment: str = Field(default="local")
    debug: bool = Field(default=True)
    mock_mode: bool = Field(default=True)
    idea_count: int = Field(default=10)
    top_k: int = Field(default=3)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        environment=os.getenv("APP_ENV", "local"),
        debug=os.getenv("DEBUG", "true").lower() == "true",
        mock_mode=os.getenv("MOCK_MODE", "true").lower() not in {"false", "0", "no"},
        idea_count=_get_int_env("IDEA_COUNT", 10),
        top_k=_get_int_env("TOP_K", 3),
    )
