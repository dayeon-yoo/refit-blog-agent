from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    openai_api_key: str = Field(default="")
    model_name: str = Field(default="gpt-4o-mini")
    environment: str = Field(default="local")
    debug: bool = Field(default=True)
    mock_mode: bool = Field(default=True)

    class Config:
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        environment=os.getenv("APP_ENV", "local"),
        debug=os.getenv("DEBUG", "true").lower() == "true",
        mock_mode=os.getenv("MOCK_MODE", "true").lower() not in {"false", "0", "no"},
    )
