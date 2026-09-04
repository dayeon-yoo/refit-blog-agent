from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from config.settings import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[T], payload: Dict[str, Any]) -> T:
        raise NotImplementedError

    @abstractmethod
    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    def generate_structured(self, prompt: str, schema: Type[T], payload: Dict[str, Any]) -> T:
        return schema.model_validate(payload)

    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        return [schema.model_validate(item) for item in payloads]


class LocalFallbackLLM(MockLLMClient):
    pass


class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when MOCK_MODE=false")
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)

    def generate_structured(self, prompt: str, schema: Type[T], payload: Dict[str, Any]) -> T:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed result for the schema")
        return parsed

    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        return [self.generate_structured(prompt, schema, payload) for payload in payloads]


def get_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.mock_mode:
        return MockLLMClient()
    return OpenAILLMClient(api_key=settings.openai_api_key, model=settings.model_name)
