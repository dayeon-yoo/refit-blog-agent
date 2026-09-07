from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, TypeVar

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
    def __init__(self, response_factory: Callable[[str, Type[BaseModel], Dict[str, Any]], Any] | None = None):
        self.response_factory = response_factory
        self.calls: List[Dict[str, Any]] = []

    def generate_structured(self, prompt: str, schema: Type[T], payload: Dict[str, Any]) -> T:
        if not payload:
            raise ValueError(f"Empty payload for schema {schema.__name__}")
        self.calls.append({"prompt": prompt, "schema": schema, "payload": payload})
        if self.response_factory is not None:
            response = self.response_factory(prompt, schema, payload)
        elif schema.__name__ == "BlogPost":
            response = self._default_blog_post(payload)
        else:
            response = payload
        return response if isinstance(response, schema) else schema.model_validate(response)

    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        if not payloads:
            return []
        return [schema.model_validate(item) for item in payloads]

    @staticmethod
    def _default_blog_post(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a topic-neutral response for local tests without an API call."""
        title = str(payload.get("title", "")).strip()
        keyword = str(payload.get("keyword", "")).strip()
        angle = str(payload.get("angle", "")).strip()
        summary = str(payload.get("summary", "")).strip()
        count = payload.get("required_item_count")

        intro = angle or summary or keyword or title
        sections = [f"## {title}", f"{intro}를 중심으로 실제로 적용할 수 있는 내용을 정리했습니다."]
        if isinstance(count, int) and count > 0:
            for index in range(1, count + 1):
                sections.extend([
                    f"## {index}. {title} 핵심 항목 {index}",
                    f"{intro}와 연결된 {index}번째 실천 항목입니다. 현재 상황에 맞게 준비하고 순서대로 적용해 보세요.",
                ])
        else:
            for index in range(1, 4):
                sections.extend([
                    f"## 핵심 포인트 {index}",
                    f"{intro}를 기준으로 {index}번째로 확인할 내용을 구체적으로 살펴보세요."
                    " 준비물과 현재 상태를 먼저 점검하면 시행착오를 줄일 수 있습니다.",
                ])
        sections.append("마지막으로 작은 범위에서 먼저 실천해 보고 자신에게 맞는 방법을 이어가 보세요.")
        return {
            "title": title,
            "keyword": keyword,
            "content": "\n\n".join(sections),
            "summary": summary or f"{title}에 대한 실천 방법을 정리했습니다.",
            "cta": "오늘 가능한 항목 하나부터 차근차근 실천해 보세요.",
        }


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
        if not payload:
            raise ValueError(f"Empty payload for schema {schema.__name__}")
        try:
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
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed for schema {schema.__name__}: {exc}") from exc

    def generate_many(self, prompt: str, schema: Type[T], payloads: List[Dict[str, Any]]) -> List[T]:
        if not payloads:
            return []
        return [self.generate_structured(prompt, schema, payload) for payload in payloads]


def get_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.mock_mode:
        return MockLLMClient()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing while MOCK_MODE=false")
    return OpenAILLMClient(api_key=settings.openai_api_key, model=settings.model_name)
