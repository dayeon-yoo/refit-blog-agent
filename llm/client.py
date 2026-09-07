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
        rifit_connection = str(payload.get("rifit_connection", "")).strip()

        intro = angle or summary or keyword or title
        sections = [
            f"## {title}",
            f"{title}을 고민하다 보면, 막상 어디서부터 시작해야 할지 망설여질 때가 있는데요."
            f" {intro}를 기준으로 지금 할 수 있는 작은 방법부터 살펴보겠습니다.",
        ]
        if isinstance(count, int) and count > 0:
            for index in range(1, count + 1):
                lead = [
                    "먼저 현재 상태를 가볍게 확인해 보세요.",
                    "이 단계에서는 준비물을 많이 늘리기보다 집에 있는 것을 먼저 살펴보면 좋습니다.",
                    "실제로 해보면 생각보다 간단한 부분부터 손이 가는데요.",
                ][(index - 1) % 3]
                sections.extend([
                    f"## {index}. {title} 핵심 항목 {index}",
                    f"{lead} {intro}와 연결된 {index}번째 방법은 현재 상황에 맞게 순서를 조절해 적용해 보세요.",
                ])
        else:
            for index in range(1, 4):
                sections.extend([
                    f"## {title}을 시작하기 전에 확인할 점 {index}",
                    f"{intro}를 살펴볼 때는 한 번에 모두 바꾸려 하지 말고, {index}번째로 눈에 들어오는 부분부터 확인해 보세요."
                    " 준비물과 현재 상태를 먼저 점검하면 시행착오를 줄일 수 있습니다.",
                ])
        cta = "오늘은 가장 부담 없는 항목 하나만 골라 실제로 해보세요."
        if rifit_connection:
            cta = f"{rifit_connection}이 필요한 순간이라면, 오늘 정리한 기준을 먼저 적용해 보세요."
        sections.append(cta)
        return {
            "title": title,
            "keyword": keyword,
            "content": "\n\n".join(sections),
            "summary": summary or f"{title}에 대한 실천 방법을 정리했습니다.",
            "cta": cta,
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
