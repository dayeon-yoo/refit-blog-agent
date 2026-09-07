from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogPost, ScoredIdea


class WriterAgent:
    """Generate a BlogPost from a BlogIdea through the configured LLM client."""

    _NUMBER_PROMISE = re.compile(r"(?P<count>\d+)\s*(?:가지|개|단계)")
    _NUMBERED_HEADING = re.compile(r"^##\s+(?P<number>\d+)[.)]\s+(?P<label>.+?)\s*$", re.MULTILINE)
    _TOKEN = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9_-]{1,}")
    _FORBIDDEN = (
        "검색 의도",
        "seo",
        "prompt",
        "agent",
        "brand fit",
        "브랜드 평가",
        "critic",
        "ai 평가",
        "콘텐츠 생성 과정",
        "검색 유입을 위해",
    )

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "writer.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def write(self, idea: ScoredIdea) -> BlogPost:
        blog_idea = idea.idea
        promised_count = self._promised_count(blog_idea.title)
        payload = {
            "title": blog_idea.title,
            "keyword": blog_idea.keyword,
            "search_intent": blog_idea.search_intent,
            "angle": blog_idea.angle,
            "summary": blog_idea.summary,
            "rifit_connection": blog_idea.rifit_connection,
            "seasonality": blog_idea.seasonality,
            "content_contract": {
                "title": blog_idea.title,
                "keyword": blog_idea.keyword,
                "angle": blog_idea.angle,
                "summary": blog_idea.summary,
                "rifit_connection": blog_idea.rifit_connection,
            },
            "required_item_count": promised_count,
        }

        post = self.client.generate_structured(self.prompt, BlogPost, payload)
        self._validate_contract(blog_idea, post, promised_count)
        return post

    @classmethod
    def _promised_count(cls, title: str) -> Optional[int]:
        match = cls._NUMBER_PROMISE.search(title or "")
        return int(match.group("count")) if match else None

    @classmethod
    def _validate_contract(cls, idea, post: BlogPost, promised_count: Optional[int]) -> None:
        if post.title.strip() != idea.title.strip():
            raise ValueError("Writer output changed the BlogIdea title")
        if post.keyword.strip() != idea.keyword.strip():
            raise ValueError("Writer output changed the BlogIdea keyword")

        content = post.content.strip()
        if not content:
            raise ValueError("Writer output contains empty content")

        lowered = content.lower()
        leaked = [phrase for phrase in cls._FORBIDDEN if phrase in lowered]
        if leaked:
            raise ValueError(f"Writer output exposes internal metadata: {leaked[0]}")

        if not cls._contains_signal(content, idea.title):
            raise ValueError("Writer output does not reflect the BlogIdea title")
        if idea.angle and not cls._contains_signal(content, idea.angle):
            raise ValueError("Writer output does not reflect the BlogIdea angle")

        if promised_count is None:
            return

        headings = cls._NUMBERED_HEADING.findall(content)
        numbers = [int(number) for number, _ in headings]
        labels = [re.sub(r"\s+", " ", label.strip().lower()) for _, label in headings]
        expected = list(range(1, promised_count + 1))
        if numbers != expected:
            raise ValueError(
                f"Writer output must contain exactly {promised_count} numbered items; found {numbers}"
            )
        if len(set(labels)) != promised_count:
            raise ValueError("Writer output contains duplicate numbered items")

    @classmethod
    def _contains_signal(cls, content: str, source: str) -> bool:
        content_lower = content.lower()
        tokens = cls._TOKEN.findall(source.lower())
        meaningful = [token for token in tokens if not token.isdigit() and token not in {"방법", "하는", "법"}]
        if not meaningful:
            return True
        return any(token in content_lower for token in meaningful[:6])
