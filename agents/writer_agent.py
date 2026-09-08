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
        "키워드 선정 이유",
        "seo",
        "prompt",
        "agent",
        "brand fit",
        "brand_fit",
        "브랜드 평가",
        "fit_score",
        "ranking",
        "metadata",
        "workflow",
        "critic",
        "ai 평가",
        "콘텐츠 생성 과정",
        "검색 유입을 위해",
    )
    _FORMAT_WORDS = {
        "방법", "하는", "법", "팁", "추천", "이유", "비교", "단계", "가지", "개",
        "정리", "안내", "정보", "내용", "중심", "실천", "가이드",
        "how", "to", "the", "and",
    }
    _SUFFIXES = (
        "으로써", "으로서", "에게서", "에게로", "하면서", "하며", "하고",
        "하는", "해야", "할", "으로", "에서", "에게", "까지", "부터",
        "을", "를", "은", "는", "이", "가", "의", "에", "로", "와", "과",
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
            "content_format": blog_idea.content_format,
            "content_perspective": blog_idea.content_perspective,
            "key_question": blog_idea.key_question,
            "outline": blog_idea.outline or [],
            "seasonality": blog_idea.seasonality,
            "content_contract": {
                "title": blog_idea.title,
                "keyword": blog_idea.keyword,
                "angle": blog_idea.angle,
                "summary": blog_idea.summary,
                "rifit_connection": blog_idea.rifit_connection,
                "content_format": blog_idea.content_format,
                "content_perspective": blog_idea.content_perspective,
                "key_question": blog_idea.key_question,
                "outline": blog_idea.outline or [],
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
        if post.keyword.strip() != idea.keyword.strip():
            raise ValueError("Writer output changed the BlogIdea keyword")

        content = post.content.strip()
        if not content:
            raise ValueError("Writer output contains empty content")

        output_text = " ".join((post.title, post.content, post.summary, post.cta)).lower()
        leaked = [phrase for phrase in cls._FORBIDDEN if phrase in output_text]
        if leaked:
            raise ValueError(f"Writer output exposes internal metadata: {leaked[0]}")

        if not cls._title_matches_contract(idea, post.title):
            raise ValueError("Writer output title does not reflect the BlogIdea topic")
        if not cls._content_reflects_contract(idea, content):
            raise ValueError("Writer output does not reflect the BlogIdea title")

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
        source_terms = cls._topic_terms(source)
        if not source_terms:
            return True
        content_lower = cls._compact(content)
        hits = sum(1 for term in source_terms if term in content_lower)
        required = 1 if len(source_terms) == 1 else 2
        return hits >= min(required, len(source_terms))

    @classmethod
    def _compact(cls, value: str) -> str:
        return re.sub(r"[^가-힣A-Za-z0-9]", "", (value or "").lower())

    @classmethod
    def _topic_terms(cls, value: str) -> list[str]:
        terms: list[str] = []
        for raw in cls._TOKEN.findall((value or "").lower()):
            term = raw
            changed = True
            while changed and len(term) > 1:
                changed = False
                for suffix in cls._SUFFIXES:
                    if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                        term = term[:-len(suffix)]
                        changed = True
                        break
            if term.isdigit() or term in cls._FORMAT_WORDS or len(term) < 2:
                continue
            if term not in terms:
                terms.append(term)
        return terms

    @classmethod
    def _contract_terms(cls, idea) -> list[str]:
        terms: list[str] = []
        for value in (idea.keyword, idea.title, idea.summary, idea.angle):
            for term in cls._topic_terms(value):
                if term not in terms:
                    terms.append(term)
        return terms

    @classmethod
    def _title_matches_contract(cls, idea, output_title: str) -> bool:
        output = cls._compact(output_title)
        if not output:
            return False

        title_terms = cls._topic_terms(idea.title)
        keyword_terms = cls._topic_terms(idea.keyword)
        contract_terms = cls._contract_terms(idea)
        if not contract_terms:
            return True

        title_hits = sum(1 for term in title_terms if term in output)
        keyword_hits = sum(1 for term in keyword_terms if term in output)
        # A title may contain only one distinctive term after format words
        # are removed, but it must still agree with the source keyword.
        return title_hits >= 2 or (title_hits >= 1 and keyword_hits >= 1)

    @classmethod
    def _content_reflects_contract(cls, idea, content: str) -> bool:
        source_terms = cls._contract_terms(idea)
        if not source_terms:
            return True
        compact_content = cls._compact(content)
        hits = sum(1 for term in source_terms if term in compact_content)
        required = 1 if len(source_terms) == 1 else 2
        return hits >= min(required, len(source_terms))
