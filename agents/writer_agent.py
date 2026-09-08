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
    _CIRCULATION_TERMS = ("기부", "수거", "수거함", "의류순환", "재사용", "재활용")
    _ABSOLUTE_CLAIM = re.compile(r"(?:반드시|무조건|필수(?:입니다|예요|에요)?|할 수 없습니다|할 수 없어요)")
    _SPECULATIVE_DONATION_CLAIM = re.compile(
        r"(?:다른 사람에게|남에게).{0,20}(?:잘 맞|어울릴)|"
        r"(?:스타일|사이즈).{0,20}(?:맞을 가능성|적합할 가능성)"
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
        cls._validate_factual_claims(idea, content)
        if not cls._preserves_title_contrast(idea, content):
            raise ValueError("Writer output does not preserve the title's contrast")
        if not cls._preserves_rifit_connection(idea, content):
            raise ValueError("Writer output omits the relevant rifit connection")

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

    @classmethod
    def _has_circulation_context(cls, idea, content: str) -> bool:
        context = cls._compact(" ".join((idea.title, idea.angle, idea.summary, content)))
        return any(term in context for term in cls._CIRCULATION_TERMS)

    @classmethod
    def _validate_factual_claims(cls, idea, content: str) -> None:
        if not cls._has_circulation_context(idea, content):
            return
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", content):
            if cls._ABSOLUTE_CLAIM.search(sentence) and any(
                term in cls._compact(sentence) for term in ("기부", "세탁", "오염", "상태", "수거", "접수")
            ):
                raise ValueError("Writer output makes an unsupported absolute clothing-handling claim")
            if cls._SPECULATIVE_DONATION_CLAIM.search(sentence):
                raise ValueError("Writer output uses a speculative donation criterion")

    @classmethod
    def _contrast_requirements(cls, idea) -> list[tuple[str, str]]:
        values = (idea.title, idea.key_question or "", idea.angle)
        requirements: list[tuple[str, str]] = []
        for value in values:
            text = (value or "").lower()
            parts = re.split(r"\s+(?:대신|말고|vs\.?|대\s*비교)\s+", text, maxsplit=1)
            if len(parts) == 2:
                left_terms = cls._topic_terms(parts[0])
                right_terms = cls._topic_terms(parts[1])
                if left_terms and right_terms:
                    requirements.append((max(left_terms, key=len), max(right_terms, key=len)))
            if "비교" in text:
                match = re.search(r"(.+?)(?:와|과)\s+(.+?)\s+비교", text)
                if match:
                    left_terms = cls._topic_terms(match.group(1))
                    right_terms = cls._topic_terms(match.group(2))
                    if left_terms and right_terms:
                        requirements.append((max(left_terms, key=len), max(right_terms, key=len)))
        return requirements

    @classmethod
    def _preserves_title_contrast(cls, idea, content: str) -> bool:
        compact_content = cls._compact(content)
        return all(left in compact_content and right in compact_content for left, right in cls._contrast_requirements(idea))

    @classmethod
    def _preserves_rifit_connection(cls, idea, content: str) -> bool:
        connection = (idea.rifit_connection or "").strip()
        if not connection:
            return True
        connection_terms = cls._topic_terms(connection)
        contract_terms = set(cls._contract_terms(idea))
        distinctive = [term for term in connection_terms if term not in contract_terms and len(term) > 1]
        if not distinctive:
            return True
        compact_content = cls._compact(content)
        return any(term in compact_content for term in distinctive)
