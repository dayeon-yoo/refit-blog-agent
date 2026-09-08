from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from agents.content_planner_agent import ContentPlannerAgent
from agents.editorial_context import build_safe_editorial_context
from llm.client import LLMClient, get_llm_client
from models.schemas import BlogPost, ContentPlan, ScoredIdea


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
    _PERSONAL_EXPERIENCE_CLAIM = re.compile(
        r"(?:저는|제가|내가|직접|돌아다니며|골랐|선택했|입었|착용했|샀|구매했|해봤|해보니)"
    )
    _EXPERIENCE_DETAIL_MARKERS = (
        "샵", "숍", "매장", "가게", "돌아다니", "화이트", "그린", "검정", "블랙",
        "브라운", "네이비", "레드", "블루", "원단", "가죽", "데님", "울",
        "면소재", "실크", "슬랙스", "로퍼", "목걸이", "귀걸이", "스니커즈", "재킷", "자켓",
    )
    _EXPERIENCE_EVENT_MARKERS = (
        "인턴으로", "근무", "일하고", "디자인회사", "회사에서", "돌아다니",
        "미팅", "회의", "행사", "참석", "주변반응", "긍정적반응", "칭찬",
        "자존감", "자신감", "만족했", "만족감을", "경제적이득", "경제적효과", "절약했",
        "80년대", "90년대", "2000년대", "높아졌다", "이득을 봤",
    )
    # Require a word boundary so the final syllable of words such as "목걸이"
    # is not mistaken for the pronoun "이" in a general styling suggestion.
    _OWNED_ITEM_CLAIM = re.compile(
        r"(?<![가-힣A-Za-z0-9])(?:이|그|내)\s+[^.!?。！？\n]{1,30}(?:은|는)"
        r"|(?<![가-힣A-Za-z0-9])제가\s+(?:산|고른)\s*[^.!?。！？\n]{0,30}(?:은|는)"
    )
    _SUGGESTION_CONTEXT = (
        "해볼 수", "할 수", "추천", "라면", "경우", "방법도", "활용해보", "매치해보",
    )
    _EXPERIENCE_CLAIM_GRAMMAR = re.compile(
        r"(?:했(?:습니다|어요|는데요)?|였(?:습니다|어요)?|고민했|느꼈)"
    )

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "writer.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def write(
        self,
        idea: ScoredIdea,
        raw_source: str = "",
        content_plan: ContentPlan | None = None,
    ) -> BlogPost:
        blog_idea = idea.idea
        promised_count = self._promised_count(blog_idea.title)
        planning_mode = ContentPlannerAgent.planning_mode(raw_source, blog_idea)
        editorial_context = build_safe_editorial_context(blog_idea, raw_source)
        safe_angle = blog_idea.angle if not raw_source.strip() else " ".join(
            value for value in (
                editorial_context.get("title"),
                editorial_context.get("keyword"),
                editorial_context.get("content_format"),
            ) if value
        )
        safe_rifit_direction = str(editorial_context.get("rifit_editorial_direction", ""))
        payload = {
            "title": blog_idea.title,
            "keyword": blog_idea.keyword,
            # Legacy keys contain only safe editorial text for local clients.
            "angle": safe_angle,
            # The mock client uses this legacy field to shape local output;
            # keep it derived from safe metadata rather than BlogIdea.summary.
            "summary": raw_source.strip() or safe_angle,
            "rifit_connection": safe_rifit_direction,
            "content_format": blog_idea.content_format,
            "target_reader": blog_idea.target_reader,
            "raw_source": raw_source,
            "planning_mode": planning_mode,
            "editorial_context": editorial_context,
            "refined_blog_idea": editorial_context,
            "content_plan": content_plan.model_dump(mode="json") if content_plan else None,
            "seasonality": blog_idea.seasonality,
            "content_contract": {
                **editorial_context,
                "raw_source": raw_source,
                "writing_script": content_plan.writing_script if content_plan else "",
            },
            "required_item_count": promised_count,
        }

        generation_prompt = self.prompt
        if planning_mode == "source_grounded_editorial":
            generation_prompt += (
                "\n\nWRITING MODE: SOURCE-GROUNDED EDITORIAL. "
                "Use raw_source as the only authority for facts, claims, relationships, "
                "and actions. Use writing_script only for ordering. Rephrase and connect "
                "the source's own propositions, but do not add external statistics, "
                "environmental effects, industry explanations, shopping tips, product "
                "examples, or new reader actions. Preserve subject/object relationships: "
                "money returned from reusing clothing is not the same claim as cheap "
                "vintage shopping. A short article is acceptable when the source is sparse. "
                "Keep any RIFIT mention at the high-level reuse/circulation direction "
                "provided by the context; do not infer product discovery, purchasing, "
                "recommendation, or service features."
            )
        elif planning_mode == "experience_deterministic":
            generation_prompt += (
                "\n\nWRITING MODE: GROUNDED EXPERIENCE. "
                "Describe only personal events and details explicitly stated in raw_source. "
                "Do not fill gaps with a first day, workplace, shopping scene, outfit detail, "
                "reaction, emotion, environmental effect, or financial result. "
                "General styling ideas are allowed only as reader suggestions or conditions, "
                "not as something the author did. If the source is brief, keep the article brief "
                "rather than inventing experience or general claims."
            )
        post = self.client.generate_structured(generation_prompt, BlogPost, payload)
        try:
            self._validate_contract(blog_idea, post, promised_count, raw_source=raw_source)
        except ValueError as error:
            self._emit_validation_diagnostics(
                error,
                blog_idea,
                post,
                raw_source,
                content_plan,
            )
            raise
        return post

    @staticmethod
    def _emit_validation_diagnostics(
        error: ValueError,
        idea,
        post: BlogPost,
        raw_source: str,
        content_plan: ContentPlan | None,
    ) -> None:
        """Expose failed intermediate artifacts only in local debug mode."""
        if not get_settings().debug:
            return
        print("Writer validation failed:", file=sys.stderr)
        print(f"reason: {error}", file=sys.stderr)
        print(f"raw_source: {raw_source!r}", file=sys.stderr)
        print(
            "refined_blog_idea: "
            + json.dumps(idea.model_dump(mode="json"), ensure_ascii=False),
            file=sys.stderr,
        )
        print(
            "content_plan.writing_script: "
            + repr(content_plan.writing_script if content_plan else ""),
            file=sys.stderr,
        )
        print(
            "writer_draft_before_validation: "
            + json.dumps(post.model_dump(mode="json"), ensure_ascii=False),
            file=sys.stderr,
        )

    @classmethod
    def _promised_count(cls, title: str) -> Optional[int]:
        match = cls._NUMBER_PROMISE.search(title or "")
        return int(match.group("count")) if match else None

    @classmethod
    def _validate_contract(
        cls,
        idea,
        post: BlogPost,
        promised_count: Optional[int],
        raw_source: str = "",
    ) -> None:
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
        cls._validate_experience_grounding(idea, content, raw_source=raw_source)
        cls._validate_rifit_grounding(idea, content)
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
    def _is_experience_contract(cls, idea) -> bool:
        contract = " ".join(
            value for value in (
                idea.title,
                idea.summary,
                idea.angle,
                idea.content_format or "",
                idea.content_perspective or "",
                idea.key_question or "",
                " ".join(idea.outline or []),
            ) if value
        ).lower()
        return any(signal in contract for signal in ("경험", "후기", "일기", "직접", "입어본", "구매한"))

    @classmethod
    def _validate_experience_grounding(cls, idea, content: str, raw_source: str = "") -> None:
        if not cls._is_experience_contract(idea):
            return
        # A raw memo is the only authority for personal facts. BlogIdea fields
        # remain useful for detecting an experience-shaped article, but their
        # editorial expansions must not authorize new autobiographical details.
        if raw_source.strip():
            contract = cls._compact(raw_source)
        else:
            contract = cls._compact(
                " ".join(
                    value for value in (
                        idea.title,
                        idea.summary,
                        idea.angle,
                        idea.content_format or "",
                        idea.content_perspective or "",
                        idea.key_question or "",
                        " ".join(idea.outline or []),
                    ) if value
                )
            )
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", content):
            is_personal_claim = cls._PERSONAL_EXPERIENCE_CLAIM.search(sentence)
            has_owned_item_claim = cls._OWNED_ITEM_CLAIM.search(sentence)
            has_event_claim = any(marker in cls._compact(sentence) for marker in cls._EXPERIENCE_EVENT_MARKERS)
            has_experience_grammar = cls._EXPERIENCE_CLAIM_GRAMMAR.search(sentence)
            if not (is_personal_claim or has_owned_item_claim or has_event_claim or has_experience_grammar):
                continue
            if any(signal in sentence.lower() for signal in cls._SUGGESTION_CONTEXT):
                continue
            compact_sentence = cls._compact(sentence)
            if raw_source.strip() and (is_personal_claim or has_owned_item_claim or has_event_claim or has_experience_grammar):
                sentence_terms = cls._topic_terms(sentence)
                source_terms = cls._topic_terms(raw_source)
                meaningful_terms = [term for term in sentence_terms if term not in {"저는", "제가", "내가", "무엇", "어떤"}]
                grounded_hits = sum(
                    1 for term in meaningful_terms
                    if any(term in source_term or source_term in term for source_term in source_terms)
                )
                # Paraphrases often add harmless connective/style words. Keep
                # the guard focused on two source anchors rather than exact
                # wording or a strict sentence-level overlap ratio.
                required_hits = 1 if len(meaningful_terms) <= 1 else 2
                if grounded_hits < required_hits:
                    raise ValueError(
                        "Writer output invents unsupported personal experience or event: "
                        f"sentence={sentence.strip()!r}"
                    )
            unsupported = [marker for marker in cls._EXPERIENCE_DETAIL_MARKERS + cls._EXPERIENCE_EVENT_MARKERS
                           if marker in compact_sentence
                           and not cls._marker_is_grounded(marker, contract)]
            if unsupported:
                raise ValueError(
                    "Writer output invents unsupported personal experience or event: "
                    f"{unsupported[0]}: marker={unsupported[0]!r}; "
                    f"sentence={sentence.strip()!r}"
                )

    @classmethod
    def _marker_is_grounded(cls, marker: str, contract: str) -> bool:
        marker = cls._compact(marker)
        if marker in contract:
            return True
        for ending in ("으로", "로", "에서", "에게", "까지", "부터", "을", "를", "은", "는", "이", "가"):
            if marker.endswith(ending) and len(marker) - len(ending) >= 2:
                return marker[:-len(ending)] in contract
        return False

    @classmethod
    def _validate_rifit_grounding(cls, idea, content: str) -> None:
        connection = (idea.rifit_connection or "").lower()
        output = content.lower()
        if not connection:
            return
        if "리패션" in output and "리패션" not in connection:
            raise ValueError("Writer output invented an unsupported RIFIT service concept")
        if re.search(r"(?:요즘|현재)?\s*많은 브랜드가", output):
            raise ValueError("Writer output invented an unsupported external brand claim")
        if any(marker in connection for marker in ("리핏", "rifit")):
            if "리핏" not in output and "rifit" not in output:
                raise ValueError("Writer output omitted the named RIFIT connection")

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
