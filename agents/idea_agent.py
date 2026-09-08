from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import List, Optional, Set

from llm.client import LLMClient, get_llm_client
from config.settings import get_settings
from models.schemas import BlogIdea, IdeaCandidate, IdeaExpansionResult
import math


def _tokens(text: str) -> Set[str]:
    # simple whitespace-based tokens, remove short tokens
    return {t for t in text.lower().split() if len(t) > 1}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    uni = a | b
    return len(inter) / len(uni)


class IdeaGenerator:
    _MAX_CORRECTION_RETRIES = 1
    _CONCEPT_GROUPS = {
        "clothing": ("옷", "의류", "의복", "의상", "헌옷", "셔츠", "패션", "아이템"),
        "discard": ("버리", "폐기", "처분", "버릴"),
        "donation": ("기부", "나눔"),
        "collection": ("수거", "수거함", "의류수거함"),
        "reuse": ("재사용", "재활용", "리폼", "업사이클링", "다시 입"),
        "vintage": ("빈티지", "세컨드핸드", "중고"),
        "shopping": ("쇼핑", "구매", "고르", "찾기"),
        "color_trend": ("코어", "색", "컬러"),
        "styling": ("코디", "스타일", "패션"),
    }
    _EXPERIENCE_SOURCE_SIGNALS = (
        "직접 샀", "샀", "구매했", "입어봤", "써봤", "사봤", "해봤", "다녀왔", "사용해봤",
        "경험했", "내가 산", "내가 입은", "해보니", "했더니", "산 경험", "인턴일기", "일기",
    )
    _EXPERIENCE_CANDIDATE_SIGNALS = (
        "개인적인 경험", "개인 경험", "경험을 공유", "경험과 교훈", "체험담",
        "후기", "내가 ", "저는 ", "제가 ", "직접 해봤", "직접 사보", "직접 입어보",
        "해보니", "사보니", "입어보니", "써보니", "경험 공유",
    )
    _EXPERIENCE_OUTPUT_NARRATIVE_SIGNALS = (
        "경험", "후기", "체험", "기록", "일기",
    )
    _EXPERIENCE_OUTPUT_ACTION_SIGNALS = (
        "구매", "샀", "사서", "입어", "착용", "출근", "코디", "활용", "해본", "해봤",
    )
    _FORMAT_FAMILIES = {
        "comparison": ("비교", "대조"),
        "guide": ("가이드", "방법", "how-to", "how to"),
        "checklist": ("체크리스트", "점검표"),
        "informational": ("정보", "설명", "해설"),
        "analysis": ("분석", "탐구"),
        "experience": ("후기", "체험", "경험", "경험담", "리뷰", "일기"),
        "interview": ("인터뷰", "문답"),
        "experiment": ("실험",),
        "styling": ("스타일링", "스타일", "코디"),
        "fact_check": ("팩트체크", "사실 확인"),
        "problem_solving": ("문제 해결",),
        "process": ("과정",),
        "trend": ("트렌드", "유행", "다시 돌아온", "스타일 흐름"),
        "editorial": ("에디토리얼", "의견", "주장", "칼럼"),
    }
    _SOURCE_DIRECTION_SIGNALS = {
        "experience": _EXPERIENCE_SOURCE_SIGNALS,
        "shopping": ("쇼핑", "찾아보", "찾기", "구매", "고르", "새 옷 말고"),
        "styling": ("코디", "스타일링", "출근룩", "입어"),
        "comparison": ("장단점", "비교", "무엇이 더", "vs", "차이점"),
        "guide": ("방법", "팁", "체크리스트", "가이드"),
    }
    _DIY_ACTION_SIGNALS = ("업사이클링", "리폼", "diy", "커팅", "패치워크")

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "idea.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()
        prompt_dir = prompt_path.parent
        self.expansion_prompt = (prompt_dir / "idea_expansion.txt").read_text(encoding="utf-8").strip()
        self.refinement_prompt = (prompt_dir / "idea_refinement.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _candidate_count(value: int) -> int:
        return min(5, max(3, int(value)))

    @staticmethod
    def _source_terms(value: str) -> Set[str]:
        particles = ("으로", "에서", "에게", "까지", "부터", "처럼", "보다", "을", "를", "은", "는", "이", "가", "의", "에", "로", "와", "과")
        terms: Set[str] = set()
        for raw in re.findall(r"[가-힣A-Za-z0-9]+", (value or "").lower()):
            term = raw
            for particle in particles:
                if term.endswith(particle) and len(term) - len(particle) >= 2:
                    term = term[: -len(particle)]
                    break
            if len(term) > 1 and term not in {"같아", "것", "요즘", "같은"}:
                terms.add(term)
        return terms

    @classmethod
    def _concept_groups(cls, value: str) -> Set[str]:
        compact = re.sub(r"\s+", "", (value or "").lower())
        return {
            group
            for group, markers in cls._CONCEPT_GROUPS.items()
            if any(marker in compact for marker in markers)
        }

    @classmethod
    def _has_explicit_experience(cls, source: str) -> bool:
        normalized = " ".join((source or "").lower().split())
        return any(signal in normalized for signal in cls._EXPERIENCE_SOURCE_SIGNALS)

    @classmethod
    def _has_unsupported_experience_claim(cls, fields: tuple[str, ...]) -> bool:
        text = " ".join(fields).lower()
        return any(signal in text for signal in cls._EXPERIENCE_CANDIDATE_SIGNALS)

    @classmethod
    def _preserves_core_concepts(cls, source: str, candidate: str) -> bool:
        source_groups = cls._concept_groups(source)
        candidate_groups = cls._concept_groups(candidate)
        if source_groups:
            # Preserve the main semantic relation, while allowing a title to
            # omit incidental context such as a city name.
            required = 2 if len(source_groups) >= 2 else 1
            return len(source_groups & candidate_groups) >= required

        source_terms = cls._source_terms(source)
        candidate_text = re.sub(r"\s+", "", (candidate or "").lower())
        return bool(source_terms and any(term in candidate_text for term in source_terms))

    @classmethod
    def _preserves_perspective(cls, perspective: str, output: str) -> bool:
        """Allow natural phrasing while requiring the perspective's core concepts."""
        perspective_text = (perspective or "").strip().lower()
        output_text = (output or "").lower()
        if not perspective_text:
            return False
        if perspective_text in output_text:
            return True

        perspective_groups = cls._concept_groups(perspective_text)
        output_groups = cls._concept_groups(output_text)
        if perspective_groups:
            required = min(2, len(perspective_groups))
            return len(perspective_groups & output_groups) >= required

        perspective_terms = cls._source_terms(perspective_text)
        output_compact = re.sub(r"\s+", "", output_text)
        return bool(perspective_terms) and sum(
            term in output_compact for term in perspective_terms
        ) >= max(1, (len(perspective_terms) + 1) // 2)

    @classmethod
    def _format_families(cls, value: str) -> Set[str]:
        compact = re.sub(r"\s+", "", (value or "").lower())
        return {
            family
            for family, markers in cls._FORMAT_FAMILIES.items()
            if any(re.sub(r"\s+", "", marker.lower()) in compact for marker in markers)
        }

    @classmethod
    def _preserves_format(cls, content_format: str, output: str) -> bool:
        format_text = (content_format or "").strip().lower()
        output_text = (output or "").lower()
        if not format_text:
            return False
        if format_text in output_text:
            return True

        format_families = cls._format_families(format_text)
        output_families = cls._format_families(output_text)
        if format_families:
            # A format such as "비교 가이드" can belong to both families;
            # retaining either core family is enough to preserve the article form.
            return bool(format_families & output_families)

        format_terms = cls._source_terms(format_text)
        output_compact = re.sub(r"\s+", "", output_text)
        return bool(format_terms) and any(term in output_compact for term in format_terms)

    @classmethod
    def _source_direction_families(cls, source: str) -> Set[str]:
        normalized = (source or "").lower()
        return {
            family
            for family, signals in cls._SOURCE_DIRECTION_SIGNALS.items()
            if any(signal.lower() in normalized for signal in signals)
        }

    @classmethod
    def _source_format_families(cls, source: str) -> Set[str]:
        """Infer only explicit article-format signals from the raw source."""
        normalized = (source or "").lower()
        families = cls._format_families(normalized)

        if cls._has_explicit_experience(normalized):
            families.add("experience")
        if any(signal in normalized for signal in ("코디", "스타일", "스타일링", "출근룩", "쇼핑", "찾아보", "즐기는 방법")):
            families.add("styling")
        if any(signal in normalized for signal in ("방법", "팁", "체크리스트", "가이드")):
            families.add("guide")
        if any(signal in normalized for signal in ("장단점", "비교", "무엇이 더", "vs", "차이점")):
            families.add("comparison")
        if any(signal in normalized for signal in ("유행", "트렌드", "다시 돌아온")):
            families.add("trend")
        if any(signal in normalized for signal in ("굳이", "필요가 있을까", "할 필요가 있을까")):
            families.add("editorial")

        return families

    @classmethod
    def _preserves_refinement_format(cls, source: str, candidate_format: str, output: str) -> bool:
        """Prefer an explicit source format, falling back to the candidate format."""
        candidate_text = re.sub(r"\s+", "", (candidate_format or "").lower())
        output_text = re.sub(r"\s+", "", (output or "").lower())
        if candidate_text and candidate_text == output_text:
            return True
        source_families = cls._source_format_families(source)
        if source_families:
            output_families = cls._format_families(output)
            # The raw source is authoritative, but incidental signals should
            # not force every related family into the refined label. For
            # example, an experience source can also mention styling without
            # requiring the output to say both "experience" and "styling".
            if "experience" in source_families:
                return "experience" in output_families
            if "comparison" in source_families:
                return "comparison" in output_families
            structured_families = source_families & {
                "guide",
                "checklist",
                "informational",
                "analysis",
                "interview",
                "experiment",
                "fact_check",
                "problem_solving",
                "process",
            }
            if structured_families:
                return bool(structured_families & output_families)
            return source_families.issubset(output_families)
        return cls._preserves_format(candidate_format, output)

    @classmethod
    def _preserves_refinement_perspective(cls, source: str, candidate_perspective: str, output: str) -> bool:
        """Prefer explicit source direction over an expansion metadata label."""
        source_families = cls._source_direction_families(source)
        if not source_families:
            return cls._preserves_perspective(candidate_perspective, output)

        output_families = cls._source_direction_families(output)
        for family in source_families:
            if family == "experience":
                if not cls._preserves_experience_direction(output):
                    return False
            elif family not in output_families:
                return False
        return True

    @classmethod
    def _preserves_experience_direction(cls, output: str) -> bool:
        normalized = (output or "").lower()
        has_narrative = any(signal in normalized for signal in cls._EXPERIENCE_OUTPUT_NARRATIVE_SIGNALS)
        has_action = any(signal in normalized for signal in cls._EXPERIENCE_OUTPUT_ACTION_SIGNALS)
        return has_narrative and has_action

    @classmethod
    def _validate_source_direction(cls, source: str, output: str) -> None:
        source_families = cls._source_direction_families(source)
        normalized_output = (output or "").lower()
        for family in source_families:
            if family == "experience":
                if not cls._preserves_experience_direction(normalized_output):
                    raise ValueError("Refined BlogIdea changed the source content direction: missing experience")
                continue
            if not any(signal.lower() in normalized_output for signal in cls._SOURCE_DIRECTION_SIGNALS[family]):
                raise ValueError(f"Refined BlogIdea changed the source content direction: missing {family}")

        source_has_diy = any(signal in source.lower() for signal in cls._DIY_ACTION_SIGNALS)
        if not source_has_diy and any(signal in normalized_output for signal in cls._DIY_ACTION_SIGNALS):
            raise ValueError("Refined BlogIdea introduced an unsupported DIY or upcycling direction")
        if not cls._preserves_source_contrast(source, normalized_output):
            raise ValueError("Refined BlogIdea changed the source contrast or editorial argument")

    @classmethod
    def _preserves_source_contrast(cls, source: str, output: str) -> bool:
        source_text = (source or "").lower()
        output_text = (output or "").lower()
        if "새 옷" in source_text and any(
            signal in source_text for signal in ("말고", "대신", "필요가 있을까", "살 필요")
        ):
            if "새 옷" not in output_text and "새로운 옷" not in output_text:
                return False
            if not any(term in output_text for term in ("빈티지", "기존 옷", "재활용", "재사용", "활용")):
                return False
        if ("버릴 필요" in source_text or "버리지" in source_text) and not any(
            term in output_text for term in ("버리", "재활용", "재사용", "활용", "빈티지")
        ):
            return False
        return True

    @classmethod
    def _source_direction_feedback(cls, source: str) -> str:
        source_text = (source or "").lower()
        requirements = []
        if "새 옷" in source_text and any(
            signal in source_text for signal in ("말고", "대신", "필요가 있을까", "살 필요")
        ):
            requirements.append("the source's choice between buying new clothes and reusing vintage or existing clothes")
        if "버릴 필요" in source_text or "버리지" in source_text:
            requirements.append("the source's point that clothes do not need to be discarded just because a trend has passed")
        if requirements:
            return (
                "The refinement changed the source contrast or editorial argument. "
                "Preserve " + "; and ".join(requirements) + ". "
                "Do not replace this argument with a shopping guide, checklist, or a new question."
            )
        return (
            "The refinement changed the source direction. Preserve the source's explicit "
            "claims, questions, actions, and subject/object relationships instead of adding a new topic."
        )

    @classmethod
    def _expansion_correction_feedback(cls, source: str, error: ValueError) -> str:
        message = str(error)
        if "invent" in message.lower() or cls._has_explicit_experience(source):
            return (
                "The previous candidates violated validation. Do not invent personal experience "
                "unless the source explicitly states it. Keep only the user's stated topic, actions, "
                "and questions, and return distinct candidates."
            )
        return (
            "The previous candidates violated validation. Preserve the source's core topic and "
            "content direction, avoid unrelated expansions, and return distinct candidates."
        )

    @classmethod
    def _refinement_correction_feedback(cls, source: str, error: ValueError) -> str:
        message = str(error)
        if "contrast or editorial argument" in message:
            return cls._source_direction_feedback(source)
        if "content direction" in message or cls._has_explicit_experience(source):
            if cls._has_explicit_experience(source):
                return (
                    "The source explicitly contains a personal experience: the user bought a "
                    "check shirt in Oslo and tried styling it for an intern commute. Preserve "
                    "that purchase, wearing, and diary/review direction in the refinement. Do not "
                    "turn it into a broad trend analysis or fashion history article."
                )
            return (
                "The previous refinement changed the source content direction. Preserve any stated "
                "experience, purchase, wearing, diary, shopping, comparison, or guide action from "
                "the source. Do not replace it with a broad analysis or an unmentioned DIY/reform direction."
            )
        return (
            "The previous refinement violated the content contract. Preserve the selected candidate "
            "and the source problem while making the BlogIdea concrete. Do not invent facts or experiences."
        )

    @classmethod
    def _validate_expansion(cls, source_input: str, result: IdeaExpansionResult, expected: int) -> IdeaExpansionResult:
        if result.source_input.strip() != source_input.strip():
            raise ValueError("Idea expansion changed the source input")
        if not 3 <= len(result.candidates) <= 5 or len(result.candidates) != expected:
            raise ValueError("Idea expansion must return the requested 3 to 5 candidates")

        compact_source = re.sub(r"[^가-힣A-Za-z0-9]", "", source_input.lower())
        titles: Set[str] = set()
        token_sets: List[Set[str]] = []
        perspective_formats: Set[tuple[str, str]] = set()
        for candidate in result.candidates:
            fields = (
                candidate.title,
                candidate.perspective,
                candidate.content_format,
                candidate.key_question,
                candidate.brief_description,
            )
            if any(not field.strip() for field in fields):
                raise ValueError("Idea candidate contains an empty required field")
            if not cls._has_explicit_experience(source_input) and cls._has_unsupported_experience_claim(fields):
                raise ValueError("Idea candidate invents personal experience not stated in the source input")
            if candidate.candidate_id in {item.candidate_id for item in result.candidates if item is not candidate}:
                raise ValueError("Idea expansion contains duplicate candidate_id")
            normalized_title = " ".join(candidate.title.lower().split())
            compact_title = re.sub(r"[^가-힣A-Za-z0-9]", "", candidate.title.lower())
            if len(compact_source) >= 12 and compact_source in compact_title:
                raise ValueError("Idea candidate copied the full source input into its title")
            if normalized_title in titles:
                raise ValueError("Idea expansion contains duplicate titles")
            titles.add(normalized_title)
            tokens = _tokens(candidate.title)
            candidate_content = " ".join((candidate.title, candidate.key_question, candidate.brief_description))
            if not cls._preserves_core_concepts(source_input, candidate_content):
                raise ValueError("Idea candidate is unrelated to the source input")
            if any(_jaccard(tokens, previous) > 0.8 for previous in token_sets):
                raise ValueError("Idea expansion contains overly similar titles")
            token_sets.append(tokens)
            perspective_formats.add((candidate.perspective.strip().lower(), candidate.content_format.strip().lower()))

        if len(result.candidates) > 1 and len(perspective_formats) < 2:
            raise ValueError("Idea candidates must provide different perspectives or formats")
        return result

    def expand(self, user_input: str, candidate_count: int = 3) -> IdeaExpansionResult:
        source_input = (user_input or "").strip()
        if not source_input:
            raise ValueError("user_input is required for idea expansion")
        expected = self._candidate_count(candidate_count)
        payload = {
            "user_input": source_input,
            "candidate_count": expected,
            "constraints": {
                "preserve_source_topic": True,
                "require_distinct_perspectives": True,
            },
        }
        prompt = self.expansion_prompt
        for attempt in range(self._MAX_CORRECTION_RETRIES + 1):
            result = self.client.generate_structured(prompt, IdeaExpansionResult, payload)
            try:
                returned_source = result.source_input
                if returned_source != source_input:
                    if get_settings().debug:
                        print(
                            "Idea expansion source_input normalized by LLM:\n"
                            f"original={source_input!r}\n"
                            f"returned={returned_source!r}",
                            file=sys.stderr,
                        )
                    # source_input is provenance, not generated content. Keep
                    # the application-owned raw source while validating ideas.
                    result.source_input = source_input
                return self._validate_expansion(source_input, result, expected)
            except ValueError as error:
                if get_settings().debug:
                    print(
                        "[Idea Expansion Validation Failure]\n"
                        f"reason={error}\n"
                        "candidates="
                        + repr([candidate.model_dump(mode="json") for candidate in result.candidates]),
                        file=sys.stderr,
                    )
                if attempt >= self._MAX_CORRECTION_RETRIES:
                    raise
                feedback = self._expansion_correction_feedback(source_input, error)
                payload = {**payload, "correction_feedback": feedback}
                prompt = f"{self.expansion_prompt}\n\nCorrection feedback:\n{feedback}"

    @classmethod
    def _validate_refinement(cls, source_input: str, candidate: IdeaCandidate, idea: BlogIdea, revision_request: str) -> BlogIdea:
        output = " ".join(
            value for value in (
                idea.title,
                idea.keyword,
                idea.angle,
                idea.summary,
                idea.content_format or "",
                idea.content_perspective or "",
                idea.key_question or "",
                " ".join(idea.outline or []),
            ) if value
        ).lower()
        if not idea.title.strip() or not idea.keyword.strip() or not idea.angle.strip() or not idea.summary.strip():
            raise ValueError("Refined BlogIdea is missing required fields")
        try:
            cls._validate_source_direction(source_input, output)
        except ValueError as error:
            if get_settings().debug:
                print(
                    "[Idea Refinement Direction Failure]\n"
                    f"source={source_input!r}\n"
                    f"title={idea.title!r}\n"
                    f"key_question={idea.key_question!r}\n"
                    f"summary={idea.summary!r}\n"
                    f"outline={idea.outline!r}\n"
                    f"reason={error}",
                    file=sys.stderr,
                )
            raise
        if not any(term in output for term in cls._source_terms(candidate.title)):
            raise ValueError("Refined BlogIdea does not reflect the selected candidate")
        if not cls._preserves_refinement_perspective(source_input, candidate.perspective, output):
            raise ValueError(
                "Refined BlogIdea does not preserve the candidate perspective: "
                f"candidate={candidate.perspective!r}, "
                f"refined={idea.content_perspective!r}"
            )
        if not cls._preserves_refinement_format(
            source_input, candidate.content_format, idea.content_format or ""
        ):
            raise ValueError(
                "Refined BlogIdea does not preserve the candidate format: "
                f"candidate={candidate.content_format!r}, "
                f"refined={idea.content_format!r}"
            )
        if candidate.key_question.strip().lower() not in output:
            raise ValueError("Refined BlogIdea does not preserve the candidate question")
        if len(idea.outline or []) < 3:
            raise ValueError("Refined BlogIdea requires a concrete outline")
        if any(term in output for term in ("seo", "검색 의도", "fit_score", "metadata", "prompt", "agent")):
            raise ValueError("Refined BlogIdea exposes internal metadata")
        if revision_request:
            revision_terms = cls._source_terms(revision_request)
            if revision_terms and not any(term in output for term in revision_terms):
                raise ValueError("Refined BlogIdea does not reflect the revision request")
        return idea

    def refine(self, source_input: str, candidate: IdeaCandidate, revision_request: str = "") -> BlogIdea:
        source_input = (source_input or "").strip()
        if not source_input:
            raise ValueError("source_input is required for idea refinement")
        payload = {
            "source_input": source_input,
            "selected_candidate": candidate.model_dump(mode="json"),
            "revision_request": (revision_request or "").strip(),
        }
        prompt = self.refinement_prompt
        for attempt in range(self._MAX_CORRECTION_RETRIES + 1):
            result = self.client.generate_structured(prompt, BlogIdea, payload)
            try:
                return self._validate_refinement(source_input, candidate, result, revision_request)
            except ValueError as error:
                if attempt >= self._MAX_CORRECTION_RETRIES:
                    raise
                feedback = self._refinement_correction_feedback(source_input, error)
                payload = {**payload, "correction_feedback": feedback}
                prompt = f"{self.refinement_prompt}\n\nCorrection feedback:\n{feedback}"

    def _fallback_candidates(self) -> List[dict]:
        return [
            {
                "title": "의류 정리와 순환, 현실적인 1가지 기준",
                "keyword": "의류 순환 방법",
                "search_intent": "정보 탐색",
                "angle": "정리 기준과 순환 경로를 함께 고려하는 실용적인 의류 관리 지침",
                "rifit_connection": "의류 정리와 수거의 자연스러운 연결",
                "seasonality": 0.85,
            },
            {
                "title": "안 입는 반팔 처리, 이렇게 정리하면 실속 있다",
                "keyword": "안 입는 반팔 처리",
                "search_intent": "문제 해결",
                "angle": "상태별 분류와 재사용 가능성을 함께 보는 실전 처리 팁",
                "rifit_connection": "헌옷 수거와 의류 재사용 관점으로 연결",
                "seasonality": 0.9,
            },
            {
                "title": "옷장 정리 시즌별 팁, 의류 순환까지 이어가기",
                "keyword": "옷장 정리 시즌별 팁",
                "search_intent": "정보 탐색",
                "angle": "계절 전환 시점에 필요한 의류 분류와 순환 전략",
                "rifit_connection": "계절별 의류 정리와 헌옷 수거를 함께 설계",
                "seasonality": 0.92,
            },
        ]

    def generate(self, trend_results, seo_results, brand_context=None) -> List[BlogIdea]:
        prompt = self.prompt
        settings = get_settings()
        idea_count = max(1, settings.idea_count)

        if not trend_results and not seo_results:
            # return fallback candidates sized to idea_count
            base = self._fallback_candidates()
            candidates = (base * ((idea_count // len(base)) + 1))[:idea_count]
            return self.client.generate_many(prompt, BlogIdea, candidates)

        # build a seed pool from trends and seo keywords (preserve order, unique)
        seeds = []
        if trend_results:
            for t in trend_results:
                if t.topic not in seeds:
                    seeds.append(t.topic)
        if seo_results:
            for s in seo_results:
                if s.keyword not in seeds:
                    seeds.append(s.keyword)

        # editorial perspectives to diversify idea topics (no seed injection into title)
        perspectives = [
            ("옷장 정리", "옷장 정리 - 공간 구성과 분류 기준, 정리 루틴 중심의 실용적 접근"),
            ("의류 관리", "의류 관리 - 소재별 세탁·보관·수선 관점의 실무 조언"),
            ("헌옷 처리", "헌옷 처리 - 기부·수거·정리 단계에서의 실용 가이드"),
            ("의류 재활용", "재활용/업사이클링 - 재료 분리와 재활용 아이디어 소개"),
            ("재사용", "재사용 - 집에서 다시 쓰기 가능한 아이템 선정과 준비 방법"),
            ("중고 거래", "중고 판매/리셀 - 사진·설명·가격 책정 팁과 플랫폼별 전략"),
            ("빈티지", "빈티지 스타일 - 리폼으로 가치 올리기와 스타일 연출"),
            ("수선/리폼", "수선·리폼 - 간단 수선으로 수명 늘리는 방법과 비용 가이드"),
            ("소비 습관", "소비 습관 - 장기적 의류 소비 절약과 계획 방법"),
            ("환경/순환", "환경과 의류 순환 - 지역 자원과 기부 연결 방법 및 영향"),
        ]

        # natural title templates per perspective — avoid English tokens and machiney patterns
        templates = {
            "옷장 정리": [
                "지금 당장 옷장 한 칸 정리로 생활을 가볍게 만드는 방법",
                "옷장 정리를 빠르게 끝내는 실전 루틴"
            ],
            "의류 관리": [
                "소재별로 알아보는 세탁·보관의 기본",
                "자주 입는 옷 오래 입히는 작은 습관"
            ],
            "헌옷 처리": [
                "헌옷 보낼 때 이것만은 꼭 확인하세요",
                "헌옷을 기부하거나 재사용할 때 실수하지 않는 법"
            ],
            "의류 재활용": [
                "버려진 옷을 새롭게 바꾸는 재활용 아이디어",
                "작은 수선으로 만드는 업사이클링 사례"
            ],
            "재사용": [
                "집에서 바로 재사용할 수 있는 의류 정리법",
                "다음 시즌까지 입을 옷을 고르는 기준"
            ],
            "중고 거래": [
                "중고로 잘 팔리는 사진과 설명의 비결",
                "리셀 전 꼭 확인할 체크포인트"
            ],
            "빈티지": [
                "빈티지 무드 내는 간단한 리폼 아이디어",
                "오래된 옷을 스타일로 살리는 방법"
            ],
            "수선/리폼": [
                "집에서 시도해볼 수 있는 쉬운 옷 수선 가이드",
                "수선 비용과 효과를 비교해보는 방법"
            ],
            "소비 습관": [
                "옷 구매 빈도를 줄이는 실천법",
                "필요한 옷만 남기는 소비 계획 세우기"
            ],
            "환경/순환": [
                "동네에서 할 수 있는 의류 순환 참여 방법",
                "의류 순환이 환경에 주는 영향과 시작 방법"
            ],
        }

        candidates = []
        seen_titles = set()
        seen_token_sets: List[Set[str]] = []

        # iterate seeds and perspectives to generate candidates; ensure intra-run diversity
        s_index = 0
        p_index = 0
        while len(candidates) < idea_count and (s_index < len(seeds)):
            seed = seeds[s_index]
            perspective = perspectives[p_index % len(perspectives)]
            key = perspective[0]
            angle_desc = perspective[1]
            tmpl_list = templates.get(key, [templates[list(templates.keys())[0]][0]])
            tmpl = tmpl_list[(s_index + p_index) % len(tmpl_list)]

            # produce title without forcing seed inclusion
            title = tmpl

            # avoid duplicate or overly similar titles within this run
            tok = _tokens(title)
            too_similar = False
            for prev in seen_token_sets:
                if _jaccard(tok, prev) > 0.45:
                    too_similar = True
                    break
            if too_similar:
                # advance perspective to try a different angle
                p_index += 1
                # if we've cycled through many perspectives, fallback to add variant with seed as suffix phrase
                if p_index - s_index > len(perspectives) * 2:
                    title = f"{tmpl} — {seed}과 연결된 실전 방법"
                    tok = _tokens(title)
                    too_similar = False

            if title in seen_titles:
                # avoid identical title
                p_index += 1
                s_index += 1
                continue

            # pick keyword and seasonality from matching seo or trend
            keyword = seed
            search_intent = "정보 탐색"
            seasonality = 0.5
            if seo_results:
                for s in seo_results:
                    if s.keyword == seed:
                        keyword = s.keyword
                        search_intent = s.search_intent
                        seasonality = round(min(max(s.seasonality, 0.0), 1.0), 2)
                        break
            if trend_results and seasonality == 0.5:
                for t in trend_results:
                    if t.topic == seed:
                        seasonality = round(min(max(t.relevance_score, 0.0), 1.0), 2)
                        break

            candidates.append(
                {
                    "title": title,
                    "keyword": keyword,
                    "search_intent": search_intent,
                    "angle": angle_desc,
                    "rifit_connection": "의류 순환/수거/재사용과 연결되는 실용적 콘텐츠",
                    "seasonality": seasonality,
                }
            )

            seen_titles.add(title)
            seen_token_sets.append(tok)

            # advance indices
            p_index += 1
            if p_index % len(perspectives) == 0:
                s_index += 1

        # if still short, fill with non-seed generic variants
        i = 1
        while len(candidates) < idea_count:
            title = f"의류 순환을 일상에 적용하는 실용적 방법 {i}"
            if title not in seen_titles:
                candidates.append({
                    "title": title,
                    "keyword": "의류 순환 방법",
                    "search_intent": "정보 탐색",
                    "angle": "생활 속에서 바로 적용 가능한 의류 재사용 실전 팁",
                    "rifit_connection": "의류 정리와 수거의 자연스러운 연결",
                    "seasonality": 0.5,
                })
                seen_titles.add(title)
            i += 1

        return self.client.generate_many(prompt, BlogIdea, candidates[:idea_count])
