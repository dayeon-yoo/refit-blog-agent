from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from agents.editorial_context import build_safe_editorial_context
from llm.client import LLMClient, get_llm_client
from models.schemas import BlogIdea, BlogPost, ContentPlan, QCIssue, QCResult


class QCAgent:
    """Inspect final content without rewriting or regenerating it."""

    _CLAIM_AUTHORITY_POLICY = {
        "PERSONAL_FACT": "raw_source_only",
        "GENERAL_SUGGESTION": "allowed within the requested topic when clearly reader-facing or conditional",
        "EXTERNAL_CLAIM": "raw_source or supplied research evidence only",
        "EDITORIAL_TRANSITION": "allowed unless promoted into a factual or personal claim",
    }

    _METADATA = (
        "검색 의도", "seo", "키워드 선정 이유", "브랜드 평가", "fit_score",
        "critic score", "ai 평가", "내부 agent", "prompt", "생성 과정",
    )
    _EXPERIENCE_EVENTS = (
        "동료", "칭찬", "반응이 좋", "첫 출근", "근무", "미팅", "회의",
        "매장을 돌아", "돌아다니며", "자존감", "자신감이 높", "절약했", "이득을 봤",
    )
    _SOURCE_EXPERIENCE = (
        "일기", "직접", "입어봤", "입어 봤", "사봤", "사 봤", "구매했", "샀다",
        "해봤", "해 봤", "다녀왔", "경험", "후기",
    )
    _RIFIT_CAPABILITIES = (
        "검색", "구매", "상품", "추천", "가격", "보상", "판매 플랫폼", "플랫폼",
    )
    _RESEARCH_MARKERS = ("매년", "통계", "연구 결과", "시장 규모", "만 톤", "% 증가", "조사 결과")
    _GENERIC_TAGS = {"기준", "방법", "팁", "차이", "비교", "가능한", "지속", "정보"}

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "qc.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"\s+", "", (value or "").lower())

    @staticmethod
    def _sentences(value: str) -> list[str]:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", value or "") if sentence.strip()]

    @classmethod
    def _evidence_for_marker(cls, post: BlogPost, marker: str) -> str:
        """Return a real content span containing the detected marker."""
        for field in (post.title, post.content, post.summary, post.cta):
            for sentence in cls._sentences(field):
                if marker.lower() in sentence.lower():
                    return sentence
        for tag in post.tags or []:
            if marker.lower() in tag.lower():
                return tag
        return ""

    @classmethod
    def _issue(cls, category: str, severity: str, message: str, evidence: str, suggestion: str) -> QCIssue:
        return QCIssue(
            category=category,
            severity=severity,
            message=message,
            evidence=evidence[:500],
            suggestion=suggestion,
        )

    @classmethod
    def _check_metadata(cls, post: BlogPost) -> list[QCIssue]:
        text = " ".join((post.title, post.content, post.summary, post.cta, " ".join(post.tags or [])))
        for marker in cls._METADATA:
            if marker.lower() in text.lower():
                return [cls._issue("metadata_leakage", "error", "내부 메타데이터가 공개 콘텐츠에 노출되었습니다.", cls._evidence_for_marker(post, marker), "내부 작업 정보를 제거하세요.")]
        return []

    @classmethod
    def _check_grounding(cls, source: str, post: BlogPost) -> list[QCIssue]:
        if not source:
            return []
        source_lower = source.lower()
        content = post.content or ""
        issues: list[QCIssue] = []
        if any(marker in source_lower for marker in cls._SOURCE_EXPERIENCE):
            for sentence in cls._sentences(content):
                if any(marker in sentence for marker in cls._EXPERIENCE_EVENTS):
                    absent = next((marker for marker in cls._EXPERIENCE_EVENTS if marker in sentence and marker not in source_lower), None)
                    if absent:
                        issues.append(cls._issue(
                            "source_grounding", "error",
                            "원본에 없는 개인 경험이나 사건을 사실처럼 서술했습니다.",
                            sentence,
                            "원본에 명시된 경험만 사실로 쓰고 나머지는 일반적인 제안으로 바꾸세요.",
                        ))
                        break
        if "빈티지" in source_lower and not any(marker in source_lower for marker in ("리폼", "업사이클")):
            for sentence in cls._sentences(content):
                if any(marker in sentence.lower() for marker in ("리폼", "업사이클링", "업사이클")):
                    issues.append(cls._issue(
                        "source_grounding", "error",
                        "원본에 없는 리폼/업사이클링 행동으로 주제가 바뀌었습니다.",
                        sentence,
                        "빈티지 활용과 리폼을 동일한 행동으로 취급하지 마세요.",
                    ))
                    break
        return issues

    @classmethod
    def _check_facts_and_rifit(cls, post: BlogPost) -> list[QCIssue]:
        issues: list[QCIssue] = []
        text = " ".join((post.title, post.content, post.summary, post.cta))
        for sentence in cls._sentences(text):
            if any(marker in sentence for marker in cls._RESEARCH_MARKERS) or re.search(r"\b\d+(?:\.\d+)?\s*(?:%|만|억|톤|명)\b", sentence):
                issues.append(cls._issue("unsupported_fact", "error", "검증 근거가 없는 구체적 사실 또는 수치가 포함되었습니다.", sentence, "출처를 제공하거나 일반적인 표현으로 완화하세요."))
                break
        lower = text.lower()
        if "리핏" in lower or "rifit" in lower:
            for capability in cls._RIFIT_CAPABILITIES:
                if capability in lower:
                    evidence = next((
                        sentence for sentence in cls._sentences(text)
                        if capability in sentence.lower()
                        and ("리핏" in sentence.lower() or "rifit" in sentence.lower())
                    ), "")
                    issues.append(cls._issue("rifit_grounding", "error", "확인되지 않은 RIFIT 기능을 단정했습니다.", evidence, "확인된 의류 재사용/순환 맥락으로만 연결하세요."))
                    break
        return issues

    @classmethod
    def _check_writing_quality(cls, post: BlogPost) -> list[QCIssue]:
        issues: list[QCIssue] = []
        if re.search(r"(?:으로를|으로은|으로는|를를|은은)", post.title):
            issues.append(cls._issue("writing_quality", "error", "제목의 조사가 어색하게 결합되었습니다.", post.title, "제목의 조사와 문장 구조를 자연스럽게 수정하세요."))
        sentences = [cls._compact(sentence) for sentence in cls._sentences(post.content)]
        duplicates = {sentence for sentence in sentences if sentence and sentences.count(sentence) > 1}
        if duplicates:
            evidence = next(sentence for sentence in cls._sentences(post.content) if cls._compact(sentence) in duplicates)
            issues.append(cls._issue("writing_quality", "error", "본문에 동일한 문장이 반복됩니다.", evidence, "반복 문장을 하나로 정리하세요."))
        if not post.content.strip():
            issues.append(cls._issue("writing_quality", "error", "본문이 비어 있습니다.", "", "본문을 작성하세요."))
        return issues

    @classmethod
    def _source_terms(cls, source: str, post: BlogPost) -> set[str]:
        text = cls._compact(" ".join((source, post.title, post.keyword, post.content)))
        terms = {
            term for term in (
                "빈티지", "구제", "체크셔츠", "셔츠", "인턴출근룩", "인턴룩", "출근룩",
                "코디", "출근", "인턴", "룩", "y2k", "쇼핑", "재활용", "재사용", "기부", "수거", "옷장", "정리",
            ) if term in text
        }
        return terms

    @classmethod
    def _check_tags(cls, source: str, post: BlogPost) -> list[QCIssue]:
        if not post.tags:
            return [cls._issue("tag_relevance", "warning", "태그가 비어 있습니다.", "", "본문의 핵심 주제에 맞는 태그를 추가하세요.")]
        source_terms = cls._source_terms(source, post)
        issues = []
        for tag in post.tags:
            compact = cls._compact(tag)
            if compact in cls._GENERIC_TAGS or not any(term in compact or compact in term for term in source_terms):
                issues.append(cls._issue("tag_relevance", "warning", "본문 및 원본 주제와 직접 연결되지 않는 태그입니다.", tag, "원본과 본문에 실제로 등장하는 핵심 주제 태그를 우선하세요."))
        return issues[:3]

    @classmethod
    def _check_images(cls, source: str, post: BlogPost) -> list[QCIssue]:
        images = (post.image_plan.images if post.image_plan else [])
        if not images:
            return [cls._issue("image_relevance", "warning", "이미지 계획이 비어 있습니다.", "", "각 핵심 section에 필요한 이미지를 계획하세요.")]
        source_terms = cls._source_terms(source, post)
        image_texts = [cls._compact(" ".join((image.purpose, image.prompt, image.alt_text))) for image in images]
        if source_terms and any(term in source_terms for term in ("빈티지", "체크셔츠", "인턴출근룩", "인턴룩", "y2k", "구제")):
            specific = {"빈티지", "체크셔츠", "인턴출근룩", "인턴룩", "y2k", "구제", "코디", "쇼핑"}
            if not any(any(term in text for term in specific.intersection(source_terms)) for text in image_texts):
                image = images[0]
                evidence = image.purpose or image.prompt or image.alt_text
                return [cls._issue("image_relevance", "warning", "이미지가 글의 구체적인 시각 주제를 반영하지 않습니다.", evidence, "본문의 실제 의류, 스타일, 쇼핑 또는 비교 주제를 이미지에 반영하세요.")]
        keys = [(image.image_type, cls._compact(image.purpose), cls._compact(image.prompt)) for image in images]
        if len(keys) != len(set(keys)):
            duplicate = next(
                image for image in images
                if sum(
                    1 for candidate in images
                    if (candidate.image_type, cls._compact(candidate.purpose), cls._compact(candidate.prompt))
                    == (image.image_type, cls._compact(image.purpose), cls._compact(image.prompt))
                ) > 1
            )
            return [cls._issue("image_duplicate", "warning", "동일한 이미지 장면이 반복됩니다.", duplicate.purpose or duplicate.prompt, "중복 장면을 하나로 줄이세요.")]
        return []

    @classmethod
    def _semantic_evidence(cls, issue: QCIssue, post: BlogPost) -> QCIssue:
        """Reject LLM-created evidence that is absent from reviewed content."""
        evidence = (issue.evidence or "").strip()
        if not evidence:
            return issue
        reviewed = " ".join((
            post.title, post.content, post.summary, post.cta,
            " ".join(post.tags or []),
            " ".join(
                value
                for image in (post.image_plan.images if post.image_plan else [])
                for value in (image.purpose, image.prompt, image.alt_text)
            ),
        ))
        if evidence.lower() in reviewed.lower():
            return issue
        return issue.model_copy(update={"evidence": ""})

    @classmethod
    def _deterministic_checks(cls, source: str, idea: BlogIdea, post: BlogPost) -> list[QCIssue]:
        issues: list[QCIssue] = []
        for checker in (
            cls._check_metadata,
            cls._check_writing_quality,
            cls._check_facts_and_rifit,
        ):
            issues.extend(checker(post))
        issues.extend(cls._check_grounding(source, post))
        issues.extend(cls._check_tags(source, post))
        issues.extend(cls._check_images(source, post))
        return issues

    @staticmethod
    def _status(issues: Iterable[QCIssue], semantic: Optional[QCResult] = None) -> str:
        all_issues = list(issues) + (semantic.issues if semantic else [])
        if any(issue.severity == "error" and issue.category in {"source_grounding", "unsupported_fact", "rifit_grounding", "metadata_leakage"} for issue in all_issues):
            return "BLOCK"
        if all_issues or (semantic and semantic.status != "PASS"):
            return "NEEDS_REVISION"
        return "PASS"

    def check(
        self,
        source: str,
        idea: BlogIdea,
        post: BlogPost,
        content_plan: Optional[ContentPlan] = None,
    ) -> QCResult:
        deterministic = self._deterministic_checks(source, idea, post)
        payload = {
            "raw_source": source,
            "refined_blog_idea": build_safe_editorial_context(idea, source),
            "blog_post": post.model_dump(mode="json"),
            "content_plan": content_plan.model_dump(mode="json") if content_plan else None,
            "claim_authority_policy": self._CLAIM_AUTHORITY_POLICY,
            "deterministic_issues": [issue.model_dump(mode="json") for issue in deterministic],
        }
        semantic = self.client.generate_structured(self.prompt, QCResult, payload)
        semantic = semantic.model_copy(update={
            "issues": [self._semantic_evidence(issue, post) for issue in semantic.issues]
        })
        issues = deterministic + semantic.issues
        status = self._status(deterministic, semantic)
        result = QCResult(
            status=status,
            issues=issues,
            summary="게시 전 자동 검수를 완료했습니다." if status == "PASS" else "게시 전 수정 또는 확인이 필요한 항목이 있습니다.",
        )
        if os.getenv("DEBUG", "").lower() == "true":
            print("[QC Debug]", file=sys.stderr)
            print("deterministic_checks:", json.dumps([item.model_dump() for item in deterministic], ensure_ascii=False), file=sys.stderr)
            print("semantic_input: fields=raw_source, refined_blog_idea, blog_post, content_plan", file=sys.stderr)
            print("semantic_result:", json.dumps(semantic.model_dump(), ensure_ascii=False), file=sys.stderr)
            print("issues:", json.dumps([item.model_dump() for item in issues], ensure_ascii=False), file=sys.stderr)
            print("final_status:", result.status, file=sys.stderr)
        return result
