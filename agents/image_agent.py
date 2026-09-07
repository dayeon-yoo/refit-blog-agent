from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogPost, ImagePlan, ImagePrompt


class ImageAgent:
    """Build a deterministic image plan from the article's visual moments."""

    _FASHION_TERMS = (
        "옷", "의류", "옷장", "셔츠", "바지", "니트", "청바지", "가방", "소재",
        "코디", "스타일", "수납", "행거", "서랍",
    )
    _CTA_MARKERS = ("마무리", "정리해보세요", "시작해보세요", "실천해보세요", "함께해요")
    _ROLE_RULES = (
        ("comparison", "비교 전후의 변화가 한눈에 보이는 장면", ("비교", "전후", "차이", "반면")),
        ("classification", "의류를 기준에 따라 나누어 놓은 장면", ("나누", "나눠", "구분", "종류별")),
        ("detail", "의류의 소재와 상태를 가까이 살펴보는 장면", ("얼룩", "손상", "소재", "늘어남", "디테일", "상태")),
        ("reuse", "기존 의류를 새로운 용도로 활용하는 장면", ("재사용", "재활용", "업사이클링", "리폼", "활용")),
        ("styling", "기존 의류를 조합해 착용한 현실적인 코디 장면", ("코디", "스타일링", "착용", "입어", "조합")),
        ("checklist", "의류 상태와 조건을 차례로 확인하는 장면", ("체크", "확인", "기준", "점검", "판단")),
        ("process", "사람의 손과 의류가 함께 보이는 실제 실행 과정", ("방법", "단계", "순서", "정리", "수납", "세탁", "수선", "해보")),
        ("organization", "가정용 옷장과 수납 공간을 정돈하는 장면", ("옷장", "정리", "보관", "수납", "비우")),
    )

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # The client and prompt remain available for a future downstream
        # generator, but planning itself is intentionally rule-based.
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "image.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def _parse_sections(self, content: str, title: str) -> list[tuple[str, str, int]]:
        lines = content.splitlines()
        sections: list[tuple[str, str, int]] = []
        current_heading = ""
        current_index = 0
        buffer: list[str] = []

        for line in lines:
            if re.match(r"^#{2,3}\s+", line.strip()):
                if buffer or current_heading:
                    sections.append((current_heading.strip(), "\n".join(buffer).strip(), current_index))
                current_heading = line.strip().lstrip("# ").strip()
                current_index = len(sections)
                buffer = []
            else:
                buffer.append(line)

        if buffer or current_heading:
            sections.append((current_heading.strip(), "\n".join(buffer).strip(), current_index))
        return sections or [(title, content, 0)]

    @staticmethod
    def _contains(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    def _classify(self, heading: str, paragraph: str, index: int, total: int) -> tuple[str, str, float]:
        text = f"{heading} {paragraph}".lower()
        for role, purpose, markers in self._ROLE_RULES:
            matched = self._contains(text, markers)
            if role == "classification" and "분류" in heading.lower():
                matched = True
            if matched:
                if role == "detail" and self._contains(text, ("소재", "원단", "봉제", "라벨", "단추", "지퍼")):
                    purpose = "소재와 봉제선 같은 의류 구조를 가까이 확인하는 장면"
                elif role == "detail" and self._contains(text, ("얼룩", "손상", "늘어남", "마모", "상태")):
                    purpose = "얼룩과 늘어남 등 의류 상태를 가까이 확인하는 장면"
                score = 5.0
                if self._contains(text, self._FASHION_TERMS):
                    score += 2.0
                if any(marker in paragraph for marker in ("해보", "나눠", "확인", "비교", "정리")):
                    score += 2.0
                if index == total - 1 and self._contains(text, self._CTA_MARKERS):
                    score -= 4.0
                return role, purpose, score

        # Long, clothing-specific sections still contain useful visual
        # context even when they do not use one of the explicit markers.
        if index < total - 1 and len(paragraph) >= 80 and self._contains(text, self._FASHION_TERMS):
            return "process", "본문의 의류 관리 장면을 보여주는 과정 이미지", 2.5

        # A short intro is only a fallback hero candidate, not a reason to
        # fill the whole plan with generic lifestyle images.
        if index == 0 and len(paragraph) >= 80:
            return "lifestyle", "글의 상황을 보여주는 도입 장면", 1.5
        return "", "", 0.0

    def _select_sections(self, sections: list[tuple[str, str, int]]) -> list[dict]:
        candidates = []
        for heading, paragraph, index in sections:
            role, purpose, score = self._classify(heading, paragraph, index, len(sections))
            if not role or score <= 0:
                continue
            candidates.append({
                "heading": heading,
                "paragraph": paragraph,
                "index": index,
                "role": role,
                "purpose": purpose,
                "score": score,
            })

        # Rank for quality, cap repeated roles, then restore article order.
        candidates.sort(key=lambda item: (-item["score"], item["index"]))
        selected: list[dict] = []
        role_counts: dict[str, int] = {}
        for candidate in candidates:
            role = candidate["role"]
            if role_counts.get(role, 0) >= 2:
                continue
            selected.append(candidate)
            role_counts[role] = role_counts.get(role, 0) + 1
            if len(selected) >= 5:
                break

        selected.sort(key=lambda item: item["index"])
        return selected

    def _visual_subject(self, role: str, paragraph: str, heading: str, blog_post: BlogPost) -> str:
        text = f"{heading} {paragraph}".lower()
        garments = [term for term in self._FASHION_TERMS if term in text]
        garment_label = "과 ".join(dict.fromkeys(garments[:2])) or "의류"
        if role == "classification":
            return f"{garment_label}를 두 그룹으로 나눈 정리 장면"
        if role == "comparison":
            return f"정리 전후의 {garment_label}와 옷장 변화"
        if role == "detail":
            if self._contains(text, ("소재", "원단", "봉제", "라벨", "단추", "지퍼")):
                return f"{garment_label}의 소재와 봉제 상태 디테일"
            if self._contains(text, ("얼룩", "손상", "늘어남", "마모", "상태")):
                return f"{garment_label}의 얼룩과 늘어남 같은 착용 상태 디테일"
            if heading:
                return f"{heading}에서 확인하는 {garment_label}의 구체적인 디테일"
            return f"{garment_label}의 소재와 착용 상태 디테일"
        if role == "reuse":
            return f"기존 {garment_label}를 새로운 용도로 활용하는 작업"
        if role == "styling":
            return f"기존 {garment_label}를 조합한 일상 코디"
        if role == "checklist":
            return f"{garment_label}의 상태와 보관 조건을 확인하는 장면"
        if role == "process":
            return f"{garment_label}를 실제로 정리하고 다루는 과정"
        if role == "organization":
            return "가정용 옷장과 수납된 계절 의류"
        return blog_post.title or blog_post.summary or "일상 속 의류 관리"

    def _make_prompt(self, role: str, subject: str, heading: str, blog_post: BlogPost) -> str:
        setting = "실제 가정의 침실 또는 옷장 앞"
        composition = "행동과 결과가 한눈에 보이는 자연스러운 사선 구도"
        style = "따뜻한 자연광, 현실적인 다큐멘터리 블로그 사진, 과도한 광고 연출과 텍스트 오버레이 없음"

        if role == "classification":
            composition = "옷의 두 그룹이 좌우로 분명히 구분되는 상단 사선 구도"
        elif role == "comparison":
            composition = "정리 전후를 좌우로 비교하는 깔끔한 split composition"
        elif role == "detail":
            setting = "밝은 테이블 위 또는 옷장 앞"
            if "얼룩" in subject or "손상" in subject or "늘어남" in subject:
                composition = "옷의 얼룩, 늘어남, 마모가 분명히 보이는 문제 부위 close-up"
            else:
                composition = "옷감의 결, 봉제선, 라벨 같은 구조가 보이는 소재 close-up"
        elif role == "reuse":
            setting = "작업 테이블과 옷장이 있는 생활 공간"
            composition = "기존 의류와 새 활용 결과물이 함께 보이는 과정 중심 구도"
        elif role == "styling":
            setting = "자연광이 들어오는 집 안 또는 현실적인 외출 준비 공간"
            composition = "옷의 조합과 착용 모습을 보여주는 반신 또는 전신 구도"
        elif role == "checklist":
            composition = "확인해야 할 부분이 잘 보이는 손과 의류의 detail shot"
        elif role == "organization":
            composition = "수납 전후의 공간감과 손이 닿는 옷 배치가 함께 보이는 wide shot"

        context = heading or blog_post.summary or blog_post.title
        return f"{subject}, {setting}, {composition}, {style}, 본문 section '{context}'의 정보를 시각적으로 보완하는 장면"

    def _make_alt_text(self, role: str, subject: str) -> str:
        if role == "classification":
            return "자주 입는 옷과 보관할 옷을 두 그룹으로 나누어 정리하는 모습"
        if role == "comparison":
            return "정리 전후로 달라진 옷장과 의류 배치를 비교한 모습"
        if role == "detail":
            if "얼룩" in subject or "손상" in subject or "늘어남" in subject:
                return "옷의 얼룩과 늘어남 등 착용 상태를 가까이 확인하는 모습"
            if "소재" in subject or "봉제" in subject or "라벨" in subject:
                return "옷감의 결과 봉제선, 라벨을 가까이 확인하는 모습"
            return "옷감과 의류의 얼룩이나 손상 상태를 가까이 확인하는 모습"
        if role == "reuse":
            return "기존 의류를 새로운 용도로 활용하는 작업 과정"
        if role == "styling":
            return "기존 옷을 조합해 일상 코디를 완성한 모습"
        if role == "checklist":
            return "의류 상태와 보관 조건을 차례로 확인하는 모습"
        if role == "organization":
            return "계절 의류를 옷장과 수납공간에 정돈한 모습"
        return subject

    def plan(self, blog_post: BlogPost) -> BlogPost:
        sections = self._parse_sections(blog_post.content, blog_post.title)
        chosen = self._select_sections(sections)
        images: list[ImagePrompt] = []

        for rank, info in enumerate(chosen):
            role = info["role"]
            subject = self._visual_subject(role, info["paragraph"], info["heading"], blog_post)
            placement = "hero-intro" if rank == 0 and info["index"] == 0 else f"section-{info['index'] + 1}"
            images.append(ImagePrompt(
                placement=placement,
                purpose=info["purpose"],
                prompt=self._make_prompt(role, subject, info["heading"], blog_post),
                image_type=role,
                alt_text=self._make_alt_text(role, subject),
            ))

        blog_post.image_plan = ImagePlan(images=images)
        return blog_post
