from __future__ import annotations

import json
import os
import re
import sys
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
    _GARMENT_TERMS = ("셔츠", "바지", "니트", "청바지", "가방", "옷장", "의류", "옷")
    _CTA_MARKERS = ("마무리", "정리해보세요", "시작해보세요", "실천해보세요", "함께해요")
    _NO_TEXT = (
        "이미지 안에 글자나 숫자, 문구, 로고, 워터마크, 읽을 수 있는 표지판과 텍스트 오버레이가 전혀 없음"
    )
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
        self._debug_trace: dict[str, object] | None = None

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("DEBUG", "").lower() == "true"

    @staticmethod
    def _topic_terms(text: str) -> dict[str, list[str]]:
        specific_terms = (
            "빈티지", "구제", "체크셔츠", "셔츠코디", "출근룩", "인턴룩",
            "인턴", "y2k", "쇼핑", "스타일링", "코디",
        )
        generic_terms = (
            "재사용", "재활용", "순환", "지속가능", "기부", "수거",
            "옷장", "정리", "분류", "보관",
        )
        lowered = (text or "").lower()
        return {
            "specific": [term for term in specific_terms if term in lowered],
            "generic": [term for term in generic_terms if term in lowered],
        }

    @staticmethod
    def _specific_role(text: str) -> str:
        lowered = (text or "").lower()
        if (
            "빈티지" in lowered and "구제" in lowered
            and any(marker in lowered for marker in ("차이", "비교", "장단점", "반면"))
        ):
            return "comparison"
        if any(marker in lowered for marker in ("출근룩", "인턴룩", "코디", "스타일링", "입어봤", "입어 보")):
            return "styling"
        if "y2k" in lowered and any(marker in lowered for marker in ("패션", "스타일", "쇼핑", "매장")):
            return "styling"
        return ""

    def _emit_debug_trace(self) -> None:
        if not self._debug_trace:
            return
        print("[Image Debug]", file=sys.stderr)
        for key in ("source_visual_topics", "post_visual_topics", "sections", "candidate_images", "deduplicated_images", "final_images"):
            value = self._debug_trace.get(key, [])
            print(f"{key}: {json.dumps(value, ensure_ascii=False)}", file=sys.stderr)

    def _parse_sections(self, content: str, title: str) -> list[tuple[str, str, int]]:
        lines = content.splitlines()
        sections: list[tuple[str, str, int]] = []
        current_heading = ""
        current_index = 0
        next_section_number = 0
        buffer: list[str] = []

        for line in lines:
            if re.match(r"^#{2,3}\s+", line.strip()):
                if buffer or current_heading:
                    sections.append((current_heading.strip(), "\n".join(buffer).strip(), current_index))
                current_heading = line.strip().lstrip("# ").strip()
                next_section_number += 1
                current_index = next_section_number
                buffer = []
            else:
                buffer.append(line)

        if buffer or current_heading:
            sections.append((current_heading.strip(), "\n".join(buffer).strip(), current_index))
        return sections or [(title, content, 0)]

    @staticmethod
    def _contains(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    def _classify(
        self,
        heading: str,
        paragraph: str,
        index: int,
        total: int,
        source_input: str = "",
    ) -> tuple[str, str, float]:
        text = f"{heading} {paragraph}".lower()
        source_text = f"{source_input} {text}".lower()
        heading_text = heading.lower()

        # Specific visual topics take precedence over broad reuse or
        # organization signals. The source is included only as authority for
        # topic detection; section text still drives the actual scene.
        specific_role = self._specific_role(source_text)
        section_topics = self._topic_terms(text)
        source_topics = self._topic_terms(source_input)
        has_section_specific = bool(section_topics["specific"])
        has_source_specific = bool(source_topics["specific"])
        if specific_role and (has_section_specific or (index == 0 and has_source_specific)):
            if specific_role == "comparison":
                return "comparison", "빈티지 의류와 구제 의류의 차이를 나란히 살펴보는 장면", 10.0
            if "y2k" in source_text and "빈티지" in source_text:
                return "styling", "Y2K 무드의 빈티지 의류를 살펴보고 조합하는 장면", 10.0
            return "styling", "빈티지 체크셔츠를 활용한 인턴 출근룩 스타일링 장면", 10.0

        heading_priority = (
            ("organization", "옷장 공간과 수납을 활용하는 장면", ("공간 활용", "공간", "옷걸이", "수납함", "선반", "바구니")),
            ("reuse", "기부하거나 다시 사용할 의류를 준비하는 장면", ("기부", "재사용", "재활용", "리폼", "새활용")),
            ("classification", "남길 옷과 정리할 옷을 기준에 따라 나누는 장면", ("기준", "분류", "나누", "구분")),
            ("detail", "얼룩과 손상, 소재 상태를 가까이 확인하는 장면", ("상태", "얼룩", "손상", "소재", "봉제", "마모")),
            ("process", "옷장에서 필요한 계절 옷을 골라 꺼내는 장면", ("꺼내", "선택", "계절에 맞춘")),
        )
        for role, purpose, markers in heading_priority:
            if any(marker in heading_text for marker in markers):
                if role == "process" and self._contains(heading_text, ("꺼내",)):
                    purpose = "옷장에서 옷을 하나씩 꺼내 모으는 장면"
                elif role == "process" and self._contains(heading_text, ("선택", "계절에 맞춘")):
                    purpose = "계절에 맞는 옷을 골라 준비하는 장면"
                if role == "detail" and self._contains(heading_text, ("소재", "봉제", "라벨", "단추", "지퍼")):
                    purpose = "소재와 봉제선, 라벨 같은 의류 구조를 가까이 확인하는 장면"
                elif role == "detail" and self._contains(heading_text, ("얼룩", "손상", "늘어남", "마모")):
                    purpose = "얼룩과 늘어남, 마모 같은 의류 상태를 가까이 확인하는 장면"
                return role, purpose, 9.0

        body_priority = (
            ("organization", "옷장 공간과 수납을 활용하는 장면", ("공간 활용", "옷걸이", "수납함", "선반", "바구니")),
            ("reuse", "기부하거나 다시 사용할 의류를 준비하는 장면", ("기부", "수거", "전달", "재사용", "재활용", "리폼", "새활용")),
            ("detail", "얼룩과 손상, 소재 상태를 가까이 확인하는 장면", ("얼룩", "손상", "소재", "봉제", "마모")),
        )
        for role, purpose, markers in body_priority:
            if any(marker in text for marker in markers):
                return role, purpose, 8.0

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

        if index == 0 and not heading and len(paragraph) >= 80:
            return "lifestyle", "옷장을 열고 여러 계절의 옷을 꺼내 펼쳐 보는 도입 장면", 1.5

        # Long, clothing-specific sections still contain useful visual
        # context even when they do not use one of the explicit markers.
        if index < total - 1 and len(paragraph) >= 80 and self._contains(text, self._FASHION_TERMS):
            return "process", "본문의 의류 관리 장면을 보여주는 과정 이미지", 2.5

        # A short intro is only a fallback hero candidate, not a reason to
        # fill the whole plan with generic lifestyle images.
        if index == 0 and len(paragraph) >= 80:
            return "lifestyle", "글의 상황을 보여주는 도입 장면", 1.5
        return "", "", 0.0

    def _select_sections(self, sections: list[tuple[str, str, int]], source_input: str = "") -> list[dict]:
        candidates = []
        for heading, paragraph, index in sections:
            role, purpose, score = self._classify(heading, paragraph, index, len(sections), source_input=source_input)
            if not role or score <= 0:
                continue
            candidates.append({
                "heading": heading,
                "paragraph": paragraph,
                "index": index,
                "role": role,
                "purpose": purpose,
                "score": score,
                "extracted_topics": self._topic_terms(f"{heading} {paragraph}"),
                "matched_concepts": [
                    marker
                    for _, _, markers in self._ROLE_RULES
                    for marker in markers
                    if marker in f"{heading} {paragraph}"
                ],
                "specific_topics": self._topic_terms(f"{heading} {paragraph}")["specific"],
                "generic_topics": self._topic_terms(f"{heading} {paragraph}")["generic"],
                "selected_rule": f"{role}_rule",
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

    def _visual_subject(
        self,
        role: str,
        paragraph: str,
        heading: str,
        blog_post: BlogPost,
        source_input: str = "",
    ) -> str:
        text = f"{source_input} {heading} {paragraph} {blog_post.title} {blog_post.keyword}".lower()
        garments = []
        for term in sorted(self._GARMENT_TERMS, key=len, reverse=True):
            if term in text and not any(term in selected for selected in garments):
                garments.append(term)
            if len(garments) >= 2:
                break
        if not garments:
            garment_label = "의류"
        elif len(garments) == 1:
            garment_label = garments[0]
        elif len(garments) == 2:
            garment_label = f"{garments[0]}{self._conjunction(garments[0])} {garments[1]}"
        else:
            garment_label = ", ".join(garments[:3])
        if role == "classification":
            return f"{garment_label}{self._object_particle(garment_label)} 두 그룹으로 나눈 정리 장면"
        if role == "comparison":
            if "빈티지" in text and "구제" in text:
                return "빈티지 의류와 구제 의류를 나란히 비교하는 장면"
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
            if self._contains(text, ("기부", "수거", "전달")):
                return "기부할 의류를 상태별로 분류해 상자에 담는 준비 과정"
            return f"기존 {garment_label}{self._object_particle(garment_label)} 새로운 용도로 활용하는 작업"
        if role == "styling":
            if "y2k" in text and "빈티지" in text:
                return "Y2K 무드의 빈티지 의류를 살펴보고 조합한 스타일링"
            if "체크셔츠" in text and ("인턴" in text or "출근룩" in text):
                return "빈티지 체크셔츠를 활용한 인턴 출근룩 스타일링"
            return f"기존 {garment_label}{self._object_particle(garment_label)} 조합한 일상 코디"
        if role == "checklist":
            return f"{garment_label}의 상태와 보관 조건을 확인하는 장면"
        if role == "process":
            if self._contains(heading.lower(), ("꺼내",)):
                return "옷장에서 옷을 하나씩 꺼내 침대 위에 모으는 장면"
            if self._contains(heading.lower(), ("선택", "계절에 맞춘")):
                return "옷장에서 가을 옷을 골라 침대 위에 모으는 장면"
            return f"{garment_label}{self._object_particle(garment_label)} 실제로 정리하고 다루는 과정"
        if role == "organization":
            if self._contains(text, ("공간", "옷걸이", "수납함", "선반", "바구니")):
                return "옷장 안 옷걸이와 수납함을 활용해 의류를 배치하는 장면"
            return "가정용 옷장과 수납된 계절 의류"
        if role == "lifestyle":
            if "y2k" in text and "빈티지" in text:
                return "빈티지 매장에서 Y2K 스타일 의류를 살펴보는 장면"
            return "옷장을 열고 여러 계절의 옷을 침대 위에 펼쳐놓은 전체 장면"
        return blog_post.title or blog_post.summary or "일상 속 의류 관리"

    @staticmethod
    def _has_final_consonant(value: str) -> bool:
        for char in reversed(value.strip()):
            if "가" <= char <= "힣":
                return (ord(char) - ord("가")) % 28 != 0
        return False

    @classmethod
    def _object_particle(cls, value: str) -> str:
        return "을" if cls._has_final_consonant(value) else "를"

    @classmethod
    def _conjunction(cls, value: str) -> str:
        return "과" if cls._has_final_consonant(value) else "와"

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
            if "기부할" in subject:
                setting = "밝은 현관 또는 정리 테이블이 있는 생활 공간"
                composition = "상태별로 분류한 옷을 상자에 접어 담고 전달 준비를 하는 손이 보이는 구도"
            else:
                setting = "작업 테이블과 옷장이 있는 생활 공간"
                composition = "기존 의류와 새 활용 결과물이 함께 보이는 과정 중심 구도"
        elif role == "styling":
            setting = "자연광이 들어오는 집 안 또는 현실적인 외출 준비 공간"
            composition = "옷의 조합과 착용 모습을 보여주는 반신 또는 전신 구도"
        elif role == "checklist":
            composition = "확인해야 할 부분이 잘 보이는 손과 의류의 detail shot"
        elif role == "organization":
            composition = "수납 전후의 공간감과 손이 닿는 옷 배치가 함께 보이는 wide shot"
        elif role == "lifestyle":
            setting = "밝은 침실과 열린 옷장이 있는 생활 공간"
            composition = "침대 위에 펼쳐진 여러 계절의 옷과 열린 옷장이 함께 보이는 넓은 사선 구도"

        return f"{subject}, {setting}, {composition}, {style}, {self._NO_TEXT}, no text, no typography, no letters, no numbers, no captions, no watermark, no logo, no readable signage, no text overlay"

    def _make_alt_text(self, role: str, subject: str, heading: str, paragraph: str) -> str:
        text = f"{heading} {paragraph}".lower()
        if role == "lifestyle":
            return "옷장에서 여러 계절의 옷을 꺼내 침대 위에 펼쳐놓은 모습"
        if role == "classification":
            if self._contains(text, ("남길", "정리할", "보관")):
                return "남길 옷과 정리할 옷을 두 그룹으로 나누는 모습"
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
            if "기부할" in subject:
                return "정리한 옷을 기부용 상자에 담는 모습"
            return "기존 의류를 새로운 용도로 활용하는 작업 과정"
        if role == "styling":
            return "기존 옷을 조합해 일상 코디를 완성한 모습"
        if role == "checklist":
            return "의류 상태와 보관 조건을 차례로 확인하는 모습"
        if role == "organization":
            return "옷걸이와 수납함을 활용해 옷장 공간을 정리하는 모습"
        if role == "process":
            if self._contains(heading.lower(), ("꺼내",)):
                return "옷장에서 옷을 하나씩 꺼내 모으는 모습"
            if self._contains(heading.lower(), ("선택", "계절에 맞춘")):
                return "가을 옷을 골라 침대 위에 모으는 모습"
            return "옷장에서 옷을 꺼내 정리하는 모습"
        return "옷을 살펴보고 정리하는 모습"

    def plan(self, blog_post: BlogPost, source_input: str = "") -> BlogPost:
        self._debug_trace = {
            # ImageAgent currently receives no raw source argument. Keeping
            # this explicit makes that missing authority visible in debug output.
            "source_visual_topics": self._topic_terms(source_input),
            "post_visual_topics": self._topic_terms(" ".join((blog_post.title, blog_post.keyword, blog_post.summary, blog_post.content))),
            "sections": [],
            "candidate_images": [],
            "deduplicated_images": [],
            "final_images": [],
        } if self._debug_enabled() else None
        sections = self._parse_sections(blog_post.content, blog_post.title)
        chosen = self._select_sections(sections, source_input=source_input)
        if self._debug_trace is not None:
            self._debug_trace["sections"] = [
                {
                    "placement": "hero-intro" if rank == 0 and info["index"] == 0 else f"section-{info['index']}",
                    "heading": info["heading"],
                    "body_summary": info["paragraph"][:160],
                    "extracted_topics": info["extracted_topics"],
                    "matched_concepts": info["matched_concepts"],
                    "specific_topics": info["specific_topics"],
                    "generic_topics": info["generic_topics"],
                    "selected_rule": info["selected_rule"],
                    "priority": info["score"],
                }
                for rank, info in enumerate(chosen)
            ]
        images: list[ImagePrompt] = []

        for rank, info in enumerate(chosen):
            role = info["role"]
            subject = self._visual_subject(
                role,
                info["paragraph"],
                info["heading"],
                blog_post,
                source_input=source_input,
            )
            placement = "hero-intro" if rank == 0 and info["index"] == 0 else f"section-{info['index']}"
            if self._debug_trace is not None:
                self._debug_trace["candidate_images"].append({
                    "placement": placement,
                    "image_type": role,
                    "subject": subject,
                    "scene": info["purpose"],
                    "purpose": info["purpose"],
                    "source_reason": f"{role}_rule",
                })
            images.append(ImagePrompt(
                placement=placement,
                purpose=info["purpose"],
                prompt=self._make_prompt(role, subject, info["heading"], blog_post),
                image_type=role,
                alt_text=self._make_alt_text(role, subject, info["heading"], info["paragraph"]),
            ))

        deduplicated_images: list[ImagePrompt] = []
        seen_scenes: set[tuple[str, str, str]] = set()
        for image in images:
            scene_key = (
                image.image_type,
                " ".join(image.purpose.split()),
                " ".join(image.prompt.split()),
            )
            if scene_key in seen_scenes:
                continue
            seen_scenes.add(scene_key)
            deduplicated_images.append(image)

        blog_post.image_plan = ImagePlan(images=deduplicated_images)
        if self._debug_trace is not None:
            self._debug_trace["deduplicated_images"] = [
                {"placement": image.placement, "image_type": image.image_type, "purpose": image.purpose, "prompt": image.prompt}
                for image in deduplicated_images
            ]
            self._debug_trace["final_images"] = [
                {"placement": image.placement, "image_type": image.image_type, "purpose": image.purpose, "alt_text": image.alt_text}
                for image in deduplicated_images
            ]
            self._emit_debug_trace()
        return blog_post
