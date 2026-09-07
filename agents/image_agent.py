from __future__ import annotations

from pathlib import Path
from typing import Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogPost, ImagePlan, ImagePrompt


class ImageAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "image.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()
        # init does not perform planning; call `plan()` with a BlogPost

    def plan(self, blog_post: BlogPost) -> BlogPost:
        content = blog_post.content

        # Parse content into sections by headings (lines starting with '## ')
        lines = content.splitlines()
        sections: list[tuple[str, str]] = []  # (heading, paragraph_text)
        current_heading = ""
        buffer: list[str] = []

        for line in lines:
            if line.strip().startswith("## "):
                # flush buffer
                if buffer or current_heading:
                    sections.append((current_heading.strip(), "\n".join(buffer).strip()))
                current_heading = line.strip().lstrip("# ").strip()
                buffer = []
            else:
                buffer.append(line)

        # flush last
        if buffer or current_heading:
            sections.append((current_heading.strip(), "\n".join(buffer).strip()))

        # If no headings detected, treat whole content as single section
        title = blog_post.title
        if not sections:
            sections = [(title, content)]

        # scoring and keyword matching to decide images
        keyword_map = {
            "정리": ("wardrobe", "정리/옷장"),
            "옷장": ("wardrobe", "정리/옷장"),
            "세탁": ("detail", "세탁/건조"),
            "건조": ("detail", "세탁/건조"),
            "수선": ("instructional", "수선/리폼"),
            "리폼": ("instructional", "수선/리폼"),
            "기부": ("reuse", "기부/재사용"),
            "중고": ("resale", "중고/리셀"),
            "리셀": ("resale", "중고/리셀"),
            "빈티지": ("editorial", "빈티지/스타일"),
            "비교": ("infographic", "비교/선택"),
            "체크": ("infographic", "체크리스트/기준"),
        }

        candidates: list[tuple[float, dict]] = []

        for idx, (heading, para) in enumerate(sections):
            text = (heading + " " + para).lower()
            score = max(1.0, len(para) / 80.0)
            matched = None
            for k, v in keyword_map.items():
                if k in text:
                    score += 5.0
                    matched = v
            candidates.append((score, {"heading": heading or title, "para": para, "matched": matched, "index": idx}))

        # choose up to 5 images, prefer higher score
        candidates.sort(key=lambda x: x[0], reverse=True)
        max_images = min(5, max(1, len(candidates)))
        chosen = candidates[:max_images]

        images: list[ImagePrompt] = []

        for rank, (_, info) in enumerate(chosen, 1):
            heading = info["heading"]
            para = info["para"]
            matched = info["matched"]

            # short subject from paragraph
            subject = para.split(". ")[0].strip() if para else heading
            if len(subject) > 120:
                subject = subject[:120].rsplit(" ", 1)[0] + "..."

            if matched:
                image_type_hint, purpose_short = matched
            else:
                # default to lifestyle for general sections
                image_type_hint, purpose_short = ("lifestyle", "실제 생활 장면")

            placement = "body"
            if rank == 1:
                placement = "hero-intro"

            # Build a detailed, content-aware prompt
            prompt_parts = [f"무엇: {subject}"]

            # action hints
            if any(w in para.lower() for w in ["정리", "정리하", "분류", "나누"]):
                prompt_parts.append("행동: 옷장/수납 공간에서 사람(또는 손)이 옷을 분류하고 정리하는 모습")
            if any(w in para.lower() for w in ["세탁", "건조", "상태", "얼룩", "손상"]):
                prompt_parts.append("행동: 옷의 상태를 클로즈업으로 확인하는 장면(얼룩, 늘어남 등 디테일 확인)")
            if any(w in para.lower() for w in ["수선", "리폼", "바느질"]):
                prompt_parts.append("행동: 손으로 바느질하거나 수선하는 가까운 샷")
            if any(w in para.lower() for w in ["기부", "기부 상자", "기부함"]):
                prompt_parts.append("행동: 기부 상자나 가방에 옷을 넣는 장면")
            if any(w in para.lower() for w in ["중고", "리셀", "판매"]):
                prompt_parts.append("행동: 제품을 촬영하거나 스마트폰으로 리스트 작성하는 모습")

            # setting/composition/mood
            prompt_parts.append("장소: 실제 가정의 침실 또는 옷장 앞, 자연광 또는 부드러운 실내조명")
            prompt_parts.append("구도: 인물의 전신 또는 반신(생활감 있는 구도), 또는 클로즈업(제품 디테일)")
            prompt_parts.append("분위기: 현실적이고 따뜻한 톤, 과장 없는 다큐멘터리 스타일 사진")
            prompt_parts.append("스타일: 모바일 블로그용 자연스러운 스틸 사진, 과도한 후처리나 텍스트 오버레이 금지")

            prompt = ", ".join(prompt_parts)

            alt_text = f"{heading}: {subject}" if heading else subject

            images.append(ImagePrompt(placement=placement, purpose=purpose_short, prompt=prompt, image_type=image_type_hint, alt_text=alt_text))

        blog_post.image_plan = ImagePlan(images=images)
        return blog_post
