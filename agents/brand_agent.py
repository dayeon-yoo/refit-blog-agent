from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogIdea, BrandEvaluation


class BrandAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "brand.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def run(self, ideas: List[BlogIdea]) -> List[BrandEvaluation]:
        prompt = self.prompt
        guidelines = {
            "themes": ["의류 순환", "헌옷 수거", "의류 재사용", "재판매", "재활용", "빈티지 의류", "의류 관리", "지속가능한 패션"],
            "content_rules": [
                "정보성 중심",
                "실제 사용자 문제 해결",
                "친근하고 이해하기 쉬운 표현",
                "과도한 환경 캠페인 느낌 금지",
                "노골적인 광고성 금지",
                "억지스러운 리핏 연결 금지",
            ],
        }

        evaluations = []
        for idea in ideas:
            fit = 0.5
            if any(keyword in idea.rifit_connection.lower() for keyword in ["수거", "순환", "재사용", "재활용", "관리"]):
                fit += 0.35
            if any(keyword in idea.title.lower() for keyword in ["정리", "처리", "관리", "순환", "헌옷", "재사용", "재활용"]):
                fit += 0.1
            if "코디" in idea.title.lower() or "패션" in idea.title.lower():
                fit -= 0.2

            fit = max(0.0, min(1.0, round(fit, 2)))
            brand_alignment = "high" if fit >= 0.8 else "medium" if fit >= 0.6 else "low"
            rationale = (
                f"{idea.title}은 실제 사용자 고민과 {', '.join(guidelines['themes'][:3])} 흐름을 연결하는 측면에서 "
                f"리핏의 브랜드 방향성과 자연스럽게 맞물린다."
                if brand_alignment != "low"
                else "패션 톤은 매력적이지만 리핏의 핵심 가치와 연결을 더 명확히 해야 한다."
            )
            evaluations.append(
                {
                    "topic": idea.title,
                    "fit_score": fit,
                    "rationale": rationale,
                    "brand_alignment": brand_alignment,
                }
            )

        return self.client.generate_many(prompt, BrandEvaluation, evaluations)
