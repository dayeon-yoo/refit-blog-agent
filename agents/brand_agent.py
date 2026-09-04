from __future__ import annotations

from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BrandEvaluation


class BrandAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()

    def run(self, ideas: List[str]) -> List[BrandEvaluation]:
        prompt = "리핏 브랜드 방향성과 연결되는지 평가한다."
        evaluations = [
            {
                "topic": "여름 지나고 남은 반팔, 어떻게 처리할까?",
                "fit_score": 0.97,
                "rationale": "계절성, 사용자 문제, 의류 처리, 의류 순환이 자연스럽게 연결된다.",
                "brand_alignment": "high",
            },
            {
                "topic": "헌옷 버리는 법, 리핏이 추천하는 의류 순환 흐름",
                "fit_score": 0.96,
                "rationale": "리핏의 핵심 가치를 설명하면서도 정보성을 유지할 수 있다.",
                "brand_alignment": "high",
            },
            {
                "topic": "의류 재사용, 왜 버리기보다 순환이 좋은가",
                "fit_score": 0.92,
                "rationale": "지속가능한 패션과 행동 가이드가 연결되며 브랜드 방향성과 부합한다.",
                "brand_alignment": "high",
            },
            {
                "topic": "옷장 정리 시즌별 체크리스트",
                "fit_score": 0.84,
                "rationale": "실용적이지만 리핏 연결을 더 잘 살려야 한다.",
                "brand_alignment": "medium",
            },
            {
                "topic": "가을 코디 10가지",
                "fit_score": 0.42,
                "rationale": "패션 콘텐츠는 매력적이지만 리핏의 서비스와 직접적 연결은 약하다.",
                "brand_alignment": "low",
            },
        ]
        return self.client.generate_many(prompt, BrandEvaluation, evaluations)
