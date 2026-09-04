from __future__ import annotations

from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogIdea, ScoredIdea


class CriticAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()

    def score(self, ideas: List[BlogIdea]) -> List[ScoredIdea]:
        scored = []
        for index, idea in enumerate(ideas, start=1):
            scores = {
                "search_intent": 18 if "정보 탐색" in idea.search_intent or "문제 해결" in idea.search_intent else 15,
                "informativeness": 18,
                "seasonality": int(idea.seasonality * 20),
                "brand_fit": 20 if "수거" in idea.rifit_connection or "순환" in idea.rifit_connection else 16,
                "differentiation": 12 if "처리" in idea.title or "정리" in idea.title else 10,
                "expandability": 9,
            }
            total = sum(scores.values())
            reason = (
                "실제 사용자 문제와 계절 변화가 연결되어 있어 검색 의도와 정보를 모두 잡기 좋고, "
                "리핏의 의류 순환 메시지와도 자연스럽게 이어진다."
            )
            scored.append(
                ScoredIdea(
                    idea_id=f"idea-{index}",
                    scores=scores,
                    total_score=total,
                    reason=reason,
                    idea=idea,
                )
            )
        return sorted(scored, key=lambda item: item.total_score, reverse=True)

    def top_3(self, ideas: List[BlogIdea]) -> List[ScoredIdea]:
        return self.score(ideas)[:3]
