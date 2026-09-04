from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogIdea, BrandEvaluation, ScoredIdea


class CriticAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "critic.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def score(self, ideas: List[BlogIdea], brand_evaluations: Optional[List[BrandEvaluation]] = None) -> List[ScoredIdea]:
        prompt = self.prompt
        eval_map = {evaluation.topic: evaluation for evaluation in (brand_evaluations or [])}
        scored = []
        for index, idea in enumerate(ideas, start=1):
            brand_eval = eval_map.get(idea.title)
            brand_fit_score = brand_eval.fit_score if brand_eval else 0.6
            search_intent = 18 if "정보 탐색" in idea.search_intent or "문제 해결" in idea.search_intent else 15
            informativeness = 18 if len(idea.angle) > 20 else 15
            seasonality = min(15, max(8, int(idea.seasonality * 20)))
            brand_fit = int(round(brand_fit_score * 20))
            differentiation = 12 if any(keyword in idea.title.lower() for keyword in ["처리", "정리", "순환", "헌옷", "재사용", "보관"]) else 10
            expandability = 9 if any(keyword in idea.title.lower() for keyword in ["정리", "처리", "관리", "순환", "보관"]) else 8
            scores = {
                "search_intent": search_intent,
                "informativeness": informativeness,
                "seasonality": seasonality,
                "brand_fit": brand_fit,
                "differentiation": differentiation,
                "expandability": expandability,
            }
            total = sum(scores.values())
            reason = (
                f"{idea.title}은 사용자가 겪는 실제 문제를 다루고 있어 검색 의도와 정보성을 확보했고, "
                f"브랜드 적합도는 {brand_fit_score:.2f}로 리핏의 의류 순환 메시지와 자연스럽게 연결된다."
            )
            scored.append(
                ScoredIdea(
                    idea_id=f"idea-{index}",
                    scores=scores,
                    total_score=total,
                    reason=reason,
                    idea=idea,
                    brand_evaluation=brand_eval,
                )
            )
        return sorted(scored, key=lambda item: item.total_score, reverse=True)

    def top_3(self, ideas: List[BlogIdea], brand_evaluations: Optional[List[BrandEvaluation]] = None) -> List[ScoredIdea]:
        return self.score(ideas, brand_evaluations)[:3]
