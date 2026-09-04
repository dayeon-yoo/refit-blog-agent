from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogIdea


class IdeaGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "idea.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

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

        if not trend_results and not seo_results:
            return self.client.generate_many(prompt, BlogIdea, self._fallback_candidates())

        trend_topics = [trend.topic for trend in (trend_results or [])]
        seo_keywords = [seo.keyword for seo in (seo_results or [])]
        generated = []
        title_templates = [
            "{topic} 어떻게 처리할까?",
            "{topic}, 이럴 때는 의류 순환이 더 낫다",
            "{topic} 전 체크할 것",
            "{topic}과 헌옷 수거, 어떤 흐름이 맞을까?",
            "{topic}로 정리하는 현실적인 방법",
        ]

        for idx, trend in enumerate((trend_results or [])[:5]):
            keyword = seo_keywords[idx] if idx < len(seo_keywords) else trend.topic
            title = title_templates[idx % len(title_templates)].format(topic=trend.topic)
            generated.append(
                {
                    "title": title,
                    "keyword": keyword,
                    "search_intent": seo_results[idx].search_intent if seo_results and idx < len(seo_results) else "정보 탐색",
                    "angle": f"{trend.topic}에 대한 현실적인 문제 해결과 의류 순환 관점의 정리",
                    "rifit_connection": "헌옷 수거, 의류 재사용, 재활용과 연결되는 실용적 콘텐츠",
                    "seasonality": round(min(max(trend.relevance_score, 0.5), 0.99), 2),
                }
            )

        for idx, seo in enumerate((seo_results or [])[:5]):
            if len(generated) >= 10:
                break
            generated.append(
                {
                    "title": f"{seo.keyword}, 리핏 관점에서 정리해보면?",
                    "keyword": seo.keyword,
                    "search_intent": seo.search_intent,
                    "angle": f"{seo.keyword}에 대한 실제 사용자 고민을 해결하는 정보형 콘텐츠",
                    "rifit_connection": "의류 순환과 지속가능한 패션 관점으로 연결",
                    "seasonality": round(min(max(seo.seasonality, 0.5), 0.99), 2),
                }
            )

        unique_candidates = []
        seen = set()
        for item in generated:
            key = item["title"]
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(item)

        while len(unique_candidates) < 10:
            unique_candidates.append(
                {
                    "title": f"의류 정리와 순환, {len(unique_candidates) + 1}가지 현실 팁",
                    "keyword": "의류 순환 방법",
                    "search_intent": "정보 탐색",
                    "angle": "생활 속 정리 습관과 의류 순환을 연결하는 실용 지침",
                    "rifit_connection": "의류 정리와 수거의 자연스러운 연결",
                    "seasonality": 0.85,
                }
            )

        return self.client.generate_many(prompt, BlogIdea, unique_candidates[:10])
