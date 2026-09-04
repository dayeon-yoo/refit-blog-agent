from __future__ import annotations

from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import SEOResult


class SEOAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()

    def _keyword_for_topic(self, topic: str) -> str:
        keyword_map = {
            "가을 시즌 의류 정리": "옷장 정리 시즌별 팁",
            "안 입는 반팔 처리 방법": "안 입는 반팔 처리",
            "의류 재사용과 리셀 팁": "의류 재사용 방법",
            "겨울 옷 보관 전 점검": "세탁 후 옷 보관 방법",
            "빈티지 의류 관리법": "빈티지 의류 관리법",
            "오래된 니트 처리 방법": "니트 보풀 제거",
            "헌옷 수거와 의류 순환": "헌옷 처리 방법",
            "여름 의류 정리 시기": "남는 옷 처리법",
            "의류 재활용 아이디어": "의류 재활용 아이디어",
            "옷장 정리와 의류 수거의 연결": "의류 순환 방법",
        }
        return keyword_map.get(topic, topic)

    def run(self, trend_results) -> List[SEOResult]:
        prompt = "Trend context를 바탕으로 리핏 블로그에 적합한 검색 키워드를 생성한다."

        if not trend_results:
            fallback_payloads = [
                {"keyword": "안 입는 반팔 처리", "search_intent": "문제 해결", "seasonality": 0.96, "content_potential": 0.91},
                {"keyword": "헌옷 처리 방법", "search_intent": "문제 해결", "seasonality": 0.84, "content_potential": 0.9},
            ]
            return self.client.generate_many(prompt, SEOResult, fallback_payloads)

        payloads = []
        for trend in trend_results:
            keyword = self._keyword_for_topic(trend.topic)
            intent = "정보 탐색" if "처리" not in keyword and "법" not in keyword else "문제 해결"
            payloads.append(
                {
                    "keyword": keyword,
                    "search_intent": intent,
                    "seasonality": round(min(max(trend.relevance_score, 0.5), 0.99), 2),
                    "content_potential": round(min(max(trend.relevance_score * 0.96, 0.6), 0.99), 2),
                }
            )

        if not payloads:
            return self.run([])

        return self.client.generate_many(prompt, SEOResult, payloads)
