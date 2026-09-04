from __future__ import annotations

from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import SEOResult


class SEOAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()

    def run(self, trend_results) -> List[SEOResult]:
        prompt = "리핏 블로그에 적합한 실제 검색 의도와 키워드를 구조화한다."
        payloads = [
            {"keyword": "안 입는 반팔 처리", "search_intent": "정보 탐색", "seasonality": 0.96, "content_potential": 0.91},
            {"keyword": "헌옷 처리 방법", "search_intent": "정보 탐색", "seasonality": 0.84, "content_potential": 0.9},
            {"keyword": "헌옷 버리는 법", "search_intent": "문제 해결", "seasonality": 0.78, "content_potential": 0.88},
            {"keyword": "옷장 정리 시즌별 팁", "search_intent": "정보 탐색", "seasonality": 0.94, "content_potential": 0.87},
            {"keyword": "의류 재사용 방법", "search_intent": "정보 탐색", "seasonality": 0.8, "content_potential": 0.89},
            {"keyword": "남는 옷 처리법", "search_intent": "문제 해결", "seasonality": 0.9, "content_potential": 0.92},
            {"keyword": "세탁 후 옷 보관 방법", "search_intent": "정보 탐색", "seasonality": 0.83, "content_potential": 0.86},
            {"keyword": "빈티지 의류 관리법", "search_intent": "정보 탐색", "seasonality": 0.75, "content_potential": 0.84},
            {"keyword": "니트 보풀 제거", "search_intent": "문제 해결", "seasonality": 0.7, "content_potential": 0.8},
            {"keyword": "의류 순환 방법", "search_intent": "정보 탐색", "seasonality": 0.82, "content_potential": 0.9},
        ]
        return self.client.generate_many(prompt, SEOResult, payloads)
