from __future__ import annotations

from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogIdea


class IdeaGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()

    def generate(self, trend_results, seo_results, brand_evaluations) -> List[BlogIdea]:
        base = [
            {
                "title": "여름 지나고 남은 반팔, 어떻게 처리할까?",
                "keyword": "안 입는 반팔 처리",
                "search_intent": "정보 탐색",
                "angle": "사용하지 않는 여름 의류의 처리 방법을 비교하고 시기별 정리 전략 제시",
                "rifit_connection": "헌옷 수거 및 의류 순환",
                "seasonality": 0.96,
            },
            {
                "title": "헌옷 버리는 법, 이럴 때는 의류 순환이 더 낫다",
                "keyword": "헌옷 처리 방법",
                "search_intent": "문제 해결",
                "angle": "의류를 버리기보다 재사용과 재판매, 재활용을 고려하는 가이드",
                "rifit_connection": "의류 순환과 수거 서비스",
                "seasonality": 0.88,
            },
            {
                "title": "한 번 입은 옷, 어떻게 재사용할 수 있을까?",
                "keyword": "의류 재사용 방법",
                "search_intent": "정보 탐색",
                "angle": "짧게 쓰고 버리는 패턴을 줄이고 오래 쓰는 실천법 제안",
                "rifit_connection": "재사용과 재판매의 가치를 강조",
                "seasonality": 0.81,
            },
            {
                "title": "옷장 정리 시즌, 남는 옷은 이렇게 처리해보세요",
                "keyword": "옷장 정리 시즌별 팁",
                "search_intent": "정보 탐색",
                "angle": "계절 전환 시점에 필요한 정리와 의류 분리 기준 안내",
                "rifit_connection": "계절별 의류 정리와 수거 연결",
                "seasonality": 0.94,
            },
            {
                "title": "가을에 버려지는 니트와 가디건, 처리 전 체크할 것",
                "keyword": "의류 관리법",
                "search_intent": "문제 해결",
                "angle": "보유 의류를 재사용 또는 재활용으로 전환하기 위한 체크리스트 제공",
                "rifit_connection": "관리와 순환을 연결하는 실용적 콘텐츠",
                "seasonality": 0.87,
            },
            {
                "title": "옷장 속 미사용 의류, 왜 정리해야 할까?",
                "keyword": "남는 옷 처리법",
                "search_intent": "문제 해결",
                "angle": "미사용 의류를 정리하고, 더 나은 순환 구조로 넘기는 이유 설명",
                "rifit_connection": "의류 순환의 실질적 의미",
                "seasonality": 0.9,
            },
            {
                "title": "의류 재활용을 고민한다면, 먼저 이 기준부터 보세요",
                "keyword": "의류 재활용 아이디어",
                "search_intent": "정보 탐색",
                "angle": "재활용 전 고려할 상태 점검과 활용 아이디어 제공",
                "rifit_connection": "재활용을 지원하는 서비스 맥락",
                "seasonality": 0.79,
            },
            {
                "title": "세탁 후 옷 보관, 의류 관리로 오래 입는 법",
                "keyword": "세탁 후 옷 보관 방법",
                "search_intent": "정보 탐색",
                "angle": "보관 전 점검과 의류 유지 관리 팁 제안",
                "rifit_connection": "의류 관리와 순환의 연장선",
                "seasonality": 0.82,
            },
            {
                "title": "빈티지 의류는 어떻게 오래 관리할까?",
                "keyword": "빈티지 의류 관리법",
                "search_intent": "정보 탐색",
                "angle": "빈티지 의류의 보관과 세탁 기준을 현실적으로 설명",
                "rifit_connection": "빈티지 의류의 재사용과 관리",
                "seasonality": 0.74,
            },
            {
                "title": "옷장 정리와 헌옷 수거, 어떤 순서가 맞을까?",
                "keyword": "의류 순환 방법",
                "search_intent": "정보 탐색",
                "angle": "정리, 분류, 수거, 재사용의 단계별 흐름 설명",
                "rifit_connection": "정리에서 수거까지의 전체 사이클",
                "seasonality": 0.86,
            },
        ]
        return self.client.generate_many("blog ideas", BlogIdea, base)
