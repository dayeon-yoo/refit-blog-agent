from __future__ import annotations

from typing import List, Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import TrendResult


class TrendAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()

    def run(self) -> List[TrendResult]:
        prompt = "최근 리핏 블로그 콘텐츠로 확장 가능한 의류 순환 트렌드 주제를 식별한다."
        payloads = [
            {
                "topic": "가을 시즌 의류 정리",
                "reason": "계절 전환으로 여름 의류를 정리하고 재사용하거나 재활용하려는 수요가 증가한다.",
                "source": "계절성 소비 패턴",
                "relevance_score": 0.95,
            },
            {
                "topic": "안 입는 반팔 처리 방법",
                "reason": "여름 끝 무렵, 사용하지 않는 반팔을 어떻게 처리할지에 대한 실질적 고민이 지속된다.",
                "source": "생활정보 검색 트렌드",
                "relevance_score": 0.92,
            },
            {
                "topic": "의류 재사용과 리셀 팁",
                "reason": "옷을 버리기보다 활용하고 싶어 하는 소비자 관심이 높다.",
                "source": "지속가능한 패션 관심",
                "relevance_score": 0.9,
            },
            {
                "topic": "겨울 옷 보관 전 점검",
                "reason": "계절 변화 전 보관 전략을 준비하는 수요가 꾸준히 존재한다.",
                "source": "가계 생활관리 관심",
                "relevance_score": 0.88,
            },
            {
                "topic": "빈티지 의류 관리법",
                "reason": "빈티지 의류에 대한 관심을 유지할 수 있는 관리·세탁 팁은 항상 유효하다.",
                "source": "패션 콘텐츠 리서치",
                "relevance_score": 0.86,
            },
            {
                "topic": "오래된 니트 처리 방법",
                "reason": "계절이 바뀌면 니트, 스웨터, 가디건 같은 의류를 정리하거나 재사용하는 고민이 함께 발생한다.",
                "source": "의류 관리 검색 패턴",
                "relevance_score": 0.87,
            },
            {
                "topic": "헌옷 수거와 의류 순환",
                "reason": "의류를 단순 폐기 대신 순환하는 흐름에 대한 관심이 꾸준하다.",
                "source": "의류 순환 커뮤니티",
                "relevance_score": 0.94,
            },
            {
                "topic": "여름 의류 정리 시기",
                "reason": "계절별 정리 시점에 대한 실제 수요가 존재한다.",
                "source": "생활정보 문맥",
                "relevance_score": 0.89,
            },
            {
                "topic": "의류 재활용 아이디어",
                "reason": "버려지는 의류를 새롭게 활용하는 방법은 지속가능한 콘텐츠로 확장성이 높다.",
                "source": "지속가능 패션 키워드",
                "relevance_score": 0.85,
            },
            {
                "topic": "옷장 정리와 의류 수거의 연결",
                "reason": "옷장 정리와 관련된 콘텐츠는 자연스럽게 수거 및 순환 메시지로 확장할 수 있다.",
                "source": "생활정리 콘텐츠",
                "relevance_score": 0.91,
            },
        ]
        return self.client.generate_many(prompt, TrendResult, payloads)
