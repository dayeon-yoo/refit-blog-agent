from __future__ import annotations

from pathlib import Path
from typing import Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogPost, ScoredIdea


class WriterAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "writer.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def write(self, idea: ScoredIdea) -> BlogPost:
        title = idea.idea.title
        keyword = idea.idea.keyword
        search_intent = idea.idea.search_intent
        angle = idea.idea.angle
        rifit_connection = idea.idea.rifit_connection
        seasonality = idea.idea.seasonality
        critic_score = idea.total_score
        critic_reason = idea.reason
        brand_eval = idea.brand_evaluation
        brand_guideline = (
            "의류 순환, 헌옷 수거, 재사용, 재활용, 의류 관리 중심의 정보성 콘텐츠를 유지하고, "
            "과도한 광고 표현을 피하되 실용적인 도움이 되는 문장으로 작성한다."
        )
        brand_note = (
            f"브랜드 평가 기준은 '{brand_eval.rationale}'이며, 현재 적합도는 {brand_eval.fit_score:.2f}입니다."
            if brand_eval
            else "브랜드 적합도는 안정적인 수준으로 유지됩니다."
        )

        content = (
            f"## {title}\n\n"
            f"이 콘텐츠는 '{keyword}'라는 검색 의도에서 출발한다. "
            f"사람들은 보통 옷장 정리 시점에 '아직 입을 수 있는지', '버려야 하는지', '다시 활용할 수 있는지'를 고민한다. "
            f"이 글의 핵심은 단순히 정리를 추천하는 것이 아니라, {angle.lower()}라는 점이다.\n\n"
            "처음부터 무조건 버리기보다, 감정적으로 오래 붙어 있던 옷을 다시 분류해보는 것이 중요하다. "
            "최근 몇 달 동안 입지 않았다면 보관 여부를 다시 점검하고, 상태가 좋다면 재사용 또는 순환으로 연결할 수 있는 경로를 생각해보는 것이 현실적이다.\n\n"
            "의류를 정리할 때 가장 큰 실수는 '다시 입을 수 있나'만 보면서 끝내는 것이다. "
            "오히려 상태와 사용 빈도를 기준으로 분류하면, 보관, 기부, 재판매, 재활용의 우선순위를 더 자연스럽게 정할 수 있다. "
            "특히 계절 변화가 큰 시점에는 정리의 감각이 더 중요해진다.\n\n"
            f"{rifit_connection}은 이렇게 실생활에서 자연스럽게 연결될 수 있다. "
            f"{brand_note} {brand_guideline} "
            f"이런 관점은 시즌성({seasonality})이 높은 콘텐츠일수록 더 큰 도움이 된다.\n\n"
            f"결국 {keyword}와 같은 문제는 단순한 정리보다, 의류를 어떻게 순환시키느냐가 더 중요하다. "
            "지금 당장 옷장을 열어보고, 다시 입지 않는 옷을 한 번 분류해보면 된다. "
            "그 과정은 더 이상 큰 부담이 아니라, 의류를 오래 쓰는 작은 습관으로 이어질 수 있다."
        )
        summary = (
            f"{title}은 실제 사용자 문제를 기준으로 시작하는 정보형 콘텐츠로, "
            f"검색 의도({search_intent})와 시즌성({seasonality})을 반영해 의류 정리와 순환을 연결한다. "
            f"평가 점수는 {critic_score}점이며, 핵심 이유는 '{critic_reason}'이다. "
            f"브랜드 기준도 함께 반영되어 '{brand_note}'가 유지된다."
        )
        cta = "다음 옷장 정리 시점에는 '버릴지'보다 '어떻게 순환시킬지'를 먼저 생각해보세요."
        return BlogPost(title=title, keyword=keyword, content=content, summary=summary, cta=cta)
