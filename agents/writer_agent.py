from __future__ import annotations

from typing import Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogPost, ScoredIdea


class WriterAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()

    def write(self, idea: ScoredIdea) -> BlogPost:
        title = idea.idea.title
        keyword = idea.idea.keyword
        content = (
            f"## {title}\n\n"
            "여름이 끝나고 옷장을 정리할 시기가 되면, 가장 먼저 떠오르는 고민은 '아직 입을 수 있는 옷'과 '버려야 할 옷'을 구분하는 일입니다. "
            "하지만 모든 의류를 무조건 버리는 것이 정답은 아닙니다. 몇 번 입지 않았더라도 상태가 좋은 옷은 재사용이나 순환의 흐름으로 이어질 수 있습니다.\n\n"
            "먼저 정리할 때는 세 가지 기준을 생각해보면 좋습니다. 먼저, 최근 1년 동안 입은 적이 있는지, 둘째, 상태가 양호한지, 셋째, 다시 입을 의향이 있는지입니다. "
            "이 기준을 기준으로 분류하면, 버릴지 보관할지, 기부할지, 재판매할지 선택의 폭이 좁혀집니다.\n\n"
            "의류를 처리할 때는 '버리기'보다 '재사용'과 '순환'을 먼저 고려하는 방식이 더 현실적입니다. "
            "해당 옷이 아직 사용 가능한 상태라면 중고 거래, 기부, 재사용 커뮤니티 등을 통해 다시 쓰이는 경로를 찾는 것이 좋습니다. "
            "반대로 손상되었거나 세탁이 어렵다면 재활용을 고려하는 것이 더 적합합니다.\n\n"
            "의류 정리는 단순히 옷장 정리를 넘어서, 생활 속에서 불필요한 소비를 줄이는 행동이기도 합니다. "
            "이 과정에서 가장 중요한 건 '다시 입을지'가 아니라 '다시 쓰일 수 있는지'를 기준으로 바라보는 것입니다. "
            "이렇게 의류 순환을 자연스럽게 생각하면, 한 번 입고 버리는 흐름보다 더 오래 사용할 수 있는 방법을 선택하게 됩니다.\n\n"
            "리핏은 의류의 새로운 순환을 돕는 방향으로 서비스를 이해할 수 있습니다. "
            "의류를 처분하는 시점에서 단순 폐기와 재사용 사이를 연결하는 고민은 실제 사용자에게 더 큰 도움이 됩니다. "
            "자연스러운 정리 습관과 의류 순환의 가치는 각자 다른 방식으로 시작할 수 있지만, 작은 선택 하나가 장기적으로는 큰 차이를 만들 수 있습니다.\n\n"
            "결국 의류 정리는 '버리는 것'이 아니라, 다음 주인에게 가는 흐름을 만드는 과정에 가깝습니다. "
            "오늘 옷장을 여는 순간, 더 이상 입지 않는 옷이 새롭게 의미를 찾는 계기가 되도록 한 번쯤 생각해보면 좋습니다."
        )
        summary = (
            "계절 변화 시기에는 남는 의류를 어떻게 처리할지 고민이 커지기 마련입니다. "
            "이 글은 의류 정리의 기준과 재사용·재활용의 실질적 방향을 함께 제시합니다."
        )
        cta = "다음 번 옷장 정리 시점에는 한 번 더 살펴보고, 오래 입을 수 있는 의류와 순환이 필요한 의류를 구분해보세요."
        return BlogPost(title=title, keyword=keyword, content=content, summary=summary, cta=cta)
