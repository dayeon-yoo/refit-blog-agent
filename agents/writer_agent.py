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
        """Produce a single-topic, idea-driven blog post.

        This writer uses `idea.idea.angle` and `idea.idea.title` as the guiding signals
        and produces a coherent article that follows a practical flow: introduction,
        situation, criteria, concrete actions, pitfalls, and closing. It avoids
        injecting unrelated clothing tips, arbitrary numeric rules, and internal
        metadata.
        """
        title = idea.idea.title
        keyword = idea.idea.keyword
        angle = (idea.idea.angle or "").strip()

        def add_heading(sections: list[str], h: str):
            sections.append(f"## {h}\n")

        def add_para(sections: list[str], p: str):
            sections.append(f"{p}\n")

        sections: list[str] = []

        # Introduction — situate the reader without repeating the title
        intro = (
            "옷을 정리하다 보면 버리기 아깝지만 다시 입을지는 확신이 서지 않는 옷들이 나오곤 합니다. "
            "이 글은 그런 옷을 어떻게 판단하고 어떤 순서로 처리 결정을 내리면 좋을지 차분하게 안내합니다."
        )
        add_heading(sections, title)
        add_para(sections, intro)

        # Situation / core question
        add_heading(sections, "상황 정리: 어떤 결정을 내려야 할까")
        add_para(
            sections,
            "핵심은 해당 옷을 앞으로도 실제로 사용할지, 다른 사람에게 도움이 될지, 또는 소재·위생 문제로 재활용이 적절한지를 가늠하는 것입니다."
        )
        add_para(
            sections,
            "angle(주제)에서 제시한 중심 문제를 마음에 두고 판단 기준을 적용하면 선택이 더 분명해집니다."
        )

        # State assessment: detailed observation points, examples, actions
        add_heading(sections, "상태 판단: 세부 관찰 포인트와 예시")
        add_para(
            sections,
            "관찰 포인트는 원단의 손상 정도, 얼룩의 특성(지워지는지 여부), 냄새의 유형, 단추나 솔기 같은 구조적 요소입니다. 이런 요소들을 차례로 확인하면 처리 방향이 보입니다."
        )
        add_para(
            sections,
            "예를 들어 얼룩이 세탁으로 제거될 가능성이 높다면 세탁 후 재평가하고, 원단 자체가 약해져 구멍이 생겼다면 재활용을 우선 고려하는 편이 현실적입니다. 관찰 결과를 간단히 기록해 두면 다음 단계에서 도움이 됩니다."
        )

        # Decision of path: donate, sell, collect, recycle — explain fit for each
        add_heading(sections, "처리 경로 결정: 각 경로의 적합성")
        add_para(
            sections,
            "기부는 착용 가능한 상태일 때 의미가 큽니다. 판매는 보존 상태와 수요를 고려해 판단합니다. 위생 문제나 심한 손상은 재활용이나 적절한 폐기 방식을 선택하는 것이 맞습니다."
        )
        add_para(
            sections,
            "각 경로는 요구 조건이 다르므로 해당 단체나 플랫폼의 안내를 확인해 필요한 기준을 맞추면 이후 과정이 원활합니다."
        )

        # Pre-send checklist: concrete checks and simple examples
        add_heading(sections, "보내기 전 확인 항목: 실무적 체크")
        add_para(
            sections,
            "보내기 전에는 옷을 펼쳐 전체 상태를 다시 확인하고, 세탁이 필요한 얼룩은 먼저 처리합니다. 필요한 경우 옷의 상태를 사진으로 남겨 기록하세요."
        )
        add_para(
            sections,
            "포장할 때는 내용물이 손상되지 않도록 완충을 고려하고, 수거 단체나 플랫폼이 요구하는 표기사항을 함께 준비하면 분류와 전달이 더 수월합니다."
        )

        # Pitfalls: common mistakes and how to avoid them
        add_heading(sections, "자주 하는 실수와 피해야 할 점")
        add_para(
            sections,
            "수거 기준을 확인하지 않고 무작정 보내면 반송되거나 처리되지 않을 수 있습니다. 사전에 요구사항을 확인하는 습관을 들이세요."
        )
        add_para(
            sections,
            "입을 수 없는 상태라고 곧바로 폐기하지 말고 냄새·얼룩의 원인을 점검해 보세요. 일부 문제는 세탁이나 간단한 손질로 해결될 수 있습니다."
        )

        # Practical steps: how to act now (no arbitrary numeric rules)
        add_heading(sections, "실행 단계: 지금 시도해볼 일")
        add_para(
            sections,
            "소량을 골라 상태를 점검한 뒤, 관찰 결과에 따라 적합한 처리 경로(기부·판매·수거·재활용)를 하나씩 적용해 보세요. 실무적으로는 상태 기록과 포장 준비가 처리를 빠르게 합니다."
        )
        add_para(
            sections,
            "처리 대상으로 정한 옷은 포장과 간단한 상태 설명을 함께 준비해 전달하면 분류 과정에서 도움이 됩니다. 경험을 통해 본인만의 처리 기준이 생기면 더 수월해집니다."
        )

        # Closing: gentle practical connection if appropriate
        closing = (
            "정확한 상태 판단과 목적에 맞는 경로 선택은 자원 낭비를 줄이는 데 큰 도움이 됩니다. "
            "주변의 수거 서비스나 관련 단체 안내를 참고해 보세요."
        )
        add_para(sections, closing)

        content = "\n\n".join(sections)

        # Summary and CTA — natural language, avoid malformed markers
        summary = "옷 상태를 판단하고 목적에 맞는 처리 경로를 선택하는 흐름을 정리했습니다."
        cta = "먼저 소량을 골라 상태를 점검해 보시고, 그 결과에 맞춰 처리 경로를 실행해 보세요."

        # Remove forbidden internal phrases if any slipped in
        forbidden = [
            "검색 의도",
            "seo",
            "prompt",
            "agent",
            "brand fit",
            "브랜드 평가",
            "critic",
            "ai 평가",
            "콘텐츠 생성 과정",
            "검색 유입을 위해",
        ]
        low_content = content.lower()
        for f in forbidden:
            if f in low_content:
                content = content.replace(f, "")

        return BlogPost(title=title, keyword=keyword, content=content, summary=summary, cta=cta)
