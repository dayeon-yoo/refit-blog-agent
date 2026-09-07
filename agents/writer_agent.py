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

        # Use brand/critic context internally to guide tone, but never expose it.
        brand_eval = getattr(idea, "brand_evaluation", None)
        critic_reason = getattr(idea, "reason", None)

        # Choose an article structure based on angle/keyword hints
        angle = idea.idea.angle or "정보형"
        lower = lambda s: (s or "").lower()

        sections: list[str] = []

        def add_heading(h: str):
            sections.append(f"## {h}\n")

        def add_paragraph(p: str):
            sections.append(f"{p}\n")

        # Intro
        intro_line = (
            f"{title}에 대해 고민하고 계신가요? "
            "일상에서 바로 적용할 수 있는 실용적 기준과 행동을 중심으로 정리해드릴게요."
        )
        add_heading(title)
        add_paragraph(intro_line)

        # Decide structure type
        if any(k in lower(angle) for k in ["체크", "체크리스트", "체크할", "전 체크", "체크리스트"] ) or any(x in lower(keyword) for x in ["체크", "팁", "방법"]):
            # checklist style
            add_heading("어떤 기준으로 판단할까요?")
            add_paragraph("다음 3가지를 기준으로 빠르게 판단해 보세요:")
            add_paragraph("1. 상태: 얼룩/늘어남/변형이 있는지 확인하세요.")
            add_paragraph("2. 사용 빈도: 지난 시즌 포함 최근 1년간 착용 횟수를 기준으로 생각하세요.")
            add_paragraph("3. 활용성: 다른 코디와의 조합 여부를 고려하세요.")
            add_heading("실제 적용 예")
            add_paragraph("예: 셔츠 하나는 상태가 좋아도 요즘 스타일과 맞지 않다면 기부나 리셀을 고려해보세요.")
        elif any(x in lower(angle) for x in ["수선", "리폼", "업사이클"] ) or any(x in lower(keyword) for x in ["수선", "리폼", "업사이클"]):
            # how-to style
            add_heading("간단한 수선·리폼으로 오래 입는 법")
            add_paragraph("집에서 시도해볼 수 있는 기본 수선과 리폼 방법을 단계별로 안내합니다.")
            add_paragraph("1. 늘어난 니트는 뜨개질 바늘과 바늘땀으로 고정해 보세요.")
            add_paragraph("2. 소매나 밑단 수선은 가까운 수선집을 이용하면 비용 대비 효과가 큽니다.")
            add_paragraph("3. 간단한 패치나 자수로 빈티지 무드를 내는 방법도 추천합니다.")
        elif any(x in lower(angle) for x in ["리셀", "중고", "판매"] ) or any(x in lower(keyword) for x in ["리셀", "중고", "판매"]):
            # resale guide
            add_heading("중고 판매 전 체크포인트")
            add_paragraph("중고로 내놓기 전, 사진과 상태 표기는 구매 결정에 큰 영향을 줍니다.")
            add_paragraph("1. 깨끗한 사진: 자연광에서 여러 각도로 촬영하세요.")
            add_paragraph("2. 상세 설명: 소재, 사이즈, 상태(특이점)를 솔직하게 적으세요.")
            add_paragraph("3. 적정 가격: 유사 매물 가격을 참고해 합리적으로 책정하세요.")
        else:
            # general informative / narrative
            add_heading("왜 이 주제가 중요한가요?")
            add_paragraph("계절이 바뀔 때마다 옷장을 정리하면 생활이 더 가벼워지고, 필요한 옷만 남길 수 있습니다.")
            add_heading("실용 팁")
            add_paragraph("간단한 분류 기준과 보관 요령을 적용하면 다음 시즌까지 옷 상태를 잘 유지할 수 있어요.")

        # Add a short practical checklist or next steps
        add_heading("오늘 바로 해볼 수 있는 행동")
        add_paragraph("1) 옷장 한 칸을 정리해 보며 상태와 활용도를 체크해보세요.")
        add_paragraph("2) 기부나 리셀용으로 분류한 옷은 오늘 바로 사진을 찍어 목록을 만드세요.")

        # Closing
        closing = "작은 실천이 모이면 옷 관리가 훨씬 수월해집니다. 다음 계절에 더 가볍게 시작해보세요."
        add_paragraph(closing)

        content = "\n\n".join(sections)

        # summary: concise human-readable summary without internal metadata
        summary = f"{title}은(는) 실생활에서 바로 적용할 수 있는 실용적인 기준과 행동을 제안합니다. 상태·빈도·활용성 중심으로 판단해 보관, 재사용, 기부, 재활용을 고려하세요."

        # CTA: contextual and not a hard sales pitch
        cta = "지금 옷장에 있는 한 벌을 골라 상태와 활용도를 점검해보세요. 필요하면 기부나 리셀을 고려해 보관 공간을 줄여보세요."

        return BlogPost(title=title, keyword=keyword, content=content, summary=summary, cta=cta)
