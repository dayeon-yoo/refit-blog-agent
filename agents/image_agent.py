from __future__ import annotations

from pathlib import Path
from typing import Optional

from llm.client import LLMClient, get_llm_client
from models.schemas import BlogPost, ImagePlan, ImagePrompt


class ImageAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.client = llm_client or get_llm_client()
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "image.txt"
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()

    def plan(self, blog_post: BlogPost) -> BlogPost:
        title = blog_post.title
        keyword = blog_post.keyword
        content = blog_post.content

        if "반팔" in title or "반팔" in keyword:
            images = [
                ImagePrompt(
                    placement="hero-intro",
                    purpose="여름이 끝나며 반팔이 한 번에 쌓이는 상황을 보여주고, 독자가 지금 정리할 타이밍임을 바로 느끼게 한다.",
                    prompt="여름이 끝나고 옷장 앞에 반팔이 쌓여 있는 장면, 손으로 반팔을 정리해 보관할 옷과 입지 않는 옷을 분리하는 행동, 실제 가정 공간, 자연광, 정리된 생활 공간, 현실적인 스틸 사진",
                    image_type="lifestyle",
                    alt_text="옷장 앞에서 반팔을 정리하고 있는 사람의 실제 생활 장면",
                ),
                ImagePrompt(
                    placement="decision-point",
                    purpose="반팔을 보관할지, 기부할지, 재판매할지 결정하는 기준을 설명하는 문단 바로 뒤에 들어가 독자가 선택을 현실적으로 떠올리게 한다.",
                    prompt="옷장 안의 반팔을 세 개의 묶음으로 나누는 장면, 보관, 기부, 재판매를 각각 다른 바구니나 선반에 두는 구도, 자연스러운 집 안 배경, 현실적인 생활 사진, 부드러운 daylight",
                    image_type="instructional",
                    alt_text="반팔을 보관, 기부, 재판매로 나누는 과정의 정리 장면",
                ),
                ImagePrompt(
                    placement="condition-check",
                    purpose="반팔의 상태와 세탁 여부를 확인하는 문단을 보완해, 어떤 옷을 다시 입을 수 있는지 판단하는 기준을 시각적으로 설명한다.",
                    prompt="테이블 위에 반팔을 펼쳐 놓고 세탁 상태와 소재 상태를 확인하는 장면, 먼지나 얼룩, 늘어짐을 살펴보는 디테일 사진, 자연스러운 실내 조명, 클로즈업 비율, 패션 에디토리얼 톤",
                    image_type="detail",
                    alt_text="반팔의 상태를 확인하고 세탁 여부를 판단하는 클로즈업 장면",
                ),
            ]
        elif "보관" in title or "보관" in keyword:
            images = [
                ImagePrompt(
                    placement="hero-intro",
                    purpose="보관 전 점검이 필요한 시점을 시각적으로 보여주고, 독자가 다음 시즌을 준비하는 순간임을 느끼게 한다.",
                    prompt="계절 옷을 정리한 뒤 보관함에 넣기 전 상태를 확인하는 장면, 옷장 안쪽에 보관 용기와 의류를 정리하는 모습, 자연스러운 생활 공간, 밝고 깔끔한 인테리어, 현실적인 스틸 사진",
                    image_type="lifestyle",
                    alt_text="계절 옷을 보관하기 전에 점검하는 실제 생활 장면",
                ),
                ImagePrompt(
                    placement="process-step-1",
                    purpose="세탁 후 완전히 건조한 상태를 확인하는 방법을 설명하는 문단 바로 뒤에 들어가 독자가 실제 행동을 떠올리도록 돕는다.",
                    prompt="세탁이 끝난 의류를 테이블 위에 펼쳐 말리는 장면, 완전히 건조된 상태와 보관 전 점검이 필요한 부분을 보여주는 클로즈업, 자연광, 실용적인 생활 사진",
                    image_type="instructional",
                    alt_text="세탁 후 의류를 건조 상태를 점검하는 장면",
                ),
                ImagePrompt(
                    placement="compare-sort",
                    purpose="보관할 옷과 정리할 옷을 비교해 구분하는 문단을 보완해, 즉시 실행 가능한 기준을 시각적으로 전달한다.",
                    prompt="옷장 내부에서 보관할 옷과 기부할 옷을 나누는 전경, 파트별 선반이나 바구니를 나누어 정리하는 상황, 깔끔한 인테리어, 집안 정리 장면, 자연스러운 컬러 톤",
                    image_type="infographic",
                    alt_text="보관할 옷과 정리할 옷을 나누는 정리 비교 장면",
                ),
            ]
        elif "정리" in title or "정리" in keyword:
            images = [
                ImagePrompt(
                    placement="hero-intro",
                    purpose="옷장 정리를 시작하는 순간을 현실적으로 보여주고, 독자가 지금 자신의 옷장을 떠올리도록 유도한다.",
                    prompt="옷장을 열어 정리를 시작하는 실제 사람, 전체적인 옷장 구도, 정리 중인 의류와 선반, 자연스러운 가정 인테리어, 매끈한 현실적 조명",
                    image_type="lifestyle",
                    alt_text="옷장을 열어 정리를 시작하는 실제 생활 장면",
                ),
                ImagePrompt(
                    placement="sorting-criteria",
                    purpose="자주 입는 옷과 아닌 옷을 구분하는 기준을 설명하는 문단을 보완해, 실제 분류 기준을 바로 이해하게 한다.",
                    prompt="옷을 세 가지 기준으로 분류하는 장면, 자주 입는 옷, 보관할 옷, 정리할 옷을 각각 다른 공간에 두는 모습, 현실적인 가정 환경, 자연광, 정보 전달형 스틸 사진",
                    image_type="infographic",
                    alt_text="옷을 기준에 따라 분류하는 장면",
                ),
                ImagePrompt(
                    placement="reuse-option",
                    purpose="기부, 재판매, 재활용 같은 대안 선택지를 자연스럽게 보여주고, 정리 이후 행동 선택을 시각화한다.",
                    prompt="입지 않는 옷을 정리해 기부 상자와 중고 판매 바구니에 넣는 장면, 현실적인 집 안, 따뜻한 조명, 잘 정리된 정리 공간, 자연스러운 생활 사진",
                    image_type="instructional",
                    alt_text="입지 않는 옷을 다시 활용할 수 있는 방법으로 분류하는 장면",
                ),
            ]
        else:
            images = [
                ImagePrompt(
                    placement="hero-intro",
                    purpose="이 글이 다루는 의류 관리 문제를 실제 상황에서 짧게 보여주고, 독자가 지금 문제를 떠올리게 한다.",
                    prompt="의류를 정리하는 현실적인 집안 장면, 옷장 혹은 침대 위에 의류가 놓여 있는 모습, 자연스러운 집 안, 다소 정돈된 생활공간, 현실적이고 차분한 조명",
                    image_type="lifestyle",
                    alt_text="의류를 정리하며 관리하는 현실적인 가정 장면",
                ),
                ImagePrompt(
                    placement="decision-criteria",
                    purpose="상태, 빈도, 계절 적합성을 판단하는 기준을 설명하는 문단을 보완해, 어떤 옷을 보관하고 어떤 옷을 정리할지 이해하게 한다.",
                    prompt="옷을 상태, 입는 빈도, 계절 적합성으로 비교하는 시각적 정리 장면, 의류들 사이에 기준을 표시한 편집 스타일, 자연스러운 테이블 위, 현실적인 패션 에디토리얼 느낌",
                    image_type="infographic",
                    alt_text="의류를 기준에 따라 비교하는 정리 기준 시각화",
                ),
                ImagePrompt(
                    placement="practical-action",
                    purpose="실제 행동으로 이어지는 문단을 보완해, 독자가 바로 옷장을 점검하고 정리할 수 있도록 돕는다.",
                    prompt="옷장 한 구역을 정리하며 필요한 옷과 안 입는 옷을 구분하는 손의 동작, 현실적인 테이블과 선반, 자연광, 다소 정돈된 가정 환경",
                    image_type="instructional",
                    alt_text="옷장을 정리하며 필요한 옷을 구분하는 실제 행동 장면",
                ),
            ]

        blog_post.image_plan = ImagePlan(images=images)
        return blog_post
