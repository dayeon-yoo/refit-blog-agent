from __future__ import annotations

import re

from agents.writer_agent import WriterAgent
from llm.client import MockLLMClient
from models.schemas import BlogIdea, BlogPost, ScoredIdea


def make_idea(title: str, keyword: str, angle: str, summary: str = "") -> ScoredIdea:
    idea = BlogIdea(
        title=title,
        keyword=keyword,
        search_intent="정보 탐색",
        angle=angle,
        rifit_connection="",
        seasonality=0.5,
        summary=summary,
    )
    return ScoredIdea(
        idea_id="test-1",
        scores={"s": 1},
        total_score=50,
        reason="mock",
        idea=idea,
    )


def numbered_headings(content: str) -> list[tuple[int, str]]:
    return [
        (int(number), label.strip())
        for number, label in re.findall(r"^##\s+(\d+)[.)]\s+(.+?)\s*$", content, re.MULTILINE)
    ]


def writer_with_response(title: str, content: str, keyword: str) -> WriterAgent:
    def response_factory(_prompt, _schema, payload):
        return BlogPost(
            title=title,
            keyword=keyword,
            content=content,
            summary=payload.get("summary", "") or "주제에 맞는 실천 내용을 정리했습니다.",
            cta="오늘 한 가지부터 적용해 보세요.",
        )

    return WriterAgent(MockLLMClient(response_factory=response_factory))


def test_writer_generates_long_structured_post_and_calls_client():
    client = MockLLMClient()
    writer = WriterAgent(client)
    scored = make_idea(
        "헌옷 보낼 때 이것만은 꼭 확인하세요",
        "헌옷 처리",
        "헌옷 처리 - 기부·수거·정리 단계에서의 실용 가이드",
    )
    post = writer.write(scored)

    assert len(client.calls) == 1
    assert client.calls[0]["schema"].__name__ == "BlogPost"
    assert post.title == scored.idea.title
    assert len(post.content) > 400
    assert len([line for line in post.content.splitlines() if line.startswith("## ")]) >= 4

    forbidden = ["검색 의도", "seo", "prompt", "agent", "brand fit", "브랜드 평가", "critic", "ai 평가", "콘텐츠 생성 과정", "검색 유입을 위해"]
    low = post.content.lower()
    for phrase in forbidden:
        assert phrase not in low


def test_numeric_title_requires_exactly_ten_distinct_items():
    writer = WriterAgent(MockLLMClient())
    post = writer.write(make_idea(
        "10가지 방법으로 옷장을 업사이클링 하는 법",
        "옷장 업사이클링",
        "오래된 옷과 옷장 속 재료를 새 용도로 바꾸는 구체적인 실천 방법",
        "버려질 옷을 생활 소품으로 바꾸는 단계별 아이디어",
    ))

    headings = numbered_headings(post.content)
    assert [number for number, _ in headings] == list(range(1, 11))
    assert len({label.lower() for _, label in headings}) == 10
    assert "업사이클링" in post.content


def test_different_numeric_title_stays_on_clothing_organization():
    writer = WriterAgent(MockLLMClient())
    post = writer.write(make_idea(
        "헌옷을 정리하는 5가지 방법",
        "헌옷 정리",
        "입지 않는 헌옷을 상태와 사용 빈도에 따라 정리하는 실천 방법",
    ))

    headings = numbered_headings(post.content)
    assert [number for number, _ in headings] == list(range(1, 6))
    assert len({label.lower() for _, label in headings}) == 5
    assert "정리" in post.content
    assert "업사이클링" not in post.content


def test_different_blog_ideas_produce_different_topics():
    client = MockLLMClient()
    writer = WriterAgent(client)
    upcycle = writer.write(make_idea(
        "옷장 속 셔츠를 새 소품으로 바꾸는 법",
        "셔츠 업사이클링",
        "입지 않는 셔츠를 실용적인 소품으로 바꾸는 과정",
    ))
    organize = writer.write(make_idea(
        "헌옷을 오래 보관하는 방법",
        "헌옷 보관",
        "계절이 지난 옷을 손상 없이 분류하고 보관하는 방법",
    ))

    assert upcycle.title != organize.title
    assert upcycle.content != organize.content
    assert "셔츠" in upcycle.content
    assert "보관" in organize.content


def test_summary_is_passed_to_writer_prompt_payload():
    client = MockLLMClient()
    writer = WriterAgent(client)
    summary = "겨울 니트를 형태가 무너지지 않게 접어서 보관하는 방법을 안내합니다."
    writer.write(make_idea("겨울 니트 보관법", "니트 보관", "니트의 형태를 지키는 보관 방법", summary))

    assert client.calls[0]["payload"]["summary"] == summary
    assert client.calls[0]["payload"]["content_contract"]["summary"] == summary


def test_writer_receives_refinement_planning_fields_as_content_contract():
    client = MockLLMClient()
    writer = WriterAgent(client)
    scored = make_idea(
        "의류 기부를 위한 분류 체크리스트",
        "의류 기부 분류",
        "의류수거함과 기부 중 선택하는 기준을 설명하는 비교 가이드",
    )
    scored.idea.content_format = "체크리스트"
    scored.idea.content_perspective = "의류 처리 선택 기준"
    scored.idea.key_question = "어떤 옷을 기부하고 어떤 옷을 수거로 보낼까요?"
    scored.idea.outline = [
        "기부 전 옷 상태 확인",
        "의류수거함과 기부의 선택 기준",
        "상태에 따른 다음 활용 방법",
    ]

    writer.write(scored)
    payload = client.calls[0]["payload"]

    assert payload["content_contract"]["outline"] == scored.idea.outline
    assert payload["content_contract"]["key_question"] == scored.idea.key_question
    assert payload["content_contract"]["content_perspective"] == scored.idea.content_perspective


def test_writer_prompt_requires_outline_fidelity_and_qualified_factual_language():
    prompt = WriterAgent(MockLLMClient()).prompt

    assert "Each meaningful outline item must have a corresponding" in prompt
    assert "observable condition details" in prompt
    assert "기부처에 따라 기준이 다를 수 있으니 확인해보세요" in prompt
    assert "Do not invent institutions, policies, statistics" in prompt
    assert "cta should be one short, non-repetitive action suggestion" in prompt


def test_writer_prompt_covers_contrast_connection_and_observable_donation_criteria():
    prompt = WriterAgent(MockLLMClient()).prompt

    assert "wearing frequency, brand, style, size" in prompt
    assert "discuss both sides in the body" in prompt
    assert "use its practical connection once" in prompt


def test_writer_rejects_unsupported_absolute_donation_claim():
    idea = make_idea(
        "의류 수거함 대신 기부하는 방법 가이드",
        "의류 기부 방법",
        "의류수거함과 기부 중 선택하는 기준",
    )
    idea.idea.rifit_connection = "의류 순환 서비스와 다음 사용으로 연결"
    writer = writer_with_response(
        idea.idea.title,
        "## 상태 확인\n의류 수거함과 기부를 비교할 때, 기부하기 전에는 반드시 옷의 상태를 점검해야 합니다. 세탁은 필수입니다. 의류 순환을 위해 다음 사용도 생각해봅니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "absolute" in str(error)
    else:
        raise AssertionError("Unsupported absolute donation claim should be rejected")


def test_writer_rejects_missing_title_contrast_and_rifit_connection():
    idea = make_idea(
        "의류 수거함 대신 기부하는 방법 가이드",
        "의류 기부 방법",
        "의류수거함과 기부 중 선택하는 기준",
    )
    idea.idea.rifit_connection = "의류 순환 서비스와 다음 사용으로 연결"
    writer = writer_with_response(
        idea.idea.title,
        "## 기부 준비\n의류 상태를 확인한 뒤 기부 준비를 합니다. 의류 순환을 위해 다음 사용도 생각해봅니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "contrast" in str(error) or "rifit" in str(error)
    else:
        raise AssertionError("Missing title contrast or rifit connection should be rejected")


def test_non_numeric_title_generates_normally():
    post = WriterAgent(MockLLMClient()).write(make_idea(
        "겨울 니트 보관법",
        "니트 보관",
        "니트의 형태와 소재를 지키는 계절 보관 방법",
    ))

    assert post.title == "겨울 니트 보관법"
    assert post.content
    assert "보관" in post.content


def test_writer_starts_with_a_reader_situation_not_a_generic_definition():
    post = WriterAgent(MockLLMClient()).write(make_idea(
        "가을 옷장 정리로 새로운 스타일 찾기",
        "가을 옷장 정리",
        "작년에 입던 가을 옷을 다시 조합해 새로운 스타일을 찾는 방법",
        "옷장 속 기존 아이템을 점검하고 조합을 바꾸는 실천 가이드",
    ))

    opening = post.content.split("\n\n", 2)[1]
    generic_openings = ["중요한 과정입니다", "필요한 과정입니다", "효율적인 방법입니다"]
    assert not any(phrase in opening for phrase in generic_openings)
    assert any(phrase in opening for phrase in ["고민하다 보면", "망설여질 때", "살펴보겠습니다"])


def test_writer_keeps_metadata_out_of_summary_and_cta():
    post = WriterAgent(MockLLMClient()).write(make_idea(
        "헌옷 정리 방법",
        "헌옷 정리",
        "입지 않는 옷을 정리하고 다시 활용할 기준",
    ))

    output = f"{post.summary} {post.cta}".lower()
    for phrase in ["검색 의도", "seo", "prompt", "agent", "critic", "workflow", "metadata"]:
        assert phrase not in output


def test_writer_does_not_use_fixed_template_headings():
    post = WriterAgent(MockLLMClient()).write(make_idea(
        "옷장 정리 팁",
        "옷장 정리",
        "옷장 정리 - 공간 구성과 분류 기준, 정리 루틴 중심의 실용적 접근",
    ))

    for disallowed in ["왜 이 주제가 중요한가요?", "실용 팁", "오늘 바로 해볼 수 있는 행동"]:
        assert disallowed not in post.content


def test_exact_title_passes_contract_validation():
    idea = make_idea("겨울 니트 보관법", "니트 보관", "니트 형태를 지키는 보관 방법")
    post = WriterAgent(MockLLMClient()).write(idea)

    assert post.title == idea.idea.title


def test_natural_title_variation_passes_when_topic_is_preserved():
    idea = make_idea(
        "헌옷을 오래 보관하는 방법",
        "헌옷 보관",
        "입지 않는 옷을 손상 없이 보관하는 방법",
    )
    writer = writer_with_response(
        "입지 않는 옷을 오래 보관하는 법",
        "입지 않는 옷을 오래 보관하면서 손상을 줄이는 방법을 설명합니다.",
        "헌옷 보관",
    )

    post = writer.write(idea)

    assert post.title != idea.idea.title
    assert "보관" in post.content


def test_unrelated_title_is_rejected():
    idea = make_idea(
        "옷장 속 셔츠를 새 소품으로 바꾸는 법",
        "셔츠 업사이클링",
        "입지 않는 셔츠를 실용적인 소품으로 바꾸는 과정",
    )
    writer = writer_with_response(
        "헌옷 기부처를 고르는 방법",
        "헌옷을 기부처에 보내는 절차를 설명합니다.",
        "셔츠 업사이클링",
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "title" in str(error).lower()
    else:
        raise AssertionError("unrelated title should fail contract validation")


def test_topic_drift_in_content_is_rejected():
    idea = make_idea(
        "옷장 속 셔츠를 새 소품으로 바꾸는 법",
        "셔츠 업사이클링",
        "입지 않는 셔츠를 실용적인 소품으로 바꾸는 과정",
    )
    writer = writer_with_response(
        idea.idea.title,
        "헌옷을 기부처에 보내는 절차와 포장 방법만 설명합니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "does not reflect" in str(error)
    else:
        raise AssertionError("topic drift should fail contract validation")


def test_metadata_in_title_is_rejected():
    idea = make_idea("옷장 정리 방법", "옷장 정리", "옷장을 정리하는 실천 방법")
    writer = writer_with_response(
        "SEO 검색 의도에 맞춘 옷장 정리",
        "옷장을 정리하는 실제 방법을 설명합니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "metadata" in str(error)
    else:
        raise AssertionError("metadata in title should fail validation")


def test_natural_title_keeps_keyword_and_angle_consistency():
    idea = make_idea(
        "가을 옷장 정리로 새로운 스타일 찾기",
        "가을 옷장 정리",
        "기존 옷을 다시 조합해 새로운 스타일을 찾는 방법",
    )
    writer = writer_with_response(
        "가을 옷장 속 기존 옷으로 스타일 찾는 방법",
        "가을 옷장을 정리한 뒤 기존 옷을 다시 조합해 새로운 스타일을 찾는 방법입니다.",
        idea.idea.keyword,
    )

    post = writer.write(idea)

    assert "기존 옷" in post.content


def test_circulation_connection_becomes_practical_mock_content_without_forced_brand_name():
    idea = make_idea(
        "지속 가능한 의류 소비를 위한 실용적인 체크리스트",
        "지속 가능한 소비 습관",
        "지속 가능한 의류 구매 및 처분 방법 중심의 실용 가이드",
        "구매 전 고려할 요소와 충동 구매를 피하는 방법",
    )
    idea.idea.rifit_connection = "의류 순환과 지속 가능한 소비에 대한 직접적인 연계성"

    post = WriterAgent(MockLLMClient()).write(idea)

    assert any(term in post.content for term in ("재사용", "재활용", "순환", "기부", "수거"))
    assert "RIFIT" not in post.content
    assert "브랜드 평가" not in post.content
