from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

import agents.writer_agent as writer_agent_module
from agents.writer_agent import WriterAgent
from llm.client import MockLLMClient
from models.schemas import BlogIdea, BlogPost, ContentPlan, ScoredIdea


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

    payload = client.calls[0]["payload"]
    assert payload["summary"] != summary
    assert payload["summary"]
    assert "summary" not in payload["editorial_context"]
    assert "summary" not in payload["content_contract"]


def test_writer_content_contract_uses_only_safe_editorial_fields():
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

    assert payload["content_contract"]["title"] == scored.idea.title
    assert payload["content_contract"]["keyword"] == scored.idea.keyword
    assert payload["content_contract"]["content_format"] == scored.idea.content_format
    assert "search_intent" not in payload["content_contract"]
    assert "key_question" not in payload["content_contract"]
    assert "content_perspective" not in payload["content_contract"]
    assert "outline" not in payload["content_contract"]
    assert "angle" not in payload["editorial_context"]
    assert "summary" not in payload["editorial_context"]


def test_writer_payload_keeps_raw_source_separate_from_safe_editorial_context():
    idea = make_idea("체크셔츠 출근 코디", "체크셔츠 코디", "구매와 착용 경험")
    idea.idea.summary = "동료들에게 좋은 반응을 얻은 오슬로 인턴 생활"
    idea.idea.angle = "오슬로 인턴 생활과 동료 반응을 중심으로 구성"
    idea.idea.outline = ["동료들의 반응 소개"]
    client = MockLLMClient()

    WriterAgent(client).write(
        idea,
        raw_source="오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다.",
        content_plan=ContentPlan(writing_script="제공된 구매와 착용 경험만 중심으로 구성합니다."),
    )

    payload = client.calls[0]["payload"]
    assert payload["raw_source"] == "오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다."
    for context_key in ("editorial_context", "refined_blog_idea", "content_contract"):
        assert "summary" not in payload[context_key]
        assert "angle" not in payload[context_key]
        assert "outline" not in payload[context_key]
        assert "search_intent" not in payload[context_key]
        assert "content_perspective" not in payload[context_key]
        assert "key_question" not in payload[context_key]


def test_writer_prompt_requires_outline_fidelity_and_qualified_factual_language():
    prompt = WriterAgent(MockLLMClient()).prompt

    assert "Each meaningful outline item must have a corresponding" in prompt
    assert "observable condition details" in prompt
    assert "기부처에 따라 기준이 다를 수 있으니 확인해보세요" in prompt
    assert "Do not invent institutions, policies, statistics" in prompt
    assert "cta should be one short, non-repetitive action suggestion" in prompt


def test_writer_prompt_defines_source_grounded_factual_whitelist_and_suggestion_boundary():
    prompt = WriterAgent(MockLLMClient()).prompt

    assert "Treat raw_source as a factual whitelist" in prompt
    assert "writing_script as authority for ordering and emphasis only" in prompt
    assert "Omission is safer than" in prompt
    assert "Never convert that suggestion into" in prompt
    assert "product adjectives or material properties" in prompt
    assert "environmental, economic, trend," in prompt
    assert "social-impact claim" in prompt


def test_source_grounded_payload_separates_framing_metadata_from_factual_source():
    idea = _experience_writer_idea()
    idea.idea.angle = "비즈니스 캐주얼 아우터 활용과 지속 가능성의 장점"
    idea.idea.summary = "오슬로 매장 분위기와 구매 이유, 동료 반응을 자세히 소개합니다."
    idea.idea.key_question = "어떤 아우터가 비즈니스 캐주얼에 가장 좋을까요?"
    idea.idea.outline = ["매장 분위기", "실제 슬랙스 착장", "환경 효과"]
    idea.idea.rifit_connection = "리핏에서 빈티지 상품을 구매하고 아우터를 추천한다고 안내"
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 체크셔츠로 인턴 출근 코디를 해봤습니다. "
        "리핏은 의류 순환이라는 맥락에서 생각해볼 수 있습니다.",
        idea.idea.keyword,
    )
    writer.write(
        idea,
        raw_source="오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다.",
        content_plan=ContentPlan(writing_script="명시된 구매와 착용 경험만 사용합니다."),
    )

    payload = writer.client.calls[0]["payload"]
    for key in ("editorial_context", "refined_blog_idea", "content_contract"):
        context = payload[key]
        assert "summary" not in context
        assert "outline" not in context
        assert "key_question" not in context
        assert "angle" not in context
        assert "rifit_connection" not in context
    assert payload["content_contract"]["metadata_role"] == "editorial framing only"
    assert payload["raw_source"] == "오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다."


def test_writer_grounded_mode_payload_marks_script_as_structure_only():
    idea = _experience_writer_idea()
    client = MockLLMClient()

    WriterAgent(client).write(
        idea,
        raw_source="오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다.",
        content_plan=ContentPlan(
            writing_script="명시된 구매와 착용 경험만 순서대로 서술합니다."
        ),
    )

    payload = client.calls[0]["payload"]
    contract = payload["content_contract"]
    assert contract["factual_authority"] == "raw_source only"
    assert contract["writing_script_role"] == "structure and ordering only"
    assert payload["raw_source"] in contract["raw_source"]


def test_writer_source_grounded_contract_declares_claim_authority_policy():
    idea = _experience_writer_idea()
    client = MockLLMClient()

    WriterAgent(client).write(
        idea,
        raw_source="오슬로에서 산 체크셔츠로 인턴 출근룩을 입어봤다.",
        content_plan=ContentPlan(writing_script="명시된 경험과 일반 제안만 구분합니다."),
    )

    policy = client.calls[0]["payload"]["content_contract"]["claim_authority_policy"]
    assert policy["PERSONAL_FACT"] == "raw_source_only"
    assert "reader-facing" in policy["GENERAL_SUGGESTION"]
    assert "research_context" in policy["EXTERNAL_CLAIM"]
    assert "not presented as an objective fact" in policy["EDITORIAL_TRANSITION"]


def test_writer_prompt_and_correction_contract_preserve_claim_authority_boundary():
    writer = WriterAgent(MockLLMClient())
    prompt = writer.prompt
    correction = writer._grounding_correction_prompt()

    for text in (prompt, correction):
        assert "PERSONAL_FACT" in text
        assert "GENERAL_SUGGESTION" in text
        assert "EXTERNAL_CLAIM" in text
        assert "raw_source" in text
    assert "shorten the article" in prompt
    assert "must not become a personal claim" in correction


def test_writer_prompt_covers_contrast_connection_and_observable_donation_criteria():
    prompt = WriterAgent(MockLLMClient()).prompt

    assert "wearing frequency, brand, style, size" in prompt
    assert "discuss both sides in the body" in prompt
    assert "use its practical connection once" in prompt


def test_writer_prompt_preserves_raw_claim_relationships_over_plan_expansion():
    prompt = WriterAgent(MockLLMClient()).prompt

    assert "raw_source is the factual, experiential, and" in prompt
    assert "subject/object relationship" in prompt
    assert "money returned from reusing clothing as cheap shopping" in prompt


def test_writer_resolves_source_grounded_editorial_mode_from_raw_source():
    idea = make_idea(
        "Y2K가 돌아와도 새 옷이 필요할까",
        "Y2K 빈티지 활용",
        "새 옷 대신 빈티지 활용",
    )
    def response_factory(_prompt, _schema, payload):
        return BlogPost(
            title=idea.idea.title,
            keyword=idea.idea.keyword,
            content="Y2K가 돌아와도 새 옷 대신 빈티지를 활용할 수 있습니다.",
            summary=payload.get("summary", "주제에 맞는 내용을 정리합니다."),
            cta="빈티지 활용 방법을 한 가지 생각해보세요.",
        )

    client = MockLLMClient(response_factory=response_factory)
    writer = WriterAgent(client)
    source = "Y2K가 다시 유행인데 새 옷 대신 빈티지에서 찾아보자."

    writer.write(idea, raw_source=source, content_plan=ContentPlan(writing_script="원문 주장 순서만 정리합니다."))

    payload = client.calls[0]["payload"]
    assert payload["planning_mode"] == "source_grounded_editorial"
    assert "WRITING MODE: SOURCE-GROUNDED EDITORIAL" in client.calls[0]["prompt"]


def test_writer_keeps_guide_in_generative_mode():
    idea = make_idea(
        "빈티지 쇼핑 체크리스트",
        "빈티지 쇼핑 체크리스트",
        "옷 상태 확인 방법을 정리하는 가이드",
    )
    client = MockLLMClient()
    WriterAgent(client).write(
        idea,
        raw_source="빈티지 쇼핑할 때 옷 상태 확인 방법을 체크리스트로 정리하고 싶다.",
        content_plan=ContentPlan(writing_script="상태 확인 기준과 실행 순서를 정리합니다."),
    )

    assert client.calls[0]["payload"]["planning_mode"] == "generative_structured"
    assert "WRITING MODE: SOURCE-GROUNDED EDITORIAL" not in client.calls[0]["prompt"]


def test_writer_prompt_limits_unverified_trends_scope_and_rifit_capabilities():
    prompt = WriterAgent(MockLLMClient()).prompt

    assert "unverified timely or popularity claims" in prompt
    assert "circular-economy argument" in prompt
    assert "lets readers find, buy, receive recommendations for, or browse vintage items" in prompt


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


def _experience_writer_idea(details: str = "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다.") -> ScoredIdea:
    idea = make_idea(
        "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다",
        "빈티지 체크셔츠 출근 코디",
        "개인적인 구매와 출근 코디 경험을 소개하는 후기",
        details,
    )
    idea.idea.content_format = "경험 후기"
    idea.idea.content_perspective = "출근 코디 경험"
    idea.idea.outline = ["구매 경험", "출근 코디", "입어본 뒤 알게 된 점"]
    return idea


def test_writer_rejects_unprovided_personal_experience_details():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "저는 오슬로의 여러 빈티지샵을 돌아다니며 화이트와 그린 조합의 체크셔츠를 골랐습니다. "
        "오래된 원단이 마음에 들었고 가죽 로퍼와 목걸이를 선택했습니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "unsupported personal experience" in str(error)
    else:
        raise AssertionError("Unprovided personal experience should be rejected")


def test_writer_allows_general_styling_suggestion_without_claiming_experience():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "빈티지 체크셔츠는 슬랙스와 매치해볼 수 있어 출근 코디에 활용할 수 있습니다.",
        idea.idea.keyword,
    )

    assert writer.write(idea).content


def test_writer_grounds_personal_claims_in_raw_source_but_allows_stated_experience():
    idea = _experience_writer_idea()
    raw_source = "오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다."

    allowed = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 체크셔츠를 인턴 출근룩으로 입어봤습니다. 슬랙스와 매치해볼 수 있어요.",
        idea.idea.keyword,
    )
    assert allowed.write(idea, raw_source=raw_source).content

    paraphrase = writer_with_response(
        idea.idea.title,
        "저도 오슬로에서 산 빈티지 체크셔츠로 출근 룩을 시도해봤습니다.",
        idea.idea.keyword,
    )
    assert paraphrase.write(idea, raw_source=raw_source).content

    invented = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 체크셔츠로 출근 코디를 했습니다. 첫 출근이라 어떤 옷을 입을지 고민했고, 진청색 진과 매치해서 입었습니다.",
        idea.idea.keyword,
    )
    try:
        invented.write(idea, raw_source=raw_source)
    except ValueError as error:
        assert "unsupported personal experience" in str(error)
    else:
        raise AssertionError("Unsupported personal claims should be rejected")


def test_writer_rejects_unprovided_first_day_and_reaction_events():
    idea = _experience_writer_idea()
    raw_source = "오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다."
    responses = [
        "오슬로에서 산 체크셔츠로 출근 코디를 했습니다. 첫 출근이라 긴장했습니다.",
        "오슬로에서 산 체크셔츠로 출근 코디를 했습니다. 동료들이 체크셔츠를 칭찬했습니다.",
        "오슬로에서 산 체크셔츠로 출근 코디를 했습니다. 진청색 청바지를 함께 입었습니다.",
    ]

    for content in responses:
        writer = writer_with_response(idea.idea.title, content, idea.idea.keyword)
        try:
            writer.write(idea, raw_source=raw_source)
        except ValueError as error:
            assert "unsupported personal experience" in str(error)
        else:
            raise AssertionError("Unsupported personal event should be rejected")


def test_writer_allows_accessory_styling_suggestion_without_owned_item_false_positive():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "빈티지 체크셔츠는 액세서리로 귀여운 목걸이나 통통 튀는 색상의 가방을 매치하면 전체적인 룩을 완성할 수 있습니다.",
        idea.idea.keyword,
    )

    assert writer.write(idea).content


def test_writer_uses_raw_source_as_personal_fact_boundary():
    idea = _experience_writer_idea(
        "오슬로에서 인턴 생활을 하며 따뜻한 색감의 체크셔츠를 저렴하게 샀다."
    )
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤습니다. 저는 재킷을 입고 출근했습니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea, raw_source="오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다.")
    except ValueError as error:
        assert "unsupported personal experience" in str(error)
    else:
        raise AssertionError("Raw source must remain the personal fact boundary")


def test_writer_debug_diagnostic_includes_marker_sentence_and_intermediate_plan(monkeypatch, capsys):
    idea = _experience_writer_idea()
    sentence = "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 하며, 제가 산 체크셔츠에 슬랙스를 실제로 입었습니다."
    writer = writer_with_response(idea.idea.title, sentence, idea.idea.keyword)
    plan = ContentPlan(writing_script="구매 경험과 출근 코디 경험을 중심으로 작성합니다.")
    monkeypatch.setattr(writer_agent_module, "get_settings", lambda: SimpleNamespace(debug=True))

    try:
        writer.write(idea, raw_source=idea.idea.summary, content_plan=plan)
    except ValueError as error:
        assert "marker='슬랙스'" in str(error)
        assert sentence in str(error)
    else:
        raise AssertionError("Unsupported personal experience should be rejected")

    diagnostic = capsys.readouterr().err
    assert "content_plan.writing_script" in diagnostic
    assert "writer_draft_before_validation" in diagnostic
    assert "marker='슬랙스'" in diagnostic
    assert sentence in diagnostic


def test_writer_allows_personal_detail_explicitly_present_in_contract():
    idea = _experience_writer_idea(
        "오슬로에서 산 빈티지 체크셔츠에 검정 슬랙스와 로퍼를 함께 입어봤다."
    )
    writer = writer_with_response(
        idea.idea.title,
        "저는 검정 슬랙스와 로퍼를 체크셔츠와 함께 입었습니다. 출근 코디로 활용해본 경험을 정리합니다.",
        idea.idea.keyword,
    )

    assert writer.write(idea).content


def test_writer_rejects_unprovided_material_in_personal_claim():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 하며, 제가 산 체크셔츠는 면 소재였습니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "면소재" in str(error)
    else:
        raise AssertionError("Unprovided material in a personal claim should be rejected")


def test_writer_allows_material_as_general_suggestion_or_condition():
    idea = _experience_writer_idea()
    responses = [
        "빈티지 체크셔츠는 면바지와 매치해볼 수 있어 출근 코디에 활용할 수 있습니다. 체크셔츠가 면 소재라면 관리 라벨을 확인해보세요.",
        "빈티지 체크셔츠는 출근 코디에 활용할 수 있습니다. 반면 체크셔츠는 다른 하의와도 조합할 수 있습니다.",
    ]

    for content in responses:
        writer = writer_with_response(idea.idea.title, content, idea.idea.keyword)
        assert writer.write(idea).content


def test_writer_does_not_false_positive_on_short_marker_inside_general_word():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 체크셔츠로 출근 코디를 해봤습니다. "
        "이 체크셔츠는 다른 옷과 잘 어울리기 때문에 일반적인 코디 제안으로 소개할 수 있습니다.",
        idea.idea.keyword,
    )

    assert writer.write(
        idea,
        raw_source="오슬로에서 산 체크셔츠로 출근 코디를 해봤다.",
    ).content


def test_writer_valid_grounded_draft_does_not_trigger_correction():
    client = MockLLMClient()
    idea = _experience_writer_idea()

    WriterAgent(client).write(
        idea,
        raw_source="오슬로에서 산 체크셔츠로 출근 코디를 해봤다.",
        content_plan=ContentPlan(writing_script="명시된 경험만 사용합니다."),
    )

    assert len(client.calls) == 1


def test_writer_corrects_multiple_grounding_issues_once():
    idea = _experience_writer_idea()
    raw_source = "오슬로에서 산 체크셔츠로 출근 코디를 해봤다."
    invalid = BlogPost(
        title=idea.idea.title,
        keyword=idea.idea.keyword,
        content=(
            "## 오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다\n\n"
            "오슬로에서 산 체크셔츠로 출근 코디를 해봤습니다.\n\n"
            "첫 출근이라 어떤 옷을 입을지 고민했고 동료들이 칭찬했습니다.\n\n"
            "진청색 청바지도 실제로 입었습니다."
        ),
        summary=raw_source,
        cta="오늘 한 가지부터 적용해 보세요.",
    )
    corrected = BlogPost(
        title=idea.idea.title,
        keyword=idea.idea.keyword,
        content=(
            "## 오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다\n\n"
            "오슬로에서 산 체크셔츠로 출근 코디를 해봤습니다.\n\n"
            "체크셔츠는 슬랙스와 매치해볼 수 있어요."
        ),
        summary=raw_source,
        cta="오늘 한 가지부터 적용해 보세요.",
    )

    def response_factory(_prompt, _schema, _payload):
        return invalid if len(client.calls) == 1 else corrected

    client = MockLLMClient(response_factory=response_factory)
    result = WriterAgent(client).write(
        idea,
        raw_source=raw_source,
        content_plan=ContentPlan(writing_script="명시된 경험만 사용합니다."),
    )

    assert result == corrected
    assert len(client.calls) == 2
    assert client.calls[1]["payload"]["correction_mode"] == "controlled_grounding_correction"
    assert len(client.calls[1]["payload"]["grounding_issues"]) >= 2
    assert client.calls[1]["payload"]["previous_draft"]["content"] == invalid.content


def test_writer_stops_after_one_failed_grounding_correction():
    idea = _experience_writer_idea()
    raw_source = "오슬로에서 산 체크셔츠로 출근 코디를 해봤다."
    invalid = BlogPost(
        title=idea.idea.title,
        keyword=idea.idea.keyword,
        content="오슬로에서 산 체크셔츠로 출근 코디를 해봤습니다. 첫 출근이라 고민했습니다.",
        summary=raw_source,
        cta="오늘 한 가지부터 적용해 보세요.",
    )

    client = MockLLMClient(response_factory=lambda _prompt, _schema, _payload: invalid)
    with pytest.raises(ValueError, match="unsupported personal experience"):
        WriterAgent(client).write(idea, raw_source=raw_source)

    assert len(client.calls) == 2


def test_writer_generation_failure_does_not_trigger_correction():
    def response_factory(_prompt, _schema, _payload):
        raise RuntimeError("network failure")

    client = MockLLMClient(response_factory=response_factory)
    with pytest.raises(RuntimeError, match="network failure"):
        WriterAgent(client).write(
            _experience_writer_idea(),
            raw_source="오슬로에서 산 체크셔츠로 출근 코디를 해봤다.",
        )

    assert len(client.calls) == 1


def test_writer_allows_material_explicitly_present_in_contract():
    idea = _experience_writer_idea("오슬로에서 산 면 소재 빈티지 체크셔츠로 출근 코디를 해봤다.")
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 하며, 제가 산 체크셔츠는 면 소재였습니다.",
        idea.idea.keyword,
    )

    assert writer.write(idea).content


def test_writer_rejects_unprovided_location_event_relationship():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 인턴으로 일하고 있어요. 오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤습니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "unsupported personal experience or event" in str(error)
    else:
        raise AssertionError("Unprovided location event should be rejected")


def test_writer_rejects_unprovided_reaction_effect_and_era_claims():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 하며, 이 셔츠는 80년대 스타일이라 학과 미팅에서 주변 반응이 좋았습니다. "
        "자존감이 높아졌고 실제로 경제적 이득도 봤습니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "unsupported personal experience or event" in str(error)
    else:
        raise AssertionError("Unprovided reaction or effect should be rejected")


def test_writer_allows_general_event_and_effect_suggestions():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "빈티지 체크셔츠는 새 옷 구매를 줄이는 선택지가 될 수 있습니다. "
        "출근 코디에서 체크셔츠를 활용하면 자신감 있는 분위기를 연출할 수 있습니다.",
        idea.idea.keyword,
    )

    assert writer.write(idea).content


def test_writer_experience_validator_does_not_classify_general_effect_as_owned_item_claim():
    idea = _experience_writer_idea()
    writer = writer_with_response(
        idea.idea.title,
        "빈티지 체크셔츠 출근 코디를 소개합니다. 이러한 선택은 자원의 낭비를 줄이고 환경 보호에도 도움이 됩니다.",
        idea.idea.keyword,
    )

    assert writer.write(idea).content


def test_writer_rejects_unsupported_rifit_service_reframing():
    idea = _experience_writer_idea()
    idea.idea.rifit_connection = "의류 순환 서비스와 다음 사용으로 연결"
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 체크셔츠로 출근 코디를 소개합니다. 리패션 서비스는 요즘 많은 브랜드가 제공합니다.",
        idea.idea.keyword,
    )

    try:
        writer.write(idea)
    except ValueError as error:
        assert "unsupported" in str(error).lower()
    else:
        raise AssertionError("Unsupported RIFIT reframing should be rejected")


def test_source_grounded_mode_does_not_require_editorial_rifit_mention():
    idea = _experience_writer_idea()
    idea.idea.rifit_connection = "리핏의 의류 순환과 다음 사용으로 연결"
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 체크셔츠로 인턴 출근 코디를 해봤습니다. "
        "체크셔츠는 다른 하의와 매치해볼 수 있습니다.",
        idea.idea.keyword,
    )

    post = writer.write(
        idea,
        raw_source="오슬로에서 산 체크셔츠로 인턴 출근 코디를 해봤다.",
        content_plan=ContentPlan(writing_script="명시된 경험만 사용합니다."),
    )

    assert "리핏" not in post.content
    assert len(writer.client.calls) == 1


def test_source_grounded_mode_keeps_rifit_safety_checks():
    idea = _experience_writer_idea()
    idea.idea.rifit_connection = "리핏의 의류 순환과 다음 사용으로 연결"
    writer = writer_with_response(
        idea.idea.title,
        "오슬로에서 산 체크셔츠로 인턴 출근 코디를 해봤습니다. "
        "리패션 서비스는 요즘 많은 브랜드가 제공합니다.",
        idea.idea.keyword,
    )

    with pytest.raises(ValueError, match="unsupported"):
        writer.write(
            idea,
            raw_source="오슬로에서 산 체크셔츠로 인턴 출근룩으로 입어봤다.",
            content_plan=ContentPlan(writing_script="명시된 경험만 사용합니다."),
        )


def test_generative_mode_keeps_rifit_connection_preservation():
    idea = make_idea(
        "지속 가능한 의류 소비 방법",
        "의류 순환 방법",
        "의류 순환을 실천하는 방법을 안내하는 가이드",
    )
    idea.idea.rifit_connection = "의류 순환 서비스와 다음 사용으로 연결"
    writer = writer_with_response(
        idea.idea.title,
        "의류 순환 방법을 정리합니다. 옷을 다시 활용하는 기준을 살펴봅니다.",
        idea.idea.keyword,
    )

    with pytest.raises(ValueError, match="rifit connection"):
        writer.write(idea, raw_source="의류 순환 방법을 가이드로 정리하고 싶다.")


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
