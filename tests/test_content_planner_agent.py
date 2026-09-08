from agents.content_planner_agent import ContentPlannerAgent
from llm.client import MockLLMClient
from models.schemas import BlogIdea, ContentPlan


def make_idea(title: str, content_format: str, perspective: str) -> BlogIdea:
    return BlogIdea(
        title=title,
        keyword=title,
        search_intent="정보 탐색",
        angle=perspective,
        rifit_connection="",
        seasonality=0.5,
        summary=title,
        content_format=content_format,
        content_perspective=perspective,
        key_question="무엇을 중심으로 살펴볼까요?",
        outline=["핵심 상황 확인", "구체적인 기준", "다음 행동"],
    )


def test_experience_plan_uses_raw_source_without_inventing_details():
    source = "오슬로에서 체크셔츠를 샀고 실제 인턴 출근룩으로 입어봤다."
    client = MockLLMClient()
    plan = ContentPlannerAgent(llm_client=client).plan(
        source,
        make_idea("빈티지 체크셔츠 출근 코디", "경험 후기", "구매와 착용 경험"),
    )

    assert isinstance(plan, ContentPlan)
    assert source in plan.writing_script
    assert "경험" in plan.writing_script
    assert len(client.calls) == 0
    assert "선택 이유" in plan.writing_script
    assert "일반적인 제안" in plan.writing_script


def test_experience_routing_uses_deterministic_plan_but_information_uses_llm():
    experience_client = MockLLMClient()
    experience = ContentPlannerAgent(llm_client=experience_client).plan(
        "오슬로에서 체크셔츠를 샀고 인턴 출근룩으로 입어봤다.",
        make_idea("체크셔츠 출근 후기", "경험 후기", "구매와 착용 경험"),
    )
    assert len(experience_client.calls) == 0
    assert "매장 분위기" not in experience.writing_script

    info_client = MockLLMClient()
    ContentPlannerAgent(llm_client=info_client).plan(
        "빈티지와 구제의 차이를 설명하고 싶다.",
        make_idea("빈티지와 구제의 차이", "정보 가이드", "차이 설명"),
    )
    assert len(info_client.calls) == 1

    comparison_client = MockLLMClient()
    ContentPlannerAgent(llm_client=comparison_client).plan(
        "의류수거함과 헌옷 기부의 차이를 비교하고 싶다.",
        make_idea("의류수거함과 기부 비교", "비교 가이드", "선택 기준 비교"),
    )
    assert len(comparison_client.calls) == 1


def test_trend_editorial_mode_uses_one_source_grounded_planner_call():
    source = "Y2K가 다시 유행인데 새 옷 대신 빈티지에서 찾아보자."
    client = MockLLMClient()
    planner = ContentPlannerAgent(llm_client=client)

    plan = planner.plan(source, make_idea("Y2K 빈티지 활용", "트렌드 에디토리얼", "유행과 선택"))

    assert plan.writing_script
    assert len(client.calls) == 1
    payload = client.calls[0]["payload"]
    assert payload["planning_mode"] == "source_grounded_editorial"
    assert "PLANNING MODE: SOURCE-GROUNDED TREND/EDITORIAL" in client.calls[0]["prompt"]


def test_content_planner_keeps_guide_and_comparison_in_generative_mode():
    guide_client = MockLLMClient()
    comparison_client = MockLLMClient()

    ContentPlannerAgent(llm_client=guide_client).plan(
        "빈티지 쇼핑할 때 옷 상태 확인법을 체크리스트로 정리하고 싶다.",
        make_idea("빈티지 쇼핑 체크리스트", "체크리스트", "상태 확인 방법"),
    )
    ContentPlannerAgent(llm_client=comparison_client).plan(
        "빈티지와 구제의 차이를 비교하고 싶다.",
        make_idea("빈티지와 구제 비교", "비교 가이드", "차이 비교"),
    )

    assert guide_client.calls[0]["payload"]["planning_mode"] == "generative_structured"
    assert comparison_client.calls[0]["payload"]["planning_mode"] == "generative_structured"


def test_raw_source_authority_beats_refined_guide_for_trend_editorial():
    source = "Y2K가 다시 유행인데 굳이 새 옷을 살 필요가 있을까? 빈티지에서 찾아보자."
    idea = make_idea("Y2K 빈티지 쇼핑", "가이드", "실용적인 쇼핑 가이드")
    client = MockLLMClient()

    assert ContentPlannerAgent.planning_mode(source, idea) == "source_grounded_editorial"
    ContentPlannerAgent(llm_client=client).plan(source, idea)
    assert client.calls[0]["payload"]["planning_mode"] == "source_grounded_editorial"


def test_explicit_raw_guide_beats_refined_experience_format():
    source = "빈티지 쇼핑할 때 옷 상태 확인하는 방법을 체크리스트로 정리하고 싶다."
    idea = make_idea("빈티지 쇼핑 체크리스트", "경험 후기", "개인 경험")

    assert ContentPlannerAgent.planning_mode(source, idea) == "generative_structured"


def test_raw_experience_beats_refined_non_experience_format():
    source = "오슬로에서 산 체크셔츠를 인턴 출근룩으로 입어봤다."
    idea = make_idea("체크셔츠 출근 코디", "코디 아이디어", "스타일 제안")
    client = MockLLMClient()

    assert ContentPlannerAgent.planning_mode(source, idea) == "experience_deterministic"
    ContentPlannerAgent(llm_client=client).plan(source, idea)
    assert len(client.calls) == 0


def test_ambiguous_raw_source_uses_refined_format_as_fallback():
    source = "가을 옷 주제로 글을 써보고 싶다."
    guide = make_idea("가을 옷 정리", "가이드", "실용 정보")
    trend = make_idea("가을 옷 이야기", "트렌드 에디토리얼", "스타일 의견")

    assert ContentPlannerAgent.planning_mode(source, guide) == "generative_structured"
    assert ContentPlannerAgent.planning_mode(source, trend) == "source_grounded_editorial"


def test_planner_keeps_information_and_trend_scripts_flexible():
    client = MockLLMClient()
    planner = ContentPlannerAgent(llm_client=client)
    info_source = "헌옷을 기부하거나 판매하는 선택 기준을 설명하고 싶다."
    trend_source = "Y2K 패션을 새 옷 말고 빈티지 쇼핑에서 찾아보자."

    info = planner.plan(info_source, make_idea("헌옷 처리 선택 기준", "비교 가이드", "선택 기준 비교"))
    trend = planner.plan(trend_source, make_idea("Y2K 빈티지 쇼핑", "스타일 가이드", "트렌드와 탐색"))

    assert "기부하거나 판매" in info.writing_script
    assert "Y2K" in trend.writing_script
    assert info.writing_script != trend.writing_script


def test_planner_prompt_prioritizes_raw_memo_and_prunes_unsupported_expansion():
    prompt = " ".join(ContentPlannerAgent(llm_client=MockLLMClient()).prompt.split())

    assert "source of truth" in prompt
    assert "You may omit outline branches" in prompt
    assert "general option or conditional suggestion" in prompt
    assert "scene-setting" in prompt


def test_planner_payload_uses_safe_editorial_context_only():
    source = "의류수거함과 기부의 차이를 설명하고 싶다."
    idea = make_idea("의류 처리 선택 기준", "정보 가이드", "선택 기준")
    idea.summary = "동료들의 반응과 오슬로 인턴 생활을 소개합니다."
    idea.angle = "오슬로 인턴 경험과 동료 반응을 중심으로 씁니다."
    idea.outline = ["동료들의 반응 소개"]
    client = MockLLMClient()

    ContentPlannerAgent(llm_client=client).plan(source, idea)

    payload = client.calls[0]["payload"]
    assert payload["raw_source"] == source
    assert payload["editorial_context"]["title"] == idea.title
    assert "summary" not in payload["editorial_context"]
    assert "angle" not in payload["editorial_context"]
    assert "outline" not in payload["editorial_context"]
    assert "search_intent" not in payload["editorial_context"]
    assert "content_perspective" not in payload["editorial_context"]
    assert "key_question" not in payload["editorial_context"]
    assert "summary" not in payload["refined_blog_idea"]
    assert "outline" not in payload["refined_blog_idea"]
    assert "search_intent" not in payload["refined_blog_idea"]
    assert "content_perspective" not in payload["refined_blog_idea"]
    assert "key_question" not in payload["refined_blog_idea"]


def test_planner_prompt_does_not_turn_missing_experience_into_a_writing_task():
    prompt = " ".join(ContentPlannerAgent(llm_client=MockLLMClient()).prompt.split())

    assert "sole factual authority" in prompt
    assert "Missing experience is a reason to omit the detail" in prompt
    assert "circular economy" in prompt
    assert "new research task" in prompt


def test_planner_rejects_empty_source():
    planner = ContentPlannerAgent(llm_client=MockLLMClient())

    try:
        planner.plan("", make_idea("제목", "정보", "정보"))
    except ValueError as error:
        assert "source" in str(error)
    else:
        raise AssertionError("Empty planner source should be rejected")
