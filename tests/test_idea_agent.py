from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import agents.idea_agent as idea_agent_module
from agents.idea_agent import IdeaGenerator
from agents.content_planner_agent import ContentPlannerAgent
from agents.writer_agent import WriterAgent
from agents.tag_agent import MockKeywordDataProvider, TagAgent
from llm.client import MockLLMClient
from main import refine_from_source, run_refined_manual_workflow
from models.schemas import BlogIdea, IdeaCandidate, IdeaExpansionResult
from orchestrator.workflow import run_manual_workflow


SOURCE = "민트코어가 유행 끝나가는 것 같고 레몬코어나 포도색이 새롭게 뜨는 것 같아."
VINTAGE_SOURCE = "인턴일기1: 빈티지 체크 셔츠의 세계(내가 직접 빈티지 체크셔츠를 산 경험을 토대로)"
RECYCLING_SOURCE = "헌옷 수거함에 넣은 옷은 정말 재활용될까?"


def test_expand_returns_distinct_source_related_candidates():
    client = MockLLMClient()
    result = IdeaGenerator(llm_client=client).expand(SOURCE)

    assert len(result.candidates) == 3
    assert result.source_input == SOURCE
    assert len({candidate.candidate_id for candidate in result.candidates}) == 3
    assert len({candidate.title for candidate in result.candidates}) == 3
    assert len({(candidate.perspective, candidate.content_format) for candidate in result.candidates}) >= 2
    assert any("민트코어" in candidate.title for candidate in result.candidates)
    assert any("레몬" in candidate.title or "포도" in candidate.title for candidate in result.candidates)
    assert all(SOURCE not in candidate.title for candidate in result.candidates)
    assert client.calls[0]["payload"]["user_input"] == SOURCE


@pytest.mark.parametrize(
    ("original", "returned"),
    [
        ("y2k 패션이 다시 유행이다.", "Y2K 패션이 다시 유행이다."),
        ("소액 의 돈으로 돌아온다", "소액의 돈으로 돌아온다"),
    ],
)
def test_expansion_restores_application_owned_source_provenance(original, returned):
    def response_factory(_prompt, _schema, _payload):
        titles = (
            ["Y2K 패션의 흐름", "Y2K 패션 스타일 관찰", "Y2K 패션 활용 이야기"]
            if "y2k" in original.lower()
            else ["소액의 돈으로 돌아오는 옷", "소액의 돈과 옷 활용", "옷이 돈으로 돌아오는 과정"]
        )
        return _expansion_with_titles(
            returned,
            titles,
        )

    client = MockLLMClient(response_factory=response_factory)
    result = IdeaGenerator(llm_client=client).expand(original)

    assert result.source_input == original
    assert len(client.calls) == 1


def test_expansion_provenance_restore_does_not_remove_candidate_semantic_guardrail():
    original = "y2k 패션이 다시 유행이다."

    def response_factory(_prompt, _schema, _payload):
        return _expansion_with_titles(
            "Y2K 패션이 다시 유행이다.",
            ["가을 데님 코디", "겨울 니트 관리", "출근 가방 추천"],
        )

    with pytest.raises(ValueError, match="unrelated"):
        IdeaGenerator(llm_client=MockLLMClient(response_factory=response_factory)).expand(original)


def test_expand_clamps_candidate_count_to_supported_range():
    generator = IdeaGenerator(llm_client=MockLLMClient())

    assert len(generator.expand(SOURCE, candidate_count=1).candidates) == 3
    assert len(generator.expand(SOURCE, candidate_count=9).candidates) == 5


def test_refine_preserves_candidate_planning_fields_and_revision_request():
    generator = IdeaGenerator(llm_client=MockLLMClient())
    candidate = generator.expand(SOURCE, candidate_count=3).candidates[2]

    idea = generator.refine(SOURCE, candidate, "실제 코디 사례를 포함해줘")

    assert isinstance(idea, BlogIdea)
    assert idea.title == candidate.title
    assert idea.keyword
    assert candidate.perspective in idea.angle
    assert candidate.content_format in idea.angle
    assert idea.key_question == candidate.key_question
    assert idea.outline and len(idea.outline) >= 3
    assert "실제 코디 사례" in idea.summary
    assert 0.0 <= idea.seasonality <= 1.0


def test_refined_idea_connects_to_existing_writer_tag_image_pipeline():
    generator = IdeaGenerator(llm_client=MockLLMClient())
    candidate = generator.expand(SOURCE).candidates[0]
    idea = generator.refine(SOURCE, candidate)
    client = MockLLMClient()

    result = run_manual_workflow(
        idea,
        writer=WriterAgent(llm_client=client),
        tag_agent=TagAgent(provider=MockKeywordDataProvider()),
    )

    assert len(result.blog_posts) == 1
    post = result.blog_posts[0]
    assert post.title == idea.title
    assert post.keyword == idea.keyword
    assert post.tags is not None
    assert post.image_plan is not None
    assert post.image_plan.images


def test_refined_candidate_runs_once_through_writer_tag_image_pipeline():
    generator_client = MockLLMClient()
    planner_client = MockLLMClient()
    writer_client = MockLLMClient()
    provider = MockKeywordDataProvider()
    idea, content_plan, result = run_refined_manual_workflow(
        SOURCE,
        "1",
        "실제 코디 사례를 포함해줘",
        generator=IdeaGenerator(llm_client=generator_client),
        content_planner=ContentPlannerAgent(llm_client=planner_client),
        writer=WriterAgent(llm_client=writer_client),
        tag_agent=TagAgent(provider=provider),
    )

    assert len(generator_client.calls) == 2
    assert len(planner_client.calls) == 1
    assert len(writer_client.calls) == 1
    assert content_plan.writing_script
    assert planner_client.calls[0]["payload"]["raw_source"] == SOURCE
    assert result.ideas == [idea]
    assert result.top_3[0].idea == idea
    assert len(result.blog_posts) == 1
    post = result.blog_posts[0]
    assert post.title == idea.title
    assert post.keyword == idea.keyword
    assert post.tags is not None
    assert post.image_plan is not None
    assert post.image_plan.images
    # Trend-only mock output may produce no valid tag candidates; if it does,
    # the provider must still be the component that receives them.
    assert not post.tags or provider.was_called
    writer_payload = writer_client.calls[0]["payload"]
    assert writer_payload["raw_source"] == SOURCE
    assert writer_payload["refined_blog_idea"]["title"] == idea.title
    assert writer_payload["content_plan"]["writing_script"] == content_plan.writing_script
    assert isinstance(result.model_dump(mode="json"), dict)


def test_experience_input_becomes_distinct_natural_content_directions():
    result = IdeaGenerator(llm_client=MockLLMClient()).expand(VINTAGE_SOURCE, candidate_count=5)

    assert len(result.candidates) == 5
    assert all(VINTAGE_SOURCE not in candidate.title for candidate in result.candidates)
    assert all("빈티지" in candidate.title or "셔츠" in candidate.title for candidate in result.candidates)
    assert {candidate.content_format for candidate in result.candidates} >= {"경험 후기", "구매 가이드", "스타일링 실험"}
    assert not all(candidate.perspective in {"흐름 분석", "원인 분석", "선택지 비교"} for candidate in result.candidates)
    assert any("직접" in candidate.title or "사보니" in candidate.title for candidate in result.candidates)


def test_question_input_generates_research_and_consumer_directions():
    result = IdeaGenerator(llm_client=MockLLMClient()).expand(RECYCLING_SOURCE, candidate_count=5)

    assert len(result.candidates) == 5
    assert all(RECYCLING_SOURCE not in candidate.title for candidate in result.candidates)
    assert all("헌옷" in candidate.title or "재활용" in candidate.title for candidate in result.candidates)
    assert len({candidate.content_format for candidate in result.candidates}) >= 3
    assert any(candidate.content_format == "팩트체크" for candidate in result.candidates)


def test_expansion_uses_the_input_as_observation_and_question_not_a_noun_list():
    source = "민트색은 예뻐서 샀지만 요즘은 손이 덜 가고, 레몬색을 대신 입어볼까 고민 중이야."
    result = IdeaGenerator(llm_client=MockLLMClient()).expand(source, candidate_count=5)

    assert len(result.candidates) == 5
    assert all(source not in candidate.title for candidate in result.candidates)
    assert any("대신" in candidate.title or "레몬" in candidate.title for candidate in result.candidates)
    assert any("손이 덜" in candidate.brief_description or "예뻐서" in candidate.brief_description for candidate in result.candidates)
    assert len({candidate.content_format for candidate in result.candidates}) >= 3


def test_non_experiential_collection_input_does_not_invent_personal_experience():
    source = "의류수거함 분류 기준: 의류 수거 대신 기부해보세요"
    result = IdeaGenerator(llm_client=MockLLMClient()).expand(source, candidate_count=5)
    combined = " ".join(
        f"{candidate.title} {candidate.perspective} {candidate.content_format} {candidate.key_question} {candidate.brief_description}"
        for candidate in result.candidates
    )

    assert "의류수거함" in combined
    assert "기부" in combined
    assert any(word in combined for word in ("분류", "보내기", "수거"))
    assert "내가 직접" not in combined
    assert "개인적인 경험" not in combined
    assert "직접 해본 결과" not in combined
    assert all(candidate.content_format not in {"경험 후기", "비교 후기"} for candidate in result.candidates)


def test_vintage_shopping_input_stays_korean_and_does_not_assume_external_research():
    source = "Y2K 패션 빈티지 쇼핑에서 찾기"
    result = IdeaGenerator(llm_client=MockLLMClient()).expand(source, candidate_count=5)
    combined = " ".join(
        f"{candidate.title} {candidate.perspective} {candidate.content_format} {candidate.key_question} {candidate.brief_description}"
        for candidate in result.candidates
    )

    assert len(result.candidates) == 5
    assert all("Y2K" in candidate.title or "빈티지" in candidate.title for candidate in result.candidates)
    assert "매장" not in combined
    assert "플랫폼" not in combined
    assert "인기 순위" not in combined
    assert "내가 직접" not in combined
    assert len({candidate.content_format for candidate in result.candidates}) >= 3


def test_non_experiential_shopping_input_does_not_use_experience_language():
    source = "y2k 패션 새 옷 말고 빈티지 쇼핑에서 찾기"
    result = IdeaGenerator(llm_client=MockLLMClient()).expand(source, candidate_count=5)
    combined = " ".join(
        f"{candidate.title} {candidate.perspective} {candidate.content_format} {candidate.key_question} {candidate.brief_description}"
        for candidate in result.candidates
    )
    forbidden_experience = ("발견한", "찾은", "찾아본", "직접 해보니", "입어보니", "사보니", "써보니", "경험해보니")

    assert not any(expression in combined for expression in forbidden_experience)
    assert "새 옷 말고" in combined or "빈티지" in combined
    assert len({candidate.content_format for candidate in result.candidates}) >= 3


def test_expansion_prompt_contains_experience_and_article_distinctness_rules():
    prompt = IdeaGenerator(llm_client=MockLLMClient()).expansion_prompt

    assert "source explicitly states" in prompt
    assert "직접 해보니" in prompt
    assert "different article brief" in prompt
    assert "새 옷 말고" in prompt


def test_expansion_prompt_avoids_unsupported_details_audience_splitting_and_authority():
    prompt = IdeaGenerator(llm_client=MockLLMClient()).expansion_prompt

    assert "unmentioned tool, product, place" in prompt
    assert "target-reader" in prompt
    assert "정리 전문가" in prompt
    assert "스타일리스트 관점" in prompt
    assert "what they explain" in prompt


def _expansion_with_titles(source: str, titles: list[str]) -> IdeaExpansionResult:
    return IdeaExpansionResult(
        source_input=source,
        candidates=[
            IdeaCandidate(
                candidate_id=f"candidate-{index}",
                title=title,
                perspective=f"관점 {index}",
                content_format=f"형식 {index}",
                key_question=f"{title}을 어떻게 볼까요?",
                brief_description=f"{title}의 선택 기준과 다음 행동을 살펴봅니다.",
            )
            for index, title in enumerate(titles, 1)
        ],
    )


def test_expansion_validation_accepts_natural_synonyms_and_ignores_incidental_region():
    source = "춘천, 옷 버리지 말고 기부하세요"
    result = _expansion_with_titles(
        source,
        [
            "버릴 옷, 기부로 다시 쓰이게 하는 방법",
            "안 입는 의류를 버리는 대신 기부하는 방법",
            "옷을 버리지 않고 기부로 보내는 선택",
        ],
    )

    assert IdeaGenerator._validate_expansion(source, result, 3) == result


def test_expansion_validation_still_rejects_unrelated_topic():
    source = "춘천, 옷 버리지 말고 기부하세요"
    result = _expansion_with_titles(
        source,
        [
            "가을 데님 코디 5가지",
            "가을 셔츠를 활용한 출근 스타일",
            "겨울 니트 색상 조합 가이드",
        ],
    )

    try:
        IdeaGenerator._validate_expansion(source, result, 3)
    except ValueError as exc:
        assert "unrelated" in str(exc)
    else:
        raise AssertionError("Unrelated expansion candidate should be rejected")


def test_expansion_validation_rejects_fabricated_personal_experience():
    source = "안 입는 옷을 팔아서 돈 벌기"
    result = IdeaExpansionResult(
        source_input=source,
        candidates=[
            IdeaCandidate(
                candidate_id="candidate-1",
                title="안 입는 옷을 직접 팔아보니 알게 된 점",
                perspective="개인적인 경험",
                content_format="후기",
                key_question="안 입는 옷을 팔아보니 무엇이 달랐을까요?",
                brief_description="내가 의류 판매를 통해 얻은 경험과 교훈을 공유합니다.",
            ),
            IdeaCandidate(
                candidate_id="candidate-2",
                title="안 입는 옷을 정리하고 판매하는 방법",
                perspective="실용 정보",
                content_format="가이드",
                key_question="판매 전에 무엇을 확인해야 할까요?",
                brief_description="판매할 옷을 고르고 준비하는 기준을 정리합니다.",
            ),
            IdeaCandidate(
                candidate_id="candidate-3",
                title="안 입는 옷을 판매할 때의 선택 기준",
                perspective="판단 기준",
                content_format="체크리스트",
                key_question="어떤 옷을 판매 대상으로 정할까요?",
                brief_description="상태와 활용 가능성을 기준으로 판단하는 방법을 설명합니다.",
            ),
        ],
    )

    try:
        IdeaGenerator._validate_expansion(source, result, 3)
    except ValueError as exc:
        assert "personal experience" in str(exc)
    else:
        raise AssertionError("Fabricated personal experience should be rejected")


def test_expansion_validation_allows_informational_comparison_without_personal_experience():
    source = "빈티지와 구제의 차이를 설명하고 싶다."
    result = IdeaExpansionResult(
        source_input=source,
        candidates=[
            IdeaCandidate(
                candidate_id="candidate-1",
                title="빈티지와 구제는 어떻게 다를까?",
                perspective="개념 비교",
                content_format="비교",
                key_question="두 용어의 차이는 무엇일까요?",
                brief_description="두 개념의 의미와 차이를 정보형으로 설명합니다.",
            ),
            IdeaCandidate(
                candidate_id="candidate-2",
                title="빈티지와 구제를 구분하는 기준",
                perspective="판단 기준",
                content_format="정보 가이드",
                key_question="두 개념을 어떤 기준으로 구분할까요?",
                brief_description="개념과 사용 맥락을 중심으로 정리합니다.",
            ),
            IdeaCandidate(
                candidate_id="candidate-3",
                title="다시 사용하는 옷을 설명하는 두 가지 표현",
                perspective="용어 해설",
                content_format="해설",
                key_question="빈티지와 구제라는 표현은 어떻게 쓰일까요?",
                brief_description="두 표현의 공통점과 차이를 일반적인 정보로 설명합니다.",
            ),
        ],
    )

    assert IdeaGenerator._validate_expansion(source, result, 3) == result


def test_expansion_validation_allows_experience_when_source_explicitly_provides_it():
    source = "내가 빈티지 체크셔츠를 직접 사봤는데 이 경험으로 글 쓰고 싶어"
    result = IdeaExpansionResult(
        source_input=source,
        candidates=[
            IdeaCandidate(
                candidate_id="candidate-1",
                title="빈티지 체크 셔츠를 직접 사보니 알게 된 점",
                perspective="구매 경험",
                content_format="후기",
                key_question="직접 구매한 셔츠에서 무엇을 확인했을까요?",
                brief_description="직접 구매하고 입어본 경험을 바탕으로 선택 기준을 정리합니다.",
            ),
            IdeaCandidate(
                candidate_id="candidate-2",
                title="빈티지 체크 셔츠를 고를 때 확인할 기준",
                perspective="선택 기준",
                content_format="구매 가이드",
                key_question="빈티지 셔츠는 무엇을 보고 골라야 할까요?",
                brief_description="구매 경험에서 얻은 기준을 정보형 가이드로 정리합니다.",
            ),
            IdeaCandidate(
                candidate_id="candidate-3",
                title="빈티지 체크 셔츠로 출근 코디 구성하기",
                perspective="스타일링",
                content_format="스타일링 가이드",
                key_question="체크 셔츠를 어떻게 일상 코디에 활용할까요?",
                brief_description="직접 입어본 셔츠를 다른 옷과 조합하는 방법을 소개합니다.",
            ),
        ],
    )

    assert IdeaGenerator._validate_expansion(source, result, 3) == result


def test_refinement_cli_helper_selects_candidate_and_returns_blog_idea():
    generator = IdeaGenerator(llm_client=MockLLMClient())
    idea = refine_from_source(
        "의류수거함 분류 기준: 의류 수거 대신 기부해보세요",
        "2",
        "기부 전 상태 확인 기준을 포함해줘",
        generator=generator,
    )

    assert isinstance(idea, BlogIdea)
    assert idea.title
    assert idea.keyword
    assert idea.angle
    assert idea.summary
    assert idea.content_format
    assert idea.content_perspective
    assert idea.key_question
    assert idea.outline and len(idea.outline) >= 3


def test_refine_from_source_matches_numeric_cli_id_to_string_candidate_id():
    source = "의류수거함 분류 기준: 의류 수거 대신 기부해보세요"
    candidates = [
        IdeaCandidate(
            candidate_id=str(index),
            title=f"후보 {index}",
            perspective=f"관점 {index}",
            content_format="가이드",
            key_question=f"질문 {index}",
            brief_description=f"설명 {index}",
        )
        for index in range(1, 4)
    ]
    expected = BlogIdea(
        title="후보 2",
        keyword="의류 기부",
        search_intent="정보 탐색",
        angle="관점 2",
        rifit_connection="의류 순환",
        seasonality=0.5,
        summary="의류 기부 방법을 안내합니다.",
    )
    generator = Mock()
    generator.expand.return_value = IdeaExpansionResult(source_input=source, candidates=candidates)
    generator.refine.return_value = expected

    result = refine_from_source(source, 2, generator=generator)

    assert result is expected
    assert generator.refine.call_args.args[1].candidate_id == "2"


def test_refine_from_source_keeps_unknown_candidate_error():
    candidates = [
        IdeaCandidate(
            candidate_id=str(index),
            title=f"후보 {index}",
            perspective=f"관점 {index}",
            content_format="가이드",
            key_question=f"질문 {index}",
            brief_description=f"설명 {index}",
        )
        for index in range(1, 4)
    ]
    generator = Mock()
    generator.expand.return_value = IdeaExpansionResult(source_input="source", candidates=candidates)

    with pytest.raises(ValueError, match=r"Unknown candidate_id 9; available: 1, 2, 3"):
        refine_from_source("source", 9, generator=generator)


def _refinement_candidate(perspective: str) -> IdeaCandidate:
    return IdeaCandidate(
        candidate_id="candidate-1",
        title="안 입는 옷의 처리 방법",
        perspective=perspective,
        content_format="가이드",
        key_question="어떤 옷을 어떻게 처리할까요?",
        brief_description="옷의 상태에 따라 적절한 처리 방법을 비교합니다.",
    )


def _refined_idea(**overrides) -> BlogIdea:
    values = {
        "title": "안 입는 옷의 처리 방법",
        "keyword": "안 입는 옷 처리",
        "search_intent": "정보 탐색",
        "angle": "기부와 의류 수거의 차이를 설명하는 가이드이자 선택 기준을 안내합니다.",
        "rifit_connection": "의류 순환으로 연결합니다.",
        "seasonality": 0.5,
        "summary": "어떤 옷을 어떻게 처리할까요? 기부 전 상태를 확인하고 적합한 처리 방법을 선택합니다.",
        "content_format": "가이드",
        "content_perspective": "의류 처리 선택 기준",
        "key_question": "어떤 옷은 기부하고 어떤 옷은 수거로 보낼까요?",
        "outline": ["상태 확인", "기부와 수거 비교", "처리 방법 선택"],
    }
    values.update(overrides)
    return BlogIdea(**values)


def test_refinement_accepts_natural_perspective_variation():
    candidate = _refinement_candidate("기부와 의류 수거 비교")
    idea = _refined_idea()

    assert IdeaGenerator._validate_refinement("source", candidate, idea, "") == idea


def test_refinement_rejects_unrelated_perspective():
    candidate = _refinement_candidate("기부와 의류 수거 비교")
    idea = _refined_idea(
        title="가을 데님 스타일링",
        keyword="가을 데님 코디",
        angle="가을 데님으로 다양한 코디를 만드는 방법입니다.",
        summary="가을 데님 스타일링 팁을 소개합니다.",
        content_perspective="스타일링 방법",
        key_question="가을 데님을 어떻게 코디할까요?",
        outline=["색상 조합", "상의 선택", "계절 코디"],
    )

    with pytest.raises(ValueError, match="candidate='기부와 의류 수거 비교'"):
        IdeaGenerator._validate_refinement("source", candidate, idea, "")


@pytest.mark.parametrize(
    ("candidate_format", "refined_format"),
    [
        ("비교 기사", "비교 가이드"),
        ("가이드", "단계별 가이드"),
        ("체크리스트", "실행 체크리스트"),
    ],
)
def test_refinement_accepts_related_content_format_variations(candidate_format, refined_format):
    candidate = _refinement_candidate("기부와 의류 수거 비교").model_copy(
        update={"content_format": candidate_format}
    )
    idea = _refined_idea(content_format=refined_format)

    assert IdeaGenerator._validate_refinement("source", candidate, idea, "") == idea


def test_refinement_prefers_explicit_source_format_over_candidate_format():
    source = "인턴일기: 체크셔츠로 출근 코디를 직접 해봤다."
    candidate = IdeaCandidate(
        candidate_id="candidate-1",
        title="체크셔츠 출근 코디",
        perspective="출근 코디 경험",
        content_format="주장 및 정보 제공",
        key_question="체크셔츠로 출근 코디를 어떻게 해봤을까요?",
        brief_description="체크셔츠를 출근 코디에 활용한 방향을 정리합니다.",
    )
    idea = _refined_idea(
        title="체크셔츠 출근 코디 경험",
        keyword="체크셔츠 출근 코디",
        angle="개인 경험 및 코디 리뷰 관점에서 체크셔츠를 출근에 활용한 과정을 정리합니다.",
        summary="체크셔츠로 출근 코디를 어떻게 해봤을까요? 직접 활용한 경험을 소개합니다.",
        content_format="개인 경험 및 코디 리뷰",
        content_perspective="개인 경험 및 코디 리뷰",
        key_question="체크셔츠로 출근 코디를 어떻게 해봤을까요?",
        outline=["체크셔츠를 산 계기", "출근 코디에 활용한 경험", "코디 아이디어"],
    )

    assert IdeaGenerator._validate_refinement(source, candidate, idea, "") == idea


def test_refinement_prefers_explicit_source_perspective_over_candidate_label():
    source = "인턴일기: 체크셔츠로 출근 코디를 직접 해봤다."
    candidate = IdeaCandidate(
        candidate_id="candidate-1",
        title="체크셔츠 출근 코디",
        perspective="소셜 미디어 트렌드",
        content_format="주장 및 정보 제공",
        key_question="체크셔츠로 출근 코디를 어떻게 해봤을까요?",
        brief_description="체크셔츠 출근 코디를 중심으로 글을 구성합니다.",
    )
    idea = _refined_idea(
        title="체크셔츠 출근 코디 경험",
        keyword="체크셔츠 출근 코디",
        angle="개인 경험 공유 관점에서 체크셔츠를 출근 코디에 활용한 과정을 정리합니다.",
        summary="체크셔츠로 출근 코디를 어떻게 해봤을까요? 직접 활용한 경험을 소개합니다.",
        content_format="개인 경험 및 코디 리뷰",
        content_perspective="개인 경험 공유",
        key_question="체크셔츠로 출근 코디를 어떻게 해봤을까요?",
        outline=["체크셔츠를 산 계기", "출근 코디에 활용한 경험", "코디 아이디어"],
    )

    assert IdeaGenerator._validate_refinement(source, candidate, idea, "") == idea


def test_refinement_perspective_uses_candidate_when_source_is_ambiguous():
    source = "가을 옷 주제로 써보고 싶다."
    assert not IdeaGenerator._preserves_refinement_perspective(
        source,
        "실용적인 옷장 정리",
        "개인 여행 후기",
    )


def test_refinement_perspective_does_not_change_y2k_exploration_to_travel_experience():
    source = "Y2K 패션을 빈티지에서 찾아보는 방법"

    assert not IdeaGenerator._preserves_refinement_perspective(
        source,
        "트렌드 탐색",
        "개인 여행 경험",
    )


def test_refinement_perspective_preserves_explicit_comparison():
    source = "의류수거함과 기부의 차이를 비교하고 싶다."

    assert IdeaGenerator._preserves_refinement_perspective(
        source,
        "비교형",
        "의류 처리 방법을 비교하고 선택 기준을 안내하는 정보형 글",
    )
    assert not IdeaGenerator._preserves_refinement_perspective(
        source,
        "비교형",
        "개인 일기로 기부 경험을 기록하는 글",
    )


def test_refinement_keeps_explicit_source_comparison_requirement():
    source = "의류수거함과 기부의 차이를 비교하고 싶다."
    assert IdeaGenerator._source_format_families(source) == {"comparison"}
    assert not IdeaGenerator._preserves_refinement_format(
        source,
        "비교형",
        "개인 경험 일기로 기부 과정을 기록하는 글",
    )


def test_refinement_allows_source_styling_format_to_specialize():
    source = "Y2K 패션을 빈티지 아이템으로 즐기는 방법"

    assert IdeaGenerator._source_format_families(source) == {"guide", "styling"}
    assert IdeaGenerator._preserves_refinement_format(
        source,
        "스타일 탐색",
        "Y2K 빈티지 스타일링 가이드: 아이템을 즐기는 방법",
    )


def test_refinement_uses_candidate_format_when_source_format_is_ambiguous():
    source = "가을 옷 정리 주제로 써보고 싶다."

    assert not IdeaGenerator._source_format_families(source)
    assert not IdeaGenerator._preserves_refinement_format(
        source,
        "실용 가이드",
        "개인 여행 후기",
    )


def test_identical_candidate_and_refined_format_passes_before_source_family_check():
    source = "Y2K가 다시 유행인데 굳이 새 옷을 살 필요가 있을까? 새 옷 대신 빈티지를 활용해보자."

    assert IdeaGenerator._source_format_families(source).isdisjoint({"comparison"})
    assert IdeaGenerator._format_families("비교") == {"comparison"}
    assert IdeaGenerator._preserves_refinement_format(source, "비교", "비교")


@pytest.mark.parametrize(
    "source",
    [
        "새 옷 말고 빈티지 옷을 입어보자.",
        "새 옷 대신 기존 옷을 활용해보자.",
        "굳이 새 옷을 살 필요가 있을까?",
        "유행이 지났다고 버릴 필요가 없다.",
    ],
)
def test_editorial_alternative_is_not_explicit_comparison(source):
    assert "comparison" not in IdeaGenerator._source_format_families(source)


@pytest.mark.parametrize(
    ("candidate_format", "refined_format"),
    [("비교 기사", "개인 후기"), ("체크리스트", "인터뷰")],
)
def test_refinement_rejects_changed_content_format_family(candidate_format, refined_format):
    candidate = _refinement_candidate("기부와 의류 수거 비교").model_copy(
        update={"content_format": candidate_format}
    )
    idea = _refined_idea(
        content_format=refined_format,
        angle="안 입는 옷을 관리하고 활용하는 실용적인 내용을 안내합니다.",
        summary="옷의 상태를 살펴보고 관리 방법을 정리합니다.",
        key_question="안 입는 옷을 어떻게 관리할까요?",
        outline=["상태 확인", "관리 방법", "실천 순서"],
    )

    with pytest.raises(ValueError, match="candidate="):
        IdeaGenerator._validate_refinement("source", candidate, idea, "")


def test_refinement_prompt_preserves_contrast_revision_and_concrete_planning_rules():
    prompt = IdeaGenerator(llm_client=MockLLMClient()).refinement_prompt

    assert '"A instead of B"' in prompt
    assert "must not\nreplace the source problem" in prompt
    assert "adds a condition to the existing plan" in prompt
    assert "specific enough to become a useful section heading" in prompt
    assert "Define target_reader by the user's actual problem" in prompt
    assert "rifit_connection as the specific point" in prompt
    assert "would require web research" in prompt


def test_refinement_prompt_preserves_source_actions_and_content_format():
    prompt = " ".join(IdeaGenerator(llm_client=MockLLMClient()).refinement_prompt.split())

    assert "Preserve the source's content action and format" in prompt
    assert "do not turn it into a broad history or trend analysis" in prompt
    assert "keep shopping, selection, and styling as the actions" in prompt
    assert "do not replace them with DIY" in prompt


def test_refinement_prompt_preserves_trend_opinion_contrast_instead_of_new_guide():
    prompt = " ".join(IdeaGenerator(llm_client=MockLLMClient()).refinement_prompt.split())

    assert "trend, style, or opinion-like source" in prompt
    assert "practical shopping guide" in prompt
    assert "굳이 새 옷을 살 필요가 있을까" in prompt
    assert "what to buy, where to shop, or what to check" in prompt


def test_trend_style_source_preserves_format_question_and_contrast():
    source = (
        "Y2K가 다시 유행인데 굳이 새 옷을 살 필요가 있을까? "
        "유행이 지났다고 버릴 필요도 없고 빈티지 매장에서 Y2K를 찾아보는 것도 재미있다."
    )
    families = IdeaGenerator._source_format_families(source)
    assert {"trend", "styling", "editorial"}.issubset(families)

    candidate = _refinement_candidate("Y2K 빈티지 스타일 탐색").model_copy(
        update={
            "title": "Y2K 빈티지 스타일 탐색",
            "content_format": "스타일 탐색",
            "key_question": "새 옷 없이 돌아온 Y2K 스타일을 어떻게 즐길 수 있을까요?",
        }
    )
    idea = _refined_idea(
        title="Y2K가 돌아와도 새 옷이 필요할까",
        keyword="Y2K 빈티지 스타일",
        angle="돌아온 Y2K 유행과 새 옷 대신 빈티지 활용에 대한 스타일 에디토리얼",
        summary="Y2K 유행을 새 옷 구매와 기존 옷 활용의 선택이라는 관점에서 이야기합니다.",
        content_format="트렌드 스타일 에디토리얼",
        content_perspective="Y2K 유행과 빈티지 활용에 대한 의견",
        key_question=candidate.key_question,
        outline=["Y2K 유행의 재등장", "새 옷 대신 빈티지 활용", "유행이 지난 옷을 버리지 않는 선택", "Y2K 빈티지 스타일 탐색"],
    )

    assert IdeaGenerator._validate_refinement(source, candidate, idea, "") == idea


def test_trend_style_source_rejects_unrequested_shopping_guide_format():
    source = "Y2K가 다시 유행인데 굳이 새 옷을 살 필요가 있을까? 빈티지 매장에서 찾아보자."
    candidate = _refinement_candidate("Y2K 빈티지 스타일 탐색").model_copy(
        update={
            "title": "Y2K 빈티지 스타일 탐색",
            "content_format": "스타일 탐색",
            "key_question": "새 옷 없이 Y2K 스타일을 어떻게 즐길 수 있을까요?",
        }
    )
    idea = _refined_idea(
        title="새 옷 대신 Y2K 빈티지 쇼핑 가이드",
        keyword="Y2K 빈티지 쇼핑",
        angle="빈티지 매장에서 Y2K 옷을 고르는 실용 가이드",
        summary="Y2K 빈티지 쇼핑에서 확인할 준비물과 품질 기준을 안내합니다.",
        content_format="가이드",
        content_perspective="빈티지 쇼핑 방법",
        key_question=candidate.key_question,
        outline=["매장 찾기", "준비물", "품질 확인", "Y2K 빈티지 스타일 탐색"],
    )

    with pytest.raises(ValueError, match="format"):
        IdeaGenerator._validate_refinement(source, candidate, idea, "")


def test_refinement_contrast_failure_retries_with_source_derived_feedback():
    source = "Y2K가 다시 유행인데 굳이 새 옷을 살 필요가 있을까? 유행이 지났다고 버릴 필요도 없다."
    candidate = IdeaCandidate(
        candidate_id="candidate-1",
        title="Y2K 빈티지 활용",
        perspective="트렌드 관찰",
        content_format="가이드",
        key_question="Y2K를 어떻게 볼까요?",
        brief_description="Y2K와 기존 옷 활용을 살펴봅니다.",
    )
    invalid = _refined_idea(
        title="Y2K 빈티지 쇼핑 방법",
        keyword="Y2K 빈티지 쇼핑",
        angle="빈티지 쇼핑 방법을 안내합니다.",
        summary="빈티지 매장에서 아이템을 고르는 방법을 정리합니다.",
        content_format="가이드",
        content_perspective="쇼핑 방법",
        key_question=candidate.key_question,
        outline=["매장 찾기", "준비물", "품질 확인"],
    )
    valid = _refined_idea(
        title="Y2K가 돌아와도 새 옷이 필요할까",
        keyword="Y2K 빈티지 활용",
        angle="새 옷 대신 빈티지와 기존 옷을 활용하는 선택을 이야기합니다.",
        summary="유행이 지나도 옷을 버리지 않고 다시 활용하는 방향을 살펴봅니다.",
        content_format="가이드",
        content_perspective="트렌드와 재활용 선택",
        key_question=candidate.key_question,
        outline=["Y2K 유행", "새 옷 대신 빈티지 활용", "유행 지난 옷을 버리지 않는 선택"],
    )

    def response_factory(_prompt, _schema, _payload):
        return invalid if len(client.calls) == 1 else valid

    client = MockLLMClient(response_factory=response_factory)
    result = IdeaGenerator(llm_client=client).refine(source, candidate)

    assert result == valid
    assert len(client.calls) == 2
    feedback = client.calls[1]["payload"]["correction_feedback"]
    assert "buying new clothes" in feedback
    assert "discarded" in feedback
    assert "shopping guide" in feedback


def test_refinement_direction_failure_debug_diagnostic(monkeypatch, capsys):
    source = "Y2K가 다시 유행인데 굳이 새 옷을 살 필요가 있을까?"
    candidate = _refinement_candidate("Y2K 빈티지 활용").model_copy(
        update={"title": "Y2K 빈티지 활용", "key_question": "Y2K를 어떻게 볼까요?"}
    )
    idea = _refined_idea(
        title="Y2K 쇼핑 가이드",
        keyword="Y2K 쇼핑",
        angle="빈티지 쇼핑 방법을 안내합니다.",
        summary="Y2K 아이템을 고르는 방법을 정리합니다.",
        key_question=candidate.key_question,
        outline=["매장 찾기", "준비물", "품질 확인"],
    )
    monkeypatch.setattr(idea_agent_module, "get_settings", lambda: SimpleNamespace(debug=True))

    with pytest.raises(ValueError, match="contrast or editorial argument"):
        IdeaGenerator._validate_refinement(source, candidate, idea, "")

    diagnostic = capsys.readouterr().err
    assert "[Idea Refinement Direction Failure]" in diagnostic
    assert "key_question=" in diagnostic
    assert "outline=" in diagnostic


def test_refinement_rejects_experience_source_changed_to_generic_analysis():
    source = "인턴일기: 오슬로에서 산 체크셔츠로 출근 코디를 해봤다."
    candidate = _refinement_candidate("출근 코디 경험")
    idea = _refined_idea(
        title="체크셔츠와 사무실 패션의 변화",
        keyword="사무실 패션 변화",
        angle="체크셔츠가 직장인 복장에 미친 변화를 분석합니다.",
        summary="사무실 패션의 역사와 변화를 분석합니다.",
        content_format="비교 분석",
        content_perspective="사무실 패션 변화",
        key_question="체크셔츠는 사무실 패션을 어떻게 바꾸었을까요?",
        outline=["체크셔츠의 역사", "사무실 복장 변화", "미래 패션"],
    )

    with pytest.raises(ValueError, match="content direction"):
        IdeaGenerator._validate_refinement(source, candidate, idea, "")


def test_refinement_rejects_vintage_shopping_changed_to_diy():
    source = "Y2K 패션, 새 옷 말고 빈티지 쇼핑에서 찾아보세요."
    candidate = _refinement_candidate("빈티지 쇼핑 선택 가이드").model_copy(
        update={"title": "Y2K 빈티지 쇼핑 가이드", "content_format": "가이드"}
    )
    idea = _refined_idea(
        title="Y2K 스타일을 빈티지 아이템으로 업사이클링하기",
        keyword="Y2K 업사이클링",
        angle="빈티지 아이템을 커팅하고 패치워크하는 DIY 방법",
        summary="Y2K 아이템을 리폼하는 방법을 소개합니다.",
        content_format="가이드",
        content_perspective="DIY 업사이클링",
        key_question="빈티지 아이템을 어떻게 업사이클링할까요?",
        outline=["DIY 업사이클링", "커팅", "패치워크"],
    )

    with pytest.raises(ValueError):
        IdeaGenerator._validate_refinement(source, candidate, idea, "")


def _experience_refinement_candidate() -> IdeaCandidate:
    return IdeaCandidate(
        candidate_id="candidate-1",
        title="빈티지 체크셔츠 출근 코디를 해봤다",
        perspective="출근 코디 경험",
        content_format="경험 후기",
        key_question="체크셔츠로 출근 코디를 어떻게 해봤을까요?",
        brief_description="직접 산 체크셔츠를 출근 코디에 활용한 경험을 정리합니다.",
    )


def _valid_experience_refinement() -> BlogIdea:
    candidate = _experience_refinement_candidate()
    return BlogIdea(
        title=candidate.title,
        keyword="빈티지 체크셔츠 출근 코디",
        search_intent="정보 탐색",
        angle="출근 코디 경험 후기 관점에서 직접 산 체크셔츠를 활용한 과정을 정리합니다.",
        rifit_connection="",
        seasonality=0.5,
        summary="오슬로에서 산 체크셔츠로 출근 코디를 해본 경험과 선택 과정을 소개합니다.",
        content_format=candidate.content_format,
        content_perspective=candidate.perspective,
        key_question=candidate.key_question,
        outline=["체크셔츠를 고른 계기", "출근 코디에 조합한 과정", "입어본 뒤 느낀 점"],
    )


def test_refinement_retries_once_with_validation_feedback():
    source = "인턴일기: 오슬로에서 산 체크셔츠로 출근 코디를 해봤다."
    candidate = _experience_refinement_candidate()
    valid = _valid_experience_refinement()
    invalid = _refined_idea(
        title="체크셔츠와 사무실 패션의 변화",
        keyword="사무실 패션 변화",
        angle="사무실 패션의 변화와 역사를 분석합니다.",
        summary="체크셔츠가 사무실 패션에 미친 영향을 분석합니다.",
        content_format="비교 분석",
        content_perspective="사무실 패션 변화",
        key_question="체크셔츠는 사무실 패션을 어떻게 바꾸었을까요?",
        outline=["체크셔츠의 역사", "사무실 복장 변화", "미래 패션"],
    )

    def response_factory(_prompt, schema, payload):
        return invalid if len(client.calls) == 1 else valid

    client = MockLLMClient(response_factory=response_factory)
    result = IdeaGenerator(llm_client=client).refine(source, candidate)

    assert result == valid
    assert len(client.calls) == 2
    assert "correction_feedback" in client.calls[1]["payload"]
    feedback = client.calls[1]["payload"]["correction_feedback"]
    assert "source explicitly" in feedback
    assert "Oslo" in feedback
    assert "intern commute" in feedback
    assert "trend analysis" in feedback


def test_expansion_retries_fabricated_experience_once():
    source = "Y2K 패션 새 옷 말고 빈티지 쇼핑에서 찾기"

    def response_factory(_prompt, _schema, payload):
        result = MockLLMClient._default_idea_expansion(payload)
        if len(client.calls) == 1:
            result = IdeaExpansionResult.model_validate(result)
            result.candidates[0] = result.candidates[0].model_copy(
                update={
                    "title": "Y2K 패션을 직접 찾아본 후기",
                    "perspective": "개인적인 경험",
                    "content_format": "후기",
                    "brief_description": "직접 쇼핑해본 경험을 공유합니다.",
                }
            )
        return result

    client = MockLLMClient(response_factory=response_factory)
    result = IdeaGenerator(llm_client=client).expand(source)

    assert len(result.candidates) == 3
    assert len(client.calls) == 2
    assert "correction_feedback" in client.calls[1]["payload"]
    assert "Do not invent personal experience" in client.calls[1]["payload"]["correction_feedback"]


def test_validation_failure_after_one_correction_retry_still_raises():
    source = "Y2K 패션 새 옷 말고 빈티지 쇼핑에서 찾기"

    def response_factory(_prompt, _schema, payload):
        result = MockLLMClient._default_idea_expansion(payload)
        result = IdeaExpansionResult.model_validate(result)
        result.candidates[0] = result.candidates[0].model_copy(
            update={
                "title": "Y2K 패션을 직접 찾아본 후기",
                "perspective": "개인적인 경험",
                "content_format": "후기",
                "brief_description": "직접 쇼핑해본 경험을 공유합니다.",
            }
        )
        return result

    client = MockLLMClient(response_factory=response_factory)
    with pytest.raises(ValueError, match="invents personal experience"):
        IdeaGenerator(llm_client=client).expand(source)

    assert len(client.calls) == 2


def test_valid_expansion_does_not_trigger_correction_retry():
    client = MockLLMClient()

    IdeaGenerator(llm_client=client).expand(SOURCE)

    assert len(client.calls) == 1


def test_experience_direction_accepts_natural_phrasing_across_refinement_fields():
    source = "인턴일기: 오슬로에서 산 체크셔츠로 출근 코디를 해봤다."
    candidate = _experience_refinement_candidate()
    idea = _valid_experience_refinement().model_copy(
        update={
            "title": "빈티지 체크셔츠 출근 코디",
            "summary": "오슬로에서 구매한 체크셔츠를 실제 인턴 출근에 활용한 경험을 정리합니다.",
            "content_format": "경험 후기",
            "content_perspective": "실제 착용 기록",
            "outline": [
                "오슬로에서 구매한 체크셔츠 소개",
                "실제로 출근할 때 입어본 코디",
                "착용 후 알게 된 점",
            ],
        }
    )

    assert IdeaGenerator._validate_refinement(source, candidate, idea, "") == idea


def test_alternative_source_does_not_require_comparison_family():
    source = "새 옷 말고 빈티지 쇼핑에서 Y2K를 찾아보세요"
    output = "새 옷을 새로 사기보다 빈티지 쇼핑에서 Y2K 아이템을 찾아 스타일링하는 방법"

    families = IdeaGenerator._source_direction_families(source)
    assert "comparison" not in families
    IdeaGenerator._validate_source_direction(source, output)


def test_exact_y2k_source_detects_shopping_and_guide_not_comparison():
    source = (
        "Y2K 패션, 새 옷 말고 빈티지 쇼핑에서 찾아보세요. "
        "다시 돌아온 Y2K 스타일을 빈티지 아이템으로 즐기는 방법"
    )

    families = IdeaGenerator._source_direction_families(source)

    assert families == {"shopping", "guide"}
    assert "comparison" not in families
    IdeaGenerator._validate_source_direction(
        source,
        "새 옷을 새로 사기보다 빈티지 쇼핑에서 Y2K 아이템을 찾아 즐기는 방법입니다.",
    )


def test_explicit_comparison_source_still_requires_comparison_direction():
    source = "의류 수거함과 기부의 장단점을 비교해보자"

    assert "comparison" in IdeaGenerator._source_direction_families(source)
    with pytest.raises(ValueError, match="missing comparison"):
        IdeaGenerator._validate_source_direction(source, "기부할 옷을 고르는 방법을 안내합니다.")
