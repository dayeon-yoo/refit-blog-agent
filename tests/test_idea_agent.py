from unittest.mock import Mock

import pytest

from agents.idea_agent import IdeaGenerator
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
    writer_client = MockLLMClient()
    provider = MockKeywordDataProvider()
    idea, result = run_refined_manual_workflow(
        SOURCE,
        "1",
        "실제 코디 사례를 포함해줘",
        generator=IdeaGenerator(llm_client=generator_client),
        writer=WriterAgent(llm_client=writer_client),
        tag_agent=TagAgent(provider=provider),
    )

    assert len(generator_client.calls) == 2
    assert len(writer_client.calls) == 1
    assert result.ideas == [idea]
    assert result.top_3[0].idea == idea
    assert len(result.blog_posts) == 1
    post = result.blog_posts[0]
    assert post.title == idea.title
    assert post.keyword == idea.keyword
    assert post.tags is not None
    assert post.image_plan is not None
    assert post.image_plan.images
    assert provider.was_called
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
