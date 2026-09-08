from __future__ import annotations

from agents.tag_agent import KeywordDataProvider, MockKeywordDataProvider, TagAgent
from models.schemas import BlogPost, BlogIdea, ScoredIdea, TagRecommendationResult
from agents.writer_agent import WriterAgent
from agents.image_agent import ImageAgent
from llm.client import MockLLMClient


def make_sample_post() -> BlogPost:
    idea = BlogIdea(
        title="How to grow tomatoes at home",
        keyword="home gardening",
        search_intent="정보형",
        angle="growing tomatoes at home with practical planting and care steps",
        rifit_connection="gardening",
        seasonality=0.5,
        summary="A practical guide to growing and caring for tomatoes at home.",
    )

    scored = ScoredIdea(
        idea_id="id1",
        scores={"score": 80},
        total_score=80,
        reason="good",
        idea=idea,
    )

    writer = WriterAgent(MockLLMClient())
    post = writer.write(scored)
    return post


def make_post(title: str, keyword: str, summary: str, content: str) -> BlogPost:
    return BlogPost(
        title=title,
        keyword=keyword,
        content=content,
        summary=summary,
        cta="작은 항목부터 실천해 보세요.",
    )


class FixedKeywordDataProvider(KeywordDataProvider):
    def __init__(self, metrics: dict[str, dict]):
        self.metrics = metrics
        self.requested_keywords: list[str] = []

    def fetch_metrics(self, keywords: list[str]) -> dict[str, dict]:
        self.requested_keywords = list(keywords)
        return {keyword: self.metrics.get(keyword, {}) for keyword in keywords}


class PartialKeywordDataProvider(KeywordDataProvider):
    def fetch_metrics(self, keywords: list[str]) -> dict[str, dict]:
        partial = {
            keyword: {
                "search_volume": None,
                "competition": None,
                "metrics_available": False,
            }
            for keyword in keywords
        }
        partial[keywords[0]] = {
            "search_volume": 100,
            "competition": 0.9,
            "metrics_available": True,
        }
        error = RuntimeError("temporary provider failure")
        error.partial_metrics = partial
        raise error


def metric(search_volume: int, competition: float, publishing_volume: int, saturation: float) -> dict:
    return {
        "search_volume": search_volume,
        "competition": competition,
        "publishing_volume": publishing_volume,
        "saturation": saturation,
    }


def test_candidates_and_recommendation_basic():
    post = make_sample_post()
    provider = MockKeywordDataProvider()
    agent = TagAgent(provider=provider, max_candidates=120)

    result = agent.recommend(post, candidate_target=60, final_k=30)
    assert isinstance(result, TagRecommendationResult)
    # some tags should be generated
    assert len(result.tags) > 0
    # provider must have been called and is mock
    assert provider.was_called is True


def test_no_irrelevant_tags_and_unique():
    post = make_sample_post()
    provider = MockKeywordDataProvider()
    agent = TagAgent(provider=provider)
    result = agent.recommend(post, candidate_target=60, final_k=30)

    tags = [t.tag for t in result.tags]
    # no duplicates
    assert len(tags) == len(set(tags))
    # limited to at most 30
    assert len(tags) <= 30
    # ensure primary keyword appears
    assert "home gardening" in tags or any("gardening" in t for t in tags)


def test_ranking_sorted_desc():
    post = make_sample_post()
    provider = MockKeywordDataProvider()
    agent = TagAgent(provider=provider)
    result = agent.recommend(post, candidate_target=80, final_k=20)

    scores = [t.ranking_score for t in result.tags]
    assert scores == sorted(scores, reverse=True)


def test_partial_provider_metrics_do_not_fail_tag_recommendation():
    post = make_post(
        "가을 옷장 정리 방법",
        "가을 옷장 정리",
        "가을 옷을 정리하는 방법",
        "## 옷장 정리\n가을 옷장을 정리합니다.",
    )

    result = TagAgent(PartialKeywordDataProvider()).recommend(
        post, candidate_target=10, final_k=10
    )

    assert result.tags
    assert any(item.metrics_available for item in result.tags)
    assert any(not item.metrics_available for item in result.tags)
    assert any(item.search_volume is None for item in result.tags)


def test_integration_writer_tag_image_flow():
    # Full small integration: writer -> tag agent -> image agent
    post = make_sample_post()
    provider = MockKeywordDataProvider()
    agent = TagAgent(provider=provider)
    image_agent = ImageAgent()

    # recommend tags and then plan images
    rec = agent.recommend(post, candidate_target=50, final_k=10)
    planned = image_agent.plan(post)

    # post.tags must be set and be strings only
    assert isinstance(post.tags, list)
    assert all(isinstance(t, str) for t in post.tags)
    # image plan must be attached
    assert planned.image_plan is not None


def test_final_tags_never_contain_spaces_and_include_natural_compounds():
    post = make_post(
        "가을 의류 정리로 새로운 스타일 찾기: 옷장 청소의 즐거움",
        "가을 시즌 의류 정리",
        "가을 옷장 속 기존 아이템을 정리하고 조합하는 방법",
        "가을 옷장을 정리하면서 기존 옷을 재조합하고 새로운 스타일을 발견하는 내용",
    )
    agent = TagAgent(MockKeywordDataProvider())
    result = agent.recommend(post, candidate_target=80, final_k=30)
    tags = [item.tag for item in result.tags]

    assert tags
    assert all(not any(char.isspace() for char in tag) for tag in tags)
    assert {"가을옷정리", "가을옷장정리", "가을의류정리", "옷장정리"}.intersection(tags)


def test_malformed_tokens_are_rejected_by_validation_and_result():
    post = make_post("가을 옷장 정리", "가을 옷장 정리", "가을 옷장 정리 방법", "가을 옷장을 정리하는 방법")
    agent = TagAgent(MockKeywordDataProvider())
    malformed = ["대한", "방법을", "새로운", "정리하고", "찾아가는", "아이템을", "시작과"]

    for tag in malformed:
        assert agent.is_valid_tag(tag, post) is False

    result = agent.recommend(post, candidate_target=60, final_k=30)
    assert not set(malformed).intersection(item.tag for item in result.tags)


def test_body_only_generic_words_do_not_become_tags():
    post = make_post(
        "옷장 업사이클링 방법",
        "옷장 업사이클링",
        "입지 않는 옷을 새 용도로 활용하는 방법",
        "본문에는 기부, 수거, 판매, 새로운 아이디어라는 표현이 잠깐 등장하지만 중심은 업사이클링입니다.",
    )
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=60, final_k=30)
    tags = {item.tag for item in result.tags}

    assert any("업사이클링" in tag for tag in tags)
    assert not any(action in tag for tag in tags for action in ("기부", "수거", "판매", "폐기"))


def test_donation_topic_can_generate_donation_tags():
    post = make_post(
        "헌옷 기부 방법",
        "헌옷 기부",
        "입을 수 있는 헌옷을 기부할 때 확인할 기준과 절차",
        "헌옷의 상태를 확인하고 세탁한 뒤 기부처의 안내에 맞춰 전달합니다.",
    )
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=60, final_k=30)
    tags = {item.tag for item in result.tags}

    assert any("기부" in tag for tag in tags)


def test_narrow_topic_is_not_padded_to_thirty_tags():
    post = make_post("니트 보관", "니트 보관", "니트 보관", "니트 보관")
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=60, final_k=30)

    assert 0 < len(result.tags) < 30


def test_provider_receives_only_validated_candidates():
    post = make_post(
        "가을 옷장 정리",
        "가을 옷장 정리",
        "가을 옷장 정리 팁",
        "가을 옷장을 정리하고 계절 옷을 보관하는 방법입니다.",
    )
    provider = MockKeywordDataProvider()
    agent = TagAgent(provider)
    agent.recommend(post, candidate_target=60, final_k=30)

    assert provider.was_called is True
    assert provider.requested_keywords
    assert all(agent.is_valid_tag(tag, post) for tag in provider.requested_keywords)
    assert all(not any(char.isspace() for char in tag) for tag in provider.requested_keywords)


def test_similar_suffix_variants_are_limited_per_root():
    post = make_post("옷장 정리 방법", "옷장 정리", "옷장 정리 팁", "옷장 정리 방법과 팁을 소개합니다.")
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=80, final_k=30)
    tags = [item.tag for item in result.tags]

    variants = [tag for tag in tags if tag.startswith("옷장정리")]
    assert len(variants) <= 3


def test_debug_trace_exposes_tag_pipeline_stages(monkeypatch, capsys):
    monkeypatch.setenv("DEBUG", "true")
    post = make_post(
        "빈티지 체크셔츠 인턴룩",
        "빈티지 셔츠, 인턴룩, 지속 가능한 패션",
        "빈티지 체크셔츠로 출근 코디를 소개합니다.",
        "오슬로에서 산 빈티지 체크셔츠로 인턴 출근 코디를 해봤습니다.",
    )

    TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=30, final_k=10)
    diagnostic = capsys.readouterr().err

    for stage in (
        "[Tag Debug]",
        "seed_keywords:",
        "generated_candidates:",
        "normalized_candidates:",
        "deduplicated_candidates:",
        "metrics_input:",
        "ranked_candidates:",
        "final_tags:",
    ):
        assert stage in diagnostic


def test_candidate_generation_preserves_phrases_without_keyword_concatenation():
    post = make_post(
        "빈티지 체크셔츠로 완벽한 인턴룩 만들기: 새 옷 없이 코디하기",
        "빈티지 셔츠, 인턴룩, 지속 가능한 패션",
        "빈티지 체크셔츠로 출근 코디를 소개합니다.",
        "오슬로에서 산 빈티지 체크셔츠로 인턴 출근 코디를 해봤습니다.",
    )
    candidates = TagAgent(MockKeywordDataProvider()).generate_candidates(post, target=80)

    assert "빈티지셔츠인턴룩지속가능한패션" not in candidates
    assert "빈티지셔츠" in candidates
    assert "인턴룩" in candidates
    assert "인턴출근룩" in candidates
    assert "체크셔츠코디" in candidates
    assert "지속" not in candidates
    assert "가능한" not in candidates
    assert "가능한패션" not in candidates


def test_comparison_and_trend_phrases_are_source_shaped():
    comparison = make_post(
        "빈티지와 구제의 차이",
        "빈티지, 구제, 재활용",
        "빈티지와 구제의 의미 차이와 의류 재활용",
        "빈티지와 구제의 의미 차이를 설명하고 의류 재활용을 연결합니다.",
    )
    comparison_candidates = TagAgent(MockKeywordDataProvider()).generate_candidates(comparison, target=80)
    assert {"빈티지", "구제", "빈티지구제차이"}.issubset(comparison_candidates)
    assert "빈티지구제재활용" not in comparison_candidates

    trend = make_post(
        "Y2K 패션이 다시 돌아온 지금",
        "Y2K 패션, 빈티지 쇼핑, 옷 재활용",
        "Y2K 스타일을 빈티지 쇼핑과 옷 재활용으로 즐기는 이야기",
        "Y2K 유행이 돌아와도 빈티지 쇼핑과 옷 재활용을 활용할 수 있습니다.",
    )
    trend_candidates = TagAgent(MockKeywordDataProvider()).generate_candidates(trend, target=80)
    assert {"y2k패션", "빈티지쇼핑", "옷재활용"}.issubset(trend_candidates)
    assert "y2k패션빈티지쇼핑옷재활용" not in trend_candidates


def test_suffix_expansion_requires_explicit_how_to_intent():
    comparison = make_post(
        "빈티지와 구제의 차이",
        "빈티지, 구제, 재활용",
        "두 개념의 차이를 설명합니다.",
        "빈티지와 구제의 차이와 공통점을 설명합니다.",
    )
    candidates = TagAgent(MockKeywordDataProvider()).generate_candidates(comparison, target=80)
    assert "재활용방법" not in candidates

    guide = make_post(
        "체크셔츠 코디 방법",
        "체크셔츠 코디",
        "체크셔츠를 활용하는 방법과 팁",
        "체크셔츠 코디 방법을 정리합니다.",
    )
    guide_candidates = TagAgent(MockKeywordDataProvider()).generate_candidates(guide, target=80)
    assert "체크셔츠코디방법" in guide_candidates


def test_raw_source_is_primary_for_core_tag_phrases_with_noisy_metadata():
    source = (
        "인턴일기 ① : 오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다. "
        "인턴 출근룩으로 직접 입어본 이야기와 체크셔츠 코디를 소개하고, "
        "새 옷을 사는 대신 빈티지 옷을 활용하는 이야기를 자연스럽게 풀어보고 싶다."
    )
    post = make_post(
        "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다. 인턴 출근룩으로를 고를 때 확인한 기준",
        "오슬로에서 산 빈티지 체크셔츠로",
        "인턴 출근룩 기준과 안 입는 옷 정리 방법",
        "빈티지 체크셔츠와 안 입는 옷 정리 기준을 설명합니다.",
    )
    agent = TagAgent(MockKeywordDataProvider())
    candidates = agent.generate_candidates(post, target=100, source_input=source)

    assert {"빈티지체크셔츠", "인턴출근룩", "체크셔츠코디"}.issubset(candidates)
    assert "뒤재사용" not in candidates
    assert "이야기와체크셔츠" not in candidates
    assert "기준" not in candidates
    assert "안입는옷코디" not in candidates
    assert "안입는옷기준" not in candidates


def test_raw_source_core_phrases_reach_metrics_input():
    source = "빈티지와 구제는 비슷하게 쓰이지만 정확히 어떤 차이가 있는지 설명하고, 옷을 다시 사용하는 재활용으로 연결하고 싶다."
    post = make_post(
        "오염된 metadata title",
        "무관한 keyword",
        "가이드 기준 방법",
        "빈티지와 구제를 설명합니다.",
    )
    provider = MockKeywordDataProvider()
    TagAgent(provider).recommend(post, candidate_target=100, final_k=20, source_input=source)

    assert {"빈티지", "구제", "빈티지구제차이"}.issubset(provider.requested_keywords)
    assert any(tag in provider.requested_keywords for tag in ("옷재활용", "의류재활용"))


def test_raw_source_does_not_trigger_unstated_unused_clothing_expansion():
    source = "오슬로에서 산 빈티지 체크셔츠로 출근 코디를 해봤다."
    post = make_post("체크셔츠 코디", "체크셔츠", "안 입는 옷 정리 기준", "안 입는 옷 코디와 정리 기준")
    candidates = TagAgent(MockKeywordDataProvider()).generate_candidates(post, target=100, source_input=source)

    assert "안입는옷코디" not in candidates
    assert "안입는옷기준" not in candidates


def test_raw_source_can_generate_unused_clothing_phrases_when_stated():
    source = "옷장에 안 입는 옷이 너무 많아서 정리하려고 한다. 안 입는 옷을 버리지 않고 재사용하거나 수거하는 방법을 소개하고 싶다."
    post = make_post("오염된 title", "무관한 keyword", "무관한 summary", "본문")
    candidates = TagAgent(MockKeywordDataProvider()).generate_candidates(post, target=100, source_input=source)

    assert "안입는옷" in candidates
    assert "안입는옷정리" in candidates or "옷정리" in candidates
    assert "옷재사용" in candidates


def test_post_tags_are_space_free_and_unique():
    post = make_post("가을 옷장 정리", "가을 옷장 정리", "가을 옷장 정리", "가을 옷장을 정리합니다.")
    TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=60, final_k=30)

    assert post.tags is not None
    assert len(post.tags) == len(set(post.tags))
    assert all(tag and not any(char.isspace() for char in tag) for tag in post.tags)


def test_primary_season_beats_secondary_season_mentions():
    post = make_post(
        "가을 옷장 정리와 스타일링",
        "가을 옷장 정리",
        "가을 옷을 중심으로 계절 옷을 정리하고 활용하는 방법",
        "가을 옷장을 정리하면서 여름 옷은 보조 사례로만 언급합니다. 여름 옷장, 여름 코디, 여름 스타일링은 핵심 주제가 아닙니다.",
    )
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=80, final_k=30)
    tags = [item.tag for item in result.tags]

    autumn_count = sum("가을" in tag for tag in tags)
    summer_count = sum("여름" in tag for tag in tags)
    assert autumn_count > 0
    assert summer_count < autumn_count


def test_korean_topic_does_not_fill_results_with_english_single_words():
    post = make_post(
        "가을 옷장 정리",
        "autumn wardrobe organization",
        "가을 옷장을 정리하는 방법",
        "가을 옷장을 계절별로 정리하는 내용입니다.",
    )
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=80, final_k=30)
    tags = {item.tag for item in result.tags}

    assert "autumn" not in tags
    assert "wardrobe" not in tags
    assert "organization" not in tags
    assert any("가을" in tag for tag in tags)


def test_repeated_body_concepts_create_search_shaped_tags():
    post = make_post(
        "가을 옷장 정리",
        "가을 옷장 정리",
        "계절 옷을 분류하고 수납하는 방법",
        "한 번도 입지 않은 옷은 따로 분류하고, 계절 옷은 수납합니다."
        " 매달 옷장 정리 루틴을 점검하면 안 입는 옷을 쉽게 찾을 수 있습니다.",
    )
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=80, final_k=30)
    tags = {item.tag for item in result.tags}

    assert {"안입는옷정리", "계절옷분류", "옷장수납", "옷장정리루틴"}.intersection(tags)


def test_high_search_volume_does_not_overrule_relevance_and_longtail_quality():
    post = make_post(
        "가을 옷장 정리 방법",
        "가을 옷장 정리",
        "계절 옷을 분류하고 수납하는 방법",
        "가을 옷장을 정리하고 계절 옷을 수납하는 내용입니다.",
    )
    provider = FixedKeywordDataProvider({
        "가을옷장": metric(100000, 0.95, 10000, 0.95),
        "가을옷장정리방법": metric(100, 0.10, 20, 0.10),
    })
    result = TagAgent(provider).recommend(post, candidate_target=80, final_k=20)
    scores = {item.tag: item.ranking_score for item in result.tags}

    assert "가을옷장정리방법" in scores
    assert scores.get("가을옷장정리방법", 0) > scores.get("가을옷장", 1)


def test_competition_penalty_is_applied():
    post = make_post("가을 옷장 정리", "가을 옷장 정리", "가을 옷장 정리 방법", "가을 옷장 정리")
    provider = FixedKeywordDataProvider({
        "가을옷장정리방법": metric(1000, 0.95, 100, 0.20),
        "가을옷장정리팁": metric(1000, 0.05, 100, 0.20),
    })
    result = TagAgent(provider).recommend(post, candidate_target=60, final_k=20)
    scores = {item.tag: item.ranking_score for item in result.tags}

    assert scores["가을옷장정리팁"] > scores["가을옷장정리방법"]


def test_saturation_penalty_is_applied():
    post = make_post("가을 옷장 정리", "가을 옷장 정리", "가을 옷장 정리 방법", "가을 옷장 정리")
    provider = FixedKeywordDataProvider({
        "가을옷장정리방법": metric(1000, 0.20, 100, 0.95),
        "가을옷장정리팁": metric(1000, 0.20, 100, 0.05),
    })
    result = TagAgent(provider).recommend(post, candidate_target=60, final_k=20)
    scores = {item.tag: item.ranking_score for item in result.tags}

    assert scores["가을옷장정리팁"] > scores["가을옷장정리방법"]


def test_final_selection_preserves_semantic_group_diversity():
    post = make_post(
        "가을 옷장 정리 방법",
        "가을 옷장 정리",
        "계절 옷을 분류하고 수납하는 방법과 옷장 정리 루틴",
        "가을 옷을 분류하고 수납하며 옷장 정리 루틴을 점검합니다.",
    )
    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=100, final_k=30)
    group_counts: dict[str, int] = {}
    agent = TagAgent(MockKeywordDataProvider())
    for item in result.tags:
        group = agent._semantic_group(item.tag)
        group_counts[group] = group_counts.get(group, 0) + 1

    assert len(group_counts) >= 2
    assert max(group_counts.values()) <= 8


def test_keyword_metrics_model_and_provider_contract_are_provider_neutral():
    post = make_post("가을 옷장 정리", "가을 옷장 정리", "가을 옷장 정리", "가을 옷장을 정리합니다.")
    provider = FixedKeywordDataProvider({"가을옷장정리": metric(123, 0.2, 45, 0.1)})
    result = TagAgent(provider).recommend(post, candidate_target=40, final_k=10)

    assert provider.requested_keywords
    assert all(isinstance(keyword, str) for keyword in provider.requested_keywords)
    assert all(item.search_volume is not None for item in result.tags if item.tag == "가을옷장정리")


def test_sustainable_consumption_tags_follow_post_topic_not_incidental_clothing_terms():
    post = make_post(
        "지속 가능한 의류 소비를 위한 실용적인 체크리스트",
        "지속 가능한 소비 습관",
        "구매 전 소재와 사용 계획을 확인하고 충동 구매를 줄이는 방법",
        """### 1. 필요성 고려하기
구매 전 실제로 필요한지 확인해 충동 구매를 줄입니다.

### 2. 소재 확인하기
소재와 재활용 소재를 확인합니다.

### 3. 브랜드 기준 살펴보기
브랜드의 윤리적 기준을 살펴봅니다.

### 4. 사용 용도 생각하기
오래 입을 수 있는지와 사용 용도를 생각합니다.

### 5. 구매 후 관리 계획 세우기
세탁과 보관으로 의류를 오래 관리합니다.""",
    )

    result = TagAgent(MockKeywordDataProvider()).recommend(post, candidate_target=80, final_k=30)
    tags = [item.tag for item in result.tags]

    assert any("지속가능한소비" in tag or "소비" in tag for tag in tags)
    assert not any(tag.startswith("안입는옷") for tag in tags)
