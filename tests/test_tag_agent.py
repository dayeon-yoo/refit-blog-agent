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
