from __future__ import annotations

from agents.tag_agent import MockKeywordDataProvider, TagAgent
from models.schemas import BlogPost, BlogIdea, ScoredIdea, TagRecommendationResult
from agents.writer_agent import WriterAgent
from agents.image_agent import ImageAgent


def make_sample_post() -> BlogPost:
    idea = BlogIdea(
        title="How to grow tomatoes at home",
        keyword="home gardening",
        search_intent="정보형",
        angle="방법",
        rifit_connection="gardening",
        seasonality=0.5,
    )

    scored = ScoredIdea(
        idea_id="id1",
        scores={"score": 80},
        total_score=80,
        reason="good",
        idea=idea,
    )

    writer = WriterAgent()
    post = writer.write(scored)
    return post


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
