from agents.brand_agent import BrandAgent
from agents.critic_agent import CriticAgent
from agents.idea_agent import IdeaGenerator
from agents.seo_agent import SEOAgent
from agents.trend_agent import TrendAgent
from agents.writer_agent import WriterAgent
from orchestrator.workflow import run_workflow


def test_mock_workflow_data_flow_is_connected():
    trend_results = TrendAgent().run()
    seo_results = SEOAgent().run(trend_results)
    ideas = IdeaGenerator().generate(trend_results, seo_results)
    brand_evaluations = BrandAgent().run(ideas)
    top_3 = CriticAgent().top_3(ideas, brand_evaluations)
    blog_posts = [WriterAgent().write(item) for item in top_3]

    assert len(trend_results) > 0
    assert len(seo_results) > 0
    assert len(ideas) >= 10
    assert len(brand_evaluations) == len(ideas)
    assert len(top_3) == 3
    assert len(blog_posts) == 3

    seo_keywords = {item.keyword for item in seo_results}
    for idea in ideas:
        assert idea.keyword
        assert idea.search_intent
        assert idea.title
        assert idea.keyword in seo_keywords or any(idea.keyword.startswith(item.topic) for item in trend_results)

    for idea, evaluation in zip(ideas, brand_evaluations):
        assert evaluation.topic == idea.title
        assert 0.0 <= evaluation.fit_score <= 1.0

    for item in top_3:
        assert item.total_score <= 100
        assert item.idea.title in {idea.title for idea in ideas}

    for post, item in zip(blog_posts, top_3):
        assert post.title == item.idea.title
        assert post.keyword == item.idea.keyword
        assert post.content
        assert item.reason in post.summary
        assert item.idea.rifit_connection in post.content


def test_workflow_result_contains_connected_data():
    result = run_workflow()

    assert "ideas" in result
    assert "top_3" in result
    assert "blog_posts" in result
    assert len(result["ideas"]) >= 10
    assert len(result["top_3"]) == 3
    assert len(result["blog_posts"]) == 3

    idea_titles = {item.title for item in result["ideas"]}
    top3_titles = {item.idea.title for item in result["top_3"]}
    assert top3_titles.issubset(idea_titles)

    for post in result["blog_posts"]:
        assert post.title
        assert post.keyword
        assert post.content
        assert post.summary
        assert post.cta
