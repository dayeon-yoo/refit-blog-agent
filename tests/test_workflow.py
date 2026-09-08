from agents.brand_agent import BrandAgent
from agents.critic_agent import CriticAgent
from agents.idea_agent import IdeaGenerator
from agents.image_agent import ImageAgent
from agents.seo_agent import SEOAgent
from agents.trend_agent import TrendAgent
from agents.writer_agent import WriterAgent
from config.settings import get_settings
from llm.client import MockLLMClient
from models.schemas import BlogIdea
from orchestrator.workflow import run_manual_workflow, run_workflow
from agents.tag_agent import MockKeywordDataProvider, TagAgent


def test_workflow_result_can_be_serialized_to_json():
    settings = get_settings()
    result = run_workflow()
    payload = result.model_dump(mode="json")
    assert isinstance(payload, dict)
    assert len(payload["ideas"]) == settings.idea_count
    assert len(payload["top_3"]) == settings.top_k
    assert len(payload["blog_posts"]) == settings.top_k
    assert result.to_json()


def test_mock_workflow_data_flow_is_connected():
    settings = get_settings()
    trend_results = TrendAgent().run()
    seo_results = SEOAgent().run(trend_results)
    ideas = IdeaGenerator().generate(trend_results, seo_results)[: settings.idea_count]
    brand_evaluations = BrandAgent().run(ideas)
    top_3 = CriticAgent().top_3(ideas, brand_evaluations)[: settings.top_k]
    image_agent = ImageAgent()
    blog_posts = [image_agent.plan(WriterAgent().write(item)) for item in top_3]

    assert len(trend_results) > 0
    assert len(seo_results) > 0
    assert len(ideas) == settings.idea_count
    assert len(brand_evaluations) == len(ideas)
    assert len(top_3) == settings.top_k
    assert len(blog_posts) == settings.top_k

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
        assert post.summary
        assert post.cta
        assert post.image_plan is not None
        assert post.image_plan.images

        for image in post.image_plan.images:
            assert image.placement
            assert image.purpose
            assert image.prompt
            assert image.image_type
            assert image.alt_text
            assert "검색 의도" not in image.prompt
            assert "브랜드 평가" not in image.prompt
            assert "critic" not in image.prompt.lower()
            assert "fit_score" not in image.prompt.lower()

        for field in ["검색 의도", "브랜드 평가", "평가 점수", "현재 적합도"]:
            assert field not in post.content
            assert field not in post.summary

        assert str(item.reason) not in post.content
        assert str(item.reason) not in post.summary
        assert str(item.total_score) not in post.content
        assert str(item.total_score) not in post.summary

        if item.brand_evaluation is not None:
            assert item.brand_evaluation.rationale not in post.content
            assert item.brand_evaluation.rationale not in post.summary
            assert str(item.brand_evaluation.fit_score) not in post.content
            assert str(item.brand_evaluation.fit_score) not in post.summary

        assert "fit_score" not in post.content
        assert "fit_score" not in post.summary
        assert "critic" not in post.content.lower()
        assert "critic" not in post.summary.lower()


def test_workflow_result_contains_connected_data():
    settings = get_settings()
    result = run_workflow()

    assert "ideas" in result
    assert "top_3" in result
    assert "blog_posts" in result
    assert len(result["ideas"]) == settings.idea_count
    assert len(result["top_3"]) == settings.top_k
    assert len(result["blog_posts"]) == settings.top_k

    idea_titles = {item.title for item in result["ideas"]}
    top3_titles = {item.idea.title for item in result["top_3"]}
    assert top3_titles.issubset(idea_titles)

    for post in result["blog_posts"]:
        assert post.title
        assert post.keyword
        assert post.content
        assert post.summary
        assert post.cta
        assert post.image_plan is not None
        assert len(post.image_plan.images) >= 3
        for image in post.image_plan.images:
            assert image.placement
            assert image.purpose
            assert image.prompt
            assert image.image_type
            assert image.alt_text


def test_manual_blog_idea_runs_writer_tag_image_pipeline_without_idea_agent():
    idea = BlogIdea(
        title="옷장에 안 입는 옷이 생기는 이유",
        keyword="안 입는 옷 정리",
        search_intent="정보 탐색",
        angle="안 입게 되는 옷의 특징을 살펴보고 재활용, 기부, 판매까지 연결",
        summary="옷장에 오래 남아 있는 옷을 점검하고 새로운 활용 방법을 소개한다.",
        rifit_connection="입지 않는 옷을 버리는 대신 다시 순환시킬 수 있는 방법을 안내한다.",
        seasonality=0.5,
    )
    client = MockLLMClient()
    provider = MockKeywordDataProvider()
    result = run_manual_workflow(
        idea,
        writer=WriterAgent(llm_client=client),
        tag_agent=TagAgent(provider=provider),
    )

    assert len(client.calls) == 1
    assert client.calls[0]["payload"]["content_contract"]["title"] == idea.title
    assert result.ideas == [idea]
    assert len(result.top_3) == 1
    assert len(result.blog_posts) == 1
    post = result.blog_posts[0]
    assert post.title == idea.title
    assert post.keyword == idea.keyword
    assert post.content
    assert post.tags
    assert post.image_plan is not None
    assert provider.was_called
    assert all(image.placement for image in post.image_plan.images)
