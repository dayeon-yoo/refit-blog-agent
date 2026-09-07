from __future__ import annotations

from agents.brand_agent import BrandAgent
from agents.critic_agent import CriticAgent
from agents.idea_agent import IdeaGenerator
from agents.image_agent import ImageAgent
from agents.seo_agent import SEOAgent
from agents.trend_agent import TrendAgent
from agents.writer_agent import WriterAgent
from agents.tag_agent import TagAgent, MockKeywordDataProvider
from config.settings import get_settings
from models.schemas import WorkflowResult
from orchestrator.dedupe import filter_duplicates


def run_workflow() -> WorkflowResult:
    settings = get_settings()
    idea_limit = max(1, settings.idea_count)
    top_k = max(1, settings.top_k)

    trend_results = TrendAgent().run()
    seo_results = SEOAgent().run(trend_results)

    if not trend_results:
        trend_results = TrendAgent().run()
    if not seo_results:
        seo_results = SEOAgent().run(trend_results)

    idea_candidates = IdeaGenerator().generate(trend_results, seo_results)[:idea_limit]
    if not idea_candidates:
        idea_candidates = IdeaGenerator().generate(trend_results, seo_results)[:idea_limit]

    # Duplicate Check: filter out ideas that are substantively similar to past outputs
    deduped_candidates = filter_duplicates(idea_candidates)
    # If all ideas removed by dedupe, fall back to original candidates to avoid empty pipeline
    if not deduped_candidates:
        deduped_candidates = idea_candidates[: max(1, min(top_k, len(idea_candidates)))]
    idea_candidates = deduped_candidates

    brand_evaluations = BrandAgent().run(idea_candidates)
    if not brand_evaluations:
        brand_evaluations = BrandAgent().run(idea_candidates)

    filtered_ideas = []
    filtered_brand_evaluations = []
    for idea, evaluation in zip(idea_candidates, brand_evaluations):
        if evaluation.fit_score >= 0.6:
            filtered_ideas.append(idea)
            filtered_brand_evaluations.append(evaluation)

    if len(filtered_ideas) < top_k:
        filtered_ideas = idea_candidates[: min(top_k, len(idea_candidates)) or len(idea_candidates)]
        filtered_brand_evaluations = brand_evaluations[: len(filtered_ideas)]

    scored = CriticAgent().top_3(filtered_ideas, filtered_brand_evaluations)
    top_3 = scored[:top_k]

    if not top_3 and idea_candidates:
        top_3 = CriticAgent().top_3(idea_candidates[:top_k], brand_evaluations[:top_k])

    writer = WriterAgent()
    image_agent = ImageAgent()

    # Tag recommendation: use Mock provider when running in mock_mode
    settings = get_settings()
    provider = MockKeywordDataProvider()
    tag_agent = TagAgent(provider=provider)

    blog_posts = []
    for idea in top_3:
        post = writer.write(idea)
        # recommend tags (adds `tags` list to BlogPost)
        _ = tag_agent.recommend(post)
        # then plan images
        planned = image_agent.plan(post)
        blog_posts.append(planned)

    return WorkflowResult(
        ideas=idea_candidates,
        top_3=top_3,
        blog_posts=blog_posts,
    )
