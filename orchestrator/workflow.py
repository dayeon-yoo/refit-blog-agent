from __future__ import annotations

from agents.brand_agent import BrandAgent
from agents.critic_agent import CriticAgent
from agents.idea_agent import IdeaGenerator
from agents.seo_agent import SEOAgent
from agents.trend_agent import TrendAgent
from agents.writer_agent import WriterAgent
from models.schemas import WorkflowResult


def run_workflow() -> WorkflowResult:
    trend_results = TrendAgent().run()
    seo_results = SEOAgent().run(trend_results)

    if not trend_results:
        trend_results = TrendAgent().run()
    if not seo_results:
        seo_results = SEOAgent().run(trend_results)

    idea_candidates = IdeaGenerator().generate(trend_results, seo_results)
    if not idea_candidates:
        idea_candidates = IdeaGenerator().generate(trend_results, seo_results)

    brand_evaluations = BrandAgent().run(idea_candidates)
    if not brand_evaluations:
        brand_evaluations = BrandAgent().run(idea_candidates)

    filtered_ideas = []
    filtered_brand_evaluations = []
    for idea, evaluation in zip(idea_candidates, brand_evaluations):
        if evaluation.fit_score >= 0.6:
            filtered_ideas.append(idea)
            filtered_brand_evaluations.append(evaluation)

    if len(filtered_ideas) < 3:
        filtered_ideas = idea_candidates[: min(3, len(idea_candidates)) or len(idea_candidates)]
        filtered_brand_evaluations = brand_evaluations[: len(filtered_ideas)]

    scored = CriticAgent().top_3(filtered_ideas, filtered_brand_evaluations)
    top_3 = scored[:3]

    if not top_3 and idea_candidates:
        top_3 = CriticAgent().top_3(idea_candidates[:3], brand_evaluations[:3])

    writer = WriterAgent()
    blog_posts = [writer.write(idea) for idea in top_3]

    return WorkflowResult(
        ideas=idea_candidates,
        top_3=top_3,
        blog_posts=blog_posts,
    )
