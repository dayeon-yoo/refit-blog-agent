from __future__ import annotations

from typing import List

from agents.brand_agent import BrandAgent
from agents.critic_agent import CriticAgent
from agents.idea_agent import IdeaGenerator
from agents.seo_agent import SEOAgent
from agents.trend_agent import TrendAgent
from agents.writer_agent import WriterAgent
from models.schemas import BlogIdea, BlogPost, ScoredIdea, WorkflowResult


def run_workflow() -> WorkflowResult:
    trend_results = TrendAgent().run()
    seo_results = SEOAgent().run(trend_results)

    idea_candidates = IdeaGenerator().generate(trend_results, seo_results, [])
    brand_evaluations = BrandAgent().run([idea.title for idea in idea_candidates])

    scored = CriticAgent().top_3(idea_candidates)
    writer = WriterAgent()
    blog_posts = [writer.write(idea) for idea in scored]

    return WorkflowResult(
        ideas=idea_candidates,
        top_3=scored,
        blog_posts=blog_posts,
    )
