from __future__ import annotations

from typing import Optional

from agents.content_planner_agent import ContentPlannerAgent
from agents.idea_agent import IdeaGenerator
from config.settings import get_settings
from llm.client import OpenAILLMClient
from models.schemas import (
    BlogIdea, BlogPost, ContentPlan, IdeaCandidate, IdeaExpansionResult,
    ImagePlan, QCResult, RIFITBaseModel,
)
from orchestrator.workflow import run_manual_workflow


class GeneratedBlog(RIFITBaseModel):
    blog_idea: BlogIdea
    content_plan: ContentPlan
    post: BlogPost
    tags: list[str]
    image_plan: Optional[ImagePlan]
    qc: QCResult


class BlogWorkflowService:
    """Stateless application operations using the existing agent workflows."""

    def expand(self, source: str, use_live: bool = False) -> IdeaExpansionResult:
        if use_live:
            settings = get_settings()
            client = OpenAILLMClient(api_key=settings.openai_api_key, model=settings.model_name)
            # Explicit paid requests should not silently retry network failures.
            client.client = client.client.with_options(max_retries=0)
            return IdeaGenerator(llm_client=client).expand(source)
        return IdeaGenerator().expand(source)

    def refine(
        self, source: str, candidate: IdeaCandidate, revision_request: str = ""
    ) -> BlogIdea:
        return IdeaGenerator().refine(source, candidate, revision_request)

    def generate(self, source: str, blog_idea: BlogIdea) -> GeneratedBlog:
        plan = ContentPlannerAgent().plan(source, blog_idea)
        result = run_manual_workflow(blog_idea, source=source, content_plan=plan)
        post = result.blog_posts[0]
        return GeneratedBlog(
            blog_idea=blog_idea, content_plan=plan, post=post,
            tags=post.tags or [], image_plan=post.image_plan, qc=result.qc_results[0],
        )
