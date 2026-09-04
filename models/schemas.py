from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


try:
    BaseModel.model_validate
except AttributeError:  # pragma: no cover
    def _model_validate(cls, obj):
        return cls.parse_obj(obj)

    BaseModel.model_validate = classmethod(_model_validate)


class TrendResult(BaseModel):
    topic: str
    reason: str
    source: str = "trend-monitor"
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class SEOResult(BaseModel):
    keyword: str
    search_intent: str
    seasonality: float = Field(..., ge=0.0, le=1.0)
    content_potential: float = Field(..., ge=0.0, le=1.0)


class BrandEvaluation(BaseModel):
    topic: str
    fit_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    brand_alignment: str


class BlogIdea(BaseModel):
    title: str
    keyword: str
    search_intent: str
    angle: str
    rifit_connection: str
    seasonality: float = Field(..., ge=0.0, le=1.0)


class ScoredIdea(BaseModel):
    idea_id: str
    scores: Dict[str, int]
    total_score: int = Field(..., ge=0, le=100)
    reason: str
    idea: BlogIdea


class BlogPost(BaseModel):
    title: str
    keyword: str
    content: str
    summary: str
    cta: str


class WorkflowResult(BaseModel):
    ideas: List[BlogIdea]
    top_3: List[ScoredIdea]
    blog_posts: List[BlogPost]

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields_set or hasattr(self, key)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)
