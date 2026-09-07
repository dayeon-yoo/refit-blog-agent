from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


try:
    BaseModel.model_validate
except AttributeError:  # pragma: no cover
    def _model_validate(cls, obj):
        return cls.parse_obj(obj)

    BaseModel.model_validate = classmethod(_model_validate)


class RIFITBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class TrendResult(RIFITBaseModel):
    topic: str
    reason: str
    source: str = "trend-monitor"
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class SEOResult(RIFITBaseModel):
    keyword: str
    search_intent: str
    seasonality: float = Field(..., ge=0.0, le=1.0)
    content_potential: float = Field(..., ge=0.0, le=1.0)


class BrandEvaluation(RIFITBaseModel):
    topic: str
    fit_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    brand_alignment: str


class BlogIdea(RIFITBaseModel):
    title: str
    keyword: str
    search_intent: str
    angle: str
    rifit_connection: str
    seasonality: float = Field(..., ge=0.0, le=1.0)


class ScoredIdea(RIFITBaseModel):
    idea_id: str
    scores: Dict[str, int]
    total_score: int = Field(..., ge=0, le=100)
    reason: str
    idea: BlogIdea
    brand_evaluation: Optional[BrandEvaluation] = None


class ImagePrompt(RIFITBaseModel):
    placement: str
    purpose: str
    prompt: str
    image_type: str
    alt_text: str


class ImagePlan(RIFITBaseModel):
    images: List[ImagePrompt]


class BlogPost(RIFITBaseModel):
    title: str
    keyword: str
    content: str
    summary: str
    cta: str
    image_plan: Optional[ImagePlan] = None
    # Final tags for publishing: list of tag strings (max 30 expected)
    tags: Optional[List[str]] = None


class WorkflowResult(RIFITBaseModel):
    ideas: List[BlogIdea]
    top_3: List[ScoredIdea]
    blog_posts: List[BlogPost]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return output_path

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields_set or hasattr(self, key)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class TagRecommendation(RIFITBaseModel):
    tag: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    # Mockable/provider metrics (may be None if provider doesn't supply them)
    search_volume: Optional[int] = None
    competition: Optional[float] = Field(None, ge=0.0, le=1.0)
    publishing_volume: Optional[int] = None
    saturation: Optional[float] = Field(None, ge=0.0, le=1.0)
    ranking_score: float = Field(..., ge=0.0, le=1.0)


class TagRecommendationResult(RIFITBaseModel):
    tags: List[TagRecommendation]
