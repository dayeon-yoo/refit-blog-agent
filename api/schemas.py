from __future__ import annotations

from pydantic import field_validator

from models.schemas import BlogIdea, IdeaCandidate, RIFITBaseModel


class ExpandRequest(RIFITBaseModel):
    source: str

    @field_validator("source")
    @classmethod
    def require_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source must not be blank")
        return value


class RefineRequest(ExpandRequest):
    candidate: IdeaCandidate
    revision_request: str = ""


class ExpandIdeasRequest(ExpandRequest):
    use_live: bool = False


class GenerateRequest(ExpandRequest):
    blog_idea: BlogIdea
