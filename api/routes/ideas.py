from fastapi import APIRouter, Depends, Response

from api.dependencies import get_blog_service
from api.schemas import ExpandIdeasRequest, RefineRequest
from models.schemas import BlogIdea, IdeaExpansionResult
from services.blog_workflow_service import BlogWorkflowService

router = APIRouter(prefix="/api/ideas", tags=["ideas"])


@router.post("/expand", response_model=IdeaExpansionResult)
def expand(request: ExpandIdeasRequest, response: Response, service: BlogWorkflowService = Depends(get_blog_service)):
    if request.use_live:
        result = service.expand(request.source, use_live=True)
        response.headers["X-RIFIT-LLM-Mode"] = "live"
        return result
    return service.expand(request.source)


@router.post("/refine", response_model=BlogIdea)
def refine(request: RefineRequest, service: BlogWorkflowService = Depends(get_blog_service)):
    return service.refine(request.source, request.candidate, request.revision_request)
