from fastapi import APIRouter, Depends

from api.dependencies import get_blog_service
from api.schemas import GenerateRequest
from services.blog_workflow_service import BlogWorkflowService, GeneratedBlog

router = APIRouter(prefix="/api/workflow", tags=["workflow"])


@router.post("/generate", response_model=GeneratedBlog)
def generate(request: GenerateRequest, service: BlogWorkflowService = Depends(get_blog_service)):
    return service.generate(request.source, request.blog_idea)
