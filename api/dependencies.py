from services.blog_workflow_service import BlogWorkflowService


def get_blog_service() -> BlogWorkflowService:
    return BlogWorkflowService()
