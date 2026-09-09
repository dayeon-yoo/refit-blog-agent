import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import ideas, workflow
from config.settings import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="RIFIT Blog Agent", debug=False)
    local = get_settings().environment in {"local", "development", "dev"}
    origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173" if local else "")
    allowed = [origin.strip() for origin in origins.split(",") if origin.strip()]
    if "*" in allowed:
        raise ValueError("CORS_ALLOW_ORIGINS must list explicit origins")
    app.add_middleware(
        CORSMiddleware, allow_origins=allowed,
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
        expose_headers=["X-RIFIT-LLM-Mode"],
    )

    @app.middleware("http")
    async def generation_mode(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-RIFIT-LLM-Mode", "mock" if get_settings().mock_mode else "live")
        return response

    @app.exception_handler(ValueError)
    async def validation_failure(request: Request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(Exception)
    async def internal_failure(request: Request, error: Exception):
        logger.exception("Blog API request failed", exc_info=error)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    app.include_router(ideas.router)
    app.include_router(workflow.router)
    return app


app = create_app()
