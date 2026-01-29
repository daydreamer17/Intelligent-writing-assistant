from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hello_agents.core.exceptions import HelloAgentsException
from fastapi.exceptions import RequestValidationError

try:
    from .deps import AppServices, get_services
    from .routes import citations_router, pipeline_router, rag_router, versions_router, writing_router
    from ..agents.base import AgentRuntimeConfig, build_llm
    from ..config import AppConfig
except ImportError:  # Allows running `python main.py` directly.
    import sys
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.api.deps import AppServices, get_services  # noqa: E402
    from app.api.routes import (  # noqa: E402
        citations_router,
        pipeline_router,
        rag_router,
        versions_router,
        writing_router,
    )
    from app.agents.base import AgentRuntimeConfig, build_llm  # noqa: E402
    from app.config import AppConfig  # noqa: E402


def create_app() -> FastAPI:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("app")
    app = FastAPI(title="HelloAgent Writing API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(writing_router, prefix="/api")
    app.include_router(pipeline_router, prefix="/api")
    app.include_router(rag_router, prefix="/api")
    app.include_router(citations_router, prefix="/api")
    app.include_router(versions_router, prefix="/api")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s %s %.2fms", request.method, request.url.path, response.status_code, duration_ms)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        if _is_rate_limit_error(exc):
            logger.warning("Rate limited on %s %s: %s", request.method, request.url.path, exc)
            return JSONResponse(
                status_code=429,
                content={"detail": "触发TPM限流，请稍后再试或降低字数/并发。"},
            )
        if isinstance(exc, HelloAgentsException):
            logger.error("LLM error on %s %s: %s", request.method, request.url.path, exc)
            return JSONResponse(status_code=500, content={"detail": str(exc)})
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/healthz")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz/detail")
    def health_detail(services: AppServices = Depends(get_services)) -> dict[str, object]:
        return {
            "status": "ok",
            "memory_mode": services.memory_mode,
            "storage_path": services.storage_path,
            "qdrant": {
                "enabled": services.qdrant_enabled,
                "available": services.qdrant_available,
                "error": services.qdrant_error,
                "url": services.qdrant_url or "unset",
                "collection": services.qdrant_collection,
                "dim": services.qdrant_dim,
                "distance": services.qdrant_distance,
            },
            "embedding": {
                "provider": services.embedding_provider or "hash",
                "model": services.embedding_model or "unset",
                "dim": services.embedding_dim,
                "mode": services.embedding_mode,
                "available": services.embedding_available,
                "error": services.embedding_error,
            },
        }

    @app.get("/healthz/llm")
    def health_llm() -> dict[str, object]:
        try:
            config = AppConfig.from_env()
            llm = build_llm(
                AgentRuntimeConfig(
                    name="llm-health",
                    system_prompt="",
                    temperature=0.0,
                    provider=config.llm_provider or None,
                    model=config.llm_model or None,
                    api_key=config.llm_api_key or None,
                    base_url=config.llm_api_base or None,
                    request_timeout=config.llm_timeout,
                    max_tokens=config.llm_max_tokens,
                )
            )
            response = llm.invoke([{"role": "user", "content": "ping"}])
            text = response if isinstance(response, str) else str(response)
            return {"status": "ok", "sample": text[:200]}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    @app.on_event("shutdown")
    def shutdown_cleanup() -> None:
        try:
            services = get_services()
            services.storage.close()
        except Exception:
            pass

    return app


app = create_app()


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "rate limit" in message
        or "rate limiting" in message
        or "tpm limit" in message
        or "too many requests" in message
        or "429" in message
    )
