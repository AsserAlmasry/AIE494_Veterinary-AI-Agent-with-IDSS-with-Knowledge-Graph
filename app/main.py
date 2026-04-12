"""
app/main.py
===========
FastAPI application entry-point.
Registers all routers, middleware, lifecycle hooks, and health endpoints.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from api.routes import predict, identity, report

# ── Logging ─────────────────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("veterinary_ai")


# ── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm-up all singleton dependencies on startup so the first request is fast.
    """
    logger.info("🚀 Starting Veterinary AI System …")
    try:
        from app.dependencies import (
            get_identity_engine,
            get_identity_bank,
            get_disease_model,
            get_risk_model,
            get_rag_service,
            get_llm_service,
            get_safety_engine,
        )
        logger.info("  Loading identity engine …")
        get_identity_engine()
        logger.info("  Loading identity bank …")
        get_identity_bank()
        logger.info("  Loading disease model …")
        get_disease_model()
        logger.info("  Loading risk model …")
        get_risk_model()
        logger.info("  Loading RAG service …")
        get_rag_service()
        logger.info("  Loading LLM service …")
        get_llm_service()
        logger.info("  Loading safety engine …")
        get_safety_engine()
        logger.info("✅ All services ready.")
    except Exception as exc:
        logger.error(f"❌ Startup warm-up failed: {exc}", exc_info=True)

    yield  # ── application is now running ──────────────────────────────

    logger.info("🛑 Shutting down Veterinary AI System …")
    try:
        from app.dependencies import get_neo4j_service
        get_neo4j_service().close()
    except Exception:
        pass


# ── Application factory ─────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Veterinary AI Agent + IDSS",
        description=(
            "Production-grade AI backend for cattle health monitoring. "
            "Integrates computer vision, disease classification, time-series "
            "risk prediction, RAG over PubMed, and knowledge graph reasoning."
        ),
        version="1.0.0",
        docs_url=f"/api/{settings.api_version}/docs",
        redoc_url=f"/api/{settings.api_version}/redoc",
        openapi_url=f"/api/{settings.api_version}/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        return response

    # ── Global exception handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc),
                "path": str(request.url),
            },
        )

    # ── Health endpoints ─────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "version": "1.0.0",
            "environment": settings.app_env,
        }

    @app.get("/", tags=["System"])
    async def root() -> Dict[str, str]:
        return {
            "service": "Veterinary AI Agent + IDSS",
            "docs": f"/api/{settings.api_version}/docs",
        }

    # ── Routers ─────────────────────────────────────────────────────────────
    prefix = f"/api/{settings.api_version}"
    app.include_router(predict.router, prefix=prefix, tags=["Prediction"])
    app.include_router(identity.router, prefix=prefix, tags=["Identity"])
    app.include_router(report.router,  prefix=prefix, tags=["Report"])

    return app


app = create_app()


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.app_env == "development"),
        log_level=settings.log_level.lower(),
    )
