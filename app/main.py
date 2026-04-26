"""
app/main.py — FastAPI application entry-point (v4.0.0 MMCOWS Integration)
Registers all routers, middleware, lifecycle hooks, and health endpoints.
"""
from __future__ import annotations
import logging, time
from contextlib import asynccontextmanager
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_settings
from api.routes import predict, identity, report
from api.routes import agent

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("veterinary_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up all singleton dependencies on startup."""
    logger.info("🚀 Starting Veterinary AI System v4.0.0 (MMCOWS) …")
    try:
        from app.dependencies import (
            get_cow_identifier, get_milk_predictor, get_heat_stress_analyzer,
            get_health_scorer, get_data_pipeline,
            get_rag_service, get_llm_service, get_safety_engine,
            get_neo4j_service, get_pipeline,
        )
        logger.info("  Loading MMCOWS cow identifier (CowReIDModel) …")
        get_cow_identifier()
        logger.info("  Loading milk predictor (TimeSeriesTransformer) …")
        get_milk_predictor()
        logger.info("  Loading heat stress analyzer (BehaviorCNNLSTM + THI) …")
        get_heat_stress_analyzer()
        logger.info("  Loading health scorer (MultiModalFusion + Autoencoder) …")
        get_health_scorer()
        logger.info("  Loading MMCOWS data pipeline …")
        get_data_pipeline()
        logger.info("  Loading RAG service …")
        try:
            get_rag_service()
        except Exception as e:
            logger.warning(f"  RAG service unavailable (non-fatal): {e}")
        logger.info("  Loading LLM service (Groq) …")
        try:
            get_llm_service()
        except Exception as e:
            logger.warning(f"  LLM service unavailable (non-fatal): {e}")
        logger.info("  Loading safety engine …")
        get_safety_engine()
        logger.info("  Loading Neo4j service …")
        neo4j = get_neo4j_service()
        logger.info("  Initializing clinical knowledge graph …")
        try:
            neo4j.initialize_clinical_schema()
        except Exception as e:
            logger.warning(f"  KG init warning: {e}")
        logger.info("  Building unified pipeline …")
        get_pipeline()
        logger.info("✅ All systems operational — MMCOWS models loaded")
    except Exception as exc:
        logger.error(f"⚠️ Startup warning: {exc}", exc_info=True)

    yield
    logger.info("🛑 Shutting down Veterinary AI System")


app = FastAPI(
    title="Veterinary AI IDSS",
    description="Production veterinary clinical decision support system powered by MMCOWS multimodal models",
    version="4.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ────────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    if elapsed > 5000:
        logger.warning(f"SLOW REQUEST: {request.method} {request.url.path} took {elapsed:.0f}ms")
    return response


# ── Register routers ─────────────────────────────────────────────────────────
prefix = f"/api/{settings.api_version}"
app.include_router(predict.router, prefix=prefix, tags=["Prediction"])
app.include_router(identity.router, prefix=prefix, tags=["Identity"])
app.include_router(report.router, prefix=prefix, tags=["Report & IDSS"])
app.include_router(agent.router, prefix=prefix, tags=["Agent"])

from api.routes import bovine_iq
app.include_router(bovine_iq.router, prefix=prefix, tags=["BovineIQ Agent"])


# ── Health / Status ──────────────────────────────────────────────────────────
@app.get(f"{prefix}/health", tags=["System"])
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "version": "4.0.0",
        "models": {
            "identification": "CowReIDModel (ViT + ArcFace, 16 cows)",
            "milk_prediction": "TimeSeriesTransformer",
            "heat_stress": "BehaviorCNNLSTM + THI",
            "health_scoring": "MultiModalFusion + SensorAutoencoder",
        },
        "dataset": "MMCOWS (14-day multimodal, 16 cows)",
    }


@app.get(f"{prefix}/status", tags=["System"])
async def system_status() -> Dict[str, Any]:
    return {
        "pipeline_version": "4.0.0",
        "known_cows": 16,
        "dataset_days": 14,
        "sensor_modalities": ["IMU", "UWB", "CBT", "Pressure", "THI", "Milk"],
        "models_loaded": 5,
    }


# ── Error handler ────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True, log_level=settings.log_level.lower())
