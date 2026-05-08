"""
app/main.py — FastAPI application entry-point (v4.0.0 MMCOWS Integration)
Registers all routers, middleware, lifecycle hooks, and health endpoints.
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    """Warm-up skip to avoid startup hangs in slow environments."""
    logger.info(f"CONFIG: RAG Model = {settings.rag_embedding_model}")
    logger.info(f"CONFIG: Neo4j User = {settings.neo4j_user}")
    logger.info(f"CONFIG: Neo4j URI = {settings.neo4j_uri}")
    
    logger.info("🚀 Starting Veterinary AI System v4.0.0 (MMCOWS) …")
    logger.info("  (Lazy model loading enabled to ensure rapid startup)")
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
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False, log_level=settings.log_level.lower())
