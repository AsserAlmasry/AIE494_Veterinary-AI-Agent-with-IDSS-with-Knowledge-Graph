"""
app/dependencies.py
===================
FastAPI dependency injection providers.
All heavy objects (models, services) are created once and reused across requests.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from models.identity.yolo_model import CowIdentityEngine
from models.identity.faiss_index import IdentityEmbeddingBank
from models.disease.maxvit_model import MaxViTDiseaseClassifier
from models.risk.transformer_model import HealthRiskTransformer
from services.rag_service import VeterinaryRAGService
from services.llm_service import GroqLLMService
from services.neo4j_service import Neo4jService
from services.safety_engine import ClinicalSafetyEngine
from services.vision_service import GroqVisionService
from pipelines.inference_pipeline import VeterinaryInferencePipeline


# ── Singleton factories ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_identity_engine() -> CowIdentityEngine:
    settings = get_settings()
    return CowIdentityEngine(
        model_path=settings.identity_model_path,
        model_url=settings.identity_model_url,
        device=None,  # auto-detect
    )


@lru_cache(maxsize=1)
def get_identity_bank() -> IdentityEmbeddingBank:
    settings = get_settings()
    return IdentityEmbeddingBank(
        embedding_dim=settings.identity_embedding_dim,
        persistence_path=settings.identity_bank_path,
        similarity_threshold=settings.identity_similarity_threshold,
        min_embeddings=settings.min_embeddings_per_cow,
        max_embeddings=settings.max_embeddings_per_cow,
    )


@lru_cache(maxsize=1)
def get_disease_model() -> MaxViTDiseaseClassifier:
    settings = get_settings()
    return MaxViTDiseaseClassifier(
        checkpoint_path=settings.disease_model_path,
        model_url=settings.disease_model_url,
    )


@lru_cache(maxsize=1)
def get_risk_model() -> HealthRiskTransformer:
    settings = get_settings()
    return HealthRiskTransformer(
        checkpoint_path=settings.early_warning_model_path,
        model_url=settings.early_warning_model_url,
    )


@lru_cache(maxsize=1)
def get_rag_service() -> VeterinaryRAGService:
    settings = get_settings()
    return VeterinaryRAGService(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.rag_embedding_model,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        kb_cache_path=settings.kb_cache_path,
        hf_token=settings.hf_token,
    )


@lru_cache(maxsize=1)
def get_llm_service() -> GroqLLMService:
    settings = get_settings()
    return GroqLLMService(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
    )


@lru_cache(maxsize=1)
def get_neo4j_service() -> Neo4jService:
    settings = get_settings()
    return Neo4jService(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


@lru_cache(maxsize=1)
def get_vision_service() -> GroqVisionService:
    settings = get_settings()
    return GroqVisionService(api_key=settings.groq_api_key)


@lru_cache(maxsize=1)
def get_safety_engine() -> ClinicalSafetyEngine:
    settings = get_settings()
    return ClinicalSafetyEngine(
        min_confidence=settings.min_confidence_for_auto_diagnosis,
        require_weight_for_dosing=settings.require_weight_for_dosing,
        block_uncertain=settings.block_uncertain_recommendations,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> VeterinaryInferencePipeline:
    return VeterinaryInferencePipeline(
        identity_engine=get_identity_engine(),
        identity_bank=get_identity_bank(),
        disease_model=get_disease_model(),
        risk_model=get_risk_model(),
        rag_service=get_rag_service(),
        llm_service=get_llm_service(),
        neo4j_service=get_neo4j_service(),
        safety_engine=get_safety_engine(),
        vision_service=get_vision_service(),
    )
