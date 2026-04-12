"""
app/config.py
=============
Centralised, environment-based configuration.
All secrets are read from .env — never hardcoded.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All settings are read from environment variables / .env file.
    Pydantic v2 handles type coercion + validation automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (Groq) ──────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2048

    # ── HuggingFace ─────────────────────────────────────────────────────────────
    hf_token: Optional[str] = None

    # ── Neo4j ───────────────────────────────────────────────────────────────────
    neo4j_uri: str = "neo4j+s://43b30b0c.databases.neo4j.io"
    neo4j_user: str = "43b30b0c"
    neo4j_password: str
    neo4j_database: str = "43b30b0c"

    # ── Model Weight URLs (Google Drive) ─────────────────────────────────────────
    identity_model_url: str = "https://drive.google.com/uc?id=1xY5WAqYsh1pEfaMbSNPgFQKt_7nq5WeC"
    disease_model_url: str = "https://drive.google.com/uc?id=1WEtfzuUFraiyPlcvRyQjW7aHpnkB8n45"
    early_warning_model_url: str = "https://drive.google.com/uc?id=1GUFj7xy8s3snXNUh47_-cDpPab-E3uTm"

    # ── Local Paths ──────────────────────────────────────────────────────────────
    weights_dir: str = "./weights"
    identity_model_name: str = "mmcows_yolo_identity.pt"
    disease_model_name: str = "disease_classifier_maxvit.pt"
    early_warning_model_name: str = "early_warning_healthrisk.pt"

    data_dir: str = "./data"
    chroma_persist_dir: str = "./data/chroma_vet_rag"
    identity_bank_path: str = "./data/identity_bank.pkl"
    kb_cache_path: str = "./data/pubmed_kb.json"

    # ── Identity System ──────────────────────────────────────────────────────────
    identity_similarity_threshold: float = 0.85
    identity_embedding_dim: int = 768  # must match ViTEmbeddingExtractor.EMBEDDING_DIM
    min_embeddings_per_cow: int = 3
    max_embeddings_per_cow: int = 20

    # ── RAG ─────────────────────────────────────────────────────────────────────
    rag_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    rag_chunk_size: int = 650
    rag_chunk_overlap: int = 80
    rag_top_k: int = 5
    rag_min_relevance_score: float = 0.55

    # ── Safety ──────────────────────────────────────────────────────────────────
    min_confidence_for_auto_diagnosis: float = 0.60  # lowered: calibrated heads need lower threshold
    require_weight_for_dosing: bool = True
    block_uncertain_recommendations: bool = True

    # ── Application ─────────────────────────────────────────────────────────────
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_version: str = "v1"
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

    # ── Performance ─────────────────────────────────────────────────────────────
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_parallel_workers: int = 4
    stage1_timeout_ms: int = 1000

    # ── Derived helpers ──────────────────────────────────────────────────────────
    @property
    def identity_model_path(self) -> str:
        return os.path.join(self.weights_dir, self.identity_model_name)

    @property
    def disease_model_path(self) -> str:
        return os.path.join(self.weights_dir, self.disease_model_name)

    @property
    def early_warning_model_path(self) -> str:
        return os.path.join(self.weights_dir, self.early_warning_model_name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton settings instance.
    Use ``get_settings()`` everywhere — never instantiate Settings() directly.
    """
    return Settings()
