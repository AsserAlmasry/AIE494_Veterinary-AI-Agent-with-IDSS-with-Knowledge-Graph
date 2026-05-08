"""
app/config.py — Centralised, environment-based configuration.
All secrets are read from .env — never hardcoded.
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", os.path.join(os.getcwd(), ".env")],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ── LLM (Groq) ──────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2048

    # ── HuggingFace ─────────────────────────────────────────────────────────
    hf_token: Optional[str] = None

    # ── Neo4j ───────────────────────────────────────────────────────────────
    neo4j_uri: str = "neo4j+s://43b30b0c.databases.neo4j.io"
    neo4j_user: str = "43b30b0c"
    neo4j_password: str
    neo4j_database: str = "neo4j"

    # ── MMCOWS Model Paths ──────────────────────────────────────────────────
    mmcows_base_path: str = "."
    mmcows_src_path: str = "."
    mmcows_saved_models_dir: str = "./saved_models"

    # Individual model checkpoint names
    identification_model_name: str = "identification_model.pt"
    milk_prediction_model_name: str = "milk_prediction_model.pth"
    behavior_model_name: str = "heat_stress_model.pt"
    heat_stress_scaler_name: str = "feature_scaler.pkl"
    health_model_name: str = "health_model.pt"
    health_scaler_name: str = "health_scaler.pkl"
    disease_model_name: str = "disease_model.pth"
    anomaly_model_name: str = "anomaly_autoencoder.pth"
    fusion_model_name: str = "fusion_model.pth"

    # ── MMCOWS Data Paths ───────────────────────────────────────────────────
    mmcows_thi_data_path: str = "./preprocessing_results/thi_station_avg.csv"
    mmcows_merged_data_path: str = "./preprocessing_results/merged_multimodal_T01_0721.csv"
    mmcows_milk_data_path: str = "./preprocessing_results/milk_all_clean.csv"

    # ── Local Data Paths ────────────────────────────────────────────────────
    data_dir: str = "./data"
    chroma_persist_dir: str = "./data/chroma_vet_rag"
    kb_cache_path: str = "./data/pubmed_kb.json"

    # ── Identity System ─────────────────────────────────────────────────────
    identity_confidence_threshold: float = 0.65

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_chunk_size: int = 650
    rag_chunk_overlap: int = 80
    rag_top_k: int = 5
    rag_min_relevance_score: float = 0.55

    # ── Safety ───────────────────────────────────────────────────────────────
    min_confidence_for_auto_diagnosis: float = 0.60
    require_weight_for_dosing: bool = True
    block_uncertain_recommendations: bool = True

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_version: str = "v1"
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

    # ── Performance ──────────────────────────────────────────────────────────
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_parallel_workers: int = 4

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def identification_model_path(self) -> str:
        return os.path.join(self.mmcows_base_path, "graduation project models/Task 1-20260426T224107Z-3-001/Task 1/Idintification model.pt")

    @property
    def milk_prediction_model_path(self) -> str:
        return os.path.join(self.mmcows_base_path, "graduation project models/Task 4-20260426T224204Z-3-001/Task 4/ensemble_TabTransformer_fold3.pth")

    @property
    def behavior_model_path(self) -> str:
        return os.path.join(self.mmcows_base_path, "graduation project models/Task 2-20260426T224124Z-3-001/Task 2/best_heat_stress_model.pt")

    @property
    def anomaly_model_path(self) -> str:
        return os.path.join(self.mmcows_saved_models_dir, self.anomaly_model_name)

    @property
    def heat_stress_scaler_path(self) -> str:
        return os.path.join(self.mmcows_base_path, "graduation project models/Task 2-20260426T224124Z-3-001/Task 2/feature_scaler.pkl")

    @property
    def fusion_model_path(self) -> str:
        return os.path.join(self.mmcows_saved_models_dir, self.fusion_model_name)

    @property
    def health_model_path(self) -> str:
        return os.path.join(self.mmcows_base_path, "graduation project models/Task 5-20260426T224142Z-3-001/Task 5/best_health_model.pt")

    @property
    def health_scaler_path(self) -> str:
        return os.path.join(self.mmcows_base_path, "graduation project models/Task 5-20260426T224142Z-3-001/Task 5/health_scaler.pkl")

    @property
    def disease_model_path(self) -> str:
        return os.path.join(self.mmcows_base_path, "graduation project models/Task 3-Disease Classification/best_model.pth")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
