"""
app/dependencies.py — FastAPI dependency injection providers.
All heavy objects (models, services) are created once and reused.
Now wired to the REAL MMCOWS models instead of fake checkpoints.
"""
from __future__ import annotations
from functools import lru_cache
from app.config import get_settings

# ── MMCOWS Model Factories ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_cow_identifier():
    from models.mmcows.cow_identifier import CowIdentifier
    s = get_settings()
    return CowIdentifier(
        checkpoint_path=s.identification_model_path,
        mmcows_src_path=s.mmcows_src_path,
        confidence_threshold=s.identity_confidence_threshold,
    )

@lru_cache(maxsize=1)
def get_milk_predictor():
    from models.mmcows.milk_predictor import MilkProductivityPredictor
    s = get_settings()
    return MilkProductivityPredictor(
        checkpoint_path=s.milk_prediction_model_path,
        mmcows_src_path=s.mmcows_src_path,
    )

@lru_cache(maxsize=1)
def get_heat_stress_analyzer():
    from models.mmcows.heat_stress_analyzer import HeatStressAnalyzer
    s = get_settings()
    return HeatStressAnalyzer(
        checkpoint_path=s.behavior_model_path,
        mmcows_src_path=s.mmcows_src_path,
    )

@lru_cache(maxsize=1)
def get_health_scorer():
    from models.mmcows.health_scorer import HealthScorer
    s = get_settings()
    return HealthScorer(
        mmcows_base_path=s.mmcows_base_path,
    )

@lru_cache(maxsize=1)
def get_data_pipeline():
    from models.mmcows.data_loader import MMCowsDataPipeline
    s = get_settings()
    return MMCowsDataPipeline(mmcows_base_path=s.mmcows_base_path)

# ── Service Factories ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_service():
    from services.rag_service import VeterinaryRAGService
    s = get_settings()
    return VeterinaryRAGService(
        persist_dir=s.chroma_persist_dir, embedding_model=s.rag_embedding_model,
        chunk_size=s.rag_chunk_size, chunk_overlap=s.rag_chunk_overlap,
        top_k=s.rag_top_k, kb_cache_path=s.kb_cache_path, hf_token=s.hf_token,
    )

@lru_cache(maxsize=1)
def get_llm_service():
    from services.llm_service import GroqLLMService
    s = get_settings()
    return GroqLLMService(api_key=s.groq_api_key, model=s.groq_model,
                          temperature=s.groq_temperature, max_tokens=s.groq_max_tokens)

@lru_cache(maxsize=1)
def get_neo4j_service():
    from services.neo4j_service import Neo4jService
    s = get_settings()
    return Neo4jService(uri=s.neo4j_uri, user=s.neo4j_user,
                        password=s.neo4j_password, database=s.neo4j_database)

@lru_cache(maxsize=1)
def get_disease_service():
    from models.mmcows.disease_classifier import MaxVitDiseaseService
    s = get_settings()
    return MaxVitDiseaseService(model_path=r"C:\Users\Dell\Downloads\best_model.pth")

@lru_cache(maxsize=1)
def get_safety_engine():
    from services.safety_engine import ClinicalSafetyEngine
    s = get_settings()
    return ClinicalSafetyEngine(
        min_confidence=s.min_confidence_for_auto_diagnosis,
        require_weight_for_dosing=s.require_weight_for_dosing,
        block_uncertain=s.block_uncertain_recommendations,
    )

@lru_cache(maxsize=1)
def get_bovine_iq_agent():
    from services.bovine_iq_service import BovineIQAgent
    s = get_settings()
    
    # Try to inject dependencies if available
    data_pipeline = rag = neo4j = pipeline = None
    try: data_pipeline = get_data_pipeline()
    except Exception: pass
    try: rag = get_rag_service()
    except Exception: pass
    try: neo4j = get_neo4j_service()
    except Exception: pass
    try: pipeline = get_pipeline()
    except Exception: pass
    
    return BovineIQAgent(
        api_key=s.groq_api_key,
        data_pipeline=data_pipeline,
        neo4j_service=neo4j,
        rag_service=rag,
        inference_pipeline=pipeline,
        model_name=s.groq_model
    )

@lru_cache(maxsize=1)
def get_pipeline():
    from pipelines.inference_pipeline import VeterinaryInferencePipeline

    # Optional services — degrade gracefully
    disease_service = rag = llm = neo4j = safety = None
    try: disease_service = get_disease_service()
    except Exception: pass
    try: rag = get_rag_service()
    except Exception: pass
    try: llm = get_llm_service()
    except Exception: pass
    try: neo4j = get_neo4j_service()
    except Exception: pass
    try: safety = get_safety_engine()
    except Exception: pass

    return VeterinaryInferencePipeline(
        cow_identifier=get_cow_identifier(),
        milk_predictor=get_milk_predictor(),
        heat_stress_analyzer=get_heat_stress_analyzer(),
        health_scorer=get_health_scorer(),
        data_pipeline=get_data_pipeline(),
        vision_service=disease_service,  # We replace the dummy vision with the real PyTorch model
        rag_service=rag,
        llm_service=llm,
        neo4j_service=neo4j,
        safety_engine=safety,
    )

