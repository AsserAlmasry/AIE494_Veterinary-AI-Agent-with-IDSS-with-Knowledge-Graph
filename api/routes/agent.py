"""
api/routes/agent.py — Unified dual-agent endpoints.
Routes /agent/chat through BovineIQ for full agentic experience.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.dependencies import (
    get_llm_service, get_rag_service, get_pipeline,
    get_data_pipeline, get_neo4j_service, get_cow_identifier,
    get_bovine_iq_agent,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    cow_id: Optional[int] = None
    disease_context: Optional[str] = None
    image_b64: Optional[str] = None

class BovineIQRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class BovineIQResumeRequest(BaseModel):
    pending_state: Dict[str, Any]
    approved: bool

class ChatResponse(BaseModel):
    answer: str
    evidence_count: int = 0
    cow_id: Optional[int] = None
    image_b64: Optional[str] = None


@router.post("/agent/chat", response_model=ChatResponse, summary="Chat with BovineIQ Veterinary Agent")
async def agent_chat(
    req: ChatRequest,
    rag_service=Depends(get_rag_service),
    neo4j_service=Depends(get_neo4j_service),
):
    """
    Routes chat messages through BovineIQ (Groq Llama 70B) for full
    agentic responses with tool calling.  Falls back to RAG-only if
    the BovineIQ agent is unavailable.
    """
    # Build a rich context string for the message
    context_parts = []
    docs = []

    if req.cow_id:
        context_parts.append(f"Current cow: #{req.cow_id}")
        try:
            history = neo4j_service.get_cow_history(req.cow_id)
            if history:
                hist_text = "\n".join(
                    f"- {h.get('disease','?')} ({h.get('confidence',0):.0%}) on {str(h.get('timestamp','?'))[:10]}"
                    for h in history[:5]
                )
                context_parts.append(f"Medical History:\n{hist_text}")
        except Exception:
            pass

    if req.disease_context:
        context_parts.append(f"Clinical context: {req.disease_context}")
        try:
            docs = rag_service.retrieve(req.disease_context, top_k=3)
            if docs:
                rag_text = "\n".join(
                    f"[{i+1}] {d.get('title','?')}\n{d.get('text','')[:200]}"
                    for i, d in enumerate(docs)
                )
                context_parts.append(f"Relevant Literature:\n{rag_text}")
        except Exception:
            pass

    # Build enriched user message
    enriched_message = req.message
    if context_parts:
        enriched_message = "\n\n".join(context_parts) + "\n\nUser question: " + req.message

    # Route through BovineIQ agent
    try:
        agent = get_bovine_iq_agent()
        result = await agent.query(enriched_message, [], image_b64=req.image_b64)
        
        image_payload = None
        if isinstance(result, dict):
            answer = result.get("content", result.get("answer", str(result)))
            image_payload = result.get("image_b64")
        else:
            answer = str(result)
            
        # EXTRACTION: Extract image_b64 if the agent tool put it in the text marker
        if "|image_b64|:" in answer:
            parts = answer.split("|image_b64|:")
            answer = parts[0].strip()
            if len(parts) > 1 and not image_payload:
                image_payload = parts[1].strip()
                if not image_payload: image_payload = None

        # FALLBACK: If still no image but we have a cow_id, try pipeline cache
        if not image_payload and req.cow_id:
            pipeline = get_pipeline()
            latest = pipeline.get_latest_status(req.cow_id)
            if latest and latest.get("image_b64"):
                image_payload = latest.get("image_b64")

        return ChatResponse(
            answer=answer, 
            evidence_count=len(docs), 
            cow_id=req.cow_id,
            image_b64=image_payload
        )
    except Exception as e:
        logger.warning(f"BovineIQ agent failed, using LLM fallback: {e}")

    # Fallback: direct LLM service
    try:
        from app.dependencies import get_llm_service
        llm_service = get_llm_service()
        context = "\n\n".join(context_parts) if context_parts else None
        answer = await llm_service.answer_clinical_question(req.message, context)
        return ChatResponse(answer=answer, evidence_count=len(docs), cow_id=req.cow_id)
    except Exception as e:
        logger.error(f"Agent chat complete failure: {e}")
        return ChatResponse(
            answer="🐄 I'm your **BovineIQ Veterinary Assistant**. I'm currently experiencing a connection issue. Please ensure the Groq API key is valid and try again.",
            evidence_count=0,
            cow_id=req.cow_id
        )


@router.get("/agent/cow/{cow_id}", summary="Get cow profile and history")
async def get_cow_profile(
    cow_id: int,
    data_pipeline=Depends(get_data_pipeline),
    neo4j_service=Depends(get_neo4j_service),
) -> Dict[str, Any]:
    milk_data = None
    try:
        milk_data = data_pipeline.get_milk_data_for_cow(cow_id)
    except Exception:
        pass
    history = []
    try:
        history = neo4j_service.get_cow_history(cow_id) or []
    except Exception:
        pass
    return {
        "cow_id": cow_id,
        "known": 1 <= cow_id <= 16,
        "milk_data": milk_data,
        "medical_history": history,
        "total_records": len(history),
    }


@router.get("/agent/daily-report/{day_index}", summary="Daily farm overview")
async def daily_farm_report(
    day_index: int = 0,
    data_pipeline=Depends(get_data_pipeline),
) -> Dict[str, Any]:
    day_data = data_pipeline.get_day_data(day_index)
    return {
        "day_index": day_index,
        "date": day_data.get("sensor_data", {}).get("date", "unknown"),
        "total_images": len(day_data.get("images", [])),
        "sensor_records": day_data.get("sensor_data", {}).get("records", 0),
        "total_cows": 16,
        "sample_images": day_data.get("images", [])[:5],
    }


@router.get("/agent/cows", summary="List all 16 known cows")
async def list_known_cows() -> Dict[str, Any]:
    return {
        "total_known_cows": 16,
        "known_cow_ids": list(range(1, 17)),
        "dataset": "MMCOWS",
        "description": "16 individually identified dairy cows from the MMCOWS multimodal dataset",
    }
