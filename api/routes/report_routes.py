"""
api/routes/report.py
====================
POST /report    — Generate a standalone clinical report from a previous analysis.
POST /report/qa — Answer a clinical question with RAG-augmented context.
GET  /report/kg — Knowledge graph stats and disease context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_llm_service, get_rag_service, get_neo4j_service, get_safety_engine

router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    cow_id: int
    disease_predictions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {disease, confidence, category} dicts"
    )
    risk_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output from the risk prediction stage"
    )
    sensor_data: Optional[Dict[str, float]] = None
    include_rag: bool = True
    include_kg: bool = True


class QARequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    disease_context: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/report",
    summary="Generate clinical report",
    description=(
        "Generate a structured Groq LLM clinical report from previously computed "
        "disease predictions and risk assessment."
    ),
)
async def generate_report(
    req: ReportRequest,
    llm_service=Depends(get_llm_service),
    rag_service=Depends(get_rag_service),
    neo4j_service=Depends(get_neo4j_service),
    safety_engine=Depends(get_safety_engine),
) -> Dict[str, Any]:
    # RAG retrieval
    rag_docs: List[Dict] = []
    if req.include_rag and req.disease_predictions:
        top_disease = req.disease_predictions[0].get("disease", "cattle health")
        try:
            rag_docs = rag_service.retrieve_for_disease(top_disease)
        except Exception as exc:
            rag_docs = []

    # KG context
    kg_context: Optional[Dict] = None
    if req.include_kg and req.disease_predictions:
        top_disease = req.disease_predictions[0].get("disease", "")
        if top_disease:
            try:
                kg_context = {
                    "disease_info": neo4j_service.get_disease_context(top_disease),
                    "related":      neo4j_service.get_related_diseases(top_disease),
                    "cow_history":  neo4j_service.get_cow_history(req.cow_id),
                }
            except Exception:
                kg_context = None

    # LLM report
    report = llm_service.generate_clinical_report(
        cow_id=req.cow_id,
        disease_predictions=req.disease_predictions,
        risk_assessment=req.risk_assessment,
        rag_context=rag_docs,
        kg_context=kg_context,
        sensor_data=req.sensor_data,
    )

    # Safety: inject disclaimer
    report["report"] = safety_engine.inject_disclaimer(report.get("report", ""))

    return {
        "cow_id":          req.cow_id,
        "report":          report.get("report"),
        "summary":         report.get("summary"),
        "llm_model":       report.get("llm_model"),
        "evidence_count":  len(rag_docs),
        "kg_enriched":     kg_context is not None,
    }


@router.post(
    "/report/qa",
    summary="Answer a clinical question",
    description="Ask a veterinary question; response is grounded by PubMed RAG retrieval.",
)
async def clinical_qa(
    req: QARequest,
    llm_service=Depends(get_llm_service),
    rag_service=Depends(get_rag_service),
) -> Dict[str, Any]:
    # Retrieve supporting evidence
    try:
        query = req.question
        if req.disease_context:
            query = f"{req.disease_context} {query}"
        docs = rag_service.retrieve(query, top_k=3)
        context = "\n\n".join(
            f"[{i+1}] {d['title']}\n{d['snippet']}"
            for i, d in enumerate(docs)
        )
    except Exception:
        docs = []
        context = None

    answer = llm_service.answer_clinical_question(
        question=req.question, context=context
    )

    return {
        "question":       req.question,
        "answer":         answer,
        "evidence_count": len(docs),
        "evidence":       [{"title": d["title"], "year": d.get("year"), "pmid": d.get("pmid")} for d in docs],
    }


@router.get(
    "/report/kg",
    summary="Knowledge graph stats",
    description="Return node counts and connectivity status of the Neo4j knowledge graph.",
)
async def kg_stats(
    neo4j_service=Depends(get_neo4j_service),
) -> Dict[str, Any]:
    stats = neo4j_service.get_graph_stats()
    return {
        "status":      "connected" if stats.get("connected") else "disconnected",
        "node_counts": {k: v for k, v in stats.items() if k != "connected"},
    }


@router.get(
    "/report/kg/disease/{disease_name}",
    summary="Disease knowledge graph context",
    description="Retrieve symptoms, treatments, and related cases for a specific disease.",
)
async def disease_kg_context(
    disease_name: str,
    neo4j_service=Depends(get_neo4j_service),
) -> Dict[str, Any]:
    disease_name = disease_name.lower().replace("-", "_")
    context = neo4j_service.get_disease_context(disease_name)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"No knowledge graph entry found for disease: {disease_name}",
        )
    return context
