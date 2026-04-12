"""
api/routes/report.py
====================
POST /report    — Generate standalone clinical report.
POST /report/qa — Clinical Q&A grounded with RAG evidence.
GET  /report/kg — Knowledge graph stats.
GET  /report/kg/disease/{name} — Disease KG context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import (
    get_llm_service,
    get_rag_service,
    get_neo4j_service,
    get_safety_engine,
)

router = APIRouter()


class ReportRequest(BaseModel):
    cow_id: int
    disease_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    sensor_data: Optional[Dict[str, float]] = None
    include_rag: bool = True
    include_kg: bool = True


class QARequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    disease_context: Optional[str] = None


@router.post("/report", summary="Generate clinical report")
async def generate_report(
    req: ReportRequest,
    llm_service=Depends(get_llm_service),
    rag_service=Depends(get_rag_service),
    neo4j_service=Depends(get_neo4j_service),
    safety_engine=Depends(get_safety_engine),
) -> Dict[str, Any]:
    rag_docs: List[Dict] = []
    if req.include_rag and req.disease_predictions:
        top_disease = req.disease_predictions[0].get("disease", "cattle health")
        try:
            rag_docs = rag_service.retrieve_for_disease(top_disease)
        except Exception:
            pass

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
                pass

    report = llm_service.generate_clinical_report(
        cow_id=req.cow_id,
        disease_predictions=req.disease_predictions,
        risk_assessment=req.risk_assessment,
        rag_context=rag_docs,
        kg_context=kg_context,
        sensor_data=req.sensor_data,
    )
    report["report"] = safety_engine.inject_disclaimer(report.get("report", ""))

    return {
        "cow_id":         req.cow_id,
        "report":         report.get("report"),
        "summary":        report.get("summary"),
        "llm_model":      report.get("llm_model"),
        "evidence_count": len(rag_docs),
        "kg_enriched":    kg_context is not None,
    }


@router.post("/report/qa", summary="Clinical Q&A with RAG")
async def clinical_qa(
    req: QARequest,
    llm_service=Depends(get_llm_service),
    rag_service=Depends(get_rag_service),
) -> Dict[str, Any]:
    try:
        query = f"{req.disease_context or ''} {req.question}".strip()
        docs = rag_service.retrieve(query, top_k=3)
        context = "\n\n".join(
            f"[{i+1}] {d['title']}\n{d['snippet']}" for i, d in enumerate(docs)
        )
    except Exception:
        docs = []
        context = None

    answer = llm_service.answer_clinical_question(req.question, context)
    return {
        "question":       req.question,
        "answer":         answer,
        "evidence_count": len(docs),
        "evidence": [
            {"title": d["title"], "year": d.get("year"), "pmid": d.get("pmid")}
            for d in docs
        ],
    }


@router.get("/report/kg", summary="Knowledge graph stats")
async def kg_stats(neo4j_service=Depends(get_neo4j_service)) -> Dict[str, Any]:
    stats = neo4j_service.get_graph_stats()
    return {
        "status":      "connected" if stats.get("connected") else "disconnected",
        "node_counts": {k: v for k, v in stats.items() if k != "connected"},
    }


@router.get(
    "/report/kg/disease/{disease_name}",
    summary="Disease knowledge graph context",
)
async def disease_kg_context(
    disease_name: str,
    neo4j_service=Depends(get_neo4j_service),
) -> Dict[str, Any]:
    disease_name = disease_name.lower().replace("-", "_")
    ctx = neo4j_service.get_disease_context(disease_name)
    if not ctx:
        raise HTTPException(
            status_code=404,
            detail=f"No KG entry for disease: {disease_name}",
        )
    return ctx
