"""
pipelines/inference_pipeline.py
================================
Unified Veterinary AI IDSS pipeline — v3.0.0

Execution stages
-----------------
1.  Image decode
2.  Cow identity (YOLO + ViT)
3.  Identity bank lookup
4.  Vision analysis (Groq Vision) ← PRIMARY disease signal (actually sees the cow)
5.  Disease classification (MaxViT → blended with vision if vision available)
6.  Safety validation + zoonotic/notifiable checks
7.  Auto-record diagnosis to Neo4j
8.  Risk prediction (Transformer + sensor data)
9.  Knowledge graph context
10. RAG retrieval (ChromaDB → fallback to Neo4j research papers)
11. Groq LLM expert IDSS clinical report (with weight-based dosing)
12. Clinical summary assembly
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VeterinaryInferencePipeline:
    """
    Production inference pipeline — stateless, thread-safe.
    All services injected via dependency injection.
    """

    def __init__(
        self,
        identity_engine,
        identity_bank,
        disease_model,
        risk_model,
        rag_service,
        llm_service,
        neo4j_service,
        safety_engine,
        vision_service=None,
    ) -> None:
        self.identity_engine = identity_engine
        self.identity_bank   = identity_bank
        self.disease_model   = disease_model
        self.risk_model      = risk_model
        self.rag_service     = rag_service
        self.llm_service     = llm_service
        self.neo4j_service   = neo4j_service
        self.safety_engine   = safety_engine
        self.vision_service  = vision_service

    # ── Main entry point ──────────────────────────────────────────────────────
    async def run_full_pipeline(
        self,
        image_bytes: Optional[bytes] = None,
        sensor_data: Optional[Dict[str, float]] = None,
        history_window: Optional[List[Dict[str, float]]] = None,
        animal_weight_kg: Optional[float] = None,
        animal_age_years: Optional[float] = None,
        cow_id_override: Optional[int] = None,
        generate_report: bool = True,
    ) -> Dict[str, Any]:
        t_pipeline_start = time.perf_counter()
        result: Dict[str, Any] = {
            "pipeline_version": "3.0.0",
            "stages":           {},
            "errors":           [],
        }

        # ── Stage 1: Image decode ─────────────────────────────────────────────
        image       = None
        image_array = None
        if image_bytes:
            try:
                from PIL import Image
                image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_array = np.array(image)
                result["stages"]["image_loaded"] = True
            except Exception as exc:
                result["errors"].append(f"Image decode: {exc}")

        # ── Step 1: Parallel Inputs (Identity, Vision, MaxViT) ───────────────
        async def _run_identity(arr):
            if cow_id_override: return {"cow_id": cow_id_override, "method": "manual_override"}
            return await asyncio.to_thread(self.identity_engine.identify, arr)

        async def _run_vision(b):
            if not self.vision_service: return {}
            return await asyncio.to_thread(self.vision_service.analyze_image, b)

        async def _run_disease(img):
            return await asyncio.to_thread(self.disease_model.predict, img)

        # Launch parallel tasks
        tasks = []
        if image_array is not None:
            tasks.append(_run_identity(image_array))
            tasks.append(_run_vision(image_bytes))
            tasks.append(_run_disease(image))

        # Wait for first stage results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Unpack results with safety
        identity_result = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else {}
        vision_result   = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {}
        disease_result  = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}

        if any(isinstance(r, Exception) for r in results):
            for r in results:
                if isinstance(r, Exception): result["errors"].append(str(r))

        cow_id = cow_id_override or identity_result.get("cow_id", 0)
        result["stages"]["identity"]        = identity_result
        result["stages"]["vision_analysis"] = vision_result
        
        # ── Identity Bank (Background) ──────────────────────────────────────
        if image_array is not None:
            try:
                embedding = await asyncio.to_thread(self.identity_engine.embedder.extract, image_array)
                bank_result = await asyncio.to_thread(self.identity_bank.identify_cow, embedding)
                result["stages"]["identity_bank"] = bank_result
            except Exception as exc:
                result["errors"].append(f"Identity bank: {exc}")

        # ── Stage 5: Disease classification (MaxViT + vision merge) ──────────
        disease_result: Dict[str, Any] = {}
        if image is not None:
            try:
                t0 = time.perf_counter()
                disease_result = self.disease_model.predict(image)

                # Merge vision candidates INTO the MaxViT probabilities
                # Vision takes strong priority (0.7 weight) over uncalibrated model (0.3 weight)
                vision_mapped = vision_result.get("mapped_classes", [])
                if vision_mapped:
                    all_probs = disease_result.get("all_probabilities", {})
                    for vc in vision_mapped:
                        dname = vc["disease"]
                        v_conf = vc["confidence"]
                        m_conf = all_probs.get(dname, 0.5)
                        # weighted blend: vision 0.70, model 0.30
                        blended = round(v_conf * 0.70 + m_conf * 0.30, 4)
                        all_probs[dname] = blended

                    # If cow appears healthy per vision, boost healthy class
                    if vision_result.get("appears_healthy"):
                        all_probs["healthy"] = max(all_probs.get("healthy", 0.5), 0.75)

                    disease_result["all_probabilities"] = all_probs

                    # Re-derive top predictions from merged probabilities
                    threshold = 0.30
                    new_preds = sorted(
                        [
                            {
                                "disease":    d,
                                "confidence": float(p),
                                "category":   self.disease_model._get_category(d),
                                "source":     "vision+model" if d in {vc["disease"] for vc in vision_mapped} else "model",
                            }
                            for d, p in all_probs.items()
                            if p >= threshold
                        ],
                        key=lambda x: x["confidence"],
                        reverse=True,
                    )
                    disease_result["predictions"] = new_preds[:3]
                    disease_result["vision_merged"] = True
                    disease_result["vision_model_used"] = vision_result.get("vision_model", "")

                safety_check = self.safety_engine.validate_predictions(
                    disease_predictions=disease_result.get("predictions", []),
                    model_uncertainty=disease_result.get("model_uncertainty", 0),
                    animal_weight_kg=animal_weight_kg,
                )
                disease_result["safety"] = safety_check
                disease_result["inference_time_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
                result["stages"]["disease"] = disease_result
                top = (disease_result.get("predictions") or [{}])[0].get("disease", "?")
                logger.info(
                    f"Disease: top={top} "
                    f"vision_merged={disease_result.get('vision_merged', False)} "
                    f"({disease_result['inference_time_ms']:.0f}ms)"
                )
            except Exception as exc:
                result["errors"].append(f"Disease: {exc}")

        # ── Stage 6: Auto-record to Neo4j ─────────────────────────────────────
        preds      = disease_result.get("predictions", [])
        top_disease    = preds[0].get("disease", "") if preds else ""
        top_confidence = preds[0].get("confidence", 0) if preds else 0
        all_disease_names = [p["disease"] for p in preds]
        if top_disease and cow_id:
            try:
                result["stages"]["diagnosis_recorded"] = await asyncio.to_thread(
                    self.neo4j_service.upsert_cow_case,
                    cow_id=cow_id, disease=top_disease, confidence=top_confidence
                )
            except Exception as exc:
                result["errors"].append(f"KG record: {exc}")

        # ── Step 2: Parallel Context (Risk, KG, RAG) ────────────────────────
        async def _run_risk():
            if not sensor_data: return {}
            # Standard risk model + safety validation
            res = await asyncio.to_thread(self.risk_model.predict_risk, sensor_data, history_window, cow_id)
            return await asyncio.to_thread(self.safety_engine.validate_risk_assessment, res)

        async def _run_kg():
            if not top_disease: return {}
            # Multi-query KG lookup
            return {
                "disease_info":        await asyncio.to_thread(self.neo4j_service.get_disease_context, top_disease),
                "treatment_protocols": await asyncio.to_thread(self.neo4j_service.get_treatment_protocol, top_disease),
                "progression_risks":   await asyncio.to_thread(self.neo4j_service.get_progression_risk, top_disease),
                "related":             await asyncio.to_thread(self.neo4j_service.get_related_diseases, top_disease),
                "cow_history":         await asyncio.to_thread(self.neo4j_service.get_cow_history, cow_id),
                "zoonotic_alerts":     await asyncio.to_thread(self.neo4j_service.get_zoonotic_info, all_disease_names),
                "neo4j_research":      [await asyncio.to_thread(self.neo4j_service.get_research_evidence, d, 3) for d in all_disease_names[:2]]
            }

        async def _run_rag():
            if not all_disease_names: return {}
            # Upgraded RAG logic
            docs = await asyncio.to_thread(self.rag_service.retrieve_for_diseases, all_disease_names, 5)
            # Fallback to KG papers if Chroma empty
            if not docs:
                neo4j_papers = []
                for d in all_disease_names[:3]:
                    papers = await asyncio.to_thread(self.neo4j_service.get_research_evidence, d, 3)
                    for p in papers:
                        neo4j_papers.append({
                            "text": f"{p.get('title','')}. {p.get('abstract','')}",
                            "snippet": p.get("abstract","")[:300],
                            "title": p.get("title","Neo4j Research"),
                            "source": f"Neo4j/PubMed:{p.get('journal','')}",
                            "disease": d,
                        })
                docs = neo4j_papers
            return {
                "query": all_disease_names,
                "retrieved": len(docs),
                "documents": docs
            }

        # Parallel gather context
        context_results = await asyncio.gather(_run_risk(), _run_kg(), _run_rag(), return_exceptions=True)
        
        risk_result = context_results[0] if not isinstance(context_results[0], Exception) else {}
        kg_context  = context_results[1] if not isinstance(context_results[1], Exception) else {}
        rag_result  = context_results[2] if not isinstance(context_results[2], Exception) else {}

        # Log context errors
        for i, res in enumerate(context_results):
            if isinstance(res, Exception):
                result["errors"].append(f"Context Stage {i}: {res}")

        result["stages"]["risk"] = risk_result
        result["stages"]["knowledge_graph"] = kg_context
        result["stages"]["rag"] = rag_result

        # ── Stage 3: LLM expert IDSS report ──────────────────────────────────
        if generate_report:
            try:
                rag_docs = rag_result.get("documents", [])
                report = await asyncio.to_thread(
                    self.llm_service.generate_clinical_report,
                    cow_id=cow_id,
                    disease_predictions=preds,
                    risk_assessment=risk_result or {"risk_level": "unknown"},
                    rag_context=rag_docs,
                    kg_context=kg_context or None,
                    sensor_data=sensor_data,
                    animal_weight_kg=animal_weight_kg,
                    animal_age_years=animal_age_years,
                    vision_analysis=vision_result or None,
                    safety_status=disease_result.get("safety", {}),
                )
                report["report"] = self.safety_engine.inject_disclaimer(report.get("report", ""))
                result["stages"]["report"] = report
            except Exception as exc:
                result["errors"].append(f"LLM report: {exc}")

        # ── Stage 11: Clinical summary ────────────────────────────────────────
        result["stages"]["clinical_summary"] = self._build_clinical_summary(
            cow_id=cow_id,
            disease_result=disease_result,
            risk_result=risk_result,
            kg_context=kg_context,
            safety_check=disease_result.get("safety", {}),
            vision_result=vision_result,
        )

        result["cow_id"]           = cow_id
        result["total_latency_ms"] = round(
            (time.perf_counter() - t_pipeline_start) * 1000, 2
        )
        result["success"] = len(result["errors"]) == 0
        logger.info(
            f"Pipeline complete | cow={cow_id} | "
            f"latency={result['total_latency_ms']:.0f}ms | "
            f"errors={len(result['errors'])}"
        )
        return result

    # ── Clinical summary ──────────────────────────────────────────────────────

    def _build_clinical_summary(
        self,
        cow_id: int,
        disease_result: Dict,
        risk_result: Dict,
        kg_context: Dict,
        safety_check: Dict,
        vision_result: Dict,
    ) -> Dict[str, Any]:
        predictions = disease_result.get("predictions", [])
        risk_level  = risk_result.get("risk_level", "unknown")
        urgency     = safety_check.get("clinical_urgency", 3)
        zoonotic    = safety_check.get("zoonotic_diseases", [])
        notifiable  = safety_check.get("notifiable_diseases", [])

        first_line_treatments = [
            t["treatment"] for t in kg_context.get("treatment_protocols", [])
            if t.get("first_line")
        ]
        progression = [
            f"{p['progresses_to']} ({p['probability']:.0%}/{p['time_days']}d)"
            for p in kg_context.get("progression_risks", [])
            if p.get("time_days")
        ]

        # Vision summary
        vision_summary = ""
        if vision_result and vision_result.get("disease_candidates"):
            top_v = vision_result["disease_candidates"][0]
            vision_summary = (
                f"{top_v.get('disease', '?')} ({top_v.get('confidence', 0):.0%} confidence "
                f"by visual inspection)"
            )

        return {
            "cow_id":                cow_id,
            "primary_finding":       predictions[0]["disease"].replace("_", " ").title() if predictions else "No findings",
            "primary_confidence":    round(predictions[0]["confidence"], 3) if predictions else 0,
            "all_findings":          [p["disease"] for p in predictions],
            "vision_primary_finding":vision_summary,
            "risk_level":            risk_level,
            "clinical_urgency":      urgency,
            "urgency_label":         self._urgency_label(urgency),
            "safety_level":          safety_check.get("safety_level", "unknown"),
            "zoonotic_risk":         bool(zoonotic),
            "notifiable_diseases":   notifiable,
            "zoonotic_diseases":     zoonotic,
            "first_line_treatments": first_line_treatments[:3],
            "progression_risks":     progression[:2],
            "action_required":       urgency >= 7,
            "veterinary_exam_needed":urgency >= 5 or risk_level == "high",
        }

    @staticmethod
    def _urgency_label(score: int) -> str:
        if score >= 9:  return "CRITICAL - Immediate action"
        if score >= 7:  return "URGENT - Contact vet within 24h"
        if score >= 5:  return "ELEVATED - Monitor closely"
        if score >= 3:  return "ROUTINE - Standard monitoring"
        return "MINIMAL - Normal health checks"

    # ── Convenience shortcuts ─────────────────────────────────────────────────

    async def run_disease_only(self, image_bytes: bytes) -> Dict[str, Any]:
        result = await self.run_full_pipeline(image_bytes=image_bytes, generate_report=False)
        return {
            "cow_id":           result.get("cow_id"),
            "disease":          result["stages"].get("disease", {}),
            "vision_analysis":  result["stages"].get("vision_analysis", {}),
            "clinical_summary": result["stages"].get("clinical_summary", {}),
        }

    async def run_risk_only(
        self,
        sensor_data: Dict[str, float],
        history_window: Optional[List] = None,
        cow_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = await self.run_full_pipeline(
            sensor_data=sensor_data,
            history_window=history_window,
            cow_id_override=cow_id,
            generate_report=False,
        )
        return {
            "cow_id":           result.get("cow_id"),
            "risk":             result["stages"].get("risk", {}),
            "rag":              result["stages"].get("rag", {}),
            "clinical_summary": result["stages"].get("clinical_summary", {}),
        }
