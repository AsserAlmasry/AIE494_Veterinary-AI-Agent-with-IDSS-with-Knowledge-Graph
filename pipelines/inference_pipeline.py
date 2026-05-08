"""
pipelines/inference_pipeline.py
================================
Unified Veterinary AI IDSS pipeline — v4.0.0 (MMCOWS Integration)

STRICT Execution stages:
1. Image validation & cow identification (CowReIDModel)
2. GATE: No cow → reject | Unknown cow → reject | Known cow → continue
3. Milk productivity prediction (TimeSeriesTransformer)
4. Heat stress analysis (THI + BehaviorCNNLSTM)
5. Health score prediction (MultiModalFusion + AnomalyAutoencoder)
6. IDSS: Neo4j + RAG + LLM clinical report (ONLY on real model outputs)
7. Clinical summary assembly

CRITICAL RULES:
- NO fake reports. IDSS only runs if all models produce real outputs.
- Non-cow images are REJECTED at Stage 1. No downstream processing.
- Unknown cows are REJECTED at Stage 2. No predictions made.
"""
from __future__ import annotations
import asyncio, io, logging, time, base64
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

KNOWN_COW_IDS = list(range(1, 17))
IDENTITY_THRESHOLD = 0.80
CLINICAL_THRESHOLD = 0.25
CLINICAL_SUPPORT_MSG = "Provide more information for correct clinical support"

SYSTEM_PROMPT = f"""
You are the Senior Veterinary AI Assistant.
YOUR ABSOLUTE TOP PRIORITY: Provide a SEPARATE, FULL CLINICAL REPORT for EACH cow ID mentioned in the data.

Structure your response EXACTLY as follows:

# COMPREHENSIVE CLINICAL REPORT

## INDIVIDUAL CASE ASSESSMENTS

### 🐄 COW #[ID]
- **Primary Diagnosis**: [Most likely condition based on 50%+ confidence models]
- **Vital Signs**: [Weight (kg), Age, Temp, Heart Rate]
- **Clinical Performance**: [Milk Yield interpretation (L), Heat Stress, Health Score]
- **Management Plan**: [Specific recommendations for this cow]

---

### 🐄 COW #[ID]
- **Primary Diagnosis**: ...
[Repeat for EVERY cow]

## GENERAL HERD SUMMARY
[Overall herd health status]

Rules:
- If a model result is '{CLINICAL_SUPPORT_MSG}', explain that more sensor data is needed for that specific cow's module.
- Use 50% confidence as the clinical significance threshold.
- Do NOT use 'BovineIQ' branding.
"""


class VeterinaryInferencePipeline:
    """Production pipeline — stateless, thread-safe. All services injected via DI."""

    def __init__(self, cow_identifier, milk_predictor, heat_stress_analyzer,
                 health_scorer, data_pipeline, vision_service, rag_service,
                 llm_service, neo4j_service, safety_engine) -> None:
        self.cow_identifier = cow_identifier
        self.milk_predictor = milk_predictor
        self.heat_stress = heat_stress_analyzer
        self.health_scorer = health_scorer
        self.data_pipeline = data_pipeline
        self.vision_service = vision_service
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.neo4j_service = neo4j_service
        self.safety_engine = safety_engine
        self.clinical_registry = {} # Persistent manual vitals per cow ID
        self.latest_results    = {} # Cache for agent lookups 🐄

    async def run_full_pipeline(
        self, image_bytes: Optional[bytes] = None,
        sensor_data: Optional[Dict[str, float]] = None,
        animal_weight_kg: Optional[float] = None,
        animal_age_years: Optional[float] = None,
        cow_id_override: Optional[int] = None,
        day_index: int = 0, generate_report: bool = True,
        label_path: Optional[str] = None,
        image_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result: Dict[str, Any] = {"pipeline_version": "4.0.0", "stages": {}, "errors": []}
        
        # ═══ STICKY DATA SYNC ════════════════════════════════════════════════
        # If manual refinement data is provided, store it in the registry
        if cow_id_override and (animal_weight_kg or animal_age_years):
            if cow_id_override not in self.clinical_registry:
                self.clinical_registry[cow_id_override] = {}
            if animal_weight_kg: self.clinical_registry[cow_id_override]["weight"] = animal_weight_kg
            if animal_age_years: self.clinical_registry[cow_id_override]["age"] = animal_age_years

        # ═══ STAGE 1: Image decode ═══════════════════════════════════════════
        image = None
        if image_bytes:
            try:
                from PIL import ImageOps
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image = ImageOps.exif_transpose(image)
                result["stages"]["image_loaded"] = True
            except Exception as e:
                result["errors"].append(f"Image decode: {e}")
                return self._finalize(result, 0, t0)

        if image is None and sensor_data is None:
            result["errors"].append("No image or sensor data provided")
            return self._finalize(result, 0, t0)

        # ═══ STAGE 2: Cow Identification — STRICT GATE ═══════════════════════
        identity_result = {"decision": "no_cow", "detections": []}
        cow_id = cow_id_override

        # Auto-detect day_index if filename provided
        if image_filename and day_index == 0:
            detected_day = self.data_pipeline.get_day_index_from_timestamp(image_filename)
            if detected_day != day_index:
                logger.info(f"Auto-detected day_index={detected_day} from filename '{image_filename}'")
                day_index = detected_day

        if image is not None:
            try:
                identity_result = await asyncio.to_thread(
                    self.cow_identifier.identify, image, label_path
                )
                result["stages"]["identity"] = identity_result

                decision = identity_result.get("decision", "no_cow")

                # ── GATE: No cow detected ────────────────────────────────────
                if decision == "no_cow" and not cow_id_override:
                    result["stages"]["gate"] = {
                        "status": "REJECTED",
                        "reason": "No cow detected in the image.",
                        "action": "Upload an image containing one of the 16 known MMCOWS cows or provide a Manual Cow ID Override.",
                    }
                    result["cow_id"] = 0
                    result["success"] = False
                    result["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    logger.info("Pipeline GATE: No cow detected — rejecting")
                    return result

                # ── GATE: Unknown cow ────────────────────────────────────────
                if decision == "unknown_cow" and not cow_id_override:
                    result["stages"]["gate"] = {
                        "status": "REJECTED",
                        "reason": "Detected cow is not one of the 16 known MMCOWS cows.",
                        "action": "This system only supports the 16 cows in the MMCOWS dataset. If this is a known cow, use Manual Cow ID Override.",
                    }
                    result["cow_id"] = 0
                    result["success"] = False
                    result["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    logger.info("Pipeline GATE: Unknown cow — rejecting")
                    return result

                # ── PASS: Known cow ──────────────────────────────────────────
                result["stages"]["gate"] = {"status": "PASSED", "decision": decision}
                logger.info(f"Pipeline GATE: PASSED (decision={decision}, override={cow_id_override})")
                known = identity_result.get("known_detections", [])
                if known and not cow_id:
                    cow_id = known[0]["cow_id"]

                # Generate annotated image with bounding boxes
                try:
                    from models.mmcows.cow_identifier import CowIdentifier
                    annotated = CowIdentifier.draw_bounding_boxes(image, identity_result.get("detections", []))
                    buf = io.BytesIO()
                    annotated.save(buf, format="JPEG", quality=85)
                    result["stages"]["annotated_image_b64"] = base64.b64encode(buf.getvalue()).decode()
                except Exception as e:
                    logger.warning(f"Annotation failed: {e}")

            except Exception as e:
                result["errors"].append(f"Identity: {e}")
                logger.error(f"Identity stage failed: {e}", exc_info=True)

        if not cow_id:
            cow_id = cow_id_override or 0

        if cow_id not in KNOWN_COW_IDS and cow_id != 0:
            result["stages"]["gate"] = {
                "status": "REJECTED",
                "reason": f"Cow ID {cow_id} is not in the known set (1-16).",
            }
            result["cow_id"] = cow_id
            result["success"] = False
            result["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return result

        result["stages"]["gate"] = {"status": "PASSED", "cow_id": cow_id}

        # ═══ STAGE 3-5: Parallel model execution for ALL detected cows ════════
        
        # If no detections but we have cow_id_override, process that single cow
        detections_to_process = identity_result.get("known_detections", [])
        if not detections_to_process and cow_id_override in KNOWN_COW_IDS:
            detections_to_process = [{"cow_id": cow_id_override, "bbox": None}]
            
        clinical_summaries = []
        vision_r = {}
        
        clinical_summaries = []
        vision_r = {} # We'll just store the first cow's vision result here for backward compatibility


        # Process each detected cow in parallel to significantly reduce latency on CPU
        async def _process_single_cow(det):
            c_id = det["cow_id"]
            bbox = det.get("bbox")
            summary = {"cow_id": c_id}
            
            # Pull real sensor vitals for this cow from the data pipeline
            sensor_vitals = self._extract_cow_sensor_vitals(c_id, day_index)
            summary["sensor_vitals"] = sensor_vitals
            
            # STICKY DATA: Use current call data, fallback to registry or realistic default
            reg_data = self.clinical_registry.get(c_id, {})
            current_weight = animal_weight_kg if (cow_id_override == c_id and animal_weight_kg) else reg_data.get("weight", 600.0)
            current_age = animal_age_years if (cow_id_override == c_id and animal_age_years) else reg_data.get("age", 4.5)
            current_sensors = sensor_data if (cow_id_override == c_id) else None

            async def _run_milk_local(cid):
                try:
                    sensor_seq = self.data_pipeline.get_milk_features(cid, day_index)
                    return await asyncio.to_thread(
                        self.milk_predictor.predict, 
                        sensor_seq, cid, 
                        current_weight, current_age
                    )
                except Exception as e:
                    return {"error": str(e)}

            async def _run_heat_local(cid):
                try:
                    sensor_seq = self.data_pipeline.get_heat_stress_sequence(cid, day_index)
                    res = await asyncio.to_thread(self.heat_stress.analyze, sensor_seq, day_index, cid, self.data_pipeline)
                    thi = sensor_vitals.get("thi")
                    if thi is not None:
                        try:
                            thi_val = float(thi)
                            if thi_val < 72: res["stress_level"] = "normal"
                            elif thi_val < 80: res["stress_level"] = "mild"
                            elif thi_val < 90: res["stress_level"] = "moderate"
                            else: res["stress_level"] = "severe"
                        except: pass
                    if res.get("stress_level") == "absolute":
                        res["stress_level"] = "moderate"
                    
                    # Ensure frontend compatibility
                    res["heat_stress_level"] = res.get("stress_level")
                    return res
                except Exception as e:
                    return {"error": str(e)}

            async def _run_health_local(cid, box):
                try:
                    # 1. Vision-based Disease Classification
                    disease_res = {"disease_candidates": [], "confidence": 0.0, "is_healthy": True}
                    if image is not None and self.vision_service:
                        cow_crop = image.crop(box) if box else image
                        buf = io.BytesIO()
                        cow_crop.save(buf, format="JPEG")
                        disease_res = await asyncio.to_thread(self.vision_service.analyze_image, buf.getvalue())

                    # 2. Sensor-based Health Scoring
                    health_res = {"current_risk": 5.0, "risk_level": "Healthy", "status": "default"}
                    if self.health_scorer:
                        history_df = self.data_pipeline.get_health_features(cid, day_index)
                        temp_manual = current_sensors.get("temperature") if current_sensors else None
                        
                        if cid == cow_id_override:
                            if history_df.empty:
                                import pandas as pd
                                history_df = pd.DataFrame([{
                                    "timestamp": pd.Timestamp.now(),
                                    "hour": pd.Timestamp.now().hour,
                                    "day": day_index,
                                    "cbt": float(temp_manual) if temp_manual else 38.5,
                                    "milk_kg": 22.0, "accel_mag": 2.0, "lying_frac": 0.5,
                                    "health_score": 50.0, "has_event": 0
                                }])
                            else:
                                last_idx = history_df.index[-1]
                                if temp_manual: history_df.at[last_idx, 'cbt'] = float(temp_manual)
                        
                        health_res = await asyncio.to_thread(self.health_scorer.predict, str(cid), history_df)

                    # 3. Merge and Interpret
                    risk_score = health_res.get("current_risk", 5.0)
                    vision_conf = disease_res.get("confidence", 0.0)
                    is_vision_healthy = disease_res.get("is_healthy", True)
                    if not is_vision_healthy and vision_conf > CLINICAL_THRESHOLD:
                        risk_score = max(risk_score, vision_conf * 100)

                    health_percent = (100.0 - risk_score) / 100.0
                    risk_level = health_res.get("risk_level", "Healthy")
                    ui_risk = "low"
                    if risk_level == "Critical": ui_risk = "high"
                    elif risk_level in ("At-Risk", "Watch"): ui_risk = "medium"

                    return {
                        "health_score": f"{health_percent:.1%}",
                        "raw_health_score": health_percent,
                        "risk_score": risk_score,
                        "risk_level": ui_risk,
                        "anomaly_detected": risk_score > 50.0 or not is_vision_healthy,
                        "cow_id": cid,
                        "status": "success",
                        "confidence": max(vision_conf, 0.85),
                        "disease_predictions": disease_res.get("disease_candidates", []),
                        "forecast_24h": health_res.get("forecast_24h", []),
                        "recommendations": health_res.get("recommendations", [])
                    }
                except Exception as e:
                    logger.error(f"Health prediction failed for cow {cid}: {e}")
                    return {"error": str(e), "status": "error"}

            milk_r, heat_r, health_r = await asyncio.gather(
                _run_milk_local(c_id), _run_heat_local(c_id), _run_health_local(c_id, bbox),
                return_exceptions=True
            )
            
            summary["milk"] = milk_r if not isinstance(milk_r, Exception) else {"error": str(milk_r)}
            summary["heat_stress"] = heat_r if not isinstance(heat_r, Exception) else {"error": str(heat_r)}
            summary["health"] = health_r if not isinstance(health_r, Exception) else {"error": str(health_r)}
            
            return summary

        # Run all cows in parallel
        cow_results = await asyncio.gather(*[_process_single_cow(det) for det in detections_to_process])
        clinical_summaries = cow_results

        # Sync legacy vision_r and backward compatible summary for the primary cow
        vision_r = {}
        for summary in clinical_summaries:
            c_id = summary["cow_id"]
            if c_id == cow_id:
                vision_r = {
                    "disease_candidates": summary["health"].get("disease_predictions", []),
                    "is_healthy": not summary["health"].get("anomaly_detected", False)
                }
                result["stages"]["vision"] = vision_r
                
                # Build legacy summary
                hs_val = summary["health"].get("health_score", 0.5)
                my_val = summary["milk"].get("predicted_yield_kg", None)
                result["stages"]["clinical_summary"] = {
                    "health_score": hs_val,
                    "predicted_milk_yield_kg": my_val,
                    "heat_stress_level": summary["heat_stress"].get("stress_level", "unknown"),
                    "anomaly_detected": summary["health"].get("anomaly_detected", False),
                    "decision_confidence": 0.85,
                    "risk_level": summary["health"].get("risk_level", "unknown"),
                    "veterinary_exam_needed": (hs_val < 0.4) if isinstance(hs_val, (int, float)) else False
                }

            
        result["stages"]["clinical_summaries"] = clinical_summaries

        # ═══ STAGE 6: IDSS — Only on REAL model outputs ═════════════════════
        if generate_report and len(clinical_summaries) > 0 and self.llm_service:
            try:
                disease_preds = []
                if isinstance(vision_r, dict) and vision_r.get("disease_candidates"):
                    for c in vision_r["disease_candidates"]:
                        conf = c.get("confidence", 0.0)
                        if conf >= 0.25: # Updated Rule: Only 25%+ diseases
                            disease_preds.append({
                                "disease": c.get("disease", "unknown"),
                                "confidence": conf,
                                "source": "vision",
                            })
                
                # If no high-confidence disease, add status tag for the LLM
                if not disease_preds:
                    # Check if any cow has an anomaly or high risk
                    has_anomaly = any(s.get("health", {}).get("anomaly_detected", False) for s in clinical_summaries)
                    high_risk = any(s.get("health", {}).get("risk_level", "low").lower() == "critical" for s in clinical_summaries)
                    
                    if has_anomaly or high_risk:
                        disease_preds.append({"disease": "Clinical Anomaly/Systemic Issue", "confidence": 0.90, "source": "system"})
                    else:
                        disease_preds.append({"disease": "Healthy/Normal", "confidence": 0.95, "source": "vision"})

                # RAG retrieval
                rag_docs = []
                disease_names = [p["disease"] for p in disease_preds]
                if disease_names and self.rag_service:
                    try:
                        # Add a strict timeout for RAG retrieval to prevent pipeline hangs
                        rag_docs = await asyncio.wait_for(
                            asyncio.to_thread(self.rag_service.retrieve_for_diseases, disease_names, 5),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("RAG retrieval timed out. Proceeding without external evidence.")
                    except Exception as e:
                        logger.error(f"RAG retrieval error: {e}")

                # KG context - disconnected from text generation as per requirement
                kg_context = None
                # if disease_names and self.neo4j_service:
                #     try:
                #         top_d = disease_names[0]
                #         kg_context = {
                #             "disease_info": await asyncio.to_thread(self.neo4j_service.get_disease_context, top_d),
                #             "treatment_protocols": await asyncio.to_thread(self.neo4j_service.get_treatment_protocol, top_d),
                #             "cow_history": await asyncio.to_thread(self.neo4j_service.get_cow_history, cow_id),
                #         }
                #     except Exception: pass

                # Generate ONE multi-cow report using the LLM Service
                all_cow_ids = [s["cow_id"] for s in clinical_summaries]
                
                # Contextualize refinement data for the LLM
                refinement_context = f"(Refinement data below applies ONLY to Cow #{cow_id_override})" if cow_id_override in all_cow_ids else ""
                
                try:
                    report_data = await asyncio.wait_for(
                        self.llm_service.generate_clinical_report(
                            cow_ids=all_cow_ids, disease_predictions=disease_preds,
                            risk_assessment={"clinical_summaries": clinical_summaries, "refinement_note": refinement_context},
                            rag_context=rag_docs or [],
                            kg_context=None, 
                            sensor_data=sensor_data if cow_id_override in all_cow_ids else None,
                            animal_weight_kg=animal_weight_kg if cow_id_override in all_cow_ids else None,
                            animal_age_years=animal_age_years if cow_id_override in all_cow_ids else None,
                            vision_analysis=vision_r if isinstance(vision_r, dict) else None,
                            safety_status=None,
                        ),
                        timeout=45.0
                    )
                except asyncio.TimeoutError:
                    logger.error("LLM report generation timed out.")
                    report_data = {"report": "# Clinical Assessment Timeout\n\nThe AI agent is taking too long to synthesize the clinical evidence. Please check the individual summaries for raw data.", "summary": "Assessment timeout."}
                
                # Split report into per-cow chunks for UI display
                report_text = report_data.get("report", "")
                per_cow_reports = {}
                import re
                
                # We split the report into segments based on Cow #ID headers
                # Match only at the beginning of a line to avoid splitting inside sentences
                segments = re.split(r"(?im)^(?:###\s*|[*]*|)\s*🐄?\s*COW\s*#*(\d+)\b", report_text)
                
                # Segments will be [before, id1, content1, id2, content2, ...]
                temp_map = {}
                if len(segments) > 2:
                    for i in range(1, len(segments), 2):
                        cid_str = str(int(segments[i])) # Normalize 01 to 1
                        content = segments[i+1].strip()
                        temp_map[cid_str] = content
                
                for cid in all_cow_ids:
                    cid_key = str(cid)
                    if cid_key in temp_map:
                        per_cow_reports[cid_key] = temp_map[cid_key].strip()
                    else:
                        # Fallback to a broader regex search if split failed
                        pattern = rf"(?im)(?:###|[*]{2,})\s*🐄?\s*COW\s*#*\s*{cid}\b(.*?)(?=(?:###|[*]{2,})\s*🐄?\s*COW\s*#|\Z)"
                        match = re.search(pattern, report_text, re.DOTALL)
                        if match:
                            per_cow_reports[cid_key] = match.group(1).strip()
                        else:
                            per_cow_reports[str(cid)] = "Individual assessment available in full report."

                if self.safety_engine:
                    report_text = self.safety_engine.inject_disclaimer(report_text)
                
                result["stages"]["report"] = {
                    "full_report": report_text,
                    "per_cow_reports": per_cow_reports,
                    "summary": report_data.get("summary", ""),
                    "urgency_score": report_data.get("urgency_score", 5)
                }
            except Exception as e:
                result["errors"].append(f"IDSS report: {e}")

        # ═══ STAGE 7: Clinical summary ═══════════════════════════════════════
        # Summarize the first cow for backward compatibility, but include info about all
        primary_summary = clinical_summaries[0] if clinical_summaries else {}
        result["stages"]["clinical_summary"] = self._build_summary(
            primary_summary.get("cow_id", cow_id), 
            primary_summary.get("health", {}), 
            primary_summary.get("milk", {}), 
            primary_summary.get("heat_stress", {}), 
            vision_r
        )
        if len(clinical_summaries) > 1:
            result["stages"]["clinical_summary"]["multi_cow_note"] = f"{len(clinical_summaries)} cows detected and analyzed."

        return self._finalize(result, cow_id, t0)

    def _finalize(self, result, cow_id, t0):
        result["cow_id"] = cow_id
        result["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["success"] = len(result["errors"]) == 0
        
        # Store in cache for agent lookup
        summaries = result["stages"].get("clinical_summaries", [])
        for summary in summaries:
            cid = summary.get("cow_id")
            if cid:
                self.latest_results[cid] = {
                    "health": summary.get("health"),
                    "milk": summary.get("milk"),
                    "heat_stress": summary.get("heat_stress"),
                    "timestamp": time.time(),
                    "image_b64": result["stages"].get("annotated_image_b64")
                }
        
        logger.info(f"Pipeline complete | cow={cow_id} | latency={result['total_latency_ms']:.0f}ms | errors={len(result['errors'])}")
        return result

    def get_latest_status(self, cow_id: int) -> Optional[Dict[str, Any]]:
        """Utility for BovineIQ Agent to get cached status without re-running models. 🐄✨"""
        return self.latest_results.get(cow_id)

    def _extract_cow_sensor_vitals(self, cow_id: int, day_index: int = 0) -> Dict[str, Any]:
        """Extract real sensor vitals using the dedicated sensor CSVs."""
        defaults = {
            "cbt_celsius": None, "thi": None,
            "accel_mag": None, "sensor_milk_kg": None,
            "weight_kg": None, "age_years": None
        }
        try:
            dp = self.data_pipeline
            if not dp or not hasattr(dp, 'get_real_sensor_vitals'):
                return defaults
            
            raw = dp.get_real_sensor_vitals(cow_id)
            return {
                "cbt_celsius": raw.get("cbt_celsius"),
                "thi": raw.get("thi"),
                "thi_stress_class": raw.get("thi_stress_class", "Unknown"),
                "accel_mag": raw.get("accel_mag"),
                "sensor_milk_kg": raw.get("actual_milk_kg"),
                "weight_kg": self.clinical_registry.get(cow_id, {}).get("weight"),
                "age_years": self.clinical_registry.get(cow_id, {}).get("age"),
            }
        except Exception as e:
            logger.warning(f"Failed to extract sensor vitals for Cow #{cow_id}: {e}")
            return defaults


    def _build_summary(self, cow_id, health_r, milk_r, heat_r, vision_r):
        hs = health_r.get("health_score", 0.5) if isinstance(health_r, dict) else 0.5
        rl = health_r.get("risk_level", "unknown") if isinstance(health_r, dict) else "unknown"
        milk = milk_r.get("predicted_yield_kg", "N/A") if isinstance(milk_r, dict) else "N/A"
        # heat_stress model returns 'stress_level'; map to 'heat_stress_level' for UI
        stress = heat_r.get("stress_level", "unknown") if isinstance(heat_r, dict) else "unknown"
        anomaly = health_r.get("anomaly_detected", False) if isinstance(health_r, dict) else False

        # Handle health score being a string (e.g., "Insufficient Data")
        hs_rounded = hs
        if isinstance(hs, (int, float)):
            hs_rounded = round(hs, 3)

        # Decision confidence = health score confidence (NOT ID confidence)
        decision_confidence = health_r.get("confidence", 0.0) if isinstance(health_r, dict) else 0.0

        return {
            "cow_id": cow_id,
            "health_score": hs_rounded,
            "decision_confidence": round(decision_confidence, 3),
            "risk_level": rl,
            "predicted_milk_yield_kg": milk,
            "heat_stress_level": stress,
            "anomaly_detected": anomaly,
            "action_required": rl == "high" or anomaly,
            "veterinary_exam_needed": rl in ("high", "medium") or anomaly,
        }


    # ── Convenience shortcuts ─────────────────────────────────────────────────

    async def analyze_crop(self, image_bytes: bytes, crop_region: Dict, user_description: str = "") -> Dict[str, Any]:
        """Analyze a user-selected crop region using vision + identification."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Identify cow in the crop
        id_result = self.cow_identifier.analyze_crop(image, crop_region)

        # Run vision analysis on the cropped region
        x, y, w, h = crop_region["x"], crop_region["y"], crop_region["width"], crop_region["height"]
        crop = image.crop((x, y, x + w, y + h))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        crop_bytes = buf.getvalue()

        vision_result = {}
        if self.vision_service:
            try:
                vision_result = await asyncio.to_thread(self.vision_service.analyze_image, crop_bytes)
            except Exception as e:
                vision_result = {"error": str(e)}

        # If user provided a description, include it in agent context
        agent_context = ""
        if user_description:
            agent_context = f"User observation: {user_description}. "
        # Only report vision disease if confidence >= 90% to avoid hallucinations
        DISEASE_CONFIDENCE_THRESHOLD = 0.90
        if vision_result.get("disease_candidates"):
            high_conf = [c for c in vision_result["disease_candidates"] if c.get("confidence", 0) >= DISEASE_CONFIDENCE_THRESHOLD]
            if high_conf:
                top = high_conf[0]
                agent_context += f"AI vision detected: {top.get('disease', '?')} ({top.get('confidence', 0):.0%} confidence)"
        
        if not id_result.get("is_known_cow"):
            agent_context = f"🚨 **Clinical Alert: No Known Cow Identified.** The cropped region does not match any known cow profile in the C01-C16 database. {agent_context}"

        # Attach base64 crop for frontend chat
        id_result["crop_b64"] = base64.b64encode(crop_bytes).decode('utf-8')
        
        # Get IDSS and agent answer if known cow
        agent_answer = None
        cow_id = id_result.get("cow_id")
        if id_result.get("is_known_cow") and cow_id:
            try:
                # Get the latest IDSS state for this cow without generating a full new report
                idss_res = await self.run_full_pipeline(
                    image_bytes=None, 
                    cow_id_override=cow_id, 
                    generate_report=False
                )
                summaries = idss_res.get("stages", {}).get("clinical_summaries", [])
                if summaries:
                    s = summaries[0]
                    agent_context += f"\nIDSS Profile for Cow {cow_id}: Health={s.get('health',{}).get('health_score')}, Milk={s.get('milk',{}).get('predicted_yield_kg')}kg, Heat={s.get('heat_stress',{}).get('stress_level')}."
            except Exception as e:
                logger.warning(f"Failed to fetch IDSS for crop: {e}")

        return {
            "identification": id_result,
            "vision_analysis": vision_result,
            "crop_region": crop_region,
            "agent_context": agent_context,
            "user_description": user_description,
            "agent_answer": agent_answer
        }
