"""
services/llm_service.py
========================
Groq LLM service (Llama 3.3 70B) for expert veterinary clinical report generation.

v3 improvements:
  - Accepts animal_weight_kg → calculated weight-based dosing in reports
  - Accepts vision_analysis → incorporates visual observations as primary evidence
  - Expert system prompt with dosing protocols table
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GroqLLMService:
    """
    Wrapper around the Groq API (groq-python SDK).
    Uses Llama 3.3 70B Versatile for clinical decision support text.

    Key features
    ------------
    • Low temperature (0.1) for deterministic, medically-safe outputs
    • Expert system prompt with BCS, FAMACHA, SCC, drug protocol knowledge
    • Weight-based calculated dosing when animal_weight_kg is provided
    • Vision analysis observations used as primary disease evidence
    • Thread-safe (client is stateless)
    """

    SYSTEM_PROMPT = (
        "You are the Senior Veterinary AI Assistant, an expert in bovine clinical diagnostics. "
        "Your mission is to provide structured, ID-specific clinical decision support.\n\n"

    "MANDATORY REPORT STRUCTURE:\n"
    "You MUST PROVIDE ONLY individual assessment blocks for each cow ID. DO NOT provide a general executive summary, herd summary, or evidence section at the top.\n\n"
    "# INDIVIDUAL CASE ASSESSMENTS\n\n"
    "### 🐄 COW #[ID]\n"
    "- **Status**: [Health Score % / Risk Level]\n"
    "- **Clinical Vitals**: [Weight, Age, Temp, Heart Rate]\n"
    "- **Production Analysis**: [Milk Yield interpretation]\n"
    "- **Diagnosis**: [Suspected condition based on 25%+ confidence. IMPORTANT: If 'anomaly_detected' is True or Risk Level is High, you MUST NOT state 'Healthy/Normal'. Instead, report the findings as a suspected clinical anomaly or specific disease.]\n"
    "- **Management Plan**: [Specific actions for THIS cow only]\n"
    "--- [End of block for this ID] ---\n\n"

    "RULES:\n"
    "1. DO NOT use 'BovineIQ' branding.\n"
    "2. DO NOT show any literature references, PubMed IDs, or 'Evidence Base' sections.\n"
    "3. **STRICT CLINICAL GATE**: Only mention a specific disease if the confidence is **25% or higher**. However, if 'anomaly_detected' is True, you must acknowledge the anomaly regardless of specific disease confidence.\n"
    "4. **FACTUAL VITALS ONLY**: DO NOT invent vitals. Use 'Not provided' if missing.\n"
    "5. **RISK ALIGNMENT**: If the provided risk level is 'High' or an anomaly is detected, YOUR REPORT MUST REFLECT A CLINICAL CONCERN. DO NOT report the animal as healthy in these cases.\n"
    "6. Always end each cow block with: 'AI-generated support. Final decisions by licensed veterinarians.'\n"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"GroqLLMService initialised | model={model}")
        except ImportError:
            raise RuntimeError("groq package not installed. Run: pip install groq")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    # ── Primary interface ─────────────────────────────────────────────────────

    def generate_clinical_report(
        self,
        cow_ids: List[int],
        disease_predictions: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any],
        rag_context: List[Dict[str, str]],
        kg_context: Optional[Dict[str, Any]] = None,
        sensor_data: Optional[Dict[str, float]] = None,
        animal_weight_kg: Optional[float] = None,
        animal_age_years: Optional[float] = None,
        vision_analysis: Optional[Dict[str, Any]] = None,
        safety_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Generate a structured IDSS clinical report using all available context."""
        prompt = self._build_idss_prompt(
            cow_ids=cow_ids,
            disease_predictions=disease_predictions,
            risk_assessment=risk_assessment,
            rag_context=rag_context,
            kg_context=kg_context,
            sensor_data=sensor_data,
            animal_weight_kg=animal_weight_kg,
            animal_age_years=animal_age_years,
            vision_analysis=vision_analysis,
            safety_status=safety_status,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            report_text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"Groq API error: {exc}", exc_info=True)
            report_text = self._fallback_report(cow_ids[0] if cow_ids else 0, disease_predictions, risk_assessment)

        summary       = self._extract_summary(report_text)
        urgency_score = self._extract_urgency_score(
            report_text, risk_assessment, disease_predictions
        )

        return {
            "report":        report_text,
            "summary":       summary,
            "llm_model":     self.model,
            "urgency_score": urgency_score,
        }

    def answer_clinical_question(
        self,
        question: str,
        context: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> str:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if context:
            messages.append({
                "role":    "system",
                "content": f"Use the following veterinary evidence to answer:\n\n{context}",
            })

        user_content = []
        if image_b64:
            # Ensure proper prefix
            if not image_b64.startswith("data:image"):
                image_b64 = f"data:image/jpeg;base64,{image_b64}"
            
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_b64}
            })
            user_content.append({
                "type": "text",
                "text": f"Study the attached cow image/crop and answer this clinical question: {question}"
            })
        else:
            user_content = question

        messages.append({"role": "user", "content": user_content})

        try:
            # For vision, we use Llama 3.2 11B or 90B Vision if specified, 
            # otherwise the default 70B Versatile (which might not support vision depending on the endpoint).
            # Groq's 70B Versatile is text-only. 
            # We must use a vision model for vision requests.
            model_to_use = self.model
            if image_b64:
                model_to_use = "llama-3.2-11b-vision-preview"

            response = self._client.chat.completions.create(
                model=model_to_use, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or "Unable to generate response."
        except Exception as exc:
            logger.error(f"Groq API error: {exc}", exc_info=True)
            return f"LLM service temporarily unavailable. Error: {exc}"

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_idss_prompt(
        self,
        cow_ids: List[int],
        disease_predictions: List[Dict],
        risk_assessment: Dict,
        rag_context: List[Dict],
        kg_context: Optional[Dict],
        sensor_data: Optional[Dict],
        animal_weight_kg: Optional[float] = None,
        animal_age_years: Optional[float] = None,
        vision_analysis: Optional[Dict] = None,
        safety_status: Optional[Dict] = None,
    ) -> str:

        # ── Vision analysis section (PRIMARY diagnostic evidence) ────────────
        vision_section = ""
        if vision_analysis and not vision_analysis.get("error"):
            obs = vision_analysis.get("visual_observations", [])
            cands = vision_analysis.get("disease_candidates", [])
            bcs   = vision_analysis.get("body_condition_score")
            assessment = vision_analysis.get("overall_health_assessment", "")
            vision_model = vision_analysis.get("vision_model", "Groq Vision")

            obs_text   = "\n".join(f"  • {o}" for o in obs) or "  • No specific observations noted"
            cands_text = "\n".join(
                f"  {i+1}. **{c.get('disease','?').replace('_',' ').title()}** "
                f"({c.get('confidence',0):.0%}) — {c.get('reasoning','')}"
                for i, c in enumerate(cands[:3])
            ) or "  No disease candidates identified (cow appears healthy)"

            vision_section = (
                f"\n## VISUAL ANALYSIS (Groq Vision: {vision_model}) — PRIMARY EVIDENCE\n"
                f"**Overall Assessment:** {assessment}\n"
                f"**BCS:** {bcs if bcs else 'Not assessed'}/5\n"
                f"**Visual Urgency:** {vision_analysis.get('clinical_urgency', '?')}/10\n\n"
                f"**Observations:**\n{obs_text}\n\n"
                f"**Vision Disease Candidates (use these as PRIMARY diagnosis basis):**\n{cands_text}"
            )

        # ── AI model disease classifications ──────────────────────────────────
        if disease_predictions:
            disease_section = "\n".join(
                f"  {i+1}. **{p['disease'].replace('_',' ').title()}** "
                f"(merged confidence: {p['confidence']:.1%}, source: {p.get('source', 'model')})"
                for i, p in enumerate(disease_predictions)
            )
        else:
            disease_section = "  No significant findings above threshold."

        # ── Multi-Cow Clinical Summaries ──────────────────────────────────────
        clinical_summaries_text = ""
        summaries = risk_assessment.get("clinical_summaries", [])
        ref_note = risk_assessment.get("refinement_note", "")
        
        allow_dosing = (safety_status or {}).get("allow_dosing", True)

        if summaries:
            clinical_summaries_text = "## INDIVIDUAL COW CLINICAL SUMMARIES\n"
            for s in summaries:
                cid = s.get("cow_id")
                hs = s.get("health", {}).get("health_score", "N/A")
                if isinstance(hs, float): hs = f"{hs:.2f}"
                risk = s.get("health", {}).get("risk_level", "Unknown").upper()
                milk = s.get("milk", {}).get("predicted_yield_kg", "N/A")
                heat = s.get("heat_stress", {}).get("heat_stress_level", "N/A")
                
                # Check if this cow has refined vitals
                is_refined = f"#{cid}" in ref_note
                w = animal_weight_kg if is_refined else None
                a = animal_age_years if is_refined else None
                
                vitals_text = ""
                if w or a:
                    vitals_text = f"  - **REFINED VITALS**: {f'{w}kg' if w else ''} {f'{a}y' if a else ''} (Manual Entry)\n"
                    if w and allow_dosing:
                        vitals_text += (
                            f"  - **PRE-CALCULATED DOSES ({w}kg)**: \n"
                            f"    • Flunixin: {2.2*w:.0f}mg | Meloxicam: {0.5*w:.0f}mg | Penicillin: {22000*w/1000000:.1f}MIU\n"
                        )

                clinical_summaries_text += (
                    f"### Cow #{cid}\n"
                    f"- **Health Status**: {hs} ({risk} RISK)\n"
                    f"{vitals_text}"
                    f"- **Production**: {milk} kg/day\n"
                    f"- **Environment**: {heat} Heat Stress\n\n"
                )
        else:
            # Fallback for old risk_assessment
            risk_factors = "\n".join(
                f"  - **{f['feature'].replace('_',' ').title()}**: {f['current_value']} "
                f"[{f['status'].upper()}] (attribution: {f['attribution_score']:.2f})"
                for f in risk_assessment.get("top_risk_factors", [])
            ) or "  No sensor data provided."
            clinical_summaries_text = f"## RISK ASSESSMENT\n- **Overall Risk Score**: {risk_assessment.get('overall_risk_score', 0):.2f}/1.0\n- **Risk Level**: {risk_assessment.get('risk_level', 'unknown').upper()}\n## SENSOR RISK FACTORS\n{risk_factors}\n"

        # ── KG treatment protocols ────────────────────────────────────────────
        treatment_section = ""
        if kg_context and kg_context.get("treatment_protocols"):
            lines = []
            for t in kg_context["treatment_protocols"][:3]:
                line = (
                    f"  - **{t.get('treatment','?')}** [Evidence {t.get('evidence_level','?')}]: "
                    f"{(t.get('protocol') or '')[:200]}…"
                )
                if t.get("withdrawal_milk_days", 0):
                    line += f" | Milk WD: {t['withdrawal_milk_days']}d"
                lines.append(line)
            treatment_section = "\n## KG TREATMENT PROTOCOLS\n" + "\n".join(lines)

        # ── Alerts ────────────────────────────────────────────────────────────
        alert_section = ""
        if kg_context and kg_context.get("zoonotic_alerts"):
            alert_lines = []
            for a in kg_context["zoonotic_alerts"]:
                if a.get("notifiable"):
                    alert_lines.append(
                        f"  NOTIFIABLE DISEASE: {a['disease']} — Immediate authority notification required."
                    )
                if a.get("zoonotic"):
                    alert_lines.append(
                        f"  ZOONOTIC RISK: {a['disease']} — PPE required for farm workers."
                    )
            if alert_lines:
                alert_section = "\n## REGULATORY & BIOSECURITY ALERTS\n" + "\n".join(alert_lines)

        # ── Progression risks ─────────────────────────────────────────────────
        progression_section = ""
        if kg_context and kg_context.get("progression_risks"):
            prog_lines = [
                f"  - May progress to **{p['progresses_to']}** "
                f"(probability {p['probability']:.0%}, within {p['time_days']}d)"
                for p in kg_context["progression_risks"][:3] if p.get("time_days")
            ]
            if prog_lines:
                progression_section = "\n## DISEASE PROGRESSION RISK\n" + "\n".join(prog_lines)

        # ── RAG evidence ──────────────────────────────────────────────────────
        evidence_section = "\n\n".join(
            f"**[{i+1}]** *{doc.get('title','')}* "
            f"({doc.get('year','?')}, {doc.get('source','PubMed')})\n"
            f"{doc.get('snippet','')}"
            for i, doc in enumerate(rag_context[:4])
        ) or "No directly relevant peer-reviewed evidence retrieved."

        # ── Cow history ───────────────────────────────────────────────────────
        history_section = ""
        if kg_context and kg_context.get("cow_history"):
            history_section = "\n## COW MEDICAL HISTORY\n" + "\n".join(
                f"  - {h.get('disease','?').replace('_',' ').title()} "
                f"({h.get('confidence',0):.0%}) — {str(h.get('timestamp','?'))[:10]}"
                for h in kg_context["cow_history"][:5]
            )

        # ── Sensor readings ───────────────────────────────────────────────────
        sensor_section = ""
        if sensor_data:
            NORMAL = {
                "body_temp": (38.0, 39.3), "heart_rate": (40, 80),
                "respiratory_rate": (10, 30), "rumination_time": (420, 600),
                "milk_yield": (10, 45), "milk_conductivity": (4.0, 7.5),
            }
            lines = []
            for k, v in sensor_data.items():
                r = NORMAL.get(k)
                flag = " OUT OF RANGE" if r and not (r[0] <= v <= r[1]) else ""
                lines.append(f"  {k.replace('_',' ').title()}: {v}{flag}")
            sensor_section = "\n## SENSOR READINGS\n" + "\n".join(lines)

        return (
            f"Generate an expert-level IDSS veterinary clinical report for the following bovine cases detected in the scene.\n\n"
            f"## PATIENTS DETECTED\n"
            f"- **Cow IDs**: {', '.join(f'#{cid}' for cid in cow_ids)}\n"
            f"- **Refinement Info**: {risk_assessment.get('refinement_note', 'No manual refinement data provided.')}\n\n"
            f"{vision_section}\n\n"
            f"## AI MODEL DISEASE CLASSIFICATIONS (supplement to visual analysis)\n"
            f"{disease_section}\n\n"
            f"{clinical_summaries_text}\n"
            f"{sensor_section}\n"
            f"{treatment_section}\n"
            f"{progression_section}\n"
            f"{alert_section}\n"
            f"{history_section}\n\n"
            f"## REPORT REQUIREMENTS\n"
            f"Generate ONLY the per-cow clinical assessments. DO NOT include a general herd summary or references.\n"
            f"Each cow block MUST include:\n"
            f"- Health status & risk\n"
            f"- Clinical vitals interpretation\n"
            f"- Diagnosis (ONLY if 50%+ confidence, else 'Healthy/Normal')\n"
            f"- Specific management plan including weight-based doses (if weight provided)\n\n"
            f"Format as Markdown."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_summary(self, report: str) -> str:
        lines = report.split("\n")
        for i, line in enumerate(lines):
            if "executive summary" in line.lower() or "summary" in line.lower():
                summary_lines = []
                for j in range(i + 1, min(i + 10, len(lines))):
                    stripped = lines[j].strip()
                    if stripped and not stripped.startswith("#"):
                        summary_lines.append(stripped)
                    elif stripped.startswith("#") and summary_lines:
                        break
                if summary_lines:
                    return " ".join(summary_lines)
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 40:
                return stripped
        return "Clinical assessment generated. See full report."

    def _extract_urgency_score(
        self,
        report: str,
        risk_assessment: Dict,
        disease_predictions: List[Dict],
    ) -> int:
        risk_level = risk_assessment.get("risk_level", "low")
        base = {"high": 7, "medium": 4, "low": 2, "unknown": 3}.get(risk_level, 3)
        HIGH_SEVERITY = {"oral_lesion", "neurological_sign", "skin_nodules", "abdominal_pain"}
        disease_names = {p["disease"] for p in disease_predictions}
        if disease_names & HIGH_SEVERITY:
            base = min(base + 2, 10)
        max_conf = max((p["confidence"] for p in disease_predictions), default=0)
        if max_conf > 0.80:
            base = min(base + 1, 10)
        return min(base, 10)

    @staticmethod
    def _fallback_report(
        cow_id: int,
        disease_predictions: List[Dict],
        risk_assessment: Dict,
    ) -> str:
        top  = disease_predictions[0]["disease"] if disease_predictions else "No findings"
        risk = risk_assessment.get("risk_level", "unknown").upper()
        return (
            f"# Clinical Assessment — Cow #{cow_id}\n\n"
            f"**Risk Level:** {risk}\n"
            f"**Primary Finding:** {top.replace('_', ' ').title()}\n\n"
            f"*LLM service temporarily unavailable. "
            f"Consult a veterinarian for full clinical assessment.*\n\n"
            "**Safety Disclaimer:** This is AI-generated decision support. "
            "Final decisions rest with the attending veterinarian."
        )
