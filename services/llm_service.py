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
        "You are an expert veterinary AI clinical decision support system (IDSS) "
        "specialising in bovine (cattle) medicine. You provide evidence-based clinical "
        "decision support to licensed veterinarians and trained farmers.\n\n"

        "CLINICAL KNOWLEDGE STANDARDS:\n"
        "• Body Temperature normal range: 38.0–39.3°C. Fever: >39.5°C. Hyperthermia: >41.0°C.\n"
        "• Heart Rate normal: 40–80 bpm. Respiratory Rate: 10–30 breaths/min.\n"
        "• BCS (Edmonson 1–5): BCS <2.5 = undernutrition; BCS >4.0 = obesity risk.\n"
        "• FAMACHA Score 1–5 (anaemia): ≥3 → anthelmintic treatment.\n"
        "• SCC: <200k cells/mL = healthy; 200–400k = subclinical mastitis; >400k = clinical mastitis.\n"
        "• Milk conductivity: 4–6 mS/cm normal; >7.5 mS/cm = mastitis indicator.\n"
        "• Rumination: 7–10 h/day normal; <4 h/day = rumen dysfunction.\n"
        "• Ketosis: BHBA >1.2 mmol/L (subclinical), >3.0 mmol/L (clinical).\n\n"

        "WEIGHT-BASED DOSING PROTOCOLS (use these to calculate doses when weight is provided):\n"
        "• Penicillin G: 22,000 IU/kg IM BID\n"
        "• Oxytetracycline LA: 11 mg/kg IM single dose\n"
        "• Florfenicol: 20 mg/kg SC or 40 mg/kg IM single dose (BRD)\n"
        "• Flunixin meglumine: 2.2 mg/kg IV/IM SID (max 5 days)\n"
        "• Meloxicam: 0.5 mg/kg SC/IV SID\n"
        "• Dexamethasone: 0.1–0.2 mg/kg IM/IV (acute inflammation)\n"
        "• Thiamine: 10 mg/kg IV slowly then 5 mg/kg IM QID (polioencephalomalacia)\n"
        "• Propylene glycol: 300 mL PO BID (ketosis, fixed dose)\n"
        "• Calcium borogluconate: 400 mL 40% solution slow IV (milk fever, fixed)\n"
        "• Magnesium sulphate: 200 mL 50% SC (hypomagnesaemia, fixed)\n"
        "• Intramammary tubes: 1 tube/affected quarter (mastitis, NOT weight-based)\n"
        "• Milk withdrawal — Penicillin: 3d; OTC LA: 0d; Flunixin: 36h; Meloxicam: min 3d\n"
        "• Meat withdrawal — Penicillin: 28d; OTC LA: 28d; Flunixin: 4d; Florfenicol: 28d\n\n"

        "MANDATORY CLINICAL RULES:\n"
        "0. ABSOLUTE DOSING BLOCK: If the prompt indicates that 'Dosing is PROHIBITED' or 'safety_level' is 'blocked', "
        "YOU MUST NOT mention any medication names, drug classes, or dosages. Focus ONLY on diagnostics and biosecurity.\n"
        "1. HIGH/CRITICAL findings → recommend immediate vet examination within 24h.\n"
        "2. When animal weight is provided AND dosing is allowed, CALCULATE AND SHOW the actual dose with arithmetic "
        "(e.g. 'Flunixin: 2.2 mg/kg × 600 kg = 1,320 mg'). Include withdrawal periods.\n"
        "3. NOTIFIABLE diseases (FMD, LSD, BSE) → immediate authority notification MANDATORY.\n"
        "4. ZOONOTIC diseases → BIOSECURITY ALERT for farm workers (PPE, hygiene).\n"
        "5. Cite evidence level: A=RCT/meta-analysis, B=observational, C=expert opinion.\n"
        "6. If visual analysis (Groq Vision) is provided, it is the PRIMARY diagnostic evidence.\n"
        "7. Always end with: 'This is AI-generated decision support. "
        "Final decisions rest with the attending veterinarian.'\n"
        "8. Include Clinical Urgency Score (1–10, 10=immediate life threat).\n"
        "9. Format output in clear Markdown with clinical headings."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
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
        cow_id: int,
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
            cow_id=cow_id,
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
            report_text = self._fallback_report(cow_id, disease_predictions, risk_assessment)

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
    ) -> str:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if context:
            messages.append({
                "role":    "system",
                "content": f"Use the following veterinary evidence to answer:\n\n{context}",
            })
        messages.append({"role": "user", "content": question})
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or "Unable to generate response."
        except Exception as exc:
            logger.error(f"Groq API error: {exc}", exc_info=True)
            return f"LLM service temporarily unavailable. Error: {exc}"

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_idss_prompt(
        self,
        cow_id: int,
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

        # ── Dosing safety logic ──────────────────────────────────────────────
        dosing_section = ""
        allow_dosing = (safety_status or {}).get("allow_dosing", True)
        safety_level = (safety_status or {}).get("safety_level", "pass")

        if animal_weight_kg and animal_weight_kg > 0 and allow_dosing:
            w = animal_weight_kg
            dosing_section = (
                f"\n## PATIENT WEIGHT & DOSING CONTEXT\n"
                f"**Animal Weight:** {w:.0f} kg\n"
                f"**Pre-calculated Reference Doses for {w:.0f} kg animal:**\n"
                f"  • Flunixin meglumine (NSAID): 2.2 × {w:.0f} = **{2.2*w:.0f} mg** IV/IM SID\n"
                f"  • Meloxicam (NSAID): 0.5 × {w:.0f} = **{0.5*w:.0f} mg** SC/IV SID\n"
                f"  • Penicillin G: 22,000 IU × {w:.0f} = **{22000*w/1000000:.1f} MIU** IM BID\n"
                f"  • Oxytetracycline LA: 11 × {w:.0f} = **{11*w:.0f} mg** IM single dose\n"
                f"  • Florfenicol (SC): 20 × {w:.0f} = **{20*w:.0f} mg** SC; (IM): {40*w:.0f} mg\n"
                f"  • Thiamine: 10 × {w:.0f} = **{10*w:.0f} mg** IV (polioencephalomalacia)\n"
                f"  • Dexamethasone: 0.1–0.2 × {w:.0f} = **{0.1*w:.0f}–{0.2*w:.0f} mg** IM/IV\n"
                f"  *(Withdrawal periods apply — verify specific product label)*\n"
                f"  *(Intramammary mastitis tubes: 1 per affected quarter — not weight-based)*"
            )
        elif not allow_dosing:
            dosing_section = (
                f"\n## PATIENT WEIGHT\n"
                f"**Animal Weight:** {animal_weight_kg or 'Not provided'} kg\n"
                f"**CRITICAL SAFETY BLOCK**: Dosing recommendations are PROHIBITED for this case "
                f"due to { 'a notifiable disease suspicion' if (safety_status or {}).get('notifiable_diseases') else 'high clinical uncertainty' }.\n"
                f"DO NOT mention any drugs, medication names, or dosages in the report."
            )

        # ── Risk section ──────────────────────────────────────────────────────
        risk_factors = "\n".join(
            f"  - **{f['feature'].replace('_',' ').title()}**: {f['current_value']} "
            f"[{f['status'].upper()}] (attribution: {f['attribution_score']:.2f})"
            for f in risk_assessment.get("top_risk_factors", [])
        ) or "  No sensor data provided."

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
            f"Generate an expert-level IDSS veterinary clinical report for the following bovine case.\n\n"
            f"## PATIENT\n"
            f"- **Cow ID**: {cow_id}\n"
            f"- **Age**: {f'{animal_age_years:.1f} years' if animal_age_years else 'Not provided'}\n"
            f"- **Weight**: {f'{animal_weight_kg:.0f} kg' if animal_weight_kg else 'Not provided'}\n\n"
            f"{vision_section}\n\n"
            f"## AI MODEL DISEASE CLASSIFICATIONS (supplement to visual analysis)\n"
            f"{disease_section}\n\n"
            f"## RISK ASSESSMENT\n"
            f"- **Overall Risk Score**: {risk_assessment.get('overall_risk_score', 0):.2f}/1.0\n"
            f"- **Risk Level**: {risk_assessment.get('risk_level', 'unknown').upper()}\n"
            f"- **Uncertainty**: {risk_assessment.get('risk_uncertainty', 0):.4f}\n\n"
            f"## SENSOR RISK FACTORS\n{risk_factors}\n"
            f"{sensor_section}\n"
            f"{dosing_section}\n"
            f"{treatment_section}\n"
            f"{progression_section}\n"
            f"{alert_section}\n"
            f"{history_section}\n\n"
            f"## RETRIEVED PEER-REVIEWED EVIDENCE (PubMed + Neo4j, 2021-2026)\n"
            f"{evidence_section}\n\n"
            f"## REPORT REQUIREMENTS\n"
            f"Generate a complete expert IDSS clinical report with ALL sections:\n"
            f"### 1. Executive Summary (2–3 sentences + Clinical Urgency Score /10)\n"
            f"### 2. Primary Diagnosis (based on VISUAL ANALYSIS as primary evidence)\n"
            f"   - State the most likely diagnosis based on what is VISUALLY OBSERVED\n"
            f"   - Explain why the visual evidence supports this diagnosis\n"
            f"### 3. Clinical Findings Interpretation (visual + sensor)\n"
            f"### 4. Differential Diagnoses (ranked by likelihood)\n"
            f"### 5. Recommended Diagnostic Tests\n"
            f"### 6. Treatment Recommendations\n"
            f"   - **Immediate (0–24h)**\n"
            f"   - If animal weight provided: show CALCULATED DOSES with arithmetic\n"
            f"   - Include withdrawal periods for milk and meat\n"
            f"   - **Short-term management (1–7 days)**\n"
            f"   - **Monitoring schedule**\n"
            f"### 7. Evidence Base (cite [1][2][3] from retrieved literature)\n"
            f"### 8. Biosecurity & Regulatory Notes\n"
            f"### 9. Safety Disclaimer\n\n"
            f"Format in Markdown. Be clinically precise, evidence-based, and actionable."
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
