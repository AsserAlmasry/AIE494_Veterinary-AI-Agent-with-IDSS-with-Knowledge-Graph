"""
services/llm_service.py
========================
Stable implementation for Groq LLM services with legacy sync support for pipelines.
Fixed to use REAL per-cow sensor vitals, not hardcoded defaults.
"""

from __future__ import annotations
import logging
import httpx
import json
import asyncio
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class GroqLLMService:
    """
    Expert Veterinary AI Clinical Report Service.
    Handles clinical synthesis and agent interactions. 🐄✨
    """

    SYSTEM_PROMPT = (
        "You are the **Veterinary AI Agent**, a senior clinical diagnostic system specializing in bovine health. 🐄✨ "
        "Your mission is to provide engaging, highly structured, and empathetic clinical decision support.\n\n"
        "MANDATORY REPORT STRUCTURE FOR EACH COW:\n"
        "### COW #[ID]\n"
        "- **Status**: [Health Score % / Risk Level] ✨\n"
        "- **Clinical Vitals**: Use ONLY the real sensor values provided in the data below. "
        "Report Core Body Temp (CBT) from sensors, milk yield from models, and THI from sensors. "
        "If a value is truly unavailable, say 'Sensor offline' NOT a generic default. 🌡️\n"
        "- **Diagnosis**: [Suspected condition based on model outputs. If anomaly_detected is True, report it as a clinical concern.] 🔬\n"
        "- **Management Plan**: [Specific actions for THIS cow based on its actual sensor readings] 💊\n\n"
        "CRITICAL RULE: Never write '(Default)' next to a vital sign. Use ONLY the sensor values provided.\n"
        "AI-generated support. Final decisions by licensed veterinarians. 🏥\n"
    )

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1, max_tokens: int = 2048) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"GroqLLMService ready | model={model}")

    async def generate_clinical_report(
        self,
        cow_ids: List[int],
        disease_predictions: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesizes a veterinary report using Groq LLM with REAL per-cow sensor vitals. 🐄✨
        """
        clinical_summaries = risk_assessment.get("clinical_summaries", [])
        
        # Build a per-cow real vitals section from the sensor data
        per_cow_vitals = {}
        for summary in clinical_summaries:
            cid = summary.get("cow_id")
            milk_data = summary.get("milk", {})
            heat_data = summary.get("heat_stress", {})
            health_data = summary.get("health", {})
            sensor_vitals = summary.get("sensor_vitals", {})
            
            # Build a rich vitals dict using real sensor data first, then model outputs
            vitals = {
                # Real sensor measurements from MMCOWS dataset
                "cbt_celsius": sensor_vitals.get("cbt_celsius") or "Sensor offline",
                "thi_heat_index": sensor_vitals.get("thi") or "Sensor offline",
                "activity_accel_mag": sensor_vitals.get("accel_mag") or "Sensor offline",
                "sensor_milk_kg": sensor_vitals.get("sensor_milk_kg") or "Sensor offline",
                "weight_kg": sensor_vitals.get("weight_kg") or "Not measured",
                "age_years": sensor_vitals.get("age_years") or "Not recorded",
                # AI model predictions
                "model_health_score": health_data.get("health_score", "N/A"),
                "model_risk_level": health_data.get("risk_level", "unknown"),
                "anomaly_detected": health_data.get("anomaly_detected", False),
                "model_milk_yield_kg": milk_data.get("predicted_yield_kg", "N/A"),
                "model_heat_stress": heat_data.get("stress_level", "unknown"),
                "disease_candidates": health_data.get("disease_predictions", []),
            }
            per_cow_vitals[str(cid)] = vitals

        # External sensor override (user-provided via the UI)
        user_sensors = kwargs.get("sensor_data") or {}
        user_weight = kwargs.get("animal_weight_kg")
        user_age = kwargs.get("animal_age_years")

        prompt = (
            f"Generate a COMPREHENSIVE clinical report for the following cows using ONLY the REAL sensor/model data provided below.\n"
            f"Do NOT invent or substitute default values for any vital sign.\n\n"
            f"=== REAL PER-COW MODEL/SENSOR DATA ===\n"
            f"{json.dumps(per_cow_vitals, indent=2)}\n\n"
            f"=== SUSPECTED CONDITIONS (from AI models) ===\n"
            f"{json.dumps(disease_predictions, indent=2)}\n\n"
        )
        
        if user_sensors or user_weight or user_age:
            prompt += (
                f"=== USER-PROVIDED MANUAL VITALS (override for primary cow) ===\n"
                f"Weight: {user_weight or 'not provided'} kg\n"
                f"Age: {user_age or 'not provided'} years\n"
                f"Manual sensors: {json.dumps(user_sensors, indent=2)}\n\n"
            )

        prompt += "Write a detailed clinical report for each cow listed. Use emojis and be engaging! 🐄✨"

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": self.max_tokens
                    }
                )
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    return {"report": content, "summary": "IDSS synthesis complete."}
                elif response.status_code == 429 or response.status_code == 400:
                    # Fallback to 8b instant on limit exhausted
                    logger.warning(f"Groq primary model failed ({response.status_code}). Falling back to 8B model...")
                    fb_response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": self.max_tokens
                        }
                    )
                    if fb_response.status_code == 200:
                        content = fb_response.json()["choices"][0]["message"]["content"]
                        return {"report": content, "summary": "IDSS synthesis complete (Fallback Model)."}
                    logger.error(f"Fallback Groq error: {fb_response.text}")
                    raise Exception(f"Fallback API Error {fb_response.status_code}")
                else:
                    logger.error(f"Groq error: {response.text}")
                    raise Exception(f"API Error {response.status_code}")
        except Exception as e:
            logger.error(f"Report synthesis failed: {e}")
            return {
                "report": "⚠️ IDSS synthesis unavailable due to API limit exhaustion. Model metrics remain accurate. 📊",
                "summary": "Synthesis failed."
            }


    async def answer_clinical_question(self, question: str, context: Optional[str] = None) -> str:
        """Answers a clinical question using Groq LLM."""
        messages = [{"role": "system", "content": "You are the Veterinary AI Agent. 🐄✨"}]
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": question})

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": messages}
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                logger.error(f"Groq API Error: {response.status_code} - {response.text}")
                return f"Connection error with Groq API (Status {response.status_code}). 🩺🩹"
        except Exception as e:
            logger.error(f"Chat answer failed: {e}")
            return "I'm having trouble with my diagnostic brain. 🩺🩹"
