"""
services/vision_service.py
===========================
Groq Vision Service — analyses cattle images using LLaMA vision models.

This is the PRIMARY disease detection layer. Because the MaxViT classification
head is not yet fine-tuned on cattle-disease data, Groq Vision provides
accurate, visually grounded disease candidates by actually *seeing* the cow.

Model used: llama-3.2-11b-vision-preview (Groq, fast, vision-capable)
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Full mapping from vision-friendly names → model class names
VISION_TO_CLASS: Dict[str, str] = {
    "mastitis":              "mastitis",
    "udder infection":       "mastitis",
    "teat infection":        "mastitis",
    "lameness":              "lameness",
    "limping":               "lameness",
    "foot disease":          "hoof_disease",
    "hoof disease":          "hoof_disease",
    "foot rot":              "hoof_disease",
    "respiratory":           "respiratory_disease",
    "pneumonia":             "respiratory_disease",
    "cough":                 "cough",
    "nasal discharge":       "nasal_discharge",
    "skin lesion":           "skin_lesion",
    "wound":                 "skin_lesion",
    "dermatitis":            "skin_lesion",
    "skin nodules":          "skin_nodules",
    "lumpy skin":            "skin_nodules",
    "eye infection":         "eye_infection",
    "pink eye":              "eye_infection",
    "keratoconjunctivitis":  "eye_infection",
    "diarrhea":              "diarrhea",
    "scouring":              "diarrhea",
    "bloat":                 "digestive_disorder",
    "rumen":                 "digestive_disorder",
    "ketosis":               "metabolic_disorder",
    "metabolic":             "metabolic_disorder",
    "reproductive":          "reproductive_issue",
    "metritis":              "reproductive_issue",
    "neural":                "neurological_sign",
    "neurological":          "neurological_sign",
    "weight loss":           "weight_loss",
    "emaciated":             "weight_loss",
    "thin body":             "weight_loss",
    "dehydrated":            "dehydration",
    "dehydration":           "dehydration",
    "fever":                 "fever",
    "lethargy":              "lethargy",
    "lethargic":             "lethargy",
    "swollen joint":         "joint_swelling",
    "joint swelling":        "joint_swelling",
    "udder":                 "udder_abnormality",
    "udder swelling":        "udder_abnormality",
    "lymph node":            "lymph_node_swelling",
    "mouth sore":            "oral_lesion",
    "oral lesion":           "oral_lesion",
    "foot mouth":            "oral_lesion",
    "healthy":               "healthy",
    "normal":                "healthy",
    "no visible":            "healthy",
    "abnormal gait":         "abnormal_gait",
    "gait abnormality":      "abnormal_gait",
    "abdominal pain":        "abdominal_pain",
    "abdominal distension":  "abdominal_pain",
}

VISION_SYSTEM_PROMPT = """You are an expert bovine (cattle) veterinary diagnostician with 20+ years of clinical experience.
You analyse cattle photographs and identify disease signs with high accuracy.
You MUST respond with valid JSON only — no explanations outside the JSON block."""

VISION_USER_PROMPT = """Analyze this cattle image for clinical signs of disease.
Focus on: Udder (mastitis), Feet (lameness/hoof rot), Skin (lesions/LSD), Eyes (pinkeye), Nose/Breathing (BRD), BCS, and Posture.

Output JSON only:
{
  "visual_observations": ["list of findings"],
  "disease_candidates": [{"disease": "name", "confidence": 0-1, "reasoning": "...", "affected_body_part": "..."}],
  "body_condition_score": 1-5,
  "overall_health_assessment": "...",
  "clinical_urgency": 1-10,
  "appears_healthy": bool
}
Rules:
1. disease names must be lowercase technical terms.
2. If healthy, set appears_healthy=true and disease_candidates=[].
3. NO explanations outside JSON."""


class GroqVisionService:
    """
    Analyses cattle images using Groq's LLaMA vision models.
    Provides visually-grounded disease candidates to replace/augment
    the untrained MaxViT classification head.
    """

    VISION_MODELS = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]

    def __init__(self, api_key: str) -> None:
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            self._model  = self._detect_working_model()
            logger.info(f"GroqVisionService ready | model={self._model}")
        except ImportError:
            raise RuntimeError("groq package not installed.")
        except Exception as exc:
            logger.warning(f"GroqVisionService init failed: {exc}")
            self._client = None
            self._model  = None

    def _detect_working_model(self) -> str:
        """Try vision models in order, return first that works."""
        # Default to first; fallback handled at call time
        return self.VISION_MODELS[0]

    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyse a cattle image and return structured veterinary assessment.

        Returns
        -------
        {
            visual_observations: [...],
            disease_candidates:  [{disease, confidence, reasoning, affected_body_part}, ...],
            body_condition_score: float,
            overall_health_assessment: str,
            clinical_urgency: int,
            appears_healthy: bool,
            mapped_classes: [{disease (model class name), confidence}, ...],
            vision_model: str,
            error: str | None,
        }
        """
        if self._client is None:
            return self._empty_result("Vision service not initialised")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        for model in self.VISION_MODELS:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": VISION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type":      "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": VISION_USER_PROMPT,
                                },
                            ],
                        },
                    ],
                    temperature=0.05,
                    max_tokens=1024,
                )
                raw_text = response.choices[0].message.content or ""
                result   = self._parse_response(raw_text)
                result["vision_model"] = model
                result["error"]        = None
                result["mapped_classes"] = self._map_to_model_classes(
                    result.get("disease_candidates", [])
                )
                logger.info(
                    f"Vision analysis complete | model={model} | "
                    f"candidates={len(result.get('disease_candidates', []))} | "
                    f"healthy={result.get('appears_healthy', False)}"
                )
                return result

            except Exception as exc:
                logger.warning(f"Vision model {model} failed: {exc}. Trying next…")
                continue

        return self._empty_result("All vision models failed")

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Extract and parse the JSON block from the model response."""
        # Try direct JSON parse
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block from markdown
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find the outermost {...} block
        match = re.search(r"\{[\s\S]+\}", raw_text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse vision response: {raw_text[:300]}")
        return {
            "visual_observations":      ["Unable to parse vision response"],
            "disease_candidates":       [],
            "body_condition_score":     3.0,
            "overall_health_assessment":"Vision parsing failed",
            "clinical_urgency":         3,
            "appears_healthy":          False,
        }

    def _map_to_model_classes(
        self, disease_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map vision disease names → model class names (25-class taxonomy).
        """
        mapped: List[Dict[str, Any]] = []
        seen: set = set()

        for cand in disease_candidates:
            raw_disease = str(cand.get("disease", "")).lower().strip()
            confidence  = float(cand.get("confidence", 0.5))

            # Direct match first
            cls = VISION_TO_CLASS.get(raw_disease)

            # Substring match if no direct hit
            if not cls:
                for key, val in VISION_TO_CLASS.items():
                    if key in raw_disease or raw_disease in key:
                        cls = val
                        break

            if cls and cls not in seen:
                seen.add(cls)
                mapped.append({
                    "disease":    cls,
                    "confidence": confidence,
                    "reasoning":  cand.get("reasoning", ""),
                    "source":     "vision",
                })

        return sorted(mapped, key=lambda x: x["confidence"], reverse=True)

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        return {
            "visual_observations":      [],
            "disease_candidates":       [],
            "body_condition_score":     None,
            "overall_health_assessment": reason,
            "clinical_urgency":         3,
            "appears_healthy":          False,
            "mapped_classes":           [],
            "vision_model":             None,
            "error":                    reason,
        }
