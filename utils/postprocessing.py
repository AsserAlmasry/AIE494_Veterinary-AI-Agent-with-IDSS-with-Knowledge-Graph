"""
utils/postprocessing.py
========================
Post-processing helpers for normalising and serialising AI outputs
before they are returned via the FastAPI response layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def format_disease_predictions(
    raw_predictions: List[Dict[str, Any]],
    top_k: int = 3,
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Sort and filter disease predictions for API response.

    Adds:
    • rank field (1-based)
    • confidence formatted as percentage string
    • alert_level: critical / warning / info based on confidence
    """
    filtered = [
        p for p in raw_predictions
        if p.get("confidence", 0) >= min_confidence
        and p.get("disease") != "healthy"
    ]
    sorted_preds = sorted(filtered, key=lambda x: x.get("confidence", 0), reverse=True)

    result = []
    for i, pred in enumerate(sorted_preds[:top_k]):
        conf = pred.get("confidence", 0)
        alert = "critical" if conf >= 0.75 else ("warning" if conf >= 0.5 else "info")
        result.append(
            {
                "rank":            i + 1,
                "disease":         pred.get("disease"),
                "confidence":      round(conf, 4),
                "confidence_pct":  f"{conf:.1%}",
                "category":        pred.get("category", "other"),
                "alert_level":     alert,
                "safety_validated": pred.get("safety_validated", True),
                "safety_note":     pred.get("safety_note"),
            }
        )
    return result


def format_risk_assessment(risk_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise risk prediction output for API response.
    Converts float tensors to Python floats, rounds scores.
    """
    if not risk_raw:
        return {}

    # Sort top disease risks
    disease_risks = risk_raw.get("disease_risks", {})
    top_diseases = sorted(
        [(d, round(float(s), 4)) for d, s in disease_risks.items() if s > 0.3],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "overall_risk_score":      round(float(risk_raw.get("overall_risk_score", 0)), 4),
        "risk_uncertainty":        round(float(risk_raw.get("risk_uncertainty", 0)), 4),
        "risk_level":              risk_raw.get("risk_level", "unknown"),
        "top_disease_risks":       [{"disease": d, "risk_score": s} for d, s in top_diseases],
        "top_risk_factors":        risk_raw.get("top_risk_factors", []),
        "recommendations":         risk_raw.get("recommendations", []),
        "prediction_horizon_days": risk_raw.get("prediction_horizon_days", 7),
        "monte_carlo_samples":     risk_raw.get("monte_carlo_samples", 5),
        "timestamp":               risk_raw.get("timestamp", datetime.now().isoformat()),
        "safety_flags":            risk_raw.get("safety_flags", []),
        "disclaimer":              risk_raw.get("disclaimer"),
    }


def format_identity_result(identity_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise identity engine output for API response."""
    return {
        "cow_id":            identity_raw.get("cow_id", 0),
        "decision":          identity_raw.get("decision", "unknown"),
        "confidence":        round(float(identity_raw.get("confidence", 0)), 4),
        "similarity_score":  round(float(identity_raw.get("similarity_score", 0)), 4),
        "method":            identity_raw.get("method", "unknown"),
        "latency_ms":        identity_raw.get("latency_ms", 0),
        "matched_reference": identity_raw.get("matched_reference"),
        "manual_override_allowed": identity_raw.get("manual_override_allowed", True),
    }


def build_summary_response(
    cow_id: int,
    identity: Optional[Dict],
    disease: Optional[Dict],
    risk: Optional[Dict],
    report: Optional[Dict],
    total_latency_ms: float,
) -> Dict[str, Any]:
    """
    Assemble a clean top-level summary suitable for frontend consumption.
    """
    summary: Dict[str, Any] = {
        "cow_id":          cow_id,
        "generated_at":    datetime.now().isoformat(),
        "total_latency_ms": round(total_latency_ms, 1),
    }

    if identity:
        summary["identity"] = format_identity_result(identity)

    if disease:
        preds = disease.get("predictions", [])
        safety = disease.get("safety", {})
        summary["disease"] = {
            "predictions":       format_disease_predictions(preds),
            "healthy_probability": round(float(disease.get("healthy_probability", 0)), 4),
            "model_uncertainty": round(float(disease.get("model_uncertainty", 0)), 4),
            "top_category":      disease.get("top_category", "unknown"),
            "inference_time_ms": disease.get("inference_time_ms", 0),
            "safety_level":      safety.get("safety_level", "pass"),
            "safety_flags":      safety.get("safety_flags", []),
        }

    if risk:
        summary["risk"] = format_risk_assessment(risk)

    if report:
        summary["clinical_report"] = {
            "summary":   report.get("summary"),
            "llm_model": report.get("llm_model"),
            "full_report_available": bool(report.get("report")),
        }

    return summary


def to_serialisable(obj: Any) -> Any:
    """
    Recursively convert numpy/torch types to Python builtins for JSON serialisation.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
    except ImportError:
        pass
    return obj
