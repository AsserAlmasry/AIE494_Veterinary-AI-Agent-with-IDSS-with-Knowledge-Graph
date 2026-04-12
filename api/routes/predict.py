"""
api/routes/predict.py
======================
POST /predict  — Full multi-modal veterinary prediction endpoint.
Accepts image upload + optional sensor JSON, returns complete pipeline result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.dependencies import get_pipeline

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class SensorData(BaseModel):
    body_temp:          Optional[float] = Field(None, description="°C")
    heart_rate:         Optional[float] = Field(None, description="bpm")
    respiratory_rate:   Optional[float] = Field(None, description="breaths/min")
    rumination_time:    Optional[float] = Field(None, description="minutes/day")
    activity_level:     Optional[float] = Field(None, description="steps/day")
    feed_intake:        Optional[float] = Field(None, description="kg/day")
    water_intake:       Optional[float] = Field(None, description="litres/day")
    milk_yield:         Optional[float] = Field(None, description="litres/day")
    milk_conductivity:  Optional[float] = Field(None, description="mS/cm")
    step_count:         Optional[float] = Field(None, description="steps/day")
    lying_time:         Optional[float] = Field(None, description="hours/day")
    weight_change:      Optional[float] = Field(None, description="kg (+ gain / - loss)")

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.dict().items() if v is not None}


class PredictResponse(BaseModel):
    cow_id:            int
    success:           bool
    total_latency_ms:  float
    stages:            Dict[str, Any]
    errors:            List[str]
    pipeline_version:  str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Full multi-modal prediction",
    description=(
        "Run the complete Veterinary AI pipeline:\n"
        "1. Cow identity (YOLO + ViT embedding)\n"
        "2. Disease classification (MaxViT)\n"
        "3. Health risk prediction (Transformer)\n"
        "4. RAG evidence retrieval (PubMed)\n"
        "5. Clinical report generation (Groq Llama 3.3 70B)"
    ),
)
async def predict(
    image: Optional[UploadFile] = File(None, description="Cattle image (JPEG/PNG)"),
    sensor_json: Optional[str] = Form(None, description="JSON-encoded SensorData"),
    animal_weight_kg: Optional[float] = Form(None),
    animal_age_years: Optional[float] = Form(None, description="Animal age in years"),
    cow_id_override: Optional[int] = Form(None, description="Skip identity; use this cow ID"),
    generate_report: bool = Form(True, description="Generate Groq LLM clinical report"),
    pipeline=Depends(get_pipeline),
) -> PredictResponse:
    # Normalise Swagger placeholder values: 0 means "not provided"
    if cow_id_override is not None and cow_id_override <= 0:
        cow_id_override = None
    if animal_weight_kg is not None and animal_weight_kg <= 0:
        animal_weight_kg = None
    if animal_age_years is not None and animal_age_years <= 0:
        animal_age_years = None
    image_bytes: Optional[bytes] = None
    if image and image.filename:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        image_bytes = await image.read()

    sensor_data: Optional[Dict[str, float]] = None
    if sensor_json:
        # Ignore Swagger UI placeholder values that are not valid JSON objects/arrays
        stripped = sensor_json.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            sensor_json = None  # treat non-JSON strings as "not provided"
        else:
            try:
                import json
                raw = json.loads(sensor_json)
                sensor_data = {k: float(v) for k, v in raw.items() if v is not None}
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid sensor_json: {exc}")

    if image_bytes is None and sensor_data is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'image' or 'sensor_json' must be provided.",
        )

    result = await pipeline.run_full_pipeline(
        image_bytes=image_bytes,
        sensor_data=sensor_data,
        animal_weight_kg=animal_weight_kg,
        animal_age_years=animal_age_years,
        cow_id_override=cow_id_override,
        generate_report=generate_report,
    )
    return PredictResponse(**result)


@router.post(
    "/predict/risk",
    summary="Sensor-only risk prediction",
    description="Predict health risk from sensor data without requiring an image.",
)
async def predict_risk(
    sensor: SensorData,
    cow_id: Optional[int] = None,
    pipeline=Depends(get_pipeline),
) -> Dict[str, Any]:
    sensor_dict = sensor.to_dict()
    if not sensor_dict:
        raise HTTPException(status_code=422, detail="No sensor readings provided.")
    return await pipeline.run_risk_only(
        sensor_data=sensor_dict,
        cow_id=cow_id,
    )


@router.post(
    "/predict/disease",
    summary="Image-only disease classification",
    description="Classify cattle diseases from an image (no sensor data required).",
)
async def predict_disease(
    image: UploadFile = File(..., description="Cattle image (JPEG/PNG)"),
    pipeline=Depends(get_pipeline),
) -> Dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    image_bytes = await image.read()
    return await pipeline.run_disease_only(image_bytes=image_bytes)
