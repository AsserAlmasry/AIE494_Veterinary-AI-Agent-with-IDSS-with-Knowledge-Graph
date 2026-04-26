"""
api/routes/predict.py — Full multi-modal veterinary prediction endpoint (v4).
Handles cow validation, annotated image return, crop analysis, and auto-YOLO label resolution.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from app.dependencies import get_pipeline, get_data_pipeline

router = APIRouter()


class PredictResponse(BaseModel):
    cow_id: int
    success: bool
    total_latency_ms: float
    stages: Dict[str, Any]
    errors: List[str]
    pipeline_version: str


def _resolve_label_path(data_pipeline, image_filename: Optional[str], day_index: int) -> Optional[str]:
    import logging, re
    logger = logging.getLogger(__name__)
    """
    Auto-resolve the YOLO label file for a given image from the MMCOWS dataset.
    This enables per-cow bounding boxes with ground-truth IDs.
    """
    if not data_pipeline or not image_filename:
        return None

    # Standardise filename: remove extension and any browser-added suffixes like (1)
    img_stem = Path(image_filename).stem
    img_stem = re.sub(r"\s*\(\d+\)$", "", img_stem) # Remove " (1)"
    img_stem = img_stem.replace(".jpg", "").replace(".png", "")
    
    logger.info(f"_resolve_label_path: Attempting to resolve label for '{img_stem}'")

    # If it matches MMCOWS pattern (10 digits + timestamp) or is just digits
    is_dataset_format = re.match(r"^\d{10}_\d{2}-\d{2}-\d{2}$", img_stem) or img_stem.isdigit()
    
    labels_dir = data_pipeline.visual / "labels" / "combined"
    if labels_dir.exists():
        # First try exact match
        matches = list(labels_dir.rglob(f"{img_stem}.txt"))
        
        # If no exact match and not in standard format, try partial matches
        if not matches and not is_dataset_format:
            logger.info(f"_resolve_label_path: No exact match for '{img_stem}'. Searching for partial matches...")
            for lp in labels_dir.rglob("*.txt"):
                if lp.stem in img_stem or img_stem in lp.stem:
                    matches = [lp]
                    break
        
        if matches:
            path = str(matches[0].absolute())
            logger.info(f"_resolve_label_path: FOUND label at {path}")
            return path
            
    return None


@router.post("/predict", response_model=PredictResponse, summary="Full multi-modal prediction")
async def predict(
    image: Optional[UploadFile] = File(None, description="Cattle image (JPEG/PNG)"),
    sensor_json: Optional[str] = Form(None, description="JSON-encoded sensor data"),
    animal_weight_kg: Optional[float] = Form(None),
    animal_age_years: Optional[float] = Form(None),
    cow_id_override: Optional[int] = Form(None, description="Manual cow ID override"),
    day_index: int = Form(0, description="MMCOWS day index (0-13)"),
    generate_report: bool = Form(True),
    pipeline=Depends(get_pipeline),
    data_pipeline=Depends(get_data_pipeline),
) -> PredictResponse:
    # Normalise zero values
    if cow_id_override is not None and cow_id_override <= 0: cow_id_override = None
    if animal_weight_kg is not None and animal_weight_kg <= 0: animal_weight_kg = None
    if animal_age_years is not None and animal_age_years <= 0: animal_age_years = None

    image_bytes: Optional[bytes] = None
    image_filename: Optional[str] = None
    if image and image.filename:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        image_bytes = await image.read()
        image_filename = image.filename

    sensor_data: Optional[Dict[str, float]] = None
    if sensor_json:
        stripped = sensor_json.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                import json
                raw = json.loads(sensor_json)
                sensor_data = {k: float(v) for k, v in raw.items() if v is not None}
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid sensor_json: {exc}")

    if image_bytes is None and sensor_data is None:
        raise HTTPException(status_code=422, detail="At least one of 'image' or 'sensor_json' must be provided.")

    # Auto-resolve YOLO label path for bounding boxes
    label_path = _resolve_label_path(data_pipeline, image_filename, day_index)

    result = await pipeline.run_full_pipeline(
        image_bytes=image_bytes, sensor_data=sensor_data,
        animal_weight_kg=animal_weight_kg, animal_age_years=animal_age_years,
        cow_id_override=cow_id_override, day_index=day_index,
        generate_report=generate_report, label_path=label_path,
    )
    return PredictResponse(**result)


class CropAnalysisRequest(BaseModel):
    x: int
    y: int
    width: int
    height: int
    description: str = ""


@router.post("/predict/crop", summary="Analyze a cropped region of a cow image")
async def analyze_crop(
    image: UploadFile = File(..., description="Full cattle image"),
    x: int = Form(...), y: int = Form(...),
    width: int = Form(...), height: int = Form(...),
    description: str = Form(""),
    pipeline=Depends(get_pipeline),
) -> Dict[str, Any]:
    image_bytes = await image.read()
    crop_region = {"x": x, "y": y, "width": width, "height": height}
    return await pipeline.analyze_crop(image_bytes, crop_region, description)


@router.post("/predict/disease", summary="Image-only disease classification")
async def predict_disease(
    image: UploadFile = File(..., description="Cattle image (JPEG/PNG)"),
    pipeline=Depends(get_pipeline),
) -> Dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    image_bytes = await image.read()
    import io
    from PIL import Image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = pipeline.cow_identifier.identify(pil_img)
    return result
