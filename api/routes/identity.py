"""
api/routes/identity.py — Cow identity verification endpoints (v4 MMCOWS).
Uses the real CowReIDModel (ViT + ArcFace, 16 cows).
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.dependencies import get_cow_identifier

router = APIRouter()


class IdentityResponse(BaseModel):
    decision: str
    detections: list
    total_cows_detected: int
    latency_ms: float


@router.post("/identity", response_model=IdentityResponse, summary="Identify cow(s) from image")
async def identify_cow(
    image: UploadFile = File(..., description="Cattle image (JPEG/PNG)"),
    cow_identifier=Depends(get_cow_identifier),
) -> IdentityResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    image_bytes = await image.read()
    try:
        import io
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = cow_identifier.identify(pil_img)
        return IdentityResponse(
            decision=result["decision"],
            detections=result.get("detections", []),
            total_cows_detected=result.get("total_cows_detected", 0),
            latency_ms=result.get("latency_ms", 0),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Identity inference failed: {exc}")


@router.get("/identity", summary="List all known cows")
async def list_cows() -> Dict[str, Any]:
    return {
        "total_known_cows": 16,
        "known_cow_ids": list(range(1, 17)),
        "model": "CowReIDModel (ViT + ArcFace)",
        "dataset": "MMCOWS",
    }
