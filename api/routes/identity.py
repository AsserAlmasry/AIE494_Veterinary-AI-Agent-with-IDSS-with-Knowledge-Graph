"""
api/routes/identity.py
=======================
POST /identity — Cow identity verification endpoint.
GET  /identity/{cow_id} — Retrieve cow identity profile.
POST /identity/{cow_id}/override — Manual identity correction.
GET  /identity — List all known cows.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.dependencies import get_identity_engine, get_identity_bank

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────────

class IdentityResponse(BaseModel):
    cow_id: int
    decision: str
    confidence: float
    similarity_score: float
    method: str
    latency_ms: float
    matched_reference: Optional[Dict[str, Any]]
    manual_override_allowed: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/identity",
    response_model=IdentityResponse,
    summary="Identify a cow from image",
    description="Two-stage: YOLO detection → ViT embedding cosine search.",
)
async def identify_cow(
    image: UploadFile = File(..., description="Cattle image (JPEG/PNG)"),
    identity_engine=Depends(get_identity_engine),
) -> IdentityResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    try:
        import io
        import numpy as np
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(pil_img)
        result = identity_engine.identify(image_array)
        return IdentityResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Identity inference failed: {exc}")


@router.get(
    "/identity/{cow_id}",
    summary="Get cow identity profile",
    description="Retrieve metadata and confidence history for a known cow.",
)
async def get_cow_profile(
    cow_id: int,
    identity_bank=Depends(get_identity_bank),
) -> Dict[str, Any]:
    profile = identity_bank.get_cow_summary(cow_id)
    if not profile.get("exists"):
        raise HTTPException(status_code=404, detail=f"Cow ID {cow_id} not found.")
    return profile


@router.post(
    "/identity/{cow_id}/override",
    summary="Manual identity correction",
    description="Veterinarian corrects a misidentified cow ID.",
)
async def override_identity(
    cow_id: int,
    image: UploadFile = File(...),
    confirmed_by: str = "veterinarian",
    identity_bank=Depends(get_identity_bank),
    identity_engine=Depends(get_identity_engine),
) -> Dict[str, Any]:
    image_bytes = await image.read()
    try:
        import io
        import numpy as np
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(pil_img)
        embedding = identity_engine.embedder.extract(image_array)
        result = identity_bank.manual_override(
            query_embedding=embedding,
            confirmed_cow_id=cow_id,
            confirmed_by=confirmed_by,
        )
        identity_engine.add_embedding(cow_id, embedding)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Override failed: {exc}")


@router.get(
    "/identity",
    summary="List all known cows",
)
async def list_cows(
    identity_bank=Depends(get_identity_bank),
) -> Dict[str, Any]:
    return {
        "total_known_cows":    len(identity_bank.KNOWN_COW_IDS),
        "total_embeddings":    identity_bank.index.ntotal,
        "next_new_cow_id":     identity_bank.NEXT_NEW_COW_ID,
        "similarity_threshold": identity_bank.SIMILARITY_THRESHOLD,
        "known_cow_ids":       identity_bank.KNOWN_COW_IDS,
    }
