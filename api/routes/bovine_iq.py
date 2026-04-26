from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, List
import logging
import base64
import os
from pydantic import BaseModel

from app.dependencies import get_bovine_iq_agent
from services.bovine_iq_service import BovineIQAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bovine_iq", tags=["BovineIQ"])

class MessageInput(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ResumeInput(BaseModel):
    pending_state: Dict[str, Any]
    approved: bool

@router.post("/chat")
async def bovine_iq_chat(
    payload: MessageInput,
    agent: BovineIQAgent = Depends(get_bovine_iq_agent)
) -> Dict[str, Any]:
    """Process a chat message through BovineIQ."""
    try:
        result = agent.query(payload.message, payload.history)
        return {"result": result}
    except Exception as e:
        logger.error(f"BovineIQ chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/resume")
async def bovine_iq_resume(
    payload: ResumeInput,
    agent: BovineIQAgent = Depends(get_bovine_iq_agent)
) -> Dict[str, Any]:
    """Resume execution after user approval/rejection of code."""
    try:
        result = agent.resume(payload.pending_state, payload.approved)
        return {"result": result}
    except Exception as e:
        logger.error(f"BovineIQ resume failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/image/{filename}")
async def get_bovine_iq_image(filename: str):
    """Serve generated temporary images."""
    from fastapi.responses import FileResponse
    file_path = os.path.join("./bovine_iq/temp_images", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/jpeg")
