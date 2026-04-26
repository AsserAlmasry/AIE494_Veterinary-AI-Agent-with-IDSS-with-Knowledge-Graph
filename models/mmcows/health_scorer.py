"""
models/mmcows/health_scorer.py
===============================
Production wrapper for MultiModalFusion + SensorAutoencoder.
Fuses visual embeddings (512-dim from CowReIDModel) with sensor data (30-dim)
to produce a health score and anomaly flag.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import timm
except ImportError:
    timm = None

logger = logging.getLogger(__name__)

from models.mmcows.original_models import MultiModalFusion, SensorAutoencoder, CowReIDModel


class HealthScorer:
    SAVED_MODELS_DIR = r"C:\Users\Dell\.gemini\antigravity\graduation project\Mmcows\mmcows\mmcow\saved_models"

    def __init__(self, mmcows_base_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path = mmcows_base_path
        self._fusion_model = None
        self._anomaly_model = None
        self._id_backbone = None
        self._available = False
        self._load_models(self.SAVED_MODELS_DIR)
        logger.info(f"HealthScorer (Fusion) ready | models_loaded={self._available}")

    def _load_models(self, models_dir: str):
        try:
            # 1. MultiModalFusion: visual_dim=512, sensor_dim=30
            self._fusion_model = MultiModalFusion(visual_dim=512, sensor_dim=30, hidden_dim=256)
            fusion_path = os.path.join(models_dir, "fusion_model.pth")
            if os.path.exists(fusion_path):
                state = torch.load(fusion_path, map_location=self.device, weights_only=False)
                self._fusion_model.load_state_dict(state, strict=False)
                self._fusion_model.to(self.device).eval()
                logger.info(f"Fusion model loaded from {fusion_path}")
            else:
                logger.warning(f"Fusion model not found at {fusion_path}")

            # 2. SensorAutoencoder — input_dim=512 (matches vis_embed)
            self._anomaly_model = SensorAutoencoder(input_dim=512, latent_dim=64)
            anomaly_path = os.path.join(models_dir, "anomaly_autoencoder.pth")
            if os.path.exists(anomaly_path):
                state = torch.load(anomaly_path, map_location=self.device, weights_only=False)
                self._anomaly_model.load_state_dict(state, strict=False)
                self._anomaly_model.to(self.device).eval()
                logger.info(f"Anomaly model loaded from {anomaly_path}")
            else:
                logger.warning(f"Anomaly model not found at {anomaly_path}")

            # 3. CowReIDModel backbone for visual embeddings
            self._id_backbone = CowReIDModel(pretrained=True)
            id_path = os.path.join(models_dir, "identification_model.pth")
            if os.path.exists(id_path):
                state = torch.load(id_path, map_location=self.device, weights_only=False)
                self._id_backbone.load_state_dict(state, strict=False)
                self._id_backbone.to(self.device).eval()
                logger.info(f"ID backbone loaded from {id_path}")

            self._available = True
        except Exception as e:
            logger.error(f"Failed to load HealthScorer models: {e}", exc_info=True)

    @staticmethod
    def _preprocess_image(crop: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(crop.convert("RGB")).unsqueeze(0)

    def predict(
        self,
        cow_crop: Optional[Image.Image] = None,
        cow_id: int = None,
        sensor_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Unified Health Score prediction using Multi-Modal Fusion."""
        t0 = time.perf_counter()

        if not self._available:
            return {"error": "Models not loaded", "health_score": 0.5, "status": "error"}

        # 1. Extract Visual Embedding (512-dim)
        vis_embed = torch.zeros((1, 512)).to(self.device)
        if cow_crop is not None:
            try:
                img_tensor = self._preprocess_image(cow_crop).to(self.device)
                with torch.no_grad():
                    vis_embed = self._id_backbone(img_tensor)
                    # Ensure correct shape [1, 512]
                    if vis_embed.ndim == 1:
                        vis_embed = vis_embed.unsqueeze(0)
            except Exception as e:
                logger.warning(f"Visual embedding failed: {e}")

        # 2. Prepare Sensor Features (30-dim)
        sensor_tensor = torch.zeros((1, 30)).to(self.device)
        if sensor_data is not None:
            try:
                arr = np.array(sensor_data, dtype=np.float32).flatten()
                # Pad or trim to exactly 30
                if len(arr) < 30:
                    arr = np.pad(arr, (0, 30 - len(arr)))
                else:
                    arr = arr[:30]
                # Replace NaN/inf
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                sensor_tensor = torch.FloatTensor(arr).unsqueeze(0).to(self.device)
            except Exception as e:
                logger.warning(f"Sensor tensor prep failed: {e}")

        # 3. Run Fusion + Anomaly
        health_score = 0.5
        anomaly_score = 0.0
        fusion_succeeded = False
        try:
            with torch.no_grad():
                score_tensor, _, _ = self._fusion_model(vis_embed, sensor_tensor)
                hs = float(score_tensor.cpu().item())
                # Guard against NaN/Inf
                if math.isnan(hs) or math.isinf(hs):
                    logger.warning(f"Fusion returned NaN/Inf health score for cow {cow_id}, using 0.5")
                    hs = 0.5
                health_score = max(0.0, min(1.0, hs))
                fusion_succeeded = True

                # Anomaly on vis_embed (512-dim)
                recon_err = self._anomaly_model.compute_anomaly_score(vis_embed)
                anom = float(recon_err.cpu().item())
                if math.isnan(anom) or math.isinf(anom):
                    anom = 0.0
                anomaly_score = max(0.0, anom)
        except Exception as e:
            logger.warning(f"Fusion/Anomaly inference failed for cow {cow_id}: {e}")

        # Interpretation
        anomaly_detected = anomaly_score > 0.15
        risk_level = "low"
        if health_score < 0.4 or anomaly_detected:
            risk_level = "high"
        elif health_score < 0.7:
            risk_level = "medium"

        hs_pct = f"{health_score:.1%}"
        if anomaly_detected:
            hs_display = f"ANOMALY ({hs_pct})"
        else:
            hs_display = hs_pct

        return {
            "health_score": hs_display,
            "raw_health_score": round(health_score, 4),
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_detected": anomaly_detected,
            "risk_level": risk_level,
            "cow_id": cow_id,
            "fusion_succeeded": fusion_succeeded,
            "recommendations": self._get_recs(risk_level, anomaly_detected),
            "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "status": "success"
        }

    def _get_recs(self, risk, anomaly):
        if risk == "high": return ["🔴 High clinical risk. Immediate vet inspection."]
        if anomaly: return ["⚠️ Sensor anomaly detected. Check equipment or behavioral deviations."]
        if risk == "medium": return ["🟡 Moderate risk. Monitor clinical signs."]
        return ["🟢 Stable health status."]
