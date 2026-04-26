"""
models/mmcows/heat_stress_analyzer.py
======================================
Production wrapper for the trained HeatStressTransformer model.
Uses 19 features (THI, CBT, IMU, UWB, etc.) to predict heat stress levels.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

STRESS_LABELS = ['Normal', 'Mild', 'Moderate', 'Severe']

from models.mmcows.original_models import HeatStressTransformer


class HeatStressAnalyzer:
    """
    Combines THI and behavioral data via the HeatStressTransformer.
    """

    def __init__(
        self,
        checkpoint_path: str,
        mmcows_src_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._available = False

        # Override with actual known path
        actual_checkpoint_path = r"C:\Users\Dell\Downloads\best_heat_stress_model.pt"
        if os.path.exists(actual_checkpoint_path):
            checkpoint_path = actual_checkpoint_path

        self._load_model(checkpoint_path)
        logger.info(
            f"HeatStressAnalyzer ready | device={self.device} | "
            f"model={self._available}"
        )

    def _load_model(self, path: str) -> None:
        try:
            self._model = HeatStressTransformer(in_dim=19)

            if os.path.exists(path):
                state = torch.load(path, map_location=self.device, weights_only=False)
                if isinstance(state, dict) and "model_state" in state:
                    self._model.load_state_dict(state["model_state"], strict=False)
                elif isinstance(state, dict) and "state_dict" in state:
                    self._model.load_state_dict(state["state_dict"], strict=False)
                else:
                    self._model.load_state_dict(state, strict=False)
                logger.info(f"HeatStressTransformer model loaded from: {path}")
                self._model.to(self.device).eval()
                self._available = True
            else:
                logger.error(f"HeatStressTransformer checkpoint not found at {path}")
        except Exception as exc:
            logger.error(f"HeatStressTransformer model load failed: {exc}", exc_info=True)

    def analyze(
        self,
        sensor_seq: Optional[np.ndarray] = None,
        day_index: int = 0,
        cow_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze heat stress conditions from a 24-step sequence of 19 features.
        """
        t0 = time.perf_counter()

        if sensor_seq is None or not self._available:
            return self._fallback(t0, cow_id)

        # Normalise to (B, 24, 19)
        arr = np.array(sensor_seq, dtype=np.float32)
        if arr.ndim == 2:
            # (24, 19) → (1, 24, 19)
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 4:
            # (1, 1, 24, 19) or similar – squeeze extra dims
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[:1]
        # arr should now be (B, T, F)
        if arr.ndim != 3:
            logger.warning(f"HeatStressAnalyzer: unexpected shape {arr.shape}, falling back")
            return self._fallback(t0, cow_id)

        # Pad/trim to exactly (B, 24, 19)
        B, T, F = arr.shape
        if T != 24 or F != 19:
            out = np.zeros((B, 24, 19), dtype=np.float32)
            t_take = min(T, 24)
            f_take = min(F, 19)
            out[:, :t_take, :f_take] = arr[:, :t_take, :f_take]
            arr = out

        try:
            tensor = torch.FloatTensor(arr).to(self.device)
            with torch.no_grad():
                logits, fc, risk = self._model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                conf, pred = torch.max(probs, dim=-1)
                
                stress_idx = int(pred.item())
                stress_level = STRESS_LABELS[stress_idx].lower()
                confidence = float(conf.item())
                predicted_risk = float(risk.item())

            return {
                "stress_level": stress_level,
                "confidence": round(confidence, 4),
                "predicted_risk_score": round(predicted_risk, 4),
                "cow_id": cow_id,
                "recommendations": self._get_recommendations(stress_level),
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        except Exception as e:
            logger.warning(f"HeatStress inference failed: {e}")
            return self._fallback(t0, cow_id)

    def _fallback(self, t0, cow_id):
        return {
            "stress_level": "unknown",
            "confidence": 0.0,
            "cow_id": cow_id,
            "recommendations": ["Awaiting 24-step sensor sequence for Heat Stress analysis"],
            "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    @staticmethod
    def _get_recommendations(stress_level: str) -> List[str]:
        recs = []
        if stress_level == "severe":
            recs.append("🔴 SEVERE heat stress — activate all cooling systems immediately")
            recs.append("Increase water availability, reduce milking times, maximize ventilation")
        elif stress_level == "moderate":
            recs.append("🟡 Moderate heat stress detected — ensure cooling and hydration")
            recs.append("Consider adjusting milking schedule to cooler hours")
        elif stress_level == "mild":
            recs.append("🟡 Mild heat stress — monitor closely, ensure shade access")
        else:
            recs.append("🟢 Normal — thermoneutral zone.")
        return recs
