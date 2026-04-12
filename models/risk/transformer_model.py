"""
models/risk/transformer_model.py
=================================
HealthRiskTransformer — time-series cattle health risk prediction.
Processes 12-channel sensor sequences of length 168 (hourly, 7 days).
Outputs multi-horizon risk scores + top contributing factors.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Sub-modules ───────────────────────────────────────────────────────────────

class HealthRiskAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.physio_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Sigmoid(),
            nn.Linear(d_model // 4, d_model),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        gated = attn_out * self.physio_gate(x)
        x = self.norm1(x + self.dropout(gated))
        return self.norm2(x + self.dropout(self.ffn(x)))


# ── Main Model ────────────────────────────────────────────────────────────────

class HealthRiskTransformer(nn.Module):
    """
    Transformer-based health risk predictor for cattle IoT sensor streams.

    Inputs : sensor_sequence (B, L, 12)  — normalised sensor readings
    Outputs: overall_risk, per-disease risks, feature attributions
    """

    SENSOR_FEATURES: List[str] = [
        "body_temp", "heart_rate", "respiratory_rate", "rumination_time",
        "activity_level", "feed_intake", "water_intake", "milk_yield",
        "milk_conductivity", "step_count", "lying_time", "weight_change",
    ]

    DISEASE_CLASSES: List[str] = [
        "healthy", "mastitis", "lameness", "respiratory_disease",
        "digestive_disorder", "skin_lesion", "eye_infection", "hoof_disease",
        "metabolic_disorder", "reproductive_issue", "neurological_sign", "fever",
        "dehydration", "weight_loss", "nasal_discharge", "cough", "diarrhea",
        "abdominal_pain", "joint_swelling", "udder_abnormality", "skin_nodules",
        "oral_lesion", "lymph_node_swelling", "abnormal_gait", "lethargy",
    ]

    PHYSIO_CONSTRAINTS: Dict[str, Dict[str, float]] = {
        "body_temp":        {"min": 37.0, "max": 41.0, "critical_low": 37.5, "critical_high": 40.0},
        "heart_rate":       {"min": 30,   "max": 120,  "critical_low": 40,   "critical_high": 100},
        "respiratory_rate": {"min": 10,   "max": 60,   "critical_low": 12,   "critical_high": 50},
        "rumination_time":  {"min": 0,    "max": 720,  "critical_low": 200,  "critical_high": 600},
        "activity_level":   {"min": 0,    "max": 3000, "critical_low": 300,  "critical_high": 2000},
        "feed_intake":      {"min": 0,    "max": 40,   "critical_low": 10,   "critical_high": 30},
        "water_intake":     {"min": 0,    "max": 150,  "critical_low": 20,   "critical_high": 100},
        "milk_yield":       {"min": 0,    "max": 60,   "critical_low": 10,   "critical_high": 45},
        "milk_conductivity":{"min": 3.0,  "max": 10.0, "critical_low": 4.0,  "critical_high": 7.5},
        "step_count":       {"min": 0,    "max": 15000,"critical_low": 2000, "critical_high": 10000},
        "lying_time":       {"min": 0,    "max": 24,   "critical_low": 6,    "critical_high": 18},
        "weight_change":    {"min": -10,  "max": 10,   "critical_low": -4,   "critical_high": 3},
    }

    SENSOR_DEFAULTS: Dict[str, float] = {
        "body_temp": 38.7, "heart_rate": 65, "respiratory_rate": 25,
        "rumination_time": 480, "activity_level": 1000, "feed_intake": 20,
        "water_intake": 50, "milk_yield": 25, "milk_conductivity": 5.0,
        "step_count": 5000, "lying_time": 12, "weight_change": 0.0,
    }

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_url: Optional[str] = None,
        input_dim: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 168,
        prediction_horizon_days: int = 7,
        num_diseases: int = 24,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon_days

        # Architecture
        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.temporal_embedding = nn.Embedding(max_seq_len, d_model)
        self.pos_dropout = nn.Dropout(dropout)
        self.temporal_blocks = nn.ModuleList(
            [HealthRiskAttentionBlock(d_model, nhead, dim_feedforward, dropout)
             for _ in range(num_layers)]
        )
        self.cross_feature_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.overall_risk_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.disease_risk_heads = nn.ModuleDict(
            {
                d: nn.Sequential(
                    nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
                )
                for d in self.DISEASE_CLASSES[1:]  # skip "healthy"
            }
        )
        self.attribution_head = nn.Linear(d_model, input_dim)

        self.to(self.device)

        if model_url and checkpoint_path and not os.path.exists(checkpoint_path):
            self._download_weights(checkpoint_path, model_url)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self.eval()
        logger.info(f"HealthRiskTransformer ready | device={self.device}")

    # ── Weight management ─────────────────────────────────────────────────────

    @staticmethod
    def _download_weights(path: str, url: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            import gdown
            logger.info(f"Downloading risk model from {url} …")
            gdown.download(url, path, quiet=False)
        except Exception as exc:
            logger.warning(f"Risk model download failed: {exc}")

    def _load_checkpoint(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, nn.Module):
                self.load_state_dict(ckpt.state_dict(), strict=False)
            elif isinstance(ckpt, dict):
                state = ckpt.get("state_dict", ckpt.get("model", ckpt))
                if isinstance(state, dict):
                    self.load_state_dict(state, strict=False)
            logger.info(f"Risk model checkpoint loaded: {path}")
        except Exception as exc:
            logger.warning(f"Risk checkpoint load error: {exc} — using random weights")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, sensor_sequence: torch.Tensor) -> Dict[str, Any]:
        B, L, _ = sensor_sequence.shape
        feat_emb = self.feature_embedding(sensor_sequence)
        pos = torch.arange(L, device=self.device).unsqueeze(0).expand(B, -1)
        x = self.pos_dropout(feat_emb + self.temporal_embedding(pos))
        for block in self.temporal_blocks:
            x = block(x)
        x_cf, _ = self.cross_feature_attn(x, x, x)
        x = x + x_cf
        pooled = x.mean(dim=1)
        return {
            "overall_risk":  self.overall_risk_head(pooled),
            "disease_risks": {d: head(pooled) for d, head in self.disease_risk_heads.items()},
            "attributions":  torch.sigmoid(self.attribution_head(pooled)),
            "pooled":        pooled,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_risk(
        self,
        sensor_data: Dict[str, float],
        history_window: Optional[List[Dict[str, float]]] = None,
        cow_id: Optional[int] = None,
        n_mc_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict health risk from current sensor readings + optional history.

        Parameters
        ----------
        sensor_data    : Dict mapping feature name → current value.
        history_window : List of past sensor dicts (most recent last).
        cow_id         : ID tag included in the result.
        n_mc_samples   : Monte-Carlo dropout samples for uncertainty estimation.
        """
        t0 = time.perf_counter()
        sequence = self._prepare_sequence(sensor_data, history_window)
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        self.train()  # enable dropout for MC sampling
        samples = []
        with torch.no_grad():
            for _ in range(n_mc_samples):
                out = self.forward(seq_tensor)
                samples.append(
                    {
                        "overall":      float(out["overall_risk"].cpu().item()),
                        "diseases":     {d: float(r.cpu().item()) for d, r in out["disease_risks"].items()},
                        "attributions": out["attributions"].cpu().numpy()[0],
                    }
                )
        self.eval()

        overall_risk = float(np.mean([s["overall"] for s in samples]))
        risk_uncertainty = float(np.std([s["overall"] for s in samples]))
        disease_risks = {
            d: float(np.mean([s["diseases"][d] for s in samples]))
            for d in self.DISEASE_CLASSES[1:]
        }
        avg_attrib = np.mean([s["attributions"] for s in samples], axis=0)
        risk_level = self._categorize_risk(overall_risk)
        top_factors = self._get_top_risk_factors(avg_attrib, sensor_data)

        return {
            "overall_risk_score":      overall_risk,
            "risk_uncertainty":        risk_uncertainty,
            "risk_level":              risk_level,
            "disease_risks":           disease_risks,
            "top_risk_factors":        top_factors,
            "prediction_horizon_days": self.prediction_horizon,
            "recommendations":         self._generate_recommendations(risk_level, top_factors, disease_risks),
            "monte_carlo_samples":     n_mc_samples,
            "inference_time_ms":       round((time.perf_counter() - t0) * 1000, 2),
            "timestamp":               datetime.now().isoformat(),
            "cow_id":                  cow_id,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_sequence(
        self,
        current: Dict[str, float],
        history: Optional[List[Dict[str, float]]],
        target_length: int = 168,
    ) -> np.ndarray:
        def norm(feat: str, val: float) -> float:
            c = self.PHYSIO_CONSTRAINTS.get(feat, {"min": 0, "max": 100})
            return float(np.clip((val - c["min"]) / (c["max"] - c["min"] + 1e-6), 0, 1))

        cur_vec = np.array(
            [norm(f, current.get(f, self.SENSOR_DEFAULTS.get(f, 0))) for f in self.SENSOR_FEATURES],
            dtype=np.float32,
        )

        if history:
            hist_vecs = [
                np.array([norm(f, e.get(f, self.SENSOR_DEFAULTS.get(f, 0))) for f in self.SENSOR_FEATURES], dtype=np.float32)
                for e in history[-(target_length - 1):]
            ]
            pad = [cur_vec] * max(0, target_length - 1 - len(hist_vecs))
            sequence = np.stack(pad + hist_vecs + [cur_vec])
        else:
            rng = np.random.default_rng(42)
            sequence = np.stack(
                [np.clip(cur_vec + rng.normal(0, 0.01, self.input_dim).astype(np.float32), 0, 1)
                 for _ in range(target_length)]
            )
        return sequence.astype(np.float32)

    @staticmethod
    def _categorize_risk(score: float) -> str:
        if score >= 0.6:
            return "high"
        if score >= 0.3:
            return "medium"
        return "low"

    def _get_top_risk_factors(
        self, attributions: np.ndarray, sensor_data: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        factors = []
        for idx in np.argsort(attributions)[::-1][:5]:
            feat = self.SENSOR_FEATURES[idx]
            c = self.PHYSIO_CONSTRAINTS.get(feat, {"min": 0, "max": 100})
            val = sensor_data.get(feat, self.SENSOR_DEFAULTS.get(feat, 0))
            if val < c.get("critical_low", c["min"]):
                status = "critically_low"
            elif val > c.get("critical_high", c["max"]):
                status = "critically_high"
            elif val < c["min"]:
                status = "low"
            elif val > c["max"]:
                status = "high"
            else:
                status = "normal"
            factors.append(
                {
                    "feature": feat,
                    "attribution_score": float(attributions[idx]),
                    "current_value": val,
                    "normal_range": c,
                    "status": status,
                }
            )
        return factors

    @staticmethod
    def _generate_recommendations(
        risk_level: str,
        factors: List[Dict],
        disease_risks: Dict[str, float],
    ) -> List[str]:
        recs = []
        if risk_level == "high":
            recs.append("🔴 HIGH RISK: Veterinary examination recommended within 24h")
            high_diseases = [d for d, r in disease_risks.items() if r > 0.7]
            if high_diseases:
                recs.append(f"   Priority diseases: {', '.join(high_diseases[:3])}")
        elif risk_level == "medium":
            recs.append("🟡 MEDIUM RISK: Increase monitoring to twice daily")
        else:
            recs.append("🟢 LOW RISK: Continue routine monitoring")

        for f in factors[:2]:
            if f["status"] in ("critically_low", "critically_high"):
                recs.append(
                    f"   ⚠️  Critical: {f['feature'].replace('_', ' ')} = {f['current_value']}"
                )
        return recs[:5]
