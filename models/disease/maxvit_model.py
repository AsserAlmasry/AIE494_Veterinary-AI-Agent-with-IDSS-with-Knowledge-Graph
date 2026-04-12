"""
models/disease/maxvit_model.py
===============================
MaxViT-based multi-label cattle disease classifier.
25 veterinary disease classes + uncertainty estimation + category hierarchy.

v2 improvements:
  - Fixed checkpoint loading order: backbone → build heads → load full state
  - Temperature scaling calibration for untrained heads
  - Calibration detection flag with informative logging
  - Healthy image prior boost using simple luminance + saturation heuristics
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ── Preprocessing helper ──────────────────────────────────────────────────────

def pil_to_tensor_normalized(image: Image.Image, size: int = 224) -> torch.Tensor:
    """NumPy 2.x-safe PIL → normalised tensor (no torchvision.transforms dependency)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((size + 32, size + 32), Image.BILINEAR)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


# ── MaxViT Disease Classifier ─────────────────────────────────────────────────

class MaxViTDiseaseClassifier(nn.Module):
    """
    MaxViT Base 21k fine-tuned for multi-label cattle disease classification.

    Architecture
    -----------
    • Backbone: maxvit_base_tf_224.in21k (from timm; falls back to ResNet50)
    • Disease head: LayerNorm → Linear(feat→512) → GELU → Dropout → Linear(512→25)
    • Uncertainty head: Linear → ReLU → Linear → Sigmoid
    • Category head: Linear(feat→7) for disease family classification

    Calibration
    -----------
    If the checkpoint only contains backbone weights (no classification head),
    temperature scaling (T=0.5) is applied automatically to sharpen predictions.
    A warning is logged informing the user that the head is uncalibrated.
    """

    DISEASE_CLASSES: List[str] = [
        "healthy", "mastitis", "lameness", "respiratory_disease",
        "digestive_disorder", "skin_lesion", "eye_infection",
        "hoof_disease", "metabolic_disorder", "reproductive_issue",
        "neurological_sign", "fever", "dehydration", "weight_loss",
        "nasal_discharge", "cough", "diarrhea", "abdominal_pain",
        "joint_swelling", "udder_abnormality", "skin_nodules",
        "oral_lesion", "lymph_node_swelling", "abnormal_gait", "lethargy",
    ]

    DISEASE_HIERARCHY: Dict[str, List[str]] = {
        "infectious":       ["mastitis", "respiratory_disease", "skin_lesion", "eye_infection", "oral_lesion", "skin_nodules"],
        "metabolic":        ["metabolic_disorder", "dehydration", "weight_loss"],
        "musculoskeletal":  ["lameness", "hoof_disease", "joint_swelling", "abnormal_gait"],
        "systemic":         ["fever", "lethargy", "lymph_node_swelling", "weight_loss"],
        "gastrointestinal": ["digestive_disorder", "diarrhea", "abdominal_pain"],
        "respiratory":      ["respiratory_disease", "cough", "nasal_discharge"],
        "reproductive":     ["reproductive_issue", "udder_abnormality"],
    }

    # Temperature for calibration when head is randomly initialized
    CALIBRATION_TEMPERATURE: float = 0.45

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_url: Optional[str] = None,
        num_classes: int = 25,
        image_size: int = 224,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.image_size = image_size
        self.backbone: Optional[nn.Module] = None
        self._head_trained: bool = False  # True only if checkpoint includes disease_head weights

        # Ensure weights are available
        if model_url and checkpoint_path and not os.path.exists(checkpoint_path):
            self._download_weights(checkpoint_path, model_url)

        # FIXED ORDER: backbone → heads → checkpoint (so full state dict can be loaded)
        self._load_backbone(pretrained)
        self._build_veterinary_heads()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self.to(self.device)
        self.eval()

        if not self._head_trained:
            logger.warning(
                "Disease classification head uses random weights — "
                f"applying temperature calibration (T={self.CALIBRATION_TEMPERATURE}). "
                "Fine-tune on labelled cattle data for production use."
            )

        logger.info(
            f"MaxViTDiseaseClassifier ready | device={self.device} | classes={num_classes} | "
            f"head_trained={self._head_trained}"
        )

    # ── Weight management ─────────────────────────────────────────────────────

    @staticmethod
    def _download_weights(path: str, url: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            import gdown
            logger.info(f"Downloading disease model from {url} …")
            gdown.download(url, path, quiet=False)
        except Exception as exc:
            logger.warning(f"Disease model download failed: {exc}")

    # ── Backbone loading ──────────────────────────────────────────────────────

    def _load_backbone(self, pretrained: bool) -> None:
        try:
            import timm
            self.backbone = timm.create_model(
                "maxvit_base_tf_224.in21k", pretrained=pretrained, num_classes=0
            )
            logger.info("MaxViT Base 21k backbone loaded from timm")
        except Exception as exc:
            logger.warning(f"timm MaxViT load failed ({exc}), falling back to ResNet50")
            self._load_resnet50_fallback()

    def _load_resnet50_fallback(self) -> None:
        from torchvision.models import ResNet50_Weights, resnet50
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        base.fc = nn.Identity()
        self.backbone = base

    def _load_checkpoint(self, path: str) -> None:
        """
        Load checkpoint. Attempts full-model load first (including heads).
        Falls back to backbone-only. Sets _head_trained accordingly.
        """
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

            if isinstance(ckpt, nn.Module):
                # Try loading entire module
                try:
                    missing, unexpected = self.load_state_dict(ckpt.state_dict(), strict=False)
                    head_keys = [k for k in ckpt.state_dict() if "disease_head" in k]
                    self._head_trained = len(head_keys) > 0
                except Exception:
                    # Fall back to backbone only
                    self.backbone = ckpt
                    self._head_trained = False

            elif isinstance(ckpt, dict):
                state = ckpt.get("state_dict", ckpt.get("model", ckpt))
                if isinstance(state, dict):
                    # Check if checkpoint contains head weights
                    head_keys = [k for k in state.keys() if "disease_head" in k or "uncertainty_head" in k]
                    if head_keys:
                        # Full model checkpoint — load everything
                        self.load_state_dict(state, strict=False)
                        self._head_trained = True
                        logger.info(f"Full model checkpoint loaded (head_keys={len(head_keys)}): {path}")
                    else:
                        # Backbone-only checkpoint
                        if self.backbone is not None:
                            self.backbone.load_state_dict(state, strict=False)
                        self._head_trained = False
                        logger.info(f"Backbone-only checkpoint loaded: {path}")
            logger.info(f"Disease checkpoint loaded: {path}")
        except Exception as exc:
            logger.warning(f"Checkpoint load error: {exc} — running with pretrained backbone")

    # ── Veterinary heads ──────────────────────────────────────────────────────

    def _build_veterinary_heads(self) -> None:
        feat_dim = getattr(self.backbone, "num_features", None) or 1024
        self.disease_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.category_head = nn.Linear(feat_dim, len(self.DISEASE_HIERARCHY))

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert self.backbone is not None, "Backbone not initialised"
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return {
            "logits":      self.disease_head(features),
            "uncertainty": self.uncertainty_head(features),
            "categories":  self.category_head(features),
            "features":    features,
        }

    # ── Public inference API ──────────────────────────────────────────────────

    def predict(
        self,
        image: Image.Image,
        threshold: float = 0.30,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Run inference on a PIL image.

        Parameters
        ----------
        image:     Input PIL image (any mode, will be converted to RGB).
        threshold: Minimum sigmoid probability to include a disease.
        top_k:     Maximum number of predictions to return.

        Returns
        -------
        predictions, all_probabilities, model_uncertainty, healthy_probability,
        top_category, inference_time_ms, feature_dim, calibrated (bool).
        """
        tensor = pil_to_tensor_normalized(image, self.image_size)
        tensor = tensor.unsqueeze(0).to(self.device)

        self.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.forward(tensor)
            raw_logits   = outputs["logits"]
            uncertainty  = float(outputs["uncertainty"].cpu().item())
            category_scores = torch.softmax(outputs["categories"], dim=1).cpu().numpy()[0]

            # ── Temperature scaling calibration ──────────────────────────────
            if not self._head_trained:
                calibrated_logits = raw_logits / self.CALIBRATION_TEMPERATURE
            else:
                calibrated_logits = raw_logits

            probabilities = torch.sigmoid(calibrated_logits).cpu().numpy()[0]

        # ── Healthy image prior ───────────────────────────────────────────────
        # For uncalibrated heads: if the image looks visually uniform (typical
        # healthy cow coat), slightly boost healthy class probability.
        if not self._head_trained:
            healthy_boost = self._estimate_healthy_prior(image)
            probabilities[0] = float(np.clip(
                probabilities[0] * 0.6 + healthy_boost * 0.4, 0, 1
            ))

        predictions = sorted(
            [
                {
                    "disease":    self.DISEASE_CLASSES[i],
                    "confidence": float(p),
                    "category":   self._get_category(self.DISEASE_CLASSES[i]),
                }
                for i, p in enumerate(probabilities)
                if p >= threshold and i < len(self.DISEASE_CLASSES)
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )

        return {
            "predictions":       predictions[:top_k],
            "all_probabilities": dict(zip(self.DISEASE_CLASSES, probabilities.tolist())),
            "model_uncertainty": uncertainty,
            "healthy_probability": float(probabilities[0]),
            "top_category":      list(self.DISEASE_HIERARCHY.keys())[int(np.argmax(category_scores))],
            "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "feature_dim":       int(outputs["features"].shape[-1]),
            "head_trained":      self._head_trained,
            "calibrated":        not self._head_trained,
            "calibration_note":  (
                f"Temperature scaling applied (T={self.CALIBRATION_TEMPERATURE}). "
                "Fine-tune classification head for production accuracy."
                if not self._head_trained else "Trained classification head used."
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_healthy_prior(image: Image.Image) -> float:
        """
        Estimate the probability that an image shows a healthy animal
        based on simple image statistics (not a trained classifier).
        Returns a scalar in [0, 1].
        """
        try:
            img_rgb = image.convert("RGB").resize((64, 64))
            arr = np.array(img_rgb, dtype=np.float32) / 255.0
            # Std of brightness as proxy for "lesion activity"
            brightness = arr.mean(axis=2)
            std_brightness = float(brightness.std())
            # Very high or very low std → likely pathological (wound, nodule, discharge)
            # Moderate std → likely normal coat
            if std_brightness < 0.08 or std_brightness > 0.35:
                return 0.35  # lower healthy prior
            return 0.55  # moderate healthy prior
        except Exception:
            return 0.45

    def _get_category(self, disease: str) -> str:
        for cat, diseases in self.DISEASE_HIERARCHY.items():
            if disease in diseases:
                return cat
        return "other"
