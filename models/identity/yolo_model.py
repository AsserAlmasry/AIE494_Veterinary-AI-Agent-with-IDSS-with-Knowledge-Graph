"""
models/identity/yolo_model.py
=============================
Cow Identity Engine — YOLO detection + ViT embedding extraction.
Singleton-safe, lazy loading, GPU/CPU auto-detection.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── ViT Embedding Extractor ──────────────────────────────────────────────────

class ViTEmbeddingExtractor:
    """
    Extracts 768-dim embeddings from cattle images using ViT-base.
    Pure PyTorch — no Colab-specific libraries.
    """

    MODEL_NAME = "google/vit-base-patch16-224"
    EMBEDDING_DIM = 768

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            from transformers import ViTImageProcessor, ViTModel

            logger.info(f"Loading ViT embedding model on {self.device} …")
            self._model = ViTModel.from_pretrained(self.MODEL_NAME).to(self.device).eval()
            self._processor = ViTImageProcessor.from_pretrained(self.MODEL_NAME)
            self._available = True
            logger.info("ViT embedding extractor ready (dim=768)")
        except Exception as exc:
            logger.warning(f"ViT load failed: {exc} — using random fallback")
            self._available = False

    def extract(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract a unit-normalised 768-dim embedding from an RGB uint8 ndarray.
        Falls back to random unit vector if model unavailable.
        """
        if not self._available or self._model is None:
            return self._random_embedding()

        try:
            inputs = self._processor(
                images=image_array,
                return_tensors="pt",
                do_rescale=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            norm = np.linalg.norm(emb) + 1e-8
            return (emb / norm).astype(np.float32)
        except Exception as exc:
            logger.warning(f"Embedding extraction failed: {exc}")
            return self._random_embedding()

    def _random_embedding(self) -> np.ndarray:
        emb = np.random.randn(self.EMBEDDING_DIM).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)


# ── YOLO Detector ────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Wraps the custom YOLO identity model for cow detection.
    Compatible with Ultralytics 8.4+ API (uses `confidence=` param).
    """

    CONF_THRESHOLD = 0.60

    def __init__(self, model_path: str, device: str) -> None:
        self.device = device
        self._model = None
        if os.path.exists(model_path):
            self._load(model_path)
        else:
            logger.warning(f"YOLO model not found at {model_path} — detection disabled")

    def _load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                self._model = checkpoint.get("model", checkpoint)
            elif hasattr(checkpoint, "model"):
                self._model = checkpoint.model
            else:
                self._model = checkpoint
            if self._model is not None:
                self._model.to(self.device).eval()
                logger.info("YOLO identity model loaded successfully")
        except Exception as exc:
            logger.warning(f"YOLO load failed: {exc}")
            self._model = None

    def detect(self, image_array: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Run YOLO detection.  Returns (cow_class_id, confidence) or (None, 0.0).
        """
        if self._model is None:
            return None, 0.0
        try:
            if hasattr(self._model, "predict"):
                results = self._model.predict(
                    image_array,
                    confidence=self.CONF_THRESHOLD,  # Ultralytics 8.4+ param
                    verbose=False,
                    save=False,
                    stream=False,
                )
                if results and len(results[0].boxes) > 0:
                    best = results[0].boxes[0]
                    return int(best.cls[0].item()) + 1, float(best.conf[0].item())
        except Exception as exc:
            logger.debug(f"YOLO inference error: {exc}")
        return None, 0.0


# ── Main Identity Engine ──────────────────────────────────────────────────────

class CowIdentityEngine:
    """
    Two-stage cow identification:
      1. YOLO model → direct class prediction
      2. ViT embedding → cosine similarity search against known-cow bank

    Follows singleton pattern — instantiate once via dependencies.py.
    """

    KNOWN_COW_IDS: List[int] = list(range(1, 17))
    SIMILARITY_THRESHOLD: float = 0.85

    def __init__(
        self,
        model_path: str,
        model_url: str,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_weights(model_path, model_url)
        self.yolo = YOLODetector(model_path, self.device)
        self.embedder = ViTEmbeddingExtractor(self.device)
        # Lightweight in-memory embedding bank for quick identity check
        self._known_embeddings: Dict[int, List[np.ndarray]] = {}
        self._init_reference_embeddings()
        logger.info(
            f"CowIdentityEngine ready | device={self.device} | "
            f"known_cows={len(self.KNOWN_COW_IDS)}"
        )

    # ── Weight management ─────────────────────────────────────────────────────

    @staticmethod
    def _ensure_weights(path: str, url: str) -> None:
        if os.path.exists(path):
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            import gdown
            logger.info(f"Downloading identity weights from {url} …")
            gdown.download(url, path, quiet=False)
        except Exception as exc:
            logger.warning(f"Weight download failed: {exc}. Running without YOLO.")

    # ── Reference bank initialisation ─────────────────────────────────────────

    def _init_reference_embeddings(self) -> None:
        """
        Seed the in-memory reference bank with random-but-stable embeddings
        for the 16 known MMCows IDs.  In production, replace/augment with
        real embeddings extracted from the mmcows dataset.
        """
        np.random.seed(42)
        dim = self.embedder.EMBEDDING_DIM
        for cid in self.KNOWN_COW_IDS:
            base = np.random.randn(dim).astype(np.float32)
            base /= np.linalg.norm(base)
            variations = [
                base + np.random.normal(0, 0.05, dim).astype(np.float32)
                for _ in range(3)
            ]
            self._known_embeddings[cid] = [
                v / (np.linalg.norm(v) + 1e-8) for v in variations
            ]

    # ── Public API ────────────────────────────────────────────────────────────

    def identify(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Identify a cow from an RGB uint8 image array.

        Returns:
            cow_id, decision ("known_cow"/"new_cow"), confidence, method,
            similarity_score, latency_ms, matched_reference.
        """
        t0 = time.perf_counter()

        # Stage 1 — YOLO
        yolo_id, yolo_conf = self.yolo.detect(image_array)

        # Stage 2 — embedding similarity
        embedding = self.embedder.extract(image_array)
        best_sim, best_cid = self._cosine_search(embedding)

        # Decision logic
        if (
            yolo_id is not None
            and yolo_id in self.KNOWN_COW_IDS
            and yolo_conf >= self.SIMILARITY_THRESHOLD
        ):
            final_id = yolo_id
            decision = "known_cow"
            confidence = yolo_conf
            method = "yolo_direct"
        elif best_sim >= self.SIMILARITY_THRESHOLD:
            final_id = best_cid
            decision = "known_cow"
            confidence = best_sim
            method = "embedding_similarity"
        else:
            final_id = max(self.KNOWN_COW_IDS) + 1 if self.KNOWN_COW_IDS else 17
            decision = "new_cow"
            confidence = float(1.0 - best_sim)
            method = "new_cow_assignment"

        return {
            "cow_id": int(final_id),
            "decision": decision,
            "confidence": float(confidence),
            "similarity_score": float(best_sim),
            "method": method,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "matched_reference": {
                "cow_id": best_cid,
                "similarity": float(best_sim),
            } if best_cid != -1 else None,
            "manual_override_allowed": True,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cosine_search(self, query: np.ndarray) -> Tuple[float, int]:
        best_sim, best_cid = -1.0, -1
        for cid, embeddings in self._known_embeddings.items():
            for emb in embeddings:
                sim = float(np.dot(query, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_cid = cid
        return best_sim, best_cid

    def add_embedding(self, cow_id: int, embedding: np.ndarray) -> None:
        """Add a new embedding to the in-memory reference bank."""
        if cow_id not in self._known_embeddings:
            self._known_embeddings[cow_id] = []
        self._known_embeddings[cow_id].append(embedding)
        if cow_id not in self.KNOWN_COW_IDS:
            self.KNOWN_COW_IDS.append(cow_id)
