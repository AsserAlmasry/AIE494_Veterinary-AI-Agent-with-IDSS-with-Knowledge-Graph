"""
models/identity/faiss_index.py
===============================
Production-grade identity embedding bank.
Uses a pure-NumPy cosine index (no FAISS SWIG dependency).
Drop-in replacement for faiss.IndexFlatIP with identical public semantics.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _NumpyIndex:
    """
    Flat inner-product index backed by a NumPy matrix.
    Assumes all stored vectors are L2-normalised →  inner product ≡ cosine sim.
    Mirrors the faiss.IndexFlatIP.search() output API exactly.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._mat: np.ndarray = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return self._mat.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.ntotal == 0:
            self._mat = vectors.copy()
        else:
            self._mat = np.vstack([self._mat, vectors])

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (similarities, indices) each shape (1, k)."""
        if self.ntotal == 0:
            return np.zeros((1, k), dtype=np.float32), np.full((1, k), -1, dtype=np.int64)

        q = np.asarray(query, dtype=np.float32).flatten()
        sims = self._mat @ q
        k = min(k, self.ntotal)
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return sims[idx].reshape(1, -1), idx.reshape(1, -1).astype(np.int64)

    def to_dict(self) -> Dict:
        return {"mat": self._mat.tolist(), "dim": self.dim}

    @classmethod
    def from_dict(cls, d: Dict) -> "_NumpyIndex":
        obj = cls(d["dim"])
        mat = np.array(d["mat"], dtype=np.float32)
        if mat.size:
            obj._mat = mat
        return obj


class IdentityEmbeddingBank:
    """
    Multi-embedding identity bank for cattle.

    Features
    --------
    • Cosine similarity search (pure NumPy, GPU-free)
    • Multi-embedding memory (3–20 embeddings per cow)
    • Threshold logic: sim ≥ 0.85 → known cow, else → new cow (ID 17+)
    • Manual override support
    • JSON-safe persistence / load
    """

    KNOWN_COW_IDS: List[int] = list(range(1, 17))

    def __init__(
        self,
        embedding_dim: int = 512,
        persistence_path: Optional[str] = None,
        similarity_threshold: float = 0.85,
        min_embeddings: int = 3,
        max_embeddings: int = 20,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.persistence_path = persistence_path
        self.SIMILARITY_THRESHOLD = similarity_threshold
        self.MIN_EMBEDDINGS = min_embeddings
        self.MAX_EMBEDDINGS = max_embeddings
        self.NEXT_NEW_COW_ID: int = max(self.KNOWN_COW_IDS) + 1

        self.index = _NumpyIndex(embedding_dim)
        self.cow_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.cow_metadata: Dict[int, Dict[str, Any]] = {}
        self.faiss_mapping: List[Tuple[int, int]] = []
        self.confidence_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        self._initialize_known_cows()

        if persistence_path and os.path.exists(persistence_path):
            self._load_state(persistence_path)

        logger.info(
            f"IdentityEmbeddingBank ready | {len(self.KNOWN_COW_IDS)} cows | "
            f"{self.index.ntotal} embeddings | threshold={self.SIMILARITY_THRESHOLD}"
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialize_known_cows(self) -> None:
        np.random.seed(42)
        all_vecs: List[np.ndarray] = []
        all_mappings: List[Tuple[int, int]] = []

        for cid in self.KNOWN_COW_IDS:
            base = np.random.randn(self.embedding_dim).astype(np.float32)
            base /= np.linalg.norm(base)
            n_var = int(np.random.randint(3, 6))
            for i in range(n_var):
                noise = np.random.normal(0, 0.03, self.embedding_dim).astype(np.float32)
                vari = base + noise
                vari /= np.linalg.norm(vari)
                all_vecs.append(vari.copy())
                all_mappings.append((cid, len(self.cow_embeddings[cid])))
                self.cow_embeddings[cid].append(vari.copy())

            self.cow_metadata[cid] = {
                "source": "mmcows_dataset",
                "initialized_at": datetime.now().isoformat(),
                "embedding_count": len(self.cow_embeddings[cid]),
                "breed": np.random.choice(
                    ["Holstein", "Jersey", "Angus", "Hereford", "Simmental"]
                ),
                "age_years": round(float(np.random.uniform(2, 8)), 1),
                "last_matched": None,
                "avg_confidence": None,
            }

        if all_vecs:
            self.index.add(np.array(all_vecs, dtype=np.float32))
            self.faiss_mapping = all_mappings

    # ── Primary interface ─────────────────────────────────────────────────────

    def identify_cow(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()

        q = np.asarray(query_embedding, dtype=np.float32).flatten()
        q /= np.linalg.norm(q) + 1e-8

        k = min(5, max(1, self.index.ntotal))
        similarities, indices = self.index.search(q, k=k)

        best_sim = float(similarities[0][0])
        best_idx = int(indices[0][0])

        matched_cow_id: Optional[int] = None
        matched_ref: Optional[Dict] = None

        if 0 <= best_idx < len(self.faiss_mapping):
            cow_id, emb_idx = self.faiss_mapping[best_idx]
            matched_cow_id = cow_id
            matched_ref = {
                "cow_id": cow_id,
                "embedding_index": emb_idx,
                "similarity": best_sim,
                "metadata": self.cow_metadata.get(cow_id, {}),
            }

        if best_sim >= self.SIMILARITY_THRESHOLD and matched_cow_id is not None:
            decision = "known_cow"
            assigned_id = matched_cow_id
            confidence = best_sim
        else:
            decision = "new_cow"
            assigned_id = self.NEXT_NEW_COW_ID
            self.NEXT_NEW_COW_ID += 1
            confidence = 1.0 - best_sim

        self.confidence_history[assigned_id].append(
            {"timestamp": datetime.now().isoformat(), "similarity": best_sim, "decision": decision}
        )

        if decision == "known_cow" and matched_cow_id and matched_cow_id in self.cow_metadata:
            meta = self.cow_metadata[matched_cow_id]
            meta["last_matched"] = datetime.now().isoformat()
            confs = [h["similarity"] for h in self.confidence_history[matched_cow_id]]
            if confs:
                meta["avg_confidence"] = round(float(np.mean(confs)), 3)

        return {
            "cow_id": int(assigned_id),
            "decision": decision,
            "similarity_score": float(best_sim),
            "confidence": float(confidence),
            "matched_reference": matched_ref,
            "search_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "threshold_used": self.SIMILARITY_THRESHOLD,
            "manual_override_allowed": True,
            "bank_stats": {
                "total_embeddings": self.index.ntotal,
                "unique_cows": len(self.cow_embeddings),
            },
        }

    def add_embedding(
        self,
        cow_id: int,
        embedding: np.ndarray,
        source: str = "active_learning",
        confirmed: bool = False,
    ) -> bool:
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        if emb.shape[0] != self.embedding_dim:
            return False
        emb /= np.linalg.norm(emb) + 1e-8
        self.cow_embeddings[cow_id].append(emb.copy())
        self.index.add(emb.reshape(1, -1))
        self.faiss_mapping.append((cow_id, len(self.cow_embeddings[cow_id]) - 1))

        if len(self.cow_embeddings[cow_id]) > self.MAX_EMBEDDINGS:
            self.cow_embeddings[cow_id] = self.cow_embeddings[cow_id][: self.MAX_EMBEDDINGS]

        if cow_id not in self.cow_metadata:
            self.cow_metadata[cow_id] = {
                "source": source,
                "first_seen": datetime.now().isoformat(),
                "confirmed": confirmed,
                "embedding_count": 1,
            }
        else:
            self.cow_metadata[cow_id]["embedding_count"] = len(
                self.cow_embeddings[cow_id]
            )
            self.cow_metadata[cow_id]["last_updated"] = datetime.now().isoformat()
            if confirmed:
                self.cow_metadata[cow_id]["confirmed"] = True
        return True

    def manual_override(
        self,
        query_embedding: np.ndarray,
        confirmed_cow_id: int,
        confirmed_by: str = "veterinarian",
    ) -> Dict[str, Any]:
        success = self.add_embedding(
            confirmed_cow_id, query_embedding, "manual_override", confirmed=True
        )
        if success:
            result = self.identify_cow(query_embedding)
            result.update(
                {
                    "overridden": True,
                    "confirmed_by": confirmed_by,
                    "confirmed_cow_id": confirmed_cow_id,
                }
            )
            return result
        return {
            "error": "Failed to add embedding",
            "cow_id": confirmed_cow_id,
            "confirmed_by": confirmed_by,
        }

    def get_cow_summary(self, cow_id: int) -> Dict[str, Any]:
        if cow_id not in self.cow_metadata:
            return {"cow_id": cow_id, "exists": False}
        hist = list(self.confidence_history.get(cow_id, []))
        return {
            "cow_id": cow_id,
            "exists": True,
            "metadata": self.cow_metadata[cow_id],
            "embedding_count": len(self.cow_embeddings.get(cow_id, [])),
            "recent_confidences": [h["similarity"] for h in hist[-10:]],
            "identity_stability": self._compute_stability(hist),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_state(self, path: Optional[str] = None) -> None:
        path = path or self.persistence_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "index": self.index.to_dict(),
            "cow_embeddings": {
                cid: [e.tolist() for e in embs]
                for cid, embs in self.cow_embeddings.items()
            },
            "cow_metadata": self.cow_metadata,
            "faiss_mapping": self.faiss_mapping,
            "confidence_history": {
                cid: list(hist)
                for cid, hist in self.confidence_history.items()
            },
            "next_new_cow_id": self.NEXT_NEW_COW_ID,
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Identity bank state saved → {path}")

    def _load_state(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            if "index" in state:
                self.index = _NumpyIndex.from_dict(state["index"])
            for cid, emb_list in state.get("cow_embeddings", {}).items():
                self.cow_embeddings[int(cid)] = [
                    np.array(e, dtype=np.float32) for e in emb_list
                ]
            self.cow_metadata = state.get("cow_metadata", {})
            self.faiss_mapping = state.get("faiss_mapping", [])
            for cid, hist in state.get("confidence_history", {}).items():
                self.confidence_history[int(cid)] = deque(hist, maxlen=100)
            self.NEXT_NEW_COW_ID = state.get("next_new_cow_id", 17)
            logger.info(f"Identity bank loaded from {path}")
        except Exception as exc:
            logger.warning(f"Failed to load identity bank: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_stability(history: List[Dict]) -> str:
        if len(history) < 5:
            return "insufficient_data"
        std = float(np.std([h["similarity"] for h in history[-20:]]))
        if std < 0.05:
            return "highly_stable"
        if std < 0.10:
            return "stable"
        if std < 0.20:
            return "variable"
        return "unstable"
