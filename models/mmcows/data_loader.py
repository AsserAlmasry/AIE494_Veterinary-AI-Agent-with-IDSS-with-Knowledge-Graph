"""
models/mmcows/data_loader.py
=============================
MMCOWS dataset pipeline — loads multimodal sensor and visual data.
Handles raw data fallbacks and feature engineering for all MMCOWS models.
"""
from __future__ import annotations
import logging, os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

class MMCowsDataPipeline:
    """Loads and serves MMCOWS data for a single day of the 14-day dataset."""

    def __init__(self, mmcows_base_path: str):
        self.base = Path(mmcows_base_path)
        self.preproc = self.base / "preprocessing_results"
        self.visual = self.base / "visual_data"
        self.sensor = self.base / "sensor_data"
        self._merged_df: Optional[pd.DataFrame] = None
        self._milk_df: Optional[pd.DataFrame] = None
        self._thi_df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self):
        try:
            merged_files = list(self.preproc.glob("merged_multimodal_*.csv"))
            if merged_files:
                mp = merged_files[0]
                self._merged_df = pd.read_csv(mp)
                if "datetime" in self._merged_df.columns:
                    self._merged_df["datetime"] = pd.to_datetime(self._merged_df["datetime"])
                logger.info(f"Merged data loaded from {mp.name}: {len(self._merged_df)} rows")
        except Exception as e: logger.warning(f"Merged data load failed: {e}")

        try:
            mk_paths = [self.preproc / "milk_all_clean.csv", self.base / "milk_all_clean.csv"]
            for p in mk_paths:
                if p.exists():
                    self._milk_df = pd.read_csv(p)
                    if "datetime" in self._milk_df.columns:
                        self._milk_df["datetime"] = pd.to_datetime(self._milk_df["datetime"])
                    logger.info(f"Milk data loaded: {len(self._milk_df)} rows")
                    break
        except Exception as e: logger.warning(f"Milk data load failed: {e}")
        
        try:
            tp = self.preproc / "thi_station_avg.csv"
            if tp.exists():
                self._thi_df = pd.read_csv(tp)
                logger.info(f"THI data: {len(self._thi_df)} rows")
        except Exception as e: logger.warning(f"THI data load failed: {e}")

    def get_day_index_from_timestamp(self, ts_str: str) -> int:
        """Parses a timestamp string and finds corresponding day index (0-13)."""
        try:
            import re
            match = re.search(r"(\d{10})", ts_str)
            if match:
                ts = int(match.group(1))
                dt = pd.to_datetime(ts, unit='s').date()
                df = self._milk_df if self._milk_df is not None else self._merged_df
                if df is not None and "datetime" in df.columns:
                    dates = sorted(df["datetime"].dt.date.unique())
                    if dt in dates: return dates.index(dt)
                    return np.argmin([abs((d - dt).days) for d in dates])
        except Exception: pass
        return 0

    def get_sensor_features_for_cow(self, cow_id: int, day_index: int = 0, window_size: int = 30) -> Optional[np.ndarray]:
        """Extract a sensor feature vector (1, 30) for Health and Milk models."""
        tag_str = f"T{cow_id:02d}"
        cow_day_data = pd.DataFrame()
        
        # 1. Try merged data
        if self._merged_df is not None:
            df = self._merged_df
            if "datetime" in df.columns:
                dates = sorted(df["datetime"].dt.date.unique())
                day_data = df[df["datetime"].dt.date == dates[min(day_index, len(dates)-1)]]
            else:
                chunk = max(1, len(df) // 14)
                day_data = df.iloc[day_index * chunk:(day_index + 1) * chunk]
            
            if "tag_id" in day_data.columns and (day_data["tag_id"] == tag_str).any():
                cow_day_data = day_data[day_data["tag_id"] == tag_str]

        # 2. Raw fallback (IMU)
        if cow_day_data.empty:
            try:
                date_part = f"07{21 + day_index:02d}" if day_index < 10 else f"08{day_index - 10 + 1:02d}"
                raw_path = self.sensor / "main_data" / "immu" / tag_str / f"{tag_str}_{date_part}.csv"
                if raw_path.exists():
                    cow_day_data = pd.read_csv(raw_path)
                    logger.info(f"Loaded raw sensor data for {tag_str} from {raw_path.name}")
            except Exception: pass

        # 3. Milk data fallback (all 16 cows have milk data)
        if cow_day_data.empty and self._milk_df is not None:
            cow_col = f"C{cow_id:02d}"
            milk_cow = self._milk_df[self._milk_df["cow_id"] == cow_col]
            if not milk_cow.empty:
                # Use milk yield + DIM as minimal feature set
                milk_num = milk_cow[["milk_weight_kg", "DIM", "milk_norm"]].dropna()
                if not milk_num.empty:
                    cow_day_data = milk_num.iloc[:min(30, len(milk_num))]
                    logger.info(f"Using milk data fallback for {tag_str} (cow {cow_id})")

        if cow_day_data.empty: return None

        # Feature Engineering: Stats over window
        num_data = cow_day_data.select_dtypes(include=[np.number])
        if "timestamp" in num_data.columns: num_data = num_data.drop(columns=["timestamp"])
        
        raw_vals = num_data.values
        if raw_vals.shape[0] < 2: return None
        
        win = raw_vals[-min(window_size, len(raw_vals)):]
        last_val = win[-1] 
        means = win.mean(axis=0)
        stds = win.std(axis=0)
        maxs = win.max(axis=0)
        
        # Concat to 30 dims (assuming ~6 numeric columns + padding)
        feat_base = np.concatenate([last_val, means, stds, maxs])
        out = np.zeros(30, dtype=np.float32)
        take = min(len(feat_base), 28)
        out[:take] = feat_base[:take]
        out[28] = day_index / 14.0
        out[29] = cow_id / 16.0
        
        # Normalize
        v_min, v_max = out.min(), out.max()
        out = (out - v_min) / (v_max - v_min + 1e-6)
        
        return out.reshape(1, 30)

    def get_heat_stress_features(self, cow_id: int, day_index: int = 0, seq_len: int = 24) -> Optional[np.ndarray]:
        """Extract a (1, 24, 19) sequence for HeatStressTransformer."""
        tag_str = f"T{cow_id:02d}"
        cow_data = pd.DataFrame()
        
        # Try raw IMU first as it's more likely to have consistent sequence
        try:
            date_part = f"07{21 + day_index:02d}" if day_index < 10 else f"08{day_index - 10 + 1:02d}"
            raw_path = self.sensor / "main_data" / "immu" / tag_str / f"{tag_str}_{date_part}.csv"
            if raw_path.exists():
                cow_data = pd.read_csv(raw_path)
        except Exception: pass

        if cow_data.empty and self._merged_df is not None:
            # Fallback to merged
            df = self._merged_df
            if "tag_id" in df.columns:
                cow_data = df[df["tag_id"] == tag_str]

        if cow_data.empty: return None

        num_data = cow_data.select_dtypes(include=[np.number])
        if "timestamp" in num_data.columns: num_data = num_data.drop(columns=["timestamp"])
        
        vals = num_data.values
        if len(vals) < 5: return None
        
        # Pad/Truncate
        if len(vals) < seq_len:
            pad = np.zeros((seq_len - len(vals), vals.shape[1]))
            vals = np.vstack([pad, vals])
        else:
            vals = vals[-seq_len:]
            
        out = np.zeros((seq_len, 19), dtype=np.float32)
        take = min(vals.shape[1], 19)
        out[:, :take] = vals[:, :take]
        
        # Normalize
        m_in, m_ax = out.min(), out.max()
        out = (out - m_in) / (m_ax - m_in + 1e-6)
        
        return out[np.newaxis, ...] # (1, 24, 19)

    def get_day_data(self, day_index: int = 0) -> Dict[str, Any]:
        result = {"day_index": day_index, "sensor_data": {}, "images": [], "labels": []}
        img_dirs = sorted((self.visual / "images").glob("*")) if (self.visual / "images").exists() else []
        if img_dirs:
            target_dir = img_dirs[min(day_index, len(img_dirs) - 1)]
            imgs = sorted(target_dir.rglob("*.jpg"))[:20]
            result["images"].extend([str(p) for p in imgs])
        return result

    def get_image_label_pair(self, index: int = 0, day_index: int = 0):
        day_data = self.get_day_data(day_index)
        imgs = day_data.get("images", [])
        if index < len(imgs):
            img_path = imgs[index]
            # Try to find matching label
            label_path = str(Path(img_path).parent.parent.parent.parent / "labels" / "combined" / Path(img_path).parent.parent.name / Path(img_path).parent.name / (Path(img_path).stem + ".txt"))
            if not os.path.exists(label_path): label_path = None
            return img_path, label_path
        return None, None
