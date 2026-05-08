import app.numpy_hack
import logging
import math
import os
import time
import pickle
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

from models.mmcows.health_transformer import HealthRiskTransformer


class HealthScorer:
    """
    Production wrapper for HealthRiskTransformer (Task 5).
    Predicts current and future (24h) health risk scores based on 48h sensor windows.
    """
    
    FEATURE_COLS = [
        'cbt', 'cbt_dev', 'cbt_trend', 'cbt_r6', 'cbt_r24',
        'milk_kg', 'milk_drop_pct', 'milk_kg_r6', 'milk_kg_r24',
        'accel_mag', 'activity_drop', 'accel_mag_r6',
        'lying_frac',
        'health_score_r6', 'health_score_r24',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'has_event'
    ]

    def __init__(self, checkpoint_path: str, scaler_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self._available = False
        self._load_assets(checkpoint_path, scaler_path)
        logger.info(f"HealthScorer (Transformer) ready | available={self._available}")

    def _load_assets(self, checkpoint_path: str, scaler_path: str):
        try:
            # 1. Load Scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Health scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Health scaler not found at {scaler_path}")

            # 2. Instantiate Model
            self.model = HealthRiskTransformer(
                in_dim=len(self.FEATURE_COLS),
                d_model=128,
                n_heads=8,
                n_layers=4,
                dropout=0.2,
                forecast_h=24
            )

            # 3. Load Checkpoint
            if os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # Extract state dict if nested
                if "model_state" in state:
                    state = state["model_state"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                
                # Handle DataParallel wrap if present in checkpoint
                if any(k.startswith('module.') for k in state.keys()):
                    state = {k.replace('module.', ''): v for k, v in state.items()}
                
                self.model.load_state_dict(state, strict=False)
                self.model.to(self.device).eval()
                logger.info(f"Health model loaded from {checkpoint_path}")
                self._available = True
            else:
                logger.warning(f"Health model not found at {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load HealthScorer assets: {e}", exc_info=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implements Cell 5 engineering logic for a single cow's window.
        Input df should have: timestamp, hour, day, cbt, milk_kg, accel_mag, lying_frac, health_score, has_event
        """
        df = df.copy()
        
        # Ensure critical columns exist for calculation
        if 'hour' not in df.columns: df['hour'] = pd.Timestamp.now().hour
        if 'day' not in df.columns: df['day'] = 0
        if 'cbt' not in df.columns: df['cbt'] = 38.5
        if 'milk_kg' not in df.columns: df['milk_kg'] = 22.0
        if 'accel_mag' not in df.columns: df['accel_mag'] = 2.0
        if 'lying_frac' not in df.columns: df['lying_frac'] = 0.5
        if 'health_score' not in df.columns: df['health_score'] = 50.0

        # Ensure sorted
        sort_col = next((c for c in df.columns if c.lower() in ('timestamp', 'datetime', 'time')), None)
        if sort_col:
            df = df.sort_values(sort_col).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        
        # Cyclical time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'].astype(float) / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'].astype(float) / 24)
        df['day_sin']  = np.sin(2 * np.pi * (df['day'].astype(float) % 14) / 14)
        df['day_cos']  = np.cos(2 * np.pi * (df['day'].astype(float) % 14) / 14)

        # Baselines (for dev/drop features)
        # In a real window, we might use global means or window means
        cbt_b = 38.5
        milk_b = 22.0
        accel_b = 2.0
        
        df['cbt_dev'] = df['cbt'] - cbt_b
        df['cbt_trend'] = df['cbt'].diff().fillna(0)
        df['milk_drop_pct'] = ((milk_b - df['milk_kg']) / (milk_b + 1e-6)).clip(0, 1) * 100
        df['activity_drop'] = ((accel_b - df['accel_mag']) / (accel_b + 1e-6)).clip(0, 1) * 100

        # Rolling averages
        for col in ['cbt', 'milk_kg', 'accel_mag', 'health_score']:
            if col in df.columns:
                df[f'{col}_r6']  = df[col].rolling(6,  min_periods=1).mean()
                df[f'{col}_r24'] = df[col].rolling(24, min_periods=1).mean()
        
        return df

    def predict(
        self,
        cow_id: str,
        history_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict health risk using 48-hour history.
        history_df: DataFrame with raw sensor columns.
        """
        t0 = time.perf_counter()

        if not self._available:
            return {"error": "Model/Scaler not loaded", "status": "error"}

        try:
            # 1. Feature Engineering
            processed = self.engineer_features(history_df)
            
            # 2. Select and Scale Features
            X = processed[self.FEATURE_COLS].fillna(0).values
            if self.scaler:
                X = self.scaler.transform(X)
            
            # 3. Ensure sequence length (48)
            if len(X) < 48:
                # Pad start with zeros
                pad = np.zeros((48 - len(X), len(self.FEATURE_COLS)))
                X = np.vstack([pad, X])
            else:
                X = X[-48:] # Take last 48 hours
            
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

            # 4. Inference
            with torch.no_grad():
                score_now, score_future, feat_w = self.model(X_tensor)
                
                now_val = float(score_now.cpu().item())
                future_vals = score_future.cpu().numpy().flatten().tolist()
                feat_attn = feat_w.cpu().numpy().flatten().tolist()

            # 5. Clinical Heuristic Overrides (Safety Layer)
            # Ensure extreme vitals always trigger high risk even if model is biased
            last_cbt = float(history_df.iloc[-1]['cbt'])
            last_milk_drop = float(processed.iloc[-1].get('milk_drop_pct', 0))
            
            # Fever heuristics
            if last_cbt >= 40.5:
                now_val = max(now_val, 85.0) # Critical
                logger.info(f"Heuristic Trigger: High Fever ({last_cbt}) -> Risk boosted to {now_val}")
            elif last_cbt >= 39.5:
                now_val = max(now_val, 60.0) # At-Risk
                logger.info(f"Heuristic Trigger: Fever ({last_cbt}) -> Risk boosted to {now_val}")
                
            # Milk drop heuristics
            if last_milk_drop >= 40.0:
                now_val = max(now_val, 70.0)
                logger.info(f"Heuristic Trigger: Severe Milk Drop ({last_milk_drop}%) -> Risk boosted to {now_val}")

            # Interpretation
            risk_level = "Healthy"
            if now_val >= 75: risk_level = "Critical"
            elif now_val >= 50: risk_level = "At-Risk"
            elif now_val >= 25: risk_level = "Watch"

            return {
                "cow_id": cow_id,
                "current_risk": round(now_val, 2),
                "forecast_24h": [round(v, 2) for v in future_vals],
                "risk_level": risk_level,
                "feature_importance": dict(zip(self.FEATURE_COLS, [round(f, 4) for f in feat_attn])),
                "recommendations": self._get_recs(risk_level, now_val),
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Health prediction failed for cow {cow_id}: {e}")
            return {"error": str(e), "status": "error"}

    def _get_recs(self, level: str, score: float) -> List[str]:
        if level == "Critical": return ["🔴 CRITICAL RISK. Immediate veterinary intervention required."]
        if level == "At-Risk": return ["🟠 AT-RISK. Clinical signs likely; schedule inspection today."]
        if level == "Watch": return ["🟡 WATCH. Behavioral deviations detected; monitor closely."]
        return ["🟢 HEALTHY. No significant health risks detected."]
