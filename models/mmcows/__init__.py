"""
models/mmcows/
==============
Production wrappers around the real MMCOWS research models.

Modules
-------
- cow_identifier     : CowReIDModel (ViT + ArcFace, 16 cows)
- milk_predictor     : TimeSeriesTransformer (sensor → yield)
- heat_stress_analyzer : THI data + BehaviorCNNLSTM
- health_scorer      : MultiModalFusion + SensorAutoencoder
- data_loader        : MMCOWS dataset pipeline
"""

from models.mmcows.cow_identifier import CowIdentifier
from models.mmcows.milk_predictor import MilkProductivityPredictor
from models.mmcows.heat_stress_analyzer import HeatStressAnalyzer
from models.mmcows.health_scorer import HealthScorer
from models.mmcows.data_loader import MMCowsDataPipeline

__all__ = [
    "CowIdentifier",
    "MilkProductivityPredictor",
    "HeatStressAnalyzer",
    "HealthScorer",
    "MMCowsDataPipeline",
]
