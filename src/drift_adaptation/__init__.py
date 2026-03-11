"""Drift Adaptation Techniques for Explainable AI.

This package provides comprehensive tools for detecting and adapting to concept drift
in machine learning models, with a focus on explainability and interpretability.
"""

__version__ = "1.0.0"
__author__ = "XAI Research Team"

from .detectors import DriftDetector, PSIDetector, KSDetector, MMDDetector
from .adapters import ModelAdapter, OnlineLearningAdapter, EnsembleAdapter
from .data import generate_synthetic_data, load_dataset
from .metrics import DriftMetrics, AdaptationMetrics
from .viz import DriftVisualizer

__all__ = [
    "DriftDetector",
    "PSIDetector", 
    "KSDetector",
    "MMDDetector",
    "ModelAdapter",
    "OnlineLearningAdapter",
    "EnsembleAdapter",
    "generate_synthetic_data",
    "load_dataset",
    "DriftMetrics",
    "AdaptationMetrics",
    "DriftVisualizer",
]
