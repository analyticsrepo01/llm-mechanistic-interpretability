"""
LLM Mechanistic Interpretability & Neural Steering Framework
============================================================
A production-grade Python library for reverse-engineering transformer internals,
inspecting residual stream representations, and steering model behaviors at runtime.
"""

from llm_interpretability.hooks import HookManager, LayerActivation
from llm_interpretability.logit_lens import LogitLens, LogitLensResult
from llm_interpretability.outliers import OutlierDetector, OutlierReport
from llm_interpretability.cka import CKAAnalyzer
from llm_interpretability.attribution import DirectLogitAttribution
from llm_interpretability.steering import ActivationSteering, SteeringHook
from llm_interpretability.geometry import RepresentationGeometry

__version__ = "0.1.0"

__all__ = [
    "HookManager",
    "LayerActivation",
    "LogitLens",
    "LogitLensResult",
    "OutlierDetector",
    "OutlierReport",
    "CKAAnalyzer",
    "DirectLogitAttribution",
    "ActivationSteering",
    "SteeringHook",
    "RepresentationGeometry",
]
