"""Hybrid transformer + rule-based web attack detection (XSS / CSRF / SQLi)."""

from __future__ import annotations

from web_attack_detector.config import (
    CLASS_NAMES,
    NUM_RULE_FEATURES,
    AttackClass,
    default_model_path,
    models_dir,
)
from web_attack_detector.dataset import LabelSchemaError, WebAttackDataset, validate_labels
from web_attack_detector.features import (
    extract_csrf_features,
    extract_rule_features,
    extract_sqli_features,
    extract_xss_features,
)
from web_attack_detector.model import HybridAttackDetector
from web_attack_detector.sample_data import create_sample_data
from web_attack_detector.trainer import (
    EpochMetrics,
    EvaluationResult,
    Prediction,
    WebAttackTrainer,
)

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "NUM_RULE_FEATURES",
    "AttackClass",
    "EpochMetrics",
    "EvaluationResult",
    "HybridAttackDetector",
    "LabelSchemaError",
    "Prediction",
    "WebAttackDataset",
    "WebAttackTrainer",
    "__version__",
    "create_sample_data",
    "default_model_path",
    "extract_csrf_features",
    "extract_rule_features",
    "extract_sqli_features",
    "extract_xss_features",
    "models_dir",
    "validate_labels",
]
