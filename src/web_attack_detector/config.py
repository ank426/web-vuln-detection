"""Central configuration: attack classes, model defaults, and repo-relative paths."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 3
DEFAULT_SEED = 42
NUM_CLASSES = 4


class AttackClass(IntEnum):
    """Canonical integer label encoding shared by every dataset and checkpoint."""

    BENIGN = 0
    XSS = 1
    CSRF = 2
    SQLI = 3


CLASS_NAMES: tuple[str, ...] = tuple(c.name.lower() for c in AttackClass)
"""Ordered class names; index i must equal ``AttackClass(i).name.lower()``."""

NUM_RULE_FEATURES = 45
"""Width of the concatenated rule-feature vector.

Frozen because trained checkpoints encode this width in
``rule_processor.0.weight``. Asserted by the test suite.
"""


def repo_root() -> Path:
    """Return the repository root (the directory containing ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the git-ignored data root."""
    return repo_root() / "data"


def models_dir() -> Path:
    """Return the git-ignored model checkpoint root."""
    return repo_root() / "models"


def raw_dir() -> Path:
    """Return the raw, never-modified dataset root."""
    return data_dir() / "raw"


def processed_dir() -> Path:
    """Return the root for builder outputs."""
    return data_dir() / "processed"


def external_dir() -> Path:
    """Return the root for downloaded embedding models (GloVe, fastText, USE)."""
    return data_dir() / "external"


def default_model_path() -> Path:
    """Return the conventional DistilBERT checkpoint path."""
    return models_dir() / "web_attack_model.pt"


def label_from_name(name: str) -> int:
    """Map a human-readable class name to its integer label.

    Args:
        name: Class name such as ``"xss"`` or ``"XSS"``.

    Returns:
        The integer label.

    Raises:
        KeyError: If the name is not a known attack class.
    """
    return int(AttackClass[name.strip().upper()])
