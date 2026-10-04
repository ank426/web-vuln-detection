"""Torch dataset wrapping the HuggingFace tokenizer and label validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from web_attack_detector.config import DEFAULT_MAX_LENGTH, NUM_CLASSES
from web_attack_detector.protocols import Tokenizer


class LabelSchemaError(ValueError):
    """Raised when a dataset's labels cannot be used for training."""


def validate_labels(labels: Sequence[int], *, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Coerce labels to a contiguous ``int64`` array and validate their range.

    The trainer feeds labels straight into ``CrossEntropyLoss``; string labels
    therefore fail deep inside torch with an opaque message. Validating here
    turns that into an actionable error.

    Args:
        labels: Raw label values from a dataframe column.
        num_classes: Exclusive upper bound for valid labels.

    Returns:
        Labels as ``int64``.

    Raises:
        LabelSchemaError: If labels are non-numeric or out of range.
    """
    array = np.asarray(labels)
    # np.bool_ is deliberately not a np.number subtype, so it needs handling
    # before the numeric check even though it is perfectly valid as a label.
    if array.dtype == np.bool_:
        return array.astype(np.int64)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        offenders = sorted({str(v) for v in array.ravel()[:20]})[:5]
        raise LabelSchemaError(
            f"Labels must be integers in [0, {num_classes}); got dtype {array.dtype} "
            f"with values such as {offenders}. Encode class names to integers first "
            f"(see `web-attack-detector-build-dataset`).",
        )
    # Reject NaN/inf before casting: astype would silently produce garbage ints.
    if not np.all(np.isfinite(array)):
        raise LabelSchemaError(
            f"Labels contain NaN or infinite values; got {int((~np.isfinite(array)).sum())} "
            f"bad row(s). Drop or impute them before training.",
        )

    as_int = array.astype(np.int64)
    bad = np.setdiff1d(np.unique(as_int), np.arange(num_classes))
    if bad.size:
        raise LabelSchemaError(
            f"Labels must be integers in [0, {num_classes}); found {bad.tolist()}. "
            f"Remap your classes onto the canonical AttackClass encoding.",
        )
    return as_int


class WebAttackDataset(Dataset):
    """Tokenised payloads plus integer labels.

    Args:
        payloads: Raw payload strings.
        labels: Integer labels aligned with ``payloads``.
        tokenizer: A HuggingFace tokenizer.
        max_length: Truncation/padding length.
    """

    def __init__(
        self,
        payloads: Sequence[str],
        labels: Sequence[int],
        tokenizer: Tokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Validate labels eagerly so bad data fails before training starts."""
        self.payloads = [str(p) for p in payloads]
        self.labels = validate_labels(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.payloads)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one tokenised sample.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, ``label`` and the raw
            ``payload`` (the trainer re-extracts rule features from it).
        """
        payload = self.payloads[index]
        encoding = self.tokenizer(
            payload,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": cast("torch.Tensor", encoding["input_ids"]).flatten(),
            "attention_mask": cast("torch.Tensor", encoding["attention_mask"]).flatten(),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "payload": payload,
        }
