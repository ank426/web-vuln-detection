"""Tests for label validation and the dataset wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from web_attack_detector.dataset import LabelSchemaError, WebAttackDataset, validate_labels


class DummyTokenizer:
    """Minimal tokenizer stand-in returning fixed-width tensors.

    Matches the ``Tokenizer`` protocol structurally so the dataset can be
    exercised without downloading DistilBERT's vocabulary.
    """

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        truncation: bool = False,
        padding: str = "max_length",
        max_length: int = 8,
        return_tensors: str = "pt",
    ) -> Mapping[str, object]:
        """Return constant tensors of the requested width, ignoring the text."""
        del text, truncation, padding, return_tensors  # the stub does not tokenise
        return {
            "input_ids": np.ones((1, max_length), dtype=np.int64),
            "attention_mask": np.ones((1, max_length), dtype=np.int64),
        }


def test_validate_labels_accepts_canonical_integers() -> None:
    """0..3 passes through as int64."""
    out = validate_labels([0, 1, 2, 3, 1, 1])
    assert out.dtype == np.int64
    assert out.tolist() == [0, 1, 2, 3, 1, 1]


def test_validate_labels_accepts_float_and_bool() -> None:
    """Numeric dtypes other than int are coerced."""
    assert validate_labels([0.0, 1.0, 2.0, 3.0]).tolist() == [0, 1, 2, 3]  # ty: ignore[invalid-argument-type]  # deliberately wrong type: asserts runtime coercion
    assert validate_labels([False, True, True, False]).tolist() == [0, 1, 1, 0]


def test_validate_labels_rejects_string_labels() -> None:
    """String class names produce an actionable error, not a torch crash."""
    with pytest.raises(LabelSchemaError, match="must be integers"):
        validate_labels(["benign", "xss", "csrf", "sqli"])  # ty: ignore[invalid-argument-type]  # deliberately wrong type: asserts runtime coercion


def test_validate_labels_rejects_out_of_range() -> None:
    """Labels outside 0..3 are rejected with the offending values listed."""
    with pytest.raises(LabelSchemaError, match=r"\[7\]"):
        validate_labels([0, 1, 7, 3])


def test_validate_labels_rejects_all_nan() -> None:
    """A fully-null label column is rejected."""
    with pytest.raises(LabelSchemaError):
        validate_labels([np.nan, np.nan, np.nan, np.nan])  # ty: ignore[invalid-argument-type]  # deliberately wrong type: asserts runtime coercion


def test_dataset_rejects_bad_labels_at_construction() -> None:
    """Validation happens eagerly, not on first access."""
    with pytest.raises(LabelSchemaError):
        WebAttackDataset(["a", "b"], ["benign", "xss"], DummyTokenizer())  # ty: ignore[invalid-argument-type]  # deliberately wrong type: asserts runtime coercion


def test_dataset_item_shapes_and_types() -> None:
    """A sample yields correctly shaped tensors plus the raw payload."""
    dataset = WebAttackDataset(
        ["<script>x</script>", "GET /"],
        [1, 0],
        DummyTokenizer(),
        max_length=16,
    )
    assert len(dataset) == 2

    item = dataset[0]
    assert item["input_ids"].shape == (16,)
    assert item["attention_mask"].shape == (16,)
    assert item["label"].dtype.is_floating_point is False
    assert int(item["label"]) == 1
    assert item["payload"] == "<script>x</script>"


def test_dataset_stringifies_payloads() -> None:
    """Non-string payloads are coerced so tokenisation never receives an int."""
    dataset = WebAttackDataset([123, None], [0, 1], DummyTokenizer())  # ty: ignore[invalid-argument-type]  # deliberately wrong type: asserts runtime coercion
    assert dataset.payloads == ["123", "None"]


def test_dataset_respects_max_length() -> None:
    """max_length controls the tensor width."""
    for width in (8, 32, 128):
        dataset = WebAttackDataset(
            ["x"], [0], DummyTokenizer(), max_length=width
        )  # deliberately wrong type: asserts runtime coercion
        assert dataset[0]["input_ids"].shape == (width,)


def test_validate_labels_rejects_inf() -> None:
    """Infinite label values are rejected before the int cast."""
    with pytest.raises(LabelSchemaError, match="NaN or infinite"):
        validate_labels([0.0, np.inf, 2.0, 3.0])  # ty: ignore[invalid-argument-type]  # deliberately wrong type: asserts runtime coercion
