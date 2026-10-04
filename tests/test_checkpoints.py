"""Compatibility tests against the shipped checkpoints.

These assert the *contract* between the feature extractors, the model and the
weights on disk, without instantiating DistilBERT (which would require a
multi-hundred-megabyte download). Skipped when ``models/`` is absent, since it is
git-ignored.
"""

from __future__ import annotations

import pytest
import torch

from web_attack_detector.config import NUM_CLASSES, NUM_RULE_FEATURES, models_dir

DISTILBERT_HIDDEN = 768
HIDDEN_SIZE = 256


def _distilbert_state() -> dict[str, torch.Tensor]:
    path = models_dir() / "web_attack_model.pt"
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if state is None:
        pytest.skip("checkpoint has no model_state_dict")
    return state


def test_rule_feature_width_matches_checkpoint() -> None:
    """The 45-dim contract the extractors must keep."""
    state = _distilbert_state()
    assert state["rule_processor.0.weight"].shape[1] == NUM_RULE_FEATURES


def test_classifier_input_width_matches_fusion() -> None:
    """classifier.0 consumes transformer_hidden + hidden_size // 4."""
    state = _distilbert_state()
    expected = DISTILBERT_HIDDEN + HIDDEN_SIZE // 4
    assert state["classifier.0.weight"].shape[1] == expected


def test_classifier_output_matches_num_classes() -> None:
    """The head emits NUM_CLASSES logits."""
    state = _distilbert_state()
    assert state["classifier.6.weight"].shape[0] == NUM_CLASSES


def test_checkpoint_class_names_are_canonical() -> None:
    """Stored class names match the canonical AttackClass ordering."""
    path = models_dir() / "web_attack_model.pt"
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert list(checkpoint["class_names"]) == ["benign", "xss", "csrf", "sqli"]


def test_uniembed_checkpoint_matches_ported_architecture() -> None:
    """The verbatim-ported MLP still fits the shipped uniembed weights."""
    path = models_dir() / "uniembed" / "mlp_xss_csrf_sqli_detector.pth"
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")

    from web_attack_detector.uniembed.model import INPUT_SIZE

    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["layer1.weight"].shape[1] == INPUT_SIZE
    assert state["layer1.weight"].shape[0] == 256
    assert state["layer2.weight"].shape == (128, 256)
    assert state["layer3.weight"].shape == (64, 128)
    assert state["output_layer.weight"].shape == (NUM_CLASSES, 64)


def test_uniembed_feature_block_widths() -> None:
    """157 rule features + 912 embedding features == 1069."""
    from web_attack_detector.uniembed.features import (
        CSRF_RULE_BASED_SIZE,
        RULE_FEATURES_TOTAL,
        SQLI_RULE_BASED_SIZE,
        XSS_RULE_BASED_SIZE,
    )
    from web_attack_detector.uniembed.model import (
        EMBEDDING_DIMS,
        INPUT_SIZE,
    )

    assert (XSS_RULE_BASED_SIZE, CSRF_RULE_BASED_SIZE, SQLI_RULE_BASED_SIZE) == (68, 52, 37)
    assert RULE_FEATURES_TOTAL == 157
    assert EMBEDDING_DIMS == 100 + 300 + 512
    assert INPUT_SIZE == 1069


def test_the_two_feature_spaces_are_distinct() -> None:
    """The 45-dim and 157-dim spaces must not be confused for one another."""
    from web_attack_detector.uniembed.features import RULE_FEATURES_TOTAL

    assert NUM_RULE_FEATURES == 45
    assert RULE_FEATURES_TOTAL == 157
