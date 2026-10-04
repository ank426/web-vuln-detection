"""Tests for configuration constants and CLI argument parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from web_attack_detector.cli.build_dataset import build_parser as build_dataset_parser
from web_attack_detector.cli.predict import build_parser as predict_parser
from web_attack_detector.cli.train import build_parser as train_parser
from web_attack_detector.cli.web import build_parser as web_parser
from web_attack_detector.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    AttackClass,
    label_from_name,
    models_dir,
    raw_dir,
    repo_root,
)


def test_class_names_align_with_enum() -> None:
    """CLASS_NAMES[i] is the lowercased name of AttackClass(i)."""
    assert len(CLASS_NAMES) == NUM_CLASSES
    for index, member in enumerate(AttackClass):
        assert CLASS_NAMES[index] == member.name.lower()


def test_attack_class_values_are_canonical() -> None:
    """The integer encoding is frozen at 0/1/2/3."""
    assert (AttackClass.BENIGN, AttackClass.XSS, AttackClass.CSRF, AttackClass.SQLI) == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("benign", 0), ("XSS", 1), (" csrf ", 2), ("SqLi", 3)],
)
def test_label_from_name(name: str, expected: int) -> None:
    """Names map to integers case-insensitively and tolerate whitespace."""
    assert label_from_name(name) == expected


def test_label_from_name_rejects_unknown() -> None:
    """An unknown class name raises."""
    with pytest.raises(KeyError):
        label_from_name("ssrf")


def test_paths_are_repo_relative_and_absolute() -> None:
    """Data/model paths resolve under the repo root as absolute paths."""
    root = repo_root()
    assert root.is_absolute()
    assert (root / "pyproject.toml").exists()
    assert raw_dir() == root / "data" / "raw"
    assert models_dir() == root / "models"


def test_train_parser_defaults() -> None:
    """Training defaults match the documented configuration."""
    args = train_parser().parse_args([])
    assert args.data is None
    assert args.epochs == 3
    assert args.batch_size == 8
    assert args.num_classes == 4
    assert args.save_path == models_dir() / "web_attack_model.pt"


def test_train_parser_accepts_overrides() -> None:
    """Flags override defaults."""
    args = train_parser().parse_args(
        ["--data", "x.csv", "--epochs", "9", "--label-col", "y", "--device", "cpu"],
    )
    assert args.data == Path("x.csv")
    assert args.epochs == 9
    assert args.label_col == "y"
    assert args.device == "cpu"


def test_predict_parser_flags() -> None:
    """Single-payload and JSON modes are selectable."""
    args = predict_parser().parse_args(["--payload", "<script>x</script>", "--json"])
    assert args.payload == "<script>x</script>"
    assert args.json is True


def test_build_dataset_parser_flags() -> None:
    """Dedupe is on by default and can be disabled."""
    assert build_dataset_parser().parse_args([]).no_dedupe is False
    args = build_dataset_parser().parse_args(["--no-dedupe", "--max-per-class", "500"])
    assert args.no_dedupe is True
    assert args.max_per_class == 500


def test_web_parser_defaults() -> None:
    """The web UI defaults to localhost:8501 and an existing script."""
    args = web_parser().parse_args([])
    assert args.port == 8501
    assert args.address == "localhost"
    assert args.app.exists()
