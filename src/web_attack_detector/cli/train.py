"""``web-attack-detector-train``: train the hybrid detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from web_attack_detector.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEST_SIZE,
    NUM_CLASSES,
    default_model_path,
    processed_dir,
)
from web_attack_detector.data.builder import build_smoke_dataset
from web_attack_detector.sample_data import create_sample_data
from web_attack_detector.trainer import WebAttackTrainer


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="web-attack-detector-train",
        description="Train the DistilBERT + rule-feature web attack detector.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Training CSV. Omit to use the 24-row synthetic smoke dataset.",
    )
    parser.add_argument("--payload-col", default="payload", help="Text column name.")
    parser.add_argument("--label-col", default="label", help="Integer label column name.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HF encoder id.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epoch count.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="AdamW lr."
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Holdout frac.")
    parser.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenisation length."
    )
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Class count.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=default_model_path(),
        help=f"Checkpoint path (default: {default_model_path()}).",
    )
    parser.add_argument(
        "--default-data",
        type=Path,
        default=processed_dir() / "web_attacks_4class.csv",
        help="Dataset used when --data is omitted but the file exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    frame = None
    if args.data is not None:
        import pandas as pd

        frame = pd.read_csv(args.data)
        print(f"Loaded {len(frame):,} rows from {args.data}")
    elif args.default_data.exists():
        import pandas as pd

        frame = pd.read_csv(args.default_data)
        print(f"Loaded {len(frame):,} rows from {args.default_data}")
    else:
        frame = create_sample_data()
        print(
            f"No dataset found at {args.default_data}; falling back to the "
            f"{len(frame)}-row synthetic smoke dataset.",
        )

    trainer = WebAttackTrainer(
        model_name=args.model_name,
        num_classes=args.num_classes,
        device=args.device,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # A stratified split needs at least two rows per class; the smoke set is
    # replicated so that train/eval splits stay possible.
    if len(frame) < args.num_classes * 4:
        frame = build_smoke_dataset()

    trainer.prepare_data(
        frame,
        payload_col=args.payload_col,
        label_col=args.label_col,
        test_size=args.test_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    trainer.train(
        epochs=args.epochs,
        save_path=args.save_path,
        config={
            "model_name": args.model_name,
            "num_classes": args.num_classes,
            "class_names": list(trainer.class_names),
            "training_args": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "test_size": args.test_size,
                "max_length": args.max_length,
            },
            "data_source": str(args.data or args.default_data),
        },
    )
    print(f"\n💡 Predict with: web-attack-detector-predict --model-path {args.save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
