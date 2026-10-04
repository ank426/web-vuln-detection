"""``web-attack-detector-predict``: classify payloads from the CLI, a file, or stdin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from web_attack_detector.config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    NUM_CLASSES,
    default_model_path,
)
from web_attack_detector.trainer import WebAttackTrainer

_BANNER = "💻 Enter a payload (blank line to quit): "


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="web-attack-detector-predict",
        description="Classify web payloads as benign / xss / csrf / sqli.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path(),
        help=f"Checkpoint path (default: {default_model_path()}).",
    )
    parser.add_argument("--model-name", default=None, help="Override the HF encoder id.")
    parser.add_argument(
        "--payload", type=str, default=None, help="Classify this single payload and exit."
    )
    parser.add_argument("--file", type=Path, default=None, help="Classify one payload per line.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write JSON results to this path."
    )
    parser.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenisation length."
    )
    return parser


def _load_trainer(args: argparse.Namespace) -> WebAttackTrainer:
    import torch

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model_name = args.model_name or checkpoint.get("model_name") or DEFAULT_MODEL_NAME
    trainer = WebAttackTrainer(
        model_name=model_name,
        num_classes=int(
            (checkpoint.get("class_names") and len(checkpoint["class_names"])) or NUM_CLASSES
        ),
        device=args.device,
        max_length=int(checkpoint.get("max_length") or args.max_length),
    )
    trainer.load_model(args.model_path, load_optimizer=False)
    return trainer


def _emit(trainer: WebAttackTrainer, payload: str, *, as_json: bool) -> dict[str, object]:
    prediction = trainer.predict(payload)
    record: dict[str, object] = {
        "payload": payload,
        "prediction": prediction.class_name,
        "label": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(trainer.class_names, prediction.probabilities, strict=True)
        },
    }
    if as_json:
        print(json.dumps(record))
    else:
        print(f"\nPayload:   {payload}")
        print(f"Prediction: {prediction.class_name} (confidence {prediction.confidence:.4f})")
        print("Probabilities:")
        for name, prob in zip(trainer.class_names, prediction.probabilities, strict=True):
            print(f"  {name:<8} {prob:.4f}")
    return record


def _read_payloads(args: argparse.Namespace) -> list[str]:
    if args.payload is not None:
        return [args.payload]
    if args.file is not None:
        if not args.file.exists():
            msg = f"File not found: {args.file}"
            raise FileNotFoundError(msg)
        return [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    trainer = _load_trainer(args)

    payloads = _read_payloads(args)
    records: list[dict[str, object]] = []

    if payloads:
        records = [_emit(trainer, p, as_json=args.json) for p in payloads]
    else:
        print("Interactive mode. Ctrl-D or a blank line to exit.")
        while True:
            try:
                payload = input(_BANNER).strip()
            except EOFError:
                break
            if not payload:
                break
            records.append(_emit(trainer, payload, as_json=args.json))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
