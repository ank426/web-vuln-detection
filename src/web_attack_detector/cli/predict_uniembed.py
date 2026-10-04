"""``web-attack-detector-predict-uniembed``: classify with the GloVe+fastText+USE MLP.

Requires the ``uniembed`` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from web_attack_detector.config import CLASS_NAMES, models_dir

DEFAULT_CHECKPOINT = models_dir() / "uniembed" / "mlp_xss_csrf_sqli_detector.pth"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="web-attack-detector-predict-uniembed",
        description="Classify payloads with the uniembed+MLP detector.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"MLP checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--payload", type=str, default=None, help="Classify and exit.")
    parser.add_argument("--file", type=Path, default=None, help="One payload per line.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda.")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    from web_attack_detector.uniembed.model import UniembedDetector

    detector = UniembedDetector(str(args.checkpoint), device=args.device)

    if args.payload is not None:
        payloads = [args.payload]
    elif args.file is not None:
        if not args.file.exists():
            print(f"File not found: {args.file}", file=sys.stderr)
            return 1
        payloads = [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payloads = []
        print("Interactive mode. Ctrl-D or a blank line to exit.")
        while True:
            try:
                entry = input("payload> ").strip()
            except EOFError:
                break
            if not entry:
                break
            payloads.append(entry)

    for payload in payloads:
        label, probabilities, confidence = detector.predict(payload)
        record = {
            "payload": payload,
            "prediction": CLASS_NAMES[label],
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities, strict=True)
            },
        }
        if args.json:
            print(json.dumps(record))
        else:
            print(f"\nPayload:    {payload}")
            print(f"Prediction: {record['prediction']} ({confidence:.2%})")
            for name, prob in record["probabilities"].items():
                print(f"  {name:<8} {prob:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
