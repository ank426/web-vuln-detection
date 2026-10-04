"""``web-attack-detector-build-dataset``: reconcile the raw corpora into one 4-class CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from web_attack_detector.config import processed_dir, raw_dir
from web_attack_detector.data.builder import build_unified_dataset, write_dataset

DEFAULT_OUTPUT = processed_dir() / "web_attacks_4class.csv"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="web-attack-detector-build-dataset",
        description="Build the unified 4-class (benign/xss/csrf/sqli) training CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=raw_dir(),
        help=f"Raw data root (default: {raw_dir()})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap each class at this many rows to reduce imbalance.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate payloads.",
    )
    parser.add_argument(
        "--no-generic",
        action="store_true",
        help="Exclude the large payloads.csv corpus.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Downsampling seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    frame, report = build_unified_dataset(
        args.raw_dir,
        max_per_class=args.max_per_class,
        dedupe=not args.no_dedupe,
        seed=args.seed,
        include_generic=not args.no_generic,
    )
    write_dataset(frame, args.output)

    print(report.render())
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
