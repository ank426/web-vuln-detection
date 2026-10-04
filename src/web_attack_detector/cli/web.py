"""``web-attack-detector-web``: launch the Streamlit UI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

APPS_DIR = Path(__file__).resolve().parents[1] / "apps"
APP = APPS_DIR / "distilbert_app.py"
UNIEMBED_APP = APPS_DIR / "uniembed_app.py"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="web-attack-detector-web",
        description="Launch the Streamlit web UI.",
    )
    parser.add_argument("--port", type=int, default=8501, help="Port to bind.")
    parser.add_argument("--address", default="localhost", help="Bind address.")
    parser.add_argument(
        "--app",
        type=Path,
        default=APP,
        help=f"Streamlit script to run (default: {APP}).",
    )
    parser.add_argument(
        "--uniembed",
        action="store_true",
        help=f"Launch the optional uniembed app instead (default: {UNIEMBED_APP}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.uniembed:
        args.app = UNIEMBED_APP
    if not args.app.exists():
        print(f"App not found: {args.app}", file=sys.stderr)
        return 1
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(args.app),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
