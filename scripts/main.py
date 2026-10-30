"""Command-line entry point for the AURA magic-mirror application."""

from __future__ import annotations

# Support direct execution from a source checkout before editable installation.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import logging

from aura.engine import AuraEngine


def parse_arguments() -> argparse.Namespace:
    """Parse source and exercise selection without changing AuraEngine's API."""
    parser = argparse.ArgumentParser(description="AURA real-time exercise coach")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--exercise", default="squats", help="Exercise profile name")
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    source = int(args.source) if str(args.source).isdigit() else args.source
    engine = AuraEngine(source=source, exercise=args.exercise)
    engine.run()


if __name__ == "__main__":
    main()
