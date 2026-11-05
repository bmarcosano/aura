"""
AURA - Main Entry Point (v4.0.0)
"""

import argparse
import sys

from aura.engine import AuraEngine


def parse_arguments():
    parser = argparse.ArgumentParser(description="AURA - Autonomous Pose Tracking")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source path or 0 for local webcam",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default="squats",
        help="Target exercise subfolder name (e.g., squats, push_up)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    app = None

    try:
        # Initialize and run the AURA engine with the specified source and exercise.
        app = AuraEngine(source=args.source, exercise=args.exercise)
        app.run()
    except Exception as e:  # noqa: BLE001
        print(f"❌ Critical System Failure: {e}")
        sys.exit(1)
    finally:
        # run() also shuts down on normal exit; shutdown() is idempotent.
        if app is not None:
            app.shutdown()
        print("AURA shutdown complete.")


if __name__ == "__main__":
    main()
