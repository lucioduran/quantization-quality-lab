"""Offline replay command."""

import argparse
import json
from pathlib import Path

from .core import verify


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify saved quantization evidence without inference"
    )
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        summary = verify(args.root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(
        json.dumps(
            {"status": "consistent", "authenticity_verified": False, "summary": summary}, indent=2
        )
    )


if __name__ == "__main__":
    main()
