"""Retired v1.1 migration entry point.

The original migration synthesized independent verification from a video allowlist.
The frozen dataset and checksum remain historical evidence; this entry point must
never overwrite them or manufacture a second annotation.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Refuse frozen writes and require an approved, evidence-backed version transition."""
    frozen = PROJECT_ROOT / "data/ground_truth/events_v1.1_snapshot.jsonl"
    if frozen.exists():
        raise SystemExit("v1.1 is frozen. Create an explicitly approved new version; do not regenerate it.")
    raise SystemExit("Legacy migration disabled: video membership is not independent annotation evidence.")


if __name__ == "__main__":
    main()
