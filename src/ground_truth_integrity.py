"""Byte-exact validation of immutable annotation snapshots."""
from __future__ import annotations

import hashlib
from pathlib import Path


def validate_frozen_snapshot(snapshot: Path, checksum_manifest: Path) -> str:
    """Validate the named snapshot, without newline normalization or hash fallback.

    Raises:
        ValueError: Missing, ambiguous, or mismatched frozen checksum.
    """
    matches = []
    for line in checksum_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == snapshot.name:
            matches.append(parts[0])
    actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if len(matches) != 1 or actual != matches[0]:
        raise ValueError(f"BLOCKED_GT_INTEGRITY: {snapshot.name}: actual SHA256 {actual}")
    return actual
