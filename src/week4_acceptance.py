"""Evidence gates for Week 4; missing evidence never implies acceptance."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

REQUIRED_REUSE_CHECKS = frozenset({
    "source_hashes", "video_id_mapping", "checkpoint_sha256", "model_preprocessing",
    "confidence_threshold", "sampling_fps", "sampling_strategy", "configuration",
    "source_commit_available", "decoded_frame_counts", "sampled_frame_count_bounds",
    "sampled_count_matches_historical_stride", "sampled_frame_counts_verified", "run_completion",
    "frame_timestamps", "error_records", "skipped_frames_accounted_for", "historical_manifest_authenticated",
})


def strict_technical_eligible(row: dict[str, Any]) -> bool:
    """Require affirmative evidence for every technical eligibility condition."""
    return (
        row.get("decode_status") == "NORMAL_EOF"
        and row.get("sha256_verified") is True
        and row.get("full_decode_verified") is True
        and row.get("full_decode_errors") == 0
        and row.get("inference_alignment_verified") is True
        and math.isfinite(float(row.get("duration_seconds", 0)))
        and float(row.get("duration_seconds", 0)) > 0
    )


def acceptance(negative_hours: float, reduction: float | None = None,
               recall: bool | None = None, delay: bool | None = None,
               fair_comparison: bool = False, provenance: bool = False,
               validation_locked: bool = False,
               positive_gt_complete: bool | None = None) -> dict[str, Any]:
    """Keep diagnostic reduction separate from independently verified FAR."""
    exposure = math.isfinite(negative_hours) and negative_hours >= 5.0
    far = reduction >= 50 if reduction is not None and fair_comparison and exposure else None
    official = all(v is True for v in (exposure, far, recall, delay, provenance, validation_locked, positive_gt_complete))
    return {
        "alert_rate_reduction_target_met": reduction >= 50 if reduction is not None else None,
        "far_reduction_verified": far,
        "event_recall_constraint_met": recall,
        "detection_delay_constraint_met": delay,
        "negative_exposure_requirement_met": exposure,
        "independent_positive_gt_complete": positive_gt_complete,
        "operating_point_official_eligible": official,
        "week4_acceptance_status": "FULL_PASS" if official else "PARTIAL",
    }


def reuse_decision(checks: dict[str, bool | None], missing_input: bool = False) -> dict[str, Any]:
    """Prioritize contradictions, then missing inputs, then unresolved checks."""
    checks = {**{key: None for key in sorted(REQUIRED_REUSE_CHECKS)}, **checks}
    if any(value is False for value in checks.values()):
        state = "RERUN_REQUIRED"
    elif missing_input:
        state = "BLOCKED_MISSING_INPUT"
    elif all(value is True for value in checks.values()) and checks:
        state = "REUSE_VERIFIED"
    elif checks.get("frame_timestamps") is None and all(
        value is True for key, value in checks.items() if key != "frame_timestamps"
    ):
        state = "REUSE_WITH_LIMITATIONS"
    else:
        state = "UNVERIFIED"
    return {"recommendation": state, "checks": checks,
            "rerun_required": True if state == "RERUN_REQUIRED" else False if state.startswith("REUSE_") else None,
            "full_timeline_independently_verified": checks.get("frame_timestamps") is True}


def preserve_review_csv(path: Path, rows: list[dict[str, Any]], key: str,
                        source: Path | None = None) -> None:
    """Merge new identities without replacing any existing human review fields.

    Raises:
        ValueError: A header, identity, source interval, or CSV record is malformed.
    """
    prior = path if path.exists() else source
    old: list[dict[str, Any]] = []
    fields = list(rows[0]) if rows else []
    if prior and prior.exists():
        with prior.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or key not in reader.fieldnames or not set(fields).issubset(reader.fieldnames)):
                raise ValueError(f"Malformed review header: {prior}")
            fields = list(dict.fromkeys(reader.fieldnames + fields))
            old = list(reader)
        if any(None in row or any(v is None for v in row.values()) or not row.get(key) for row in old):
            raise ValueError(f"Malformed review record: {prior}")
        if len({row[key] for row in old}) != len(old):
            raise ValueError(f"Duplicate review identity: {prior}")
    existing = {row[key]: row for row in old}
    if len({row[key] for row in rows}) != len(rows):
        raise ValueError("Duplicate generated review identity")
    immutable = ("video_id", "relative_path", "onset_sec", "offset_sec", "class")
    for row in rows:
        previous = existing.get(row[key])
        if previous:
            for field in immutable:
                if field in previous and field in row and str(previous[field]) != str(row[field]):
                    raise ValueError(f"Source changed for {row[key]}: {field}; review version transition required")
        else:
            old.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(old)
    content = buffer.getvalue().encode("utf-8")
    if path.exists() and content == path.read_bytes():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import hashlib
        backup = path.with_name(path.name + "." + hashlib.sha256(path.read_bytes()).hexdigest() + ".bak")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
