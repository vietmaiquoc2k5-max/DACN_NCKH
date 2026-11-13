"""Measured local source catalog, retaining history and separating semantic candidates."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import catalog, frozen, sha256


def save_json(path: Path, value: Any) -> None:
    """Atomically update a current-run artifact; callers must choose a new run directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def probe_metadata(path: Path, ffprobe: str | None) -> dict[str, Any]:
    """Read stream metadata only, never call metadata hours verified exposure."""
    if not ffprobe:
        return {"probe_status": "BLOCKED_FFPROBE_MISSING"}
    try:
        result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,time_base:format=duration",
                                 "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            return {"probe_status": "PROBE_ERROR", "probe_exit_code": result.returncode, "stderr": result.stderr}
        raw = json.loads(result.stdout)
        stream = raw.get("streams", [{}])[0]
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        fps = float(Fraction(rate)) if rate and rate != "0/0" else None
        duration = raw.get("format", {}).get("duration") or stream.get("duration")
        frames = stream.get("nb_frames")
        return {"probe_status": "METADATA_ONLY", "probe_exit_code": 0, "codec": stream.get("codec_name"),
                "fps": fps, "duration_seconds": float(duration) if duration not in (None, "N/A") else None,
                "frame_count": int(frames) if frames not in (None, "N/A") else None,
                "stream_metadata": stream, "raw_probe": raw}
    except (ValueError, ZeroDivisionError, subprocess.TimeoutExpired) as error:
        return {"probe_status": "PROBE_ERROR", "reason": str(error)}


def recover_catalog(settings: Settings, output: Path, *, publish: bool = False) -> dict[str, Any]:
    """Hash/probe physical sources, retaining missing records and recording duplicates."""
    source_rows = catalog(settings)
    records = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for index, source in enumerate(source_rows, 1):
        path = confined(settings.repo, source["relative_path"])
        record = dict(source, source_available=path.is_file(), sha256=None, file_size_bytes=None,
                      duration_seconds=None, fps=None, frame_count=None, technical_status="NOT_VERIFIED", split=None)
        if path.is_file():
            record["sha256"] = sha256(path)
            record["file_size_bytes"] = path.stat().st_size
            record.update(probe_metadata(path, settings.ffprobe))
            expected = source.get("expected_sha256")
            record["historical_sha256_matches"] = record["sha256"] == expected if expected else None
            if expected and expected != record["sha256"]:
                record["exclusion_reason"] = "SOURCE_SHA256_MISMATCH"
            hashes[record["sha256"]].append(record["video_id"])
        else:
            record["exclusion_reason"] = "EXCLUDED_MISSING_SOURCE"
        records.append(record)
        print(f"CATALOG {index}/{len(source_rows)} {record['video_id']} {record.get('probe_status', 'MISSING')}", flush=True)
    negatives = [row for row in records if row["source_group"] == "VN_NEG" and row["source_available"]]
    physical = [row for row in records if row["source_group"] in {"VN_NEG", "VN_POS"} and row["source_available"]]
    missing = [row for row in records if row["manifest_origin"] == "historical" and not row["source_available"]]
    identity = [{"video_id": row["video_id"], "sha256": row["sha256"], "relative_path": row["relative_path"]} for row in negatives]
    scope_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    payload = {"schema_version": 2, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope_id": "vn_neg_local_v2_" + scope_hash, "scope_identity": identity,
               "technical_audit_path": (output / "technical_audit_63.json").relative_to(settings.repo).as_posix(),
               "videos": records, "semantic_labels_verified": False,
               "notes": "Filename/manifest groups are candidates only. Metadata duration is not FAR exposure."}
    duplicates = [{"sha256": digest, "video_ids": ids} for digest, ids in hashes.items() if len(ids) > 1]
    save_json(output / "video_catalog_local_v2.json", payload)
    save_json(output / "current_physical_catalog.json", {"video_count": len(physical), "videos": physical})
    save_json(output / "missing_historical_sources.json", {"status": "EXCLUDED_MISSING_SOURCE", "count": len(missing), "videos": missing})
    manifest = json.loads((settings.repo / "data/negative_videos_manifest.json").read_text(encoding="utf-8"))
    original = {row["video_id"]: row for row in manifest["videos"]}
    deltas = [{"video_id": row["video_id"], "historical_duration_seconds": original[row["video_id"]].get("duration_seconds"),
               "ffprobe_duration_seconds": row["duration_seconds"],
               "historical_reported_frame_count": original[row["video_id"]].get("reported_frame_count"), "ffprobe_native_frame_count": row["frame_count"]}
              for row in negatives if row["video_id"] in original]
    report = {"historical_negative_records": len(manifest["videos"]), "physical_prefixed_videos": len(physical),
              "source_group_counts_available": dict(Counter(row["source_group"] for row in records if row["source_available"])),
              "missing_historical_count": len(missing), "new_video_ids": [row["video_id"] for row in records if row["manifest_origin"] == "discovered"],
              "sha_mismatches": [row["video_id"] for row in records if row.get("historical_sha256_matches") is False],
              "duplicate_hash_groups": duplicates, "metadata_differences": deltas,
              "metadata_hours_by_group": {group: sum(row["duration_seconds"] or 0 for row in physical if row["source_group"] == group) / 3600 for group in ("VN_NEG", "VN_POS")},
              "probe_errors": [row["video_id"] for row in physical if row.get("probe_status") != "METADATA_ONLY"],
              "semantic_review_pending": [row["video_id"] for row in records],
              "scope_id": payload["scope_id"], "far_per_hour": None, "split_assignment_performed": False}
    save_json(output / "catalog_reconciliation_report.json", report)
    frozen_hash, events = frozen(settings)
    save_json(output / "historical_claims_reauthentication.json", {
        "frozen_sha256": frozen_hash, "events": [{"historical_index": i, "video_id": event["video_id"],
        "claim": event["verification_status"], "status": "UNVERIFIED", "verified_evidence": ["FROZEN_BYTES_AUTHENTICATED"],
        "missing_evidence": ["ORIGINAL_INDEPENDENT_REVIEW_RECORDS", "HUMAN_IDENTITY_AND_SIGNOFF_PROVENANCE"]}
        for i, event in enumerate(events, 1) if event.get("verification_status") == "cross_verified"]})
    if publish:
        target = settings.repo / "data/video_catalog_local_v2.json"
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
    return payload
