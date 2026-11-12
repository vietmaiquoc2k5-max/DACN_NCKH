"""Phase 3: Strict full video decode audit for all 79 negative surveillance videos.
Outputs to runs/benchmarks/week4_final_recovery/
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_audit import (
    CACHE_SCHEMA_VERSION,
    AUDIT_POLICY_VERSION,
    VideoDecodeCacheManager,
    audit_single_video,
)

out_dir = ROOT / "runs/benchmarks/week4_final_recovery"
out_dir.mkdir(parents=True, exist_ok=True)

manifest_path = ROOT / "data/negative_videos_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)

manifest_entries = manifest_data.get("videos", [])
print(f"Loaded {len(manifest_entries)} videos from manifest.")

cache_path = ROOT / "data/.video_decode_cache.json"
cache_mgr = VideoDecodeCacheManager(cache_file=cache_path, root_dir=ROOT)

log_path = out_dir / "video_audit_execution.log"
with open(log_path, "w", encoding="utf-8") as log_f:
    log_f.write(
        f"=== Phase 3 Video Audit Execution Log ===\n"
        f"Started: {datetime.now(timezone.utc).isoformat()}\n"
        f"Policy: {AUDIT_POLICY_VERSION} (schema {CACHE_SCHEMA_VERSION})\n"
        f"Total videos to audit: {len(manifest_entries)}\n\n"
    )

results = []
t0 = time.time()

for idx, entry in enumerate(manifest_entries, 1):
    f_t0 = time.time()
    rel_path = entry.get("relative_path") or entry.get("video_path")
    full_path = ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
    
    print(f"[{idx:02d}/{len(manifest_entries):02d}] Auditing {rel_path}...", end=" ", flush=True)
    
    if not full_path.is_file():
        # Missing file error record
        rec = {
            "video_id": entry.get("video_id"),
            "relative_path": rel_path,
            "filename": Path(rel_path).name,
            "sha256": "",
            "sha256_verified": False,
            "file_size_bytes": 0,
            "status": "BLOCKED_INPUT_MISSING",
            "decode_status": "BLOCKED_INPUT_MISSING",
            "full_decode_verified": False,
            "error_message": "File not found on disk",
            "reported_frame_count": 0,
            "full_decoded_frames": 0,
            "duration_seconds": 0.0,
            "duration_hours": 0.0,
            "audit_policy_version": AUDIT_POLICY_VERSION,
            "schema_version": CACHE_SCHEMA_VERSION,
            "last_audited_at": datetime.now(timezone.utc).isoformat()
        }
    else:
        rec = audit_single_video(
            full_path,
            root_dir=ROOT,
            full_decode=True,
            cache_manager=cache_mgr
        )
    
    f_elapsed = time.time() - f_t0
    results.append(rec)
    
    ext = rec.get("external_tool_check") or {}
    ff_read = ext.get("nb_read_frames")
    
    # Classify strict status for logging
    status_summary = (
        f"status={rec.get('status')}, decode={rec.get('decode_status')}, "
        f"opencv={rec.get('full_decoded_frames')}, reported={rec.get('reported_frame_count')}, "
        f"ffmpeg={ff_read}, verified={rec.get('full_decode_verified')}"
    )
    print(f"({f_elapsed:.1f}s) -> {status_summary}")
    
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"[{idx:02d}/{len(manifest_entries):02d}] {rel_path} ({f_elapsed:.2f}s): {status_summary}\n")
        if rec.get("error_message"):
            log_f.write(f"   Note/Error: {rec.get('error_message')}\n")

total_elapsed = time.time() - t0
print(f"\nAudit completed in {total_elapsed:.1f}s.")

# 1. Output video_audit_final.json
json_path = out_dir / "video_audit_final.json"
summary_counts = {}
for r in results:
    st = r.get("decode_status", "UNKNOWN")
    summary_counts[st] = summary_counts.get(st, 0) + 1

full_json_data = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "audit_policy_version": AUDIT_POLICY_VERSION,
    "cache_schema_version": CACHE_SCHEMA_VERSION,
    "total_videos_audited": len(results),
    "decode_status_counts": summary_counts,
    "strict_valid_count": sum(1 for r in results if r.get("decode_status") == "NORMAL_EOF" and r.get("full_decode_verified")),
    "total_decoded_frames": sum(r.get("full_decoded_frames", 0) for r in results),
    "total_evaluated_duration_hours": round(sum(r.get("duration_hours", 0.0) for r in results), 4),
    "videos": results
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(full_json_data, f, indent=2, ensure_ascii=False)
print(f"Saved: {json_path}")

# 2. Output video_audit_summary_final.csv
csv_path = out_dir / "video_audit_summary_final.csv"
fieldnames = [
    "video_id", "relative_path", "file_size_bytes", "sha256", "fps",
    "reported_frame_count", "full_decoded_frames", "ffmpeg_read_frames",
    "decode_status", "status", "source_label_status", "human_review_status",
    "full_decode_verified", "duration_seconds", "duration_hours", "error_message"
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        ext = r.get("external_tool_check") or {}
        writer.writerow({
            "video_id": r.get("video_id"),
            "relative_path": r.get("relative_path"),
            "file_size_bytes": r.get("file_size_bytes"),
            "sha256": r.get("sha256"),
            "fps": r.get("fps"),
            "reported_frame_count": r.get("reported_frame_count"),
            "full_decoded_frames": r.get("full_decoded_frames"),
            "ffmpeg_read_frames": ext.get("nb_read_frames"),
            "decode_status": r.get("decode_status"),
            "status": r.get("status"),
            "source_label_status": r.get("source_label_status"),
            "human_review_status": r.get("human_review_status"),
            "full_decode_verified": r.get("full_decode_verified"),
            "duration_seconds": r.get("duration_seconds"),
            "duration_hours": r.get("duration_hours"),
            "error_message": r.get("error_message")
        })
print(f"Saved: {csv_path}")

# 3. Output video_decode_failures_final.csv (non-NORMAL_EOF or not verified)
failures = [r for r in results if r.get("decode_status") != "NORMAL_EOF" or not r.get("full_decode_verified")]
fail_path = out_dir / "video_decode_failures_final.csv"
with open(fail_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in failures:
        ext = r.get("external_tool_check") or {}
        writer.writerow({
            "video_id": r.get("video_id"),
            "relative_path": r.get("relative_path"),
            "file_size_bytes": r.get("file_size_bytes"),
            "sha256": r.get("sha256"),
            "fps": r.get("fps"),
            "reported_frame_count": r.get("reported_frame_count"),
            "full_decoded_frames": r.get("full_decoded_frames"),
            "ffmpeg_read_frames": ext.get("nb_read_frames"),
            "decode_status": r.get("decode_status"),
            "status": r.get("status"),
            "source_label_status": r.get("source_label_status"),
            "human_review_status": r.get("human_review_status"),
            "full_decode_verified": r.get("full_decode_verified"),
            "duration_seconds": r.get("duration_seconds"),
            "duration_hours": r.get("duration_hours"),
            "error_message": r.get("error_message")
        })
print(f"Saved: {fail_path} ({len(failures)} failures/needs_review)")

