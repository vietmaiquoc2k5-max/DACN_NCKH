"""Script chạy Phase 2 Video Audit đầy đủ cho 79 tệp video âm tính.
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
    VideoDecodeCacheManager,
    audit_single_video,
)

out_dir = ROOT / "runs/benchmarks/week4_revalidation"
out_dir.mkdir(parents=True, exist_ok=True)

cache_path = ROOT / "data/.video_decode_cache.json"
cache_mgr = VideoDecodeCacheManager(cache_file=cache_path, root_dir=ROOT)

dirs = [ROOT / "datasets/negative_videos", ROOT / "datasets/test_videos/neg"]
exts = {".mp4", ".avi", ".mkv", ".mov"}
all_files = []
for d in dirs:
    if d.is_dir():
        for f in sorted(d.glob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                all_files.append(f)

print(f"Starting Phase 2 audit on {len(all_files)} files...")
results = []
t0 = time.time()

log_path = out_dir / "audit_execution_log.txt"
with open(log_path, "w", encoding="utf-8") as log_f:
    log_f.write(
        f"=== Video Audit Execution Log ===\nStarted: {datetime.now(timezone.utc).isoformat()}\nTotal files: {len(all_files)}\n\n"
    )

for idx, f in enumerate(all_files, 1):
    f_t0 = time.time()
    rel = str(f.relative_to(ROOT)).replace("\\", "/")
    print(f"[{idx:02d}/{len(all_files):02d}] Auditing {rel}...", end=" ", flush=True)
    res = audit_single_video(f, root_dir=ROOT, full_decode=True, cache_manager=cache_mgr)
    f_elapsed = time.time() - f_t0
    results.append(res)

    ext = res.get("external_tool_check") or {}
    ff_read = ext.get("nb_read_frames")
    status_str = (
        f"status={res.get('status')}, decode={res.get('decode_status')}, "
        f"opencv={res.get('full_decoded_frames')}, reported={res.get('reported_frame_count')}, "
        f"ffmpeg={ff_read}, verified={res.get('full_decode_verified')}"
    )
    print(f"({f_elapsed:.1f}s) -> {status_str}")

    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"[{idx:02d}/{len(all_files):02d}] {rel} ({f_elapsed:.2f}s): {status_str}\n")
        if res.get("error_message"):
            log_f.write(f"   Error/Note: {res.get('error_message')}\n")

total_elapsed = time.time() - t0
print(f"Completed Phase 2 audit in {total_elapsed:.1f}s.")

# 1. Save full JSON
json_path = out_dir / "video_audit_full.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "total_videos": len(results),
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "videos": results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

# 2. Save summary CSV
csv_path = out_dir / "video_audit_summary.csv"
fieldnames = [
    "video_id",
    "relative_path",
    "file_size_bytes",
    "sha256",
    "fps",
    "reported_frame_count",
    "full_decoded_frames",
    "ffmpeg_read_frames",
    "decode_status",
    "status",
    "source_label_status",
    "human_review_status",
    "full_decode_verified",
    "duration_seconds",
    "duration_hours",
    "error_message",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        ext = r.get("external_tool_check") or {}
        writer.writerow(
            {
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
                "error_message": r.get("error_message"),
            }
        )

# 3. Save failures CSV (non-NORMAL_EOF or not full_decode_verified)
failures = [
    r
    for r in results
    if r.get("decode_status") != "NORMAL_EOF" or not r.get("full_decode_verified")
]
fail_path = out_dir / "video_decode_failures.csv"
with open(fail_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in failures:
        ext = r.get("external_tool_check") or {}
        writer.writerow(
            {
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
                "error_message": r.get("error_message"),
            }
        )

print(f"Failures/Needs Review count: {len(failures)}")
for fl in failures:
    print(f"  -> {fl.get('video_id')}: {fl.get('decode_status')} | {fl.get('error_message')}")

