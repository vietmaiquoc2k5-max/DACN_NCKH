"""Phase 1: Video inventory and path reconciliation.
Verifies all 79 negative surveillance videos on disk against manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_audit import compute_streaming_sha256

out_dir = ROOT / "runs/benchmarks/week4_final_recovery"
out_dir.mkdir(parents=True, exist_ok=True)

manifest_path = ROOT / "data/negative_videos_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)

manifest_entries = manifest_data.get("videos", [])
print(f"Total manifest entries: {len(manifest_entries)}")

# Scan target directories
scan_dirs = [
    ROOT / "datasets/negative_videos",
    ROOT / "datasets/test_videos/neg",
]

disk_files = []
exts = {".mp4", ".avi", ".mkv", ".mov"}
for d in scan_dirs:
    if d.is_dir():
        for f in sorted(d.glob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                disk_files.append(f)

print(f"Total physical video files found: {len(disk_files)}")

# Inventory audit
inventory_records = []
missing_records = []
path_reconciliations = []

manifest_by_id = {v["video_id"]: v for v in manifest_entries}
seen_hashes = {}
duplicate_hashes = []
seen_ids = set()
duplicate_ids = []

for idx, entry in enumerate(manifest_entries, 1):
    vid = entry["video_id"]
    decl_path = entry.get("relative_path") or entry.get("video_path") or ""
    decl_sha256 = entry.get("sha256", "")
    
    # Check ID collision
    if vid in seen_ids:
        duplicate_ids.append(vid)
    seen_ids.add(vid)

    full_path = ROOT / decl_path if not Path(decl_path).is_absolute() else Path(decl_path)
    
    file_exists = full_path.is_file()
    actual_size = full_path.stat().st_size if file_exists else 0
    actual_sha256 = ""
    sha_match = False
    
    if file_exists:
        actual_sha256 = compute_streaming_sha256(full_path)
        sha_match = (actual_sha256.lower() == decl_sha256.lower()) if decl_sha256 else True
        if actual_sha256 in seen_hashes:
            duplicate_hashes.append({
                "video_id": vid,
                "path": decl_path,
                "duplicate_of": seen_hashes[actual_sha256]
            })
        else:
            seen_hashes[actual_sha256] = decl_path
    else:
        missing_records.append({
            "video_id": vid,
            "declared_path": decl_path,
            "expected_sha256": decl_sha256,
            "reason": "FILE_NOT_FOUND_ON_DISK"
        })

    record = {
        "video_id": vid,
        "declared_relative_path": decl_path.replace("\\", "/"),
        "physical_path": str(full_path.relative_to(ROOT)).replace("\\", "/") if file_exists else None,
        "file_exists": file_exists,
        "file_size_bytes": actual_size,
        "declared_sha256": decl_sha256,
        "actual_sha256": actual_sha256,
        "sha256_match": sha_match,
        "fps": entry.get("fps"),
        "reported_frame_count": entry.get("reported_frame_count"),
        "codec": entry.get("codec"),
        "source_label_status": entry.get("source_label_status", "SURVEILLANCE_NEGATIVE_BATCH"),
        "human_review_status": entry.get("human_review_status", "PENDING_HUMAN_REVIEW"),
    }
    inventory_records.append(record)
    
    path_reconciliations.append({
        "video_id": vid,
        "raw_manifest_path": decl_path,
        "resolved_canonical_path": str(full_path.relative_to(ROOT)).replace("\\", "/") if file_exists else None,
        "is_valid": file_exists and sha_match,
        "notes": "Exact match" if (file_exists and sha_match) else ("Missing file" if not file_exists else "SHA mismatch")
    })

# 1. Output video_inventory.json
git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

inventory_output = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "git_head": git_head,
    "manifest_path": "data/negative_videos_manifest.json",
    "total_manifest_entries": len(manifest_entries),
    "total_physical_files_found": len(disk_files),
    "files_present": len(inventory_records) - len(missing_records),
    "files_missing": len(missing_records),
    "duplicate_video_ids": duplicate_ids,
    "duplicate_hashes": duplicate_hashes,
    "recovery_status": "RECOVERED_FULL_DATASET" if len(missing_records) == 0 else "PARTIAL_DATASET_ONLY",
    "all_accessible": len(missing_records) == 0,
    "videos": inventory_records
}

with open(out_dir / "video_inventory.json", "w", encoding="utf-8") as f:
    json.dump(inventory_output, f, indent=2, ensure_ascii=False)

# 2. Output missing_videos.csv
missing_fields = ["video_id", "declared_path", "expected_sha256", "reason"]
with open(out_dir / "missing_videos.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=missing_fields)
    writer.writeheader()
    writer.writerows(missing_records)

# 3. Output path_reconciliation.json
with open(out_dir / "path_reconciliation.json", "w", encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_checked": len(path_reconciliations),
        "valid_count": sum(1 for p in path_reconciliations if p["is_valid"]),
        "reconciliations": path_reconciliations
    }, f, indent=2, ensure_ascii=False)

print("\n--- PHASE 1 SUMMARY ---")
print(f"Manifest entries: {len(manifest_entries)}")
print(f"Files present on disk: {len(inventory_records) - len(missing_records)}")
print(f"Files missing: {len(missing_records)}")
print(f"Duplicate IDs: {len(duplicate_ids)}")
print(f"Duplicate SHA-256: {len(duplicate_hashes)}")
print(f"Recovery status: {inventory_output['recovery_status']}")

