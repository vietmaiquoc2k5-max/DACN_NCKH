"""Repository Size Audit and Duplicate Analysis Tool.

This module inspects tracked files, working-tree objects, Git object database,
and historical blobs to identify oversized files, redundant data, and duplicate
assets in the DACN_NCKH repository.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("RepoSizeAudit")

ROOT = Path(__file__).resolve().parents[1]


def get_sha256(path: Path) -> str:
    """Calculate SHA256 checksum in chunks without excessive RAM consumption."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError) as err:
        LOG.warning("Could not hash %s: %s", path, err)
        return "ERROR_HASHING"


def run_cmd(cmd: list[str], cwd: Path = ROOT) -> str:
    """Run subprocess command safely and return stdout string."""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        LOG.warning("Command %s failed: %s", " ".join(cmd), res.stderr.strip())
    return res.stdout


def get_tracked_files() -> list[str]:
    """Get list of tracked files via git ls-files -z."""
    res = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    raw = res.stdout.split(b"\0")
    files = [f.decode("utf-8", errors="replace") for f in raw if f]
    return files


def check_gitignored_batch(files: list[str]) -> set[str]:
    """Check which tracked files match current .gitignore rules using git check-ignore."""
    ignored = set()
    chunk_size = 500
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        payload = "\n".join(chunk).encode("utf-8")
        try:
            p = subprocess.Popen(
                ["git", "check-ignore", "--stdin"],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, _ = p.communicate(input=payload)
            for line in out.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if line:
                    ignored.add(line)
        except Exception as e:
            LOG.warning("Batch check-ignore error: %s", e)
    return ignored


def categorize_file(rel_path: str, ext: str, size: int) -> tuple[str, str, str, str, str]:
    """Classify file into category, action, justification, risk level, and required_by.
    
    Returns:
        (suspected_category, recommended_action, justification, risk_level, required_by)
    """
    top = rel_path.split("/")[0] if "/" in rel_path else "."
    
    # 1. Video files
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}:
        cat = "VIDEO"
        action = "UNTRACK_KEEP_LOCAL"
        just = "Large raw video stream; should remain local or external storage, not in git history."
        risk = "MEDIUM"
        req = "Temporal evaluation / RTSP simulation"
        return cat, action, just, risk, req

    # 2. Checkpoint weights
    if ext in {".pt", ".onnx", ".engine", ".xml", ".bin"}:
        cat = "MODEL_WEIGHTS"
        if rel_path in {
            "runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt",
            "yolov8n.pt",
            "best.pt"
        }:
            action = "KEEP_TRACKED"
            just = "Core baseline model or reference weight required for inference/training."
            risk = "HIGH"
            req = "System runtime & Week 4/5 evaluation baseline"
        elif "weights/epoch" in rel_path or "weights/last.pt" in rel_path:
            action = "DELETE_CANDIDATE"
            just = "Intermediate training checkpoint; best.pt is already retained."
            risk = "LOW"
            req = "None (Intermediate)"
        else:
            action = "UNTRACK_KEEP_LOCAL"
            just = "Trained experiment weights; should be stored in artifact registry or local runs/."
            risk = "MEDIUM"
            req = "Experiment audit"
        return cat, action, just, risk, req

    # 3. Datasets images
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        if top in {"datasets", "dataset", "FireSmokeLabeler", "uploads"}:
            cat = "IMAGE_DATASET"
            if top == "datasets" and "datasets/vn_dataset/" in rel_path:
                action = "MANUAL_REVIEW"
                just = "Vietnamese camera frames. Human-verified frames exist; raw untracked candidates should be separated."
                risk = "HIGH"
                req = "MT1, MT4 training & domain adaptation"
            elif top in {"dataset", "FireSmokeLabeler", "uploads"}:
                action = "UNTRACK_KEEP_LOCAL"
                just = "Duplicate or legacy staged dataset directory; should be consolidated into datasets/vn_dataset."
                risk = "MEDIUM"
                req = "Legacy annotation tool / preview uploads"
            else:
                action = "UNTRACK_KEEP_LOCAL"
                just = "Raw dataset images; should be managed via DVC, LFS, or private dataset transfer."
                risk = "MEDIUM"
                req = "Dataset storage"
            return cat, action, just, risk, req
        elif "runs/" in rel_path:
            cat = "TRAINING_OUTPUT"
            if any(name in rel_path for name in ["results.png", "confusion_matrix", "curve"]):
                action = "KEEP_TRACKED"
                just = "Lightweight scientific visual metric artifact."
                risk = "LOW"
                req = "Research documentation & paper figures"
            else:
                action = "DELETE_CANDIDATE"
                just = "Generated train/val batch visualization previews (train_batch*.jpg)."
                risk = "LOW"
                req = "None (Preview)"
            return cat, action, just, risk, req

    # 4. Labels
    if ext == ".txt" and any(k in rel_path for k in ["/labels/", "labels\\"]):
        cat = "LABEL_DATASET"
        if "datasets/vn_dataset/labels/" in rel_path:
            action = "KEEP_TRACKED"
            just = "YOLO annotation labels for Vietnamese dataset."
            risk = "HIGH"
            req = "MT1 Ground Truth"
        elif "handoff/week5/labels/" in rel_path:
            action = "KEEP_TRACKED"
            just = "Human-approved QA verified labels for Week 5 domain adaptation."
            risk = "CRITICAL"
            req = "Week 5 verified training & MT4 adaptation"
        else:
            action = "MANUAL_REVIEW"
            just = "Dataset label file in auxiliary or legacy directory."
            risk = "MEDIUM"
            req = "Auxiliary labels"
        return cat, action, just, risk, req

    # 5. Training outputs / tensorboard / cache
    if "runs/" in rel_path:
        cat = "TRAINING_OUTPUT"
        if ext in {".csv", ".json", ".yaml", ".yml"}:
            action = "KEEP_TRACKED"
            just = "Lightweight metric logs, training parameters, or receipt records."
            risk = "MEDIUM"
            req = "Benchmark reproducibility"
        elif ext == ".cache" or "events.out.tfevents" in rel_path:
            action = "DELETE_CANDIDATE"
            just = "Disposable YOLO dataset scan cache or TensorBoard log."
            risk = "LOW"
            req = "None"
        else:
            action = "UNTRACK_KEEP_LOCAL"
            just = "Generated training run artifact."
            risk = "LOW"
            req = "Training output"
        return cat, action, just, risk, req

    # 6. Source code
    if ext in {".py", ".sh", ".ps1", ".yaml", ".yml", ".toml", ".json", ".sql", ".html", ".css", ".js"}:
        if top in {"src", "scripts", "configs", "baselines", "tests", "ground_truth_studio", ".github"}:
            cat = "SOURCE_CODE"
            action = "KEEP_TRACKED"
            just = "Active pipeline source code or configuration."
            risk = "CRITICAL"
            req = "Core application & pipeline"
            return cat, action, just, risk, req

    # 7. Documentation
    if ext in {".md", ".txt", ".rst", ".pdf", ".docx"}:
        cat = "DOCUMENTATION"
        if ext in {".pdf", ".docx"} and size > 2 * 1024 * 1024:
            action = "EXTERNAL_STORAGE_REQUIRED"
            just = "Large binary document; recommend hosting on cloud or GitHub release."
            risk = "LOW"
            req = "Documentation reference"
        else:
            action = "KEEP_TRACKED"
            just = "Project documentation or outline specification."
            risk = "HIGH"
            req = "Project specifications & scientific reports"
        return cat, action, just, risk, req

    # 8. Archive
    if ext in {".zip", ".rar", ".tar", ".gz", ".whl", ".7z"}:
        cat = "ARCHIVE"
        action = "EXTERNAL_STORAGE_REQUIRED"
        just = "Compressed archive file; should not be tracked directly in Git history."
        risk = "MEDIUM"
        req = "External distribution"
        return cat, action, just, risk, req

    # 9. Cache / Temp / Database
    if ext in {".cache", ".sqlite", ".sqlite3", ".log", ".tmp", ".bak", ".pyc"}:
        cat = "CACHE_TEMP"
        action = "DELETE_CANDIDATE"
        just = "Temporary runtime cache or local database."
        risk = "LOW"
        req = "None"
        return cat, action, just, risk, req

    # Default fallback
    cat = "MISC"
    action = "KEEP_TRACKED" if size < 100 * 1024 else "MANUAL_REVIEW"
    just = "Unclassified project asset."
    risk = "LOW" if size < 100 * 1024 else "MEDIUM"
    req = "Unspecified"
    return cat, action, just, risk, req


def get_git_objects_info() -> dict[str, Any]:
    """Inspect Git count-objects and find largest historical blobs."""
    raw_count = run_cmd(["git", "count-objects", "-vH"])
    count_data = {}
    for line in raw_count.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            count_data[k.strip()] = v.strip()

    LOG.info("Finding top 30 largest reachable blobs in Git history...")
    blob_cmd = [
        "git", "rev-list", "--objects", "--all"
    ]
    p_rev = subprocess.Popen(blob_cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True, errors="replace")
    
    # We will pass objects through git cat-file --batch-check
    p_cat = subprocess.Popen(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)"],
        cwd=ROOT,
        stdin=p_rev.stdout,
        stdout=subprocess.PIPE,
        text=True,
        errors="replace"
    )
    if p_rev.stdout:
        p_rev.stdout.close()
    
    blobs = []
    if p_cat.stdout:
        for line in p_cat.stdout:
            parts = line.strip().split(" ", 3)
            if len(parts) >= 3 and parts[1] == "blob":
                sha = parts[0]
                sz = int(parts[2])
                name = parts[3] if len(parts) > 3 else "unknown"
                blobs.append((sz, sha, name))
    p_cat.wait()

    blobs.sort(key=lambda x: x[0], reverse=True)
    top_blobs = blobs[:50]
    
    return {
        "count_objects": count_data,
        "top_blobs": [
            {"size_bytes": b[0], "size_mb": round(b[0] / (1024 * 1024), 2), "sha": b[1], "path": b[2]}
            for b in top_blobs
        ]
    }


def main() -> None:
    LOG.info("Starting comprehensive repository size audit...")
    output_dir = ROOT / "reports/repository_cleanup"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracked_files = get_tracked_files()
    LOG.info("Enumerated %d tracked files via git ls-files.", len(tracked_files))

    LOG.info("Checking gitignore matching status across tracked files...")
    ignored_set = check_gitignored_batch(tracked_files)
    LOG.info("Found %d tracked files matching current .gitignore rules.", len(ignored_set))

    rows = []
    top_dir_sizes: dict[str, int] = defaultdict(int)
    top_dir_counts: dict[str, int] = defaultdict(int)
    ext_sizes: dict[str, int] = defaultdict(int)
    ext_counts: dict[str, int] = defaultdict(int)
    cat_sizes: dict[str, int] = defaultdict(int)
    cat_counts: dict[str, int] = defaultdict(int)
    action_sizes: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    
    hash_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_tracked_bytes = 0

    LOG.info("Analyzing file metadata, categories, and checksums...")
    for i, rel in enumerate(tracked_files):
        p = ROOT / rel
        ext = p.suffix.lower()
        top_dir = rel.split("/")[0] if "/" in rel else "."
        
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
            
        total_tracked_bytes += sz
        top_dir_sizes[top_dir] += sz
        top_dir_counts[top_dir] += 1
        ext_sizes[ext or "(none)"] += sz
        ext_counts[ext or "(none)"] += 1
        
        is_ignored = rel in ignored_set
        cat, action, just, risk, req = categorize_file(rel, ext, sz)
        
        cat_sizes[cat] += sz
        cat_counts[cat] += 1
        action_sizes[action] += sz
        action_counts[action] += 1
        
        # Only compute SHA256 for files > 0 bytes
        sha = get_sha256(p) if sz > 0 and sz < 50 * 1024 * 1024 else ("LARGE_FILE_SKIPPED" if sz >= 50 * 1024 * 1024 else "EMPTY_FILE")
        
        row = {
            "path": rel,
            "extension": ext,
            "size_bytes": sz,
            "size_mb": round(sz / (1024 * 1024), 4),
            "tracked": True,
            "gitignored": is_ignored,
            "top_level_directory": top_dir,
            "sha256": sha,
            "suspected_category": cat,
            "recommended_action": action,
            "justification": just,
            "risk_level": risk,
            "required_by": req
        }
        rows.append(row)
        
        if sha not in {"ERROR_HASHING", "LARGE_FILE_SKIPPED", "EMPTY_FILE"}:
            hash_map[sha].append(row)

        if (i + 1) % 5000 == 0:
            LOG.info("Processed %d / %d files...", i + 1, len(tracked_files))

    # Sort rows by size descending
    rows.sort(key=lambda r: r["size_bytes"], reverse=True)

    # 1. Write size_audit.csv
    csv_path = output_dir / "size_audit.csv"
    LOG.info("Writing size audit to %s...", csv_path)
    fieldnames = [
        "path", "extension", "size_bytes", "size_mb", "tracked", "gitignored",
        "top_level_directory", "sha256", "suspected_category",
        "recommended_action", "justification", "risk_level", "required_by"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 2. Duplicate Detection
    LOG.info("Searching for duplicate files across repository...")
    duplicates = []
    dup_total_wasted_bytes = 0
    for sha, group in hash_map.items():
        if len(group) > 1:
            total_instances = len(group)
            single_size = group[0]["size_bytes"]
            wasted = single_size * (total_instances - 1)
            dup_total_wasted_bytes += wasted
            for item in group:
                duplicates.append({
                    "sha256": sha,
                    "file_size_bytes": single_size,
                    "file_size_mb": round(single_size / (1024 * 1024), 4),
                    "duplicate_count": total_instances,
                    "wasted_bytes": wasted,
                    "wasted_mb": round(wasted / (1024 * 1024), 4),
                    "path": item["path"],
                    "top_level_directory": item["top_level_directory"],
                    "category": item["suspected_category"],
                    "action": item["recommended_action"]
                })

    duplicates.sort(key=lambda d: (-d["wasted_bytes"], d["sha256"]))
    dup_csv_path = output_dir / "duplicates.csv"
    LOG.info("Writing duplicates to %s (Found %d duplicate groups)...", dup_csv_path, len([g for g in hash_map.values() if len(g) > 1]))
    dup_fields = [
        "sha256", "file_size_bytes", "file_size_mb", "duplicate_count",
        "wasted_bytes", "wasted_mb", "path", "top_level_directory", "category", "action"
    ]
    with open(dup_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dup_fields)
        writer.writeheader()
        writer.writerows(duplicates)

    # 3. Git Objects & Blobs Inspection
    git_info = get_git_objects_info()

    # 4. Generate Summary JSON for reporting
    summary = {
        "total_tracked_files": len(tracked_files),
        "total_tracked_bytes": total_tracked_bytes,
        "total_tracked_mb": round(total_tracked_bytes / (1024 * 1024), 2),
        "total_tracked_gb": round(total_tracked_bytes / (1024 ** 3), 3),
        "git_object_database": git_info["count_objects"],
        "top_directories_by_bytes": [
            {"directory": d, "files": top_dir_counts[d], "bytes": sz, "mb": round(sz / (1024 * 1024), 2), "pct": round(sz / total_tracked_bytes * 100, 2) if total_tracked_bytes else 0}
            for d, sz in sorted(top_dir_sizes.items(), key=lambda x: x[1], reverse=True)
        ],
        "top_extensions_by_bytes": [
            {"extension": e, "files": ext_counts[e], "bytes": sz, "mb": round(sz / (1024 * 1024), 2)}
            for e, sz in sorted(ext_sizes.items(), key=lambda x: x[1], reverse=True)[:20]
        ],
        "category_summary": [
            {"category": c, "files": cat_counts[c], "bytes": sz, "mb": round(sz / (1024 * 1024), 2), "pct": round(sz / total_tracked_bytes * 100, 2) if total_tracked_bytes else 0}
            for c, sz in sorted(cat_sizes.items(), key=lambda x: x[1], reverse=True)
        ],
        "action_summary": [
            {"action": a, "files": action_counts[a], "bytes": sz, "mb": round(sz / (1024 * 1024), 2), "pct": round(sz / total_tracked_bytes * 100, 2) if total_tracked_bytes else 0}
            for a, sz in sorted(action_sizes.items(), key=lambda x: x[1], reverse=True)
        ],
        "duplicate_summary": {
            "unique_duplicate_hashes": len([g for g in hash_map.values() if len(g) > 1]),
            "total_duplicate_file_entries": len(duplicates),
            "total_wasted_bytes": dup_total_wasted_bytes,
            "total_wasted_mb": round(dup_total_wasted_bytes / (1024 * 1024), 2)
        },
        "tracked_files_matching_gitignore": {
            "count": len(ignored_set),
            "sample": sorted(list(ignored_set))[:50]
        },
        "top_20_largest_tracked_files": [
            {"path": r["path"], "size_mb": r["size_mb"], "category": r["suspected_category"], "action": r["recommended_action"]}
            for r in rows[:20]
        ],
        "top_20_historical_blobs": git_info["top_blobs"][:20]
    }

    summary_path = output_dir / "audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOG.info("Saved summary report to %s", summary_path)

    print("\n" + "=" * 80)
    print("  REPOSITORY SIZE AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Tracked Files: {summary['total_tracked_files']:,}")
    print(f"Total Tracked Size:  {summary['total_tracked_mb']:,} MB ({summary['total_tracked_gb']} GB)")
    print(f"Git Pack Size:       {summary['git_object_database'].get('size-pack', 'N/A')}")
    print("\nTop Directories by Tracked Size:")
    for d in summary["top_directories_by_bytes"][:10]:
        print(f"  - {d['directory']:<25}: {d['mb']:>10.2f} MB ({d['pct']:>5.1f}%) | {d['files']:>6} files")
    
    print("\nRecommended Action Breakdown:")
    for a in summary["action_summary"]:
        print(f"  - {a['action']:<25}: {a['mb']:>10.2f} MB ({a['pct']:>5.1f}%) | {a['files']:>6} files")

    print("\nDuplicates Summary:")
    print(f"  - Unique duplicate groups: {summary['duplicate_summary']['unique_duplicate_hashes']}")
    print(f"  - Total duplicate entries: {summary['duplicate_summary']['total_duplicate_file_entries']}")
    print(f"  - Total wasted storage:    {summary['duplicate_summary']['total_wasted_mb']} MB")

    print(f"\nFiles Tracked but matching .gitignore: {summary['tracked_files_matching_gitignore']['count']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

