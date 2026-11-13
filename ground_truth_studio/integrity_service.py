"""Read-only frozen GT and source integrity guard."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from src.ground_truth_integrity import validate_frozen_snapshot
from ground_truth_studio.config import Settings, confined


def sha256(path: Path) -> str:
    """Hash source bytes using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen(settings: Settings) -> tuple[str, list[dict[str, Any]]]:
    """Authenticate historical bytes without promoting their annotation claims."""
    path = settings.repo / "data/ground_truth/events_v1.1_snapshot.jsonl"
    checksum = validate_frozen_snapshot(path, path.parent / "checksums_v1.1.sha256")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return checksum, rows


def catalog(settings: Settings) -> list[dict[str, Any]]:
    """Map actual catalog filenames, never infer POS_ID.avi aliases from GT times."""
    rows = []
    with (settings.repo / "data/video_dataset_list.csv").open(encoding="utf-8-sig", newline="") as stream:
        for entry in csv.DictReader(stream):
            if not entry["video_id"].startswith("POS_"):
                continue
            candidates = [entry["relative_path"], "FireSmokeLabeler/videos/" + entry["file_name"]]
            paths = [confined(settings.repo, candidate) for candidate in candidates]
            index = next((i for i, path in enumerate(paths) if path.is_file()), 0)
            rows.append({"video_id": entry["video_id"], "kind": "positive", "relative_path": candidates[index],
                         "expected_sha256": None, "available": paths[index].is_file(), "technical": {}})
    manifest = json.loads((settings.repo / "data/negative_videos_manifest.json").read_text(encoding="utf-8"))
    audit_path = settings.repo / "runs/benchmarks/week4_final_recovery/video_audit_final.json"
    audits = {row["video_id"]: row for row in json.loads(audit_path.read_text(encoding="utf-8"))["videos"]} if audit_path.exists() else {}
    rec_path = settings.repo / "runs/benchmarks/week4_fix4_verification/recalculated/inference_reconciliation_final.json"
    coverage = {row["video_id"]: row for row in json.loads(rec_path.read_text(encoding="utf-8")).get("coverage", [])} if rec_path.exists() else {}
    for entry in manifest["videos"]:
        path = confined(settings.repo, entry["relative_path"])
        technical = dict(audits.get(entry["video_id"], {}))
        technical["inference_alignment_verified"] = coverage.get(entry["video_id"], {}).get("inference_alignment_verified") is True
        rows.append({"video_id": entry["video_id"], "kind": "negative", "relative_path": entry["relative_path"],
                     "expected_sha256": entry.get("sha256"), "available": path.is_file(), "technical": technical})
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Catalog video ID collision")
    return rows
