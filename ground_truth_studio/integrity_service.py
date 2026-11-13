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
    for row in rows:
        row.update(source_group="POS_HISTORICAL" if row["kind"] == "positive" else "VN_NEG" if row["video_id"].startswith("VN_NEG_") else "LEGACY_NEGATIVE",
                   semantic_status="PENDING_HUMAN_REVIEW", manifest_origin="historical",
                   exclusion_reason=None if row["available"] else "EXCLUDED_MISSING_SOURCE")
    known = {row["video_id"] for row in rows}
    directory = settings.repo / "datasets/negative_videos"
    if directory.is_dir():
        import re
        for path in sorted(directory.iterdir()):
            if not path.is_file() or not re.fullmatch(r"VN_POS_\d+\.mp4", path.name, flags=re.IGNORECASE):
                continue
            video_id = path.stem.upper()
            if video_id in known:
                raise ValueError(f"Catalog video ID collision: {video_id}")
            relative = path.relative_to(settings.repo).as_posix()
            confined(settings.repo, relative)
            rows.append({"video_id": video_id, "kind": "positive_candidate", "relative_path": relative,
                         "expected_sha256": None, "available": True, "technical": {}, "source_group": "VN_POS",
                         "semantic_status": "PENDING_HUMAN_REVIEW", "manifest_origin": "discovered", "exclusion_reason": None})
            known.add(video_id)
    # Persisted source hashes are identity pins, not an implicit technical/semantic approval.
    local_path = settings.repo / "data/video_catalog_local_v2.json"
    if local_path.exists():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        pins = {row["video_id"]: row for row in local["videos"]}
        if len(pins) != len(local["videos"]):
            raise ValueError("Duplicate ID in local source catalog")
        current_audits = {}
        if local.get("technical_audit_path"):
            audit_file = confined(settings.repo, local["technical_audit_path"])
            if audit_file.exists():
                current_audits = {item["video_id"]: item for item in json.loads(audit_file.read_text(encoding="utf-8"))["videos"]}
        for row in rows:
            pin = pins.get(row["video_id"])
            if pin and pin["relative_path"] == row["relative_path"]:
                row["local_source_sha256"] = pin.get("sha256")
                row["expected_sha256"] = row["expected_sha256"] or pin.get("sha256")
                audited = current_audits.get(row["video_id"])
                if audited and audited.get("sha256") == pin.get("sha256") and audited.get("sha256_verified") is True:
                    alignment = row["technical"].get("inference_alignment_verified") is True
                    row["technical"] = dict(audited, inference_alignment_verified=alignment)
                    row["technical_status"] = audited.get("decode_status", "NOT_VERIFIED")
    return rows
