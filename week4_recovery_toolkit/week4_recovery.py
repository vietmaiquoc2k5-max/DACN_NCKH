#!/usr/bin/env python3
"""Non-destructive Week 4 source, inference, human-GT and evaluation readiness tool.

Designed for DACN_NCKH. It does not perform ML inference, invent human reviews,
change checkpoints, or edit historical artifacts. Python 3.10+; stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("week4_recovery")

MANIFEST_REL = Path("data/negative_videos_manifest.json")
INFERENCE_DIR = Path("runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full")
REUSE_REL = Path("runs/benchmarks/week4_fix4_verification/recalculated/inference_reuse_per_video.json")
AUDIT_REL = Path("runs/benchmarks/week4_final_recovery/video_audit_final.json")
GT_REL = Path("data/ground_truth/events.jsonl")
FORM_KEYS = ("negative_annotator_01.csv", "negative_annotator_02.csv", "negative_adjudication.csv", "positive_annotator_02.csv", "positive_adjudication.csv")


def now_utc() -> str:
    """Return an offset-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Hash files incrementally without loading video into RAM.

    Args:
        path: Local file path.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Load JSON file, failing explicitly if absent or invalid."""
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV with strict row shape and unique header names."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid CSV header: {path}")
        data = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in data):
        raise ValueError(f"Malformed CSV row: {path}")
    return data


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, preserve: bool = False) -> bool:
    """Write CSV, never replacing human forms when preserve=True."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if preserve and path.exists():
        LOGGER.warning("PRESERVED existing human review file: %s", path)
        return False
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_json(path: Path, value: Any) -> None:
    """Write machine-readable output into a dedicated output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def manifest_entries(root: Path) -> list[dict[str, Any]]:
    """Load and validate unique manifest video IDs and hashes."""
    items = read_json(root / MANIFEST_REL)["videos"]
    ids: set[str] = set()
    for item in items:
        video_id, digest = item.get("video_id"), item.get("sha256")
        if not isinstance(video_id, str) or not video_id or video_id in ids:
            raise ValueError(f"Duplicate/invalid video_id in manifest: {video_id!r}")
        if not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest.lower()):
            raise ValueError(f"Manifest lacks valid SHA-256 for {video_id}")
        ids.add(video_id)
    return items


def safe_target(root: Path, rel: str) -> Path:
    """Resolve declared video path without allowing filesystem escape."""
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or not rel:
        raise ValueError(f"Unsafe manifest path: {rel}")
    path = (root / rel_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes repository: {rel}")
    return path


def source_roots(root: Path, explicit: Iterable[Path]) -> list[Path]:
    """Restrict scanning to the project's data locations and specified directories."""
    candidates = [root / "datasets", root / "FireSmokeLabeler" / "videos", root / "negative_videos", *explicit]
    found: list[Path] = []
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.is_dir() and path not in found:
            found.append(path)
    return found


def locate_videos(root: Path, entries: list[dict[str, Any]], search: list[Path], *, recover: str = "none") -> list[dict[str, Any]]:
    """Locate videos by name, size and SHA-256; optionally restore verified originals.

    Args:
        root: Repository root.
        entries: Validated manifest entries.
        search: Directories that the operator explicitly allows for scanning.
        recover: none, copy or hardlink; never overwrites an existing target.

    Returns:
        Detailed inventory records per manifest video.
    """
    missing_basenames = {Path(str(item["relative_path"])).name.lower() for item in entries}
    candidates: dict[str, list[Path]] = defaultdict(list)
    for directory in source_roots(root, search):
        LOGGER.info("Scanning allowed directory %s", directory)
        for path in directory.rglob("*"):
            if path.is_file() and path.name.lower() in missing_basenames:
                candidates[path.name.lower()].append(path)
    digest_cache: dict[Path, str] = {}

    def checked_digest(path: Path) -> str:
        if path not in digest_cache:
            digest_cache[path] = sha256_file(path)
        return digest_cache[path]

    results: list[dict[str, Any]] = []
    for item in entries:
        rel = str(item["relative_path"])
        target = safe_target(root, rel)
        want = str(item["sha256"]).lower()
        size = item.get("file_size_bytes")
        result: dict[str, Any] = {"video_id": item["video_id"], "declared_path": rel,
                                  "expected_sha256": want, "source_path": None,
                                  "actual_sha256": None, "status": "MISSING", "matching_paths": None}
        if target.exists():
            if not target.is_file():
                result["status"] = "NON_FILE_AT_EXPECTED_PATH"
            else:
                actual = checked_digest(target)
                result["actual_sha256"] = actual
                result["status"] = "PRESENT_VERIFIED" if actual == want else "HASH_MISMATCH"
                if actual == want:
                    result["source_path"] = str(target)
            results.append(result)
            continue
        matches: list[Path] = []
        for candidate in candidates.get(target.name.lower(), []):
            try:
                if size is not None and candidate.stat().st_size != int(size):
                    continue
                if checked_digest(candidate) == want:
                    matches.append(candidate)
            except (OSError, ValueError) as exc:
                LOGGER.warning("Skipping unreadable candidate %s: %s", candidate, exc)
        if len(matches) == 1:
            result["source_path"] = str(matches[0])
            result["actual_sha256"] = want
            result["status"] = "FOUND_ELSEWHERE"
            if recover != "none":
                target.parent.mkdir(parents=True, exist_ok=True)
                if recover == "hardlink":
                    # os.link does not overwrite an existing path.
                    os.link(matches[0], target)
                elif recover == "copy":
                    # Exclusive mode forbids replacing source files or existing targets.
                    try:
                        with matches[0].open("rb") as src, target.open("xb") as dst:
                            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                        if sha256_file(target) != want:
                            target.unlink()  # Only removes file freshly created by this call.
                            raise ValueError("SHA-256 mismatch after copying")
                    except Exception:
                        # A failure during copying must not leave a partial file accepted at the manifest path.
                        # Do not remove files that another process created concurrently.
                        LOGGER.exception("Cannot copy source for %s; inspect any partial target: %s", item["video_id"], target)
                        raise
                result["status"] = "RESTORED_VERIFIED"
        elif len(matches) > 1:
            result["status"] = "MULTIPLE_IDENTICAL_MATCHES_SELECT_MANUALLY"
            result["matching_paths"] = ";".join(str(path) for path in matches)
        results.append(result)
    return results


def cmd_inventory(args: argparse.Namespace) -> int:
    """Create recovery plan; recover matched files only with explicit flag."""
    root = args.repo.resolve()
    results = locate_videos(root, manifest_entries(root), args.search_dir, recover=args.recover)
    counts = dict(Counter(item["status"] for item in results))
    out = args.output.resolve()
    write_json(out / "video_inventory_verified.json", {"generated_at": now_utc(), "repository_root": str(root),
                                                      "requested_recovery_mode": args.recover, "counts": counts,
                                                      "results": results})
    write_csv(out / "video_recovery_plan.csv", results,
              ["video_id", "declared_path", "expected_sha256", "source_path", "actual_sha256", "status", "matching_paths"])
    LOGGER.info("Inventory: %s; report: %s", counts, out)
    return 0 if all(r["status"] in {"PRESENT_VERIFIED", "RESTORED_VERIFIED"} for r in results) else 2


def cmd_provenance(args: argparse.Namespace) -> int:
    """Reconcile evidence without interpreting missing timestamps as full coverage."""
    root, out = args.repo.resolve(), args.output.resolve()
    entries = manifest_entries(root)
    inf_dir = root / INFERENCE_DIR
    run = read_json(inf_dir / "run_manifest.json")
    inf = read_csv(inf_dir / "video_inference_summary.csv")
    audit = read_json(root / AUDIT_REL)
    audit_by_id = {item["video_id"]: item for item in audit["videos"]}
    reuse_path = getattr(args, "reuse_report", None) or root / REUSE_REL
    reuse_by_id = {}
    if reuse_path.exists():
        reuse_by_id = {item["video_id"]: item for item in read_json(reuse_path)}
    inf_by_id = {item["video_id"]: item for item in inf}
    if len(inf_by_id) != len(inf):
        raise ValueError("Duplicate video IDs in inference summary")
    records = []
    for item in entries:
        video_id = item["video_id"]
        path = safe_target(root, str(item["relative_path"]))
        exists = path.is_file()
        actual_sha = sha256_file(path) if exists else None
        sha_ok = exists and actual_sha == str(item["sha256"]).lower()
        inference = inf_by_id.get(video_id)
        audited = audit_by_id.get(video_id)
        prior = reuse_by_id.get(video_id) or {}
        decode_ok = False
        if inference and audited:
            try:
                decode_ok = int(inference["total_decoded_frames"]) == int(audited["full_decoded_frames"])
            except (ValueError, KeyError, TypeError):
                decode_ok = False
        timestamp_proven = (prior.get("recommendation") == "REUSE_VERIFIED"
                            and prior.get("full_timeline_independently_verified") is True)
        if not exists:
            decision = "BLOCKED_MISSING_INPUT"
        elif not sha_ok:
            decision = "BLOCKED_HASH_MISMATCH"
        elif not inference or not audited or not decode_ok:
            decision = "RERUN_CANDIDATE_RECONCILIATION_REQUIRED"
        elif timestamp_proven:
            decision = "REUSE_VERIFIED_BY_EXISTING_REPORT_RECHECK_RUN_BINDING"
        else:
            decision = "UNVERIFIED_TIMELINE_DO_NOT_RERUN_AUTOMATICALLY"
        records.append({"video_id": video_id, "declared_path": item["relative_path"],
                        "source_exists": exists, "manifest_source_sha256_match": bool(sha_ok),
                        "audit_inference_decoded_count_match": decode_ok,
                        "historical_run_id": run.get("run_id"),
                        "current_source_sha256": actual_sha,
                        "historical_frame_timestamp_proof": bool(timestamp_proven),
                        "decision": decision})
    counts = dict(Counter(row["decision"] for row in records))
    report = {"generated_at": now_utc(), "scope": "local input evidence only; no model inference executed",
              "historical_run_id": run.get("run_id"),
              "historical_source_commit": run.get("source_code_base_commit"),
              "historical_packaged_commit": run.get("full_git_sha"),
              "historical_manifest_sha256": run.get("manifest_sha256"),
              "current_manifest_sha256": sha256_file(root / MANIFEST_REL),
              "historical_checkpoint_sha256": run.get("checkpoint_sha256"),
              "counts": counts, "videos": records}
    checkpoint_rel = str(run.get("model_checkpoint") or "").replace("\\", "/")
    if checkpoint_rel:
        checkpoint = safe_target(root, checkpoint_rel)
        report["checkpoint_sha256_current"] = sha256_file(checkpoint) if checkpoint.is_file() else None
        report["checkpoint_sha256_matches"] = (report["checkpoint_sha256_current"] == run.get("checkpoint_sha256"))
    write_json(out / "inference_provenance_gate.json", report)
    write_csv(out / "inference_per_video_decisions.csv", records, list(records[0]))
    LOGGER.info("Provenance: %s; no automatic inference", counts)
    return 0 if all(r["decision"].startswith("REUSE_VERIFIED") for r in records) else 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL with explicit line number for malformed records."""
    data: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for index, raw in enumerate(stream, 1):
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError(f"Non-object JSONL record at {path}:{index}")
                data.append(value)
    return data


def cmd_prepare_reviews(args: argparse.Namespace) -> int:
    """Create blank, independent human-review forms without rewriting prior work."""
    root, out = args.repo.resolve(), args.review_dir.resolve()
    entries = manifest_entries(root)
    positives = read_jsonl(root / GT_REL)
    positive_video_ids = sorted({row["video_id"] for row in positives})
    neg_fields = ["video_id", "relative_path", "source_sha256", "reviewer_id", "reviewed_at_utc",
                  "decision", "fire_intervals_json", "smoke_intervals_json", "evidence_ref", "signoff"]
    neg_rows = [{**{key: "" for key in neg_fields}, "video_id": item["video_id"],
                 "relative_path": item["relative_path"], "source_sha256": item["sha256"]} for item in entries]
    created = {}
    for number in (1, 2):
        filename = f"negative_annotator_0{number}.csv"
        created[filename] = write_csv(out / filename, neg_rows, neg_fields, preserve=True)
    negative_adj_fields = ["video_id", "source_sha256", "approval", "adjudicator_id",
                           "adjudicated_at_utc", "evidence_ref", "notes"]
    negative_adj_rows = [{"video_id": item["video_id"], "source_sha256": item["sha256"],
                          **{key: "" for key in negative_adj_fields[2:]}} for item in entries]
    created["negative_adjudication.csv"] = write_csv(out / "negative_adjudication.csv",
                                                     negative_adj_rows, negative_adj_fields, preserve=True)
    positive_fields = ["video_id", "relative_path", "source_sha256", "reviewer_id", "reviewed_at_utc",
                       "annotation_events_json", "evidence_ref", "signoff"]
    pos_rows = [{"video_id": video_id, "relative_path": f"FireSmokeLabeler/videos/{video_id}.avi",
                 **{key: "" for key in positive_fields[2:]}} for video_id in positive_video_ids]
    created["positive_annotator_02.csv"] = write_csv(out / "positive_annotator_02.csv",
                                                     pos_rows, positive_fields, preserve=True)
    positive_adj_fields = ["gt_index", "gt_snapshot_sha256", "video_id", "label", "start_sec", "end_sec", "reviewer_02_event_index",
                           "approval", "adjudicator_id", "adjudicated_at_utc", "evidence_ref", "notes"]
    gt_sha = sha256_file(root / GT_REL)
    positive_adj_rows = [{"gt_index": index, "gt_snapshot_sha256": gt_sha, "video_id": event["video_id"], "label": event["label"],
                          "start_sec": event["start_sec"], "end_sec": event["end_sec"],
                          **{key: "" for key in positive_adj_fields[6:]}} for index, event in enumerate(positives, 1)]
    created["positive_adjudication.csv"] = write_csv(out / "positive_adjudication.csv",
                                                     positive_adj_rows, positive_adj_fields, preserve=True)
    instructions = out / "HUMAN_REVIEW_PROTOCOL.md"
    if not instructions.exists():
        instructions.write_text("""# Quy trình gán nhãn độc lập — Không được dùng kết quả AI làm GT

1. Mỗi annotator mở video nguồn gốc, xem toàn bộ video (không xem inference boxes, sự kiện dự đoán hoặc nhãn người kia).
2. `negative_annotator_01.csv` và `_02.csv` phải được điền bởi hai người thực sự khác nhau, mỗi người ký `signoff=YES`, ghi ID, thời điểm UTC, bằng chứng review và nhãn `NEGATIVE`, `FIRE`, `SMOKE` hoặc `UNCERTAIN`. Nếu có FIRE/SMOKE, ghi các khoảng thời gian JSON và gửi adjudication.
3. `positive_annotator_02.csv` chỉ cho reviewer 02 xem danh sách video, KHÔNG cho thấy nhãn và mốc sự kiện của reviewer 01. `annotation_events_json` là danh sách JSON `[ {"label":"FIRE","start_sec":0.5,"end_sec":4.2} ]` (nhập bằng chứng thật, không copy ví dụ). Điền SHA-256 video thực tế, ID, timestamp, evidence_ref và signoff.
4. Chỉ SAU khi reviewer 02 hoàn tất mới mở các form `*_adjudication.csv`, đối chiếu nhãn, onset/offset và giải quyết bất đồng bằng xem lại video. Ghi approval `APPROVE` và ID/timestamp/evidence của người đối soát; không tự điền.
5. Nếu nguồn bị lỗi giải mã, thời lượng chưa rõ, người xem không chắc chắn hoặc file chưa có, để PENDING và ghi nguyên nhân. Không ép VERIFIED.
6. Các form được tạo một lần. Chạy lại lệnh không ghi đè tiến độ review đã nhập. Lưu bản gốc và chữ ký bên ngoài CSV nếu cần cho phản biện.
7. Chỉ chạy đánh giá chính thức sau khi đủ >=5 giờ video âm tính đáp ứng audit + inference timeline và GT dương tính được kiểm tra chéo, có validation lock trước test.
""", encoding="utf-8")
    write_json(out / "review_preparation_status.json", {"generated_at": now_utc(),
                "negative_video_rows": len(entries), "positive_video_rows": len(pos_rows),
                "positive_event_rows": len(positives), "files_created": created,
                "human_reviews_performed_by_this_script": 0, "gt_status": "PENDING_HUMAN_REVIEW"})
    LOGGER.info("Human review forms prepared at %s, preserving existing forms", out)
    return 0


def valid_utc_timestamp(value: str | None) -> bool:
    """Accept only timezone-aware review timestamps, not placeholder text."""
    if not value:
        return False
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return instant.tzinfo is not None and instant.utcoffset() is not None
    except ValueError:
        return False


def verified_negative_ids(root: Path, review_dir: Path) -> tuple[set[str], dict[str, str]]:
    """Validate explicit human signoffs; never change repository GT statuses."""
    entries = {row["video_id"]: row for row in manifest_entries(root)}
    files = [review_dir / name for name in ("negative_annotator_01.csv", "negative_annotator_02.csv", "negative_adjudication.csv")]
    if any(not f.is_file() for f in files):
        return set(), {video_id: "REVIEW_FORMS_MISSING" for video_id in entries}
    sources = []
    for path in files:
        rows = read_csv(path)
        mapped = {row["video_id"]: row for row in rows}
        if len(mapped) != len(rows):
            raise ValueError(f"Duplicate video_id in {path}")
        sources.append(mapped)
    verified, reasons = set(), {}
    for video_id, manifest in entries.items():
        first, second, adjudicator = (source.get(video_id, {}) for source in sources)
        if any(not row for row in (first, second, adjudicator)):
            reasons[video_id] = "REVIEW_ROW_MISSING"
            continue
        ids = [first.get("reviewer_id", "").strip(), second.get("reviewer_id", "").strip()]
        if not all(ids) or ids[0] == ids[1]:
            reasons[video_id] = "TWO_INDEPENDENT_REVIEWERS_REQUIRED"
            continue
        if not all(row.get("source_sha256", "").lower() == str(manifest["sha256"]).lower()
                   for row in (first, second, adjudicator)):
            reasons[video_id] = "REVIEW_SOURCE_SHA_MISMATCH"
            continue
        if not all(row.get("decision", "").upper().strip() == "NEGATIVE" for row in (first, second)):
            reasons[video_id] = "NEGATIVE_CONSENSUS_NOT_ESTABLISHED"
            continue
        if any(row.get("fire_intervals_json", "").strip() not in ("", "[]")
               or row.get("smoke_intervals_json", "").strip() not in ("", "[]")
               for row in (first, second)):
            reasons[video_id] = "NEGATIVE_LABEL_CONTRADICTS_POSITIVE_INTERVALS"
            continue
        if not all(row.get("signoff", "").upper().strip() == "YES"
                   and valid_utc_timestamp(row.get("reviewed_at_utc")) and row.get("evidence_ref")
                   for row in (first, second)):
            reasons[video_id] = "REVIEW_SIGNOFF_OR_EVIDENCE_MISSING"
            continue
        if (adjudicator.get("approval", "").upper().strip() != "APPROVE"
            or not adjudicator.get("adjudicator_id")
            or not valid_utc_timestamp(adjudicator.get("adjudicated_at_utc"))
            or not adjudicator.get("evidence_ref")):
            reasons[video_id] = "HUMAN_ADJUDICATION_PENDING"
            continue
        video_path = safe_target(root, str(manifest["relative_path"]))
        if not video_path.is_file() or sha256_file(video_path) != str(manifest["sha256"]).lower():
            reasons[video_id] = "SOURCE_VIDEO_MISSING_OR_HASH_MISMATCH"
            continue
        verified.add(video_id)
    return verified, reasons


def verified_positive_events(root: Path, review_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Require explicit human adjudication and independently authored event bounds."""
    originals = read_jsonl(root / GT_REL)
    reviewer_path, adj_path = review_dir / "positive_annotator_02.csv", review_dir / "positive_adjudication.csv"
    if not reviewer_path.is_file() or not adj_path.is_file():
        return [], {str(i): "REVIEW_FORMS_MISSING" for i in range(1, len(originals)+1)}
    reviewers = read_csv(reviewer_path)
    approvals = read_csv(adj_path)
    if len({r["video_id"] for r in reviewers}) != len(reviewers):
        raise ValueError("Duplicate positive reviewer video ID")
    if len({r["gt_index"] for r in approvals}) != len(approvals):
        raise ValueError("Duplicate adjudicated GT index")
    by_video = {r["video_id"]: r for r in reviewers}
    by_index = {int(r["gt_index"]): r for r in approvals}
    verified: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    matched_second_indices: set[tuple[str, int]] = set()
    gt_sha = sha256_file(root / GT_REL)
    for idx, original in enumerate(originals, 1):
        vid = original["video_id"]
        review = by_video.get(vid)
        adj = by_index.get(idx)
        if not review or not adj:
            reasons[str(idx)] = "REVIEW_OR_ADJUDICATION_ROW_MISSING"
            continue
        if adj.get("gt_snapshot_sha256") != gt_sha:
            reasons[str(idx)] = "GROUND_TRUTH_VERSION_MISMATCH"
            continue
        if (review.get("signoff", "").upper().strip() != "YES"
            or not valid_utc_timestamp(review.get("reviewed_at_utc"))
            or not all(review.get(field) for field in ("reviewer_id", "evidence_ref", "source_sha256"))):
            reasons[str(idx)] = "INDEPENDENT_REVIEW_MISSING"
            continue
        if review["reviewer_id"] == original.get("annotator_01"):
            reasons[str(idx)] = "ANNOTATORS_NOT_INDEPENDENT"
            continue
        path = safe_target(root, review["relative_path"])
        if not path.is_file() or sha256_file(path) != review["source_sha256"].lower():
            reasons[str(idx)] = "POSITIVE_VIDEO_MISSING_OR_SOURCE_HASH_MISMATCH"
            continue
        try:
            events = json.loads(review["annotation_events_json"])
            second_index = int(adj["reviewer_02_event_index"])
            second = events[second_index]
            start_02, end_02 = float(second["start_sec"]), float(second["end_sec"])
            agrees = (0 <= start_02 <= end_02 and second["label"].upper() == original["label"].upper()
                      and abs(start_02 - float(original["start_sec"])) <= 0.5
                      and abs(end_02 - float(original["end_sec"])) <= 0.5)
        except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
            reasons[str(idx)] = "REVIEW_EVENT_INVALID_OR_UNMAPPED"
            continue
        if not agrees:
            reasons[str(idx)] = "REVIEWER_DISAGREEMENT_ADJUDICATE"
            continue
        if (vid, second_index) in matched_second_indices:
            reasons[str(idx)] = "REVIEW_EVENT_MAPPED_MORE_THAN_ONCE"
            continue
        if (adj.get("approval", "").upper().strip() != "APPROVE"
            or not adj.get("adjudicator_id")
            or not valid_utc_timestamp(adj.get("adjudicated_at_utc"))
            or not adj.get("evidence_ref")
            or adj.get("video_id") != vid or adj.get("label", "").upper() != original["label"].upper()):
            reasons[str(idx)] = "HUMAN_ADJUDICATION_PENDING"
            continue
        matched_second_indices.add((vid, second_index))
        verified.append({"gt_index": idx, **original, "human_evidence": {"annotator_02": review["reviewer_id"],
                         "adjudicator": adj["adjudicator_id"], "reviewer_02_event_index": second_index}})
    return verified, reasons


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Gate verified evaluation; never compute official metrics from pending GT."""
    root, out, review_dir = args.repo.resolve(), args.output.resolve(), args.review_dir.resolve()
    neg, neg_reasons = verified_negative_ids(root, review_dir)
    pos, pos_reasons = verified_positive_events(root, review_dir)
    all_entries = manifest_entries(root)
    audit = read_json(root / AUDIT_REL)
    audits = {item["video_id"]: item for item in audit["videos"]}
    reuse_path = getattr(args, "reuse_report", None) or root / REUSE_REL
    reuse = {item["video_id"]: item for item in read_json(reuse_path)} if reuse_path.is_file() else {}
    eligible = []
    for entry in all_entries:
        vid = entry["video_id"]
        if vid not in neg:
            continue
        evidence = audits.get(vid, {})
        frame = reuse.get(vid, {})
        if not (evidence.get("decode_status") == "NORMAL_EOF" and evidence.get("full_decode_verified") is True
                and evidence.get("sha256_verified") is True and frame.get("recommendation") == "REUSE_VERIFIED"
                and frame.get("full_timeline_independently_verified") is True):
            neg_reasons[vid] = "TECHNICAL_OR_INFERENCE_EVIDENCE_NOT_VERIFIED"
            continue
        fps = evidence.get("fps")
        frames = evidence.get("full_decoded_frames")
        if not isinstance(fps, (int, float)) or not isinstance(frames, int) or fps <= 0 or frames <= 0:
            neg_reasons[vid] = "AUTHORITATIVE_DURATION_UNAVAILABLE"
            continue
        eligible.append({"video_id": vid, "exposure_sec": frames / fps})
    seconds = sum(item["exposure_sec"] for item in eligible)
    blockers = []
    if seconds < 5 * 3600:
        blockers.append("LESS_THAN_5_HOURS_VERIFIED_NEGATIVE_EXPOSURE")
    if len(pos) != len(read_jsonl(root / GT_REL)):
        blockers.append("POSITIVE_GROUND_TRUTH_NOT_FULLY_INDEPENDENTLY_VERIFIED")
    if not args.validation_lock or not args.validation_lock.is_file():
        blockers.append("VALIDATION_OPERATING_POINT_LOCK_EVIDENCE_MISSING")
    # Existing inference still lacks bound frame timestamps. This evaluator must not
    # fabricate recall, latency or FAR from nominal-negative historical summaries.
    report = {"generated_at": now_utc(), "source": "human forms and local technical evidence",
              "verified_negative_human_count": len(neg), "strict_and_timeline_eligible_negative_count": len(eligible),
              "verified_negative_exposure_hours": seconds / 3600,
              "positive_events_independently_adjudicated": len(pos),
              "positive_events_total": len(read_jsonl(root / GT_REL)),
              "negative_blockers": neg_reasons, "positive_blockers": pos_reasons,
              "scientific_acceptance": "BLOCKED" if blockers else "READY_FOR_INDEPENDENT_EVENT_EVALUATION",
              "blockers": blockers, "far_per_hour": None, "poisson_ci95": None,
              "event_recall": None, "event_f1": None, "detection_delay_sec": None,
              "automatic_gt_promotion_performed": False,
              "model_inference_executed": False,
              "note": "No official metrics are generated until all human/technical/lock gates pass. Use the repository's audited event evaluator on the exact verified video IDs after resolving these gates."}
    write_json(out / "week4_evaluation_readiness.json", report)
    LOGGER.info("Evaluation readiness: %s; negative hours=%.4f; positive verified=%s", report["scientific_acceptance"], seconds / 3600, len(pos))
    return 0 if not blockers else 2



def cmd_recompute(args: argparse.Namespace) -> int:
    """Recompute metrics only after verified human, timeline and validation-lock gates.

    This delegates event matching and Garwood Poisson CI to the repository's
    existing src.event_eval implementation, avoiding a second metric definition.
    """
    root, out, review_dir = args.repo.resolve(), args.output.resolve(), args.review_dir.resolve()
    # First write a durable BLOCKED report if evidence is incomplete.
    if cmd_evaluate(args) != 0:
        LOGGER.error("BLOCKED: independent GT, timeline coverage or >=5h negative exposure is incomplete")
        return 2
    positives, pos_errors = verified_positive_events(root, review_dir)
    verified_neg, _ = verified_negative_ids(root, review_dir)
    audits = {r["video_id"]: r for r in read_json(root / AUDIT_REL)["videos"]}
    reuse_path = args.reuse_report or root / REUSE_REL
    reuse = {r["video_id"]: r for r in read_json(reuse_path)}
    neg_scope = {v for v in verified_neg if v in audits and v in reuse
                 and audits[v].get("decode_status") == "NORMAL_EOF"
                 and audits[v].get("full_decode_verified") is True
                 and reuse[v].get("recommendation") == "REUSE_VERIFIED"
                 and reuse[v].get("full_timeline_independently_verified") is True}
    hours = sum(float(audits[v]["full_decoded_frames"]) / float(audits[v]["fps"]) for v in neg_scope) / 3600.0
    if not args.positive_provenance or not args.positive_provenance.is_file():
        LOGGER.error("BLOCKED: independently authenticated positive prediction provenance required")
        return 2
    provenance = read_json(args.positive_provenance)
    by_video = {r["video_id"]: r for r in provenance.get("videos", [])}
    pos_ids = {r["video_id"] for r in positives}
    positive_review_by_video = {r["video_id"]: r for r in read_csv(review_dir / "positive_annotator_02.csv")}
    checkpoint_expected = read_json(root / INFERENCE_DIR / "run_manifest.json").get("checkpoint_sha256")
    if not all(v in by_video and by_video[v].get("full_timeline_independently_verified") is True
               and by_video[v].get("source_sha256") == positive_review_by_video[v].get("source_sha256")
               and by_video[v].get("checkpoint_sha256") == checkpoint_expected
               and by_video[v].get("run_id") for v in pos_ids):
        LOGGER.error("BLOCKED: source/run/checkpoint/PTS provenance missing for positive videos")
        return 2
    lock = read_json(args.validation_lock)
    if (lock.get("selected_on") != "validation" or lock.get("locked_before_test") is not True
            or not lock.get("source_commit") or not lock.get("validation_evidence_ref")):
        LOGGER.error("BLOCKED: validation-lock document is incomplete; no test-time tuning allowed")
        return 2
    try:
        sys.path.insert(0, str(root))
        from src.event_eval import Event, evaluate, exact_poisson_ci
    except (ImportError, OSError) as exc:
        LOGGER.error("BLOCKED: cannot import authoritative event evaluator: %s", exc)
        return 3
    neg_events_path = args.negative_episodes or root / "runs/benchmarks/week4_fix4_verification/recalculated/normalized_event_episodes.json"
    pos_a_path = args.positive_baseline or root / "runs/benchmarks/week4_final/frame_level_events.json"
    pos_b_path = args.positive_temporal or root / "runs/benchmarks/week4_final/temporal_confirmed_events.json"
    if (provenance.get("baseline_events_sha256") != sha256_file(pos_a_path)
            or provenance.get("temporal_events_sha256") != sha256_file(pos_b_path)):
        LOGGER.error("BLOCKED: positive predictions do not match authenticated artifact hashes")
        return 2
    neg_data, pos_a, pos_b = (read_json(p) for p in (neg_events_path, pos_a_path, pos_b_path))
    tolerance = lock.get("tolerance_sec")
    early_tol = lock.get("early_tolerance_sec")
    if not isinstance(tolerance, (int, float)) or tolerance < 0 or not isinstance(early_tol, (int, float)) or early_tol < 0:
        LOGGER.error("BLOCKED: validation-locked onset matching tolerances are missing")
        return 2

    def to_event(row: dict[str, Any]) -> Any:
        return Event(video_id=row["video_id"], camera_id=row["camera_id"], label=row["label"],
                     start_sec=float(row["start_sec"]), end_sec=float(row["end_sec"]),
                     confidence=float(row.get("confidence", 1.0)))

    truth = [to_event(row) for row in positives]
    results: dict[str, Any] = {}
    for name, pos_predictions, neg_predictions in (
            ("A", pos_a.get("merged_events", []), neg_data.get("pipeline_a", [])),
            ("B2", pos_b.get("b2_feature_gate_events", []), neg_data.get("pipeline_b2", []))):
        preds = [to_event(row) for row in pos_predictions if row["video_id"] in pos_ids]
        for row in neg_predictions:
            if row["video_id"] not in neg_scope:
                continue
            video = audits[row["video_id"]]
            upper = float(video["full_decoded_frames"]) / float(video["fps"])
            if not (0 <= float(row["start_sec"]) <= float(row["end_sec"]) < upper + 0.001):
                raise ValueError(f"Event outside independently verified exposure: {row['video_id']}")
            preds.append(to_event(row))
        report = evaluate(truth, preds, pos_ids, neg_scope, hours,
                          tolerance_sec=float(tolerance), early_tolerance_sec=float(early_tol))
        count = int(report["counts"]["negative_false_alarms"])
        results[name] = {"counts": report["counts"], "metrics": report["metrics"],
                         "poisson_ci95_far_per_hour": list(exact_poisson_ci(count, hours))}
    far_a = results["A"]["metrics"]["false_alarms_per_hour"]
    far_b = results["B2"]["metrics"]["false_alarms_per_hour"]
    reduction = (far_a - far_b) / far_a if far_a and far_a > 0 else None
    recall_a = results["A"]["metrics"]["event_instance_recall"]
    recall_b = results["B2"]["metrics"]["event_instance_recall"]
    delta_recall_pp = 100.0 * (recall_b - recall_a)
    latency = results["B2"]["metrics"]["mean_detection_delay_sec"]
    criteria = {"negative_exposure_ge_5h": hours >= 5.0,
                "far_reduction_ge_50pct": reduction is not None and reduction >= 0.5,
                "event_recall_drop_le_5pp": delta_recall_pp >= -5.0,
                "mean_delay_le_10_sec": latency is not None and latency <= 10.0}
    write_json(out / "week4_metrics_recomputed.json", {
        "generated_at": now_utc(), "negative_video_ids": sorted(neg_scope),
        "positive_video_ids": sorted(pos_ids), "negative_exposure_hours": hours,
        "positive_gt_events": len(positives), "source_hashes": {
            "negative_episodes": sha256_file(neg_events_path),
            "positive_baseline": sha256_file(pos_a_path),
            "positive_temporal": sha256_file(pos_b_path),
            "validation_lock": sha256_file(args.validation_lock),
            "positive_provenance": sha256_file(args.positive_provenance)},
        "matching_policy": {"tolerance_sec": tolerance, "early_tolerance_sec": early_tol},
        "pipelines": results, "relative_far_reduction": reduction,
        "event_recall_delta_percentage_points": delta_recall_pp,
        "criteria": criteria,
        "status": "MEASURED_REQUIRES_INDEPENDENT_PROVENANCE_AUDIT",
        "note": "Source-bound, human-reviewed results; manual independent confirmation of lock provenance and annotations remains required before official acceptance."})
    LOGGER.info("Metric recomputation finished using authoritative src.event_eval; no GT/source mutations")
    return 0


def parser() -> argparse.ArgumentParser:
    """Build a deterministic CLI with non-destructive defaults."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, required=True, help="Local DACN_NCKH repository path")
    p.add_argument("--output", type=Path, help="Output folder (default under repository runs/benchmarks/week4_assisted_recovery)")
    commands = p.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="Locate missing videos and optionally restore hash-matched files")
    inv.add_argument("--search-dir", type=Path, action="append", default=[], help="Extra permitted directory, repeatable")
    inv.add_argument("--recover", choices=["none", "copy", "hardlink"], default="none", help="Explicitly restore verified video at declared path")
    commands.add_parser("provenance", help="Recheck 79 video inputs and historical inference summaries")
    prep = commands.add_parser("prepare-reviews", help="Prepare separate, blank human review forms")
    prep.add_argument("--review-dir", type=Path, help="Review forms directory")
    ev = commands.add_parser("evaluate", help="Validate human and technical evidence gates before official evaluation")
    ev.add_argument("--review-dir", type=Path, help="Review forms directory")
    ev.add_argument("--reuse-report", type=Path, default=None, help="Updated per-video inference evidence JSON")
    ev.add_argument("--validation-lock", type=Path, default=None, help="Validation-selected operating point lock evidence")
    metrics = commands.add_parser("recompute", help="After ALL gates pass, recompute FAR, Recall, F1 and Delay with repo evaluator")
    metrics.add_argument("--review-dir", type=Path, help="Independent human review directory")
    metrics.add_argument("--reuse-report", type=Path, default=None, help="Source-bound per-video negative inference report")
    metrics.add_argument("--validation-lock", type=Path, required=True, help="Locked validation-selected operating point evidence JSON")
    metrics.add_argument("--positive-provenance", type=Path, required=True, help="Independently verified positive video inference provenance JSON")
    metrics.add_argument("--negative-episodes", type=Path, default=None, help="Normalized negative events from the source-bound inference run")
    metrics.add_argument("--positive-baseline", type=Path, default=None)
    metrics.add_argument("--positive-temporal", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    """Execute selected command with error reporting and meaningful exit codes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = parser()
    args = ap.parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = args.repo / "runs/benchmarks/week4_assisted_recovery" / stamp
    if args.command in ("prepare-reviews", "evaluate", "recompute") and args.review_dir is None:
        args.review_dir = args.repo.parent / "week4_private_human_review"
    try:
        if args.command == "inventory":
            return cmd_inventory(args)
        if args.command == "provenance":
            return cmd_provenance(args)
        if args.command == "prepare-reviews":
            return cmd_prepare_reviews(args)
        if args.command == "evaluate":
            return cmd_evaluate(args)
        if args.command == "recompute":
            return cmd_recompute(args)
        raise ValueError(f"Unknown command: {args.command}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        LOGGER.error("BLOCKED: %s: %s", type(exc).__name__, exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
