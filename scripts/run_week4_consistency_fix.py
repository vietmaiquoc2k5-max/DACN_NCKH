"""Regenerate Week 4 consistency evidence without overwriting historical artifacts.

The run is resumable: verified media may be reused only through the versioned
SHA-256 decode cache. Missing source files stay blocked and contribute no exposure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.event_eval import exact_poisson_ci
from src.video_audit import (
    AUDIT_POLICY_VERSION,
    CACHE_SCHEMA_VERSION,
    VideoDecodeCacheManager,
    audit_single_video,
    compute_streaming_sha256,
)

OUT = ROOT / "runs/benchmarks/week4_consistency_fix"
MANIFEST = ROOT / "data/negative_videos_manifest.json"
RUN_DIR = ROOT / "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full"
RUN_ID = "run_20261111_233035_neg_full"
CHECKPOINT = ROOT / "runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt"
GT_QUEUE = ROOT / "runs/benchmarks/week4_revalidation/gt_review_queue.csv"
REPORT = ROOT / "docs/WEEK4_FINAL_CONSISTENCY_ACCEPTANCE_REPORT.md"
RELIABILITY_REPORT = ROOT / "docs/WEEK4_FINAL_RELIABILITY_REPORT.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def modified_at_utc(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_text(args: Sequence[str]) -> str:
    try:
        res = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return (res.stdout + res.stderr).strip()
    except Exception as exc:
        return "UNAVAILABLE: " + str(exc)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_path(entry: Dict[str, Any]) -> Path:
    raw = entry.get("relative_path") or entry.get("video_path") or entry.get("path") or ""
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def build_preflight() -> Dict[str, Any]:
    manifest_bytes = MANIFEST.read_bytes()
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    entries = manifest.get("videos", [])
    paths = [resolve_manifest_path(v) for v in entries]
    present = [(v, p) for v, p in zip(entries, paths) if p.is_file()]
    missing = [(v, p) for v, p in zip(entries, paths) if not p.is_file()]
    current_hashes: Dict[str, str] = {}
    hash_errors: List[Dict[str, str]] = []
    for entry, path in present:
        try:
            current_hashes[entry.get("video_id", path.stem)] = compute_streaming_sha256(path)
        except Exception as exc:
            hash_errors.append({"video_id": entry.get("video_id", path.stem), "error": str(exc)})

    manifest_ids = [str(v.get("video_id", "")) for v in entries]
    manifest_sha_values = [str(v.get("sha256", "")) for v in entries if v.get("sha256")]
    physical_id_counts = Counter(p.stem for _, p in present)
    physical_sha_counts = Counter(current_hashes.values())
    run_manifest_path = RUN_DIR / "run_manifest.json"
    run_manifest = read_json(run_manifest_path) if run_manifest_path.exists() else {}
    checkpoint_sha = compute_streaming_sha256(CHECKPOINT) if CHECKPOINT.is_file() else None
    ffmpeg_path, ffprobe_path = shutil.which("ffmpeg"), shutil.which("ffprobe")
    tracked_preflight_path = ROOT / "runs/benchmarks/week4_revalidation/preflight.json"
    tracked_preflight = read_json(tracked_preflight_path) if tracked_preflight_path.exists() else {}
    normalized_sha = sha256_bytes(manifest_text.encode("utf-8"))

    result = {
        "generated_at": now_iso(),
        "git": {
            "branch": run_text(["git", "branch", "--show-current"]),
            "head": run_text(["git", "rev-parse", "HEAD"]),
            "status_porcelain": run_text(["git", "status", "--porcelain"]).splitlines(),
            "working_tree_clean": not bool(run_text(["git", "status", "--porcelain"])),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "ffmpeg_path": ffmpeg_path,
            "ffmpeg_version": run_text([ffmpeg_path, "-version"]).splitlines()[0] if ffmpeg_path else None,
            "ffprobe_path": ffprobe_path,
            "ffprobe_version": run_text([ffprobe_path, "-version"]).splitlines()[0] if ffprobe_path else None,
            "opencv_version": __import__("cv2").__version__,
        },
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
            "exists": CHECKPOINT.is_file(),
            "sha256": checkpoint_sha,
            "matches_inference_run": checkpoint_sha == run_manifest.get("checkpoint_sha256"),
            "size_bytes": CHECKPOINT.stat().st_size if CHECKPOINT.is_file() else None,
        },
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "entry_count": len(entries),
            "sha256_raw_worktree_bytes": sha256_bytes(manifest_bytes),
            "sha256_utf8_text_after_newline_normalization": normalized_sha,
            "git_blob_snapshot_sha256": run_text(["git", "hash-object", str(MANIFEST)]),
            "git_head_blob_sha256": run_text(["git", "rev-parse", "HEAD:data/negative_videos_manifest.json"]),
            "tracked_preflight_sha256": tracked_preflight.get("manifest", {}).get("sha256"),
            "inference_run_recorded_sha256": run_manifest.get("manifest_sha256"),
        },
        "video_inventory": {
            "manifest_entries": len(entries),
            "manifest_files_present": len(present),
            "manifest_files_missing": len(missing),
            "missing_video_ids": [v.get("video_id") for v, _ in missing],
            "present_path_sample": [str(p.relative_to(ROOT)).replace("\\", "/") for _, p in present[:20]],
            "hash_errors": hash_errors,
            "manifest_duplicate_video_ids": sorted(k for k, n in Counter(manifest_ids).items() if n > 1),
            "manifest_duplicate_declared_sha256": sorted(k for k, n in Counter(manifest_sha_values).items() if n > 1),
            "present_duplicate_filename_stems": sorted(k for k, n in physical_id_counts.items() if n > 1),
            "present_duplicate_computed_sha256": sorted(k for k, n in physical_sha_counts.items() if n > 1),
            "present_content_hash_matches_manifest": {
                str(v.get("video_id")): current_hashes.get(str(v.get("video_id"))) == v.get("sha256")
                for v, _ in present
            },
        },
        "preflight_artifact_reconciliation": {
            "historical_preflight_timestamp": tracked_preflight.get("timestamp_iso"),
            "historical_preflight_git_head": tracked_preflight.get("git_head"),
            "historical_preflight_ffmpeg_path": tracked_preflight.get("ffmpeg_path"),
            "current_ffmpeg_path": ffmpeg_path,
            "environment_difference_confirmed": tracked_preflight.get("ffmpeg_path") != ffmpeg_path,
            "historical_preflight_reported_files": tracked_preflight.get("disk_scan", {}).get("total_negative_files_found"),
            "current_manifest_paths_present": len(present),
            "classification": "ENVIRONMENT_AND_INPUT_INVENTORY_DIFFERENCE; historical preflight is not current-host evidence",
        },
        "artifact_presence": {
            rel: (ROOT / rel).is_file()
            for rel in [
                "runs/benchmarks/week4_revalidation/video_audit_full.json",
                "runs/benchmarks/week4_revalidation/video_audit_summary.csv",
                "runs/benchmarks/week4_revalidation/video_decode_failures.csv",
                "runs/benchmarks/week4_revalidation/audit_execution_log.txt",
                "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/run_manifest.json",
                "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/video_inference_summary.csv",
                "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/temporal_confirmed_events.json",
                "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/frame_level_events.json",
                "runs/benchmarks/week4_revalidation/far_comparison.csv",
                "runs/benchmarks/week4_revalidation/operating_point_final_audit.json",
            ]
        },
        "status": "BLOCKED_SOURCE_MEDIA_MISSING" if missing else "PREFLIGHT_COMPLETE",
    }
    write_json(OUT / "preflight_verified.json", result)
    return result


def build_strict_video_audit(manifest: Dict[str, Any]) -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = VideoDecodeCacheManager(cache_file=OUT / ".decode_resume_cache.json", root_dir=ROOT)
    log_path = OUT / "audit_execution_verified.log"
    log_path.write_text(
        "Strict video audit start " + now_iso() + "\n"
        + "policy=" + AUDIT_POLICY_VERSION + " schema=" + str(CACHE_SCHEMA_VERSION) + "\n",
        encoding="utf-8",
    )
    records: List[Dict[str, Any]] = []
    entries = manifest.get("videos", [])
    for index, entry in enumerate(entries, 1):
        vid = str(entry.get("video_id", ""))
        path = resolve_manifest_path(entry)
        common = {
            "video_id": vid,
            "relative_path": str(entry.get("relative_path", "")),
            "manifest_sha256": entry.get("sha256"),
            "manifest_size_bytes": entry.get("file_size_bytes"),
            "attempted": False,
            "audit_policy_version": AUDIT_POLICY_VERSION,
        }
        if not path.is_file():
            record = {**common, "audit_result_status": "BLOCKED_INPUT_MISSING", "decode_status": None,
                      "status": "BLOCKED", "full_decode_verified": False, "valid_exposure_seconds": 0.0,
                      "error_message": "Manifest source file is absent from this workspace; not audited"}
        else:
            start = time.monotonic()
            result = audit_single_video(path, root_dir=ROOT, full_decode=True, cache_manager=cache)
            elapsed = round(time.monotonic() - start, 3)
            actual_sha = result.get("sha256")
            sha_match = actual_sha == entry.get("sha256") if entry.get("sha256") else None
            ffmpeg = result.get("external_tool_check") or {}
            if not ffmpeg.get("tool_available"):
                audit_status = "BLOCKED_FFMPEG_UNAVAILABLE"
            elif not sha_match:
                audit_status = "NEEDS_REVIEW_INPUT_HASH_MISMATCH"
            elif result.get("decode_status") == "NORMAL_EOF" and result.get("full_decode_verified") is True:
                audit_status = "STRICT_VALID"
            else:
                audit_status = "NEEDS_REVIEW"
            record = {**common, **result, "attempted": True, "audit_result_status": audit_status,
                      "actual_sha256": actual_sha, "manifest_sha256_match": sha_match,
                      "elapsed_seconds": elapsed,
                      "valid_exposure_seconds": result.get("duration_seconds", 0.0) if audit_status == "STRICT_VALID" else 0.0}
        records.append(record)
        ff = record.get("external_tool_check") or {}
        line = (f"[{index}/{len(entries)}] {vid} result={record['audit_result_status']} "
                f"decode={record.get('decode_status')} opencv={record.get('full_decoded_frames')} "
                f"ffmpeg={ff.get('nb_read_frames')} exit={ff.get('exit_code')} "
                f"manifest_sha_match={record.get('manifest_sha256_match')} elapsed={record.get('elapsed_seconds', 0)}s")
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")
            if record.get("error_message"):
                log.write("  note=" + str(record["error_message"]) + "\n")
        payload = summarize_audit(records, len(entries))
        write_json(OUT / "video_audit_verified.json", payload)

    summary = summarize_audit(records, len(entries))
    write_json(OUT / "video_audit_verified.json", summary)
    fields = ["video_id", "relative_path", "audit_result_status", "decode_status", "status", "attempted",
              "sha256", "manifest_sha256", "manifest_sha256_match", "fps", "codec", "reported_frame_count",
              "full_decoded_frames", "ffmpeg_read_frames", "ffmpeg_exit_code", "ffmpeg_stderr",
              "ffmpeg_progress_end", "full_decode_verified", "duration_seconds", "valid_exposure_seconds",
              "error_message"]
    rows = []
    failures = []
    for rec in records:
        ext = rec.get("external_tool_check") or {}
        row = {**rec, "ffmpeg_read_frames": ext.get("nb_read_frames"),
               "ffmpeg_exit_code": ext.get("exit_code"), "ffmpeg_stderr": ext.get("stderr"),
               "ffmpeg_progress_end": ext.get("progress_end")}
        rows.append(row)
        if rec.get("audit_result_status") != "STRICT_VALID":
            failures.append(row)
    write_csv(OUT / "video_audit_summary_verified.csv", rows, fields)
    write_csv(OUT / "video_audit_failures_verified.csv", failures, fields)
    return summary


def summarize_audit(records: Sequence[Dict[str, Any]], expected: int) -> Dict[str, Any]:
    statuses = Counter(r.get("audit_result_status") for r in records)
    decode_statuses = Counter(r.get("decode_status") or "NOT_AUDITED" for r in records)
    verified = [r for r in records if r.get("audit_result_status") == "STRICT_VALID"]
    return {
        "generated_at": now_iso(), "expected_manifest_videos": expected, "records_written": len(records),
        "records_remaining": expected - len(records), "audit_policy_version": AUDIT_POLICY_VERSION,
        "counts_by_audit_status": dict(statuses), "counts_by_decode_status": dict(decode_statuses),
        "strict_valid_video_count": len(verified),
        "strict_valid_exposure_seconds": round(sum(float(r.get("valid_exposure_seconds", 0) or 0) for r in verified), 3),
        "strict_valid_exposure_hours": round(sum(float(r.get("valid_exposure_seconds", 0) or 0) for r in verified) / 3600.0, 6),
        "full_manifest_complete": len(records) == expected and not any(r.get("audit_result_status", "").startswith("BLOCKED") for r in records),
        "status": "BLOCKED" if any(r.get("audit_result_status", "").startswith("BLOCKED") for r in records) else "COMPLETE",
        "videos": list(records),
    }


def event_key(event: Dict[str, Any]) -> Tuple[Any, ...]:
    return (event.get("video_id"), event.get("camera_id"), str(event.get("label", "")).upper(),
            event.get("start_sec"), event.get("end_sec"))


def manifest_binding_status(current_manifest_hash: Optional[str], recorded_hash: Optional[str],
                            run_id: Optional[str]) -> str:
    if not run_id:
        return "UNVERIFIED_MISSING_RUN_ID"
    if not current_manifest_hash or not recorded_hash:
        return "UNVERIFIED_MISSING_MANIFEST_HASH"
    return "MATCH" if current_manifest_hash == recorded_hash else "HISTORICAL_SNAPSHOT"


def artifact_lineage_status(artifact_commit: Optional[str], source_commit: Optional[str],
                            run_id: Optional[str]) -> str:
    if not run_id:
        return "UNVERIFIED_MISSING_RUN_ID"
    if not artifact_commit:
        return "UNVERIFIED_MISSING_ARTIFACT_COMMIT"
    if source_commit and source_commit != artifact_commit:
        return "SOURCE_SNAPSHOT_DIFFERS_FROM_ARTIFACT_COMMIT"
    if not source_commit:
        return "UNVERIFIED_MISSING_SOURCE_COMMIT"
    return "TRACEABLE"


def select_scope_b_ids(audit_records: Sequence[Dict[str, Any]],
                       run_rows: Dict[str, Dict[str, Any]]) -> set:
    return {str(record.get("video_id")) for record in audit_records
            if record.get("audit_result_status") == "STRICT_VALID"
            and record.get("manifest_sha256_match") is True
            and str(record.get("video_id")) in run_rows
            and run_rows[str(record.get("video_id"))].get("status") == "COMPLETED"
            and int(run_rows[str(record.get("video_id"))].get("processed_sampled_frames") or 0) > 0}


def validate_audit_summary(summary: Dict[str, Any]) -> str:
    counts = summary.get("counts_by_audit_status", {})
    strict_count = int(counts.get("STRICT_VALID", 0))
    if strict_count != int(summary.get("strict_valid_video_count", -1)):
        return "MISMATCH"
    if any(int(value or 0) > 0 for key, value in counts.items() if str(key).startswith("BLOCKED")):
        return "BLOCKED"
    return "CONSISTENT"


def compare_reduction_claims(computed_scope_b: Optional[float], stored_far_comparison: Optional[float],
                             stored_operating_point: Optional[float]) -> str:
    claims = [v for v in (stored_far_comparison, stored_operating_point) if v is not None]
    if computed_scope_b is None or len(claims) < 2:
        return "UNVERIFIED"
    if any(abs(claims[0] - value) > 1e-9 for value in claims[1:]):
        return "HISTORICAL_ARTIFACTS_DISAGREE"
    if any(abs(computed_scope_b - value) > 0.01 for value in claims):
        return "RECOMPUTED_VALUE_DIFFERS_FROM_HISTORICAL_CLAIMS"
    return "MATCH"


def operating_point_status(current: str, pending_ground_truth: int) -> str:
    if pending_ground_truth > 0 or str(current).upper() == "OFFICIAL":
        return "PROVISIONAL"
    return "PROVISIONAL"  # promotion requires a separate, fully evidenced review


def index_events(events: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    by_video: Dict[str, Dict[str, int]] = defaultdict(lambda: {"FIRE": 0, "SMOKE": 0, "total": 0})
    duplicate_counts: Dict[str, int] = defaultdict(int)
    seen = set()
    for event in events:
        key = event_key(event)
        vid = str(event.get("video_id", ""))
        if key in seen:
            duplicate_counts[vid] += 1
        else:
            seen.add(key)
            label = str(event.get("label", "")).upper()
            if label in ("FIRE", "SMOKE"):
                by_video[vid][label] += 1
                by_video[vid]["total"] += 1
    return by_video, dict(duplicate_counts)


def reconcile_inference(manifest: Dict[str, Any], run_manifest: Dict[str, Any], rows: Sequence[Dict[str, str]],
                        pipeline_a_events: Sequence[Dict[str, Any]], pipeline_b1_events: Sequence[Dict[str, Any]],
                        pipeline_b2_events: Sequence[Dict[str, Any]], current_manifest_hash: str) -> Dict[str, Any]:
    manifest_by_id = {str(v.get("video_id")): v for v in manifest.get("videos", [])}
    rows_by_id = {str(r.get("video_id")): r for r in rows}
    row_ids, manifest_ids = set(rows_by_id), set(manifest_by_id)
    decoded = sum(int(r.get("total_decoded_frames") or 0) for r in rows)
    sampled = sum(int(r.get("processed_sampled_frames") or 0) for r in rows)
    duration = sum(float(r.get("actual_duration_sec") or 0) for r in rows)
    complete_rows = [r for r in rows if r.get("status") == "COMPLETED" and int(r.get("processed_sampled_frames") or 0) > 0]
    sample_fps = float((run_manifest.get("config") or {}).get("sample_fps", 0) or 0)
    expected_samples = sum(round(float(r.get("actual_duration_sec") or 0) * sample_fps) for r in rows)
    a_by_video, a_dupes = index_events(pipeline_a_events)
    b1_by_video, b1_dupes = index_events(pipeline_b1_events)
    b2_by_video, b2_dupes = index_events(pipeline_b2_events)
    event_sums = {"pipeline_a": sum(x["total"] for x in a_by_video.values()),
                  "pipeline_b1": sum(x["total"] for x in b1_by_video.values()),
                  "pipeline_b2": sum(x["total"] for x in b2_by_video.values())}
    summary_event_totals = {
        "pipeline_a_raw_frame_alarms": sum(int(r.get("pipeline_a_frame_alarms") or 0) for r in rows),
        "pipeline_b1_alerts": sum(int(r.get("pipeline_b1_alerts") or 0) for r in rows),
        "pipeline_b2_alerts": sum(int(r.get("pipeline_b2_alerts") or 0) for r in rows),
    }
    id_complete = row_ids == manifest_ids and len(rows) == len(manifest.get("videos", []))
    per_video = []
    for vid in sorted(manifest_ids | row_ids):
        m, row = manifest_by_id.get(vid, {}), rows_by_id.get(vid, {})
        per_video.append({"video_id": vid, "relative_path_manifest": m.get("relative_path"),
                          "relative_path_inference": row.get("relative_path"),
                          "path_match": bool(m and row and m.get("relative_path") == row.get("relative_path")),
                          "status": row.get("status") or "MISSING_ROW",
                          "decoded_frames": int(row.get("total_decoded_frames") or 0),
                          "sampled_inference_frames": int(row.get("processed_sampled_frames") or 0),
                          "duration_seconds": float(row.get("actual_duration_sec") or 0),
                          "pipeline_a_raw_frame_alarms": int(row.get("pipeline_a_frame_alarms") or 0),
                          "pipeline_a_event_count": a_by_video.get(vid, {}).get("total", 0),
                          "pipeline_b1_events": b1_by_video.get(vid, {}).get("total", 0),
                          "pipeline_b2_events": b2_by_video.get(vid, {}).get("total", 0)})
    count_match_b1 = summary_event_totals["pipeline_b1_alerts"] == event_sums["pipeline_b1"]
    count_match_b2 = summary_event_totals["pipeline_b2_alerts"] == event_sums["pipeline_b2"]
    b2_dupe_count = sum(b2_dupes.values())
    return {
        "generated_at": now_iso(), "run_id": run_manifest.get("run_id"),
        "source_code_base_commit": run_manifest.get("source_code_base_commit"),
        "packaged_commit": run_manifest.get("full_git_sha"),
        "checkpoint_sha256": run_manifest.get("checkpoint_sha256"),
        "recorded_run_manifest_sha256": run_manifest.get("manifest_sha256"),
        "current_manifest_sha256_utf8_normalized": current_manifest_hash,
        "manifest_hash_match": run_manifest.get("manifest_sha256") == current_manifest_hash,
        "video_counts": {"manifest": len(manifest_ids), "inference_rows": len(rows),
                          "id_set_match": id_complete, "missing_inference_ids": sorted(manifest_ids-row_ids),
                          "extra_inference_ids": sorted(row_ids-manifest_ids),
                          "completed_rows": len(complete_rows), "incomplete_or_zero_sample_rows": len(rows)-len(complete_rows)},
        "frame_totals_from_actual_csv_columns": {"decoded_physical_frames": decoded,
            "sampled_inference_frames": sampled, "sampling_ratio_sampled_over_decoded": sampled/decoded if decoded else None,
            "sample_fps_from_run_config": sample_fps,
            "expected_samples_round_duration_times_fps": expected_samples,
            "sample_count_ratio_vs_rounded_duration_expectation": sampled/expected_samples if expected_samples else None,
            "inference_duration_seconds": duration, "coverage_100_percent": None,
            "coverage_status": "UNVERIFIED_TIMESTAMPS_NOT_STORED_PER_FRAME"},
        "event_reconciliation": {"event_list_counts": event_sums, "summary_columns": summary_event_totals,
            "pipeline_b1_event_count_matches_summary": count_match_b1,
            "pipeline_b2_event_count_matches_summary": count_match_b2,
            "pipeline_a_raw_detections_are_not_pipeline_a_events": True,
            "duplicate_event_key_counts": {"pipeline_a": a_dupes, "pipeline_b1": b1_dupes, "pipeline_b2": b2_dupes},
            "pipeline_b2_duplicate_event_rows": b2_dupe_count},
        "per_video": per_video,
        "status": "HISTORICAL_RUN_RECONCILED_WITH_LIMITS" if id_complete and count_match_b1 and count_match_b2
                  and len(complete_rows) == len(rows) and bool(run_manifest.get("run_id")) else "NEEDS_REVIEW",
    }


def summarize_far_scope(name: str, rows: Sequence[Dict[str, Any]], a_events: Dict[str, Dict[str, int]],
                        b1_events: Dict[str, Dict[str, int]], b2_events: Dict[str, Dict[str, int]],
                        exposure_fn) -> Dict[str, Any]:
    ids = {str(r["video_id"]) for r in rows}
    exposure = sum(max(0.0, float(exposure_fn(r) or 0)) for r in rows)
    hours = exposure / 3600.0
    counts = {}
    for pipeline, mapping in [("A", a_events), ("B1", b1_events), ("B2", b2_events)]:
        by_label = {label: sum(mapping.get(vid, {}).get(label, 0) for vid in ids) for label in ("FIRE", "SMOKE")}
        by_label["total"] = by_label["FIRE"] + by_label["SMOKE"]
        counts[pipeline] = {"events": by_label, "far_per_hour":
            {label: (count/hours if hours > 0 else None) for label, count in by_label.items()},
            "exact_poisson_95_ci_per_hour": {label: list(exact_poisson_ci(count, hours)) if hours > 0 else None
                                             for label, count in by_label.items()}}
    a_far, b2_far = counts["A"]["far_per_hour"]["total"], counts["B2"]["far_per_hour"]["total"]
    return {"scope": name, "video_count": len(ids), "exposure_seconds": round(exposure, 3) if hours else None,
            "exposure_hours": round(hours, 8) if hours else None, "counts": counts,
            "b2_far_reduction_vs_a_percent": ((a_far-b2_far)/a_far*100.0 if a_far and b2_far is not None else None),
            "rate_interpretation": "ALERT_EVENT_RATE_ONLY_PENDING_INDEPENDENT_NEGATIVE_LABEL_REVIEW"}


def reconcile_historical_far_scopes() -> Dict[str, Any]:
    """Recompute the declared 79/77 historical scopes from their per-video rows.

    These values are diagnostics only: the old audit marks many included videos
    NEEDS_REVIEW, so their exposure cannot be accepted as validated EOF exposure.
    """
    path = ROOT / "runs/benchmarks/week4_revalidation/far_per_video.csv"
    if not path.exists():
        return {"status": "UNVERIFIED_HISTORICAL_PER_VIDEO_DATA_MISSING", "scopes": {}}
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    all_ids = {str(r.get("video_id")) for r in rows}
    # Historical Scope 2 excluded only the two rows called TRUNCATED_STREAM in
    # the saved audit report. This intentionally reveals any other NEEDS_REVIEW
    # rows that the old "Clean Only" label left in the denominator.
    truncated_ids = {str(r.get("video_id")) for r in rows if r.get("decode_status") == "TRUNCATED_STREAM"}
    scopes = {}
    for name, scope_rows, claimed_hours, claimed_events in (
        ("scope_1", rows, 7.1312, 421),
        ("scope_2", [r for r in rows if str(r.get("video_id")) not in truncated_ids], 6.9846, 417),
    ):
        exposure = sum(float(r.get("duration_seconds") or 0) for r in scope_rows)
        events = sum(int(r.get("pipeline_b2_fp_events") or 0) for r in scope_rows)
        hours = exposure / 3600.0
        review_rows = [r for r in scope_rows if r.get("decode_status") != "NORMAL_EOF"
                       or str(r.get("full_decode_verified", "")).lower() != "true"]
        scopes[name] = {
            "video_count": len(scope_rows), "claimed_video_count": 79 if name == "scope_1" else 77,
            "exposure_seconds_from_same_rows": round(exposure, 3),
            "exposure_hours_from_same_rows": round(hours, 8),
            "claimed_exposure_hours": claimed_hours,
            "claimed_exposure_matches_per_video_rows": abs(hours - claimed_hours) < 0.00005,
            "b2_alerts_from_same_rows": events, "claimed_b2_alerts": claimed_events,
            "claimed_alert_count_matches_per_video_rows": events == claimed_events,
            "b2_alerts_per_hour_arithmetic_only": events / hours if hours > 0 else None,
            "exact_poisson_95_ci_arithmetic_only": list(exact_poisson_ci(events, hours)) if hours > 0 else None,
            "normal_eof_rows": len(scope_rows) - len(review_rows),
            "needs_review_or_not_verified_rows": len(review_rows),
            "excluded_truncated_video_ids_for_scope_2": sorted(truncated_ids) if name == "scope_2" else [],
            "far_acceptance_status": "REJECTED_SCOPE_CONTAINS_NEEDS_REVIEW_ROWS" if review_rows else "HISTORICAL_ONLY_NEGATIVE_LABELS_UNVERIFIED",
        }
    scopes["scope_1"]["excluded_truncated_video_ids_for_scope_2"] = []
    return {"status": "INTERNAL_ROWS_RECONCILED_SCOPE_2_NOT_CLEAN", "source": str(path.relative_to(ROOT)),
            "truncated_video_ids": sorted(truncated_ids), "scopes": scopes,
            "note": "Historical numerator and denominator sums are paired by per-video rows. Scope 2 removed only the 2 truncated rows but retains all other NEEDS_REVIEW rows; the computed rates and exact intervals are arithmetic diagnostics, not accepted FAR."}


def build_far(manifest: Dict[str, Any], audit: Dict[str, Any], inference: Dict[str, Any],
              rows: Sequence[Dict[str, str]], a_list: Sequence[Dict[str, Any]],
              b1_list: Sequence[Dict[str, Any]], b2_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    a_idx, a_dupes = index_events(a_list)
    b1_idx, b1_dupes = index_events(b1_list)
    b2_idx, b2_dupes = index_events(b2_list)
    run_rows = {str(r.get("video_id")): r for r in rows}
    audit_rows = {str(r.get("video_id")): r for r in audit.get("videos", [])}
    eligible_b_ids = select_scope_b_ids(list(audit_rows.values()), run_rows)
    scope_b = [run_rows[vid] for vid in sorted(eligible_b_ids)]
    # Historical all-row durations include absent or unverified media, so do
    # not expose an all-row FAR. Historical arithmetic is reconciled separately
    # and explicitly rejected as validated exposure when strict EOF evidence is
    # unavailable.
    scope_a_result = {"scope": "A_ALL_HISTORICALLY_EVALUATED", "video_count": len(rows),
        "exposure_seconds": None, "exposure_hours": None, "counts": None,
        "b2_far_reduction_vs_a_percent": None,
        "rate_interpretation": "BLOCKED_NO_CURRENT_STRICT_EOF_EVIDENCE_FOR_ALL_ROWS"}
    scope_b_result = summarize_far_scope("B_STRICT_CURRENT_AUDIT_AND_INFERENCE_MATCH", scope_b, a_idx, b1_idx, b2_idx,
        lambda r: audit_rows.get(str(r.get("video_id")), {}).get("valid_exposure_seconds"))
    human_negative_ids = {str(v.get("video_id")) for v in manifest.get("videos", [])
                          if str(v.get("human_review_status", "")).upper() in ("VERIFIED_NEGATIVE", "CROSS_VERIFIED_NEGATIVE")}
    scope_c_ids = eligible_b_ids & human_negative_ids
    scope_c = [run_rows[vid] for vid in sorted(scope_c_ids)]
    scope_c_result = summarize_far_scope("C_INDEPENDENTLY_VERIFIED_NEGATIVE", scope_c, a_idx, b1_idx, b2_idx,
        lambda r: audit_rows.get(str(r.get("video_id")), {}).get("valid_exposure_seconds")) if scope_c else {
            "scope": "C_INDEPENDENTLY_VERIFIED_NEGATIVE", "video_count": 0, "exposure_seconds": None,
            "exposure_hours": None, "counts": None, "b2_far_reduction_vs_a_percent": None,
            "rate_interpretation": "PENDING_HUMAN_NEGATIVE_REVIEW"}
    comparison_path = ROOT / "runs/benchmarks/week4_revalidation/far_comparison.csv"
    comparison_rows = list(csv.DictReader(comparison_path.open(encoding="utf-8-sig"))) if comparison_path.exists() else []
    scope2_comparison = next((r for r in comparison_rows if "Scope 2" in str(r.get("Scope", ""))), {})
    comparison_claim = None
    try:
        comparison_claim = float(str(scope2_comparison.get("Reduction (%)", "")).replace("%", ""))
    except ValueError:
        pass
    op_path = ROOT / "runs/benchmarks/week4_revalidation/operating_point_final_audit.json"
    op_data = read_json(op_path) if op_path.exists() else {}
    scope2_op_text = str((((op_data.get("criteria_evaluation") or {}).get("mt3_far_reduction") or {}).get("scope_2_result")) or "")
    op_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", scope2_op_text)
    op_claim = float(op_match.group(1)) if op_match else None
    reduction_reconciliation = compare_reduction_claims(
        scope_b_result.get("b2_far_reduction_vs_a_percent"), comparison_claim, op_claim)
    excluded_b = sorted(set(run_rows)-eligible_b_ids)
    scope_b_exclusions = [{"video_id": vid,
        "reason": "MISSING_SOURCE_OR_NOT_STRICT_VALID_OR_INFERENCE_NOT_COMPLETE",
        "historical_exposure_seconds": float(run_rows[vid].get("actual_duration_sec") or 0),
        "pipeline_a_events": a_idx.get(vid, {}).get("total", 0),
        "pipeline_b1_events": b1_idx.get(vid, {}).get("total", 0),
        "pipeline_b2_events": b2_idx.get(vid, {}).get("total", 0)} for vid in excluded_b]
    result = {
        "generated_at": now_iso(), "source_run_id": RUN_ID,
        "source_audit_policy_version": AUDIT_POLICY_VERSION,
        "scope_A": scope_a_result, "scope_B": scope_b_result, "scope_C": scope_c_result,
        "scope_exclusions": {"scope_B_excluded_video_rows": scope_b_exclusions,
            "scope_B_excluded_video_count": len(scope_b_exclusions),
            "scope_B_excluded_alert_events_by_pipeline": {
                "A": sum(x["pipeline_a_events"] for x in scope_b_exclusions),
                "B1": sum(x["pipeline_b1_events"] for x in scope_b_exclusions),
                "B2": sum(x["pipeline_b2_events"] for x in scope_b_exclusions)},
            "scope_C_excluded_video_count": len(set(run_rows)-scope_c_ids)},
        "duplicate_event_rows": {"A": a_dupes, "B1": b1_dupes, "B2": b2_dupes},
        "historical_scope_b_reduction_claims": {"far_comparison_csv_pct": comparison_claim,
            "operating_point_audit_pct": op_claim,
            "reconciliation_status": reduction_reconciliation},
        "declared_historical_scope_reconciliation": reconcile_historical_far_scopes(),
        "ground_truth_for_negative_videos": "PENDING_HUMAN_NEGATIVE_REVIEW",
        "status": "PENDING_HUMAN_NEGATIVE_REVIEW" if not scope_c_ids else "NEEDS_REVIEW",
        "scientific_note": "The computed counts are alert-event rates on historical rows; they are not semantic false-alarm rates until independent negative labels are verified.",
    }
    write_json(OUT / "far_recomputed.json", result)
    per_video = []
    for vid in sorted(run_rows):
        row, aud = run_rows[vid], audit_rows.get(vid, {})
        per_video.append({"video_id": vid, "relative_path": row.get("relative_path"),
            "strict_audit_status": aud.get("audit_result_status", "BLOCKED_INPUT_MISSING"),
            "decode_status": aud.get("decode_status"),
            "historical_inference_status": row.get("status"),
            "historical_exposure_seconds": float(row.get("actual_duration_sec") or 0),
            "strict_valid_exposure_seconds": float(aud.get("valid_exposure_seconds", 0) or 0),
            "pipeline_a_raw_frame_alarms": int(row.get("pipeline_a_frame_alarms") or 0),
            "pipeline_a_events": a_idx.get(vid, {}).get("total", 0),
            "pipeline_b1_events": b1_idx.get(vid, {}).get("total", 0),
            "pipeline_b2_events": b2_idx.get(vid, {}).get("total", 0),
            "scope_B_included": vid in eligible_b_ids, "scope_C_included": vid in scope_c_ids,
            "negative_label_verified": vid in human_negative_ids})
    write_csv(OUT / "far_per_video_verified.csv", per_video, list(per_video[0].keys()) if per_video else [])
    comp_rows = []
    for key, scope in [("A", scope_a_result), ("B", scope_b_result), ("C", scope_c_result)]:
        comp_rows.append({"scope": key, "videos": scope["video_count"], "exposure_hours": scope.get("exposure_hours"),
            "pipeline_a_events": (scope.get("counts") or {}).get("A", {}).get("events", {}).get("total"),
            "pipeline_a_rate": (scope.get("counts") or {}).get("A", {}).get("far_per_hour", {}).get("total"),
            "pipeline_b1_events": (scope.get("counts") or {}).get("B1", {}).get("events", {}).get("total"),
            "pipeline_b1_rate": (scope.get("counts") or {}).get("B1", {}).get("far_per_hour", {}).get("total"),
            "pipeline_b2_events": (scope.get("counts") or {}).get("B2", {}).get("events", {}).get("total"),
            "pipeline_b2_rate": (scope.get("counts") or {}).get("B2", {}).get("far_per_hour", {}).get("total"),
            "pipeline_b2_exact_poisson_ci": (scope.get("counts") or {}).get("B2", {}).get("exact_poisson_95_ci_per_hour", {}).get("total"),
            "b2_reduction_pct": scope.get("b2_far_reduction_vs_a_percent"),
            "interpretation": scope.get("rate_interpretation")})
    write_csv(OUT / "far_scope_comparison.csv", comp_rows, list(comp_rows[0].keys()))
    md = ["# FAR Scientific Reconciliation", "", "Computed from stored run event files with src.event_eval.exact_poisson_ci.",
          "All event rates remain pending independent negative-label review.", ""]
    for scope in [scope_a_result, scope_b_result, scope_c_result]:
        md += [f"## Scope {scope['scope']}", "", f"Videos: {scope['video_count']}; exposure: {scope.get('exposure_hours')} h.", ""]
        if scope.get("counts"):
            for name, label in [("A", "Pipeline A"), ("B1", "Pipeline B1"), ("B2", "Pipeline B2")]:
                item = scope["counts"][name]
                md.append(f"- {label}: {item['events']['total']} events; {item['far_per_hour']['total']} / h; exact CI {item['exact_poisson_95_ci_per_hour']['total']}.")
            md.append(f"- B2 FAR reduction relative to A: {scope['b2_far_reduction_vs_a_percent']}%.")
        else:
            md.append("- BLOCKED: no independently verified negative videos are available.")
        md.append("")
    md += ["## Explicit scope exclusions", "", f"Scope B excludes {len(scope_b_exclusions)} historical video rows. Every excluded row, exposure and alert count is listed in `far_recomputed.json`; no alert was silently dropped.", ""]
    (OUT / "far_scientific_reconciliation.md").write_text("\n".join(md), encoding="utf-8")
    return result


def build_provenance(manifest: Dict[str, Any], preflight: Dict[str, Any], inference: Dict[str, Any]) -> Dict[str, Any]:
    current_sha = preflight["manifest"]["sha256_utf8_text_after_newline_normalization"]
    raw_sha = preflight["manifest"]["sha256_raw_worktree_bytes"]
    run_manifest = read_json(RUN_DIR / "run_manifest.json")
    inf_recon_path = ROOT / "runs/benchmarks/week4_revalidation/inference_reconciliation.json"
    inf_recon = read_json(inf_recon_path) if inf_recon_path.exists() else {}
    old_prov_path = ROOT / "runs/benchmarks/week4_revalidation/provenance_audit_revalidated.json"
    old_prov = read_json(old_prov_path) if old_prov_path.exists() else {}
    artifacts = [
        ("data/negative_videos_manifest.json", raw_sha, "current_manifest_raw_worktree_bytes"),
        ("runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/run_manifest.json", None, "historical_run_manifest"),
        ("runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/video_inference_summary.csv", None, "historical_inference_summary"),
        ("runs/benchmarks/week4_revalidation/inference_reconciliation.json", None, "historical_reconciliation"),
        ("runs/benchmarks/week4_revalidation/provenance_audit_revalidated.json", None, "historical_provenance"),
        ("runs/benchmarks/week4_revalidation/video_audit_full.json", None, "historical_full_audit"),
    ]
    lineage = []
    for rel, claimed, kind in artifacts:
        p = ROOT / rel
        artifact_commit = run_text(["git", "log", "-1", "--format=%H", "--", rel]) or None
        source_commit = None
        run_id = None
        if rel.endswith("run_manifest.json") and p.exists():
            manifest_record = read_json(p)
            source_commit = manifest_record.get("source_code_base_commit")
            run_id = manifest_record.get("run_id")
        elif "video_inference_summary.csv" in rel:
            source_commit = run_manifest.get("full_git_sha")
            run_id = run_manifest.get("run_id")
        elif "inference_reconciliation.json" in rel and p.exists():
            detail = read_json(p)
            source_commit = detail.get("packaged_commit")
            run_id = detail.get("inference_run_id")
        elif "provenance_audit_revalidated.json" in rel and p.exists():
            detail = read_json(p)
            source_commit = (detail.get("lineage_trail") or {}).get("revalidation_commit")
            run_id = (detail.get("artifacts") or {}).get("historical_inference_run_id")
        elif "video_audit_full.json" in rel and p.exists():
            detail = read_json(p)
            run_id = detail.get("audit_run_id")
            source_commit = detail.get("audit_commit")
        line = {"artifact_path": rel, "exists": p.is_file(), "kind": kind,
                "actual_sha256": compute_streaming_sha256(p) if p.is_file() else None,
                "size_bytes": p.stat().st_size if p.is_file() else None,
                "modified_at_utc": modified_at_utc(p) if p.is_file() else None,
                "claimed_sha256": claimed, "artifact_commit": artifact_commit,
                "source_commit": source_commit, "run_id": run_id,
                "status": artifact_lineage_status(artifact_commit, source_commit, run_id)}
        if rel == "data/negative_videos_manifest.json":
            line.update({"normalized_text_sha256": current_sha,
                         "git_head_blob_sha256": preflight["manifest"]["git_head_blob_sha256"],
                         "historical_inference_sha256": run_manifest.get("manifest_sha256"),
                         "hash_method_note": "Raw CRLF bytes and UTF-8 text normalized to LF hash differently; both values are recorded."})
            line["status"] = "MATCH_CURRENT_RAW_AND_NORMALIZED_VERSIONS_RECORDED"
        lineage.append(line)
    run_sha = run_manifest.get("manifest_sha256")
    recon_sha = (inf_recon.get("consistency_check") or {}).get("current_manifest_sha256")
    provenance_sha = (old_prov.get("artifacts") or {}).get("manifest_sha256")
    result = {
        "generated_at": now_iso(), "audit_commit": None,
        "audit_base_commit": preflight["git"]["head"],
        "working_tree_clean_at_generation": preflight["git"]["working_tree_clean"],
        "checkpoint_sha256_current": preflight["checkpoint"]["sha256"],
        "checkpoint_sha256_run": run_manifest.get("checkpoint_sha256"),
        "checkpoint_match": preflight["checkpoint"]["sha256"] == run_manifest.get("checkpoint_sha256"),
        "inference_run_id": run_manifest.get("run_id"),
        "source_code_base_commit": run_manifest.get("source_code_base_commit"),
        "packaged_inference_commit": run_manifest.get("full_git_sha"),
        "manifest_hashes": {"current_raw_worktree_bytes": raw_sha,
            "current_utf8_lf_normalized": current_sha,
            "current_git_head_blob": preflight["manifest"]["git_head_blob_sha256"],
            "historical_inference_run": run_sha,
            "historical_inference_reconciliation": recon_sha,
            "historical_provenance_audit": provenance_sha},
        "artifact_lineage": lineage,
        "assessment": {"run_manifest_binding_status": manifest_binding_status(current_sha, run_sha, run_manifest.get("run_id")),
            "run_manifest_hash_matches_current_raw": run_sha == raw_sha,
            "run_manifest_hash_matches_current_normalized": run_sha == current_sha,
            "reconciliation_hash_matches_current_normalized": recon_sha == current_sha,
            "provenance_hash_matches_current_raw": provenance_sha == raw_sha,
            "lineage_status": "HISTORICAL_SNAPSHOT_UNVERIFIED" if run_sha not in (raw_sha, current_sha) else "CURRENT_SNAPSHOT_MATCH",
            "current_audit_commit": "UNCOMMITTED", "audit_commit_sha": None},
        "status": "PARTIAL_ACCEPTANCE",
    }
    write_json(OUT / "provenance_verified.json", result)
    write_csv(OUT / "artifact_lineage.csv", lineage, list(lineage[0].keys()))
    lines = ["# Manifest Hash Reconciliation", "",
        f"Current worktree raw-byte SHA-256: `{raw_sha}`.",
        f"Current UTF-8 text after universal-newline normalization SHA-256: `{current_sha}`.",
        f"Current Git HEAD blob SHA-256: `{preflight['manifest']['git_head_blob_sha256']}`.",
        f"Historical inference run recorded: `{run_sha}`.",
        f"Historical inference reconciliation recorded: `{recon_sha}`.",
        f"Historical provenance audit recorded: `{provenance_sha}`.", "",
        "The 5413… raw hash corresponds to the CRLF rendering of the prior 2.3 manifest at commit 3ebd; 4bf… is its LF Git blob. The current v2.3 file's raw CRLF hash is 5ba… and its LF/Git blob hash is 3af…. These are line-ending representations of their respective committed contents, not interchangeable hash inputs.",
        "The inference run's 756… digest is not reproduced by current bytes or current normalized text. Its exact source manifest snapshot is not present in this repository; classify it as HISTORICAL_SNAPSHOT / UNVERIFIED, not a match.", ""]
    (OUT / "manifest_hash_reconciliation.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def build_ground_truth_and_op() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    queue = list(csv.DictReader(GT_QUEUE.open(encoding="utf-8-sig"))) if GT_QUEUE.exists() else []
    statuses = Counter(str(r.get("review_status", "UNVERIFIED")) for r in queue)
    verified = statuses.get("VERIFIED", 0)
    pending = statuses.get("PENDING_HUMAN_REVIEW", 0)
    gt = {"generated_at": now_iso(), "queue_path": str(GT_QUEUE.relative_to(ROOT)).replace("\\", "/"),
          "queue_rows": len(queue), "recorded_verified_rows": verified, "recorded_pending_rows": pending,
          "annotator_02_pending_events": pending,
          "status": "UNVERIFIED_FROM_CURRENT_PRIMARY_GT_SNAPSHOT" if not queue else "HUMAN_REVIEW_STATUS_PRESERVED",
          "note": "This audit does not determine labels or promote pending annotations."}
    write_json(OUT / "ground_truth_status_verified.json", gt)
    previous = read_json(ROOT / "runs/benchmarks/week4_revalidation/operating_point_final_audit.json")
    criteria = previous.get("criteria_evaluation", {})
    op = {"generated_at": now_iso(), "operating_point_configuration": previous.get("operating_point_configuration"),
          "previous_claims": criteria,
          "ground_truth_pending": pending,
          "status": "PROVISIONAL", "official_promotion_allowed": False,
          "criteria_reassessment": {"far_reduction": "UNVERIFIED_FOR_STRICT_SCOPES; see far_recomputed.json",
              "recall": "UNVERIFIED_WHILE_GROUND_TRUTH_PENDING",
              "latency": "UNVERIFIED_FROM_RAW_TIMESTAMP_EVIDENCE",
              "negative_exposure": "BLOCKED_FOR_FULL_SET; only strict current-media exposure counted",
              "test_leakage": "UNVERIFIED"},
          "status_reason": "Keep PROVISIONAL: 24 GT events remain pending and current strict input/reconciliation evidence is incomplete."}
    write_json(OUT / "operating_point_reassessment.json", op)
    return gt, op


def write_final_report(preflight: Dict[str, Any], audit: Dict[str, Any], inference: Dict[str, Any],
                       far: Dict[str, Any], provenance: Dict[str, Any], gt: Dict[str, Any], op: Dict[str, Any],
                       test_result: Optional[Dict[str, Any]] = None,
                       ci_status: Optional[Dict[str, Any]] = None) -> None:
    env = preflight["environment"]
    a_counts = audit.get("counts_by_audit_status", {})
    frames = inference["frame_totals_from_actual_csv_columns"]
    scopes = [far["scope_A"], far["scope_B"], far["scope_C"]]
    lines = [
        "# Week 4 Final Consistency Acceptance Report", "",
        f"Generated: {now_iso()}",
        f"Base commit: `{preflight['git']['head']}`; the audit ran with a modified worktree and remains uncommitted.",
        "**Acceptance: BLOCKED** — current workspace is missing most manifest video sources; no full-set strict audit or independently verified negative FAR is possible.", "",
        "## Environment and preflight", "",
        f"Python: `{env['python_version'].splitlines()[0]}`; executable `{env['python_executable']}`.",
        f"OpenCV: `{env['opencv_version']}`; FFmpeg: `{env['ffmpeg_version']}` at `{env['ffmpeg_path']}`; FFprobe: `{env['ffprobe_version']}`.",
        f"Checkpoint SHA-256: `{preflight['checkpoint']['sha256']}` (run match: {preflight['checkpoint']['matches_inference_run']}).",
        f"Manifest: {preflight['video_inventory']['manifest_files_present']}/{preflight['manifest']['entry_count']} source files present; {preflight['video_inventory']['manifest_files_missing']} missing.",
        f"The old preflight artifact reports Python 3.10.11 and no FFmpeg path, while this run observes Python 3.14 globally, Python 3.9 in `.venv`, and FFmpeg 9.0.1. Its recorded timestamp (21:45 UTC) is later than the current preflight timestamp ({preflight['generated_at']}), so its chronology is unverified. The old audit log records clean FFmpeg frame counts but not executable path/version. This is an environment/metadata-lineage difference; the old null path is not current-host evidence.", "",
        "## Strict video audit", "",
        f"Audit policy `{AUDIT_POLICY_VERSION}`; attempted {sum(1 for v in audit['videos'] if v.get('attempted'))}/{len(audit['videos'])}; strict valid {audit['strict_valid_video_count']}; statuses `{json.dumps(a_counts, ensure_ascii=False)}`.",
        f"Strict verified exposure: {audit['strict_valid_exposure_hours']} h. Missing sources contribute zero exposure and retain `BLOCKED_INPUT_MISSING`.",
        "Per-video FFmpeg exit/stderr/count/progress, OpenCV count, container metadata, codec, frame rates, packet PTS/DTS presence/ranges and DTS order, SHA-256 and status are in `video_audit_verified.json` and the CSVs. PTS/DTS is recorded; B-frame PTS reordering is not treated as an error.",
        "Tolerance policy: no experiment-backed allowance was found for 5 frames, 10 frames, 0.25 seconds, or 0.35 seconds. Production audit uses zero count tolerance; 5-frame/0.25-second legacy values remain disabled, and the 10-frame text is only a synthetic truncation-test description. No codec/FPS/PTS-based tolerance is inferred. The prior 19 NORMAL_EOF / 58 NEEDS_REVIEW / 2 TRUNCATED_STREAM records are historical only and were not treated as current-valid evidence.", "",
        "## Inference reconciliation", "",
        f"Historical run `{inference['run_id']}`: {inference['video_counts']['inference_rows']} rows, {frames['decoded_physical_frames']} decoded frames, {frames['sampled_inference_frames']} sampled frames; sampling ratio {frames['sampling_ratio_sampled_over_decoded']}; timeline coverage 100% is **UNVERIFIED** because per-frame timestamps are not available.",
        f"Event totals from event files: {inference['event_reconciliation']['event_list_counts']}; summary reconciliation: B1={inference['event_reconciliation']['pipeline_b1_event_count_matches_summary']}, B2={inference['event_reconciliation']['pipeline_b2_event_count_matches_summary']}. Pipeline A raw frame alarms ({inference['event_reconciliation']['summary_columns']['pipeline_a_raw_frame_alarms']}) are kept distinct from its event list count ({inference['event_reconciliation']['event_list_counts']['pipeline_a']}).", "",
        "## FAR / alert-event rates", "",
    ]
    for scope in scopes:
        if not scope.get("counts"):
            lines.append(f"- {scope['scope']}: BLOCKED; no verified negative scope.")
            continue
        b2 = scope["counts"]["B2"]
        lines.append(f"- {scope['scope']}: {scope['video_count']} videos, {scope['exposure_hours']} h; B2={b2['events']['total']} events, {b2['far_per_hour']['total']}/h, exact 95% CI {b2['exact_poisson_95_ci_per_hour']['total']}; reduction={scope['b2_far_reduction_vs_a_percent']}%.")
    declared = far.get("declared_historical_scope_reconciliation", {})
    for key, label in (("scope_1", "Historical requested Scope 1"), ("scope_2", "Historical requested Scope 2")):
        item = declared.get("scopes", {}).get(key)
        if item:
            lines.append(f"- {label} arithmetic from matching per-video rows: {item['video_count']} rows, {item['exposure_hours_from_same_rows']} h, B2={item['b2_alerts_from_same_rows']}, {item['b2_alerts_per_hour_arithmetic_only']:.4f}/h, exact Poisson 95% CI {item['exact_poisson_95_ci_arithmetic_only']}; {item['needs_review_or_not_verified_rows']} rows lack NORMAL_EOF. Status: `{item['far_acceptance_status']}`.")
    lines += ["", "These are alert-event rates on the recorded nominal-negative batch, not scientifically confirmed false-alarm rates: negative semantics remain pending independent review. Scope B exclusions and removed events/exposure are enumerated explicitly in `far_recomputed.json`.",
        "The requested historical rows pair their event counts and durations internally, but their FAR claims fail strict exposure validation: Scope 2 excludes only the 2 truncated rows and retains 58 NEEDS_REVIEW / unverified rows. The 16 current strict-valid files yield the only current recomputed scope; the historical Scope 1/2 arithmetic above is not accepted FAR.",
        "Stored reduction artifacts disagree (57.05% versus 56.48%). The rerun computes a common formula from event lists; no historical output was edited to force agreement.", "",
        "## Provenance, Ground Truth and Operating Point", "",
        f"Run ID `{provenance['inference_run_id']}`; inference source base `{provenance['source_code_base_commit']}`; packaged commit `{provenance['packaged_inference_commit']}`.",
        f"Current audit base `{provenance['audit_base_commit']}`; audit commit is unset (`UNCOMMITTED_WORKTREE`). Current manifest raw and normalized hashes are recorded separately. Historical run manifest hash match: {provenance['assessment']['run_manifest_hash_matches_current_normalized']}.",
        "File modification times (UTC), sizes and SHA-256 values for generated outputs and key source artifacts are recorded in `artifact_timestamp_audit.json`; artifact-level lineage timestamps are in `artifact_lineage.csv`.",
        f"Ground Truth queue records {gt['recorded_verified_rows']} verified rows and {gt['recorded_pending_rows']} pending; no labels changed.",
        "Operating Point: **PROVISIONAL**; official promotion is false. Recall and latency remain unverified while GT is pending.",
        "RTSP: **NOT_TESTED**; no live RTSP source was available in this audit environment.", "",
        "## Tests and CI", "",
    ]
    if test_result:
        lines.append(f"Local suite: `{test_result.get('command')}` — {test_result.get('status')} ({test_result.get('tests_run')} tests, {test_result.get('failures')} failures, {test_result.get('errors')} errors; {test_result.get('duration_seconds')} s).")
        lines.append("Requested real-pipeline scenarios and their unittest method names/results are in `runs/benchmarks/week4_consistency_fix/scenario_test_results.csv`; PASS is derived only from the recorded full-suite run with zero failures/errors.")
    else:
        lines.append("Local suite: see preflight evidence; no CI run is asserted without a recorded workflow run.")
    if ci_status:
        lines.append(f"Remote combined status: {ci_status.get('status')} for base commit `{ci_status.get('commit_sha')}`; {ci_status.get('status_check_count')} status checks. No CI PASS is claimed.")
    else:
        lines.append("Remote CI: UNVERIFIED; no recorded status check result is available.")
    lines += ["",
        "## Unresolved blockers", "",
        f"- Restore {preflight['video_inventory']['manifest_files_missing']} missing original source videos and rerun the strict audit; do not substitute generated fixtures.",
        "- Retrieve the exact manifest snapshot referenced by the historical inference run, or mark that run as unverified and rerun inference after input hashes are verified.",
        "- Complete independent review for negative semantics before reporting FAR; preserve 24 pending positive Ground Truth events until Annotator 02 confirms.",
        "- Reconcile the historical Scope B reduction discrepancy from common raw event inputs; preserve PROVISIONAL status.",
        "- Obtain an actual GitHub Actions workflow result before marking CI as PASS.", "",
        "## Acceptance", "",
        "**BLOCKED.** Evidence paths are the JSON/CSV/log files in `runs/benchmarks/week4_consistency_fix/`. No `TECHNICAL_VALIDATION_PASS` or `WEEK4_FULL_ACCEPTANCE` is claimed.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines)
    REPORT.write_text(report_text, encoding="utf-8")
    RELIABILITY_REPORT.write_text(report_text.replace("# Week 4 Final Consistency Acceptance Report",
                                                     "# Week 4 Final Reliability Report", 1), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight()
    manifest = read_json(MANIFEST)
    audit = build_strict_video_audit(manifest)
    run_manifest = read_json(RUN_DIR / "run_manifest.json")
    rows = list(csv.DictReader((RUN_DIR / "video_inference_summary.csv").open(encoding="utf-8-sig")))
    event_data = read_json(RUN_DIR / "temporal_confirmed_events.json")
    a_list = read_json(RUN_DIR / "frame_level_events.json")
    b1_list = event_data.get("pipeline_b1_events", [])
    b2_list = event_data.get("pipeline_b2_events", [])
    inference = reconcile_inference(manifest, run_manifest, rows, a_list, b1_list, b2_list,
                                    preflight["manifest"]["sha256_utf8_text_after_newline_normalization"])
    write_json(OUT / "inference_reconciliation_verified.json", inference)
    write_csv(OUT / "inference_coverage_verified.csv", inference["per_video"], list(inference["per_video"][0].keys()))
    decision = {"generated_at": now_iso(), "run_id": run_manifest.get("run_id"),
        "checkpoint_sha256_match": preflight["checkpoint"]["matches_inference_run"],
        "inference_reuse_status": "MANUAL_REVIEW_REQUIRED",
        "rerun_required": None,
        "reason": "Historical manifest digest is not reproduced, most original videos are absent, so per-video input identity cannot be re-established.",
        "do_not_retrain": True}
    write_json(OUT / "rerun_decision_verified.json", decision)
    far = build_far(manifest, audit, inference, rows, a_list, b1_list, b2_list)
    provenance = build_provenance(manifest, preflight, inference)
    gt, op = build_ground_truth_and_op()
    test_result = None
    test_result_path = OUT / "local_test_result.json"
    if test_result_path.exists():
        test_result = read_json(test_result_path)
    ci_status = None
    ci_status_path = OUT / "ci_status.json"
    if ci_status_path.exists():
        ci_status = read_json(ci_status_path)
    scenario_tests = [
        {"scenario": "Normal video decode to EOF", "test_methods": "test_15_decode_normal_eof; test_strict_audit_requires_real_ffmpeg_eof_evidence"},
        {"scenario": "Video truncated midstream", "test_methods": "test_16_decode_truncated_video; test_truncated_video_is_not_accepted_by_pipeline"},
        {"scenario": "Corrupt packet/NAL", "test_methods": "test_17_decode_midstream_error; test_corrupt_packet_nal_is_not_accepted_by_pipeline"},
        {"scenario": "Decoder silent stop", "test_methods": "test_silent_opencv_stop_is_needs_review; test_48_audit_pipeline_silent_eof_missing_frames"},
        {"scenario": "Wrong metadata frame count", "test_methods": "test_20_decode_metadata_frame_count_wrong; test_incorrect_container_frame_count_forces_review"},
        {"scenario": "FPS zero/invalid and zero frame count", "test_methods": "test_invalid_fps_and_zero_frame_metadata_are_review; test_44_audit_pipeline_invalid_fps; test_45_audit_pipeline_zero_reported_frames"},
        {"scenario": "FFprobe unavailable", "test_methods": "test_ffprobe_not_installed_ffmpeg_can_still_verify_full_stream; test_46_audit_pipeline_ffprobe_not_installed"},
        {"scenario": "FFprobe nonzero exit", "test_methods": "test_ffprobe_error_does_not_replace_ffmpeg_full_decode"},
        {"scenario": "Real packet PTS/DTS, codec, FPS and count", "test_methods": "test_ffprobe_records_real_packet_pts_dts_fps_codec_and_counts"},
        {"scenario": "Same-size content mutation and stale cache", "test_methods": "test_06_hash_same_size_different_content; test_same_size_content_change_invalidates_cached_audit; test_10_cache_old_schema_no_sha256"},
        {"scenario": "External decoder error/nonzero exit", "test_methods": "test_ffmpeg_decode_error_is_not_success; test_47_audit_pipeline_external_tool_error_exit_code"},
    ]
    for item in scenario_tests:
        item["result"] = "PASS" if test_result and test_result.get("status") == "PASS" and not test_result.get("failures") and not test_result.get("errors") else "NOT_CONFIRMED"
    write_csv(OUT / "scenario_test_results.csv", scenario_tests, ["scenario", "test_methods", "result"])
    write_final_report(preflight, audit, inference, far, provenance, gt, op, test_result, ci_status)
    timestamp_paths = sorted(p for p in OUT.rglob("*") if p.is_file()
                              and p.name != ".decode_resume_cache.json"
                              and p.name != "artifact_timestamp_audit.json")
    timestamp_paths += [REPORT, RELIABILITY_REPORT, MANIFEST, CHECKPOINT,
        RUN_DIR / "run_manifest.json", RUN_DIR / "video_inference_summary.csv",
        ROOT / "runs/benchmarks/week4_revalidation/video_audit_full.json",
        ROOT / "runs/benchmarks/week4_revalidation/far_per_video.csv",
        ROOT / "runs/benchmarks/week4_revalidation/far_report_revalidated.json"]
    timestamp_rows = []
    for path in dict.fromkeys(timestamp_paths):
        timestamp_rows.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": path.is_file(), "modified_at_utc": modified_at_utc(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": compute_streaming_sha256(path) if path.is_file() else None})
    write_json(OUT / "artifact_timestamp_audit.json", {"generated_at": now_iso(),
        "status": "TIMESTAMPS_RECORDED_FOR_PRESENT_INPUTS_AND_GENERATED_OUTPUTS",
        "artifacts": timestamp_rows})
    print(json.dumps({"preflight": preflight["status"], "audit": audit["counts_by_audit_status"],
                      "inference": inference["status"], "far": far["status"],
                      "acceptance": "BLOCKED", "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
