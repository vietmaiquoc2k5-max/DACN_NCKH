"""Recalculate Week 4 evidence without modifying historical artifacts or source GT.

Usage: python scripts/run_phases4_to_8_recovery.py [--output-dir PATH]
No model inference or annotation verification is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.event_eval import exact_poisson_ci
from src.week4_acceptance import acceptance, preserve_review_csv, reuse_decision, strict_technical_eligible
from src.ground_truth_integrity import validate_frozen_snapshot

HISTORICAL = ROOT / "runs/benchmarks/week4_final_recovery"
INF = ROOT / "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full"
out_dir = ROOT / "runs/benchmarks/week4_critical_fix/recalculated"
CONFIG = {"sample_fps": 5.0, "conf_thresh": 0.25, "eval_conf_thresh": 0.35,
          "window_n": 8, "confirm_k": 3, "cooldown_sec": 5.0,
          "feature_threshold": 0.02, "max_event_gap_sec": 2.0}


def read_json(path: Path) -> Any:
    """Read required JSON evidence; missing or malformed evidence is an error."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read an existing CSV without modifying annotations."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str | None:
    """Hash bytes on disk; missing input remains explicitly absent."""
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(name: str, value: Any) -> None:
    """Write a new report, with JSON NaN and infinity prohibited."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def keyed(rows: list[dict[str, Any]], key: str = "video_id") -> dict[str, dict[str, Any]]:
    """Reject collisions instead of silently dropping rows."""
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def duration(row: dict[str, Any]) -> float:
    """Derive unrounded exposure from decoded frame count and native FPS."""
    fps = float(row.get("fps") or 0)
    count = int(row.get("full_decoded_frames") or 0)
    if fps <= 0 or count <= 0 or not math.isfinite(fps):
        return 0.0
    return count / fps


def run_phase4_inference_reconciliation() -> dict[str, Any]:
    """Reconcile current inputs with historical evidence, without running inference."""
    manifest_path = ROOT / "data/negative_videos_manifest.json"
    manifest = keyed(read_json(manifest_path)["videos"])
    audit = keyed(read_json(HISTORICAL / "video_audit_final.json")["videos"])
    records = keyed(read_csv(INF / "video_inference_summary.csv"))
    run = read_json(INF / "run_manifest.json")
    config = read_json(INF / "inference_config.json")
    checkpoint = ROOT / run["model_checkpoint"].replace("\\", "/")
    current_checkpoint = digest(checkpoint)
    coverage = []
    for vid, entry in manifest.items():
        aud, rec = audit.get(vid, {}), records.get(vid, {})
        source_hash = digest(ROOT / entry["relative_path"])
        expected = entry.get("sha256")
        hash_ok = source_hash == expected == aud.get("sha256") if source_hash and expected else None
        frames_match = (int(rec.get("total_decoded_frames", -1)) == aud.get("full_decoded_frames")) if rec else False
        count = int(rec.get("processed_sampled_frames", 0))
        decoded = int(rec.get("total_decoded_frames", 0))
        native_fps = float(aud.get("fps") or 0)
        stride = max(1, round(native_fps / CONFIG["sample_fps"])) if native_fps > 0 else None
        expected_samples = (decoded + stride - 1) // stride if stride else None
        # Aggregate bounds do not prove the exact sampling schedule or timeline coverage.
        sampled_sane = 0 < count <= decoded if decoded else False
        coverage.append({"video_id": vid, "relative_path": entry["relative_path"],
                         "current_source_sha256": source_hash, "expected_sha256": expected,
                         "source_hash_matches": hash_ok, "decoded_frames_match": frames_match,
                         "processed_sampled_frames": count, "sampled_count_bounds_valid": sampled_sane,
                         "expected_samples_historical_stride": expected_samples,
                         "sampled_count_matches_historical_stride": count == expected_samples if expected_samples is not None else None,
                         "effective_sampling_fps_from_stride": native_fps / stride if stride else None,
                         "audit_decoded_frames": aud.get("full_decoded_frames"),
                         "inference_decoded_frames": decoded,
                         "path_matches": rec.get("relative_path") == entry["relative_path"] == aud.get("relative_path"),
                         "completed": rec.get("status") == "COMPLETED",
                         "frame_timestamps_verified": None,
                         "inference_alignment_verified": False})
    missing = any(row["current_source_sha256"] is None for row in coverage) or current_checkpoint is None
    def all_evidence(field: str) -> bool | None:
        values = [row[field] for row in coverage]
        return False if False in values else None if None in values else bool(values)
    source_commit = run.get("source_code_base_commit")
    commit_available = bool(source_commit) and subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT,
        capture_output=True).returncode == 0
    checks = {
        "source_hashes": all_evidence("source_hash_matches"),
        "video_id_mapping": set(manifest) == set(audit) == set(records) and all_evidence("path_matches"),
        "checkpoint_sha256": current_checkpoint == run.get("checkpoint_sha256") if current_checkpoint else None,
        "model_preprocessing": None,  # historical manifest lacks image size/preprocessing fingerprint
        "confidence_threshold": config.get("eval_conf_thresh") == CONFIG["eval_conf_thresh"],
        "sampling_fps": config.get("sample_fps") == CONFIG["sample_fps"],
        "sampling_strategy": None,  # no persisted per-frame schedule
        "configuration": config == CONFIG and run.get("config") == config,
        "source_commit_available": commit_available,
        "decoded_frame_counts": all_evidence("decoded_frames_match"),
        "sampled_frame_count_bounds": all_evidence("sampled_count_bounds_valid"),
        "sampled_count_matches_historical_stride": all_evidence("sampled_count_matches_historical_stride"),
        "sampled_frame_counts_verified": None,
        "run_completion": all_evidence("completed"),
        "frame_timestamps": None,
        "error_records": read_json(INF / "inference_errors.json") == [],
        "skipped_frames_accounted_for": None,
        # Current manifest mismatch is unresolved historical lineage, not proof of changed videos.
        "historical_manifest_authenticated": True if digest(manifest_path) == run.get("manifest_sha256") else None,
    }
    decision = reuse_decision(checks, missing_input=missing)
    decision["missing_source_video_ids"] = [row["video_id"] for row in coverage if row["current_source_sha256"] is None]
    decision["limitations"] = [key for key, value in checks.items() if value is None]
    decision["automatic_inference_executed"] = False
    save("inference_rerun_decision.json", decision)
    result = {"inference_run_id": run.get("run_id"),
              "source_code_base_commit": source_commit, "packaged_commit": run.get("full_git_sha"),
              "historical_manifest_sha256": run.get("manifest_sha256"),
              "current_manifest_sha256": digest(manifest_path),
              "checkpoint_sha256": current_checkpoint, "coverage": coverage,
              "timeline_coverage": "UNVERIFIED_TIMESTAMPS_NOT_STORED_PER_FRAME",
              "decision": decision}
    save("inference_reconciliation_final.json", result)
    return result


def run_phase5_negative_ground_truth() -> None:
    """Generate templates while preserving every existing review record and column."""
    videos = read_json(ROOT / "data/negative_videos_manifest.json")["videos"]
    queue, master = [], []
    for video in videos:
        base = {"video_id": video["video_id"], "relative_path": video["relative_path"],
                "duration_seconds": video.get("duration_seconds", 0), "duration_hours": video.get("duration_hours", 0)}
        queue.append(dict(base, annotator_01_id="", annotator_01_status="PENDING_REVIEW",
                          annotator_02_id="", annotator_02_status="PENDING_REVIEW",
                          consensus_status="PENDING_REVIEW", fire_intervals="", smoke_intervals="",
                          notes="Awaiting dual independent human annotation."))
        master.append(dict(base, semantic_label="PENDING_REVIEW", is_verified_negative=False,
                           verified_negative_exposure_sec=0.0, adjudication_notes="Pending human review."))
    for name, rows in (("negative_video_review_queue.csv", queue), ("negative_ground_truth_master.csv", master)):
        preserve_review_csv(out_dir / name, rows, "video_id", HISTORICAL / name)


def normalize_episodes(events: list[dict[str, Any]], intervals: dict[str, float], gap: float = 2.0) -> list[dict[str, Any]]:
    """Clip and merge all pipelines under the same video/class/camera episode protocol."""
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for event in events:
        vid = event["video_id"]
        if vid not in intervals:
            raise ValueError(f"Event video missing from authoritative scope: {vid}")
        start, end = float(event["start_sec"]), float(event["end_sec"])
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < start:
            raise ValueError("Invalid event interval")
        if start >= intervals[vid]:
            continue
        key = (vid, event.get("camera_id", f"BENCH_{vid}"), event["label"].upper())
        grouped.setdefault(key, []).append((start, min(end, intervals[vid])))
    result = []
    for (vid, camera, label), spans in sorted(grouped.items()):
        merged: list[list[float]] = []
        for start, end in sorted(spans):
            if merged and start - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        result.extend({"video_id": vid, "camera_id": camera, "label": label,
                       "start_sec": start, "end_sec": end} for start, end in merged)
    return result


def summarize_scope(rows: list[dict[str, Any]], events: dict[str, list[dict[str, Any]]], diagnostic: bool) -> dict[str, Any]:
    """Calculate matched numerators and denominators from the same video intervals."""
    ids = {row["video_id"] for row in rows}
    seconds = math.fsum(row["duration_seconds"] for row in rows)
    hours = seconds / 3600
    pipelines = {}
    for pipeline, episodes in events.items():
        count = sum(event["video_id"] in ids for event in episodes)
        ci = exact_poisson_ci(count, hours) if hours > 0 else (None, None)
        pipelines[pipeline] = {"event_count": count, "exposure_seconds": seconds,
                               "alert_event_rate_per_hour": count / hours if hours > 0 else None,
                               "exact_95_poisson_ci": list(ci)}
    baseline, temporal = pipelines["pipeline_a"]["event_count"], pipelines["pipeline_b2"]["event_count"]
    reduction = (baseline - temporal) / baseline * 100 if baseline > 0 else None
    return {"video_ids": sorted(ids), "video_count": len(ids), "exposure_seconds": seconds,
            "exposure_hours": hours, "diagnostic_only": diagnostic,
            "evaluated_intervals": {row["video_id"]: [0, row["duration_seconds"]] for row in rows},
            "pipelines": pipelines, "alert_rate_reduction_percent": reduction,
            "scientific_acceptance_eligible": False}


def run_phase6_far_recalculation() -> dict[str, Any]:
    """Evaluate diagnostic, strict, and independently verified scopes separately."""
    audit = read_json(HISTORICAL / "video_audit_final.json")
    reconciliation = read_json(out_dir / "inference_reconciliation_final.json")
    coverage = keyed(reconciliation["coverage"])
    rows = []
    for record in audit["videos"]:
        row = dict(record, duration_seconds=duration(record),
                   inference_alignment_verified=coverage.get(record["video_id"], {}).get("inference_alignment_verified", False))
        # Historical audit hash evidence is retained separately; current eligibility requires current bytes.
        row["sha256_verified"] = coverage.get(record["video_id"], {}).get("source_hash_matches") is True
        rows.append(row)
    intervals = {row["video_id"]: row["duration_seconds"] for row in rows}
    temporal = read_json(INF / "temporal_confirmed_events.json")
    raw_events = {"pipeline_a": read_json(INF / "frame_level_events.json"),
                  "pipeline_b1": temporal["pipeline_b1_events"], "pipeline_b2": temporal["pipeline_b2_events"]}
    events = {name: normalize_episodes(value, intervals) for name, value in raw_events.items()}
    episode_reconciliation = {
        name: {"source_episode_count": len(value), "normalized_episode_count": len(events[name]),
               "outside_authoritative_interval": sum(float(event["start_sec"]) >= intervals[event["video_id"]] for event in value),
               "clipped_at_interval_end": sum(float(event["start_sec"]) < intervals[event["video_id"]] < float(event["end_sec"]) for event in value)}
        for name, value in raw_events.items()
    }
    save("normalized_event_episodes.json", events)
    strict = [row for row in rows if strict_technical_eligible(row)]
    # Semantic scope remains empty unless the existing human review provides two independent IDs,
    # whole-video consensus, and positive exposure in addition to technical eligibility.
    reviews = keyed(read_csv(out_dir / "negative_video_review_queue.csv"))
    masters = keyed(read_csv(out_dir / "negative_ground_truth_master.csv"))
    semantic = []
    for row in strict:
        review, master = reviews.get(row["video_id"], {}), masters.get(row["video_id"], {})
        first, second = review.get("annotator_01_id"), review.get("annotator_02_id")
        if (first and second and first != second
            and review.get("annotator_01_status") == review.get("annotator_02_status") == review.get("consensus_status") == "VERIFIED_NEGATIVE"
            and master.get("semantic_label") == "VERIFIED_NEGATIVE"
            and str(master.get("is_verified_negative")).lower() == "true"
            and float(master.get("verified_negative_exposure_sec", 0)) >= row["duration_seconds"]):
            semantic.append(row)
    scopes = {
        "scope_a_historical_diagnostic": summarize_scope(rows, events, True),
        "scope_b_strict_technical_validation": summarize_scope(strict, events, False),
        "scope_b3_preliminary_nontruncated_diagnostic": summarize_scope(
            [row for row in rows if row.get("decode_status") in {"NORMAL_EOF", "NEEDS_REVIEW"}], events, True),
        "scope_c_scientifically_verified_negative_exposure": summarize_scope(semantic, events, False),
    }
    diagnostic_reduction = scopes["scope_a_historical_diagnostic"]["alert_rate_reduction_percent"]
    criteria = acceptance(scopes["scope_c_scientifically_verified_negative_exposure"]["exposure_hours"], diagnostic_reduction)
    result = {"terminology": "Diagnostic alert-event rate is not verified FAR.",
              "event_protocol": "Per video/camera/class; clip to [0, decoded_frames/native_fps); merge gaps <=2s for A, B1, B2.",
              "baseline_comparison": "Identical IDs, intervals and episode units; no global cross-scope baseline.",
              "event_reconciliation": episode_reconciliation,
              "timeline_alignment": "UNVERIFIED; diagnostic comparisons cannot establish scientific improvement",
              "audit_status_counts": dict(Counter(row.get("decode_status") for row in rows)),
              "decode_only_normal_eof": {"video_count": sum(row.get("decode_status") == "NORMAL_EOF" and row.get("full_decode_verified") is True for row in rows),
                                        "exposure_hours": math.fsum(duration(row) for row in rows if row.get("decode_status") == "NORMAL_EOF" and row.get("full_decode_verified") is True) / 3600},
              "criteria": criteria, **scopes}
    save("far_final.json", result)
    with (out_dir / "far_scope_comparison_final.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scope", "videos", "exposure_seconds", "events_a", "events_b2", "diagnostic_reduction_percent", "scientific_acceptance"])
        for name, scope in scopes.items():
            writer.writerow([name, scope["video_count"], scope["exposure_seconds"],
                             scope["pipelines"]["pipeline_a"]["event_count"],
                             scope["pipelines"]["pipeline_b2"]["event_count"],
                             scope["alert_rate_reduction_percent"], "PARTIAL"])
    lines = ["# Week 4: diagnostic rates and scientific acceptance", "",
             "Scientific acceptance: PARTIAL. Operating point: PROVISIONAL.", "",
             "| Scope | Videos | Hours | A episodes | B2 episodes | Diagnostic reduction (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, scope in scopes.items():
        reduction = scope["alert_rate_reduction_percent"]
        displayed_reduction = f"{reduction:.4f}" if reduction is not None else "UNVERIFIED"
        lines.append(f"| {name} | {scope['video_count']} | {scope['exposure_hours']:.8f} | "
                     f"{scope['pipelines']['pipeline_a']['event_count']} | {scope['pipelines']['pipeline_b2']['event_count']} | {displayed_reduction} |")
    lines.extend(["", "Rates and exact 95% Poisson intervals are stored in far_final.json.",
                  "Missing exposure yields null rate and null CI, never FAR=0.",
                  "Historical nontruncated scope is diagnostic only. NEEDS_REVIEW is excluded from strict scope.",
                  "All pipelines use identical IDs, exposure intervals and gap=2s episode merging.",
                  "Timeline alignment, independent negative GT, Recall/Delay, and validation-lock provenance remain blockers."])
    (out_dir / "far_final_scientific_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_phase7_positive_ground_truth() -> None:
    """Preserve source verification metadata; never invent a second annotator."""
    source = ROOT / "data/ground_truth/events_v1.1_snapshot.jsonl"
    validate_frozen_snapshot(source, ROOT / "data/ground_truth/checksums_v1.1.sha256")
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for index, event in enumerate(events, 1):
        vid, label = event["video_id"], event["label"]
        rows.append({"video_id": vid, "event_id": event.get("event_id", f"{vid}_E{index:02d}_{label}"),
                     "class": label, "onset_sec": event["start_sec"], "offset_sec": event["end_sec"],
                     "annotator_1": event.get("annotator_01") or "", "annotator_2": event.get("annotator_02") or "",
                     "review_status": event.get("verification_status", "pending_verification"),
                     "disagreement_flag": False, "adjudication_notes": event.get("notes", ""),
                     "evidence_ref": f"FireSmokeLabeler/videos/{vid}.avi"})
    preserve_review_csv(out_dir / "gt_positive_review_queue.csv", rows, "event_id", HISTORICAL / "gt_positive_review_queue.csv")
    save("positive_gt_source_state.json", {"snapshot_sha256": digest(source), "event_count": len(events),
         "verification_status_counts": dict(Counter(event.get("verification_status", "unknown") for event in events)),
         "source_event_ids_present": all("event_id" in event for event in events),
         "legacy_queue_ids": "Legacy ordinal IDs retained; timestamps and video/class identity checked on merge.",
         "independent_evidence_audit": "Historical cross_verified claims preserved; external annotation records not reauthenticated."})


def run_phase8_operating_point() -> None:
    """Keep the unchanged operating point provisional until all evidence gates pass."""
    far = read_json(out_dir / "far_final.json")
    save("operating_point_final.json", {
        "operating_point_configuration": CONFIG, "classification": "PROVISIONAL",
        "official_promotion_allowed": False, "criteria_verification": far["criteria"],
        "blocking_reasons": ["Independent negative exposure <5h", "Positive GT provenance and eligibility incomplete",
                             "Event Recall and Delay not verified on independent eligible events",
                             "Inference frame timestamps/preprocessing/sampling lineage unverified",
                             "Validation lock provenance not independently authenticated"],
        "retuning_performed": False})


def tool_version(name: str) -> str | None:
    """Record available tool versions without requiring tool installation."""
    binary = shutil.which(name)
    if not binary:
        return None
    return subprocess.run([binary, "-version"], capture_output=True, text=True).stdout.splitlines()[0]


def run_lineage(started: str) -> None:
    """Record historical source and current producer separately, plus duration reconciliation."""
    import cv2
    audit = read_json(HISTORICAL / "video_audit_final.json")
    run = read_json(INF / "run_manifest.json")
    lineage = []
    for path in sorted(HISTORICAL.glob("*")):
        if not path.is_file():
            continue
        producer = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
        metadata = read_json(path) if path.suffix == ".json" else {}
        embedded = {key: metadata[key] for key in ("generated_at", "git_head", "source_code_base_commit", "packaged_commit", "audit_policy_version")
                    if isinstance(metadata, dict) and key in metadata}
        lineage.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                        "artifact_packaging_commit": producer,
                        "artifact_executing_commit": None, "embedded_historical_metadata": embedded,
                        "historical_environment": None,
                        "note": "Packaging commit does not establish the executing dirty worktree; missing historical environment/start/completion remain unknown."})
    videos = audit["videos"]
    exact = math.fsum(duration(row) for row in videos) / 3600
    stored_seconds = math.fsum(float(row.get("duration_seconds", 0)) for row in videos) / 3600
    rounded_hours = math.fsum(float(row.get("duration_hours", 0)) for row in videos)
    save("consistency_report.json", {
        "status": "PARTIAL", "audit_reported_hours": audit.get("total_evaluated_duration_hours"),
        "sum_per_video_rounded_hours": rounded_hours,
        "sum_per_video_stored_seconds_hours": stored_seconds, "decoded_frames_over_native_fps_hours": exact,
        "historical_inference_exposure_hours": run.get("exposure_summary", {}).get("total_exposure_hours"),
        "explanation": "7.1309 is the sum of per-video rounded duration_hours; ~7.1312 uses duration_seconds. Both use the same audit IDs. Inference 7.2026 uses a different historical denominator; timeline evidence remains missing.",
        "historical_audit_observations_not_rerun": {"total_decoded_frames": audit.get("total_decoded_frames"), "decode_status_counts": audit.get("decode_status_counts")},
        "full_decode_executed_current_host": False})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    save("artifact_lineage.json", {
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "producer_base_commit": head, "producer_worktree_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)).hexdigest(),
        "producer_state": "UNCOMMITTED_WORKTREE; new source files hashed below",
        "producer_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [
            Path(__file__), ROOT / "src/week4_acceptance.py", ROOT / "src/ground_truth_integrity.py", ROOT / "src/event_eval.py"]},
        "historical_inference_source_commit": run.get("source_code_base_commit"),
        "historical_inference_packaged_commit": run.get("full_git_sha"),
        "manifest_sha256": digest(ROOT / "data/negative_videos_manifest.json"),
        "checkpoint_sha256": digest(ROOT / run["model_checkpoint"].replace("\\", "/")),
        "audit_policy_version": audit.get("audit_policy_version"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "opencv": cv2.__version__, "ffmpeg": tool_version("ffmpeg"), "ffprobe": tool_version("ffprobe")},
        "historical_artifacts": lineage,
        "new_artifacts": {p.name: digest(p) for p in out_dir.glob("*") if p.is_file() and p.name != "artifact_lineage.json"},
        "historical_start_completion": "Use original generated_at/last_audited_at; missing timestamps remain unknown"})


def main() -> None:
    """Run independent software recovery phases without overwriting historical artifacts."""
    global out_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=out_dir)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if ROOT not in out_dir.parents or out_dir == HISTORICAL or HISTORICAL in out_dir.parents:
        parser.error("Use a new output directory within the workspace, outside historical recovery artifacts")
    started = datetime.now(timezone.utc).isoformat()
    run_phase4_inference_reconciliation()
    run_phase5_negative_ground_truth()
    run_phase6_far_recalculation()
    run_phase7_positive_ground_truth()
    run_phase8_operating_point()
    run_lineage(started)
    print(f"Recovery complete: {out_dir}; scientific acceptance PARTIAL")


if __name__ == "__main__":
    main()
