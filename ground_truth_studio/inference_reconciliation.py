"""Reconcile historical observations without inventing runtime provenance."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ground_truth_studio.catalog_recovery import save_json
from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import sha256


def classify_reuse(*, available: bool, contradiction: bool, complete_evidence: bool) -> str:
    """Separate missing evidence from demonstrated runtime contradictions."""
    if not available:
        return "BLOCKED_MISSING_INPUT"
    if contradiction:
        return "RERUN_REQUIRED"
    return "REUSE_VERIFIED" if complete_evidence else "UNVERIFIED_MISSING_EVIDENCE"


def reconcile(settings: Settings, output: Path) -> dict[str, Any]:
    """Write a per-video decision for the referenced historical negative run.

    The run has aggregate CSV and event episodes, but no persisted sampled-frame
    index/PTS log. Counts alone cannot authenticate completion or preprocessing.
    """
    history = settings.repo / "runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full"
    manifest = json.loads((history / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((history / "inference_config.json").read_text(encoding="utf-8"))
    legacy = json.loads((settings.repo / "data/negative_videos_manifest.json").read_text(encoding="utf-8"))["videos"]
    local = json.loads((output / "video_catalog_local_v2.json").read_text(encoding="utf-8"))
    sources = {row["video_id"]: row for row in local["videos"]}
    with (history / "video_inference_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = {row["video_id"]: row for row in csv.DictReader(stream)}
    audit_path = output / "technical_audit_63.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"videos": [], "status": "NOT_RUN"}
    audited = {row["video_id"]: row for row in audit["videos"]}
    checkpoint = confined(settings.repo, manifest["model_checkpoint"].replace("\\", "/"))
    checkpoint_digest = sha256(checkpoint) if checkpoint.is_file() else None
    errors = json.loads((history / "inference_errors.json").read_text(encoding="utf-8"))
    records = []
    for old in legacy:
        video_id = old["video_id"]
        source, aggregate = sources[video_id], summaries.get(video_id)
        current_audit = audited.get(video_id, {})
        sampled = int(aggregate["processed_sampled_frames"]) if aggregate else None
        decoded = current_audit.get("full_decoded_frames")
        fps = current_audit.get("fps") or source.get("fps")
        step = max(1, round(fps / config["sample_fps"])) if fps else None
        expected = math.ceil(decoded / step) if decoded is not None and step else None
        missing = ["RUN_BOUND_SOURCE_MANIFEST", "RUNTIME_IMAGE_SIZE_AND_PREPROCESSING", "SAMPLED_FRAME_INDEX_SOURCE_PTS_LOG",
                   "INDEPENDENT_SOURCE_PTS_COVERAGE", "PER_VIDEO_COMPLETION_BINDING"]
        if checkpoint_digest is None:
            missing.append("LOCAL_CHECKPOINT")
        # A different current manifest is not evidence that the historical model
        # saw different bytes. Keep that source binding explicitly unverified.
        contradiction = (source.get("historical_sha256_matches") is False or
                         (checkpoint_digest is not None and checkpoint_digest != manifest["checkpoint_sha256"]))
        state = classify_reuse(available=source["source_available"], contradiction=contradiction, complete_evidence=False)
        records.append({"video_id": video_id, "decision": state, "run_id": manifest["run_id"],
                        "source_sha256_current": source.get("sha256"), "source_sha256_historical_catalog": old.get("sha256"),
                        "historical_catalog_hash_matches_current": source.get("historical_sha256_matches"),
                        "source_run_binding_verified": False, "checkpoint_sha256_claimed": manifest["checkpoint_sha256"],
                        "checkpoint_sha256_current": checkpoint_digest, "checkpoint_matches": checkpoint_digest == manifest["checkpoint_sha256"] if checkpoint_digest else None,
                        "source_code_base_commit_claimed": manifest.get("source_code_base_commit"),
                        "packaged_commit_claimed": manifest.get("full_git_sha"), "config_claimed": config,
                        "image_size_runtime": None, "image_size_source_code_observation": 416,
                        "sampling_source_code_observation": "frame_index % max(1, round(native_fps / 5)) == 0; time = frame_index / native_fps",
                        "sampling_mode_claimed": "FULL_DURATION_SAMPLED_5_FPS_NOT_NATIVE_FRAME_INFERENCE",
                        "native_decoded_frames_fresh": decoded, "expected_sampled_frames_from_fresh_decode": expected,
                        "actual_processed_frames_claimed": sampled, "sample_count_arithmetic_matches": sampled == expected if expected is not None and sampled is not None else None,
                        "frame_index_evidence": None, "source_pts_evidence": None, "full_timeline_verified": False,
                        "aggregate_completion_claim": aggregate, "global_error_records": errors,
                        "event_merge_protocol_claimed": {"max_event_gap_sec": config["max_event_gap_sec"], "unit": "episode"},
                        "fresh_decode_status": current_audit.get("decode_status"),
                        "technical_eligible_for_new_inference": current_audit.get("decode_status") == "NORMAL_EOF" and current_audit.get("full_decode_verified") is True and current_audit.get("sha256_verified") is True,
                        "missing_evidence": missing, "reason": "EXCLUDED_MISSING_SOURCE" if not source["source_available"] else "Missing runtime evidence does not prove incorrect model execution"})
    result = {"run_id": manifest["run_id"], "scope_id": local["scope_id"], "technical_audit_status": audit["status"],
              "decision_counts": dict(Counter(row["decision"] for row in records)), "videos": records,
              "historical_artifact_sha256": {path.name: sha256(path) for path in history.iterdir() if path.is_file()},
              "current_manifest_sha256": sha256(settings.repo / "data/negative_videos_manifest.json"),
              "source_code_observation": {"path": "scripts/run_week4_complete_validation.py", "sha256_current": sha256(settings.repo / "scripts/run_week4_complete_validation.py"), "runtime_binding_verified": False},
              "run_manifest_digest_claim": manifest.get("manifest_sha256"),
              "rerun_completed": 0, "new_inference_decision": "NOT_RUN",
              "reason": "No submitted human GT currently available; runtime evidence is missing, not a proven execution failure. Complete strict preflight and human review before an instrumented comparison run.",
              "proposed_future_scope": [row["video_id"] for row in records if row["decision"] != "BLOCKED_MISSING_INPUT"],
              "rerun_required_scope": [row["video_id"] for row in records if row["decision"] == "RERUN_REQUIRED"],
              "operating_point": "PROVISIONAL", "scientific_status": "PARTIAL"}
    save_json(output / "inference_reconciliation.json", result)
    return result
