"""Conditional single-human evaluation with identical A/B source intervals."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from src.event_eval import Event, evaluate, exact_poisson_ci
from src.inference_frame_evidence import frame_evidence_for_video


def evaluate_provisional(reviews: list[dict[str, Any]], audits: dict[str, dict[str, Any]],
                         pipelines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate submitted full-video reviews only when both timelines are bound.

    Args:
        reviews: Primary submitted records from a single-reviewer export.
        audits: Fresh full-decode records indexed by video ID.
        pipelines: A and B evidence, each containing inspected frame JSONL audits,
            independently decoded reference_timestamps, expected_samples, run_id,
            checkpoint_sha256, detector_config, merge_gap_sec and merged events.
            Reference PTS must come from source decoding, never inference logs.

    Returns:
        Provisional metrics and the immutable shared video/hash/interval scope.
        Missing evidence produces exclusions and null metrics, never acceptance.
    """
    excluded, scope, gt, positive, negative = [], [], [], set(), set()
    selected: dict[str, dict[str, Any]] = {}
    for row in reviews:
        if row.get("is_primary_round_for_video") is True:
            if row["video_id"] in selected:
                raise ValueError("Duplicate primary review for one video")
            selected[row["video_id"]] = row
    for video_id, row in sorted(selected.items()):
        reasons = []
        digest, duration = row.get("source_sha256"), row.get("duration_seconds")
        if (row.get("review_status") != "SINGLE_HUMAN_REVIEWED_PROVISIONAL" or
                not row.get("submitted_at_utc") or not row.get("reviewer_person_id") or
                row.get("attest_full_view") is not True or not row.get("evidence_ref")):
            reasons.append("HUMAN_SUBMISSION_EVIDENCE_MISSING")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            reasons.append("INVALID_REVIEWED_DURATION")
        if row.get("source_timestamp_provenance") != "SOURCE_PTS":
            reasons.append("SOURCE_PTS_REQUIRED")
        if row.get("uncertain_intervals") or row.get("decision") == "UNCERTAIN":
            reasons.append("UNCERTAIN_CONTENT")
        audit = audits.get(video_id, {})
        if (audit.get("sha256") != digest or not digest or audit.get("sha256_verified") is not True or
                audit.get("decode_status") != "NORMAL_EOF" or audit.get("full_decode_verified") is not True):
            reasons.append("TECHNICAL_AUDIT_NOT_ELIGIBLE")
        decisions = row.get("decision")
        events = row.get("events", [])
        is_negative = row.get("kind") == "negative" and decisions == "VERIFIED_NEGATIVE" and not events
        if not is_negative and not events:
            reasons.append("NO_POSITIVE_EVENTS_OR_VERIFIED_NEGATIVE_DECISION")
        if any(event.get("source_sha256") != digest or event.get("timestamp_provenance") != "SOURCE_PTS" for event in events):
            reasons.append("EVENT_SOURCE_PROVENANCE_MISMATCH")
        for name in ("A", "B"):
            pipeline = pipelines.get(name, {})
            pts = pipeline.get("reference_timestamps", {}).get(video_id)
            expected = pipeline.get("expected_samples", {}).get(video_id, 0)
            native_count, fps = audit.get("full_decoded_frames"), audit.get("fps")
            step = max(1, round(fps / 5.0)) if fps and fps > 0 else None
            indices = list(range(0, native_count, step)) if type(native_count) is int and native_count > 0 and step else []
            if (not indices or not pts or sorted(pts) != indices or expected != len(indices) or
                    pipeline.get("sampling_mode") != "SOURCE_NATIVE_STRIDE_5_FPS" or
                    audit.get("duration_seconds") != duration):
                reasons.append(name + "_FULL_DURATION_SAMPLING_UNVERIFIED")
            check = frame_evidence_for_video(video_id, pipeline.get("frame_audits", []),
                run_id=pipeline.get("run_id"), source_sha256=digest,
                checkpoint_sha256=pipeline.get("checkpoint_sha256"), expected_samples=expected,
                detector_config=pipeline.get("detector_config"), reference_timestamps=pts)
            if not check["full_timeline_independently_verified"] or expected <= 0:
                reasons.append(name + "_TIMELINE_UNVERIFIED")
            if pipeline.get("merge_gap_sec") != 2.0:
                reasons.append(name + "_EPISODE_PROTOCOL_UNVERIFIED")
            if pipeline.get("covered_intervals", {}).get(video_id) != [0.0, duration]:
                reasons.append(name + "_REVIEW_INTERVAL_NOT_COVERED")
        if reasons:
            excluded.append({"video_id": video_id, "reasons": reasons})
            continue
        scope.append({"video_id": video_id, "source_sha256": digest, "start_sec": 0.0, "end_sec": duration,
                      "kind": "negative" if is_negative else "positive", "review_id": row["review_id"]})
        if is_negative:
            negative.add(video_id)
        else:
            positive.add(video_id)
            gt.extend(Event(video_id, "BENCH_" + video_id, event["label"], event["start_sec"], event["end_sec"]) for event in events)
    hours = sum(row["end_sec"] for row in scope if row["kind"] == "negative") / 3600
    results = {}
    for name in ("A", "B"):
        predictions = [Event(**event) for event in pipelines.get(name, {}).get("events", [])
                       if event["video_id"] in positive | negative]
        # Reject a malformed episode instead of silently clipping or dropping it.
        limits = {row["video_id"]: row["end_sec"] for row in scope}
        if any(event.camera_id != "BENCH_" + event.video_id or event.end_sec > limits[event.video_id] for event in predictions):
            raise ValueError("Prediction outside authenticated camera/interval scope")
        merged: list[Event] = []
        for event in sorted(predictions, key=lambda item: (item.video_id, item.camera_id, item.label, item.start_sec)):
            if merged and (merged[-1].video_id, merged[-1].camera_id, merged[-1].label) == (event.video_id, event.camera_id, event.label) and event.start_sec - merged[-1].end_sec <= 2.0:
                previous = merged.pop()
                event = Event(event.video_id, event.camera_id, event.label, previous.start_sec, max(previous.end_sec, event.end_sec))
            merged.append(event)
        predictions = merged
        metrics = {"event_recall": None, "event_precision": None, "event_f1": None, "detection_delay_sec": None,
                   "far_per_hour": None, "poisson_ci_95": None}
        if positive:
            measured = evaluate(gt, predictions, positive, negative or None, hours or None, allow_missing_negative=True)["metrics"]
            metrics.update(event_recall=measured["event_instance_recall"], event_precision=measured["event_precision"],
                           event_f1=measured["event_f1"], detection_delay_sec=measured["mean_detection_delay_sec"])
        if hours > 0:
            count = sum(event.video_id in negative for event in predictions)
            metrics.update(far_per_hour=count / hours, poisson_ci_95=list(exact_poisson_ci(count, hours)), negative_episode_count=count)
        results[name] = metrics
    return {"mode": "PROVISIONAL_SINGLE_REVIEWER", "scientific_status": "PARTIAL", "official": False,
            "status": "PARTIAL_INSUFFICIENT_VERIFIED_EXPOSURE" if hours < 5 else "PARTIAL_INDEPENDENT_GT_AND_VALIDATION_LOCK_REQUIRED",
            "operating_point": "PROVISIONAL", "human_review_status": "HUMAN_REVIEW_PENDING" if not selected else "SINGLE_HUMAN_REVIEWED_PROVISIONAL",
            "verified_negative_hours": hours, "scope": scope, "excluded": excluded,
            "scope_id": hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest(),
            "pipeline_metrics": results, "independent_cross_verification": False}
