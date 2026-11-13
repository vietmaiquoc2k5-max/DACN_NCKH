"""Export evidence for a real single reviewer without claiming cross-verification."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import sha256, frozen
from ground_truth_studio.models import utc_now, compare_events
from ground_truth_studio.review_repository import ReviewRepository


STATUS = "SINGLE_HUMAN_REVIEWED_PROVISIONAL"


def _write_new_json(path: Path, value: Any) -> None:
    """Create one immutable export document, refusing to overwrite existing data."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_new_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Serialize provisional event candidates, retaining source evidence."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export_single_reviewer(repository: ReviewRepository, token: str) -> Path:
    """Export submitted own-review records as non-official research candidates.

    Args:
        repository: The existing Studio private repository.
        token: Authenticated annotator session token.

    Returns:
        Path to a newly created immutable private export directory.

    Raises:
        PermissionError: When the actor lacks the annotation role.
        ValueError: When no submissions or a source/provenance check fails.
    """
    actor = repository.actor(token, {"annotator_01", "annotator_02"})
    frozen_hash, _ = frozen(repository.settings)
    reviews = [row for row in repository.own_review_history(token) if row["status"] == "SUBMITTED"]
    if not reviews:
        raise ValueError("Chưa có review đã Submit; Save Draft không phải là đánh giá hoàn tất.")

    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    latest = {row["video_id"]: row["id"] for row in reviews}
    with repository.connection() as db:
        links = {row["review_id"]: dict(row) for row in db.execute("SELECT * FROM blind_review_links")}
    by_id = {row["id"]: row for row in reviews}
    intra = []
    for review in reviews:
        video = repository.video(review["video_id"])
        source = review.get("source") or {}
        expected = source.get("source_sha256")
        path = confined(repository.settings.repo, video["relative_path"])
        if not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"BLOCKED_SOURCE: {review['video_id']} thiếu video hoặc source SHA-256 không khớp")
        payload = review["payload"]
        if not payload.get("evidence_ref") or payload.get("attest_full_view") is not True:
            raise ValueError(f"BLOCKED_EVIDENCE: {review['video_id']} thiếu bằng chứng xác nhận xem đầy đủ")
        base = {
            "video_id": review["video_id"],
            "relative_path": video["relative_path"],
            "source_sha256": expected,
            "review_id": review["id"],
            "reviewer_id": actor["id"],
            "reviewer_person_id": actor["person_id"],
            "submitted_at_utc": review["submitted_at"],
            "evidence_ref": payload["evidence_ref"],
            "review_status": STATUS,
            "cross_verified": False,
            "scientific_acceptance": "PARTIAL",
            "is_primary_round_for_video": review["id"] == latest[review["video_id"]],
            "blind_rereview": links.get(review["id"]),
        }
        records.append({**base, "kind": video["kind"], "decision": payload.get("decision"),
                        "duration_seconds": source.get("duration"), "attest_full_view": True,
                        "events": payload.get("events", []),
                        "uncertain_intervals": payload.get("uncertain_intervals", []),
                        "source_timestamp_provenance": source.get("timestamp_provenance"),
                        "duration_provenance": source.get("duration_provenance")})
        link = links.get(review["id"])
        if link and link["original_review_id"] in by_id:
            intra.append({"video_id": review["video_id"], "first_review_id": link["original_review_id"],
                          "rereview_id": review["id"], "independent_cross_verification": False,
                          "comparison": compare_events(by_id[link["original_review_id"]]["payload"]["events"], payload.get("events", []))})
        if not base["is_primary_round_for_video"]:
            continue
        if payload.get("events"):
            for event in payload.get("events", []):
                if event.get("source_sha256") != expected:
                    raise ValueError(f"BLOCKED_EVENT_SOURCE: {review['video_id']}")
                candidates.append({**event, **base,
                                   "camera_id": f"BENCH_{review['video_id']}",
                                   "verification_status": "pending_verification",
                                   "annotation_status": STATUS})
        if video["kind"] == "negative":
            negative.append({**base, "semantic_decision": payload.get("decision"),
                             "events": payload.get("events", []),
                             "uncertain_intervals": payload.get("uncertain_intervals", []),
                             "verified_negative_exposure_sec": None,
                             "far_eligible": False,
                             "reason": "Single reviewer only; independent GT and technical gates not passed"})

    # Complete all validation before creating a new output directory.
    root = repository.settings.private / "exports"
    root.mkdir(parents=True, exist_ok=True)
    folder = root / ("single_reviewer_" + uuid.uuid4().hex)
    folder.mkdir(mode=0o700)
    _write_new_json(folder / "single_review_summary.json", {
        "generated_at_utc": utc_now(), "mode": STATUS,
        "reviewer_id": actor["id"], "reviewer_person_id": actor["person_id"],
        "submitted_videos": len(latest), "submitted_rounds": len(records), "positive_event_candidates": len(candidates),
        "negative_reviewed_videos": len(negative),
        "scientific_gt_status": "PARTIAL_SINGLE_REVIEWER",
        "official_far_per_hour": None, "official_event_recall": None,
        "official_event_f1": None, "official_detection_delay_sec": None,
        "cross_verified": False, "approved": False,
        "notes": "No second independent person. This bundle must not enter scientific final acceptance without separate approval and evidence.",
    })
    _write_new_json(folder / "single_review_records.json", records)
    _write_new_json(folder / "single_negative_reviews.json", negative)
    _write_new_json(folder / "intra_annotator_consistency.json", intra)
    _write_new_jsonl(folder / "single_reviewer_gt_candidate.jsonl", candidates)
    _write_new_json(folder / "ground_truth_provenance.json", {
        "generated_at_utc": utc_now(), "frozen_snapshot_sha256": frozen_hash,
        "reviewer_id": actor["id"], "identity_reference": actor["identity_ref"],
        "review_ids": [row["id"] for row in reviews], "source_sha256": {row["video_id"]: row["source_sha256"] for row in records},
        "mode": STATUS, "cross_verified": False, "official": False})
    with repository.connection() as db:
        repository.log(db, actor["id"], "EXPORT_SINGLE_REVIEWER_PROVISIONAL", folder.name,
                       {"review_ids": [row["id"] for row in reviews], "cross_verified": False})
    _write_new_json(folder / "file_sha256.json", {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in folder.iterdir() if file.is_file()
    })
    return folder
