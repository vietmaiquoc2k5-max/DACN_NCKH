"""Versioned, conflict-aware candidate exports; frozen GT is never a destination."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import frozen, sha256
from ground_truth_studio.models import compare_events, utc_now
from ground_truth_studio.review_repository import CSV_FIELDS, ConflictError, ReviewRepository
from src.week4_acceptance import strict_technical_eligible


def _current(repository: ReviewRepository) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    with repository.connection() as db:
        db.execute("BEGIN")  # Reviews and adjudications must come from one WAL read snapshot.
        reviews = [repository.decode_review(row) for row in db.execute("SELECT r.* FROM review_sessions r JOIN review_heads h ON h.review_id=r.id")]
        adjudications = {}
        for row in db.execute("SELECT * FROM adjudications ORDER BY rowid"):
            adjudications[row["video_id"]] = dict(row, data=json.loads(row["data"]))
    return reviews, adjudications


def summary(repository: ReviewRepository, *, captured: tuple[list[dict[str, Any]], dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Return aggregate progress, keeping semantic signoff separate from FAR eligibility."""
    _, historical = frozen(repository.settings)
    videos = repository.videos()
    reviews, adjudications = captured if captured is not None else _current(repository)
    candidates, semantic_negatives, eligible_seconds = 0, 0, 0.0
    by_id = {row["id"]: row for row in reviews}
    for video_id, adjudication in adjudications.items():
        data = adjudication["data"]
        pair = [by_id.get(item) for item in data["review_ids"]]
        if adjudication["decision"] != "ACCEPT" or not all(pair):
            continue
        video = repository.video(video_id)
        candidates += len(data["final_events"]) if video["kind"] != "negative" else 0
        if video["kind"] == "negative" and all(review["payload"].get("decision") == "VERIFIED_NEGATIVE"
                and not review["payload"].get("events") and not review["payload"].get("uncertain_intervals") for review in pair) and not data["final_events"]:
            semantic_negatives += 1
            technical = dict(video["technical"])
            source = confined(repository.settings.repo, video["relative_path"])
            technical["sha256_verified"] = source.is_file() and sha256(source) == data["source_sha256"] == video["expected_sha256"]
            technical["duration_seconds"] = min(float(technical.get("duration_seconds") or 0), *(review["source"]["duration"] for review in pair))
            if strict_technical_eligible(technical):
                eligible_seconds += technical["duration_seconds"]
    return {"positive_videos": sum(row["kind"] == "positive" for row in videos),
            "positive_candidate_videos": sum(row["kind"] == "positive_candidate" for row in videos),
            "negative_candidates_available": len(repository.working_videos("negative")),
            "missing_historical_references": sum(not row["available"] and row.get("manifest_origin") == "historical" for row in videos),
            "technical_verified_available": sum(row["available"] and row.get("technical_status") == "NORMAL_EOF" for row in videos),
            "negative_videos": sum(row["kind"] == "negative" for row in videos),
            "available_sources": sum(row["available"] for row in videos),
            "historical_event_count": len(historical),
            "historical_pending_events": sum(row.get("verification_status") == "pending_verification" for row in historical),
            "historical_cross_verified_claims": sum(row.get("verification_status") == "cross_verified" for row in historical),
            "human_adjudicated_candidate_events": candidates,
            "videos_without_studio_review": len(videos) - len({row["video_id"] for row in reviews}),
            "submitted_reviews": sum(row["status"] == "SUBMITTED" for row in reviews),
            "disputed_videos": sum(row["decision"] == "DISPUTED" for row in adjudications.values()),
            "adjudicated_videos": len(adjudications), "semantic_negative_consensus_videos": semantic_negatives,
            "eligible_negative_exposure_seconds": eligible_seconds,
            "eligible_negative_exposure_hours": eligible_seconds / 3600,
            "scientific_status": "BLOCKED_HUMAN_REVIEW" if not adjudications else "CANDIDATE_ONLY_PENDING_OFFICIAL_APPROVAL",
            "operating_point_promotion_allowed": False}


class ExportService:
    def __init__(self, repository: ReviewRepository):
        self.repository = repository

    def export(self, token: str) -> Path:
        """Export a fresh bundle with original CSV snapshots and provenance, never overwrite.

        Raises:
            ConflictError: A private CSV changed externally, or another human owns its row.
        """
        actor = self.repository.actor(token, {"adjudicator", "admin"})
        snapshot_hash, historical = frozen(self.repository.settings)
        if self.repository.snapshot_csv():
            raise ConflictError("Private CSV changed outside Studio; operator must inspect and explicitly snapshot changes")
        reviews, adjudications = _current(self.repository)
        with self.repository.connection() as db:
            snapshots = {}
            for row in db.execute("SELECT * FROM csv_snapshots ORDER BY id"):
                snapshots[row["filename"]] = dict(row)
        rows: dict[str, list[dict[str, Any]]] = {
            name: list(csv.DictReader(io.StringIO(item["content"]))) for name, item in snapshots.items()}
        for name in CSV_FIELDS:
            rows.setdefault(name, [])
        rows["positive_annotator_01.csv"] = []
        for review in reviews:
            if review["status"] != "SUBMITTED":
                continue
            video = self.repository.video(review["video_id"])
            payload = review["payload"]
            source_path = confined(self.repository.settings.repo, video["relative_path"])
            if not source_path.is_file() or sha256(source_path) != review["source"]["source_sha256"]:
                raise ConflictError("Submitted source changed or is missing; preserve review and resolve provenance before export")
            common = {"video_id": video["video_id"], "relative_path": video["relative_path"],
                      "source_sha256": review["source"]["source_sha256"], "reviewer_id": review["user_id"],
                      "reviewed_at_utc": review["submitted_at"], "evidence_ref": payload["evidence_ref"], "signoff": "YES"}
            if video["kind"] != "negative":
                name = "positive_" + review["slot"] + ".csv"
                common["annotation_events_json"] = json.dumps(payload["events"], ensure_ascii=False)
            else:
                name = "negative_" + review["slot"] + ".csv"
                common.update(decision={"VERIFIED_NEGATIVE": "NEGATIVE", "CONTAINS_FIRE": "FIRE", "CONTAINS_SMOKE": "SMOKE", "UNCERTAIN": "UNCERTAIN"}[payload["decision"]],
                              fire_intervals_json=json.dumps([e for e in payload["events"] if e["label"] == "FIRE"]),
                              smoke_intervals_json=json.dumps([e for e in payload["events"] if e["label"] == "SMOKE"]),
                              uncertain_intervals_json=json.dumps(payload.get("uncertain_intervals", [])))
            existing = next((row for row in rows[name] if row["video_id"] == video["video_id"]), None)
            if existing and existing.get("reviewer_id") and existing["reviewer_id"] != review["user_id"]:
                raise ConflictError("External human row belongs to another reviewer; export requires manual reconciliation")
            if existing:
                existing.update(common)
            else:
                rows[name].append(common)
        candidates = []
        by_id = {review["id"]: review for review in reviews}
        for video_id, adjudication in adjudications.items():
            data = adjudication["data"]
            pair = [by_id.get(item) for item in data["review_ids"]]
            if not all(pair):
                continue
            approved = adjudication["decision"] == "ACCEPT"
            for event in data["final_events"] if approved else []:
                candidates.append(dict(event, video_id=video_id, camera_id=f"BENCH_{video_id}", confidence=1.0,
                                       annotator_id=pair[0]["user_id"], annotator_01=pair[0]["user_id"], annotator_02=pair[1]["user_id"],
                                       reviewer_id=adjudication["user_id"], reviewed_at_utc=adjudication["created_at"],
                                       evidence_ref=data["evidence_ref"], annotation_status="HUMAN_ADJUDICATED_CANDIDATE",
                                       verification_status="pending_verification", adjudication_id=adjudication["id"],
                                       review_session_ids=data["review_ids"], source_sha256=data["source_sha256"]))
            video = self.repository.video(video_id)
            approval = "APPROVE" if approved else ""
            if video["kind"] == "negative":
                record = next((row for row in rows["negative_adjudication.csv"] if row["video_id"] == video_id), None)
                negative_consensus = (not data["final_events"] and all(review["payload"].get("decision") == "VERIFIED_NEGATIVE"
                                      and not review["payload"].get("uncertain_intervals") for review in pair))
                fields = {"video_id": video_id, "source_sha256": data["source_sha256"], "approval": approval if negative_consensus else "",
                          "adjudicator_id": adjudication["user_id"], "adjudicated_at_utc": adjudication["created_at"],
                          "evidence_ref": data["evidence_ref"], "notes": data["reason"]}
                if record:
                    record.update(fields)
                else:
                    rows["negative_adjudication.csv"].append(fields)
            else:
                originals = [(index, event) for index, event in enumerate(historical, 1) if event["video_id"] == video_id]
                mapping = compare_events([event for _, event in originals], pair[1]["payload"]["events"])
                matched = {item["first"]: item["second"] for item in mapping["pairs"]}
                final_matched = {item["first"] for item in compare_events([event for _, event in originals], data["final_events"])["pairs"]}
                for i, (index, event) in enumerate(originals):
                    record = next((row for row in rows["positive_adjudication.csv"] if str(row["gt_index"]) == str(index)), None)
                    fields = {"gt_index": index, "gt_snapshot_sha256": snapshot_hash, "video_id": video_id,
                              "label": event["label"], "start_sec": event["start_sec"], "end_sec": event["end_sec"],
                              "reviewer_02_event_index": matched.get(i, ""), "approval": approval if i in matched and i in final_matched else "",
                              "adjudicator_id": adjudication["user_id"], "adjudicated_at_utc": adjudication["created_at"],
                              "evidence_ref": data["evidence_ref"], "notes": data["reason"]}
                    if record:
                        record.update(fields)
                    else:
                        rows["positive_adjudication.csv"].append(fields)
        backup = self.repository.backup()
        destination = self.repository.settings.private / "exports" / uuid.uuid4().hex
        destination.mkdir(parents=True, exist_ok=False)
        for name, items in rows.items():
            fields = list(CSV_FIELDS.get(name, CSV_FIELDS["positive_annotator_02.csv"]))
            fields = list(dict.fromkeys(fields + [key for row in items for key in row]))
            with (destination / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                # Keep original cells in external_csv_snapshots.json; escape spreadsheet formulas in CSV text.
                writer.writerows({key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value
                                  for key, value in row.items()} for row in items)
        (destination / "verified_gt_candidate.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates), encoding="utf-8")
        (destination / "ground_truth_review_summary.json").write_text(json.dumps(summary(self.repository, captured=(reviews, adjudications)), indent=2), encoding="utf-8")
        (destination / "external_csv_snapshots.json").write_text(json.dumps(snapshots, indent=2, ensure_ascii=False), encoding="utf-8")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repository.settings.repo, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=self.repository.settings.repo, capture_output=True, text=True)
        with self.repository.connection() as db:
            identities = [dict(row) for row in db.execute("SELECT id,person_id,role,identity_ref FROM users")]
        provenance = {"created_at_utc": utc_now(), "exported_by": actor["id"], "frozen_snapshot_sha256": snapshot_hash,
                      "source_commit": git.stdout.strip() if git.returncode == 0 else None, "registered_identity_references": identities,
                      "source_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                      "studio_source_sha256": {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")},
                      "backup": str(backup), "review_ids": [row["id"] for row in reviews],
                      "adjudication_ids": [row["id"] for row in adjudications.values()],
                      "identity_policy": "Unique operator-provisioned person IDs with evidence references; not biometric proof of independence",
                      "official_ground_truth": False, "operating_point_promoted": False,
                      "files": {path.name: sha256(path) for path in destination.iterdir()}}
        (destination / "ground_truth_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        with self.repository.connection() as db:
            db.execute("INSERT INTO exports VALUES(?,?,?)", (destination.name, json.dumps(provenance), utc_now()))
            self.repository.log(db, actor["id"], "EXPORT_CANDIDATE", destination.name, provenance)
        return destination
