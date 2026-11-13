"""Explicit independent rounds; prior consensus is retained without promotion."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ground_truth_studio.integrity_service import frozen, sha256
from ground_truth_studio.config import confined
from ground_truth_studio.models import compare_events
from ground_truth_studio.review_repository import ReviewRepository


class IndependentReview:
    """Use existing transactional storage and audit trail without schema changes."""

    def __init__(self, repository: ReviewRepository):
        self.repository = repository

    def start(self, token: str, video_id: str, *, joint_prior: bool = False) -> dict[str, Any]:
        """Back up storage and create an empty round after a deliberate human action."""
        repo = self.repository
        actor = repo.actor(token, {"annotator_01", "annotator_02"})
        repo.video(video_id)
        backup = repo.backup()
        action = "INDEPENDENT_REVIEW_ROUND_" + ("A" if actor["role"] == "annotator_01" else "B")
        with repo.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT r.* FROM review_heads h JOIN review_sessions r ON h.review_id=r.id WHERE h.video_id=? AND h.slot=?", (video_id, actor["role"])).fetchone()
            if previous and previous["user_id"] != actor["id"]:
                raise PermissionError("Review slot belongs to another account")
            if previous and db.execute("SELECT 1 FROM audit_log WHERE ref=? AND action=?", (previous["id"], action)).fetchone():
                return repo.decode_review(previous)
            review_id = str(uuid.uuid4())
            payload = {"events": [], "uncertain_intervals": [], "decision": "", "evidence_ref": "", "notes": "", "attest_full_view": False}
            db.execute("INSERT INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)", (review_id,video_id,actor["id"],actor["role"],"DRAFT",0,None,json.dumps(payload),None))
            db.execute("INSERT INTO review_heads VALUES(?,?,?) ON CONFLICT(video_id,slot) DO UPDATE SET review_id=excluded.review_id", (video_id,actor["role"],review_id))
            repo.log(db, actor["id"], action, review_id, {"prior_review_id": previous["id"] if previous else None,
                "prior_status": "JOINT_CONSENSUS_PRIOR" if joint_prior else "PRIOR_UNCLASSIFIED",
                "backup": backup.name, "human_independence_attestation_required_at_submit": True})
        return repo.review(token, review_id)

    def comparison(self, token: str, video_id: str) -> dict[str, Any]:
        """Expose two real independent submissions only after both are locked."""
        repo = self.repository
        actor = repo.actor(token, {"annotator_01","annotator_02","adjudicator","admin"})
        with repo.connection() as db:
            rows = db.execute("SELECT r.*,u.person_id,u.identity_ref FROM review_heads h JOIN review_sessions r ON h.review_id=r.id JOIN users u ON u.id=r.user_id WHERE h.video_id=? ORDER BY r.slot", (video_id,)).fetchall()
            if len(rows) != 2 or any(row["status"] != "SUBMITTED" for row in rows) or len({row["person_id"] for row in rows}) != 2:
                raise PermissionError("Two submitted rounds by distinct real people required")
            for row in rows:
                action = "INDEPENDENT_REVIEW_ROUND_" + ("A" if row["slot"] == "annotator_01" else "B")
                if not db.execute("SELECT 1 FROM audit_log WHERE action=? AND ref=?", (action,row["id"])).fetchone():
                    raise PermissionError("Prior consensus/submission is not an independent round")
            if actor["role"].startswith("annotator_") and actor["id"] not in {row["user_id"] for row in rows}:
                raise PermissionError("Not a reviewer of this pair")
        reviews = [repo.decode_review(row) for row in rows]
        if any(row['payload'].get('attest_independent') is not True or row['payload'].get('attest_full_view') is not True for row in reviews):
            raise PermissionError("Independent full-view attestations required")
        digests = {row["source"]["source_sha256"] for row in reviews}
        video = repo.video(video_id)
        if len(digests) != 1 or sha256(confined(repo.settings.repo,video["relative_path"])) not in digests:
            raise ValueError("Source hash mismatch")
        comparison = compare_events(reviews[0]["payload"]["events"], reviews[1]["payload"]["events"])
        if any(row["payload"].get("uncertain_intervals") for row in reviews) or reviews[0]["payload"].get("decision") != reviews[1]["payload"].get("decision"):
            comparison["status"] = "DISPUTED"
        return {"video_id":video_id,"reviews":reviews,"comparison":comparison,
            "status":"INDEPENDENT_DUAL_REVIEW_PENDING_ADJUDICATION","official":False,
            "negative_consensus_candidate": video["kind"]=="negative" and comparison["status"] != "DISPUTED" and all(row["payload"].get("decision")=="VERIFIED_NEGATIVE" and not row["payload"].get("events") and not row["payload"].get("uncertain_intervals") for row in reviews),
            "scientific_negative_hours": None}

    def export(self, token: str, video_id: str) -> Path:
        """Write a new private research bundle; no adjudication or signoff is invented."""
        result = self.comparison(token, video_id)
        result["frozen_sha256"] = frozen(self.repository.settings)[0]
        folder = self.repository.settings.private / "exports" / ("independent_dual_" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        (folder/"dual_review.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        (folder/"file_sha256.json").write_text(json.dumps({"dual_review.json":sha256(folder/"dual_review.json")}),encoding="utf-8")
        return folder
