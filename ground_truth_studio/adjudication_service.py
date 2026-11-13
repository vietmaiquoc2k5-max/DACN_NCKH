"""Evidence-requiring human adjudication; independent submissions stay immutable."""
from __future__ import annotations

import json
import uuid
from typing import Any

from ground_truth_studio.models import compare_events, utc_now, validate_events
from ground_truth_studio.review_repository import ReviewRepository


class AdjudicationService:
    def __init__(self, repository: ReviewRepository):
        self.repository = repository

    def comparison(self, token: str, video_id: str) -> dict[str, Any]:
        """Expose both reviewers/history only after two locked independent submissions."""
        pair = self.repository.submitted_pair(token, video_id)
        historical = self.repository.historical(token, video_id)
        result = compare_events(pair[0]["payload"]["events"], pair[1]["payload"]["events"])
        if pair[0]["payload"].get("decision") != pair[1]["payload"].get("decision"):
            result["status"] = "DISPUTED"
        return {"reviews": pair, "historical": historical, "comparison": result,
                "historical_claims_reauthenticated": False}

    def decide(self, token: str, video_id: str, decision: str, reason: str, evidence_ref: str,
               final_events: list[dict[str, Any]], omissions: dict[str, str]) -> str:
        """Record an explicit, immutable decision without averaging disagreement away."""
        actor = self.repository.actor(token, {"adjudicator"})
        if decision not in {"ACCEPT", "DISPUTED", "REQUIRE_REVIEW", "REJECT"} or not reason.strip() or not evidence_ref.strip():
            raise ValueError("Adjudication requires an explicit action, reason and evidence")
        context = self.comparison(token, video_id)
        pair = context["reviews"]
        if pair[0]["source"]["source_sha256"] != pair[1]["source"]["source_sha256"]:
            raise ValueError("Source bytes differ across reviews")
        duration = min(review["source"]["duration"] for review in pair)
        validate_events(final_events, duration, submit=decision == "ACCEPT")
        if decision == "ACCEPT":
            source_hash = pair[0]["source"]["source_sha256"]
            from ground_truth_studio.config import confined
            from ground_truth_studio.integrity_service import sha256
            source = confined(self.repository.settings.repo, self.repository.video(video_id)["relative_path"])
            if not source.is_file() or sha256(source) != source_hash:
                raise ValueError("Source changed or is missing at adjudication")
            if any(event.get("source_sha256") != source_hash for event in final_events):
                raise ValueError("Final events must retain the reviewed source hash")
            missing = compare_events(context["historical"], final_events)["unmatched_first"]
            for index in missing:
                historical_id = context["historical"][index]["historical_event_id"]
                if not omissions.get(historical_id, "").strip():
                    raise ValueError(f"Explicit disagreement/evidence required for missing historical event {historical_id}")
        record_id = str(uuid.uuid4())
        data = {"reason": reason, "evidence_ref": evidence_ref, "final_events": final_events,
                "historical_omissions": omissions, "comparison": context["comparison"],
                "review_ids": [review["id"] for review in pair], "source_sha256": pair[0]["source"]["source_sha256"],
                "reviewer_ids": [review["user_id"] for review in pair],
                "independence_basis": "Operator-provisioned distinct person IDs; external identity evidence still human responsibility"}
        with self.repository.connection() as db:
            db.execute("INSERT INTO adjudications VALUES(?,?,?,?,?,?)",
                       (record_id, video_id, actor["id"], decision, json.dumps(data, ensure_ascii=False), utc_now()))
            self.repository.log(db, actor["id"], "ADJUDICATION", video_id, data)
        return record_id
