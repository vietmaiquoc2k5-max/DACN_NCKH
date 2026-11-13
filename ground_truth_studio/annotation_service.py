"""Human-owned drafts, resumable sessions and deliberate locked submissions."""
from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from ground_truth_studio.integrity_service import frozen
from ground_truth_studio.models import validate_events
from ground_truth_studio.review_repository import ReviewRepository
from ground_truth_studio.video_service import VideoService


class AnnotationService:
    def __init__(self, repository: ReviewRepository, videos: VideoService):
        self.repository = repository
        self.videos = videos

    def save(self, token: str, review_id: str, revision: int, payload: dict[str, Any], *, submit: bool = False) -> dict[str, Any]:
        """Validate source and annotations before an atomic draft/submission write."""
        review = self.repository.review(token, review_id)
        video = self.repository.video(review["video_id"])
        metadata = self.videos.inspect(video)
        if review["source"] and review["source"]["source_sha256"] != metadata["source_sha256"]:
            raise ValueError("Source changed since draft; retain old review and resolve source conflict")
        value = copy.deepcopy(payload)
        validate_events(value.get("events", []), metadata["duration"], submit=submit)
        validate_events(value.get("uncertain_intervals", []), metadata["duration"], submit=submit, uncertain=True)
        if video["kind"] == "negative":
            decision = value.get("decision")
            if submit and decision not in {"VERIFIED_NEGATIVE", "CONTAINS_FIRE", "CONTAINS_SMOKE", "UNCERTAIN"}:
                raise ValueError("Choose an explicit semantic decision")
            labels = {event["label"] for event in value.get("events", [])}
            if decision == "VERIFIED_NEGATIVE" and (labels or value.get("uncertain_intervals")):
                raise ValueError("Negative decision contradicts fire/smoke/uncertain intervals")
            if submit and decision in {"CONTAINS_FIRE", "CONTAINS_SMOKE"} and decision.removeprefix("CONTAINS_") not in labels:
                raise ValueError("A positive interval is required for this decision")
        if submit:
            frozen(self.repository.settings)
            if not value.get("evidence_ref", "").strip() or value.get("attest_full_view") is not True:
                raise ValueError("Explicit full-source-view attestation and evidence reference required")
            # Rehash source at signoff; a stat-cache alone is not scientific integrity evidence.
            from ground_truth_studio.config import confined
            from ground_truth_studio.integrity_service import sha256
            if sha256(confined(self.repository.settings.repo, video["relative_path"])) != metadata["source_sha256"]:
                raise ValueError("Source bytes changed before submission")
            for event in value.get("events", []) + value.get("uncertain_intervals", []):
                if event.get("source_sha256") != metadata["source_sha256"]:
                    raise ValueError("Reconfirm imported/onset/offset frames on the current source")
                if event["frame_end"] >= metadata["frame_count"]:
                    raise ValueError("Event frame outside decoded source")
        # Persist lightweight metadata; the potentially large PTS array lives in the video service.
        source = {key: item for key, item in metadata.items() if key != "source_pts"}
        return self.repository.save(token, review_id, revision, value, source, submit=submit)

    def import_own_csv_draft(self, token: str, review_id: str, revision: int) -> dict[str, Any]:
        """Explicitly restore only this account's CSV as a draft, never as verification."""
        actor = self.repository.actor(token)
        review = self.repository.review(token, review_id)
        video = self.repository.video(review["video_id"])
        name = "positive_annotator_02.csv" if video["kind"] != "negative" else f"negative_{actor['role']}.csv"
        row = next((row for row in self.repository.own_csv_rows(token, name) if row["video_id"] == video["video_id"]), None)
        if row is None:
            raise ValueError("No external CSV row belonging to this authenticated reviewer")
        metadata = self.videos.inspect(video)
        if row.get("source_sha256") != metadata["source_sha256"]:
            raise ValueError("External CSV source hash is missing or differs; manual reconciliation required")
        raw_events = json.loads(row.get("annotation_events_json") or "[]") if video["kind"] != "negative" else []
        if not isinstance(raw_events, list) or any(not isinstance(event, dict) for event in raw_events):
            raise ValueError("External annotation_events_json must contain event objects")
        if video["kind"] == "negative":
            for label, key in (("FIRE", "fire_intervals_json"), ("SMOKE", "smoke_intervals_json")):
                spans = json.loads(row.get(key) or "[]")
                if not isinstance(spans, list):
                    raise ValueError("External intervals must be a JSON list")
                for span in spans:
                    if not isinstance(span, dict) and not (isinstance(span, list) and len(span) == 2):
                        raise ValueError("Malformed external interval")
                    raw_events.append(dict(span, label=label) if isinstance(span, dict) else {"label": label, "start_sec": span[0], "end_sec": span[1]})
        events = [dict(event, event_id=event.get("event_id") or str(uuid.uuid4()), frame_start=None, frame_end=None,
                       source_sha256=metadata["source_sha256"], timestamp_provenance="IMPORTED_UNVERIFIED") for event in raw_events]
        raw_uncertain = json.loads(row.get("uncertain_intervals_json") or "[]")
        if not isinstance(raw_uncertain, list) or any(not isinstance(event, dict) for event in raw_uncertain):
            raise ValueError("External uncertain intervals must contain event objects")
        uncertain = [dict(event, label="UNCERTAIN", event_id=event.get("event_id") or str(uuid.uuid4()), frame_start=None, frame_end=None,
                          source_sha256=metadata["source_sha256"], timestamp_provenance="IMPORTED_UNVERIFIED")
                     for event in raw_uncertain]
        payload = dict(review["payload"], events=events, evidence_ref=row.get("evidence_ref", ""),
                       notes="Imported own external row as draft; source frames need confirmation.",
                       uncertain_intervals=uncertain,
                       decision={"NEGATIVE": "VERIFIED_NEGATIVE", "FIRE": "CONTAINS_FIRE", "SMOKE": "CONTAINS_SMOKE", "UNCERTAIN": "UNCERTAIN"}.get(row.get("decision"), ""), attest_full_view=False)
        return self.save(token, review_id, revision, payload)
