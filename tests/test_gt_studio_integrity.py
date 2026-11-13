"""Frozen data, external CSV conflicts, safe candidates and exposure eligibility."""
from __future__ import annotations

import json
import os
import csv
from ground_truth_studio.integrity_service import sha256

from gt_studio_fixtures import StudioCase
from ground_truth_studio.adjudication_service import AdjudicationService
from ground_truth_studio.config import Settings
from ground_truth_studio.export_service import ExportService, summary
from ground_truth_studio.review_repository import ConflictError, ReviewRepository


class TestStudioIntegrity(StudioCase):
    def test_frozen_mutation_blocks_start_and_private_path_cannot_be_source(self):
        path = self.root / "data/ground_truth/events_v1.1_snapshot.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "BLOCKED_GT_INTEGRITY"):
            ReviewRepository(self.settings)
        with self.assertRaises(ValueError):
            Settings(self.root, path.parent)

    def test_export_preserves_sources_and_never_promotes_pending(self):
        before = {p.name: p.read_bytes() for p in self.private.glob("*.csv")}
        snapshot = (self.root / "data/ground_truth/events_v1.1_snapshot.jsonl").read_bytes()
        result = ExportService(self.repository).export(self.tokens["owner"])
        # A SQLite context manager alone does not close a connection on Windows.
        # The completed backup must be immediately reopenable with no live writer.
        backup = next((self.private / "backups").glob("*.sqlite3"))
        renamed = backup.with_suffix(".checked.sqlite3")
        os.replace(backup, renamed)
        os.replace(renamed, backup)
        self.assertEqual((result / "verified_gt_candidate.jsonl").read_text(), "")
        self.assertFalse(json.loads((result / "ground_truth_provenance.json").read_text())["official_ground_truth"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.private.glob("*.csv")})
        self.assertEqual(snapshot, (self.root / "data/ground_truth/events_v1.1_snapshot.jsonl").read_bytes())

    def test_external_csv_change_blocks_export_without_overwrite(self):
        path = self.private / "positive_annotator_02.csv"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("POS_01,source,hash,second,time,[],reference,YES\n")
        before = path.read_bytes()
        with self.assertRaises(ConflictError):
            ExportService(self.repository).export(self.tokens["owner"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.repository.snapshot_csv(accept_external_changes=True), [])

    def test_existing_own_external_fire_review_restores_only_draft(self):
        path = self.private / "negative_annotator_02.csv"
        with path.open(encoding="utf-8") as stream:
            fields = csv.DictReader(stream).fieldnames
        event = {"event_id": "external-stable-id", "start_sec": .2, "end_sec": .8}
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"video_id": "NEG_01", "relative_path": "../untrusted/path", "source_sha256": sha256(self.negative),
                             "reviewer_id": "second", "reviewed_at_utc": "2026-11-13T00:00:00+00:00", "decision": "FIRE",
                             "fire_intervals_json": json.dumps([event]), "smoke_intervals_json": "[]", "evidence_ref": "SYNTHETIC_FIXTURE_ONLY", "signoff": "YES"})
        self.repository.snapshot_csv(accept_external_changes=True)
        review = self.repository.start_review(self.tokens["second"], "NEG_01")
        imported = self.annotations.import_own_csv_draft(self.tokens["second"], review["id"], 0)
        self.assertEqual(imported["status"], "DRAFT")
        self.assertEqual(imported["payload"]["decision"], "CONTAINS_FIRE")
        self.assertEqual(imported["payload"]["events"][0]["event_id"], "external-stable-id")
        self.assertFalse(imported["payload"]["attest_full_view"])
        with self.assertRaises(ValueError):
            self.annotations.save(self.tokens["second"], review["id"], imported["revision"], imported["payload"], submit=True)

    def test_human_adjudicated_candidate_is_schema_compatible_but_not_official(self):
        first = self.submit("first", events=[self.event()])
        self.submit("second", events=[self.event()])
        adjudication = AdjudicationService(self.repository)
        context = adjudication.comparison(self.tokens["judge"], "POS_01")
        omissions = {row["historical_event_id"]: "Synthetic fixture: resolve duplicate historical event explicitly" for row in context["historical"]}
        adjudication.decide(self.tokens["judge"], "POS_01", "ACCEPT", "Synthetic fixture decision", "synthetic evidence",
                            first["payload"]["events"], omissions)
        target = ExportService(self.repository).export(self.tokens["judge"])
        candidate = target / "verified_gt_candidate.jsonl"
        rows = [json.loads(line) for line in candidate.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verification_status"], "pending_verification")
        self.assertEqual(rows[0]["annotation_status"], "HUMAN_ADJUDICATED_CANDIDATE")
        from src.event_eval import load_ground_truth_events
        self.assertEqual(len(load_ground_truth_events(candidate, only_verified=False)), 1)
        self.assertEqual(len(load_ground_truth_events(candidate, only_verified=True)), 0)

    def test_negative_exposure_requires_semantic_and_technical_evidence(self):
        self.submit("first", "NEG_01", decision="VERIFIED_NEGATIVE")
        self.submit("second", "NEG_01", decision="VERIFIED_NEGATIVE")
        self.assertEqual(summary(self.repository)["eligible_negative_exposure_seconds"], 0)
        AdjudicationService(self.repository).decide(self.tokens["judge"], "NEG_01", "ACCEPT", "Synthetic consensus", "synthetic evidence", [], {})
        self.assertEqual(summary(self.repository)["semantic_negative_consensus_videos"], 1)
        self.assertEqual(summary(self.repository)["eligible_negative_exposure_seconds"], 0)
        video = self.repository.video("NEG_01")
        video["technical"] = {"decode_status": "NORMAL_EOF", "full_decode_verified": True, "full_decode_errors": 0,
                              "inference_alignment_verified": True, "duration_seconds": 2.}
        with self.repository.connection() as db:
            db.execute("UPDATE videos SET data=? WHERE id=?", (json.dumps(video), "NEG_01"))
        self.assertAlmostEqual(summary(self.repository)["eligible_negative_exposure_seconds"], 2., places=4)
        video["technical"]["decode_status"] = "NEEDS_REVIEW"
        with self.repository.connection() as db:
            db.execute("UPDATE videos SET data=? WHERE id=?", (json.dumps(video), "NEG_01"))
        self.assertEqual(summary(self.repository)["eligible_negative_exposure_seconds"], 0)
