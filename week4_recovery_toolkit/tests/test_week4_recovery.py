"""Synthetic unit tests for non-destructive recovery and evidence gates."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from sys import path as sys_path

sys_path.insert(0, str(Path(__file__).resolve().parents[1]))
from week4_recovery import (cmd_prepare_reviews, locate_videos, manifest_entries,
                            safe_target, sha256_file, verified_negative_ids, verified_positive_events,
                            read_csv)


class TestWeek4Recovery(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        (self.root / "data" / "ground_truth").mkdir(parents=True)
        self.data = b"real-original-synthetic-test-video\x00\x01"
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.entry = {"video_id": "negative_01", "relative_path": "datasets/test_videos/neg/sample.avi",
                      "sha256": self.digest, "file_size_bytes": len(self.data), "duration_seconds": 4.0}
        (self.root / "data/negative_videos_manifest.json").write_text(
            json.dumps({"videos": [self.entry]}), encoding="utf-8")
        (self.root / "data/ground_truth/events.jsonl").write_text(
            json.dumps({"video_id": "POS_01", "label": "FIRE", "start_sec": 1.0,
                        "end_sec": 3.0, "annotator_01": "A", "verification_status": "pending_verification"}) + "\n", encoding="utf-8")
        self.search = self.root / "external_media"
        self.search.mkdir()
        self.original = self.search / "sample.avi"
        self.original.write_bytes(self.data)
        self.review = self.root / "reviews"

    def test_hash_matches_manifest(self) -> None:
        self.assertEqual(sha256_file(self.original), self.digest)
        self.assertEqual(len(manifest_entries(self.root)), 1)

    def test_locator_does_not_touch_sources(self) -> None:
        rows = locate_videos(self.root, manifest_entries(self.root), [self.search])
        self.assertEqual(rows[0]["status"], "FOUND_ELSEWHERE")
        self.assertFalse((self.root / self.entry["relative_path"]).exists())
        self.assertEqual(self.original.read_bytes(), self.data)

    def test_restore_is_explicit_and_hash_verified(self) -> None:
        rows = locate_videos(self.root, manifest_entries(self.root), [self.search], recover="copy")
        self.assertEqual(rows[0]["status"], "RESTORED_VERIFIED")
        self.assertEqual(sha256_file(self.root / self.entry["relative_path"]), self.digest)
        self.assertEqual(self.original.read_bytes(), self.data)

    def test_hash_mismatch_fails_closed(self) -> None:
        (self.root / self.entry["relative_path"]).parent.mkdir(parents=True)
        (self.root / self.entry["relative_path"]).write_bytes(b"invalid")
        rows = locate_videos(self.root, manifest_entries(self.root), [self.search], recover="copy")
        self.assertEqual(rows[0]["status"], "HASH_MISMATCH")
        self.assertEqual((self.root / self.entry["relative_path"]).read_bytes(), b"invalid")

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            safe_target(self.root, "../evil")

    def test_review_forms_never_overwritten(self) -> None:
        args = Namespace(repo=self.root, review_dir=self.review)
        self.assertEqual(cmd_prepare_reviews(args), 0)
        form = self.review / "negative_annotator_01.csv"
        content = form.read_bytes()
        modified = content + b"\n# Manual reviewer notes\n"
        form.write_bytes(modified)
        self.assertEqual(cmd_prepare_reviews(args), 0)
        self.assertEqual(form.read_bytes(), modified)

    def test_blank_gt_never_verified(self) -> None:
        cmd_prepare_reviews(Namespace(repo=self.root, review_dir=self.review))
        negative_ids, negative_reasons = verified_negative_ids(self.root, self.review)
        pos, pos_reasons = verified_positive_events(self.root, self.review)
        self.assertEqual(negative_ids, set())
        self.assertEqual(pos, [])
        self.assertIn("negative_01", negative_reasons)
        self.assertIn("1", pos_reasons)

    def test_signed_negative_requires_adjudication_and_source(self) -> None:
        cmd_prepare_reviews(Namespace(repo=self.root, review_dir=self.review))
        for n in (1, 2):
            file = self.review / f"negative_annotator_0{n}.csv"
            rows = read_csv(file)
            rows[0].update(reviewer_id=f"human_{n}", reviewed_at_utc="2026-11-13T05:00:00Z",
                           decision="NEGATIVE", evidence_ref=f"review-log-{n}", signoff="YES")
            with file.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        neg, why = verified_negative_ids(self.root, self.review)
        self.assertNotIn("negative_01", neg)
        self.assertEqual(why["negative_01"], "HUMAN_ADJUDICATION_PENDING")
        adj_path = self.review / "negative_adjudication.csv"
        rows = read_csv(adj_path)
        rows[0].update(approval="APPROVE", adjudicator_id="human_1",
                       adjudicated_at_utc="2026-11-13T07:00:00Z", evidence_ref="independent-consensus")
        with adj_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        neg, why = verified_negative_ids(self.root, self.review)
        self.assertEqual(why["negative_01"], "SOURCE_VIDEO_MISSING_OR_HASH_MISMATCH")
        target = self.root / self.entry["relative_path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(self.data)
        neg, why = verified_negative_ids(self.root, self.review)
        self.assertEqual(neg, {"negative_01"})


if __name__ == "__main__":
    unittest.main()

class TestEvaluationFailClosed(unittest.TestCase):
    def test_unreviewed_gt_reports_blocked_not_zero_metrics(self) -> None:
        from week4_recovery import cmd_evaluate, cmd_prepare_reviews
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "data/ground_truth").mkdir(parents=True)
            (root / "data/negative_videos_manifest.json").write_text(json.dumps({"videos": [{
                "video_id": "N", "relative_path": "datasets/N.mp4",
                "sha256": hashlib.sha256(b"x").hexdigest(), "file_size_bytes": 1,
            }]}), encoding="utf-8")
            (root / "data/ground_truth/events.jsonl").write_text(json.dumps({
                "video_id": "P", "label": "FIRE", "start_sec": 0.0, "end_sec": 1.0,
                "camera_id": "CAM_P", "annotator_01": "a1",
            }) + "\n", encoding="utf-8")
            folder = root / "reviews"
            cmd_prepare_reviews(Namespace(repo=root, review_dir=folder))
            audit_path = root / "runs/benchmarks/week4_final_recovery/video_audit_final.json"
            audit_path.parent.mkdir(parents=True)
            audit_path.write_text(json.dumps({"videos": []}), encoding="utf-8")
            args = Namespace(repo=root, review_dir=folder, output=root / "reports",
                             reuse_report=None, validation_lock=None)
            self.assertEqual(cmd_evaluate(args), 2)
            output = json.loads((root / "reports/week4_evaluation_readiness.json").read_text(encoding="utf-8"))
            self.assertIsNone(output["far_per_hour"])
            self.assertIsNone(output["event_recall"])
            self.assertIsNone(output["detection_delay_sec"])
            self.assertEqual(output["scientific_acceptance"], "BLOCKED")
