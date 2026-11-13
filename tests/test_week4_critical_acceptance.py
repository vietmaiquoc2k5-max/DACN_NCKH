"""Regression tests for scientific gates, scope matching and human-review retention."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import run_phases4_to_8_recovery as recovery
from src.week4_acceptance import (
    REQUIRED_REUSE_CHECKS, acceptance, preserve_review_csv, reuse_decision, strict_technical_eligible,
)


class TestCriticalAcceptance(unittest.TestCase):
    def test_missing_gt_cannot_pass_despite_large_reduction(self):
        result = acceptance(0.0, 99.0)
        self.assertTrue(result["alert_rate_reduction_target_met"])
        self.assertIsNone(result["far_reduction_verified"])
        self.assertFalse(result["operating_point_official_eligible"])
        self.assertEqual(result["week4_acceptance_status"], "PARTIAL")

    def test_complete_evidence_required_for_acceptance(self):
        self.assertFalse(acceptance(5, 60, True, True, True, True)["operating_point_official_eligible"])
        self.assertTrue(acceptance(5, 60, True, True, True, True, True)["operating_point_official_eligible"])
        self.assertFalse(acceptance(5, 60, False, True, True, True, True)["operating_point_official_eligible"])
        self.assertFalse(acceptance(float("inf"), 60, True, True, True, True, True)["operating_point_official_eligible"])

    def test_strict_scope_requires_every_evidence_field(self):
        row = {"decode_status": "NORMAL_EOF", "sha256_verified": True, "full_decode_verified": True,
               "full_decode_errors": 0, "inference_alignment_verified": True, "duration_seconds": 60.0}
        self.assertTrue(strict_technical_eligible(row))
        for status in ("NEEDS_REVIEW", "DECODE_ERROR", "TRUNCATED_STREAM", "UNKNOWN", "", "CANNOT_OPEN"):
            with self.subTest(status=status):
                self.assertFalse(strict_technical_eligible(dict(row, decode_status=status)))
        for field in row:
            with self.subTest(missing=field):
                missing = dict(row)
                missing.pop(field)
                self.assertFalse(strict_technical_eligible(missing))
        for value in (0, -1, float("nan"), float("inf")):
            self.assertFalse(strict_technical_eligible(dict(row, duration_seconds=value)))

    def test_reuse_decision_states(self):
        complete = {key: True for key in REQUIRED_REUSE_CHECKS}
        self.assertEqual(reuse_decision({"checkpoint": False})["recommendation"], "RERUN_REQUIRED")
        self.assertEqual(reuse_decision({"checkpoint": None}, True)["recommendation"], "BLOCKED_MISSING_INPUT")
        self.assertEqual(reuse_decision(complete)["recommendation"], "REUSE_VERIFIED")
        limited = reuse_decision(dict(complete, frame_timestamps=None))
        self.assertEqual(limited["recommendation"], "REUSE_WITH_LIMITATIONS")
        self.assertFalse(limited["full_timeline_independently_verified"])
        unknown = reuse_decision({"checkpoint": True, "sampling": None})
        self.assertEqual(unknown["recommendation"], "UNVERIFIED")
        self.assertIsNone(unknown["rerun_required"])
        self.assertEqual(reuse_decision({})["recommendation"], "UNVERIFIED")
        for field in REQUIRED_REUSE_CHECKS - {"frame_timestamps"}:
            missing = dict(complete)
            missing.pop(field)
            self.assertEqual(reuse_decision(missing)["recommendation"], "UNVERIFIED")

    def test_episodes_clipped_and_merged_under_common_protocol(self):
        events = [{"video_id": "v", "label": "FIRE", "start_sec": s, "end_sec": e}
                  for s, e in [(1, 1), (2, 2), (6, 11), (11, 12)]]
        result = recovery.normalize_episodes(events, {"v": 10})
        self.assertEqual([(r["start_sec"], r["end_sec"]) for r in result], [(1, 2), (6, 10)])
        for bad in [(float("nan"), 1), (3, 2), (-1, 1)]:
            with self.assertRaises(ValueError):
                recovery.normalize_episodes([dict(events[0], start_sec=bad[0], end_sec=bad[1])], {"v": 10})

    def test_scope_counts_identical_videos_and_intervals(self):
        a = [{"video_id": "v"}, {"video_id": "excluded"}]
        b = [{"video_id": "excluded"}]
        result = recovery.summarize_scope([{"video_id": "v", "duration_seconds": 3600.123456}],
                                          {"pipeline_a": a, "pipeline_b2": b}, True)
        self.assertEqual(result["pipelines"]["pipeline_a"]["event_count"], 1)
        self.assertEqual(result["pipelines"]["pipeline_b2"]["event_count"], 0)
        self.assertEqual(result["exposure_seconds"], 3600.123456)
        self.assertEqual(result["pipelines"]["pipeline_a"]["exposure_seconds"],
                         result["pipelines"]["pipeline_b2"]["exposure_seconds"])
        empty = recovery.summarize_scope([], {"pipeline_a": [], "pipeline_b2": []}, False)
        self.assertIsNone(empty["pipelines"]["pipeline_b2"]["alert_event_rate_per_hour"])

    def test_review_regeneration_preserves_partial_and_verified_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            existing = [{"event_id": "e1", "video_id": "v", "onset_sec": 1.0, "offset_sec": 2.0,
                         "annotator_1": "human-a", "annotator_2": "", "status": "PENDING",
                         "reviewed_at": "2026-11-13", "evidence_ref": "manual/source"},
                        {"event_id": "e2", "video_id": "v", "onset_sec": 3.0, "offset_sec": 4.0,
                         "annotator_1": "human-a", "annotator_2": "human-b", "status": "VERIFIED",
                         "reviewed_at": "2026-11-13", "evidence_ref": "independent/source"}]
            preserve_review_csv(path, existing, "event_id")
            before = path.read_bytes()
            regenerated = [dict(row, annotator_1="", annotator_2="", status="PENDING", reviewed_at="", evidence_ref="") for row in existing]
            preserve_review_csv(path, regenerated, "event_id")
            self.assertEqual(path.read_bytes(), before)
            new = dict(existing[0], event_id="e3")
            preserve_review_csv(path, regenerated + [new], "event_id")
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["event_id"] for row in rows], ["e1", "e2", "e3"])
            self.assertEqual(rows[1]["annotator_2"], "human-b")
            self.assertEqual(len(list(path.parent.glob("*.bak"))), 1)
            with self.assertRaisesRegex(ValueError, "Source changed"):
                preserve_review_csv(path, [dict(existing[0], onset_sec=1.5)], "event_id")

    def test_malformed_review_rejected_without_modifying_bytes(self):
        for content in ("video_id,status\nv,pending\n", "event_id,status\ne,pending\ne,verified\n",
                        "event_id,status\ne\n", "event_id,status\ne,pending,extra\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "review.csv"
                path.write_text(content)
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    preserve_review_csv(path, [{"event_id": "e", "status": "PENDING"}], "event_id")
                self.assertEqual(path.read_bytes(), before)

    def test_duplicate_video_mapping_rejected(self):
        with self.assertRaises(ValueError):
            recovery.keyed([{"video_id": "v"}, {"video_id": "v"}])

    def test_duration_uses_unrounded_frame_evidence(self):
        self.assertEqual(recovery.duration({"fps": 3, "full_decoded_frames": 10, "duration_seconds": 3.33}), 10 / 3)


if __name__ == "__main__":
    unittest.main()
