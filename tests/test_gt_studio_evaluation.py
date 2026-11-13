"""Evidence gates use synthetic submitted reviews and independently bound logs."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ground_truth_studio.inference_reconciliation import classify_reuse
from ground_truth_studio.provisional_evaluation import evaluate_provisional
from src.inference_frame_evidence import inspect_frame_jsonl


class TestProvisionalEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.digest = "a" * 64
        base = {"review_status": "SINGLE_HUMAN_REVIEWED_PROVISIONAL", "source_sha256": self.digest,
                "review_id": "synthetic-review", "reviewer_person_id": "SYNTHETIC_ONLY", "submitted_at_utc": "2026-11-13T00:00:00Z",
                "attest_full_view": True, "evidence_ref": "SYNTHETIC_FIXTURE", "is_primary_round_for_video": True,
                "source_timestamp_provenance": "SOURCE_PTS", "duration_seconds": 3600.0, "events": [], "uncertain_intervals": []}
        self.reviews = [dict(base, video_id="VN_NEG_01", kind="negative", decision="VERIFIED_NEGATIVE"),
                        dict(base, video_id="VN_POS_01", kind="positive_candidate", decision=None,
                             events=[{"label": "FIRE", "start_sec": 10., "end_sec": 20., "source_sha256": self.digest, "timestamp_provenance": "SOURCE_PTS"}])]
        self.audits = {row["video_id"]: {"sha256": self.digest, "sha256_verified": True, "full_decode_verified": True, "decode_status": "NORMAL_EOF", "full_decoded_frames": 3, "fps": 1/1200, "duration_seconds": 3600.} for row in self.reviews}
        self.pipelines = {}
        for name in ("A", "B"):
            path = Path(self.temp.name) / (name + ".jsonl")
            rows = [{"video_id": row["video_id"], "run_id": "fixture-" + name, "source_sha256": self.digest,
                     "checkpoint_sha256": "b" * 64, "timestamp_source": "source_pts", "detector_input_config": {"imgsz": 416},
                     "frame_index": index, "timestamp_sec": float(index * 1200), "detections": []}
                    for row in self.reviews for index in range(3)]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            events = [{"video_id": "VN_POS_01", "camera_id": "BENCH_VN_POS_01", "label": "FIRE", "start_sec": 11., "end_sec": 12.},
                      {"video_id": "VN_NEG_01", "camera_id": "BENCH_VN_NEG_01", "label": "SMOKE", "start_sec": 30., "end_sec": 31.},
                      {"video_id": "VN_NEG_01", "camera_id": "BENCH_VN_NEG_01", "label": "SMOKE", "start_sec": 32., "end_sec": 33.}]
            self.pipelines[name] = {"run_id": "fixture-" + name, "checkpoint_sha256": "b" * 64, "detector_config": {"imgsz": 416},
                "frame_audits": [inspect_frame_jsonl(path)], "merge_gap_sec": 2., "events": events, "sampling_mode": "SOURCE_NATIVE_STRIDE_5_FPS",
                "reference_timestamps": {row["video_id"]: {index: float(index * 1200) for index in range(3)} for row in self.reviews},
                "expected_samples": {row["video_id"]: 3 for row in self.reviews},
                "covered_intervals": {row["video_id"]: [0., 3600.] for row in self.reviews}}

    def test_no_gt_or_inference_returns_null_not_zero(self):
        result = evaluate_provisional([], {}, {})
        self.assertEqual(result["status"], "PARTIAL_INSUFFICIENT_VERIFIED_EXPOSURE")
        self.assertEqual(result["human_review_status"], "HUMAN_REVIEW_PENDING")
        self.assertIsNone(result["pipeline_metrics"]["A"]["far_per_hour"])
        self.assertIsNone(result["pipeline_metrics"]["B"]["event_recall"])

    def test_shared_scope_episode_counts_and_exact_ci(self):
        result = evaluate_provisional(self.reviews, self.audits, self.pipelines)
        self.assertEqual(len(result["scope"]), 2)
        self.assertEqual(result["verified_negative_hours"], 1.)
        for metrics in result["pipeline_metrics"].values():
            self.assertEqual(metrics["event_recall"], 1.)
            self.assertEqual(metrics["detection_delay_sec"], 1.)
            self.assertEqual(metrics["far_per_hour"], 1.)
            self.assertLess(metrics["poisson_ci_95"][0], 1.)
            self.assertGreater(metrics["poisson_ci_95"][1], 1.)
        self.assertFalse(result["official"])

    def test_unverified_audit_missing_pts_and_one_pipeline_exclude_both(self):
        for change in ("audit", "pts", "interval", "pipeline", "incomplete_pts", "source_hash"):
            audits, pipelines = copy.deepcopy(self.audits), copy.deepcopy(self.pipelines)
            if change == "audit":
                audits["VN_NEG_01"]["decode_status"] = "NEEDS_REVIEW"
            elif change == "pts":
                pipelines["B"]["reference_timestamps"] = {}
            elif change == "interval":
                pipelines["B"]["covered_intervals"]["VN_NEG_01"] = [0., 3599.]
            elif change == "incomplete_pts":
                pipelines["B"]["reference_timestamps"]["VN_NEG_01"].pop(2)
            elif change == "source_hash":
                audits["VN_NEG_01"]["sha256"] = "c" * 64
            else:
                pipelines.pop("B")
            result = evaluate_provisional(self.reviews, audits, pipelines)
            self.assertEqual(result["verified_negative_hours"], 0.)
            self.assertIsNone(result["pipeline_metrics"]["A"]["far_per_hour"])

    def test_candidate_negative_with_events_or_uncertainty_never_far(self):
        for changes in ({"kind": "positive_candidate"}, {"decision": "CONTAINS_FIRE", "events": self.reviews[1]["events"]}, {"uncertain_intervals": [{"start_sec": 1}]}):
            reviews = copy.deepcopy(self.reviews)
            reviews[0].update(changes)
            result = evaluate_provisional(reviews, self.audits, self.pipelines)
            self.assertEqual(result["verified_negative_hours"], 0.)

    def test_missing_metadata_does_not_prove_rerun_required(self):
        self.assertEqual(classify_reuse(available=True, contradiction=False, complete_evidence=False), "UNVERIFIED_MISSING_EVIDENCE")
        self.assertEqual(classify_reuse(available=False, contradiction=True, complete_evidence=False), "BLOCKED_MISSING_INPUT")
        self.assertEqual(classify_reuse(available=True, contradiction=True, complete_evidence=False), "RERUN_REQUIRED")
        self.assertEqual(classify_reuse(available=True, contradiction=False, complete_evidence=True), "REUSE_VERIFIED")
