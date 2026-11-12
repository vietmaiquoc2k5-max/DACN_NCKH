from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_week4_consistency_fix import (
    artifact_lineage_status,
    compare_reduction_claims,
    index_events,
    manifest_binding_status,
    operating_point_status,
    reconcile_historical_far_scopes,
    reconcile_inference,
    select_scope_b_ids,
    validate_audit_summary,
)
from src.video_audit import run_external_stream_decode_check


class Week4ConsistencyFixTests(unittest.TestCase):
    def test_ffmpeg_missing_is_not_success(self):
        with patch("scripts.run_week4_consistency_fix.shutil.which", return_value=None):
            # The production helper is in src.video_audit, patch its tool lookup.
            with patch("src.video_audit.shutil.which", return_value=None):
                result = run_external_stream_decode_check("not-opened.mp4")
        self.assertFalse(result["tool_available"])
        self.assertFalse(result["is_clean_decode"])

    def test_ffmpeg_decode_error_is_not_success(self):
        failed = subprocess.CompletedProcess(["ffmpeg"], 1, "frame=7\nprogress=continue\n", "Invalid NAL unit")
        with patch("src.video_audit.shutil.which", return_value="ffmpeg"), \
                patch("src.video_audit.subprocess.run", return_value=failed):
            result = run_external_stream_decode_check("corrupt.mp4")
        self.assertTrue(result["error_detected"])
        self.assertFalse(result["is_clean_decode"])
        self.assertEqual(result["exit_code"], 1)

    def test_long_video_timeout_is_explicit_failure(self):
        timeout = subprocess.TimeoutExpired(["ffmpeg"], 3600)
        with patch("src.video_audit.shutil.which", return_value="ffmpeg"), \
                patch("src.video_audit.subprocess.run", side_effect=timeout):
            result = run_external_stream_decode_check("long.mp4", timeout_sec=3600)
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("timed out", result["error_message"])
        self.assertFalse(result["is_clean_decode"])

    def test_wrong_container_count_cannot_enter_scope_b(self):
        audit = [{"video_id": "A", "audit_result_status": "STRICT_VALID", "manifest_sha256_match": True},
                 {"video_id": "B", "audit_result_status": "NEEDS_REVIEW", "manifest_sha256_match": True}]
        rows = {"A": {"status": "COMPLETED", "processed_sampled_frames": "10"},
                "B": {"status": "COMPLETED", "processed_sampled_frames": "10"}}
        self.assertEqual(select_scope_b_ids(audit, rows), {"A"})

    def test_inference_zero_samples_never_claim_100_percent_coverage(self):
        result = reconcile_inference(
            {"videos": [{"video_id": "A", "relative_path": "A.mp4"}]},
            {"run_id": "run-A", "config": {"sample_fps": 5}},
            [{"video_id": "A", "relative_path": "A.mp4", "status": "COMPLETED",
              "total_decoded_frames": "100", "processed_sampled_frames": "0", "actual_duration_sec": "20"}],
            [], [], [], "manifest-hash")
        self.assertIsNone(result["frame_totals_from_actual_csv_columns"]["coverage_100_percent"])
        self.assertEqual(result["frame_totals_from_actual_csv_columns"]["sampled_inference_frames"], 0)
        self.assertEqual(result["status"], "NEEDS_REVIEW")

    def test_historical_scope_b_far_reduction_disagreement_is_reported(self):
        self.assertEqual(compare_reduction_claims(57.05, 57.05, 56.48), "HISTORICAL_ARTIFACTS_DISAGREE")

    def test_historical_scope2_claim_is_row_paired_but_not_strictly_clean(self):
        result = reconcile_historical_far_scopes()
        scope1 = result["scopes"]["scope_1"]
        scope2 = result["scopes"]["scope_2"]
        self.assertEqual((scope1["video_count"], scope1["b2_alerts_from_same_rows"]), (79, 421))
        self.assertTrue(scope1["claimed_exposure_matches_per_video_rows"])
        self.assertEqual((scope2["video_count"], scope2["b2_alerts_from_same_rows"]), (77, 417))
        self.assertTrue(scope2["claimed_exposure_matches_per_video_rows"])
        self.assertEqual(scope2["needs_review_or_not_verified_rows"], 58)
        self.assertEqual(scope2["far_acceptance_status"], "REJECTED_SCOPE_CONTAINS_NEEDS_REVIEW_ROWS")

    def test_manifest_hash_difference_is_historical_not_match(self):
        self.assertEqual(manifest_binding_status("new", "old", "run-1"), "HISTORICAL_SNAPSHOT")

    def test_provenance_without_run_id_is_unverified(self):
        self.assertEqual(manifest_binding_status("same", "same", None), "UNVERIFIED_MISSING_RUN_ID")

    def test_artifact_created_from_historical_source_is_not_relabelled_current(self):
        self.assertEqual(artifact_lineage_status("artifact-commit", "source-commit", "run-1"),
                         "SOURCE_SNAPSHOT_DIFFERS_FROM_ARTIFACT_COMMIT")
        self.assertEqual(artifact_lineage_status("current-commit", "current-commit", "run-1"), "TRACEABLE")

    def test_operating_point_cannot_become_official_with_gt_pending(self):
        self.assertEqual(operating_point_status("OFFICIAL", 24), "PROVISIONAL")

    def test_audit_summary_cannot_claim_every_video_valid_when_review_exists(self):
        summary = {"strict_valid_video_count": 3,
                   "counts_by_audit_status": {"STRICT_VALID": 2, "NEEDS_REVIEW": 1}}
        self.assertEqual(validate_audit_summary(summary), "MISMATCH")

    def test_events_are_counted_by_unique_event_key_and_duplicates_are_visible(self):
        event = {"video_id": "A", "camera_id": "C", "label": "FIRE", "start_sec": 1.0, "end_sec": 2.0}
        counts, duplicates = index_events([event, dict(event)])
        self.assertEqual(counts["A"]["total"], 1)
        self.assertEqual(duplicates["A"], 1)


if __name__ == "__main__":
    unittest.main()
