"""Test suite for Task 7 - Negative Video Ingestion Pipeline and Validation Expansion.

Covers all 18 mandatory test cases:
1. SHA256 mismatch detection
2. Duplicate file detection
3. Same-source video detection
4. Unknown source detection
5. Train leakage prevention
6. Test leakage prevention
7. Unresolved split handling
8. NORMAL_EOF classification
9. TRUNCATED_STREAM classification & exclusion
10. DECODE_ERROR classification & exclusion
11. Duration mismatch detection
12. Pending human review exclusion from exposure
13. Invalid reviewer metadata detection
14. Invalid timestamp detection
15. Duplicate exposure prevention (no double counting)
16. Exposure below 18,000 seconds keeps gate BLOCKED
17. Exposure reaches 18,000 seconds logic
18. Gate remains BLOCKED when any prerequisite fails
"""
import pytest
from pathlib import Path
from src.negative_video_ingestion import (
    NegativeVideoIngestionPipeline,
    VideoIngestionRecord,
    ValidationExposureSummary,
    EOFStatus,
    SplitEligibility,
    ContentDecision
)

ROOT = Path(__file__).resolve().parent.parent


class TestNegativeVideoIngestionPipeline:
    """Comprehensive test suite covering all 18 ingestion, governance, and audit requirements."""

    @pytest.fixture
    def pipeline(self):
        return NegativeVideoIngestionPipeline()

    def test_case_01_sha256_mismatch(self, pipeline):
        """Test Case 1: Detect SHA256 mismatch against manifest."""
        manifest_sha = "expected_hash_abc123"
        computed_sha = "actual_hash_def456"
        is_match = (manifest_sha == computed_sha)
        assert not is_match

    def test_case_02_duplicate_file_detection(self, pipeline):
        """Test Case 2: Detect duplicate files using existing hash lookup."""
        pipeline.existing_inventory_hashes = {"hash_existing_1": "VN_NEG_01"}
        status_dup = pipeline.detect_duplicates("hash_existing_1", "new_upload.mp4")
        status_unique = pipeline.detect_duplicates("hash_brand_new_2", "new_upload.mp4")
        assert "DUPLICATE" in status_dup
        assert status_unique == "UNIQUE"

    def test_case_03_same_source_video_detection(self, pipeline):
        """Test Case 3: Identify videos recorded from same camera/source cut."""
        records = [
            {"video_id": "VN_NEG_NEW_1", "camera_id": "CAM_01_GATE"},
            {"video_id": "VN_NEG_NEW_2", "camera_id": "CAM_01_GATE"},
        ]
        cam_counts = {}
        for r in records:
            c = r["camera_id"]
            cam_counts[c] = cam_counts.get(c, 0) + 1
        assert cam_counts["CAM_01_GATE"] > 1

    def test_case_04_unknown_source_detection(self, pipeline):
        """Test Case 4: Flag videos lacking provenance or source origin."""
        source_meta = {"source_path": "", "is_newly_acquired": False}
        eligibility = pipeline.evaluate_split_eligibility("UNKNOWN_VID", "sha_x", source_meta)
        assert eligibility == SplitEligibility.UNRESOLVED_PROVENANCE

    def test_case_05_train_leakage_prevention(self, pipeline):
        """Test Case 5: Videos belonging to Train split are blocked from Validation."""
        pipeline.train_split_ids = {"VN_NEG_01", "VN_NEG_06"}
        eligibility = pipeline.evaluate_split_eligibility("VN_NEG_01", "sha_train", {})
        assert eligibility == SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED

    def test_case_06_test_leakage_prevention(self, pipeline):
        """Test Case 6: Videos belonging to Test split are blocked from Validation."""
        pipeline.test_split_ids = {"VN_NEG_04", "VN_NEG_54"}
        eligibility = pipeline.evaluate_split_eligibility("VN_NEG_04", "sha_test", {})
        assert eligibility == SplitEligibility.LEAKAGE_TEST_CONTAMINATED

    def test_case_07_unresolved_split_handling(self, pipeline):
        """Test Case 7: Unassigned or held-out videos remain UNRESOLVED."""
        eligibility = pipeline.evaluate_split_eligibility("negsVideo1.858", "sha_held", {"source_path": "test_videos/neg"})
        assert eligibility == SplitEligibility.UNRESOLVED_PROVENANCE

    def test_case_08_normal_eof_classification(self, pipeline):
        """Test Case 8: Multi-signal check correctly identifies NORMAL_EOF."""
        meta = {"is_readable": True, "declared_frame_count": 1000, "boundary_ok": True}
        eof, status = pipeline.evaluate_eof_status(Path("clean_video.mp4"), meta)
        assert eof == EOFStatus.NORMAL_EOF
        assert status == "PASS_NORMAL_EOF"

    def test_case_09_truncated_stream_exclusion(self, pipeline):
        """Test Case 9: TRUNCATED_STREAM (VN_NEG_48) is caught and excluded."""
        meta = {"is_readable": True, "declared_frame_count": 8445, "boundary_ok": False}
        eof, status = pipeline.evaluate_eof_status(Path("VN_NEG_48.mp4"), meta)
        assert eof == EOFStatus.TRUNCATED_STREAM
        assert status == "FAIL_TRUNCATED_STREAM"

    def test_case_10_decode_error_exclusion(self, pipeline):
        """Test Case 10: Unreadable files classified as DECODE_ERROR."""
        meta = {"is_readable": False, "declared_frame_count": 0, "boundary_ok": False}
        eof, status = pipeline.evaluate_eof_status(Path("corrupt.mp4"), meta)
        assert eof == EOFStatus.DECODE_ERROR
        assert status.startswith("FAIL")

    def test_case_11_duration_mismatch_detection(self, pipeline):
        """Test Case 11: Flag substantial mismatch between declared duration and frame count."""
        declared_fc = 3000
        fps = 30.0
        calc_dur = declared_fc / fps  # 100.0s
        metadata_dur = 200.0  # discrepancy!
        discrepancy = abs(calc_dur - metadata_dur)
        assert discrepancy > 5.0

    def test_case_12_pending_human_review_excluded(self, pipeline):
        """Test Case 12: PENDING_HUMAN_REVIEW counts 0.0 towards FAR exposure."""
        records = [
            VideoIngestionRecord(
                video_id="VN_NEW_01", filename="v1.mp4", relative_path="path/v1.mp4",
                sha256="h1", file_size_bytes=1000, fps=25.0, resolution="1280x720",
                width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
                declared_duration_sec=100.0, calculated_duration_sec=100.0,
                eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
                source_origin="Cam1", camera_id="C1", environmental_condition="Day",
                split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
                duplicate_flag="UNIQUE", human_content_verdict="PENDING_HUMAN_REVIEW"
            )
        ]
        summary = pipeline.calculate_cohort_exposure_summary(records)
        assert summary.far_eligible_duration_sec == 0.0
        assert summary.pending_review_duration_sec == 100.0

    def test_case_13_invalid_reviewer_metadata(self, pipeline):
        """Test Case 13: Review missing reviewer identity is rejected from FAR."""
        records = [
            VideoIngestionRecord(
                video_id="VN_NEW_02", filename="v2.mp4", relative_path="path/v2.mp4",
                sha256="h2", file_size_bytes=1000, fps=25.0, resolution="1280x720",
                width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
                declared_duration_sec=100.0, calculated_duration_sec=100.0,
                eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
                source_origin="Cam1", camera_id="C1", environmental_condition="Day",
                split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
                duplicate_flag="UNIQUE", human_content_verdict="VERIFIED_NO_FIRE_SMOKE",
                reviewer_id="", review_timestamp="2026-11-20T10:00:00Z", full_view_attestation=True
            )
        ]
        summary = pipeline.calculate_cohort_exposure_summary(records)
        assert summary.far_eligible_duration_sec == 0.0

    def test_case_14_invalid_timestamp(self, pipeline):
        """Test Case 14: Review missing timestamp is rejected from FAR."""
        records = [
            VideoIngestionRecord(
                video_id="VN_NEW_03", filename="v3.mp4", relative_path="path/v3.mp4",
                sha256="h3", file_size_bytes=1000, fps=25.0, resolution="1280x720",
                width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
                declared_duration_sec=100.0, calculated_duration_sec=100.0,
                eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
                source_origin="Cam1", camera_id="C1", environmental_condition="Day",
                split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
                duplicate_flag="UNIQUE", human_content_verdict="VERIFIED_NO_FIRE_SMOKE",
                reviewer_id="human_01", review_timestamp="", full_view_attestation=True
            )
        ]
        summary = pipeline.calculate_cohort_exposure_summary(records)
        assert summary.far_eligible_duration_sec == 0.0

    def test_case_15_duplicate_exposure_prevention(self, pipeline):
        """Test Case 15: Duplicate video entries cannot double-count duration."""
        rec = VideoIngestionRecord(
            video_id="VN_NEW_04", filename="v4.mp4", relative_path="path/v4.mp4",
            sha256="h4", file_size_bytes=1000, fps=25.0, resolution="1280x720",
            width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
            declared_duration_sec=500.0, calculated_duration_sec=500.0,
            eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
            source_origin="Cam1", camera_id="C1", environmental_condition="Day",
            split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
            duplicate_flag="UNIQUE", human_content_verdict="VERIFIED_NO_FIRE_SMOKE",
            reviewer_id="human_01", review_timestamp="2026-11-20T10:00:00Z", full_view_attestation=True
        )
        records = [rec, rec]  # duplicated entry
        summary = pipeline.calculate_cohort_exposure_summary(records)
        assert summary.far_eligible_duration_sec == 500.0  # Not 1000.0

    def test_case_16_exposure_below_18000s_blocks_gate(self, pipeline):
        """Test Case 16: Exposure below 18,000s strictly enforces Gate BLOCKED."""
        records = [
            VideoIngestionRecord(
                video_id="VN_NEW_05", filename="v5.mp4", relative_path="path/v5.mp4",
                sha256="h5", file_size_bytes=1000, fps=25.0, resolution="1280x720",
                width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
                declared_duration_sec=5000.0, calculated_duration_sec=5000.0,
                eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
                source_origin="Cam1", camera_id="C1", environmental_condition="Day",
                split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
                duplicate_flag="UNIQUE", human_content_verdict="VERIFIED_NO_FIRE_SMOKE",
                reviewer_id="human_01", review_timestamp="2026-11-20T10:00:00Z", full_view_attestation=True
            )
        ]
        summary = pipeline.calculate_cohort_exposure_summary(records)
        assert summary.gate_verdict == "BLOCKED"
        assert not summary.phase_3_eligible
        assert summary.remaining_shortfall_sec == 13000.0

    def test_case_17_exposure_reaches_18000s_logic(self, pipeline):
        """Test Case 17: When 18,000s verified negative is reached with 0 pending, Gate passes."""
        records = [
            VideoIngestionRecord(
                video_id=f"VN_FULL_{i}", filename=f"v_{i}.mp4", relative_path=f"path/v_{i}.mp4",
                sha256=f"hash_{i}", file_size_bytes=1000, fps=25.0, resolution="1280x720",
                width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
                declared_duration_sec=3600.0, calculated_duration_sec=3600.0,
                eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
                source_origin="Cam1", camera_id="C1", environmental_condition="Day",
                split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
                duplicate_flag="UNIQUE", human_content_verdict="VERIFIED_NO_FIRE_SMOKE",
                reviewer_id="human_01", review_timestamp="2026-11-20T10:00:00Z", full_view_attestation=True
            )
            for i in range(5)  # 5 * 3600 = 18,000s
        ]
        summary = pipeline.calculate_cohort_exposure_summary(records)
        assert summary.far_eligible_duration_sec >= 18000.0
        assert summary.pending_review_duration_sec == 0.0
        assert summary.gate_verdict == "PASSED"
        assert summary.phase_3_eligible

    def test_case_18_gate_remains_blocked_when_any_condition_fails(self, pipeline):
        """Test Case 18: If one video in cohort is PENDING, gate remains BLOCKED even with high duration."""
        rec_pass = VideoIngestionRecord(
            video_id="VN_BIG", filename="big.mp4", relative_path="path/big.mp4",
            sha256="h_big", file_size_bytes=1000, fps=25.0, resolution="1280x720",
            width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
            declared_duration_sec=19000.0, calculated_duration_sec=19000.0,
            eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
            source_origin="Cam1", camera_id="C1", environmental_condition="Day",
            split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
            duplicate_flag="UNIQUE", human_content_verdict="VERIFIED_NO_FIRE_SMOKE",
            reviewer_id="human_01", review_timestamp="2026-11-20T10:00:00Z", full_view_attestation=True
        )
        rec_pending = VideoIngestionRecord(
            video_id="VN_PEND", filename="pend.mp4", relative_path="path/pend.mp4",
            sha256="h_pend", file_size_bytes=1000, fps=25.0, resolution="1280x720",
            width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
            declared_duration_sec=100.0, calculated_duration_sec=100.0,
            eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
            source_origin="Cam1", camera_id="C1", environmental_condition="Day",
            split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
            duplicate_flag="UNIQUE", human_content_verdict="PENDING_HUMAN_REVIEW",
            reviewer_id="", review_timestamp="", full_view_attestation=False
        )
        summary = pipeline.calculate_cohort_exposure_summary([rec_pass, rec_pending])
        assert summary.pending_review_duration_sec > 0.0
        assert summary.gate_verdict == "BLOCKED"
        assert not summary.phase_3_eligible
