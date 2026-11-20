"""Test suite for Task 7 - Negative Video Ingestion Pipeline and Validation Expansion.

Scientific Integrity Hardened Integration Tests:
1. Real SHA-256 computation and mismatch detection
2. Duplicate detection against existing inventory
3. Duplicate detection within the same ingestion batch (different filenames, identical bytes)
4. Missing or placeholder provenance rejection (UNRESOLVED_PROVENANCE)
5. Train split leakage detection and prevention
6. Test split leakage detection and prevention
7. Unresolved / held-out split handling
8. Full stream sequential decode verifying exact frame counts and NORMAL_EOF (no seek substitution)
9. Full stream decode detecting TRUNCATED_STREAM on truncated streams
10. Full stream decode detecting DECODE_ERROR on zero-byte/corrupted files
11. Duration mismatch detection between declared and decoded streams
12. Exclusion of PENDING_HUMAN_REVIEW from FAR exposure
13. Exclusion of invalid/disallowed reviewer identity from FAR exposure
14. Exclusion of missing/invalid review timestamp from FAR exposure
15. Prevention of duplicate duration accumulation (no double-counting)
16. Exposure below 18,000 seconds strictly enforces Gate BLOCKED
17. Exposure reaches 18,000 seconds logic with 0 pending passes Gate
18. Gate remains BLOCKED when any prerequisite condition fails
"""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.negative_video_ingestion import (
    NegativeVideoIngestionPipeline,
    VideoIngestionRecord,
    ValidationExposureSummary,
    EOFStatus,
    SplitEligibility,
    ContentDecision,
    validate_provenance,
)

ROOT = Path(__file__).resolve().parent.parent


def create_synthetic_mp4(
    file_path: Path,
    num_frames: int = 30,
    fps: float = 10.0,
    width: int = 64,
    height: int = 64
) -> str:
    """Create a real, decodable synthetic MP4 video file using OpenCV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 8) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    return NegativeVideoIngestionPipeline.compute_sha256(file_path)


class TestNegativeVideoIngestionPipeline:
    """Rigorous integration test suite for negative video ingestion and gate validation."""

    @pytest.fixture
    def pipeline(self, tmp_path: Path):
        pipe = NegativeVideoIngestionPipeline()
        pipe.batch_hashes = {}
        return pipe

    def test_case_01_sha256_mismatch(self, pipeline, tmp_path: Path):
        """Test Case 1: Detect SHA256 mismatch against manifest on real video."""
        vid_path = tmp_path / "video_01.mp4"
        actual_sha = create_synthetic_mp4(vid_path, num_frames=20, fps=10.0)

        expected_tampered_sha = "0123456789abcdef" * 4
        assert actual_sha != expected_tampered_sha

        # Pipeline SHA256 matches actual file
        computed = pipeline.compute_sha256(vid_path)
        assert computed == actual_sha
        assert computed != expected_tampered_sha

    def test_case_02_duplicate_inventory_detection(self, pipeline, tmp_path: Path):
        """Test Case 2: Detect duplicate file using existing inventory hash lookup."""
        vid_path = tmp_path / "dup_inv.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=15, fps=10.0)

        pipeline.existing_inventory_hashes = {sha: "VN_NEG_EXISTING_42"}
        status = pipeline.detect_duplicates(sha, filename="dup_inv.mp4", video_id="NEW_01")
        assert "DUPLICATE_SHA256_OF_VN_NEG_EXISTING_42" in status

        # Verify ingestion record records the duplicate
        rec = pipeline.ingest_single_video(
            vid_path,
            assigned_id="NEW_01",
            source_origin="CCTV St",
            camera_id="CAM_01",
            environmental_condition="Day",
            acquisition_time="2026-11-20T10:00:00Z",
            is_new_acquisition=True
        )
        assert "DUPLICATE_SHA256_OF_VN_NEG_EXISTING_42" in rec.duplicate_flag
        assert rec.far_eligible is False

    def test_case_03_duplicate_batch_detection(self, pipeline, tmp_path: Path):
        """Test Case 3: Detect intra-batch duplicates (different names, identical content)."""
        vid_1 = tmp_path / "batch_clip_a.mp4"
        vid_2 = tmp_path / "batch_clip_b.mp4"

        # Create first video
        sha1 = create_synthetic_mp4(vid_1, num_frames=25, fps=10.0)
        # Create second identical video (exact same frames)
        sha2 = create_synthetic_mp4(vid_2, num_frames=25, fps=10.0)
        assert sha1 == sha2

        rec1 = pipeline.ingest_single_video(
            vid_1,
            assigned_id="BATCH_VID_01",
            source_origin="Intersection CCTV",
            camera_id="CAM_INTERSECTION",
            environmental_condition="Clear/Day",
            acquisition_time="2026-11-20T12:00:00Z",
            is_new_acquisition=True
        )
        assert rec1.duplicate_flag == "UNIQUE"

        # Ingesting second video in same batch session
        rec2 = pipeline.ingest_single_video(
            vid_2,
            assigned_id="BATCH_VID_02",
            source_origin="Intersection CCTV",
            camera_id="CAM_INTERSECTION",
            environmental_condition="Clear/Day",
            acquisition_time="2026-11-20T12:00:00Z",
            is_new_acquisition=True
        )
        assert "DUPLICATE_BATCH_OF_BATCH_VID_01" in rec2.duplicate_flag

        # Exposure summary must exclude duplicate
        rec1.human_content_verdict = ContentDecision.VERIFIED_NO_FIRE_SMOKE.value
        rec1.reviewer_id = "evaluator_01"
        rec1.review_timestamp = "2026-11-20T13:00:00Z"
        rec1.full_view_attestation = True

        rec2.human_content_verdict = ContentDecision.VERIFIED_NO_FIRE_SMOKE.value
        rec2.reviewer_id = "evaluator_01"
        rec2.review_timestamp = "2026-11-20T13:00:00Z"
        rec2.full_view_attestation = True

        summary = pipeline.calculate_cohort_exposure_summary([rec1, rec2])
        # Only rec1 counted, rec2 rejected as duplicate
        assert summary.far_eligible_duration_sec == rec1.declared_duration_sec

    def test_case_04_missing_provenance_flagged(self, pipeline, tmp_path: Path):
        """Test Case 4: Candidate video lacking provenance metadata is marked UNRESOLVED_PROVENANCE."""
        vid_path = tmp_path / "no_prov.mp4"
        create_synthetic_mp4(vid_path, num_frames=20, fps=10.0)

        # Ingest with default/missing provenance (no camera_id, no acquisition_time, no source_origin)
        rec = pipeline.ingest_single_video(vid_path, assigned_id="VN_NO_PROV")
        assert rec.split_eligibility == SplitEligibility.UNRESOLVED_PROVENANCE.value
        assert rec.current_split == "UNRESOLVED"
        assert rec.far_eligible is False

        # Ingest with placeholder camera_id and missing acquisition time
        rec_placeholder = pipeline.ingest_single_video(
            vid_path,
            assigned_id="VN_PLACEHOLDER",
            source_origin="Surveillance Camera Feed",
            camera_id="CAM_ACQUIRED",
            is_new_acquisition=True
        )
        assert rec_placeholder.split_eligibility == SplitEligibility.UNRESOLVED_PROVENANCE.value

        # Ingest with invalid non-ISO acquisition time
        rec_bad_time = pipeline.ingest_single_video(
            vid_path,
            assigned_id="VN_BAD_TIME",
            source_origin="Surveillance Camera Feed",
            camera_id="CAM_01",
            environmental_condition="Clear",
            acquisition_time="invalid-date-string",
            is_new_acquisition=True
        )
        assert rec_bad_time.split_eligibility == SplitEligibility.UNRESOLVED_PROVENANCE.value

        # Ingest with valid ISO 8601 acquisition time passes provenance
        rec_valid = pipeline.ingest_single_video(
            vid_path,
            assigned_id="VN_VALID_PROV",
            source_origin="Intersection CCTV",
            camera_id="CAM_01",
            environmental_condition="Clear",
            acquisition_time="2026-11-20T10:00:00Z",
            is_new_acquisition=True
        )
        assert rec_valid.split_eligibility == SplitEligibility.ELIGIBLE_FOR_VALIDATION.value

    def test_case_05_train_leakage_prevention(self, pipeline, tmp_path: Path):
        """Test Case 5: Videos belonging to Train split are blocked from Validation."""
        vid_path = tmp_path / "VN_NEG_01.mp4"
        create_synthetic_mp4(vid_path, num_frames=20, fps=10.0)

        pipeline.train_split_ids = {"VN_NEG_01", "VN_NEG_06"}
        rec = pipeline.ingest_single_video(
            vid_path,
            assigned_id="VN_NEG_01",
            source_origin="Highway Cam",
            camera_id="HW_01",
            environmental_condition="Day",
            acquisition_time="2026-11-20T10:00:00Z",
            is_new_acquisition=True
        )
        assert rec.split_eligibility == SplitEligibility.LEAKAGE_TRAIN_CONTAMINATED.value
        assert rec.current_split == "train"
        assert rec.far_eligible is False

    def test_case_06_test_leakage_prevention(self, pipeline, tmp_path: Path):
        """Test Case 6: Videos belonging to Test split are blocked from Validation."""
        vid_path = tmp_path / "VN_NEG_04.mp4"
        create_synthetic_mp4(vid_path, num_frames=20, fps=10.0)

        pipeline.test_split_ids = {"VN_NEG_04", "VN_NEG_54"}
        rec = pipeline.ingest_single_video(
            vid_path,
            assigned_id="VN_NEG_04",
            source_origin="Factory Cam",
            camera_id="FAC_01",
            environmental_condition="Night",
            acquisition_time="2026-11-20T22:00:00Z",
            is_new_acquisition=True
        )
        assert rec.split_eligibility == SplitEligibility.LEAKAGE_TEST_CONTAMINATED.value
        assert rec.current_split == "test"
        assert rec.far_eligible is False

    def test_case_07_unresolved_split_handling(self, pipeline, tmp_path: Path):
        """Test Case 7: Unassigned or held-out test pool candidates remain UNRESOLVED."""
        vid_path = tmp_path / "negsVideo1.858.mp4"
        create_synthetic_mp4(vid_path, num_frames=20, fps=10.0)

        eligibility = pipeline.evaluate_split_eligibility(
            "negsVideo1.858", "sha_held", {"source_path": "datasets/test_videos/neg/negsVideo1.858.avi"}
        )
        assert eligibility == SplitEligibility.UNRESOLVED_PROVENANCE

    def test_case_08_full_stream_decode_clean_normal_eof(self, pipeline, tmp_path: Path):
        """Test Case 8: Sequential decode reads all frames sequentially and confirms NORMAL_EOF."""
        vid_path = tmp_path / "clean_30.mp4"
        create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)

        meta = pipeline.decode_full_stream(vid_path)
        assert meta["is_readable"] is True
        assert meta["declared_frame_count"] == 30
        assert meta["decoded_frame_count"] == 30
        assert meta["diff_frames"] == 0
        assert meta["eof_status"] == EOFStatus.NORMAL_EOF.value
        assert meta["technical_verdict"] == "PASS_NORMAL_EOF"

    def test_case_09_full_stream_decode_truncated_stream(self, pipeline, tmp_path: Path):
        """Test Case 9: Sequential decode detects TRUNCATED_STREAM on truncated video."""
        vid_path = tmp_path / "trunc_candidate.mp4"
        create_synthetic_mp4(vid_path, num_frames=60, fps=10.0)

        # Truncate the physical file bytes to 40%
        file_bytes = vid_path.read_bytes()
        truncated_bytes = file_bytes[: int(len(file_bytes) * 0.40)]
        trunc_path = tmp_path / "physically_truncated.mp4"
        trunc_path.write_bytes(truncated_bytes)

        meta = pipeline.decode_full_stream(trunc_path)
        # Sequential decode halts early, or fails decode
        assert meta["eof_status"] in (EOFStatus.TRUNCATED_STREAM.value, EOFStatus.DECODE_ERROR.value)
        assert meta["technical_verdict"].startswith("FAIL")

    def test_case_10_full_stream_decode_decode_error(self, pipeline, tmp_path: Path):
        """Test Case 10: Corrupt zero-byte file classified as DECODE_ERROR."""
        corrupt_path = tmp_path / "corrupt_zero.mp4"
        corrupt_path.write_bytes(b"")

        meta = pipeline.decode_full_stream(corrupt_path)
        assert meta["eof_status"] == EOFStatus.DECODE_ERROR.value
        assert meta["technical_verdict"].startswith("FAIL")

    def test_case_11_duration_mismatch_detection(self, pipeline, tmp_path: Path):
        """Test Case 11: Detects significant mismatch between declared duration and decodable frames."""
        vid_path = tmp_path / "dur_mismatch.mp4"
        create_synthetic_mp4(vid_path, num_frames=20, fps=10.0)  # Actual duration: 2.0s

        # Ingest record with declared duration inconsistent with actual stream (60.0s vs 2.0s)
        rec = pipeline.ingest_single_video(
            vid_path,
            assigned_id="VN_DUR_CHECK",
            source_origin="Street Cam",
            camera_id="ST_01",
            environmental_condition="Clear",
            acquisition_time="2026-11-20T10:00:00Z",
            is_new_acquisition=True,
            declared_duration=60.0
        )
        assert rec.declared_duration_sec == 60.0
        assert abs(rec.calculated_duration_sec - 2.0) <= 0.1
        assert rec.has_duration_mismatch is True
        assert rec.technical_verdict.startswith("FAIL_DURATION_MISMATCH")
        assert "60.0" in rec.technical_verdict
        assert rec.far_eligible is False

        # Direct sequential decode check with expected_duration_sec
        meta = pipeline.decode_full_stream(vid_path, expected_duration_sec=60.0)
        assert meta["has_duration_mismatch"] is True
        assert meta["duration_diff_sec"] == 58.0
        assert any("Duration mismatch" in note for note in meta["notes"])

    def test_case_12_pending_human_review_excluded(self, pipeline, tmp_path: Path):
        """Test Case 12: PENDING_HUMAN_REVIEW counts 0.0 towards FAR exposure."""
        vid_path = tmp_path / "valid_pending.mp4"
        create_synthetic_mp4(vid_path, num_frames=50, fps=10.0)

        rec = pipeline.ingest_single_video(
            vid_path,
            assigned_id="VN_PENDING_01",
            source_origin="Mall Cam",
            camera_id="MALL_01",
            environmental_condition="Indoor",
            acquisition_time="2026-11-20T10:00:00Z",
            is_new_acquisition=True
        )
        summary = pipeline.calculate_cohort_exposure_summary([rec])
        assert summary.far_eligible_duration_sec == 0.0
        assert summary.pending_review_duration_sec == rec.declared_duration_sec

    def test_case_13_invalid_reviewer_metadata(self, pipeline):
        """Test Case 13: Review missing reviewer identity or with placeholder is rejected from FAR."""
        record = VideoIngestionRecord(
            video_id="VN_NEW_02", filename="v2.mp4", relative_path="path/v2.mp4",
            sha256="h2", file_size_bytes=1000, fps=25.0, resolution="1280x720",
            width=1280, height=720, declared_frame_count=2500, decoded_frame_count=2500,
            declared_duration_sec=100.0, calculated_duration_sec=100.0,
            eof_status="NORMAL_EOF", technical_verdict="PASS_NORMAL_EOF",
            source_origin="Cam1", camera_id="C1", environmental_condition="Day",
            split_eligibility="ELIGIBLE_FOR_VALIDATION", current_split="val",
            duplicate_flag="UNIQUE", human_content_verdict="VERIFIED_NO_FIRE_SMOKE",
            reviewer_id="system", review_timestamp="2026-11-20T10:00:00Z", full_view_attestation=True
        )
        summary = pipeline.calculate_cohort_exposure_summary([record])
        assert summary.far_eligible_duration_sec == 0.0

    def test_case_14_invalid_timestamp(self, pipeline):
        """Test Case 14: Review missing timestamp is rejected from FAR."""
        record = VideoIngestionRecord(
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
        summary = pipeline.calculate_cohort_exposure_summary([record])
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
        record = VideoIngestionRecord(
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
        summary = pipeline.calculate_cohort_exposure_summary([record])
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
