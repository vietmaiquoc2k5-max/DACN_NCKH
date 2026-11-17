#!/usr/bin/env python3
"""Comprehensive test suite for Negative Video Technical Audit and FAR Exposure.

Verifies all 12 mandatory scientific integrity requirements:
1. Normal video identification (NORMAL_EOF, technical PASS).
2. Truncated video identification (TRUNCATED_STREAM, technical FAIL).
3. Video decode error identification (DECODE_ERROR, technical FAIL).
4. Unknown EOF identification (UNKNOWN_EOF, technical FAIL).
5. SHA-256 hash mismatch blocks technical verdict.
6. Missing/disallowed human reviewer identity prevents FAR eligibility.
7. Pending human content review blocks FAR exposure.
8. Video with fire or smoke hazard flagged is blocked from negative exposure.
9. Duplicate video IDs in manifest are handled safely and audited.
10. FAR exposure shortfall calculation (remaining_duration_to_5h_sec = max(0, 18000 - far_eligible)).
11. Duration mismatch between declared and decoded duration detected and recorded.
12. Ineligible technical video is strictly excluded from verified and FAR exposure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit_negative_video_technical_integrity import (
    CONTAINER_TOLERANCE_FRAMES,
    CONTAINER_TOLERANCE_PERCENT,
    DISALLOWED_REVIEWERS,
    EOFStatus,
    NegativeVideoTechnicalAuditor,
    TechnicalVerdict,
    ContentDecision,
    sha256_file,
)


def create_synthetic_mp4(
    file_path: Path,
    num_frames: int = 30,
    fps: float = 10.0,
    width: int = 64,
    height: int = 64
) -> str:
    """Create a lightweight synthetic MP4 video for fast testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create non-empty frame
        img = np.full((height, width, 3), (i * 7) % 255, dtype=np.uint8)
        out.write(img)
    out.release()
    return sha256_file(file_path)


@pytest.fixture
def test_env(tmp_path: Path):
    """Setup isolated test fixtures for auditor."""
    manifest_path = tmp_path / "negative_videos_manifest.json"
    split_manifest_path = tmp_path / "split_manifest.csv"
    video_dir = tmp_path / "videos"
    output_dir = tmp_path / "runs/audits"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Empty split manifest
    with open(split_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "video_id", "split", "category"])

    return {
        "manifest_path": manifest_path,
        "split_manifest_path": split_manifest_path,
        "video_dir": video_dir,
        "output_dir": output_dir,
    }


class TestNegativeVideoTechnicalAudit:
    """Test suite covering the 12 scientific video audit requirements."""

    def test_case_01_normal_video_normal_eof(self, test_env):
        """Case 1: A normal intact video is classified as NORMAL_EOF with PASS verdict."""
        vid_path = test_env["video_dir"] / "NORMAL_01.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)

        manifest_data = {
            "videos": [{
                "video_id": "NORMAL_01",
                "filename": "NORMAL_01.mp4",
                "sha256": sha,
                "fps": 10.0,
                "reported_frame_count": 30,
                "duration_seconds": 3.0,
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        records, summary = auditor.audit_all(probe_decode=True)

        assert len(records) == 1
        rec = records[0]
        assert rec.eof_status == EOFStatus.NORMAL_EOF.value
        assert rec.technical_verdict == TechnicalVerdict.PASS.value
        assert rec.file_exists is True
        assert rec.sha256_match is True
        assert rec.decoded_frame_count >= 28  # within standard container tolerance

    def test_case_02_truncated_video_truncated_stream(self, test_env):
        """Case 2: A stream that stops prematurely is classified as TRUNCATED_STREAM with FAIL."""
        vid_path = test_env["video_dir"] / "TRUNC_01.mp4"
        # Write only 15 frames, but manifest expects 60 frames
        sha = create_synthetic_mp4(vid_path, num_frames=15, fps=10.0)

        manifest_data = {
            "videos": [{
                "video_id": "TRUNC_01",
                "filename": "TRUNC_01.mp4",
                "sha256": sha,
                "fps": 10.0,
                "reported_frame_count": 60,  # Manifest expects 60 frames
                "duration_seconds": 6.0,
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )

        dec_fc, decl_fc, fps, w, h, last_pts, eof_status, notes = auditor.probe_video_stream(
            vid_path, expected_frames=60, expected_fps=10.0, expected_duration=6.0
        )
        assert eof_status == EOFStatus.TRUNCATED_STREAM.value
        assert decl_fc == 60
        assert dec_fc <= 16

        records, summary = auditor.audit_all(probe_decode=True)
        rec = records[0]
        assert rec.eof_status == EOFStatus.TRUNCATED_STREAM.value
        assert rec.technical_verdict == TechnicalVerdict.FAIL.value
        assert rec.eligible_for_far is False
        assert summary["audit_statistics"]["technical_fail_count"] == 1
        assert summary["duration_indices"]["verified_technical_duration_sec"] == 0.0

    def test_case_03_decode_error(self, test_env):
        """Case 3: Unreadable or corrupted video file classified as DECODE_ERROR with FAIL."""
        corrupt_path = test_env["video_dir"] / "CORRUPT_01.mp4"
        # Write random corrupt bytes
        corrupt_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00corrupt_payload_garbage")
        sha = sha256_file(corrupt_path)

        manifest_data = {
            "videos": [{
                "video_id": "CORRUPT_01",
                "filename": "CORRUPT_01.mp4",
                "sha256": sha,
                "fps": 10.0,
                "reported_frame_count": 30,
                "duration_seconds": 3.0,
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        records, summary = auditor.audit_all(probe_decode=True)

        assert len(records) == 1
        rec = records[0]
        assert rec.eof_status == EOFStatus.DECODE_ERROR.value
        assert rec.technical_verdict == TechnicalVerdict.FAIL.value
        assert rec.eligible_for_far is False

    def test_case_04_unknown_eof(self, test_env):
        """Case 4: Non-positive fps, resolution, or frames triggers UNKNOWN_EOF with FAIL."""
        vid_path = test_env["video_dir"] / "NORMAL_FOR_UNKNOWN.mp4"
        create_synthetic_mp4(vid_path, num_frames=10, fps=10.0)

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )

        class MockVideoCapture:
            def __init__(self, path):
                pass
            def isOpened(self):
                return True
            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS:
                    return 0.0  # Non-positive FPS
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 0
                return 0
            def release(self):
                pass

        import unittest.mock as mock
        with mock.patch("cv2.VideoCapture", MockVideoCapture):
            dec_fc, decl_fc, fps, w, h, last_pts, eof_status, notes = auditor.probe_video_stream(
                vid_path, expected_frames=0, expected_fps=0.0, expected_duration=0.0
            )
            assert eof_status == EOFStatus.UNKNOWN_EOF.value

    def test_case_05_sha256_mismatch_blocks_technical_verdict(self, test_env):
        """Case 5: Checksum mismatch triggers technical FAIL and blocks FAR."""
        vid_path = test_env["video_dir"] / "SHA_MISMATCH.mp4"
        actual_sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)
        fake_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        manifest_data = {
            "videos": [{
                "video_id": "SHA_MISMATCH",
                "filename": "SHA_MISMATCH.mp4",
                "sha256": fake_sha,  # Intentional mismatch
                "fps": 10.0,
                "reported_frame_count": 30,
                "duration_seconds": 3.0,
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        records, summary = auditor.audit_all(probe_decode=True)

        rec = records[0]
        assert rec.sha256_match is False
        assert rec.technical_verdict == TechnicalVerdict.FAIL.value
        assert "SHA256_MISMATCH" in rec.ineligibility_reasons
        assert rec.eligible_for_far is False

    def test_case_06_human_review_missing_prevents_far_eligibility(self, test_env):
        """Case 6: Missing or disallowed reviewer name blocks FAR eligibility."""
        vid_path = test_env["video_dir"] / "VAL_NO_REV.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)

        for bad_reviewer in ["", "SYSTEM", "AUTO", "null", "annotator_pending", "unknown"]:
            manifest_data = {
                "videos": [{
                    "video_id": "VAL_NO_REV",
                    "filename": "VAL_NO_REV.mp4",
                    "sha256": sha,
                    "fps": 10.0,
                    "reported_frame_count": 30,
                    "duration_seconds": 3.0,
                    "human_review_status": "VERIFIED_NEGATIVE",
                    "reviewer_name": bad_reviewer,
                    "review_timestamp": "2026-11-17T18:00:00Z",
                    "attest_full_video_viewed": "TRUE",
                }]
            }
            test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

            auditor = NegativeVideoTechnicalAuditor(
                manifest_path=test_env["manifest_path"],
                split_manifest_path=test_env["split_manifest_path"],
                video_dirs=[test_env["video_dir"]],
                output_dir=test_env["output_dir"]
            )
            records, _ = auditor.audit_all(probe_decode=True)
            assert records[0].eligible_for_far is False
            assert "INVALID_REVIEWER" in records[0].ineligibility_reasons

    def test_case_07_pending_human_review_blocks_far_exposure(self, test_env):
        """Case 7: Video in PENDING_HUMAN_REVIEW yields zero FAR exposure."""
        vid_path = test_env["video_dir"] / "PENDING_01.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)

        manifest_data = {
            "videos": [{
                "video_id": "PENDING_01",
                "filename": "PENDING_01.mp4",
                "sha256": sha,
                "fps": 10.0,
                "reported_frame_count": 30,
                "duration_seconds": 3.0,
                "human_review_status": "PENDING_HUMAN_REVIEW",
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        records, summary = auditor.audit_all(probe_decode=True)

        assert records[0].eligible_for_far is False
        assert "PENDING_HUMAN_CONTENT_REVIEW" in records[0].ineligibility_reasons
        assert summary["duration_indices"]["far_eligible_duration_sec"] == 0.0

    def test_case_08_fire_smoke_hazard_flagged_blocks_exposure(self, test_env):
        """Case 8: Video flagged with FIRE_PRESENT or SMOKE_PRESENT is rejected from FAR."""
        for hazard in ["FIRE_PRESENT", "SMOKE_PRESENT"]:
            vid_path = test_env["video_dir"] / f"{hazard}.mp4"
            sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)

            manifest_data = {
                "videos": [{
                    "video_id": hazard,
                    "filename": f"{hazard}.mp4",
                    "sha256": sha,
                    "fps": 10.0,
                    "reported_frame_count": 30,
                    "duration_seconds": 3.0,
                    "human_review_status": hazard,
                    "reviewer_name": "researcher_01",
                    "review_timestamp": "2026-11-17T18:00:00Z",
                    "attest_full_video_viewed": "TRUE",
                }]
            }
            test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

            auditor = NegativeVideoTechnicalAuditor(
                manifest_path=test_env["manifest_path"],
                split_manifest_path=test_env["split_manifest_path"],
                video_dirs=[test_env["video_dir"]],
                output_dir=test_env["output_dir"]
            )
            records, summary = auditor.audit_all(probe_decode=True)

            assert records[0].eligible_for_far is False
            assert f"HAZARD_PRESENT_{hazard}" in records[0].ineligibility_reasons
            assert summary["duration_indices"]["far_eligible_duration_sec"] == 0.0

    def test_case_09_duplicate_video_id_handling(self, test_env):
        """Case 9: Manifest containing duplicate video IDs is audited and tracked."""
        vid_path = test_env["video_dir"] / "DUP_01.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)

        manifest_data = {
            "videos": [
                {
                    "video_id": "DUP_01",
                    "filename": "DUP_01.mp4",
                    "sha256": sha,
                    "fps": 10.0,
                    "reported_frame_count": 30,
                    "duration_seconds": 3.0,
                },
                {
                    "video_id": "DUP_01",  # duplicate ID
                    "filename": "DUP_01.mp4",
                    "sha256": sha,
                    "fps": 10.0,
                    "reported_frame_count": 30,
                    "duration_seconds": 3.0,
                }
            ]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        records, summary = auditor.audit_all(probe_decode=True)
        assert len(records) == 2
        assert records[0].video_id == "DUP_01"
        assert records[1].video_id == "DUP_01"

    def test_case_10_far_exposure_shortfall_calculation(self, test_env):
        """Case 10: Accurate calculation of remaining shortfall to 18000s (5.0 hours)."""
        vid_path = test_env["video_dir"] / "SHORTFALL_01.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=100, fps=10.0)  # 10s video

        # 1. When far_eligible is 0.0
        manifest_data = {
            "videos": [{
                "video_id": "SHORTFALL_01",
                "filename": "SHORTFALL_01.mp4",
                "sha256": sha,
                "fps": 10.0,
                "reported_frame_count": 100,
                "duration_seconds": 10.0,
                "human_review_status": "PENDING_HUMAN_REVIEW"
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        _, summary = auditor.audit_all(probe_decode=True)
        assert summary["duration_indices"]["far_eligible_duration_sec"] == 0.0
        assert summary["duration_indices"]["remaining_duration_to_5h_sec"] == 18000.0
        assert summary["duration_indices_hours"]["remaining_duration_to_5h_hours"] == 5.0

        # 2. When a video is genuinely human verified and eligible (10s)
        manifest_data["videos"][0]["human_review_status"] = "VERIFIED_NEGATIVE"
        manifest_data["videos"][0]["reviewer_name"] = "researcher_01"
        manifest_data["videos"][0]["review_timestamp"] = "2026-11-17T18:00:00Z"
        manifest_data["videos"][0]["attest_full_video_viewed"] = "TRUE"
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        _, summary2 = auditor.audit_all(probe_decode=True)
        eligible_sec = summary2["duration_indices"]["far_eligible_duration_sec"]
        assert eligible_sec > 0.0
        expected_remaining = round(max(0.0, 18000.0 - eligible_sec), 2)
        assert summary2["duration_indices"]["remaining_duration_to_5h_sec"] == expected_remaining

    def test_case_11_duration_mismatch_detection(self, test_env):
        """Case 11: Declared vs decoded duration discrepancy detected and reported."""
        vid_path = test_env["video_dir"] / "MISMATCH_01.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)  # 3.0s

        manifest_data = {
            "videos": [{
                "video_id": "MISMATCH_01",
                "filename": "MISMATCH_01.mp4",
                "sha256": sha,
                "fps": 10.0,
                "reported_frame_count": 30,
                "duration_seconds": 15.0,  # Manifest declares 15s instead of 3s
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        records, summary = auditor.audit_all(probe_decode=True)

        rec = records[0]
        assert rec.declared_duration_sec == 15.0
        assert rec.decoded_duration_sec <= 3.1
        assert abs(rec.declared_duration_sec - rec.decoded_duration_sec) > 10.0

    def test_case_12_ineligible_technical_video_excluded_from_exposure(self, test_env):
        """Case 12: Ineligible technical video has zero contribution to verified and FAR exposure."""
        vid_path = test_env["video_dir"] / "INELIGIBLE_01.mp4"
        sha = create_synthetic_mp4(vid_path, num_frames=30, fps=10.0)

        # Force a technical failure: bad sha
        manifest_data = {
            "videos": [{
                "video_id": "INELIGIBLE_01",
                "filename": "INELIGIBLE_01.mp4",
                "sha256": "wrong_sha256",
                "fps": 10.0,
                "reported_frame_count": 30,
                "duration_seconds": 3.0,
                "human_review_status": "VERIFIED_NEGATIVE",
                "reviewer_name": "researcher_01",
                "review_timestamp": "2026-11-17T18:00:00Z",
                "attest_full_video_viewed": "TRUE",
            }]
        }
        test_env["manifest_path"].write_text(json.dumps(manifest_data), encoding="utf-8")

        auditor = NegativeVideoTechnicalAuditor(
            manifest_path=test_env["manifest_path"],
            split_manifest_path=test_env["split_manifest_path"],
            video_dirs=[test_env["video_dir"]],
            output_dir=test_env["output_dir"]
        )
        records, summary = auditor.audit_all(probe_decode=True)

        rec = records[0]
        assert rec.technical_verdict == TechnicalVerdict.FAIL.value
        assert rec.eligible_for_far is False
        assert summary["duration_indices"]["verified_technical_duration_sec"] == 0.0
        assert summary["duration_indices"]["far_eligible_duration_sec"] == 0.0
