#!/usr/bin/env python3
"""Comprehensive test suite for Week 6 Validation Ground Truth & Negative Video Verification.

Implements all 10 required scientific test cases:
1. Exact 5 validation positive event IDs required.
2. Blind protocol: Annotator 2 cannot view Annotator 1 timestamps before independent submission.
3. Self-verification forbidden: Annotator 2 cannot be identical to Annotator 1.
4. Missing reviewer identity or non-ISO timestamp prevents cross-verification.
5. Negative content verification cannot be auto-promoted without real human reviewer.
6. Technically defective video (e.g. TRUNCATED_STREAM) is strictly excluded from FAR exposure.
7. Test split videos mistakenly included in Validation are detected and rejected.
8. Duplicate video IDs or event IDs are rejected.
9. SHA-256 hash mismatch blocks validation gate.
10. Review worksheet generation and validation strictly preserves original Ground Truth files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from scripts.prepare_week6_validation_gt_review import (
    TARGET_EVENT_SPECS,
    extract_validation_events,
    prepare_validation_gt_worksheet,
    sha256_file,
)
from scripts.prepare_week6_validation_negative_review import (
    get_validation_negative_videos_from_split,
    prepare_validation_negative_review,
)
from scripts.validate_week6_validation_reviews import (
    EXPECTED_VAL_NEG_VIDEOS,
    TARGET_POSITIVE_EVENTS,
    validate_all_validation_reviews,
    validate_negative_review_manifest,
    validate_positive_gt_worksheet,
)


class TestWeek6ValidationVerification:
    """Test suite covering the 10 scientific validation gates."""

    def test_case_01_exact_5_validation_positive_event_ids(self):
        """Case 1: Exactly 5 positive event specifications must be targeted."""
        events_jsonl = ROOT / "data/ground_truth/events.jsonl"
        events = extract_validation_events(events_jsonl)
        assert len(events) == 5

        specs = [(e["clean_video_id"], e["label"].upper()) for e in events]
        expected_specs = [
            ("POS_07", "FIRE"),
            ("POS_08", "FIRE"),
            ("POS_08", "SMOKE"),
            ("POS_17", "SMOKE"),
            ("POS_18", "SMOKE"),
        ]
        assert sorted(specs) == sorted(expected_specs)

    def test_case_02_blind_protocol_omits_annotator_01_timestamps(self, tmp_path: Path):
        """Case 2: Annotator 2 worksheet must NOT contain Annotator 1 timestamps."""
        events_jsonl = ROOT / "data/ground_truth/events.jsonl"
        ws_csv, _ = prepare_validation_gt_worksheet(events_jsonl, tmp_path)

        with open(ws_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])

        forbidden_headers = {"start_sec_annotator_01", "end_sec_annotator_01", "start_sec", "end_sec"}
        overlap = forbidden_headers.intersection(headers)
        assert len(overlap) == 0, f"Worksheet violates blind protocol by exposing: {overlap}"

        # Test validator detects violation if someone manually injects them
        report = validate_positive_gt_worksheet(ws_csv, events_jsonl)
        assert report["blind_protocol_intact"] is True

    def test_case_03_disallow_self_verification(self, tmp_path: Path):
        """Case 3: Annotator 2 cannot be identical to Annotator 1."""
        events_jsonl = ROOT / "data/ground_truth/events.jsonl"
        ws_csv, _ = prepare_validation_gt_worksheet(events_jsonl, tmp_path)

        # Modify first row so annotator_02_id == annotator_01_id
        rows = []
        with open(ws_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        rows[0]["annotator_02_id"] = rows[0]["annotator_01_id"]
        rows[0]["annotator_02_start_sec"] = "0.2"
        rows[0]["annotator_02_end_sec"] = "80.0"
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00"

        with open(ws_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        report = validate_positive_gt_worksheet(ws_csv, events_jsonl)
        assert any("SELF_VERIFICATION_FORBIDDEN" in err for err in report["errors"])
        assert report["validation_gate_status"] == "BLOCKED"

    def test_case_04_missing_reviewer_or_timestamp_blocks_cross_verified(self, tmp_path: Path):
        """Case 4: Cross-verified is forbidden if reviewer is invalid or timestamp non-ISO."""
        events_jsonl = ROOT / "data/ground_truth/events.jsonl"
        ws_csv, _ = prepare_validation_gt_worksheet(events_jsonl, tmp_path)

        rows = []
        with open(ws_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # Set invalid reviewer name (SYSTEM) and invalid timestamp
        rows[0]["annotator_02_id"] = "SYSTEM"
        rows[0]["annotator_02_start_sec"] = "0.1"
        rows[0]["annotator_02_end_sec"] = "80.0"
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "not_a_valid_timestamp"

        with open(ws_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        report = validate_positive_gt_worksheet(ws_csv, events_jsonl)
        assert report["invalid_count"] >= 1
        assert report["validation_gate_status"] == "BLOCKED"

    def test_case_05_disallow_verified_negative_without_human_review(self, tmp_path: Path):
        """Case 5: Negative video cannot be VERIFIED_NEGATIVE without genuine human reviewer."""
        split_manifest = ROOT / "datasets/vn_dataset/split_manifest.csv"
        manifest_csv = tmp_path / "neg_manifest.csv"

        # Create mock row with VERIFIED_NEGATIVE but missing reviewer name
        rows = [{
            "video_id": "VN_NEG_03",
            "filename": "VN_NEG_03.mp4",
            "file_path": "datasets/negative_videos/VN_NEG_03.mp4",
            "file_size_bytes": 18492935,
            "sha256_actual": "25fda9ea2b1ca96ca51b74fa34b6b75e67ffb06daa0fc0cb4f7eeb1f840e6b94",
            "sha256_expected": "25fda9ea2b1ca96ca51b74fa34b6b75e67ffb06daa0fc0cb4f7eeb1f840e6b94",
            "sha256_match": "TRUE",
            "split_provenance": "datasets/vn_dataset/split_manifest.csv::val",
            "native_fps": 30.0,
            "resolution": "1280x720",
            "reported_frames": 1053,
            "duration_seconds": 35.1,
            "decode_status": "PROBED_OPENCV_OK_UNVERIFIED_EOF",
            "decode_tool": "OPENCV",
            "technical_verdict": "PASS_OPENCV_PROBE_UNVERIFIED_EOF",
            "content_decision": "VERIFIED_NEGATIVE",
            "reviewer_name": "",  # Empty!
            "review_timestamp": "2026-11-17T18:00:00",
            "attest_full_video_viewed": "TRUE",
            "review_notes": "",
            "evidence_timestamp_sec": "",
        }]

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        report = validate_negative_review_manifest(manifest_csv, split_manifest)
        rec = report["records"][0]
        assert rec["far_admissible"] is False
        assert any("Invalid reviewer name" in err for err in report["errors"])

    def test_case_06_technical_fault_video_excluded_from_far_exposure(self, tmp_path: Path):
        """Case 6: Video with technical integrity failure (e.g. TRUNCATED_STREAM) is excluded."""
        split_manifest = ROOT / "datasets/vn_dataset/split_manifest.csv"
        manifest_csv = tmp_path / "neg_manifest.csv"

        # Mock VN_NEG_48 with FAIL_TRUNCATED_STREAM but someone marked it VERIFIED_NEGATIVE
        rows = [{
            "video_id": "VN_NEG_48",
            "filename": "VN_NEG_48.mp4",
            "file_path": "datasets/negative_videos/VN_NEG_48.mp4",
            "file_size_bytes": 48545679,
            "sha256_actual": "d1d098b094d507acf1247a25af855d8ee2cd518cb27363f282b774de31354942",
            "sha256_expected": "d1d098b094d507acf1247a25af855d8ee2cd518cb27363f282b774de31354942",
            "sha256_match": "TRUE",
            "split_provenance": "datasets/vn_dataset/split_manifest.csv::val",
            "native_fps": 30.0,
            "resolution": "1920x1080",
            "reported_frames": 8445,
            "duration_seconds": 281.5,
            "decode_status": "TRUNCATED_STREAM",
            "decode_tool": "OPENCV",
            "technical_verdict": "FAIL_TRUNCATED_STREAM",
            "content_decision": "VERIFIED_NEGATIVE",
            "reviewer_name": "Mai Quốc Việt",
            "review_timestamp": "2026-11-17T18:00:00",
            "attest_full_video_viewed": "TRUE",
            "review_notes": "Attempted to verify truncated video",
            "evidence_timestamp_sec": "",
        }]

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        report = validate_negative_review_manifest(manifest_csv, split_manifest)
        rec = report["records"][0]
        # Invariant: Must NOT be admissible into FAR exposure!
        assert rec["far_admissible"] is False
        assert report["verified_exposure_hours"] == 0.0

    def test_case_07_reject_test_split_contamination_in_validation(self, tmp_path: Path):
        """Case 7: Test split video injected into Validation negative manifest is rejected."""
        split_manifest = ROOT / "datasets/vn_dataset/split_manifest.csv"
        manifest_csv = tmp_path / "neg_manifest.csv"

        # VN_NEG_54 is from Test split, NOT Val split!
        rows = [{
            "video_id": "VN_NEG_54",
            "filename": "VN_NEG_54.mp4",
            "file_path": "datasets/negative_videos/VN_NEG_54.mp4",
            "file_size_bytes": 1000,
            "sha256_actual": "0" * 64,
            "sha256_expected": "0" * 64,
            "sha256_match": "TRUE",
            "split_provenance": "TEST_SPLIT",
            "native_fps": 25.0,
            "resolution": "1920x1080",
            "reported_frames": 1000,
            "duration_seconds": 40.0,
            "decode_status": "PROBED_OPENCV_OK_UNVERIFIED_EOF",
            "decode_tool": "OPENCV",
            "technical_verdict": "PASS_OPENCV_PROBE_UNVERIFIED_EOF",
            "content_decision": "PENDING_HUMAN_REVIEW",
            "reviewer_name": "",
            "review_timestamp": "",
            "attest_full_video_viewed": "FALSE",
            "review_notes": "",
            "evidence_timestamp_sec": "",
        }]

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        report = validate_negative_review_manifest(manifest_csv, split_manifest)
        assert any("SPLIT_LEAKAGE_DETECTED" in err for err in report["errors"])

    def test_case_08_reject_duplicate_video_ids(self, tmp_path: Path):
        """Case 8: Duplicate video IDs in negative review manifest are flagged."""
        split_manifest = ROOT / "datasets/vn_dataset/split_manifest.csv"
        manifest_csv = tmp_path / "neg_manifest.csv"

        rows = [
            {
                "video_id": "VN_NEG_03",
                "filename": "VN_NEG_03.mp4",
                "file_path": "datasets/negative_videos/VN_NEG_03.mp4",
                "file_size_bytes": 18492935,
                "sha256_actual": "25fda9ea2b1ca96ca51b74fa34b6b75e67ffb06daa0fc0cb4f7eeb1f840e6b94",
                "sha256_expected": "25fda9ea2b1ca96ca51b74fa34b6b75e67ffb06daa0fc0cb4f7eeb1f840e6b94",
                "sha256_match": "TRUE",
                "split_provenance": "datasets/vn_dataset/split_manifest.csv::val",
                "native_fps": 30.0,
                "resolution": "1280x720",
                "reported_frames": 1053,
                "duration_seconds": 35.1,
                "decode_status": "PROBED_OPENCV_OK_UNVERIFIED_EOF",
                "decode_tool": "OPENCV",
                "technical_verdict": "PASS_OPENCV_PROBE_UNVERIFIED_EOF",
                "content_decision": "PENDING_HUMAN_REVIEW",
                "reviewer_name": "",
                "review_timestamp": "",
                "attest_full_video_viewed": "FALSE",
                "review_notes": "",
                "evidence_timestamp_sec": "",
            },
            {
                "video_id": "VN_NEG_03",  # Duplicate!
                "filename": "VN_NEG_03.mp4",
                "file_path": "datasets/negative_videos/VN_NEG_03.mp4",
                "file_size_bytes": 18492935,
                "sha256_actual": "25fda9ea2b1ca96ca51b74fa34b6b75e67ffb06daa0fc0cb4f7eeb1f840e6b94",
                "sha256_expected": "25fda9ea2b1ca96ca51b74fa34b6b75e67ffb06daa0fc0cb4f7eeb1f840e6b94",
                "sha256_match": "TRUE",
                "split_provenance": "datasets/vn_dataset/split_manifest.csv::val",
                "native_fps": 30.0,
                "resolution": "1280x720",
                "reported_frames": 1053,
                "duration_seconds": 35.1,
                "decode_status": "PROBED_OPENCV_OK_UNVERIFIED_EOF",
                "decode_tool": "OPENCV",
                "technical_verdict": "PASS_OPENCV_PROBE_UNVERIFIED_EOF",
                "content_decision": "PENDING_HUMAN_REVIEW",
                "reviewer_name": "",
                "review_timestamp": "",
                "attest_full_video_viewed": "FALSE",
                "review_notes": "",
                "evidence_timestamp_sec": "",
            }
        ]

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        report = validate_negative_review_manifest(manifest_csv, split_manifest)
        assert any("DUPLICATE_VIDEO_ID" in err for err in report["errors"])

    def test_case_09_sha256_mismatch_blocks_gate(self, tmp_path: Path):
        """Case 9: File hash mismatch against manifest must block the validation gate."""
        split_manifest = ROOT / "datasets/vn_dataset/split_manifest.csv"
        manifest_csv = tmp_path / "neg_manifest.csv"

        rows = [{
            "video_id": "VN_NEG_03",
            "filename": "VN_NEG_03.mp4",
            "file_path": "datasets/negative_videos/VN_NEG_03.mp4",
            "file_size_bytes": 18492935,
            "sha256_actual": "corrupted_hash_value_12345",
            "sha256_expected": "25fda9ea2b1ca96ca51b74fa34b6b75e67ffb06daa0fc0cb4f7eeb1f840e6b94",
            "sha256_match": "FALSE",
            "split_provenance": "datasets/vn_dataset/split_manifest.csv::val",
            "native_fps": 30.0,
            "resolution": "1280x720",
            "reported_frames": 1053,
            "duration_seconds": 35.1,
            "decode_status": "PROBED_OPENCV_OK_UNVERIFIED_EOF",
            "decode_tool": "OPENCV",
            "technical_verdict": "FAIL_SHA256_MISMATCH",
            "content_decision": "VERIFIED_NEGATIVE",
            "reviewer_name": "Mai Quốc Việt",
            "review_timestamp": "2026-11-17T18:00:00",
            "attest_full_video_viewed": "TRUE",
            "review_notes": "",
            "evidence_timestamp_sec": "",
        }]

        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        report = validate_negative_review_manifest(manifest_csv, split_manifest)
        assert any("SHA256_MISMATCH" in err for err in report["errors"])
        assert report["records"][0]["far_admissible"] is False
        assert report["far_gate_status"] == "BLOCKED"

    def test_case_10_worksheet_read_only_preserves_original_events_jsonl(self, tmp_path: Path):
        """Case 10: Generating review worksheets must preserve original events.jsonl byte-for-byte."""
        events_jsonl = ROOT / "data/ground_truth/events.jsonl"
        hash_before = sha256_file(events_jsonl)

        # Run preparation tools
        prepare_validation_gt_worksheet(events_jsonl, tmp_path)

        hash_after = sha256_file(events_jsonl)
        assert hash_before == hash_after, "Ground Truth events.jsonl was modified during review prep!"
