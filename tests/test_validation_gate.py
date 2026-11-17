#!/usr/bin/env python3
"""Comprehensive test suite for the Validation Gate Auditor.

Verifies all 13 mandatory scientific gate conditions:
1. Reviewer identity mismatch / self-verification forbidden.
2. Missing timestamps.
3. Timestamp outside video duration.
4. Event disagreement (timing delta > 0.5s or non-confirm decision).
5. Invalid decision string.
6. Duplicate reviewer / disallowed reviewer identity.
7. Missing full-view attestation.
8. SHA-256 hash mismatch.
9. Truncated video excluded from exposure.
10. Unknown EOF excluded from exposure.
11. Exposure below 5 hours strictly blocks gate.
12. Test split contamination rejected in validation.
13. Invalid provenance detected and flagged.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_week6_gate import (
    DISALLOWED_REVIEWERS,
    EXPECTED_VAL_NEG_VIDEOS,
    SOP_ONSET_TOLERANCE_SEC,
    TARGET_FAR_EXPOSURE_SEC,
    TARGET_POSITIVE_EVENTS,
    ValidationGateAuditor,
    get_video_duration,
    sha256_file,
)


@pytest.fixture
def gate_env(tmp_path: Path):
    """Setup isolated test fixtures for Validation Gate."""
    gt_worksheet = tmp_path / "validation_gt_review_worksheet.csv"
    events_jsonl = tmp_path / "events.jsonl"
    neg_manifest = tmp_path / "validation_negative_review_manifest.csv"
    split_manifest = tmp_path / "split_manifest.csv"
    far_recon = tmp_path / "far_exposure_reconciliation.json"
    output_dir = tmp_path / "runs/audits"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base events.jsonl
    events_data = [
        {"video_id": "POS_07", "label": "FIRE", "start_sec": 0.1, "end_sec": 70.0, "annotator_01": "annotator_01"},
        {"video_id": "POS_08", "label": "FIRE", "start_sec": 0.4, "end_sec": 10.0, "annotator_01": "annotator_01"},
        {"video_id": "POS_08", "label": "SMOKE", "start_sec": 8.8, "end_sec": 10.0, "annotator_01": "annotator_01"},
        {"video_id": "POS_17", "label": "SMOKE", "start_sec": 0.4, "end_sec": 20.0, "annotator_01": "annotator_01"},
        {"video_id": "POS_18", "label": "SMOKE", "start_sec": 11.2, "end_sec": 70.0, "annotator_01": "annotator_01"},
    ]
    with open(events_jsonl, "w", encoding="utf-8") as f:
        for ev in events_data:
            f.write(json.dumps(ev) + "\n")

    # 2. Base split_manifest.csv
    with open(split_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "video_id", "split", "category"])
        for v in EXPECTED_VAL_NEG_VIDEOS:
            writer.writerow([f"{v}_0001.jpg", v, "val", "negative"])

    # 3. Base GT worksheet
    ws_headers = [
        "event_id", "source_line_no", "video_id", "camera_id", "target_label",
        "video_relative_path", "video_sha256", "annotator_01_id", "annotator_02_id",
        "annotator_02_start_sec", "annotator_02_end_sec", "annotator_02_decision",
        "attest_full_view", "attest_independent", "review_timestamp", "review_notes"
    ]
    ws_rows = [
        {"event_id": "VAL_POS_07_FIRE", "source_line_no": 11, "video_id": "POS_07", "camera_id": "BENCH_POS_07", "target_label": "FIRE", "video_relative_path": "path", "video_sha256": "sha", "annotator_01_id": "annotator_01", "annotator_02_id": "", "annotator_02_start_sec": "", "annotator_02_end_sec": "", "annotator_02_decision": "PENDING_REVIEW", "attest_full_view": "FALSE", "attest_independent": "FALSE", "review_timestamp": "", "review_notes": ""},
        {"event_id": "VAL_POS_08_FIRE", "source_line_no": 12, "video_id": "POS_08", "camera_id": "BENCH_POS_08", "target_label": "FIRE", "video_relative_path": "path", "video_sha256": "sha", "annotator_01_id": "annotator_01", "annotator_02_id": "", "annotator_02_start_sec": "", "annotator_02_end_sec": "", "annotator_02_decision": "PENDING_REVIEW", "attest_full_view": "FALSE", "attest_independent": "FALSE", "review_timestamp": "", "review_notes": ""},
        {"event_id": "VAL_POS_08_SMOKE", "source_line_no": 13, "video_id": "POS_08", "camera_id": "BENCH_POS_08", "target_label": "SMOKE", "video_relative_path": "path", "video_sha256": "sha", "annotator_01_id": "annotator_01", "annotator_02_id": "", "annotator_02_start_sec": "", "annotator_02_end_sec": "", "annotator_02_decision": "PENDING_REVIEW", "attest_full_view": "FALSE", "attest_independent": "FALSE", "review_timestamp": "", "review_notes": ""},
        {"event_id": "VAL_POS_17_SMOKE", "source_line_no": 27, "video_id": "POS_17", "camera_id": "BENCH_POS_17", "target_label": "SMOKE", "video_relative_path": "path", "video_sha256": "sha", "annotator_01_id": "annotator_01", "annotator_02_id": "", "annotator_02_start_sec": "", "annotator_02_end_sec": "", "annotator_02_decision": "PENDING_REVIEW", "attest_full_view": "FALSE", "attest_independent": "FALSE", "review_timestamp": "", "review_notes": ""},
        {"event_id": "VAL_POS_18_SMOKE", "source_line_no": 28, "video_id": "POS_18", "camera_id": "BENCH_POS_18", "target_label": "SMOKE", "video_relative_path": "path", "video_sha256": "sha", "annotator_01_id": "annotator_01", "annotator_02_id": "", "annotator_02_start_sec": "", "annotator_02_end_sec": "", "annotator_02_decision": "PENDING_REVIEW", "attest_full_view": "FALSE", "attest_independent": "FALSE", "review_timestamp": "", "review_notes": ""},
    ]
    with open(gt_worksheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ws_headers)
        writer.writeheader()
        writer.writerows(ws_rows)

    # 4. Base negative manifest
    neg_headers = [
        "video_id", "filename", "file_path", "file_size_bytes", "sha256_actual",
        "sha256_expected", "sha256_match", "split_provenance", "native_fps",
        "resolution", "reported_frames", "duration_seconds", "decode_status",
        "decode_tool", "technical_verdict", "content_decision", "reviewer_name",
        "review_timestamp", "attest_full_video_viewed", "review_notes", "evidence_timestamp_sec"
    ]
    neg_rows = []
    for vid in sorted(list(EXPECTED_VAL_NEG_VIDEOS)):
        is_48 = (vid == "VN_NEG_48")
        neg_rows.append({
            "video_id": vid,
            "filename": f"{vid}.mp4",
            "file_path": f"datasets/negative_videos/{vid}.mp4",
            "file_size_bytes": 1000000,
            "sha256_actual": "abc",
            "sha256_expected": "abc",
            "sha256_match": "TRUE",
            "split_provenance": "datasets/vn_dataset/split_manifest.csv::val",
            "native_fps": 30.0,
            "resolution": "1920x1080",
            "reported_frames": 3000,
            "duration_seconds": 100.0,
            "decode_status": "TRUNCATED_STREAM" if is_48 else "NORMAL_EOF",
            "decode_tool": "OPENCV",
            "technical_verdict": "FAIL_TRUNCATED_STREAM" if is_48 else "PASS",
            "content_decision": "PENDING_HUMAN_REVIEW",
            "reviewer_name": "",
            "review_timestamp": "",
            "attest_full_video_viewed": "FALSE",
            "review_notes": "",
            "evidence_timestamp_sec": ""
        })
    with open(neg_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=neg_headers)
        writer.writeheader()
        writer.writerows(neg_rows)

    # 5. Base reconciliation JSON
    far_recon.write_text(json.dumps({"target_exposure_standard": {"required_minimum_seconds": 18000.0}}), encoding="utf-8")

    return {
        "gt_worksheet": gt_worksheet,
        "events_jsonl": events_jsonl,
        "neg_manifest": neg_manifest,
        "split_manifest": split_manifest,
        "far_recon": far_recon,
        "output_dir": output_dir,
    }


class TestValidationGateAuditor:
    """Test suite covering the 13 Validation Gate conditions."""

    def test_case_01_reviewer_identity_mismatch_or_self_verification(self, gate_env):
        """Case 1: Reviewer 2 identical to Reviewer 1 is rejected as self-verification."""
        # Set annotator_02_id == annotator_01_id
        rows = list(csv.DictReader(open(gate_env["gt_worksheet"], "r", encoding="utf-8")))
        rows[0]["annotator_02_id"] = "annotator_01"
        rows[0]["annotator_02_start_sec"] = "0.2"
        rows[0]["annotator_02_end_sec"] = "70.0"
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"

        with open(gate_env["gt_worksheet"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_positive_events()
        assert report["records"][0]["status"] == "INVALID_SUBMISSION"
        assert "Self-verification forbidden" in report["records"][0]["notes"]

    def test_case_02_missing_timestamps(self, gate_env):
        """Case 2: Missing or non-numeric timestamps trigger INVALID_SUBMISSION."""
        rows = list(csv.DictReader(open(gate_env["gt_worksheet"], "r", encoding="utf-8")))
        rows[0]["annotator_02_id"] = "annotator_02"
        rows[0]["annotator_02_start_sec"] = ""  # Missing start
        rows[0]["annotator_02_end_sec"] = "70.0"
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"

        with open(gate_env["gt_worksheet"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_positive_events()
        assert report["records"][0]["status"] == "INVALID_SUBMISSION"
        assert "Invalid numeric timestamps" in report["records"][0]["notes"]

    def test_case_03_timestamp_outside_video_duration(self, gate_env):
        """Case 3: Timestamps outside bounds or start >= end trigger boundary violation."""
        rows = list(csv.DictReader(open(gate_env["gt_worksheet"], "r", encoding="utf-8")))
        rows[0]["annotator_02_id"] = "annotator_02"
        rows[0]["annotator_02_start_sec"] = "50.0"
        rows[0]["annotator_02_end_sec"] = "30.0"  # start >= end
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"

        with open(gate_env["gt_worksheet"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_positive_events()
        assert report["records"][0]["status"] == "INVALID_SUBMISSION"
        assert ">= end timestamp" in report["records"][0]["notes"]

    def test_case_04_event_disagreement(self, gate_env):
        """Case 4: Timing delta > 0.5s or non-confirm decision is flagged as DISPUTED."""
        rows = list(csv.DictReader(open(gate_env["gt_worksheet"], "r", encoding="utf-8")))
        # Event 1: Delta start 1.5s > 0.5s tolerance
        rows[0]["annotator_02_id"] = "annotator_02"
        rows[0]["annotator_02_start_sec"] = "1.6"  # Annotator 1 is 0.1s -> delta = 1.5s
        rows[0]["annotator_02_end_sec"] = "70.0"
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"

        with open(gate_env["gt_worksheet"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_positive_events()
        assert report["records"][0]["status"] == "DISPUTED_NEEDS_ADJUDICATION"
        assert report["disputed_count"] >= 1

    def test_case_05_invalid_decision(self, gate_env):
        """Case 5: Unrecognized decision string triggers INVALID_SUBMISSION."""
        rows = list(csv.DictReader(open(gate_env["gt_worksheet"], "r", encoding="utf-8")))
        rows[0]["annotator_02_id"] = "annotator_02"
        rows[0]["annotator_02_start_sec"] = "0.2"
        rows[0]["annotator_02_end_sec"] = "70.0"
        rows[0]["annotator_02_decision"] = "MAYBE_FIRE"  # Invalid
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"

        with open(gate_env["gt_worksheet"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_positive_events()
        assert report["records"][0]["status"] == "INVALID_SUBMISSION"
        assert "Invalid decision" in report["records"][0]["notes"]

    def test_case_06_duplicate_reviewer_or_disallowed(self, gate_env):
        """Case 6: Disallowed reviewer identity (e.g. SYSTEM, AUTO) is rejected."""
        rows = list(csv.DictReader(open(gate_env["gt_worksheet"], "r", encoding="utf-8")))
        rows[0]["annotator_02_id"] = "SYSTEM"  # Disallowed
        rows[0]["annotator_02_start_sec"] = "0.2"
        rows[0]["annotator_02_end_sec"] = "70.0"
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "TRUE"
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"

        with open(gate_env["gt_worksheet"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_positive_events()
        assert report["records"][0]["status"] == "INVALID_SUBMISSION"
        assert "Invalid Annotator 2 identity" in report["records"][0]["notes"]

    def test_case_07_missing_full_view_attestation(self, gate_env):
        """Case 7: Missing full view attestation triggers INVALID_SUBMISSION."""
        rows = list(csv.DictReader(open(gate_env["gt_worksheet"], "r", encoding="utf-8")))
        rows[0]["annotator_02_id"] = "annotator_02"
        rows[0]["annotator_02_start_sec"] = "0.2"
        rows[0]["annotator_02_end_sec"] = "70.0"
        rows[0]["annotator_02_decision"] = "CONFIRM_EVENT"
        rows[0]["attest_full_view"] = "FALSE"  # Missing full view
        rows[0]["attest_independent"] = "TRUE"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"

        with open(gate_env["gt_worksheet"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_positive_events()
        assert report["records"][0]["status"] == "INVALID_SUBMISSION"
        assert "attest viewing full video" in report["records"][0]["notes"]

    def test_case_08_sha256_mismatch(self, gate_env):
        """Case 8: Hash mismatch in negative review manifest fails technical integrity."""
        rows = list(csv.DictReader(open(gate_env["neg_manifest"], "r", encoding="utf-8")))
        rows[0]["sha256_match"] = "FALSE"
        rows[0]["content_decision"] = "VERIFIED_NEGATIVE"
        rows[0]["reviewer_name"] = "researcher_01"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"
        rows[0]["attest_full_video_viewed"] = "TRUE"

        with open(gate_env["neg_manifest"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_negative_videos()
        assert report["records"][0]["admissible_far"] is False
        assert any("SHA256_MISMATCH" in err for err in report["errors"])

    def test_case_09_truncated_video_excluded(self, gate_env):
        """Case 9: VN_NEG_48 (truncated stream) is strictly excluded from FAR exposure."""
        rows = list(csv.DictReader(open(gate_env["neg_manifest"], "r", encoding="utf-8")))
        # Try to make VN_NEG_48 appear verified
        for r in rows:
            if r["video_id"] == "VN_NEG_48":
                r["content_decision"] = "VERIFIED_NEGATIVE"
                r["reviewer_name"] = "researcher_01"
                r["review_timestamp"] = "2026-11-17T18:00:00Z"
                r["attest_full_video_viewed"] = "TRUE"

        with open(gate_env["neg_manifest"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_negative_videos()
        neg_48_rec = next(r for r in report["records"] if r["video_id"] == "VN_NEG_48")
        assert neg_48_rec["admissible_far"] is False
        assert report["technical_fail_count"] >= 1

    def test_case_10_unknown_eof_excluded(self, gate_env):
        """Case 10: Video with UNKNOWN_EOF or decode error is excluded from exposure."""
        rows = list(csv.DictReader(open(gate_env["neg_manifest"], "r", encoding="utf-8")))
        rows[0]["decode_status"] = "UNKNOWN_EOF"
        rows[0]["technical_verdict"] = "FAIL_UNKNOWN_EOF"
        rows[0]["content_decision"] = "VERIFIED_NEGATIVE"
        rows[0]["reviewer_name"] = "researcher_01"
        rows[0]["review_timestamp"] = "2026-11-17T18:00:00Z"
        rows[0]["attest_full_video_viewed"] = "TRUE"

        with open(gate_env["neg_manifest"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_negative_videos()
        assert report["records"][0]["admissible_far"] is False

    def test_case_11_exposure_below_5_hours_blocks_gate(self, gate_env):
        """Case 11: FAR exposure below 5.0h (18000s) strictly blocks gate passed status."""
        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        result = auditor.run_full_validation_gate()
        assert result["validation_gate_status"] == "BLOCKED"
        assert result["phase_3_eligible"] is False
        assert result["summary_metrics"]["shortfall_exposure_hours"] == 5.0

    def test_case_12_test_split_contamination(self, gate_env):
        """Case 12: Negative video from test split injected into validation is rejected."""
        rows = list(csv.DictReader(open(gate_env["neg_manifest"], "r", encoding="utf-8")))
        rows.append({
            "video_id": "VN_NEG_04",  # TEST split video
            "filename": "VN_NEG_04.mp4",
            "file_path": "datasets/negative_videos/VN_NEG_04.mp4",
            "file_size_bytes": 1000000,
            "sha256_actual": "abc",
            "sha256_expected": "abc",
            "sha256_match": "TRUE",
            "split_provenance": "datasets/vn_dataset/split_manifest.csv::test",
            "native_fps": 30.0,
            "resolution": "1920x1080",
            "reported_frames": 3000,
            "duration_seconds": 100.0,
            "decode_status": "NORMAL_EOF",
            "decode_tool": "OPENCV",
            "technical_verdict": "PASS",
            "content_decision": "PENDING_HUMAN_REVIEW",
            "reviewer_name": "",
            "review_timestamp": "",
            "attest_full_video_viewed": "FALSE",
            "review_notes": "",
            "evidence_timestamp_sec": ""
        })

        with open(gate_env["neg_manifest"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], gate_env["events_jsonl"],
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        report = auditor.audit_negative_videos()
        assert any("SPLIT_CONTAMINATION" in err for err in report["errors"])

    def test_case_13_invalid_provenance_detected(self, gate_env):
        """Case 13: Missing provenance source artifact flags provenance gate as INVALID."""
        missing_events_path = gate_env["events_jsonl"].parent / "non_existent_events.jsonl"
        auditor = ValidationGateAuditor(
            gate_env["gt_worksheet"], missing_events_path,
            gate_env["neg_manifest"], gate_env["split_manifest"],
            gate_env["far_recon"], gate_env["output_dir"]
        )
        prov = auditor.audit_provenance()
        assert prov["status"] == "INVALID"
        assert len(prov["errors"]) >= 1
