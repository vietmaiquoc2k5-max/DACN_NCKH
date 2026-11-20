#!/usr/bin/env python3
"""Unit and regression tests for Week 6 Phase 2: Data & Ground Truth Validation.
Verifies scientific audit mechanisms:
- Ground truth schema, timestamp consistency, validation gate, test GT completeness.
- Negative video manifest, hash integrity, decode fallback, split leakage, FAR tiers.
- Error review manifest, taxonomy, human attribution rules, anti-auto-promotion gate.
- Integrated Phase 2 runner and report generation.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from scripts.audit_week6_ground_truth import EventAuditRecord, GroundTruthAuditor
from scripts.audit_week6_negative_videos import NegativeVideoAuditRecord, NegativeVideoAuditor
from scripts.run_week6_phase2_audit import audit_environment_and_checkpoints, run_phase2_audit
from scripts.validate_week6_error_reviews import (
    DISALLOWED_REVIEWER_NAMES,
    STANDARD_TAXONOMY,
    parse_iso_timestamp,
    validate_error_reviews,
)

ROOT = Path(__file__).resolve().parent.parent


class TestGroundTruthAudit:
    """Tests for Ground Truth auditing and validation gate."""

    def test_timestamp_and_record_dataclass(self):
        rec = EventAuditRecord(
            line_number=1,
            video_id='POS_07',
            camera_id='CAM_01',
            label='FIRE',
            start_sec=10.0,
            end_sec=25.0,
            duration_sec=15.0,
            verification_status='pending_verification',
            annotator_01='annotator_01',
            annotator_02=None,
            split_assignment='val',
            status_classification='VALID',
            audit_notes='OK'
        )
        assert rec.start_sec < rec.end_sec
        assert rec.duration_sec == 15.0
        assert rec.verification_status == 'pending_verification'

    def test_validation_gate_blocks_on_pending(self, tmp_path: Path):
        # Create mock events.jsonl with 1 pending validation event
        events_file = tmp_path / 'events.jsonl'
        events_file.write_text(
            json.dumps({
                'video_id': 'POS_07',
                'camera_id': 'CAM_01',
                'label': 'fire',
                'start_sec': 5.0,
                'end_sec': 12.0,
                'verification_status': 'pending_verification',
                'annotator_01': 'user1'
            }) + '\n',
            encoding='utf-8'
        )

        split_manifest = tmp_path / 'split_manifest.json'
        split_manifest.write_text(
            json.dumps({
                'val_videos': ['POS_07'],
                'test_videos': ['POS_09']
            }),
            encoding='utf-8'
        )

        auditor = GroundTruthAuditor(
            events_path=events_file,
            split_manifest_path=split_manifest,
            output_dir=tmp_path / 'out'
        )
        records, summary = auditor.audit_events()

        assert len(records) == 1
        assert summary['validation_gate']['status'] == 'BLOCKED'
        assert summary['validation_gate']['pending_val_events'] == 1
        assert summary['validation_gate']['verified_val_events'] == 0

    def test_test_gt_completeness_flags_missing(self, tmp_path: Path):
        events_file = tmp_path / 'events.jsonl'
        events_file.write_text(
            json.dumps({
                'video_id': 'POS_09',
                'camera_id': 'CAM_01',
                'label': 'smoke',
                'start_sec': 2.0,
                'end_sec': 8.0,
                'verification_status': 'cross_verified',
                'annotator_01': 'user1',
                'annotator_02': 'user2'
            }) + '\n',
            encoding='utf-8'
        )

        split_manifest = tmp_path / 'split_manifest.json'
        split_manifest.write_text(
            json.dumps({
                'videos': []
            }),
            encoding='utf-8'
        )

        auditor = GroundTruthAuditor(
            events_path=events_file,
            split_manifest_path=split_manifest,
            output_dir=tmp_path / 'out'
        )
        _, summary = auditor.audit_events()

        test_audit = summary['test_gt_completeness']
        assert test_audit['total_test_videos_target'] == 16
        assert test_audit['annotated_test_videos_count'] == 1
        assert test_audit['missing_test_videos_count'] == 15
        assert 'POS_10' in test_audit['missing_test_videos']
        assert 'POS_69' in test_audit['missing_test_videos']


class TestNegativeVideoAudit:
    """Tests for Negative Video auditing, tool fallback, and leakage."""

    def test_leakage_detection_logic(self, tmp_path: Path):
        # Mock negative manifest
        neg_manifest = tmp_path / 'neg_manifest.json'
        neg_manifest.write_text(
            json.dumps({
                'videos': [
                    {'video_id': 'NEG_01', 'filename': 'NEG_01.mp4', 'sha256': 'fake1', 'duration_seconds': 100},
                    {'video_id': 'NEG_02', 'filename': 'NEG_02.mp4', 'sha256': 'fake2', 'duration_seconds': 200}
                ]
            }),
            encoding='utf-8'
        )

        # Mock dataset structure where NEG_01 has an image in train
        train_img_dir = tmp_path / 'dataset' / 'images' / 'train'
        train_img_dir.mkdir(parents=True, exist_ok=True)
        (train_img_dir / 'NEG_01_frame_0001.jpg').write_bytes(b'dummy')

        auditor = NegativeVideoAuditor(
            manifest_path=neg_manifest,
            video_dir=tmp_path / 'videos',
            dataset_dir=tmp_path / 'dataset',
            output_dir=tmp_path / 'out'
        )
        records, summary = auditor.audit_negative_videos()

        assert len(records) == 2
        rec_01 = next(r for r in records if r.video_id == 'NEG_01')
        rec_02 = next(r for r in records if r.video_id == 'NEG_02')

        assert rec_01.leakage_status == 'TRAIN_EXPOSURE_LEAKAGE_RISK'
        assert rec_01.far_eligibility_status == 'BLOCKED_PENDING_HUMAN_CONTENT_REVIEW'
        assert rec_02.leakage_status == 'CLEAN_HELD_OUT_NEGATIVE'

        assert summary['split_membership_breakdown']['train_sampled_videos_count'] == 1
        assert summary['split_membership_breakdown']['held_out_negative_videos_count'] == 1
        assert summary['far_eligibility_gate']['status'] == 'BLOCKED'


class TestErrorReviewValidation:
    """Tests for Error Review Validation compliance and anti-auto-promotion."""

    def test_iso_timestamp_parsing(self):
        assert parse_iso_timestamp('2026-11-17T17:00:00') is True
        assert parse_iso_timestamp('2026-11-17') is True
        assert parse_iso_timestamp('invalid-time') is False
        assert parse_iso_timestamp('') is False

    def test_disallowed_reviewers(self):
        for name in ['SYSTEM', 'auto', 'None', '', 'bot']:
            assert name.lower() in DISALLOWED_REVIEWER_NAMES

    def test_rejects_unreviewed_as_verified(self, tmp_path: Path):
        csv_file = tmp_path / 'manifest.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'case_id', 'video_id', 'timestamp', 'frame_index', 'checkpoint_sha256',
                'evidence_path', 'suspected_cause', 'reviewer_name', 'review_timestamp',
                'reviewer_status', 'review_notes'
            ])
            writer.writeheader()
            writer.writerow({
                'case_id': 'CASE_01',
                'video_id': 'NEG_01',
                'timestamp': '12.0s',
                'frame_index': '300',
                'checkpoint_sha256': 'dummy',
                'evidence_path': 'fake.jpg',
                'suspected_cause': 'VEHICLE_LIGHT',
                'reviewer_name': 'AUTO_SYSTEM',  # Disallowed
                'review_timestamp': '2026-11-17T17:00:00',
                'reviewer_status': 'VERIFIED',
                'review_notes': 'Automated classification'
            })

        report = validate_error_reviews(csv_file)
        assert report['invalid_count'] == 1
        assert report['verified_count'] == 0
        assert report['validation_status'] == 'VALIDATION_FAILED'

    def test_accepts_valid_human_review(self, tmp_path: Path):
        # Create a real evidence file so existence check passes
        evidence_file = tmp_path / 'evidence.jpg'
        evidence_file.write_bytes(b'\xFF\xD8\xFF\xE0')

        csv_file = tmp_path / 'manifest.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'case_id', 'video_id', 'timestamp', 'frame_index', 'checkpoint_sha256',
                'evidence_path', 'suspected_cause', 'reviewer_name', 'review_timestamp',
                'reviewer_status', 'review_notes'
            ])
            writer.writeheader()
            # Valid human review row
            writer.writerow({
                'case_id': 'CASE_01',
                'video_id': 'NEG_01',
                'timestamp': '12.0s',
                'frame_index': '300',
                'checkpoint_sha256': 'dummy',
                'evidence_path': str(evidence_file.relative_to(ROOT)).replace('\\', '/') if evidence_file.is_relative_to(ROOT) else str(evidence_file),
                'suspected_cause': 'VEHICLE_LIGHT',
                'reviewer_name': 'Mai Quốc Việt',
                'review_timestamp': '2026-11-17T17:00:00',
                'reviewer_status': 'VERIFIED',
                'review_notes': 'Headlights reflected on pavement'
            })

        report = validate_error_reviews(csv_file)
        assert report['invalid_count'] == 0
        assert report['verified_count'] == 1
        assert report['cause_distribution']['VEHICLE_LIGHT'] == 1

    def test_current_manifest_state_is_pending_human_review(self):
        manifest_csv = ROOT / 'runs/audits/week6/error_review_manifest.csv'
        if manifest_csv.exists():
            report = validate_error_reviews(manifest_csv)
            assert report['validation_status'] == 'PENDING_HUMAN_REVIEW'
            assert report['pending_count'] == 60
            assert report['verified_count'] == 0
            assert report['invalid_count'] == 0


class TestPhase2MasterRunner:
    """Tests for Phase 2 Master Runner environment audit and report generation."""

    def test_audit_environment_and_checkpoints(self):
        prov = audit_environment_and_checkpoints()
        assert 'environment' in prov
        assert 'checkpoints' in prov
        ckpts = prov['checkpoints']
        assert 'strategy_c_best_pt' in ckpts
        if ckpts['strategy_c_best_pt']['exists']:
            assert '356a288f045dcc' in ckpts['strategy_c_best_pt']['sha256']
        else:
            assert ckpts['strategy_c_best_pt']['sha256'] == 'FILE_NOT_FOUND'

    def test_run_phase2_audit_execution(self, tmp_path: Path):
        # Run audit with output to tmp_path using existing negative video audit json
        neg_json_source = ROOT / 'runs/audits/week6/negative_video_audit.json'
        if neg_json_source.exists():
            (tmp_path / 'negative_video_audit.json').write_text(
                neg_json_source.read_text(encoding='utf-8'),
                encoding='utf-8'
            )

        err_manifest_source = ROOT / 'runs/audits/week6/error_review_manifest.csv'
        if err_manifest_source.exists():
            (tmp_path / 'error_review_manifest.csv').write_text(
                err_manifest_source.read_text(encoding='utf-8'),
                encoding='utf-8'
            )

        summary = run_phase2_audit(output_dir=tmp_path, reprobe_videos=False)

        assert summary['overall_verdict'] == 'BLOCKED'
        assert len(summary['blockers']) >= 4
        assert (tmp_path / 'phase2_summary.json').exists()
        assert (tmp_path / 'phase2_final_report.md').exists()

        report_text = (tmp_path / 'phase2_final_report.md').read_text(encoding='utf-8')
        assert 'PHẦN 1. XÁC MINH NGUỒN GỐC REPOSITORY' in report_text
        assert 'PHẦN 2. KIỂM ĐỊNH TÍNH TOÀN VẸN CỦA GROUND TRUTH' in report_text
        assert 'PHẦN 3. KIỂM ĐỊNH VIDEO ÂM TÍNH' in report_text
        assert 'PHẦN 4. QUẢN LÝ VÀ THẨM ĐỊNH CÁC CA BÁO ĐỘNG GIẢ' in report_text
        assert 'PHẦN 5. KẾT LUẬN KHOA HỌC' in report_text
