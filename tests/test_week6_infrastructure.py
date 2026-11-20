"""Unit and System Tests for Week 6 Infrastructure.
Tests experiment configuration, split protection, operating point locking,
ablation runner guards, error extraction schema, export dependency detection, and GT invariants.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_week6_ablation import (
    check_split_protection,
    check_validation_gt_status,
    sha256_file
)
from scripts.export_and_benchmark_models import inspect_environment
from scripts.extract_week6_error_cases import extract_and_catalog_errors


class TestWeek6ExperimentProtocol:
    """Validate configs/experiments/week6_protocol.yaml against research requirements."""

    def test_protocol_file_exists(self):
        protocol_path = ROOT / 'configs/experiments/week6_protocol.yaml'
        assert protocol_path.exists(), 'week6_protocol.yaml must exist'

    def test_protocol_checkpoint_hashes(self):
        protocol_path = ROOT / 'configs/experiments/week6_protocol.yaml'
        cfg = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
        
        # Strategy C check
        strat_c_path = ROOT / cfg['checkpoints']['winning_model']['path']
        if not strat_c_path.exists():
            pytest.skip(f"Winning model file missing on local disk (pending transfer from GPU machine): {strat_c_path}")
        else:
            actual_sha = sha256_file(strat_c_path)
            assert actual_sha == cfg['checkpoints']['winning_model']['sha256'], "Winning model SHA mismatch"

        # Baseline check
        base_path = ROOT / cfg['checkpoints']['baseline_model']['path']
        assert base_path.exists(), f"Baseline model file missing: {base_path}"
        actual_base_sha = sha256_file(base_path)
        assert actual_base_sha == cfg['checkpoints']['baseline_model']['sha256'], "Baseline model SHA mismatch"

    def test_operating_point_is_strictly_provisional(self):
        protocol_path = ROOT / 'configs/experiments/week6_protocol.yaml'
        cfg = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
        op = cfg['operating_point']
        assert op['status'] == 'PROVISIONAL', "Operating point must remain PROVISIONAL"
        assert op['lock_status'] == 'UNLOCKED', "Operating point lock_status must be UNLOCKED"
        assert 'pending' in op['reason'].lower(), "Operating point must state pending GT as reason"

    def test_manifest_hashes_match(self):
        protocol_path = ROOT / 'configs/experiments/week6_protocol.yaml'
        cfg = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
        manifests = cfg['manifests']

        dataset_path = ROOT / manifests['dataset_manifest']['path']
        assert sha256_file(dataset_path) == manifests['dataset_manifest']['sha256']

        neg_path = ROOT / manifests['negative_videos_manifest']['path']
        assert sha256_file(neg_path) == manifests['negative_videos_manifest']['sha256']

        gt_path = ROOT / manifests['ground_truth_events']['path']
        assert sha256_file(gt_path) == manifests['ground_truth_events']['sha256']


class TestSplitProtectionAndAblationGuards:
    """Enforce test set isolation and GT validation blocking."""

    def test_split_protection_rejects_test(self):
        with pytest.raises(ValueError, match="TEST_SPLIT_FORBIDDEN_FOR_TUNING"):
            check_split_protection("test")

        with pytest.raises(ValueError, match="TEST_SPLIT_FORBIDDEN_FOR_TUNING"):
            check_split_protection("TEST")

    def test_split_protection_accepts_val(self):
        # Should not raise
        check_split_protection("val")
        check_split_protection("validation")

    def test_split_protection_rejects_unknown(self):
        with pytest.raises(ValueError, match="INVALID_SPLIT"):
            check_split_protection("train")
        with pytest.raises(ValueError, match="INVALID_SPLIT"):
            check_split_protection("random_split")

    def test_validation_gt_blocks_when_events_pending(self):
        gt_path = ROOT / 'data/ground_truth/events.jsonl'
        status = check_validation_gt_status(gt_path)
        # In current repo, validation events (POS_07, POS_08, POS_17, POS_18) are pending
        assert status['is_blocked'] is True
        assert status['pending_val_events'] > 0
        assert status['verified_val_events'] == 0
        assert "VALIDATION_EVENT_GROUND_TRUTH_PENDING_VERIFICATION" in status['reason']

    def test_synthetic_pending_gt_cannot_be_auto_verified(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({
                'video_id': 'POS_07',
                'verification_status': 'pending_verification'
            }) + '\n')
            temp_path = Path(f.name)

        try:
            res = check_validation_gt_status(temp_path)
            assert res['is_blocked'] is True
            assert res['verified_val_events'] == 0
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestExportAndBenchmarkInspection:
    """Test runtime inspection and export dependency detection."""

    def test_inspect_environment_structure(self):
        report = inspect_environment()
        assert 'environment' in report
        assert 'backends' in report
        env = report['environment']
        assert 'python_version' in env
        assert 'torch_version' in env
        assert 'cuda_available' in env

    def test_missing_dependencies_reported_accurately(self):
        report = inspect_environment()
        backends = report['backends']
        assert 'onnx' in backends
        assert 'onnxruntime' in backends
        assert 'openvino' in backends
        assert 'tensorrt' in backends
        # These packages are currently missing in .venv and must be marked as missing
        assert backends['onnx']['status'] == 'MISSING_PACKAGE'
        assert backends['onnxruntime']['status'] == 'MISSING_PACKAGE'
        assert backends['openvino']['status'] == 'MISSING_PACKAGE'
        assert backends['tensorrt']['status'] == 'MISSING_PACKAGE'

    def test_pytorch_cuda_available_on_local_gpu(self):
        report = inspect_environment()
        backends = report['backends']
        if torch.cuda.is_available():
            assert backends['pytorch_cuda']['available'] is True
            assert backends['pytorch_cuda']['status'] == 'AVAILABLE'
        else:
            assert backends['pytorch_cuda']['available'] is False
            assert backends['pytorch_cuda']['status'] == 'UNAVAILABLE_NO_CUDA'


class TestErrorCasesExtraction:
    """Test error case extraction, merging, and schema compliance."""

    def test_error_catalog_exists_and_schema_valid(self):
        catalog_path = ROOT / 'data/error_cases_catalog.csv'
        assert catalog_path.exists(), "data/error_cases_catalog.csv must exist"
        
        with open(catalog_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        
        assert len(rows) >= 50, f"Must have >= 50 error cases, found {len(rows)}"

        expected_fields = [
            'error_id', 'video_id', 'timestamp', 'frame_index', 'model_sha256',
            'prediction', 'ground_truth', 'error_type', 'severity',
            'suspected_cause', 'reviewer_status', 'evidence_path'
        ]
        assert list(rows[0].keys()) == expected_fields

    def test_reviewer_status_strictly_pending(self):
        catalog_path = ROOT / 'data/error_cases_catalog.csv'
        with open(catalog_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        
        for r in rows:
            assert r['reviewer_status'] == 'PENDING_REVIEW', (
                f"Error case {r['error_id']} reviewer_status must be PENDING_REVIEW, got {r['reviewer_status']}"
            )
            assert r['suspected_cause'] == 'UNASSIGNED_REQUIRES_HUMAN_REVIEW', (
                f"Error case {r['error_id']} suspected_cause must be UNASSIGNED_REQUIRES_HUMAN_REVIEW"
            )

    def test_deduplication_merges_consecutive_events(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            fake_log = {
                'pipeline_b2_events': [
                    {'video_id': 'TEST_VID', 'label': 'FIRE', 'start_sec': 10.0, 'end_sec': 12.0, 'confidence': 0.4},
                    {'video_id': 'TEST_VID', 'label': 'FIRE', 'start_sec': 13.0, 'end_sec': 15.0, 'confidence': 0.6},
                    {'video_id': 'TEST_VID', 'label': 'FIRE', 'start_sec': 30.0, 'end_sec': 32.0, 'confidence': 0.5},
                ]
            }
            f.write(json.dumps(fake_log))
            temp_log = Path(f.name)

        out_csv = Path(tempfile.mktemp(suffix='.csv'))
        dummy_manifest = Path(tempfile.mktemp(suffix='.json'))
        dummy_manifest.write_text(json.dumps({'videos': []}))

        try:
            results = extract_and_catalog_errors(
                input_events_json=temp_log,
                negative_manifest_json=dummy_manifest,
                output_csv=out_csv,
                model_sha256='dummy_sha',
                merge_gap_sec=5.0
            )
            # Events at 10-12s and 13-15s (gap 1.0s <= 5.0s) should be merged into 1 case.
            # Event at 30-32s (gap 15.0s > 5.0s) is a second case.
            assert len(results) == 2, f"Expected 2 merged cases, got {len(results)}"
            assert results[0]['severity'] == 'HIGH'  # peak_conf is 0.6 >= 0.50
        finally:
            if temp_log.exists():
                temp_log.unlink()
            if out_csv.exists():
                out_csv.unlink()
            if dummy_manifest.exists():
                dummy_manifest.unlink()
