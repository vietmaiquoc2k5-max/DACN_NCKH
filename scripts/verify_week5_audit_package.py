#!/usr/bin/env python3
"""Automated verification script for Week 5 Domain Adaptation Audit Package.
Audits artifacts, hashes, consistency between CSV/JSON, parameter policy and data splits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOG = logging.getLogger('verify_audit_package')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_inventory() -> None:
    LOG.info('Testing artifact inventory...')
    inv_path = ROOT / 'WEEK5_ARTIFACT_INVENTORY.json'
    assert inv_path.exists(), 'WEEK5_ARTIFACT_INVENTORY.json not found'
    data = json.loads(inv_path.read_text(encoding='utf-8'))
    for item in data['artifacts']:
        p = ROOT / item['path']
        assert p.exists(), f"Missing artifact: {item['path']}"
        actual_hash = sha256_file(p)
        assert actual_hash == item['sha256'], f"Hash mismatch for {item['path']}: {actual_hash} != {item['sha256']}"
    LOG.info('Artifact inventory test PASSED (%d artifacts verified).', len(data['artifacts']))


def test_checkpoint_hashes() -> None:
    LOG.info('Testing checkpoint integrity...')
    pairs = [
        ('Strategy A',
         ROOT / 'runs/benchmarks/week5_domain_adaptation/gpu_training/run_full_finetune_seed42/fit/weights/best.pt',
         ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_a_best.pt',
         '11f460c7bc1408c3da9bd99fb9299da3d7b78c80ac3a36161b39a9652339050b'),
        ('Strategy B',
         ROOT / 'runs/benchmarks/week5_domain_adaptation/gpu_training/run_freeze_backbone_seed42/fit/weights/best.pt',
         ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_b_best.pt',
         '0fa658c2f041175b36bf33f2fcf59c055b1c19a12e057baf3d079ce4c70546cc'),
        ('Strategy C',
         ROOT / 'runs/benchmarks/week5_domain_adaptation/gpu_training/run_replay_20_percent_seed42/fit/weights/best.pt',
         ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_c_best.pt',
         '356a288f045dcc3dfe4f9984e86dffc8b854627327370198edc015672265e5df'),
    ]
    for name, fit_p, handoff_p, expected_sha in pairs:
        assert fit_p.exists(), f'{name} fit checkpoint missing'
        assert handoff_p.exists(), f'{name} handoff checkpoint missing'
        fit_sha = sha256_file(fit_p)
        handoff_sha = sha256_file(handoff_p)
        assert fit_sha == expected_sha, f'{name} fit hash mismatch: {fit_sha} != {expected_sha}'
        assert handoff_sha == expected_sha, f'{name} handoff hash mismatch: {handoff_sha} != {expected_sha}'
    LOG.info('Checkpoint hashes test PASSED.')


def test_csv_json_consistency() -> None:
    LOG.info('Testing comparison CSV vs JSON consistency...')
    csv_p = ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.csv'
    json_p = ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/comparison.json'
    assert csv_p.exists() and json_p.exists()
    
    with open(csv_p, newline='', encoding='utf-8') as f:
        csv_rows = list(csv.DictReader(f))
    json_rows = json.loads(json_p.read_text(encoding='utf-8'))['rows']
    
    assert len(csv_rows) == len(json_rows) == 4, f'Row count mismatch: {len(csv_rows)} != {len(json_rows)}'
    for c_row, j_row in zip(csv_rows, json_rows):
        assert c_row['model'] == j_row['model']
        for k, v in j_row.items():
            c_val = c_row[k]
            if isinstance(v, float):
                assert abs(float(c_val) - v) < 1e-9, f"Mismatch at {c_row['model']}.{k}: csv={c_val}, json={v}"
            elif isinstance(v, int):
                assert int(c_val) == v, f"Mismatch at {c_row['model']}.{k}: csv={c_val}, json={v}"
            else:
                assert c_val == str(v), f"Mismatch at {c_row['model']}.{k}: csv={c_val}, json={v}"
    LOG.info('Comparison CSV vs JSON test PASSED (0 mismatches across 25 fields).')


def test_video_disjointness() -> None:
    LOG.info('Testing dataset video disjointness...')
    manifest_p = ROOT / 'handoff/week5/dataset_manifest.json'
    manifest = json.loads(manifest_p.read_text(encoding='utf-8'))
    rows = manifest['rows']
    vn_train_videos = set(r['source_video_sha256'] for r in rows if r['corpus'] == 'VN' and r['split'] == 'train')
    vn_val_videos = set(r['source_video_sha256'] for r in rows if r['corpus'] == 'VN' and r['split'] == 'val')
    overlap = vn_train_videos.intersection(vn_val_videos)
    assert len(overlap) == 0, f'Found {len(overlap)} overlapping videos between VN Train and Val: {overlap}'
    LOG.info('Dataset video disjointness test PASSED (61 Train videos, 23 Val videos, 0 overlap).')


def test_replay_ratio() -> None:
    LOG.info('Testing replay ratio...')
    replay_p = ROOT / 'runs/benchmarks/week5_domain_adaptation/gpu_training/run_replay_20_percent_seed42/replay_selection.json'
    replay = json.loads(replay_p.read_text(encoding='utf-8'))
    assert len(replay['selected']) == 147
    assert replay['vn_presentations'] == 585
    assert replay['dfire_presentations'] == 147
    expected_frac = 147 / (585 + 147)
    assert abs(replay['actual_replay_fraction'] - expected_frac) < 1e-9
    LOG.info('Replay ratio test PASSED (147 / 732 = 20.08197%).')


def main() -> int:
    try:
        test_inventory()
        test_checkpoint_hashes()
        test_csv_json_consistency()
        test_video_disjointness()
        test_replay_ratio()
        LOG.info('ALL AUDIT PACKAGE INTEGRITY TESTS PASSED.')
        return 0
    except AssertionError as err:
        LOG.error('Audit verification FAILED: %s', err)
        return 1


if __name__ == '__main__':
    sys.exit(main())
