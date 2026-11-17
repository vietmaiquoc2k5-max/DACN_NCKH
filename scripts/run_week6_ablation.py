#!/usr/bin/env python3
"""Ablation Runner for Week 6: One-factor sensitivity sweep on Validation split only.
Enforces strict test split protection and blocks event optimization when GT is pending.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
LOG = logging.getLogger('week6_ablation')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_split_protection(split: str) -> None:
    """Enforce strict isolation: Test split is forbidden for tuning or ablation."""
    if split.lower() == 'test':
        raise ValueError(
            'TEST_SPLIT_FORBIDDEN_FOR_TUNING: Week 6 ablation is strictly confined to '
            'the validation split. The test split is held-out and must never be used '
            'for parameter sweeps or model tuning.'
        )
    if split.lower() not in ('val', 'validation'):
        raise ValueError(f'INVALID_SPLIT: Expected "val", got "{split}"')


def check_validation_gt_status(gt_path: Path) -> dict[str, Any]:
    """Inspect Ground Truth verification status for validation positive videos."""
    if not gt_path.exists():
        return {'status': 'MISSING_GT_FILE', 'verified_count': 0, 'pending_count': 0, 'blocked': True}

    lines = [json.loads(l) for l in gt_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    val_videos = {'POS_07', 'POS_08', 'POS_17', 'POS_18', 'VN_POS_07', 'VN_POS_08', 'VN_POS_17', 'VN_POS_18'}
    
    val_events = [e for e in lines if e.get('video_id') in val_videos]
    verified_val = [e for e in val_events if e.get('verification_status') == 'cross_verified']
    pending_val = [e for e in val_events if e.get('verification_status') == 'pending_verification']

    is_blocked = len(val_events) == 0 or len(verified_val) == 0
    return {
        'total_gt_events': len(lines),
        'val_events_count': len(val_events),
        'verified_val_events': len(verified_val),
        'pending_val_events': len(pending_val),
        'is_blocked': is_blocked,
        'reason': (
            'VALIDATION_EVENT_GROUND_TRUTH_PENDING_VERIFICATION: All validation positive events '
            f'({[e.get("video_id") for e in val_events]}) are pending second annotator verification. '
            'Event-level positive recall cannot be tuned without validated ground truth.'
        ) if is_blocked else 'VERIFIED'
    }


def run_ablation_sweep(
    plan_path: Path,
    protocol_path: Path,
    checkpoint_path: Path,
    data_yaml: Path,
    output_dir: Path,
    split: str = 'val',
    resume: bool = True,
    dry_run: bool = False
) -> dict[str, Any]:
    """Execute one-factor sensitivity ablation according to plan."""
    check_split_protection(split)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    protocol = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
    
    # Verify checkpoint
    ckpt_sha = sha256_file(checkpoint_path)
    expected_sha = protocol['checkpoints']['winning_model']['sha256']
    if ckpt_sha != expected_sha:
        raise ValueError(f'CHECKPOINT_SHA_MISMATCH: {ckpt_sha} != {expected_sha}')

    gt_path = ROOT / protocol['manifests']['ground_truth_events']['path']
    gt_check = check_validation_gt_status(gt_path)

    results_csv = output_dir / 'ablation_results.csv'
    results_json = output_dir / 'ablation_results.json'

    existing_runs = {}
    if resume and results_json.exists():
        try:
            prev = json.loads(results_json.read_text(encoding='utf-8'))
            for r in prev.get('runs', []):
                key = f"{r.get('axis')}_{r.get('value')}"
                existing_runs[key] = r
            LOG.info('Resuming ablation sweep: %d runs already recorded.', len(existing_runs))
        except Exception as e:
            LOG.warning('Failed to load existing results for resume: %s', e)

    executed_runs = list(existing_runs.values())
    matrix = plan.get('ablation_one_factor_matrix', [])
    model = None if dry_run else YOLO(str(checkpoint_path))

    for item in matrix:
        axis = item['axis']
        val = item['value']
        run_key = f"{axis}_{val}"

        if run_key in existing_runs:
            LOG.info('Skipping already completed run: %s', run_key)
            continue

        baseline_cfg = item.get('baseline', {'imgsz': 640, 'conf': 0.35, 'k': 3, 'n': 8})
        current_cfg = dict(baseline_cfg)
        if axis == 'k_n':
            current_cfg['k'] = val[0]
            current_cfg['n'] = val[1]
        else:
            current_cfg[axis] = val

        config_str = json.dumps(current_cfg, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]

        LOG.info('Executing ablation run [%s]: axis=%s, value=%s, cfg=%s', run_key, axis, val, current_cfg)

        run_record: dict[str, Any] = {
            'run_id': f"ablation_{run_key}_{config_hash}",
            'axis': axis,
            'value': val,
            'config': current_cfg,
            'config_hash': config_hash,
            'model_sha256': ckpt_sha,
            'split': split,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'event_level_status': 'BLOCKED' if gt_check['is_blocked'] else 'MEASURED',
            'event_level_block_reason': gt_check['reason'] if gt_check['is_blocked'] else None
        }

        if dry_run:
            run_record.update({
                'status': 'DRY_RUN',
                'map50': None,
                'map50_95': None,
                'smoke_map50': None,
                'fire_map50': None,
            })
        else:
            # Evaluate detector at specified imgsz and conf on validation set
            eval_imgsz = current_cfg['imgsz']
            eval_conf = current_cfg['conf'] if axis == 'conf' else 0.001
            try:
                metrics = model.val(
                    data=str(data_yaml),
                    split=split,
                    imgsz=eval_imgsz,
                    conf=eval_conf,
                    iou=protocol['inference_parameters']['iou_nms'],
                    device='cuda:0',
                    workers=0,
                    plots=False,
                    verbose=False
                )
                box = metrics.box
                ap_idx = list(box.ap_class_index)
                per_class = {}
                for pos, idx in enumerate(ap_idx):
                    c_name = {0: 'smoke', 1: 'fire'}.get(int(idx), str(idx))
                    per_class[c_name] = float(box.ap50[pos])

                run_record.update({
                    'status': 'MEASURED',
                    'map50': float(box.map50),
                    'map50_95': float(box.map),
                    'smoke_map50': per_class.get('smoke', None),
                    'fire_map50': per_class.get('fire', None),
                })
            except Exception as err:
                LOG.error('Evaluation failed for run %s: %s', run_key, err)
                run_record.update({
                    'status': 'ERROR',
                    'error': str(err)
                })

        executed_runs.append(run_record)

    summary = {
        'protocol_version': protocol.get('protocol_version'),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': ckpt_sha,
        'validation_gt_status': gt_check,
        'operating_point_status': protocol.get('operating_point', {}).get('status', 'PROVISIONAL'),
        'total_planned_runs': len(matrix),
        'total_completed_runs': len(executed_runs),
        'runs': executed_runs
    }

    results_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # Write CSV
    if executed_runs:
        fieldnames = [
            'run_id', 'axis', 'value', 'map50', 'map50_95', 'smoke_map50', 'fire_map50',
            'status', 'event_level_status', 'config_hash', 'timestamp'
        ]
        with open(results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in executed_runs:
                writer.writerow(r)

    LOG.info('Ablation sweep completed: %d runs recorded to %s and %s', len(executed_runs), results_csv, results_json)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/research/week6_experiment_plan.json')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/experiments/week6_protocol.yaml')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_c_best.pt'
    )
    parser.add_argument(
        '--data-yaml',
        type=Path,
        default=ROOT / 'runs/benchmarks/week5_domain_adaptation/gpu_training/run_replay_20_percent_seed42/vn.yaml'
    )
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/benchmarks/week6_ablation')
    parser.add_argument('--split', type=str, default='val', help='Evaluation split (strictly val only)')
    parser.add_argument('--no-resume', action='store_true', help='Do not resume from existing results')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup and generate plan without running inference')
    args = parser.parse_args()

    try:
        run_ablation_sweep(
            plan_path=args.plan,
            protocol_path=args.protocol,
            checkpoint_path=args.checkpoint,
            data_yaml=args.data_yaml,
            output_dir=args.output_dir,
            split=args.split,
            resume=not args.no_resume,
            dry_run=args.dry_run
        )
        return 0
    except Exception as err:
        LOG.error('Ablation runner aborted: %s', err)
        return 1


if __name__ == '__main__':
    sys.exit(main())
