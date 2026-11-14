"""Audit real Week 5 inputs and publish a blocked, versioned bbox review queue."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import sha256
from src.domain_adaptation import (configure_parameters, read_yolo_boxes,
                                   restrict_optimizer, state_digest,
                                   validate_class_mapping, verify_file_hash)
from src.preannotation import match_frames
from src.video_audit_v2 import write_json

LOG = logging.getLogger(__name__)
BASELINE = 'runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt'
BASELINE_SHA = 'f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6'
PREVIOUS = ROOT / 'runs/benchmarks/vn_pos_preannotation/run_20261114_6942fa56'


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write metadata to a new file without editing original labels."""
    if not rows:
        raise ValueError('A review queue must contain actual image records')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_preflight() -> dict[str, Any]:
    """Load the authentic baseline and inspect real parameter/optimizer coverage."""
    path = ROOT / BASELINE
    verify_file_hash(path, BASELINE_SHA)
    wrapper = YOLO(str(path))
    model = wrapper.model.float().cpu()
    validate_class_mapping(model.names)
    policies = {}
    for strategy in ['full_finetune', 'freeze_backbone', 'replay_20_percent']:
        policy = configure_parameters(model, strategy)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.937,
                                    weight_decay=.0005)
        restrict_optimizer(optimizer, model)
        policies[strategy] = {
            key: value for key, value in policy.items()
            if key not in {'frozen_parameter_names', 'trainable_parameter_names'}
        }
        policies[strategy]['optimizer_trainable_parameters'] = sum(
            p.numel() for group in optimizer.param_groups for p in group['params'])
        policies[strategy]['frozen_state_digest'] = state_digest(
            model, tuple(policy['backbone_prefixes'])) if policy['backbone_prefixes'] else None
        policies[strategy]['real_training_executed'] = False
        policies[strategy]['post_training_unchanged_verified'] = None
    memory = {}
    try:
        import psutil
        ram = psutil.virtual_memory()
        memory = {'ram_total_bytes': ram.total, 'ram_available_bytes': ram.available}
    except ImportError:
        memory = {'ram_total_bytes': None, 'ram_available_bytes': None}
    cuda = torch.cuda.is_available()
    verify_file_hash(path, BASELINE_SHA)
    return {
        'checkpoint': BASELINE, 'checkpoint_sha256': BASELINE_SHA,
        'checkpoint_bytes_unchanged': True, 'loaded': True,
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'class_mapping': model.names, 'backbone_modules': len(model.yaml['backbone']),
        'parameter_policies_checked_on_actual_baseline': policies,
        'cuda_available': cuda, 'device_used_for_preflight': 'cpu',
        'vram_bytes': torch.cuda.get_device_properties(0).total_memory if cuda else None,
        'memory': memory, 'training_budget_estimate': None,
        'training_budget_status': 'REQUIRES_VALID_DATA_AND_TIMED_PILOT',
        'cpu_fallback_supported': True, 'baseline_vn_map': None, 'baseline_dfire_map': None,
        'map_status': 'BLOCKED_DATA_READINESS',
    }


def contact_sheets(rows: list[dict[str, Any]], output: Path) -> list[str]:
    """Show actual empty-label images without model predictions or event overlays."""
    output.mkdir()
    names = []
    for page, first in enumerate(range(0, len(rows), 24), start=1):
        canvas = np.full((6*210, 4*320, 3), 245, np.uint8)
        for position, row in enumerate(rows[first:first+24]):
            image = cv2.imread(str(confined(ROOT, row['image_path'])))
            if image is None:
                raise ValueError('UNREADABLE_REVIEW_IMAGE')
            scale = min(310/image.shape[1], 172/image.shape[0])
            small = cv2.resize(image, (max(1, round(image.shape[1]*scale)),
                                       max(1, round(image.shape[0]*scale))))
            y, x = divmod(position, 4)
            top, left = y*210+5, x*320+5
            canvas[top:top+small.shape[0], left:left+small.shape[1]] = small
            cv2.putText(canvas, row['image_name'], (left, y*210+194),
                        cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 0, 0), 1, cv2.LINE_AA)
        path = output / f'empty_labels_{page:02d}.jpg'
        if not cv2.imwrite(str(path), canvas):
            raise OSError('CONTACT_SHEET_WRITE_FAILED')
        names.append(path.relative_to(ROOT).as_posix())
    return names


def prepare(output: Path, discovery_path: Path, closure: Path) -> dict[str, Any]:
    """Inspect current bytes and keep all unverified images outside training.

    Args:
        output: Fresh version directory below week5_domain_adaptation.
        discovery_path: Actual bounded D-Fire discovery receipt.
        closure: Authenticated Week 4 GT audit output.

    Returns:
        Dataset readiness and precise blockers, with no invented annotation signoff.
    """
    if not output.is_relative_to(ROOT / 'runs/benchmarks/week5_domain_adaptation'):
        raise ValueError('Unsafe output directory')
    output.mkdir(parents=True, exist_ok=False)
    discovery = json.loads(discovery_path.read_text(encoding='utf-8'))
    # A discovery receipt cannot promote an unaudited D-Fire corpus to training data.
    dfire = {'status': discovery['dfire_status'], 'locations_checked': discovery['dfire_locations'],
             'workspace_directories_visited': discovery['workspace_directories_visited'],
             'errors': discovery['errors'], 'discovery_receipt_sha256': sha256(discovery_path),
             'discovery_scope': discovery['scope'], 'content_manifest_sha256': None,
             'train_images_verified': 0, 'evaluation_images_verified': 0,
             'required': 'Original D-Fire images, YOLO labels and train/val/test split provenance'}
    write_json(output / 'dfire_availability.json', dfire)
    manifest = ROOT / 'datasets/vn_dataset/split_manifest.csv'
    review = ROOT / 'datasets/vn_dataset/review_status.csv'
    inputs = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p)
              for p in [manifest, review, ROOT/'data/vn_pos_preannotation_split_manifest.json',
                        ROOT/'data/vn_pos_scene_review_required.csv']}
    legacy = list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    reviewed = {r['image_name']: r for r in csv.DictReader(review.open(encoding='utf-8-sig'))}
    split = json.loads((ROOT/'data/vn_pos_preannotation_split_manifest.json').read_text(encoding='utf-8'))
    split_by_id = {r['video_id']: r for r in split['videos']}
    sources = {p.stem: json.loads(p.read_text(encoding='utf-8'))
               for p in (PREVIOUS/'fingerprints/videos').glob('*.json')}
    # Current source hashes identify the available asset, not the historical extractor's runtime bytes.
    for source in sources.values():
        verify_file_hash(confined(ROOT, source['source_path']), source['source_sha256'])
    risk = json.loads((PREVIOUS/'week5_legacy_semantic_gaps.json').read_text(encoding='utf-8'))
    risk_ids = {r['video_id'] for r in risk['risk_videos']}
    candidates, high_risk, bbox_queue, errors = [], [], [], []
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in legacy:
        vid, name, declared_split = row['video_id'], row['image_name'], row['split']
        image = ROOT/'datasets/vn_dataset/images'/declared_split/name
        label = ROOT/'datasets/vn_dataset/labels'/declared_split/(Path(name).stem+'.txt')
        boxes, error = [], ''
        try:
            boxes = read_yolo_boxes(label)
            if cv2.imread(str(image)) is None:
                raise ValueError('IMAGE_DECODE_FAILED')
        except (ValueError, OSError) as exc:
            error = str(exc)
            errors.append({'image_name': name, 'error': error})
        pin = split_by_id.get(vid)
        source = sources.get(vid, {})
        current_sha = source.get('source_sha256', '')
        source_path = source.get('source_path', '')
        proposed = pin['proposed_split'] if pin else declared_split.upper()+'_LEGACY_RESERVATION'
        scene = pin['scene_group_id'] if pin else 'UNCONFIRMED_'+vid
        declared_index = int(row['frame_idx'])
        count = source.get('sampling_frame_count')
        in_bounds = 0 <= declared_index < count if count is not None else None
        item = {
            'image_name': name, 'video_id': vid, 'legacy_split': declared_split,
            'reserved_split': proposed, 'image_path': image.relative_to(ROOT).as_posix(),
            'label_path': label.relative_to(ROOT).as_posix(),
            'image_sha256': sha256(image), 'label_sha256': sha256(label),
            'label_syntax_and_bounds_valid': not error, 'bbox_count': len(boxes),
            'empty_label': not boxes and not error,
            'source_path_candidate': source_path, 'current_candidate_source_sha256': current_sha,
            'scene_group_candidate': scene, 'scene_human_confirmed': False,
            'source_frame_index_declared': declared_index,
            'timestamp_sec_declared': row['timestamp_sec'], 'declared_index_in_sampling_bounds': in_bounds,
            'frame_source_lineage': 'UNVERIFIED_LEGACY_EXTRACTION;NO_RUNTIME_SOURCE_SHA_OR_SOURCE_PTS',
            'legacy_reviewed_flag': reviewed.get(name, {}).get('is_reviewed', ''),
            'bbox_review_status': 'BBOX_GROUND_TRUTH_REQUIRED', 'approved_for_training': False,
        }
        candidates.append(item)
        groups[('image_sha256', item['image_sha256'])].add(declared_split)
        groups[('video_id', vid)].add(declared_split)
        groups[('candidate_source', current_sha or vid)].add(declared_split)
        if vid in risk_ids and item['empty_label']:
            high_risk.append(item)
        if proposed in {'TRAIN_CANDIDATE', 'VALIDATION_CANDIDATE', 'TRAIN_LEGACY_RESERVATION', 'VAL_LEGACY_RESERVATION'}:
            bbox_queue.append(item)
    write_csv(output/'vn_image_candidates.csv', candidates)
    write_csv(output/'empty_label_review_129.csv', high_risk)
    write_csv(output/'bbox_annotation_queue.csv', bbox_queue)
    LOG.info('Audited %s images/labels; %s high-risk empty labels; %s bbox queue images',
             len(candidates), len(high_risk), len(bbox_queue))
    image_sha = {r['image_sha256'] for r in candidates}
    fingerprints = json.loads((PREVIOUS/'fingerprints/image_fingerprints.json').read_text(encoding='utf-8'))
    fingerprints = [r for r in fingerprints if r['corpus']=='datasets/vn_dataset/images'
                    and r['image_sha256'] in image_sha]
    if len(fingerprints) != len(candidates):
        raise ValueError('CACHED_PIXEL_EVIDENCE_DOES_NOT_COVER_CURRENT_IMAGES')
    # Retain hypotheses across legacy splits. No hash similarity can certify independence.
    near = []
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        a = [r for r in fingerprints if r['declared_split']==left]
        b = [r for r in fingerprints if r['declared_split']==right]
        for hit in match_frames(a, b):
            first, second = a[hit['first']], b[hit['second']]
            near.append({'image_a': first['relative_path'], 'image_b': second['relative_path'],
                         'split_a': left, 'split_b': right,
                         'phash_distance': hit['phash_distance'], 'dhash_distance': hit['dhash_distance'],
                         'color_intersection': hit['color_intersection'],
                         'status': 'PIXEL_SIMILARITY_CANDIDATE_REQUIRES_HUMAN_SCENE_REVIEW'})
    if near:
        write_csv(output/'cross_split_pixel_candidates.csv', near)
    sheets = contact_sheets(high_risk, output/'contact_sheets')
    leaks = [{'identity_kind': field, 'identity': identity, 'splits': sorted(values)}
             for (field, identity), values in groups.items() if len(values)>1]
    readiness = {
        'status': 'BLOCKED_BBOX_AND_LINEAGE_REQUIRED', 'version': output.name,
        'images': len(candidates), 'counts_by_legacy_split': dict(Counter(r['legacy_split'] for r in candidates)),
        'label_errors': errors, 'exact_cross_split_leakage': leaks,
        'cross_split_pixel_candidate_pairs': len(near), 'near_duplicate_independence_proven': False,
        'human_hash_bound_bbox_approvals_found': 0, 'approved_train_images': 0, 'approved_val_images': 0,
        'approved_test_images': 0, 'bbox_review_queue_images': len(bbox_queue),
        'empty_label_risk_images': len(high_risk), 'empty_label_risk_videos': len(risk_ids),
        'all_risk_frames_assumed_wrong': False, 'human_review_still_required': True,
        'contact_sheets': sheets, 'event_gt_does_not_approve_bboxes': True,
        'legacy_source_runtime_hash_available': False, 'source_pts_lineage_verified': False,
        'test_candidates_used_for_training': False, 'original_images_or_labels_changed': False,
        'input_sha256': inputs,
    }
    write_json(output/'vn_dataset_readiness.json', readiness)
    # This is an explicit blocked version, not an empty but "valid" training corpus.
    write_json(output/'dataset_manifest.json', {
        'schema_version': 1, 'dataset_version': output.name,
        'status': 'BLOCKED_NOT_A_TRAINABLE_DATASET', 'class_mapping': {'0': 'smoke', '1': 'fire'},
        'rows': [], 'candidate_manifest': 'vn_image_candidates.csv',
        'candidate_manifest_sha256': sha256(output/'vn_image_candidates.csv'),
        'prior_split_manifest_sha256': inputs['data/vn_pos_preannotation_split_manifest.json'],
        'event_gt_scope_sha256': sha256(closure/'vn_pos_gt_scope.csv'),
        'event_gt_events_sha256': sha256(closure/'vn_pos_gt_events.jsonl'),
        'split_assignment_changed': False,
        'blockers': ['BBOX_GROUND_TRUTH_REQUIRED', 'FRAME_SOURCE_LINEAGE_NOT_VERIFIED',
                     'SCENE_RELATIONS_REQUIRE_REVIEW', 'BLOCKED_DATASET_MISSING_DFIRE'],
    })
    preflight = baseline_preflight()
    write_json(output/'preflight.json', preflight)
    for name, digest in inputs.items():
        verify_file_hash(ROOT/name, digest)
    result = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'training_executed': False,
              'status': 'BLOCKED_DATA_READINESS', 'vn_dataset': readiness['status'],
              'dfire': dfire['status'], 'baseline_checkpoint_verified': preflight['loaded'],
              'original_inputs_unchanged': True, 'selected_model': None}
    write_json(output/'summary.json', result)
    return result


def main() -> None:
    """Run read-only preparation with a fresh version directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--discovery-receipt', required=True, type=Path)
    parser.add_argument('--closure-dir', required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    print(json.dumps(prepare(args.output_dir.resolve(), args.discovery_receipt.resolve(),
                             args.closure_dir.resolve())), flush=True)


if __name__ == '__main__':
    main()
