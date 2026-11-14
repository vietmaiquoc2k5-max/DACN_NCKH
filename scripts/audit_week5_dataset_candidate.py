"""Audit an explicitly supplied corpus without promoting it to D-Fire ground truth."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ground_truth_studio.integrity_service import sha256
from src.domain_adaptation import read_yolo_boxes

LOG = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write actual records to a fresh metadata file."""
    if not rows:
        return
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit(candidate: Path, output: Path) -> dict[str, Any]:
    """Hash and inspect a candidate while preserving all source bytes.

    Args:
        candidate: Existing directory explicitly supplied by the operator.
        output: New metadata directory inside the Week 5 benchmark root.

    Returns:
        Real counts, label/decode findings and unresolved identity/split blockers.
    """
    if not candidate.is_dir() or output.is_relative_to(candidate):
        raise ValueError('Invalid candidate or output inside source corpus')
    if not output.is_relative_to(ROOT / 'runs/benchmarks/week5_domain_adaptation'):
        raise ValueError('Output must be inside the Week 5 metadata root')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files = sorted(p for p in candidate.rglob('*') if p.is_file())
    snapshots = [{'relative_path': p.relative_to(candidate).as_posix(),
                  'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files]
    digest_by_path = {r['relative_path']: r['sha256'] for r in snapshots}
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [p for p in files if p.suffix.lower()=='.txt' and p.parent.name=='labels']
    label_set = set(labels)
    parsed, errors, class_counts = {}, [], Counter()
    for path in labels:
        name = path.relative_to(candidate).as_posix()
        try:
            parsed[path] = read_yolo_boxes(path)
            class_counts.update(int(row[0]) for row in parsed[path])
        except (ValueError, OSError) as error:
            errors.append({'relative_path': name, 'error': str(error)})
    by_digest, by_base = defaultdict(list), defaultdict(list)
    image_rows, unreadable, matched_labels = [], [], set()
    for index, image in enumerate(images, 1):
        relative = image.relative_to(candidate).as_posix()
        expected = image.parent.parent / 'labels' / (image.stem+'.txt')
        exists = expected in label_set
        if exists:
            matched_labels.add(expected)
        decoded = cv2.imread(str(image))
        if decoded is None:
            unreadable.append(relative)
        digest = digest_by_path[relative]
        base = image.stem.split('.rf.')[0]
        by_digest[digest].append(relative)
        by_base[base].append(relative)
        image_rows.append({'relative_path': relative, 'image_sha256': digest,
            'width': decoded.shape[1] if decoded is not None else None,
            'height': decoded.shape[0] if decoded is not None else None,
            'decode_valid': decoded is not None, 'label_exists': exists,
            'label_path': expected.relative_to(candidate).as_posix() if exists else '',
            'label_sha256': digest_by_path[expected.relative_to(candidate).as_posix()] if exists else '',
            'label_valid': expected in parsed if exists else False,
            'bbox_count': len(parsed[expected]) if expected in parsed else None,
            'filename_base_hypothesis': base, 'approved_for_training': False})
        if index % 1000==0:
            LOG.info('Inspected %s/%s actual images', index, len(images))
    duplicates = [{'sha256': key, 'paths': value} for key, value in by_digest.items() if len(value)>1]
    flat_hashes = {r['image_sha256'] for r in image_rows if not r['label_exists']}
    paired_hashes = {r['image_sha256'] for r in image_rows if r['label_exists']}
    configs = {}
    for name in ['data_legacy.yaml', 'data_dfire.yaml']:
        path = ROOT / name
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        configs[name] = {'sha256': sha256(path), 'dataset_path': data.get('path'),
                         'train': data.get('train'), 'val': data.get('val'),
                         'class_mapping': data.get('names')}
    write_csv(output/'source_files.csv', snapshots)
    write_csv(output/'image_label_inventory.csv', image_rows)
    # Verify again after reading pixels; the source is never repaired in place.
    for row in snapshots:
        if sha256(candidate / row['relative_path'])!=row['sha256']:
            raise ValueError('SOURCE_CHANGED_DURING_AUDIT: '+row['relative_path'])
    if {p.relative_to(candidate).as_posix() for p in candidate.rglob('*') if p.is_file()}!=set(digest_by_path):
        raise ValueError('SOURCE_FILE_SET_CHANGED_DURING_AUDIT')
    result = {
        'audit_status': 'DONE', 'corpus_identity_status': 'NOT VERIFIED',
        'training_status': 'BLOCKED', 'candidate': str(candidate),
        'head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': time.perf_counter()-started,
        'source_files': len(files), 'images': len(images), 'labels': len(labels),
        'image_counts_by_directory': dict(Counter(p.parent.relative_to(candidate).as_posix() for p in images)),
        'images_with_adjacent_label': sum(r['label_exists'] for r in image_rows),
        'images_without_label': sum(not r['label_exists'] for r in image_rows),
        'missing_labels_treated_as_negative': False,
        'empty_label_files': sum(not boxes for boxes in parsed.values()),
        'valid_label_files': len(parsed), 'invalid_label_files': errors,
        'orphan_label_files': [p.relative_to(candidate).as_posix() for p in labels if p not in matched_labels],
        'numeric_class_box_counts': dict(class_counts), 'unreadable_images': unreadable,
        'rf_named_images': sum('.rf.' in p.name for p in images),
        'exact_image_duplicate_groups': duplicates,
        'unlabelled_image_hashes_also_in_paired_images': len(flat_hashes & paired_hashes),
        'repeated_filename_base_groups': sum(len(v)>1 for v in by_base.values()),
        'filename_groups_are_source_or_scene_proof': False,
        'configs': configs,
        'legacy_train_equals_validation': configs['data_legacy.yaml']['train']==configs['data_legacy.yaml']['val'],
        'class_mapping_matches_baseline': configs['data_legacy.yaml']['class_mapping']==configs['data_dfire.yaml']['class_mapping'],
        'user_confirmed_source_url': 'https://github.com/gaia-solutions-on-demand/DFireDataset',
        'local_files_authenticated_against_author_manifest': False,
        'source_readme_or_download_manifest_found': any(p.name.lower().startswith('readme') or p.suffix.lower() in {'.yaml','.yml','.json'} for p in files),
        'vn_bbox_export_available': False,
        'original_files_unchanged': True, 'labels_remapped_or_created': False,
        'detector_or_training_executed': False, 'map': None,
        'blockers': ['LOCAL_DFIRE_PROVENANCE_NOT_AUTHENTICATED', 'TRAIN_VAL_PATH_ALIAS',
                     'CLASS_MAPPING_REVERSED_RELATIVE_TO_BASELINE', 'VN_BBOX_EXPORT_MISSING'],
    }
    with (output/'candidate_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output/'artifact_sha256.json').open('x', encoding='utf-8') as stream:
        json.dump({p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='artifact_sha256.json'},stream,indent=2)
    return result


def main() -> None:
    """Run a read-only corpus audit with explicit input and fresh output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    result = audit(args.candidate_root.resolve(), args.output_dir.resolve())
    print(json.dumps({key:result[key] for key in ['audit_status','images','labels','images_without_label',
        'valid_label_files','legacy_train_equals_validation','corpus_identity_status','training_status','duration_seconds']}), flush=True)


if __name__=='__main__':
    main()
