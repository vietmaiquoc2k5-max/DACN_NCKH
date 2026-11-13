"""Audit exposure/overlap and publish a provisional source/scene split, without AI inference."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import sha256
from src.preannotation import (POLICY, PROVISIONAL, canonical_hash, classify_exposure,
                              match_frames, propose_split, scene_components, similarity_candidate)
from src.video_audit_v2 import resume_record, write_json

LOGGER = logging.getLogger(__name__)


def validate_candidate_ids(entries: list[dict[str, Any]]) -> None:
    """Reject missing, repeated or unexpected VN_POS identities before any split."""
    if len(entries) != 80 or {r['video_id'] for r in entries} != {f'VN_POS_{i:02d}' for i in range(1, 81)}:
        raise ValueError('Inventory mismatch: expected exactly VN_POS_01..80')


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write real rows, preserving list/dict values as explicit JSON cells."""
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in row.items() if k in fields})


def inventory_record(root: Path, entry: dict[str, Any], audit: dict[str, Any], pixels: dict[str, Any]) -> dict[str, Any]:
    """Measure source identity and keep container metadata separate from decode facts.

    Args:
        root: Authorized repository root.
        entry: One catalog source, including a prior source hash pin.
        audit: Completed strict v2 record, or missing evidence.
        pixels: Current pixel-sampling record with independent stream metadata.

    Returns:
        Inventory fields with None for unmeasured/unverified full duration.
    """
    path = confined(root, entry['relative_path'])
    present = path.is_file()
    digest = sha256(path) if present else None
    matches = digest == entry.get('sha256') if present else False
    bound = present and matches and audit.get('source_sha256') == digest and pixels.get('source_sha256') == digest
    valid = bound and audit.get('technical_eligible') is True and audit.get('decode_status') == 'NORMAL_EOF'
    raw = pixels.get('metadata', {})
    stream = next((s for s in raw.get('streams', []) if s.get('codec_type') == 'video'), {})
    def number(value: Any) -> float | None:
        try:
            return float(Fraction(value)) if value not in (None, 'N/A', '0/0') else None
        except (ValueError, ZeroDivisionError, TypeError):
            return None
    return {
        'video_id': entry['video_id'], 'source_path': entry['relative_path'],
        'relative_path': entry['relative_path'], 'source_sha256': digest,
        'expected_source_sha256': entry.get('sha256'), 'source_available': present,
        'source_sha_verified': matches, 'file_size_bytes': path.stat().st_size if present else None,
        'container': raw.get('format', {}).get('format_name'), 'codec': stream.get('codec_name'),
        'pix_fmt': stream.get('pix_fmt'), 'resolution': [stream.get('width'), stream.get('height')],
        'fps_nominal': number(stream.get('r_frame_rate')), 'fps_average_metadata': number(stream.get('avg_frame_rate')),
        'frame_count_decoded': audit.get('opencv_actual_decoded_frames'),
        'frame_count_ffprobe': audit.get('ffprobe_nb_read_frames'),
        'frame_count_container_metadata': audit.get('nb_frames_metadata'),
        'opencv_reported_frame_count': audit.get('opencv_reported_frame_count'),
        'duration_seconds': audit.get('duration_seconds') if valid else None,
        'duration_container_seconds': number(raw.get('format', {}).get('duration')),
        'duration_stream_metadata_seconds': number(stream.get('duration')),
        'duration_provenance': audit.get('duration_provenance') if valid else 'FULL_SOURCE_DURATION_NOT_VERIFIED',
        'has_audio': pixels.get('has_audio'), 'time_base': audit.get('time_base'),
        'timestamp_provenance': 'SOURCE_PTS' if valid else 'SOURCE_TIMELINE_NOT_ACCEPTED',
        'decode_status': audit.get('decode_status', 'NOT_AUDITED') if bound else 'SOURCE_MISSING_OR_BINDING_MISMATCH',
        'technical_valid': valid, 'decode_exclusion_reasons': audit.get('exclusion_reasons', []),
        'source_group_id': 'SOURCE_' + digest if digest else None,
        'scene_group_id': None, 'historical_exposure_status': 'UNKNOWN_HISTORY',
        'proposed_split': 'UNASSIGNED', 'annotation_status': 'PENDING_HUMAN_REVIEW',
    }


def analyze(run: Path, analysis_id: str, publish: bool, seed: int) -> dict[str, Any]:
    """Compute content hypotheses and write a checksum-bound provisional manifest."""
    if not run.is_relative_to(ROOT / 'runs/benchmarks/vn_pos_preannotation') or not analysis_id.replace('_', '').isalnum():
        raise ValueError('Unsafe analysis directory')
    fp = run / 'fingerprints'
    completed = json.loads((fp / 'completion.json').read_text(encoding='utf-8'))
    technical = json.loads((run / 'technical_audit_80.json').read_text(encoding='utf-8'))
    if not completed['complete'] or completed['errors'] or not technical['complete'] or technical['failures']:
        raise ValueError('Finish and resolve collection errors before publishing the split')
    out = run / ('analysis_' + analysis_id)
    out.mkdir(exist_ok=False)
    (out / 'pair_evidence').mkdir()
    catalog_path = ROOT / 'data/video_catalog_local_v2.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))['videos']
    candidates = [r for r in catalog if r['video_id'].startswith('VN_POS_')]
    validate_candidate_ids(candidates)
    sources = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted((fp / 'videos').glob('*.json'))}
    # Legacy extracted frames may have been chosen around AI-suggested onsets.
    # Keep those extra requests for lineage only. Group/split screening uses the
    # uniformly requested source ordinals, independent of old onset predictions.
    uniform = {}
    uniform_indices = {}
    for vid, source in sources.items():
        count = source['sampling_frame_count']
        requested = set(np.linspace(0, max(0, count-1), min(48, count), dtype=int).tolist())
        uniform_indices[vid] = requested
        uniform[vid] = [(i, frame) for i, frame in enumerate(source['frames']) if frame['requested_frame_index'] in requested]
    write_json(out / 'pixel_sampling_interpretation.json', {
        'split_screen_query_sampling': '48 uniform ordinal requests from source count; no model outputs/onsets',
        'legacy_extra_ordinal_requests': 'Lineage checks only; may originate from historical AI-onset-conditioned extraction',
        'initial_collector_false_flag_means': 'No direct model/onset values read or inference executed; it does not certify legacy extraction was model-independent',
        'prior_image_corpus': 'Compared as existing content for leakage audit, never treated as human GT',
        'no_model_performance_or_semantic_balance_used_to_allocate': True,
    })
    images = json.loads((fp / 'image_fingerprints.json').read_text(encoding='utf-8'))
    old_rows = list(csv.DictReader((ROOT / 'datasets/vn_dataset/split_manifest.csv').open(encoding='utf-8-sig')))
    old_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        old_by_id[row['video_id']].append(row)
    csv_sources = {r['video_id']: r for r in csv.DictReader((ROOT / 'data/video_dataset_list.csv').open(encoding='utf-8-sig'))}
    inventory = []
    for entry in sorted(candidates, key=lambda r:r['video_id']):
        audit_entry = {k: entry[k] for k in ['video_id', 'relative_path', 'sha256']}
        record = resume_record(run / 'records' / entry['video_id'], audit_entry, ROOT)
        row = inventory_record(ROOT, entry, record, sources[entry['video_id']])
        row.update(audit_record_path=(run / 'records' / entry['video_id'] / 'record.json').relative_to(ROOT).as_posix(),
                   audit_record_sha256=sha256(run / 'records' / entry['video_id'] / 'record.json'),
                   legacy_splits=sorted({x['split'] for x in old_by_id[row['video_id']]}))
        inventory.append(row)
    by_id = {row['video_id']: row for row in inventory}
    edges: list[tuple[str, str]] = []
    exact = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for vid, source in sources.items():
        hashes[source['source_sha256']].append(vid)
    for digest, ids in sorted(hashes.items()):
        for first, second in itertools.combinations(sorted(ids), 2):
            if first in by_id or second in by_id:
                edges.append((first, second))
                exact.append({'source_sha256': digest, 'first_video_id': first, 'second_video_id': second, 'relation': 'EXACT_FILE_BYTES_DUPLICATE'})
    near = []
    for vid in sorted(by_id):
        for other in sorted(sources):
            if other == vid or (other in by_id and other < vid) or sources[vid]['source_sha256'] == sources[other]['source_sha256']:
                continue
            pairs = match_frames([frame for _,frame in uniform[vid]], [frame for _,frame in uniform[other]])
            relation = similarity_candidate(pairs)
            if relation is None:
                continue
            edges.append((vid, other))
            evidence = out / 'pair_evidence' / f'{vid}__{other}.json'
            for pair in pairs:
                pair['first_source_sample_index'] = uniform[vid][pair['first']][0]
                pair['second_source_sample_index'] = uniform[other][pair['second']][0]
            write_json(evidence, {'first_video_id': vid, 'second_video_id': other, **relation, 'matched_pixel_observations': pairs,
                                  'position_provenance': sources[vid]['position_provenance'], 'not_human_or_semantic_verification': True})
            near.append({'first_video_id': vid, 'second_video_id': other, **relation, 'evidence_ref': evidence.relative_to(ROOT).as_posix()})
        LOGGER.info('SOURCE_SIMILARITY %s candidates=%s', vid, len(near))
    image_hits = []
    lineage = defaultdict(int)
    image_foreign: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for vid in sorted(by_id):
        for start in range(0, len(images), 512):
            chunk = images[start:start+512]
            pairs = match_frames(sources[vid]['frames'], chunk)
            for pair in pairs:
                image = chunk[pair['second']]
                if image['source_video_id_hint'] == vid and image['corpus'] == 'datasets/vn_dataset/images':
                    lineage[vid] += 1
                    continue
                if sources[vid]['frames'][pair['first']]['requested_frame_index'] not in uniform_indices[vid]:
                    continue
                row = {'video_id': vid, 'image_path': image['relative_path'], 'image_sha256': image['image_sha256'],
                       'image_corpus': image['corpus'], 'image_declared_split': image['declared_split'],
                       'reference_video_id_hint': image['source_video_id_hint'], **pair}
                image_hits.append(row)
                hint = image['source_video_id_hint']
                reference = hint if hint and hint != vid else ('IMAGE_' + image['image_sha256'])
                image_foreign[(vid, reference)].append({**pair, 'second': start + pair['second']})
        LOGGER.info('IMAGE_SIMILARITY %s foreign_pixel_hits=%s', vid, len(image_hits))
    external_nodes = set(sources)
    for (vid, reference), pairs in image_foreign.items():
        relation = similarity_candidate(pairs)
        if relation:
            external_nodes.add(reference)
            edges.append((vid, reference))
    groups = scene_components(external_nodes, edges)
    group_members: dict[str, list[str]] = defaultdict(list)
    for vid, group in groups.items():
        group_members[group].append(vid)
    exposure = []
    for row in inventory:
        vid = row['video_id']; declared = csv_sources.get(vid, {})
        # Actual saved onset records explicitly marked AI are prior model use by ID.
        # Original runtime source hashes are absent; never claim authenticated train history.
        notes = declared.get('notes', '')
        model_onset_record = 'AI_GOI_Y_MOC' in notes or 'AI_DO_MOC' in notes
        flags = {'KNOWN_EVALUATION_EXPOSURE'} if model_onset_record else set()
        status = classify_exposure(flags, history_available=False)
        group = groups[vid]
        row.update(scene_group_id=group, historical_exposure_status=status,
                   external_scene_overlap=any(member not in by_id for member in group_members[group]))
        exposure.append({'video_id': vid, 'source_sha256': row['source_sha256'], 'historical_exposure_status': status,
                         'legacy_frame_split': row['legacy_splits'], 'legacy_frames': len(old_by_id[vid]),
                         'recorded_ai_onset_use_by_video_id': model_onset_record, 'historical_runtime_source_sha_available': False,
                         'actual_training_use': 'UNKNOWN_HISTORY_ORIGINAL_CORPUS_HASHES_UNAVAILABLE',
                         'parameter_tuning_use': 'UNKNOWN_HISTORY', 'proven_unseen': False,
                         'scene_group_id': group, 'external_scene_overlap': row['external_scene_overlap'],
                         'evidence_ref': 'data/video_dataset_list.csv; datasets/vn_dataset/split_manifest.csv; scripts/auto_detect_timestamps.py; scripts/process_new_vn_batches.py',
                         'evidence_strength': 'ID_BASED_RECORDED_AI_ONSETS; NOT_RUNTIME_SHA_AUTHENTICATED',
                         'model_onset_values_or_predictions_published': False})
    split = propose_split(inventory, seed)
    proposed = {row['video_id']: row['proposed_split'] for row in split}
    for row in inventory:
        row['proposed_split'] = proposed[row['video_id']]
    timestamp = datetime.now(timezone.utc).isoformat()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    wrapper = {'status': PROVISIONAL, 'policy': POLICY, 'seed': seed, 'head': head, 'created_at_utc': timestamp,
               'input_catalog_sha256': sha256(catalog_path), 'videos': split,
               'rule': 'Preserve legacy source reservations when whole scene group agrees; quarantine conflicts/invalid/external overlap; seeded duration allocation only for previously unassigned groups',
               'independence_verified': False, 'model_inference_executed_for_this_split': False,
               'human_annotation_status': 'HUMAN_ANNOTATION_PENDING', 'camera_metadata_available': False}
    wrapper['manifest_sha256'] = canonical_hash(wrapper)
    names = []
    def export_json(name: str, value: Any) -> None:
        write_json(out / name, value); names.append(name)
    def export_csv(name: str, values: list[dict[str, Any]], fields: list[str]) -> None:
        write_csv(out / name, values, fields); names.append(name)
    export_json('vn_pos_preannotation_inventory.json', {'expected_videos': 80, 'found_sources': sum(r['source_available'] for r in inventory), 'videos': inventory})
    export_csv('vn_pos_preannotation_inventory.csv', inventory, list(inventory[0]))
    export_csv('vn_pos_exact_duplicates.csv', exact, ['source_sha256', 'first_video_id', 'second_video_id', 'relation'])
    export_csv('vn_pos_near_duplicate_candidates.csv', near, ['first_video_id', 'second_video_id', 'relation', 'status', 'distinct_matched_frames', 'ordered_chain_length', 'best_phash_distance', 'human_confirmed', 'evidence_ref'])
    scene_rows = [{'video_id': r['video_id'], 'source_group_id': r['source_group_id'], 'scene_group_id': r['scene_group_id'],
                   'members': sorted(group_members[r['scene_group_id']]), 'camera_id': None,
                   'relation_status': 'SCENE_RELATION_UNCERTAIN' if len(group_members[r['scene_group_id']]) > 1 else 'SOURCE_IDENTITY_ONLY_CAMERA_UNKNOWN',
                   'source_metadata_missing': True} for r in inventory]
    export_csv('vn_pos_scene_groups.csv', scene_rows, list(scene_rows[0]))
    scene_review = [{'video_id': r['video_id'], 'scene_group_id': r['scene_group_id'], 'related_sources': sorted(group_members[r['scene_group_id']]),
                     'status': 'PENDING_HUMAN_SCENE_REVIEW', 'reason': 'UNKNOWN_SOURCE_CAMERA_METADATA' + ('; PIXEL_RELATION_REQUIRES_CONFIRMATION' if len(group_members[r['scene_group_id']]) > 1 else ''),
                     'must_not_cross_splits_until_resolved': True} for r in inventory]
    export_csv('vn_pos_scene_review_required.csv', scene_review, list(scene_review[0]))
    export_csv('vn_pos_historical_exposure_audit.csv', exposure, list(exposure[0]))
    export_json('vn_pos_preannotation_split_manifest.json', wrapper)
    export_csv('vn_pos_preannotation_split_manifest.csv', split, list(split[0]))
    provenance = {
        'status': PROVISIONAL, 'head': head, 'timestamp_utc': timestamp, 'policy': POLICY, 'seed': seed,
        'input_catalog_sha256': sha256(catalog_path), 'manifest_canonical_sha256': wrapper['manifest_sha256'],
        'manifest_json_sha256': sha256(out / 'vn_pos_preannotation_split_manifest.json'),
        'manifest_csv_sha256': sha256(out / 'vn_pos_preannotation_split_manifest.csv'),
        'input_sha256': {name: sha256(ROOT / name) for name in ['data/video_dataset_list.csv', 'datasets/vn_dataset/split_manifest.csv', 'datasets/vn_dataset/review_status.csv', 'data_dfire.yaml', 'data_vn.yaml']},
        'implementation_sha256': {name: sha256(ROOT / name) for name in ['src/preannotation.py', 'scripts/prepare_vn_pos_preannotation.py', 'scripts/fingerprint_vn_pos_sources.py', 'scripts/audit_vn_pos_decode.py']},
        'payload_sha256': {name: sha256(out / name) for name in names},
        'independence_verified': False, 'split_manifest_bytes_frozen_for_annotation': True,
        'model_outputs_or_semantic_labels_used_to_allocate': False,
        'historical_ai_exposure_used_only_to_flag_non_independence': True,
        'scene_similarity_is_hypothesis_not_human_confirmation': True,
        'after_annotation_policy': 'Do not rebalance using model/test performance; publish a new explicit manifest version for label-coverage issues, retaining source groups, prior manifest hash and reasons',
    }
    export_json('vn_pos_split_provenance.json', provenance)
    write_csv(out / 'image_overlap_candidates.csv', image_hits, ['video_id', 'image_path', 'image_sha256', 'image_corpus', 'image_declared_split', 'reference_video_id_hint', 'first', 'second', 'phash_distance', 'dhash_distance', 'color_intersection'])
    summary = {
        'status': 'PARTIAL', 'inventory_status': 'INVENTORY_PASS' if all(r['source_available'] and r['source_sha_verified'] for r in inventory) else 'INVENTORY_PARTIAL',
        'split_status': PROVISIONAL, 'split_ready': 'PENDING_HUMAN_SCENE_REVIEW_AND_ANNOTATION',
        'test_independence_status': 'TEST_INDEPENDENCE_NOT_VERIFIED', 'human_annotation_status': 'HUMAN_ANNOTATION_PENDING',
        'expected': 80, 'found': sum(r['source_available'] for r in inventory), 'technical_valid': sum(r['technical_valid'] for r in inventory),
        'decode_counts': dict(Counter(r['decode_status'] for r in inventory)),
        'technical_valid_duration_seconds': sum(r['duration_seconds'] or 0 for r in inventory),
        'source_sha_verified': sum(r['source_sha_verified'] for r in inventory),
        'exact_duplicate_pairs': len(exact), 'near_duplicate_candidate_pairs': len(near),
        'scene_groups': len({r['scene_group_id'] for r in inventory}),
        'uncertain_multi_source_groups': sum(len(group_members[g]) > 1 for g in {r['scene_group_id'] for r in inventory}),
        'foreign_image_pixel_hits': len(image_hits), 'own_source_pixel_lineage_hits': dict(lineage),
        'historical_exposure_counts': dict(Counter(r['historical_exposure_status'] for r in inventory)),
        'proposed_split_counts': dict(Counter(r['proposed_split'] for r in split)),
        'group_leakage': 0, 'source_leakage': 0, 'manifest_sha256': wrapper['manifest_sha256'],
        'negative_independent_test': 'BLOCKED', 'verified_independent_negative_hours': 0,
        'independent_negative_hours_required_additional': 5,
        'detector_inference_executed': False, 'training_executed': False,
        'analysis_directory': out.relative_to(ROOT).as_posix(),
    }
    write_json(out / 'summary.json', summary)
    if publish:
        targets = [ROOT / 'data' / name for name in names]
        if any(path.exists() for path in targets):
            raise FileExistsError('Published preannotation files exist; do not overwrite a frozen manifest. Use a versioned publication after review.')
        for name, target in zip(names, targets):
            with target.open('xb') as stream:
                stream.write((out / name).read_bytes())
    print('PREANNOTATION_ANALYZED', json.dumps(summary), flush=True)
    return summary


def main() -> None:
    """Publish an auditable provisional manifest only after actual source collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--analysis-id', default='v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    analyze(args.run_dir.resolve(), args.analysis_id, args.publish, args.seed)


if __name__ == '__main__':
    main()
