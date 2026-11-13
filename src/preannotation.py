"""Content-only similarity and conservative, group-safe pre-annotation contracts."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable

import cv2
import numpy as np

POLICY = 'vn_pos_group_preannotation_v1'
PROVISIONAL = 'PRE_ANNOTATION_SPLIT_PROVISIONAL'
SPLITS = {'TRAIN_CANDIDATE', 'VALIDATION_CANDIDATE', 'INDEPENDENT_TEST_CANDIDATE', 'BLOCKED_UNASSIGNED'}
EXPOSURE_PRIORITY = ['KNOWN_TRAIN', 'KNOWN_TUNING_EXPOSURE', 'KNOWN_VALIDATION', 'KNOWN_EVALUATION_EXPOSURE']


def canonical_hash(value: Any) -> str:
    """Hash a JSON value with a stable, explicit representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def fingerprint(image: np.ndarray) -> dict[str, Any]:
    """Describe real pixels using three spatial views; never predict semantic labels.

    Args:
        image: A decoded BGR image, kept only for this function call.

    Returns:
        DCT pHash, gradient dHash and texture per view, plus an HSV histogram.
    """
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 8:
        raise ValueError('An actual decoded color frame is required')
    height, width = image.shape[:2]
    variants = []
    for margin in (0.0, .1, .2):
        y, x = int(height * margin), int(width * margin)
        gray = cv2.cvtColor(image[y:height-y, x:width-x], cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        coefficients = cv2.dct(small.astype(np.float32))[:8, :8].flatten()[1:]
        phash = sum(int(bit) << index for index, bit in enumerate(coefficients > np.median(coefficients)))
        grad = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        dhash = sum(int(bit) << index for index, bit in enumerate((grad[:, 1:] > grad[:, :-1]).flatten()))
        variants.append({'phash': f'{phash:016x}', 'dhash': f'{dhash:016x}', 'std': float(np.std(small))})
    hsv = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 2], [0, 180, 0, 256]).flatten()
    hist /= max(float(hist.sum()), 1)
    return {'variants': variants, 'color_hist': [float(value) for value in hist]}


def match_frames(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find high-similarity pixel observations, rejecting low-information frames.

    Args:
        first: Pixel fingerprints in temporal order from one source.
        second: Fingerprints from another source, or ordered still-image references.

    Returns:
        Candidate frame pairs. These are hypotheses, never exact-duplicate proof.
    """
    if not first or not second:
        return []
    a = [(i, v) for i, row in enumerate(first) for v in row['variants'] if v['std'] >= 10]
    b = [(i, v) for i, row in enumerate(second) for v in row['variants'] if v['std'] >= 10]
    if not a or not b:
        return []
    ap = np.array([int(v['phash'], 16) for _, v in a], dtype=np.uint64)
    bp = np.array([int(v['phash'], 16) for _, v in b], dtype=np.uint64)
    xor = ap[:, None] ^ bp[None, :]
    if hasattr(np, 'bitwise_count'):
        distances = np.bitwise_count(xor)
    else:
        distances = np.unpackbits(xor.view(np.uint8).reshape(*xor.shape, 8), axis=-1).sum(axis=-1)
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for ai, bi in zip(*np.where(distances <= 6)):
        i, av = a[ai]; j, bv = b[bi]
        dh = (int(av['dhash'], 16) ^ int(bv['dhash'], 16)).bit_count()
        if dh > 8:
            continue
        color = sum(min(x, y) for x, y in zip(first[i]['color_hist'], second[j]['color_hist']))
        if color < .8:
            continue
        row = {'first': i, 'second': j, 'phash_distance': int(distances[ai, bi]), 'dhash_distance': dh, 'color_intersection': color}
        key = (i, j)
        if key not in pairs or (row['phash_distance'], dh) < (pairs[key]['phash_distance'], pairs[key]['dhash_distance']):
            pairs[key] = row
    return [pairs[key] for key in sorted(pairs)]


def ordered_chain(pairs: list[dict[str, Any]]) -> int:
    """Count distinct, monotonically ordered matched observations in both sources."""
    ordered = sorted({(p['first'], p['second']) for p in pairs})
    if not ordered:
        return 0
    lengths = [1] * len(ordered)
    for i, (a, b) in enumerate(ordered):
        lengths[i] = 1 + max((lengths[j] for j, (x, y) in enumerate(ordered[:i]) if x < a and y < b), default=0)
    return max(lengths)


def similarity_candidate(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Require multiple views over time or one exceptionally close frame match."""
    if not pairs:
        return None
    count = min(len({p['first'] for p in pairs}), len({p['second'] for p in pairs}))
    close = any(p['phash_distance'] <= 2 and p['dhash_distance'] <= 3 and p['color_intersection'] >= .9 for p in pairs)
    if count < 2 and not close:
        return None
    chain = ordered_chain(pairs)
    return {'relation': 'TEMPORAL_SEQUENCE_CANDIDATE' if chain >= 3 else 'VISUAL_SCENE_CANDIDATE',
            'status': 'SCENE_RELATION_UNCERTAIN', 'distinct_matched_frames': count,
            'ordered_chain_length': chain, 'best_phash_distance': min(p['phash_distance'] for p in pairs),
            'human_confirmed': False}


def scene_components(ids: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Keep every suspected or proven relation in one deterministic component."""
    parent = {identity: identity for identity in sorted(set(ids))}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for first, second in sorted(set(tuple(sorted(edge)) for edge in edges)):
        if first not in parent or second not in parent:
            raise ValueError('Scene edge refers to a missing source identity')
        a, b = find(first), find(second)
        parent[max(a, b)] = min(a, b)
    members: dict[str, list[str]] = defaultdict(list)
    for identity in parent:
        members[find(identity)].append(identity)
    names = {key: 'SCENE_' + canonical_hash(sorted(value))[:16] for key, value in members.items()}
    return {identity: names[find(identity)] for identity in parent}


def classify_exposure(flags: set[str], *, history_available: bool) -> str:
    """Distinguish recorded exposure, missing history, and absence of found evidence."""
    for status in EXPOSURE_PRIORITY:
        if status in flags:
            return status
    return 'NO_EVIDENCE_OF_PRIOR_USE' if history_available else 'UNKNOWN_HISTORY'


def propose_split(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Preserve compatible prior assignments; quarantine conflicts and invalid groups.

    Args:
        rows: Source identity, scene group, duration, technical and history metadata.
            Detector outputs and semantic labels are forbidden inputs.
        seed: Deterministic tie-breaking seed for groups without prior assignments.

    Returns:
        Provisional assignments. No row is promoted to verified independent test.
    """
    forbidden = {'predictions', 'detections', 'confidence', 'event_recall', 'event_f1', 'model_labels', 'ground_truth_events'}
    if any(forbidden.intersection(row) for row in rows):
        raise ValueError('Inference/label-driven split inputs are forbidden')
    if len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate video identity')
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get('scene_group_id') or not row.get('source_sha256'):
            raise ValueError('Missing source or scene identity; do not silently allocate')
        groups[row['scene_group_id']].append(row)
    assignments: dict[str, tuple[str, str]] = {}
    load = {'train': 0., 'val': 0., 'test': 0.}
    free = []
    translation = {'train': 'TRAIN_CANDIDATE', 'val': 'VALIDATION_CANDIDATE', 'test': 'INDEPENDENT_TEST_CANDIDATE'}
    for group, members in sorted(groups.items()):
        old = set().union(*(set(row.get('legacy_splits', [])) for row in members))
        if any(row.get('decode_status') != 'NORMAL_EOF' for row in members):
            assignments[group] = ('BLOCKED_UNASSIGNED', 'TECHNICAL_AUDIT_NOT_ELIGIBLE_IN_GROUP')
        elif any(row.get('external_scene_overlap') for row in members):
            assignments[group] = ('BLOCKED_UNASSIGNED', 'SCENE_OVERLAP_WITH_HISTORICAL_REFERENCE_REQUIRES_REVIEW')
        elif len(old) > 1 or old - translation.keys():
            assignments[group] = ('BLOCKED_UNASSIGNED', 'LEGACY_SOURCE_SCENE_SPLIT_CONFLICT')
        elif old:
            split = next(iter(old))
            assignments[group] = (translation[split], 'PRESERVE_LEGACY_VIDEO_RESERVATION; INDEPENDENCE_NOT_PROVEN')
            load[split] += sum(float(r.get('duration_seconds') or 0) for r in members)
        else:
            free.append(group)
    targets = {'train': .6, 'val': .2, 'test': .2}
    for group in sorted(free, key=lambda g: canonical_hash([seed, g])):
        split = min(targets, key=lambda key: (load[key] / targets[key], key))
        assignments[group] = (translation[split], 'SEEDED_GROUP_DURATION_ALLOCATION_NO_MODEL_OUTPUT')
        load[split] += sum(float(r.get('duration_seconds') or 0) for r in groups[group])
    output = []
    for row in sorted(rows, key=lambda r: r['video_id']):
        split, reason = assignments[row['scene_group_id']]
        output.append({**row, 'proposed_split': split, 'reason': reason,
                       'split_status': PROVISIONAL, 'split_policy_version': POLICY, 'seed': seed,
                       'annotation_status': 'PENDING_HUMAN_REVIEW',
                       'independence_verified': False,
                       'independence_status': 'TEST_INDEPENDENCE_NOT_VERIFIED',
                       'candidate_readiness': 'BLOCKED' if split in {'INDEPENDENT_TEST_CANDIDATE', 'BLOCKED_UNASSIGNED'} else 'PENDING_HUMAN_ANNOTATION',
                       'annotation_priority': 1 if 'test' in row.get('legacy_splits', []) or split == 'INDEPENDENT_TEST_CANDIDATE' else 2 if split == 'VALIDATION_CANDIDATE' else 3})
    validate_split(output)
    return output


def validate_split(rows: list[dict[str, Any]]) -> None:
    """Reject source/scene crossing and all automatic annotation/test promotion."""
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        split = row['proposed_split']
        if split not in SPLITS or row['split_status'] != PROVISIONAL:
            raise ValueError('Invalid split status or assignment')
        if row.get('independence_verified') is not False or row.get('annotation_status') != 'PENDING_HUMAN_REVIEW':
            raise ValueError('Preannotation cannot promote human GT or independent test')
        for key in ('source_sha256', 'scene_group_id'):
            groups[(key, row[key])].add(split)
        if row.get('duration_seconds') is not None and (not math.isfinite(row['duration_seconds']) or row['duration_seconds'] <= 0):
            raise ValueError('Invalid measured source duration')
    if any(len(values) > 1 for values in groups.values()):
        raise ValueError('Source/scene group crosses split boundaries')
