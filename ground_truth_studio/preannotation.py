"""Read verified split metadata and audited PTS without importing predictions."""
from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import sha256
from src.preannotation import PROVISIONAL, canonical_hash, validate_split
from src.video_audit_v2 import classify


def _manifest(settings: Settings) -> list[dict[str, Any]]:
    """Authenticate manifest bytes, catalog binding and provisional group contracts."""
    path = settings.repo / 'data/vn_pos_preannotation_split_manifest.json'
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding='utf-8'))
    provenance = json.loads((settings.repo / 'data/vn_pos_split_provenance.json').read_text(encoding='utf-8'))
    if sha256(path) != provenance['manifest_json_sha256']:
        raise ValueError('Preannotation manifest checksum changed')
    digest = value.pop('manifest_sha256')
    if canonical_hash(value) != digest or value['status'] != PROVISIONAL:
        raise ValueError('Invalid provisional manifest integrity/status')
    if sha256(settings.repo / 'data/video_catalog_local_v2.json') != value['input_catalog_sha256']:
        raise ValueError('Preannotation catalog version differs')
    validate_split(value['videos'])
    return value['videos']


def split_metadata(settings: Settings) -> dict[str, dict[str, Any]]:
    """Expose only source-bound routing fields, never human or AI annotation values.

    Args:
        settings: Repository and private storage configuration.

    Returns:
        Read-only provisional split hints for the annotation UI.
    """
    fields = {'video_id', 'source_sha256', 'scene_group_id', 'proposed_split', 'split_status',
              'candidate_readiness', 'annotation_priority', 'decode_status', 'independence_status'}
    return {row['video_id']: {k: row[k] for k in fields if k in row} for row in _manifest(settings)}


def audited_source_metadata(settings: Settings, video: dict[str, Any], digest: str) -> dict[str, Any] | None:
    """Use exact v2 source PTS/durations bound to freshly verified source bytes.

    Args:
        settings: Local repository configuration.
        video: Catalog source identity, without predictions or annotations.
        digest: SHA-256 freshly measured by VideoService on the current source.

    Returns:
        Explicit-duration source PTS metadata for an eligible VN_POS, or None
        when no completed eligible audit exists. Never promotes GT status.
    """
    if not video.get('video_id', '').startswith('VN_POS_'):
        return None
    row = next((r for r in _manifest(settings) if r['video_id'] == video['video_id']), None)
    if row is None:
        return None
    if row['source_sha256'] != digest:
        raise ValueError('BLOCKED_INPUT: preannotation source SHA-256 mismatch')
    if row['decode_status'] != 'NORMAL_EOF':
        return None
    record_path = confined(settings.repo, row['audit_record_path'])
    if sha256(record_path) != row['audit_record_sha256']:
        raise ValueError('BLOCKED_INPUT: technical audit record changed')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record['source_sha256'] != digest or classify(record)['technical_eligible'] is not True:
        raise ValueError('BLOCKED_INPUT: technical audit/source eligibility differs')
    timeline = record_path.parent / 'timeline.stdout'
    if sha256(timeline) != record['artifact_sha256']['timeline.stdout']:
        raise ValueError('BLOCKED_INPUT: audited source timeline changed')
    pts, durations = [], []
    with timeline.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.startswith('frame|'):
                continue
            fields = dict(part.split('=', 1) for part in line.strip().split('|')[1:] if '=' in part)
            pts.append(int(fields['pts'])); durations.append(int(fields['duration']))
    if not pts or len(pts) != record['opencv_actual_decoded_frames']:
        raise ValueError('BLOCKED_INPUT: audited frame count differs')
    if any(b != a + duration for a, b, duration in zip(pts, pts[1:], durations)):
        raise ValueError('BLOCKED_INPUT: audited source timeline discontinuity')
    base = Fraction(record['time_base'])
    return {
        'source_sha256': digest, 'frame_count': len(pts),
        'fps': float(Fraction(record['r_frame_rate'])), 'resolution': row['resolution'],
        'duration': float((pts[-1] + durations[-1] - pts[0]) * base),
        'source_pts': [float(value * base) for value in pts],
        'playback_time_offset_sec': float(pts[0] * base),
        'timestamp_provenance': 'SOURCE_PTS',
        'duration_provenance': 'LAST_SOURCE_PTS_PLUS_EXPLICIT_FRAME_DURATION_MINUS_FIRST_PTS',
        'technical_audit_policy': record['audit_policy_version'],
        'codec': row['codec'], 'pix_fmt': row.get('pix_fmt'),
    }
