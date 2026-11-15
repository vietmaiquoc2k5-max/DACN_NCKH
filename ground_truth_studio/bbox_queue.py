"""Stable, deduplicated bbox navigation independent of annotation mutations."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def describe_queue(queue: dict[str, dict], history: dict[str, dict], repo: Path) -> list[dict]:
    """Attach current actor decisions and explicit data investigation reasons."""
    rows = []
    for identity, item in queue.items():
        row = dict(item)
        row['image_id'] = row.get('image_name') or Path(row['image_path']).name
        label = repo / row['label_path'] if row.get('label_path') else None
        row['empty_label'] = bool(label and label.is_file() and not label.read_text(encoding='utf-8').strip())
        row['declared_empty_label'] = str(item.get('empty_label', '')).lower() == 'true'
        row['suspicious_negative'] = bool(row.get('priority'))
        row['review_status'] = {'APPROVE': 'APPROVED', 'REJECT': 'REJECTED'}.get(
            history.get(identity, {}).get('decision'), 'PENDING')
        reasons = []
        if not (repo / row['image_path']).is_file():
            reasons.append('IMAGE_NOT_FOUND')
        if not label or not label.is_file():
            reasons.append('LABEL_NOT_FOUND')
        if not row.get('source_path_candidate') or not (repo / row['source_path_candidate']).is_file():
            reasons.append('SOURCE_NOT_FOUND')
        if not row.get('lineage_verified'):
            reasons.append('LINEAGE_NOT_VERIFIED')
        if any(word in row.get('reserved_split', '') for word in ('QUARANTINE', 'BLOCKED', 'EXCLUDED')):
            reasons.append('SPLIT_QUARANTINED')
        row['investigation_reasons'] = reasons
        row['queue_reason'] = ('SUSPICIOUS_EMPTY_LABEL_REQUIRES_HUMAN_INSPECTION' if row['suspicious_negative']
                               else 'BBOX_APPROVAL_REQUIRED')
        rows.append(row)
    return rows


def filter_queue(rows: list[dict], *, query: str = '', priority: bool = False,
                 empty: bool = False, suspicious: bool = False, status: str = 'ALL',
                 investigation: bool = False) -> list[dict]:
    """Filter all IDs and aliases before pagination; never modify review records."""
    query = query.strip().casefold()
    return [row for row in rows
            if (not priority or row.get('priority'))
            and (not empty or row['empty_label'])
            and (not suspicious or row['suspicious_negative'])
            and (status == 'ALL' or row['review_status'] == status)
            and (not investigation or row['investigation_reasons'])
            and (not query or query in ' '.join(str(row.get(k, '')) for k in
                 ('image_id', 'image_path', 'image_sha256', 'video_id', 'aliases')).casefold())]


def paginate(rows: list[dict], page: int, page_size: int = 25) -> tuple[list[dict], int, int]:
    """Return a complete deterministic slice and clamped one-based page index."""
    if page_size < 1:
        raise ValueError('INVALID_PAGE_SIZE')
    count = max(1, math.ceil(len(rows) / page_size))
    page = min(max(1, page), count)
    return rows[(page - 1) * page_size:page * page_size], page, count
