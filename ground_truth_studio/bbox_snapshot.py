"""Read-only live Studio snapshots; never restore over an active database."""
from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from ground_truth_studio.integrity_service import sha256


def committed_reviews(private: Path, *, original_private: Path | None = None) -> list[dict]:
    """Read immutable versions published by their final checksum marker.

    An in-progress submission without its marker is not a committed review.
    Existing versions are validated on every read, including original label bytes.
    """
    records = []
    for marker in sorted((private / 'bbox_qa/versions').glob('*/review.sha256')):
        path = marker.with_name('review.json')
        if sha256(path) != marker.read_text(encoding='utf-8').strip():
            raise ValueError('REVIEW_METADATA_HASH_MISMATCH')
        row = json.loads(path.read_text(encoding='utf-8'))
        label = path.with_name('labels.txt')
        expected_label = ((original_private / label.relative_to(private)) if original_private else label)
        if Path(row['label_version_path']).resolve() != expected_label.resolve():
            raise ValueError('REVIEW_LABEL_PATH_MISMATCH')
        if sha256(label) != row['label_sha256']:
            raise ValueError('APPROVED_LABEL_HASH_MISMATCH')
        records.append(row)
    return sorted(records, key=lambda r: (r['reviewed_at_utc'], r['version_id']))


def snapshot_studio(private: Path, target: Path) -> dict:
    """Back up SQLite through its API and copy a fixed committed bbox version set.

    Args:
        private: Existing Studio private root; read only.
        target: New private/ignored destination. Existing destinations are refused.

    Returns:
        Checksums and counts. New concurrent submissions are outside this snapshot.
    """
    target.mkdir(parents=True, exist_ok=False)
    database = private / 'ground_truth_studio.sqlite3'
    if database.exists():
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(target / database.name)) as destination:
                source.backup(destination, pages=256)
                if destination.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLITE_BACKUP_INTEGRITY_FAILURE')
    records = committed_reviews(private)
    checksums = {}
    for row in records:
        folder = private / 'bbox_qa/versions' / row['version_id']
        for name in ('review.json', 'review.sha256', 'labels.txt'):
            src = folder / name
            relative = src.relative_to(private)
            dst = target / relative
            expected = sha256(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if sha256(src) != expected or sha256(dst) != expected:
                raise ValueError('LIVE_IMMUTABLE_REVIEW_CHANGED_DURING_BACKUP')
            checksums[relative.as_posix()] = expected
    if database.exists():
        checksums[database.name] = sha256(target / database.name)
    receipt = {'schema_version': 1, 'bbox_versions': len(records),
               'unique_reviewed_images': len({r['image_sha256'] for r in records}),
               'checksums': checksums, 'sqlite_integrity': 'ok' if database.exists() else 'absent',
               'concurrency_scope': 'SQLite consistent point-in-time; fixed committed immutable bbox version set'}
    (target / 'snapshot_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt
