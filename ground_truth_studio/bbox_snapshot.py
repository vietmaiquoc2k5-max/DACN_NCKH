"""Read-only live Studio snapshots; never restore over an active database."""
from __future__ import annotations

import json
import shutil
import sqlite3
import logging
import re
from contextlib import closing
from pathlib import Path

from ground_truth_studio.integrity_service import sha256
from ground_truth_studio.bbox_resolution import apply_resolutions


def scan_reviews(private: Path, *, original_private: Path | None = None) -> tuple[list[dict],list[dict]]:
    """Inspect every version independently; corrupt bytes remain untouched and quarantined."""
    records=[];issues=[]
    for folder in sorted((private/'bbox_qa/versions').glob('*')):
        if not folder.is_dir():continue
        path=folder/'review.json';marker=folder/'review.sha256';label=folder/'labels.txt'
        data=None;error=None;expected=None;actual=None
        try:
            if not path.is_file():raise ValueError('MISSING_REVIEW_METADATA')
            actual=sha256(path)
            if not marker.is_file():raise ValueError('MISSING_CHECKSUM_OR_INFLIGHT_SUBMISSION')
            expected=marker.read_text(encoding='utf-8').strip()
            if not re.fullmatch('[0-9a-f]{64}',expected) or actual!=expected:
                raise ValueError('REVIEW_METADATA_HASH_MISMATCH')
            data=json.loads(path.read_text(encoding='utf-8'))
            if data['version_id']!=folder.name:raise ValueError('REVIEW_VERSION_ID_MISMATCH')
            expected_label=(original_private/label.relative_to(private)) if original_private else label
            if Path(data['label_version_path']).resolve()!=expected_label.resolve():raise ValueError('REVIEW_LABEL_PATH_MISMATCH')
            if not label.is_file() or sha256(label)!=data['label_sha256']:raise ValueError('APPROVED_LABEL_HASH_MISMATCH')
            for key in ('image_sha256','reviewer_id','reviewed_at_utc','decision'):data[key]
            if any(not isinstance(data[key],str) or not data[key] for key in ('image_sha256','reviewer_id','reviewed_at_utc','decision')):
                raise ValueError('INVALID_REVIEW_SCHEMA')
            records.append(data)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            error=str(exc)
            try:claimed=json.loads(path.read_text(encoding='utf-8')).get('image_sha256')
            except (OSError,ValueError,AttributeError):claimed=None
            if not isinstance(claimed,str):claimed=None
            issue={'version_id':folder.name,'path':str(path),'status':'INTEGRITY_QUARANTINED',
                'error':error,'expected_sha256':expected,'actual_sha256':actual,
                'file_size':path.stat().st_size if path.exists() else None,
                'mtime_ns':path.stat().st_mtime_ns if path.exists() else None,
                'untrusted_claimed_image_sha256':claimed}
            issues.append(issue)
            logging.getLogger(__name__).error('BBox integrity quarantine version=%s path=%s error=%s',folder.name,path,error)
    # A corrupt newer decision must not silently resurrect an older approval.
    issues=apply_resolutions(private,records,issues)
    tainted={i['untrusted_claimed_image_sha256'] for i in issues if i['untrusted_claimed_image_sha256'] and i['status']=='INTEGRITY_QUARANTINED'}
    records=[r for r in records if r['image_sha256'] not in tainted]
    return sorted(records,key=lambda r:(r['reviewed_at_utc'],r['version_id'])),issues


def committed_reviews(private: Path, *, original_private: Path | None = None) -> list[dict]:
    """Read immutable versions published by their final checksum marker.

    An in-progress submission without its marker is not a committed review.
    Existing versions are validated on every read, including original label bytes.
    """
    checked,issues=scan_reviews(private,original_private=original_private)
    blocking=[i for i in issues if i['status']=='INTEGRITY_QUARANTINED' and i['error']!='MISSING_CHECKSUM_OR_INFLIGHT_SUBMISSION']
    if blocking:
        raise ValueError(blocking[0]['error']+'; version_id='+blocking[0]['version_id'])
    return checked


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
    paths=list((private/'bbox_qa/versions').glob('*/*'))+list((private/'bbox_qa/resolutions').glob('*/*'))
    for src in paths:
        if src.is_file():
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
