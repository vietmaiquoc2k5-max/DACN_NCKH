"""Run the existing strict v2 decode policy on the catalog's 80 VN_POS sources."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import sha256
from src.video_audit_v2 import POLICY, audit_video, resume_record, write_json

LOGGER = logging.getLogger(__name__)


def run_audit(out: Path, workers: int, resume: bool, timeout: int) -> dict[str, Any]:
    """Collect actual full-stream evidence without repairing or moving sources.

    Args:
        out: Fresh benchmark directory, or this run's directory with resume enabled.
        workers: Number of concurrent videos; each external decoder uses two threads.
        resume: Reuse completed records only after strict checksum validation.
        timeout: Timeout per external decoder/probe invocation, in seconds.

    Returns:
        Technical audit results; these never constitute human semantic approval.
    """
    if not out.is_relative_to(ROOT / 'runs/benchmarks/vn_pos_preannotation'):
        raise ValueError('Run directory must be beneath runs/benchmarks/vn_pos_preannotation')
    settings = Settings.load()
    if not settings.ffmpeg or not settings.ffprobe:
        raise ValueError('FFmpeg and FFprobe are required; no clean-EOF fallback')
    catalog_path = ROOT / 'data/video_catalog_local_v2.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    sources = [r for r in catalog['videos'] if r['video_id'].startswith('VN_POS_')]
    expected = {f'VN_POS_{i:02d}' for i in range(1, 81)}
    if len(sources) != 80 or {r['video_id'] for r in sources} != expected:
        raise ValueError('Expected exactly VN_POS_01..80; investigate the catalog mismatch')
    entries = [{'video_id': r['video_id'], 'relative_path': r['relative_path'], 'sha256': r['sha256']} for r in sources]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    identity = {
        'catalog_sha256': sha256(catalog_path), 'sources': entries, 'head': head,
        'audit_policy': POLICY, 'audit_implementation_sha256': sha256(ROOT / 'src/video_audit_v2.py'),
        'runner_sha256': sha256(Path(__file__)),
    }
    if out.exists():
        if not resume:
            raise FileExistsError('Use a fresh directory or explicit --resume; no artifacts overwritten')
        previous = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
        if any(previous[k] != v for k, v in identity.items()):
            raise ValueError('Resume input/code/catalog binding changed; choose a fresh run')
    else:
        out.mkdir(parents=True)
        (out / 'records').mkdir()
        write_json(out / 'run_manifest.json', {
            **identity, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'VN_POS_TECHNICAL_PREANNOTATION_ONLY', 'workers': workers,
            'timeout_per_external_pass_seconds': timeout, 'detector_inference_enabled': False,
            'frozen_inputs': {name: sha256(ROOT / name) for name in [
                'data/ground_truth/events.jsonl', 'data/ground_truth/events_v1.1_snapshot.jsonl',
                'datasets/vn_dataset/split_manifest.csv', 'datasets/vn_dataset/review_status.csv',
            ]},
        })

    def one(entry: dict[str, Any]) -> dict[str, Any]:
        folder = out / 'records' / entry['video_id']
        if (folder / 'record.sha256').exists():
            return resume_record(folder, entry, ROOT)
        if folder.exists():
            raise ValueError('Partial evidence retained; use a fresh run, never overwrite a partial record')
        if not confined(ROOT, entry['relative_path']).is_file():
            raise FileNotFoundError(entry['video_id'])
        return audit_video(entry, ROOT, folder, settings.ffmpeg, settings.ffprobe, timeout)

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, entry): entry['video_id'] for entry in entries}
        for future in as_completed(pending):
            vid = pending[future]
            try:
                row = future.result()
                rows.append(row)
                LOGGER.info('COMPLETE %s/80 %s %s', len(rows) + len(failures), vid, row['decode_status'])
            except Exception as error:
                failures.append({'video_id': vid, 'error': f'{type(error).__name__}: {error}'})
                LOGGER.exception('AUDIT_FAILED %s', vid)
            write_json(out / 'technical_audit_80.json', {
                'videos': sorted(rows, key=lambda r: r['video_id']), 'failures': failures,
                'complete': len(rows) + len(failures) == 80,
                'counts': dict(Counter(r['decode_status'] for r in rows)),
                'policy': POLICY, 'semantic_approval': False,
            })
    result = json.loads((out / 'technical_audit_80.json').read_text(encoding='utf-8'))
    write_json(out / 'audit_run_status.json', {
        'complete': True, 'audited_records': len(rows), 'worker_failures': failures,
        'sources_unchanged': all(sha256(confined(ROOT, e['relative_path'])) == e['sha256'] for e in entries),
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'detector_inference_executed': False,
    })
    return result


def main() -> None:
    """Run the source audit with bounded concurrency and per-pass deadlines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_audit(args.run_dir.resolve(), args.workers, args.resume, args.timeout)
    print('AUDIT_FINISHED', json.dumps(result['counts']), 'WORKER_FAILURES', len(result['failures']), flush=True)
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
