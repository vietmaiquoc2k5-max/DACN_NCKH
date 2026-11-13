"""Collect pixel fingerprints for overlap screening; never load a detector."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ground_truth_studio.config import Settings, confined
from ground_truth_studio.integrity_service import sha256
from src.preannotation import fingerprint
from src.video_audit_v2 import write_json

LOGGER = logging.getLogger(__name__)


def video_fingerprint(entry: dict[str, Any], run: Path, out: Path, legacy: dict[str, list[int]]) -> dict[str, Any]:
    """Sample actual source pixels for content screening, not annotation timestamps.

    Args:
        entry: An existing catalog record with its source identity pin.
        run: Technical audit directory; VN_POS count comes from its completed audit.
        out: Derived pixel-evidence directory.
        legacy: Previously declared image ordinals, used without reading labels/onsets.

    Returns:
        Actual pixel hashes and requested/backend-reported positions. Random-seek
        positions are explicitly non-authoritative for GT or event-time evaluation.
    """
    vid = entry['video_id']
    path = confined(ROOT, entry['relative_path'])
    digest = sha256(path)
    if digest != entry['sha256']:
        raise ValueError(f'Source bytes differ from catalog: {vid}')
    settings = Settings.load()
    probe = subprocess.run([settings.ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], capture_output=True, timeout=60)
    metadata = json.loads(probe.stdout) if probe.returncode == 0 else {}
    stream = next((s for s in metadata.get('streams', []) if s.get('codec_type') == 'video'), {})
    count = None
    basis = None
    if vid.startswith('VN_POS_'):
        record = run / 'records' / vid / 'record.json'
        deadline = time.monotonic() + 7200
        while not record.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f'Technical audit not completed: {vid}')
            time.sleep(2)
        measured = json.loads(record.read_text(encoding='utf-8'))
        if measured.get('source_sha256') != digest:
            raise ValueError('Technical audit/source binding differs')
        count = measured.get('opencv_actual_decoded_frames')
        basis = 'CURRENT_V2_ACTUAL_DECODE_COUNT'
    else:
        pts = ROOT / 'runs/benchmarks/week4_final_scientific/run_20261113_consensus_v1/pts' / (vid + '.json')
        previous = ROOT / 'runs/benchmarks/week4_decode_audit_v2/run_20261113/records' / vid / 'record.json'
        if previous.exists():
            measured = json.loads(previous.read_text(encoding='utf-8'))
            if measured.get('source_sha256') == digest:
                count = measured.get('opencv_actual_decoded_frames')
                basis = 'PREVIOUS_V2_DECODE_COUNT_SAME_SOURCE_SHA'
        elif pts.exists():
            count = len(json.loads(pts.read_text())['pts'])
            basis = 'PREVIOUS_SOURCE_PTS_FRAME_COUNT_SAME_CATALOG_PIN'
    capture = cv2.VideoCapture(str(path))
    frames = []
    requested = []
    try:
        if not capture.isOpened():
            raise ValueError('Content sampler cannot open source')
        if not isinstance(count, int) or count <= 0:
            raw = stream.get('nb_frames')
            count = int(raw) if raw and raw != 'N/A' else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            basis = 'METADATA_ONLY_CONTENT_SAMPLING_NOT_EOF_PROOF'
        requested = sorted(set(np.linspace(0, max(0, count - 1), min(48, count), dtype=int).tolist() + legacy.get(vid, [])))
        for index in requested:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if ok and image is not None:
                frames.append({**fingerprint(image), 'requested_frame_index': index,
                               'backend_reported_next_index': capture.get(cv2.CAP_PROP_POS_FRAMES)})
    finally:
        capture.release()
    if sha256(path) != digest:
        raise ValueError('Source changed during pixel fingerprinting')
    result = {
        'video_id': vid, 'source_path': entry['relative_path'], 'source_sha256': digest,
        'frames': frames, 'requested_samples': len(requested), 'decoded_samples': len(frames),
        'position_provenance': 'CONTENT_ONLY_RANDOM_SEEK_REQUEST_AND_BACKEND_REPORT; NOT_AUTHORITATIVE_GT_ORDINAL_OR_PTS',
        'frame_count_basis': basis, 'sampling_frame_count': count,
        'metadata': metadata, 'probe_exit_code': probe.returncode,
        'probe_stderr': probe.stderr.decode('utf-8', errors='replace'),
        'has_audio': any(s.get('codec_type') == 'audio' for s in metadata.get('streams', [])) if metadata else None,
        'decoder_inference_executed': False,
    }
    write_json(out / 'videos' / (vid + '.json'), result)
    return result


def collect(run: Path, workers: int) -> None:
    """Collect source/reference pixels and existing image identities in a fresh run."""
    if not run.is_relative_to(ROOT / 'runs/benchmarks/vn_pos_preannotation'):
        raise ValueError('Unsafe run directory')
    out = run / 'fingerprints'
    out.mkdir(exist_ok=False)
    (out / 'videos').mkdir()
    rows = json.loads((ROOT / 'data/video_catalog_local_v2.json').read_text(encoding='utf-8'))['videos']
    sources = [r for r in rows if confined(ROOT, r['relative_path']).is_file()]
    legacy: dict[str, list[int]] = {}
    with (ROOT / 'datasets/vn_dataset/split_manifest.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            legacy.setdefault(row['video_id'], []).append(int(row['frame_idx']))
    cv2.setNumThreads(1)
    write_json(out / 'protocol.json', {
        'method': '48 uniform source-ordinal requests plus legacy frame requests; three-crop DCT pHash/dHash + HSV histogram',
        'purpose': 'Content overlap hypotheses only; no FIRE/SMOKE labels or detector inference',
        'model_outputs_or_ai_onsets_used_for_sampling': False,
        'source_order': 'references first, then VN_POS; only metadata/identity',
        'module_sha256': sha256(ROOT / 'src/preannotation.py'),
        'runner_sha256': sha256(Path(__file__)),
        'limitations': 'Finite sampled pixels cannot rule out short clips, arbitrary crops, scene changes or unseen training footage; no candidate is automatically independent',
    })
    ordered = sorted(sources, key=lambda r: (r['video_id'].startswith('VN_POS_'), r['video_id']))
    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(video_fingerprint, e, run, out, legacy): e['video_id'] for e in ordered}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                row = future.result()
                completed.append({'video_id': vid, 'samples': row['decoded_samples'], 'source_sha256': row['source_sha256']})
                LOGGER.info('PIXELS %s/%s %s samples=%s', len(completed), len(sources), vid, row['decoded_samples'])
            except Exception as error:
                errors.append({'video_id': vid, 'error': f'{type(error).__name__}: {error}'})
                LOGGER.exception('PIXEL_FAILED %s', vid)
    images = []
    source_stems = {Path(r['relative_path']).stem: r['video_id'] for r in sources}
    directories = ['datasets/vn_dataset/images', 'FireSmokeLabeler/dataset/images', 'dataset']
    for directory in directories:
        for path in sorted((ROOT / directory).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                errors.append({'image': path.relative_to(ROOT).as_posix(), 'error': 'UNREADABLE_REFERENCE_IMAGE'})
                continue
            identity = next((vid for stem, vid in source_stems.items() if path.stem.startswith(stem + '_f') or path.stem.startswith(stem + '_frame')), None)
            images.append({**fingerprint(image), 'relative_path': path.relative_to(ROOT).as_posix(),
                           'image_sha256': sha256(path), 'source_video_id_hint': identity,
                           'corpus': directory, 'declared_split': next((x for x in path.parts if x in {'train', 'val', 'test'}), 'UNKNOWN')})
        LOGGER.info('IMAGE_CORPUS %s cumulative=%s', directory, len(images))
    write_json(out / 'image_fingerprints.json', images)
    write_json(out / 'completion.json', {'complete': True, 'source_videos': len(sources), 'completed': sorted(completed, key=lambda r:r['video_id']), 'images': len(images), 'errors': errors})
    print('PIXEL_SCREENING_COLLECTED', len(completed), 'VIDEOS', len(images), 'IMAGES', 'ERRORS', len(errors), flush=True)
    if errors:
        raise SystemExit(1)


def main() -> None:
    """Collect only non-semantic pixel fingerprints and source metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    collect(args.run_dir.resolve(), args.workers)


if __name__ == '__main__':
    main()
