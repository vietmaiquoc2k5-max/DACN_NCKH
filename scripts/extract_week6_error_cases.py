#!/usr/bin/env python3
"""Error Case Extraction Tool for Week 6.
Extracts real false positive (FP) and false negative (FN) error incidents from measured evaluation logs.
Merges consecutive detections to avoid duplicate counts; enforces strict PENDING_REVIEW status.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
LOG = logging.getLogger('error_extractor')


def extract_and_catalog_errors(
    input_events_json: Path,
    negative_manifest_json: Path,
    output_csv: Path,
    model_sha256: str,
    merge_gap_sec: float = 5.0,
    max_cases: int = 60
) -> list[dict[str, Any]]:
    """Extract real error cases from pipeline B2 evaluation logs on negative videos."""
    if not input_events_json.exists():
        raise FileNotFoundError(f'Input events file not found: {input_events_json}')

    data = json.loads(input_events_json.read_text(encoding='utf-8'))
    raw_events = data.get('pipeline_b2_events', [])
    if not raw_events:
        raise ValueError('No pipeline_b2_events found in evaluation log')

    # Load video fps from manifest if available
    video_fps_map = {}
    if negative_manifest_json.exists():
        manifest = json.loads(negative_manifest_json.read_text(encoding='utf-8'))
        for v in manifest.get('videos', []):
            video_fps_map[v.get('video_id')] = v.get('fps', 25.0)

    # Group events by video_id
    by_video: dict[str, list[dict[str, Any]]] = {}
    for ev in raw_events:
        by_video.setdefault(ev['video_id'], []).append(ev)

    # Merge consecutive detections within merge_gap_sec
    merged_episodes: list[dict[str, Any]] = []
    for vid, ev_list in sorted(by_video.items()):
        ev_list.sort(key=lambda x: x['start_sec'])
        current: dict[str, Any] | None = None
        for ev in ev_list:
            if current is None:
                current = dict(ev)
                current['peak_conf'] = ev['confidence']
                current['detection_count'] = 1
            else:
                if (ev['start_sec'] - current['end_sec'] <= merge_gap_sec) and (ev['label'] == current['label']):
                    current['end_sec'] = max(current['end_sec'], ev['end_sec'])
                    current['peak_conf'] = max(current['peak_conf'], ev['confidence'])
                    current['detection_count'] += 1
                else:
                    merged_episodes.append(current)
                    current = dict(ev)
                    current['peak_conf'] = ev['confidence']
                    current['detection_count'] = 1
        if current is not None:
            merged_episodes.append(current)

    LOG.info(
        'Processed %d raw FP events into %d distinct merged error episodes (gap <= %.1fs).',
        len(raw_events), len(merged_episodes), merge_gap_sec
    )

    # Build error catalog
    catalog: list[dict[str, Any]] = []
    # Sort to sample across diverse videos, both Fire and Smoke
    fire_episodes = [e for e in merged_episodes if e['label'] == 'FIRE']
    smoke_episodes = [e for e in merged_episodes if e['label'] == 'SMOKE']

    # Interleave to ensure balanced representation of both Fire and Smoke false alarms
    interleaved = []
    max_len = max(len(fire_episodes), len(smoke_episodes))
    for i in range(max_len):
        if i < len(fire_episodes):
            interleaved.append(fire_episodes[i])
        if i < len(smoke_episodes):
            interleaved.append(smoke_episodes[i])

    selected = interleaved[:max_cases] if max_cases > 0 else interleaved

    for idx, ep in enumerate(selected, start=1):
        vid = ep['video_id']
        fps = video_fps_map.get(vid, 25.0)
        start_sec = ep['start_sec']
        end_sec = ep['end_sec']
        duration = end_sec - start_sec
        peak_conf = ep['peak_conf']
        label = ep['label']
        frame_idx = int(round(start_sec * fps))

        # Severity rule based strictly on objective duration and peak confidence
        if duration >= 3.0 or peak_conf >= 0.50:
            severity = 'HIGH'
        elif duration >= 1.0 or peak_conf >= 0.35:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        catalog.append({
            'error_id': f'ERR_{idx:03d}',
            'video_id': vid,
            'timestamp': f'{start_sec:.2f}s - {end_sec:.2f}s' if duration > 0 else f'{start_sec:.2f}s',
            'frame_index': frame_idx,
            'model_sha256': model_sha256,
            'prediction': f'{label} (peak_conf: {peak_conf:.4f}, count: {ep["detection_count"]})',
            'ground_truth': 'BACKGROUND (NO_FIRE_NO_SMOKE)',
            'error_type': 'FP',
            'severity': severity,
            'suspected_cause': 'UNASSIGNED_REQUIRES_HUMAN_REVIEW',
            'reviewer_status': 'PENDING_REVIEW',
            'evidence_path': f'datasets/negative_videos/{vid}.mp4'
        })

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'error_id', 'video_id', 'timestamp', 'frame_index', 'model_sha256',
        'prediction', 'ground_truth', 'error_type', 'severity',
        'suspected_cause', 'reviewer_status', 'evidence_path'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)

    LOG.info('Successfully wrote %d error cases to %s', len(catalog), output_csv)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-events',
        type=Path,
        default=ROOT / 'runs/benchmarks/week4_full_inference/run_20261111_233035_neg_full/temporal_confirmed_events.json'
    )
    parser.add_argument(
        '--negative-manifest',
        type=Path,
        default=ROOT / 'data/negative_videos_manifest.json'
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=ROOT / 'data/error_cases_catalog.csv'
    )
    parser.add_argument(
        '--model-sha256',
        type=str,
        default='356a288f045dcc3dfe4f9984e86dffc8b854627327370198edc015672265e5df'
    )
    parser.add_argument('--gap', type=float, default=5.0)
    parser.add_argument('--max-cases', type=int, default=60, help='Number of error cases to catalog (>= 50 required)')
    args = parser.parse_args()

    try:
        extract_and_catalog_errors(
            input_events_json=args.input_events,
            negative_manifest_json=args.negative_manifest,
            output_csv=args.output_csv,
            model_sha256=args.model_sha256,
            merge_gap_sec=args.gap,
            max_cases=args.max_cases
        )
        return 0
    except Exception as err:
        LOG.error('Error extraction failed: %s', err)
        return 1


if __name__ == '__main__':
    sys.exit(main())
