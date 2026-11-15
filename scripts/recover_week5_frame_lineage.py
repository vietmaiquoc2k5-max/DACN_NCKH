"""Recover selected VN frame receipts without editing pixels, labels or human reviews."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import sha256
from scripts.audit_week5_readiness import read_csv, write_json, QUEUE
from src.frame_lineage import verify_video_frames


def main() -> None:
    """Use the declared priority scope; accept an explicit alternative input CSV."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/QUEUE/'empty_label_review_129.csv')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers', type=int, choices=[1,2], default=1,
                        help='Use one decoder by default while Studio is active.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    settings=Settings.load()
    if not settings.ffprobe:raise ValueError('FFPROBE_REQUIRED')
    rows=read_csv(args.input);groups=defaultdict(list)
    for row in rows:groups[row['source_path_candidate']].append(row)
    cv2.setNumThreads(1);results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(verify_video_frames,ROOT,ROOT/path,group,settings.ffprobe):path for path,group in groups.items()}
        for future in as_completed(futures):
            batch=future.result();results.extend(batch)
            for row in batch:write_json(args.output/(row['image_sha256']+'.json'),row)
            print(json.dumps({'source':futures[future],'statuses':dict(Counter(r['status'] for r in batch))}),flush=True)
    write_json(args.output/'summary.json',{'scope_images':len(rows),'scope_videos':len(groups),
        'statuses':dict(Counter(r['status'] for r in results)),'input_manifest_sha256':sha256(args.input),
        'receipts':{p.name:sha256(p) for p in args.output.glob('*.json')},
        'labels_transformed':False,'human_approvals_changed':False,'scene_split_independence':False})


if __name__=='__main__':main()
