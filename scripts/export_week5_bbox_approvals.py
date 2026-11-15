"""Export a checksum-verified private snapshot; never manufacture human decisions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.bbox_export_adapter import export_approvals
from ground_truth_studio.bbox_service import load_queue
from ground_truth_studio.bbox_snapshot import committed_reviews
from ground_truth_studio.config import Settings


def main() -> None:
    """Export to ignored local storage; optional verified splits enable eligible rows."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lineage-dir',type=Path)
    parser.add_argument('--split-manifest',type=Path)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'.tools'):
        raise ValueError('PRIVATE_APPROVAL_EXPORT_MUST_STAY_IN_IGNORED_TOOLS')
    rows=committed_reviews(args.snapshot,original_private=Settings.load().private)
    result=export_approvals(ROOT,rows,{r['image_sha256']:r for r in load_queue(ROOT)},args.output,
        lineage_dir=args.lineage_dir.resolve() if args.lineage_dir else None,
        split_manifest=args.split_manifest,version_root=args.snapshot/'bbox_qa/versions')
    print(json.dumps({'human_approved':result['human_approved_count'],'eligible':result['approved_for_training'],
                      'excluded_statuses':dict(Counter(r['status'] for r in result['excluded_rows']))}))


if __name__=='__main__':main()
