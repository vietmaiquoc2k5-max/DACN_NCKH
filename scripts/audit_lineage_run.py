"""Audit existing receipts and emit a missing-only resume plan to a fresh output."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lineage_run_audit import audit_run


def main() -> None:
    """Require observed lifecycle evidence; never infer an exit from file presence."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',type=Path,required=True)
    parser.add_argument('--receipts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--process-alive',action='store_true')
    parser.add_argument('--observed-exit-code',type=int)
    parser.add_argument('--resume-plan',type=Path)
    args=parser.parse_args()
    with args.scope.open(encoding='utf-8-sig') as stream:scope=list(csv.DictReader(stream))
    result=audit_run(scope,args.receipts,ROOT,process_alive=args.process_alive,exit_code=args.observed_exit_code)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    if args.resume_plan:
        if args.process_alive:raise ValueError('CANNOT_PLAN_RESUME_WHILE_ORIGINAL_PROCESS_RUNNING')
        if result['invalid_receipts']:raise ValueError('INVALID_RECEIPTS_REQUIRE_INVESTIGATION')
        missing=set(result['missing_ids'])
        with args.resume_plan.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=scope[0].keys());writer.writeheader()
            writer.writerows(row for row in scope if row['image_sha256'] in missing)
    print(json.dumps({key:result[key] for key in ['process_status','requested','processed','verified','unresolved','not_audited','summary_valid','observed_exit_code']}))


if __name__=='__main__':main()
