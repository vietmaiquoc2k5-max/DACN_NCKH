"""Fail-closed completeness checks for resumable lineage runs."""
from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ground_truth_studio.integrity_service import sha256


def audit_run(scope: list[dict], folder: Path, root: Path, *, process_alive: bool,
              exit_code: int | None = None, verify_sources: bool = True) -> dict:
    """Validate a fixed manifest against committed receipts and completion evidence.

    Args:
        scope: Original image rows with declared frame and source identities.
        folder: Decoder output directory. Files still being written remain pending.
        root: Owning repository root.
        process_alive: Independently observed process state, never inferred from files.
        exit_code: Observed original process exit code, if available.
        verify_sources: Authenticate each distinct source once for this audit.

    Returns:
        Exact counts, issues and finality. No inferred successful exit code.
    """
    expected={r['image_sha256']:r for r in scope}
    if len(expected)!=len(scope):raise ValueError('DUPLICATE_SCOPE_IDS')
    statuses=Counter();issues=[];missing=[];sources={};receipts={}
    for key,row in expected.items():
        path=folder/(key+'.json')
        if not path.is_file():missing.append(key);continue
        try:
            proof=json.loads(path.read_text(encoding='utf-8'))
            if proof['image_sha256']!=key:raise ValueError('IMAGE_ID_MISMATCH')
            if proof['source_frame_index']!=int(row['source_frame_index_declared']):raise ValueError('FRAME_INDEX_MISMATCH')
            if proof['source_video_path']!=row['source_path_candidate']:raise ValueError('SOURCE_PATH_MISMATCH')
            if sha256(root/row['image_path'])!=key:raise ValueError('IMAGE_HASH_MISMATCH')
            if proof.get('source_video_sha256')!=row['current_candidate_source_sha256']:raise ValueError('SOURCE_HASH_BINDING_MISMATCH')
            if verify_sources:
                source=root/proof['source_video_path']
                if str(source) not in sources:sources[str(source)]=sha256(source)
                if sources[str(source)]!=proof['source_video_sha256']:raise ValueError('SOURCE_HASH_MISMATCH')
            if proof['status'] in {'VERIFIED_NATIVE_DECODE','VERIFIED_DERIVED_IMAGE'}:
                if type(proof.get('source_pts_ticks')) is not int:raise ValueError('INTEGER_PTS_REQUIRED')
                if Fraction(proof['time_base'])<=0:raise ValueError('INVALID_TIME_BASE')
                if proof['status']=='VERIFIED_DERIVED_IMAGE' and (
                    proof.get('match_evidence')!='EXACT_ENCODED_IMAGE_SHA256' or not proof.get('transformation')):
                    raise ValueError('TRANSFORMATION_EVIDENCE_REQUIRED')
            statuses[proof['status']]+=1;receipts[path.name]=sha256(path)
        except (ValueError,KeyError,OSError,TypeError,ZeroDivisionError) as error:
            issues.append({'image_sha256':key,'reason':str(error)})
    extra=[p.name for p in folder.glob('*.json') if p.name!='summary.json' and p.stem not in expected]
    summary_path=folder/'summary.json';summary_valid=False
    if summary_path.is_file():
        try:
            summary=json.loads(summary_path.read_text())
            summary_valid=summary['scope_images']==len(scope) and summary['receipts']==receipts and not missing and not issues and not extra
        except (ValueError,KeyError):pass
    if process_alive:state='RUNNING'
    elif exit_code is not None and exit_code!=0:state='FAILED'
    elif summary_valid and exit_code==0:state='COMPLETED'
    elif missing or issues:state='INTERRUPTED'
    else:state='NOT_VERIFIED'
    return {'process_status':state,'requested':len(scope),'processed':sum(statuses.values()),
        'verified':sum(statuses[s] for s in ('VERIFIED_NATIVE_DECODE','VERIFIED_DERIVED_IMAGE')),
        'unresolved':sum(v for s,v in statuses.items() if s not in ('VERIFIED_NATIVE_DECODE','VERIFIED_DERIVED_IMAGE')),
        'not_audited':len(missing),'invalid_receipts':issues,'unexpected_receipts':extra,
        'summary_valid':summary_valid,'observed_exit_code':exit_code,'statuses':dict(statuses),
        'missing_ids':missing,'receipt_sha256':receipts,'distinct_sources_hashed':len(sources)}
