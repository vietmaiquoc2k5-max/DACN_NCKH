"""Read-only human bundle audit and a separate versioned research event scope."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ground_truth_studio.config import Settings,confined
from ground_truth_studio.integrity_service import sha256
from src.preannotation import canonical_hash,validate_split
from src.video_audit_v2 import resume_record,write_json
from src.week45_ground_truth import duplicate_policy,validate_submission


def audit(bundle: Path,out: Path,protocol: str) -> dict[str,Any]:
    """Authenticate human records and create a sanitized derived bundle.

    Args:
        bundle: Existing private export; only read access is used.
        out: A fresh versioned benchmark directory inside this repository.
        protocol: Operator-declared human review protocol, never inferred from accounts.

    Returns:
        Actual audited counts, exclusions and unresolved issues.
    """
    if not out.is_relative_to(ROOT/'runs/benchmarks/week4_acceptance_closure'):
        raise ValueError('Unsafe output directory')
    if out.exists():raise FileExistsError('Choose a fresh closure run; originals are never overwritten')
    hashes=json.loads((bundle/'file_sha256.json').read_text(encoding='utf-8'))
    if not hashes or any(sha256(confined(bundle,name))!=digest for name,digest in hashes.items()):
        raise ValueError('Private export checksum failure')
    records=json.loads((bundle/'single_review_records.json').read_text(encoding='utf-8'))
    primary=[r for r in records if r['video_id'].startswith('VN_POS_') and r.get('is_primary_round_for_video') is True]
    if len({r['video_id'] for r in primary})!=len(primary):raise ValueError('Ambiguous primary review identity')
    if any(r['video_id'] not in {f'VN_POS_{i:02d}' for i in range(1,81)} for r in primary):raise ValueError('Unexpected VN_POS identity')
    manifest_path=ROOT/'data/vn_pos_preannotation_split_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));declared=manifest.pop('manifest_sha256')
    if canonical_hash(manifest)!=declared:raise ValueError('Preannotation canonical integrity failure')
    split_provenance=json.loads((ROOT/'data/vn_pos_split_provenance.json').read_text(encoding='utf-8'))
    if sha256(manifest_path)!=split_provenance['manifest_json_sha256']:raise ValueError('Preannotation bytes changed')
    validate_split(manifest['videos'])
    by_id={r['video_id']:r for r in manifest['videos']}
    # Snapshot reads never instantiate ReviewRepository, which has write initialization.
    db=sqlite3.connect((Settings.load().private/'ground_truth_studio.sqlite3').as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
    for record in records:
        actual=db.execute('SELECT r.*,u.person_id FROM review_sessions r JOIN users u ON r.user_id=u.id WHERE r.id=?',(record['review_id'],)).fetchone()
        if not actual or actual['status']!='SUBMITTED' or actual['video_id']!=record['video_id'] or actual['submitted_at']!=record['submitted_at_utc'] or actual['person_id']!=record['reviewer_person_id']:
            raise ValueError('Submission DB binding mismatch')
        source=json.loads(actual['source']);payload=json.loads(actual['payload'])
        if source['source_sha256']!=record['source_sha256'] or payload['events']!=record['events'] or payload['uncertain_intervals']!=record['uncertain_intervals'] or payload.get('decision')!=record['decision'] or payload.get('attest_full_view') is not True or payload.get('evidence_ref')!=record['evidence_ref']:
            raise ValueError('Submitted payload/source differs from export')
        path=confined(ROOT,record['relative_path'])
        if sha256(path)!=record['source_sha256']:raise ValueError('Export/source bytes mismatch')
    db_counts=dict(db.execute('SELECT status,COUNT(*) FROM review_sessions GROUP BY status'))
    vn_counts=dict(db.execute("SELECT status,COUNT(*) FROM review_sessions WHERE video_id LIKE 'VN_POS_%' GROUP BY status"))
    db.close()
    out.mkdir(parents=True)
    duplicate_exclusions,duplicates=duplicate_policy(primary)
    scope=[];accepted_events=[];exclusions=[];issues=[]
    for record in sorted(primary,key=lambda r:r['video_id']):
        vid=record['video_id'];pin=by_id[vid];reasons=[]
        if record['source_sha256']!=pin['source_sha256'] or record['relative_path']!=pin['source_path']:
            reasons.append('PREANNOTATION_SOURCE_BINDING_MISMATCH')
        folder=confined(ROOT,pin['audit_record_path']).parent
        technical=resume_record(folder,{'video_id':vid,'relative_path':record['relative_path'],'sha256':record['source_sha256']},ROOT)
        if technical['decode_status']!='NORMAL_EOF':reasons.append('TECHNICAL_DECODE_NOT_ELIGIBLE')
        pts=[]
        with (folder/'timeline.stdout').open(encoding='utf-8') as stream:
            for line in stream:
                if line.startswith('frame|'):
                    fields=dict(p.split('=',1) for p in line.strip().split('|')[1:] if '=' in p);pts.append(int(fields['pts']))
        duration=technical.get('duration_seconds')
        if duration is not None:reasons.extend(validate_submission(record,pts,technical['time_base'],duration))
        reasons.extend(duplicate_exclusions.get(vid,[]))
        if not record['events']:reasons.append('NO_POSITIVE_EVENTS;NOT_AUTOMATIC_NEGATIVE')
        accepted=not reasons
        if accepted:
            for event in record['events']:
                accepted_events.append({k:event[k] for k in ['label','start_sec','end_sec','frame_start','frame_end','timestamp_provenance','source_sha256']} | {'video_id':vid,'camera_id':'BENCH_'+vid,'GT_PROTOCOL':protocol})
        row={'video_id':vid,'source_path':record['relative_path'],'source_sha256':record['source_sha256'],
             'submitted':True,'full_view_attested':record['attest_full_view'],'timestamp_provenance':record['source_timestamp_provenance'],
             'source_duration_seconds':duration,'reviewed_duration_seconds':record.get('duration_seconds'),
             'decode_status':technical['decode_status'],'gt_event_count':len(record['events']),
             'fire_events':sum(e['label']=='FIRE' for e in record['events']),'smoke_events':sum(e['label']=='SMOKE' for e in record['events']),
             'uncertain_intervals':len(record['uncertain_intervals']),'event_positive_scope_accepted':accepted,'reasons':sorted(set(reasons)),
             'proposed_split':pin['proposed_split'],'scene_group_id':pin['scene_group_id'],
             'test_independence_verified':False,'bbox_ground_truth_status':'BBOX_GROUND_TRUTH_REQUIRED'}
        scope.append(row)
        if reasons:exclusions.append({'video_id':vid,'reasons':sorted(set(reasons))})
        if any('OVERLAP' in reason or 'MISMATCH' in reason or 'BOUNDS' in reason for reason in reasons):issues.append({'video_id':vid,'reasons':sorted(set(reasons))})
    missing=sorted(set(by_id)-{r['video_id'] for r in primary})
    for vid in missing:exclusions.append({'video_id':vid,'reasons':['NO_SUBMITTED_VN_POS_RECORD','TECHNICAL_TRUNCATION_ALREADY_EXCLUDED' if by_id[vid]['decode_status']=='TRUNCATED_STREAM' else 'MISSING_HUMAN_SUBMISSION']})
    labels=[{e['label'] for e in r['events']} for r in primary]
    summary={
        'GT_PROTOCOL':protocol,'protocol_basis':'OPERATOR_DECLARED_THIS_CONVERSATION','cross_verified':False,
        'export_id':bundle.name,'bundle_checksum_payloads':len(hashes),'all_bundle_checksums_match':True,
        'bundle_records':len(records),'bundle_submitted_bound_to_db':len(records),'bundle_drafts':0,
        'private_db_status_counts':db_counts,'vn_pos_db_status_counts':vn_counts,
        'vn_pos_primary_submitted':len(primary),'vn_pos_unique_sha256':len({r['source_sha256'] for r in primary}),
        'videos_with_fire':sum('FIRE' in s for s in labels),'videos_with_smoke':sum('SMOKE' in s for s in labels),
        'videos_with_both':sum({'FIRE','SMOKE'}<=s for s in labels),'videos_without_events':sum(not s for s in labels),
        'videos_with_uncertainty':sum(bool(r['uncertain_intervals']) for r in primary),
        'raw_gt_event_count':sum(len(r['events']) for r in primary),
        'raw_event_counts':dict(Counter(e['label'] for r in primary for e in r['events'])),
        'source_duration_sum_seconds':sum(r['source_duration_seconds'] or 0 for r in scope),
        'source_duration_once_per_sha256_seconds':sum(r['source_duration_seconds'] or 0 for r in {r['source_sha256']:r for r in scope}.values()),
        'technically_valid_submitted':sum(r['decode_status']=='NORMAL_EOF' for r in scope),
        'accepted_event_positive_videos':sum(r['event_positive_scope_accepted'] for r in scope),
        'accepted_event_counts':dict(Counter(e['label'] for e in accepted_events)),
        'accepted_event_positive_unique_sources':len({r['source_sha256'] for r in scope if r['event_positive_scope_accepted']}),
        'missing_submissions':missing,'exclusions':exclusions,'duplicate_ledger':duplicates,'localized_review_issues':issues,
        'split_counts_preserved':dict(Counter(r['proposed_split'] for r in by_id.values())),
        'accepted_positive_by_split':dict(Counter(r['proposed_split'] for r in scope if r['event_positive_scope_accepted'])),
        'negative_far_exposure_promoted':False,'originals_modified':False,'independent_test_valid':False,
    }
    write_json(out/'week4_gt_bundle_audit.json',summary)
    with (out/'vn_pos_gt_events.jsonl').open('x',encoding='utf-8') as stream:
        for event in accepted_events:stream.write(json.dumps(event,ensure_ascii=False)+'\n')
    for name,rows,fields in [('vn_pos_gt_scope.csv',scope,list(scope[0])),('vn_pos_gt_exclusions.csv',exclusions,['video_id','reasons'])]:
        with (out/name).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    write_json(out/'vn_pos_gt_provenance.json',{'GT_PROTOCOL':protocol,'export_id':bundle.name,'bundle_file_hashes':hashes,
               'bundle_checksum_manifest_sha256':sha256(bundle/'file_sha256.json'),'private_db_access':'READ_ONLY_SNAPSHOT',
               'identities_or_private_review_ids_published':False,'cross_verified':False,'official_frozen_gt_modified':False,
               'derived_gt_events_sha256':sha256(out/'vn_pos_gt_events.jsonl')})
    write_json(out/'week4_gt_lineage.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'created_at_utc':datetime.now(timezone.utc).isoformat(),'prior_preannotation_manifest_sha256':sha256(manifest_path),
               'prior_split_canonical_sha256':declared,'new_research_gt_version':out.name,'duplicate_policy':'Quarantine all conflicting aliases; lowest video ID only for identical boundaries',
               'no_event_policy':'Not an accepted positive event source and never auto-promoted to negative FAR exposure',
               'source_and_human_records_preserved':True})
    print('HUMAN_GT_AUDIT',json.dumps({k:summary[k] for k in ['bundle_records','vn_pos_primary_submitted','raw_event_counts','accepted_event_positive_videos','accepted_event_counts','exclusions']}),flush=True)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--gt-protocol',required=True,choices=['JOINT_HUMAN_CONSENSUS','SINGLE_HUMAN_REVIEW'])
    args=parser.parse_args();audit(args.bundle.resolve(),args.output_dir.resolve(),args.gt_protocol)


if __name__=='__main__':main()
