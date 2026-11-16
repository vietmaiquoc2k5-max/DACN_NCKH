"""Audit a frozen final bbox snapshot and produce traceable blocked/eligible manifests."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ground_truth_studio.bbox_snapshot import committed_reviews,scan_reviews
from ground_truth_studio.bbox_service import load_queue,QUEUE
from ground_truth_studio.bbox_export_adapter import export_approvals
from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import sha256
from src.bbox_final_audit import summarize_decisions
from src.domain_adaptation import read_yolo_boxes
from src.scene_subset import build_subset
from scripts.audit_week5_readiness import read_csv


def audit(snapshot: Path, output: Path, export: Path) -> dict:
    """Validate all decisions and lineage without modifying original sources or labels."""
    output.mkdir(parents=True,exist_ok=False)
    private=Settings.load().private;snapshot=snapshot.resolve()
    manifest=json.loads((snapshot/'snapshot_receipt.json').read_text(encoding='utf-8'))
    for path,digest in manifest['checksums'].items():
        if sha256(snapshot/path)!=digest:raise ValueError('SNAPSHOT_CHECKSUM_MISMATCH')
    records=committed_reviews(snapshot,original_private=private)
    _,issues=scan_reviews(snapshot,original_private=private)
    queue={r['image_sha256']:r for r in load_queue(ROOT)}
    counts=summarize_decisions(set(queue),records);latest={r['image_sha256']:r for r in records}
    proofdir=ROOT/'runs/benchmarks/week5_domain_adaptation/finalize_20261115_9f8bacb3/lineage'
    reservations={'TRAIN_CANDIDATE':'train','TRAIN_LEGACY_RESERVATION':'train',
                  'VALIDATION_CANDIDATE':'val','VAL_LEGACY_RESERVATION':'val',
                  'INDEPENDENT_TEST_CANDIDATE':'test','TEST_LEGACY_RESERVATION':'test'}
    sources={};rows=[];boxes_count=Counter();composition=Counter();lineage=Counter();class_by_split={}
    for key,q in queue.items():
        image=ROOT/q['image_path']
        if sha256(image)!=key:raise ValueError('IMAGE_HASH_MISMATCH')
        for original in q['aliases']:
            if sha256(ROOT/original['label_path'])!=original['label_sha256']:raise ValueError('ORIGINAL_LABEL_CHANGED')
        review=latest.get(key)
        if not review:continue
        label=snapshot/'bbox_qa/versions'/review['version_id']/'labels.txt'
        if sha256(label)!=review['label_sha256']:raise ValueError('HUMAN_LABEL_HASH_MISMATCH')
        if not review.get('reviewer_id') or not review.get('evidence_ref') or not review.get('attest_frame_review'):
            raise ValueError('HUMAN_REVIEW_EVIDENCE_MISSING')
        boxes=read_yolo_boxes(label)
        if review['decision']=='APPROVE' and not boxes and not review.get('negative_frame_approved'):
            raise ValueError('NEGATIVE_ATTESTATION_REQUIRED')
        split=reservations.get(q['reserved_split'],'blocked')
        found={int(b[0]) for b in boxes}
        kind='NEGATIVE' if not found else 'BOTH' if len(found)==2 else 'SMOKE_ONLY' if 0 in found else 'FIRE_ONLY'
        if review['decision']=='APPROVE':
            composition[kind]+=1;boxes_count.update('SMOKE' if int(b[0])==0 else 'FIRE' for b in boxes)
            class_by_split.setdefault(split,Counter())[kind]+=1
        proof_path=proofdir/(key+'.json');proof=json.loads(proof_path.read_text(encoding='utf-8')) if proof_path.exists() else {}
        status=proof.get('status','NO_LINEAGE_RECEIPT');source=proof.get('source_video_sha256')
        if proof:
            if proof['image_sha256']!=key or proof['source_frame_index']!=int(q['source_frame_index_declared']):raise ValueError('LINEAGE_BINDING_MISMATCH')
            if source!=q['current_candidate_source_sha256']:raise ValueError('SOURCE_IDENTITY_MISMATCH')
            source_path=ROOT/proof['source_video_path']
            if str(source_path) not in sources:sources[str(source_path)]=sha256(source_path)
            if sources[str(source_path)]!=source:raise ValueError('SOURCE_HASH_MISMATCH')
            if status in {'VERIFIED_NATIVE_DECODE','VERIFIED_DERIVED_IMAGE'}:
                if type(proof.get('source_pts_ticks')) is not int or not proof.get('time_base'):raise ValueError('NATIVE_PTS_REQUIRED')
                if status=='VERIFIED_DERIVED_IMAGE' and (not proof.get('transformation') or proof.get('match_evidence')!='EXACT_ENCODED_IMAGE_SHA256'):
                    raise ValueError('TRANSFORMATION_EVIDENCE_REQUIRED')
        lineage[status]+=1
        rows.append({'image_sha256':key,'image_path':q['image_path'],'annotation_version':review['version_id'],
            'annotation_metadata_sha256':sha256(snapshot/'bbox_qa/versions'/review['version_id']/'review.json'),
            'label_sha256':review['label_sha256'],'label_path':label.relative_to(ROOT).as_posix(),
            'decision':review['decision'],'human_approved':review['decision']=='APPROVE',
            'source_video_id':q['video_id'],'source_video_sha256':source,
            'source_video_path':proof.get('source_video_path'),'source_frame_index':proof.get('source_frame_index'),
            'source_pts_ticks':proof.get('source_pts_ticks'),'time_base':proof.get('time_base'),
            'lineage_status':status,'lineage_verified':status in {'VERIFIED_NATIVE_DECODE','VERIFIED_DERIVED_IMAGE'},
            'frame_lineage_receipt':proof_path.relative_to(ROOT).as_posix() if proof else None,
            'frame_lineage_receipt_sha256':sha256(proof_path) if proof else None,
            'scene_group_id':None,'scene_status':'NOT_VERIFIED','split':split,'original_reservation':q['reserved_split'],
            'class_mapping':['smoke','fire'],'class_composition':kind,'bbox_count':len(boxes),
            'negative_frame_approved':review.get('negative_frame_approved',False),'verified_hard_negative_tags':[],
            'parent_queue_sha256':sha256(ROOT/QUEUE/'bbox_annotation_queue.csv')})
    blocked={'3ed5b015393ce74d1da992dece66ce4a4b7abcd4b77ce5d3d23264794c4eacbe'}
    subset=build_subset(rows,set(),blocked)
    parent=ROOT/'runs/benchmarks/week5_domain_adaptation/policy_20261116_20b050e6'
    def save(name,value):
        with (output/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    save('review_summary.json',dict(counts,integrity_resolved=sum(i['status']=='RESOLVED_BY_HUMAN_RECONFIRMATION' for i in issues),
        integrity_unresolved=sum(i['status']=='INTEGRITY_QUARANTINED' for i in issues),image_composition=dict(composition),
        bbox_counts=dict(boxes_count),approved_class_by_reservation=class_by_split,lineage_status_counts=dict(lineage),
        distinct_sources_hashed=len(sources),snapshot_manifest_sha256=sha256(snapshot/'snapshot_receipt.json')))
    save('annotation_lineage_manifest.json',{'status':'AUDITED_NOT_TRAINING_READY','rows':rows,
        'parent_snapshot_manifest_sha256':sha256(snapshot/'snapshot_receipt.json'),'original_labels_unchanged':True})
    save('clean_vn_manifest.json',subset)
    for name in ['scene_group_manifest.json','human_source_shortlist_current.json','baseline_membership.json','new_baseline_protocol.json','clean_dfire_manifest.json']:
        save(name,json.loads((parent/name).read_text(encoding='utf-8')))
    save('readiness.json',{'BBOX_REVIEW_COMPLETE':counts['pending']==0,'BBOX_INTEGRITY_READY':True,
        'SPLIT_READY':False,'VN_BBOX_READY':False,'DFIRE_READY':False,'RESOURCE_PILOT_READY':False,'WEEK5_TRAINING_READY':False,
        'eligible_train':subset['train_count'],'eligible_validation':subset['val_count'],'eligible_test':0,
        'reason':'No authenticated opposing source/scene independence decisions; UNKNOWN remains excluded',
        'resource_pilot':None,'training_runs':0,'epochs':0,'metrics':None,
        'parent_readiness_sha256':sha256(parent/'readiness.json')})
    result=export_approvals(ROOT,records,queue,export.resolve(),lineage_dir=proofdir,version_root=snapshot/'bbox_qa/versions')
    committed_reviews(snapshot,original_private=private)
    for path,digest in manifest['checksums'].items():
        if sha256(snapshot/path)!=digest:raise ValueError('SNAPSHOT_CHANGED')
    save('export_receipt.json',{'path':str(export.resolve()),'approved':result['human_approved_count'],
        'rejected':result['rejected_count'],'pending':result['pending_count'],'eligible':result['approved_for_training'],
        'checksums_sha256':sha256(export/'sha256.json')})
    save('artifact_sha256.json',{p.name:sha256(p) for p in output.glob('*.json')})
    return {k:v for k,v in counts.items() if k not in ['latest_versions','missing_image_sha256']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--export',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.snapshot,args.output,args.export)))
