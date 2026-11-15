"""Join immutable human decisions with independent lineage and split evidence."""
from __future__ import annotations

import json
from pathlib import Path

from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import sha256
from src.domain_adaptation import read_yolo_boxes, validate_class_mapping, validate_dataset_rows


def validate_lineage(root: Path, proof: dict, record: dict) -> None:
    """Check exact native/derived receipts and current source content bindings."""
    if proof.get('status') not in {'VERIFIED_NATIVE_DECODE','VERIFIED_DERIVED_IMAGE'}:
        raise ValueError('LINEAGE_NOT_VERIFIED')
    if proof.get('image_sha256') != record['image_sha256']:
        raise ValueError('LINEAGE_IMAGE_MISMATCH')
    if not isinstance(proof.get('source_frame_index'),int) or proof['source_frame_index']<0:
        raise ValueError('FRAME_INDEX_REQUIRED')
    if not isinstance(proof.get('source_pts_ticks'),int) or not proof.get('time_base'):
        raise ValueError('NATIVE_PTS_REQUIRED')
    if proof['status']=='VERIFIED_DERIVED_IMAGE' and (
        proof.get('match_evidence')!='EXACT_ENCODED_IMAGE_SHA256' or not proof.get('transformation')):
        raise ValueError('DERIVED_TRANSFORM_PROOF_REQUIRED')
    source=confined(root,proof['source_video_path'])
    if sha256(source)!=proof['source_video_sha256']:
        raise ValueError('SOURCE_HASH_MISMATCH')


def export_approvals(root: Path, records: list[dict], queue: dict[str,dict], target: Path,
                     *, lineage_dir: Path | None = None, split_manifest: Path | None = None,
                     version_root: Path | None = None) -> dict:
    """Export human approvals even when independent training evidence is pending.

    Args:
        root: Repository root owning image/source bytes.
        records: Authenticated, checksum-verified committed versions from storage.
        queue: Reserved source candidate records keyed by image SHA.
        target: Fresh private/ignored export directory; originals are never edited.
        lineage_dir: Independently computed per-image receipts, if available.
        split_manifest: Locked, provenance reviewed split with parent manifest binding.
        version_root: Snapshot versions directory, for exporting a consistent backup.

    Returns:
        Manifest separating annotation approval from training eligibility.
    """
    root=root.resolve();target=target.resolve()
    if split_manifest and not target.is_relative_to(root):
        raise ValueError('TRAINING_EXPORT_REQUIRES_WORKSPACE_STAGING')
    target.mkdir(parents=True,exist_ok=False)
    latest={}
    # Caller supplies one reviewer's history or reconciled non-conflicting history.
    owners={}
    for record in records:
        key=record['image_sha256'];owners.setdefault(key,set()).add(record['reviewer_id'])
        latest[key]=record
    if any(len(ids)>1 for ids in owners.values()):
        raise ValueError('MULTI_REVIEWER_ADJUDICATION_REQUIRED')
    splits={};split_sha=None
    if split_manifest:
        split=json.loads(split_manifest.read_text(encoding='utf-8'))
        if split.get('split_locked') is not True or split.get('leakage_review_complete') is not True:
            raise ValueError('SPLIT_REVIEW_REQUIRED')
        parent=confined(root,split['parent_manifest_path'])
        if sha256(parent)!=split['parent_manifest_sha256']:raise ValueError('SPLIT_PARENT_HASH_MISMATCH')
        validate_dataset_rows(split['rows'],require_human=False)
        if any(not r.get('provenance_evidence_ref') or r.get('provenance_verified') is not True for r in split['rows']):
            raise ValueError('SPLIT_PROVENANCE_REQUIRED')
        splits={r['image_sha256']:r for r in split['rows']};split_sha=sha256(split_manifest)
    approved=[];eligible=[];excluded=[];reservations=[]
    for key,record in latest.items():
        if record['decision']!='APPROVE':
            excluded.append({'image_sha256':key,'status':'REJECTED'});continue
        validate_class_mapping(record['class_mapping'])
        if record.get('status')!='SUBMITTED_BBOX_REVIEW' or record.get('attest_frame_review') is not True or not record.get('evidence_ref') or not record.get('reviewed_at_utc'):
            raise ValueError('HUMAN_SIGNOFF_REQUIRED')
        original=Path(record['label_version_path']) if version_root is None else version_root/record['version_id']/'labels.txt'
        if sha256(confined(root,record['image_path']))!=key or sha256(original)!=record['label_sha256']:
            raise ValueError('APPROVAL_CONTENT_HASH_MISMATCH')
        boxes=read_yolo_boxes(original)
        if not boxes and record.get('negative_frame_approved') is not True:
            raise ValueError('EXPLICIT_NEGATIVE_APPROVAL_REQUIRED')
        label=target/(key+'.txt');label.write_bytes(original.read_bytes())
        review_path=target/(key+'.approval.json')
        review_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        approved.append(record)
        source_row=queue[key]
        reservation=source_row.get('reserved_split','UNKNOWN')
        reservations.append({'image_sha256':key,'reservation':reservation})
        proof_path=lineage_dir/(key+'.json') if lineage_dir else None
        if not proof_path or not proof_path.is_file():
            excluded.append({'image_sha256':key,'status':'APPROVED_PENDING_LINEAGE'});continue
        proof=json.loads(proof_path.read_text(encoding='utf-8'))
        try:validate_lineage(root,proof,record)
        except (ValueError,OSError) as error:
            excluded.append({'image_sha256':key,'status':'APPROVED_PENDING_LINEAGE','reason':str(error)});continue
        if key not in splits:
            excluded.append({'image_sha256':key,'status':'APPROVED_LINEAGE_VERIFIED_PENDING_SPLIT'});continue
        split_row=splits[key];split_name=split_row['split']
        allowed={'TRAIN_CANDIDATE':'train','TRAIN_LEGACY_RESERVATION':'train',
                 'VALIDATION_CANDIDATE':'val','VAL_LEGACY_RESERVATION':'val'}
        if split_name not in {'train','val'} or allowed.get(reservation)!=split_name:
            excluded.append({'image_sha256':key,'status':'RESERVED_OR_EXCLUDED_SCOPE'});continue
        if split_row['source_video_sha256']!=proof['source_video_sha256']:
            raise ValueError('SPLIT_SOURCE_BINDING_MISMATCH')
        row={'corpus':'VN','split':split_name,'image_path':record['image_path'],'image_sha256':key,
             'label_path':label.relative_to(root).as_posix(),'label_sha256':record['label_sha256'],
             'bbox_count':len(boxes),'scene_group_id':split_row['scene_group_id'],'lineage_verified':True,
             **{k:proof[k] for k in ('source_video_path','source_video_sha256','source_frame_index','source_pts_ticks','time_base')},
             'frame_lineage_receipt':proof_path.relative_to(root).as_posix(),'frame_lineage_receipt_sha256':sha256(proof_path),
             'bbox_approval_receipt':review_path.relative_to(root).as_posix(),'bbox_approval_receipt_sha256':sha256(review_path),
             'bbox_review':dict(record,source_video_sha256=proof['source_video_sha256'],
                                source_binding_origin='INDEPENDENT_LINEAGE_RECEIPT; ORIGINAL_APPROVAL_UNCHANGED')}
        eligible.append(row)
    if eligible:validate_dataset_rows(eligible)
    annotated={r['image_sha256'] for r in approved}
    if lineage_dir:
        for key in queue.keys()-annotated:
            path=lineage_dir/(key+'.json')
            if path.is_file() and json.loads(path.read_text(encoding='utf-8')).get('status') in {'VERIFIED_NATIVE_DECODE','VERIFIED_DERIVED_IMAGE'}:
                excluded.append({'image_sha256':key,'status':'LINEAGE_VERIFIED_PENDING_HUMAN_APPROVAL'})
    payload={'status':'ANNOTATION_EXPORT_NOT_WEEK5_TRAINING_APPROVAL','approved_bbox_reviews':approved,
             'rows':eligible,'approved_for_training':len(eligible),'human_approved_count':len(approved),
             'pending_count':len(queue)-len(latest),'rejected_count':sum(r['decision']=='REJECT' for r in latest.values()),
             'excluded_rows':excluded,'reservations':reservations,'split_manifest_sha256':split_sha}
    (target/'approved_annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in approved),encoding='utf-8')
    (target/'bbox_export.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    (target/'sha256.json').write_text(json.dumps({p.name:sha256(p) for p in target.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    return payload
