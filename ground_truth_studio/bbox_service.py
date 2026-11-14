"""Authenticated, append-only bbox review; event reviews and source labels are immutable."""
from __future__ import annotations

import csv
import json
import math
import uuid
from pathlib import Path
from typing import Any
import cv2

from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import sha256
from ground_truth_studio.models import utc_now

QUEUE = 'runs/benchmarks/week5_domain_adaptation/run_20261114_73034262_v2'


def load_queue(repo: Path) -> list[dict[str, Any]]:
    """Deduplicate actual priority and general queues by authenticated image SHA."""
    result = {}
    for name in ['empty_label_review_129.csv', 'bbox_annotation_queue.csv']:
        with (repo/QUEUE/name).open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                key = row['image_sha256']
                if key not in result:
                    result[key] = dict(row, aliases=[], priority=name.startswith('empty'))
                result[key]['aliases'].append({k:row[k] for k in ['image_path','label_path','label_sha256','reserved_split']})
    return list(result.values())


def validate_boxes(boxes: list[dict]) -> str:
    """Validate normalized boxes without clipping or changing semantic classes."""
    lines = []
    for box in boxes:
        try:
            cls = box['class_id']
            x,y,w,h = (float(box[k]) for k in ['x','y','w','h'])
        except (TypeError,ValueError,KeyError) as error:
            raise ValueError('INVALID_BBOX_ROW') from error
        if cls not in (0,1) or isinstance(cls,bool) or not all(math.isfinite(v) for v in [x,y,w,h]):
            raise ValueError('INVALID_BBOX_CLASS_OR_NUMBER')
        if not (0<w<=1 and 0<h<=1) or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:
            raise ValueError('INVALID_BBOX_BOUNDS')
        lines.append(f'{int(cls)} {x:.9f} {y:.9f} {w:.9f} {h:.9f}')
    return '\n'.join(lines)+ ('\n' if lines else '')


class BBoxReviewService:
    """Reuse Studio actor authentication and private storage, without DB migration."""

    def __init__(self, repository: Any):
        self.repository = repository
        self.repo = repository.settings.repo
        self.private = repository.settings.private/'bbox_qa'
        self.queue = {r['image_sha256']:r for r in load_queue(self.repo)}

    def inspect(self, token: str, identity: str) -> tuple[dict, Path, list[dict]]:
        """Read existing pixels/labels only after authentication and hash checks."""
        self.repository.actor(token)
        row = self.queue[identity]
        image = confined(self.repo,row['image_path'])
        if sha256(image)!=identity:
            raise ValueError('IMAGE_HASH_MISMATCH')
        if cv2.imread(str(image)) is None:
            raise ValueError('CORRUPT_IMAGE')
        boxes = []
        label = confined(self.repo,row['label_path']) if row.get('label_path') else None
        if label and label.exists():
            if sha256(label)!=row['label_sha256']:
                raise ValueError('LABEL_HASH_MISMATCH')
            try:
                for line in label.read_text(encoding='utf-8').splitlines():
                    c,x,y,w,h = map(float,line.split())
                    if c not in (0,1):raise ValueError('Invalid original class')
                    boxes.append(dict(class_id=int(c),x=x,y=y,w=w,h=h))
                validate_boxes(boxes)
            except (ValueError,TypeError):
                row=dict(row,original_label_error=True)
                boxes=[]
        return row,image,boxes

    def history(self, token: str) -> list[dict]:
        """Only expose the current reviewer's immutable annotation versions."""
        actor = self.repository.actor(token)
        records = []
        for path in sorted((self.private/'versions').glob('*/review.json')):
            if not (path.parent/'review.sha256').exists() or sha256(path)!=(path.parent/'review.sha256').read_text().strip():
                raise ValueError('REVIEW_METADATA_HASH_MISMATCH')
            data = json.loads(path.read_text(encoding='utf-8'))
            if data['reviewer_id']==actor['id']:
                records.append(data)
        return sorted(records,key=lambda r:r['reviewed_at_utc'])

    def submit(self, token: str, identity: str, boxes: list[dict], decision: str,
               evidence: str, attest: bool, negative: bool=False) -> dict:
        """Save an explicit human action in a new private annotation version.

        Args:
            token: Existing Studio login session, never a caller-supplied reviewer ID.
            identity: SHA of an image in the approved review queue.
            boxes: Human edited/drawn normalized boxes (0 smoke, 1 fire).
            decision: APPROVE or REJECT, never inferred from detector predictions.
            evidence: Human evidence reference or review explanation.
            attest: Explicit full-image review signoff.
            negative: Explicit negative-frame approval, required for an empty annotation.

        Returns:
            Immutable review metadata. Human bbox approval alone does not verify lineage.
        """
        actor = self.repository.actor(token,{'annotator_01','annotator_02','adjudicator'})
        row,_,_ = self.inspect(token,identity)
        if decision not in {'APPROVE','REJECT'} or not evidence.strip() or attest is not True:
            raise ValueError('HUMAN_SIGNOFF_REQUIRED')
        body = validate_boxes(boxes)
        if decision=='APPROVE' and not body and negative is not True:
            raise ValueError('EXPLICIT_NEGATIVE_APPROVAL_REQUIRED')
        if negative and body:
            raise ValueError('NEGATIVE_APPROVAL_HAS_BOXES')
        folder = self.private/'versions'/str(uuid.uuid4())
        folder.mkdir(parents=True,exist_ok=False)
        label = folder/'labels.txt'
        label.write_text(body,encoding='utf-8')
        record = {'version_id':folder.name,'image_sha256':identity,'image_path':row['image_path'],
            'original_label_sha256':row.get('label_sha256'), 'label_sha256':sha256(label),
            'label_version_path':str(label),'boxes':boxes,'decision':decision,
            'status':'SUBMITTED_BBOX_REVIEW' if decision=='APPROVE' else 'REJECTED_BBOX_REVIEW',
            'reviewer_id':actor['id'],'reviewed_at_utc':utc_now(),'evidence_ref':evidence.strip(),
            'attest_frame_review':True,'negative_frame_approved':bool(negative),
            'source_video_sha256':None,
            'candidate_source_video_sha256':row.get('current_candidate_source_sha256') or None,
            'scene_group_id':row.get('scene_group_candidate'), 'aliases':row['aliases'],
            'lineage_status':'LINEAGE_NOT_VERIFIED','lineage_verified':False,
            'approved_for_training':False,'class_mapping':{'0':'smoke','1':'fire'}}
        with (folder/'review.json').open('x',encoding='utf-8') as stream:
            json.dump(record,stream,ensure_ascii=False,indent=2)
        (folder/'review.sha256').write_text(sha256(folder/'review.json'),encoding='utf-8')
        return record

    def export(self, token: str) -> Path:
        """Export actual latest approvals separately from training eligibility."""
        latest = {r['image_sha256']:r for r in self.history(token)}
        approved = []
        for record in latest.values():
            self.inspect(token,record['image_sha256'])
            if sha256(Path(record['label_version_path']))!=record['label_sha256']:
                raise ValueError('APPROVED_LABEL_HASH_MISMATCH')
            if record['decision']=='APPROVE':
                approved.append(record)
        folder = self.private/'exports'/str(uuid.uuid4())
        folder.mkdir(parents=True,exist_ok=False)
        for record in approved:
            original = Path(record['label_version_path'])
            (folder/(record['image_sha256']+'.txt')).write_bytes(original.read_bytes())
        payload = {'status':'BLOCKED_LINEAGE_NOT_VERIFIED','approved_bbox_reviews':approved,
                   'rows':[],'approved_for_training':0,'human_approved_count':len(approved),
                   'pending_count':len(self.queue)-len(latest),'rejected_count':sum(r['decision']=='REJECT' for r in latest.values())}
        (folder/'bbox_export.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/'sha256.json').write_text(json.dumps({p.name:sha256(p) for p in folder.iterdir() if p.is_file()},indent=2),encoding='utf-8')
        return folder
