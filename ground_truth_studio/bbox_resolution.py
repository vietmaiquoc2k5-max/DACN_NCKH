"""Append-only evidence for human reconfirmation; never repair corrupt checksums."""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from ground_truth_studio.integrity_service import sha256


def fingerprints(folder: Path) -> dict:
    """Bind the exact forensic bytes, including the damaged checksum marker."""
    return {name:sha256(folder/name) if (folder/name).is_file() else None
            for name in ('review.json','review.sha256','labels.txt')}


def apply_resolutions(private: Path, records: list[dict], issues: list[dict]) -> list[dict]:
    """Release only exact affected versions bound to a valid new human annotation."""
    valid={r['version_id']:r for r in records};by_id={i['version_id']:i for i in issues}
    resolved={};invalid=[]
    for marker in sorted((private/'bbox_qa/resolutions').glob('*/resolution.sha256')):
        try:
            path=marker.with_name('resolution.json')
            if sha256(path)!=marker.read_text().strip():raise ValueError('RESOLUTION_HASH_MISMATCH')
            receipt=json.loads(path.read_text(encoding='utf-8'))
            review=valid[receipt['replacement_version_id']]
            if receipt['mode']!='HUMAN_RECONFIRMATION' or receipt['attest_reconfirmed'] is not True:
                raise ValueError('RESOLUTION_SIGNOFF_REQUIRED')
            if not receipt['evidence_ref'] or any(receipt[k]!=review[k] for k in ('reviewer_id','reviewed_at_utc','image_sha256','label_sha256','evidence_ref')):
                raise ValueError('RESOLUTION_REVIEW_BINDING_MISMATCH')
            if not review.get('attest_frame_review') or review.get('decision') not in ('APPROVE','REJECT'):
                raise ValueError('RESOLUTION_NEW_HUMAN_REVIEW_REQUIRED')
            replacement=private/'bbox_qa/versions'/review['version_id']
            if sha256(replacement/'review.json')!=receipt['replacement_metadata_sha256']:
                raise ValueError('RESOLUTION_REPLACEMENT_HASH_MISMATCH')
            affected=receipt['affected_versions']
            if not isinstance(affected,dict) or not affected:raise ValueError('RESOLUTION_AFFECTED_IDS_REQUIRED')
            if any(version not in by_id or
                   by_id[version]['untrusted_claimed_image_sha256']!=review['image_sha256'] or
                   fingerprints(private/'bbox_qa/versions'/version)!=binding
                   for version,binding in affected.items()):
                raise ValueError('RESOLUTION_FORENSIC_BYTES_CHANGED')
            for version in affected:resolved[version]={'resolution_id':path.parent.name,'resolution_sha256':sha256(path)}
        except (OSError,ValueError,KeyError,TypeError) as error:
            logging.getLogger(__name__).error('Invalid bbox resolution %s: %s',marker.parent.name,error)
            invalid.append({'version_id':'RESOLUTION:'+marker.parent.name,'path':str(marker.with_name('resolution.json')),
                            'status':'INTEGRITY_QUARANTINED','error':str(error),
                            'untrusted_claimed_image_sha256':None})
    return [dict(i,status='RESOLVED_BY_HUMAN_RECONFIRMATION',**resolved[i['version_id']])
            if i['version_id'] in resolved else i for i in issues]+invalid


def publish_resolution(private: Path, review: dict, affected: dict, evidence: str) -> dict:
    """Publish a new receipt after authenticated service submission, preserving originals."""
    folder=private/'bbox_qa/resolutions'/str(uuid.uuid4());folder.mkdir(parents=True,exist_ok=False)
    receipt={'resolution_id':folder.name,'mode':'HUMAN_RECONFIRMATION','attest_reconfirmed':True,
             'reviewer_id':review['reviewer_id'],'reviewed_at_utc':review['reviewed_at_utc'],
             'image_sha256':review['image_sha256'],'label_sha256':review['label_sha256'],
             'replacement_version_id':review['version_id'],
             'replacement_metadata_sha256':sha256(private/'bbox_qa/versions'/review['version_id']/'review.json'),
             'affected_versions':affected,'evidence_ref':evidence,
             'original_checksum_authenticated':False,'original_versions_preserved':True}
    path=folder/'resolution.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    pending=folder/'resolution.sha256.pending'
    with pending.open('x',encoding='utf-8') as stream:
        stream.write(sha256(path));stream.flush();os.fsync(stream.fileno())
    pending.rename(folder/'resolution.sha256')
    return receipt
