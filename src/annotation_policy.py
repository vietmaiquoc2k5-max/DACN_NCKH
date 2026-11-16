"""Proposed ontology guidance; never turn observations into human ground truth."""
from __future__ import annotations

from typing import Any

POLICY_VERSION = 'week5_ontology_proposal_v1'
CATEGORIES = {'FIRE','SMOKE','WELDING_SPARKS','WELDING_FUME','CONTROLLED_FLAME','ACTUAL_FIRE_INCIDENT','AMBIGUOUS'}


def proposed_guidance(category: str) -> dict[str, Any]:
    """Return non-binding review guidance without changing existing YOLO annotations.

    Args:
        category: Human-supplied observation category, not a model prediction.

    Returns:
        A proposal requiring policy approval and per-image human review.
    """
    if category not in CATEGORIES:
        raise ValueError('UNKNOWN_OBSERVATION_CATEGORY')
    suggested = {'FIRE':1,'SMOKE':0,'CONTROLLED_FLAME':1}.get(category)
    return {'policy_version':POLICY_VERSION,'policy_status':'PROPOSED_PENDING_HUMAN_APPROVAL',
            'review_status':'HUMAN_REVIEW_REQUIRED','observation':category,
            'suggested_frame_class_id':suggested,'class_mapping':{'0':'smoke','1':'fire'},
            'metadata_tags':[category],
            'hard_negative_candidate':category in {'WELDING_SPARKS','WELDING_FUME'},
            'confirmed_fire_incident':None,'auto_approved':False,
            'rewrite_existing_labels':False,'negative_attestation_required':suggested is None}


def require_policy_approval(receipt: dict | None) -> None:
    """Fail closed before any future annotation-convention migration."""
    if not receipt or receipt.get('policy_version')!=POLICY_VERSION or receipt.get('decision')!='APPROVE' or any(
        not receipt.get(key) for key in ('reviewer_id','reviewed_at_utc','evidence_ref','policy_sha256')):
        raise ValueError('HUMAN_POLICY_APPROVAL_REQUIRED; no relabeling authorized')
