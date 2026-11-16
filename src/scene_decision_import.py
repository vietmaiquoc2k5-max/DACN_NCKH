"""Authenticate evidence-bound scene decisions before selecting a pilot candidate."""
from __future__ import annotations

from datetime import datetime,timezone,timedelta
from pathlib import Path

from ground_truth_studio.config import confined
from ground_truth_studio.integrity_service import sha256
from src.scene_subset import validate_scene_decision


def verify_batch(root: Path, scope: dict, decisions: list[dict], actor: dict,
                 existing: list[dict] | None = None) -> list[dict]:
    """Check authenticated uploader, context, exact source pairs and evidence bytes.

    UNKNOWN is not imported as independence. Conflicts and duplicates fail closed.
    """
    if actor.get('role') not in {'annotator_01','annotator_02','adjudicator'}:
        raise PermissionError('SCENE_REVIEW_ROLE_REQUIRED')
    allowed={frozenset(p['member_sha256']) for p in scope['pairs']}
    seen={frozenset(d['member_sha256']) for d in (existing or [])};verified=[]
    for decision in decisions:
        pair=frozenset(decision['member_sha256'])
        if pair not in allowed or len(pair)!=2:raise ValueError('PAIR_OUTSIDE_REVIEW_SCOPE')
        if pair in seen:raise ValueError('DUPLICATE_OR_CONFLICTING_SCENE_DECISION')
        if decision.get('reviewer_id')!=actor['id']:raise PermissionError('SCENE_REVIEWER_IDENTITY_MISMATCH')
        if decision.get('scope_sha256')!=scope['scope_sha256']:raise ValueError('SCENE_CONTEXT_HASH_MISMATCH')
        validate_scene_decision(decision,set(pair))
        stamp=datetime.fromisoformat(decision['reviewed_at_utc'])
        if stamp.tzinfo is None or stamp>datetime.now(timezone.utc)+timedelta(minutes=5):raise ValueError('INVALID_REVIEW_TIME')
        if not decision.get('scene_reason','').strip():raise ValueError('SCENE_REASON_REQUIRED')
        for evidence in decision['evidence']:
            if sha256(confined(root,evidence['path']))!=evidence['sha256']:raise ValueError('SCENE_EVIDENCE_HASH_MISMATCH')
        for source in pair:
            entry=scope['sources'][source]
            if sha256(confined(root,entry['path']))!=source:raise ValueError('SOURCE_SHA_MISMATCH')
        seen.add(pair);verified.append(decision)
    combined=(existing or [])+verified;parents={}
    def find(source: str) -> str:
        parents.setdefault(source,source)
        if parents[source]!=source:parents[source]=find(parents[source])
        return parents[source]
    for item in combined:
        if item['classification']=='CONFIRMED_OVERLAP':
            a,b=item['member_sha256'];parents[find(a)]=find(b)
    for item in combined:
        if item['classification']=='CONFIRMED_INDEPENDENT':
            a,b=item['member_sha256']
            if find(a)==find(b):raise ValueError('TRANSITIVE_SCENE_DECISION_CONFLICT')
    return verified
