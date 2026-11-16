"""Preserve scoped conversational observations without forging whole-source decisions."""
from __future__ import annotations

from datetime import datetime,timezone


def record_observation(statement: str, evidence: list[dict]) -> dict:
    """Record supplied user text and scope; recording time is not claimed review time."""
    if not statement.strip() or not evidence:
        raise ValueError('USER_STATEMENT_AND_EXACT_EVIDENCE_REQUIRED')
    return {'statement':statement,'attribution':'USER_IN_CURRENT_CONVERSATION',
            'studio_reviewer_id':None,'review_completed_at':None,
            'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'12 contact sheets: displayed sampled frames only',
            'observation':'USER_REPORTED_VISUAL_NON_OVERLAP',
            'whole_source_independence':'UNKNOWN','evidence':evidence,
            'reconfirmation_requested':False,'training_eligibility_granted':False}
