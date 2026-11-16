"""Account for final human decisions without equating review completion with readiness."""
from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_decisions(queue_ids: set[str], records: list[dict]) -> dict[str, Any]:
    """Select latest immutable decision per image and report exact queue coverage.

    Args:
        queue_ids: Unique image SHA identities from the frozen review queue.
        records: Previously integrity-verified annotation versions.

    Returns:
        Counts and missing IDs, without granting dataset eligibility.
    """
    latest={}
    for record in sorted(records,key=lambda r:(r['reviewed_at_utc'],r['version_id'])):
        if record['decision'] not in {'APPROVE','REJECT'}:
            raise ValueError('UNSUPPORTED_FINAL_REVIEW_DECISION')
        latest[record['image_sha256']]=record
    extra=set(latest)-queue_ids
    if extra:raise ValueError('REVIEW_OUTSIDE_FROZEN_QUEUE')
    counts=Counter(r['decision'] for r in latest.values())
    return {'queue_unique_images':len(queue_ids),'valid_annotation_versions':len(records),
            'reviewed_unique_images':len(latest),'approved':counts['APPROVE'],'rejected':counts['REJECT'],
            'pending':len(queue_ids-set(latest)),'missing_image_sha256':sorted(queue_ids-set(latest)),
            'duplicate_history_versions':len(records)-len(latest),
            'training_eligible':False,'latest_versions':{k:v['version_id'] for k,v in latest.items()}}
