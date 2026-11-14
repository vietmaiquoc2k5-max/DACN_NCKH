"""Validate submitted source-time events without altering human annotation records."""
from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Any


def validate_submission(record: dict[str, Any], pts: list[int], time_base: str,
                        duration: float) -> list[str]:
    """Return evidence failures for a submitted review on its authenticated source.

    Args:
        record: Original human export record, retained without edits.
        pts: Exact source-frame presentation timestamps in integer ticks.
        time_base: Source rational time base.
        duration: Accepted source duration from its strict technical audit.

    Returns:
        Error codes. Cross-class overlap is permitted; ambiguous same-class
        duplicate/overlapping intervals require review rather than automatic merge.
    """
    errors = []
    if not record.get('submitted_at_utc') or not record.get('reviewer_id') or not record.get('reviewer_person_id'):
        errors.append('SUBMISSION_OR_REVIEWER_PROVENANCE_MISSING')
    if record.get('review_status') != 'SINGLE_HUMAN_REVIEWED_PROVISIONAL':
        errors.append('UNEXPECTED_EXPORT_SUBMISSION_STATUS')
    if not str(record.get('evidence_ref') or '').strip() or record.get('attest_full_view') is not True:
        errors.append('FULL_VIEW_OR_EVIDENCE_MISSING')
    if not pts or any(b <= a for a,b in zip(pts,pts[1:])) or duration <= 0:
        return sorted(set(errors + ['INVALID_SOURCE_TIMELINE']))
    base = Fraction(time_base)
    identities = set()
    signatures = set()
    events = record.get('events', [])
    uncertain = record.get('uncertain_intervals', [])
    if not isinstance(events, list) or not isinstance(uncertain, list):
        return sorted(set(errors + ['INVALID_INTERVAL_CONTAINER']))
    for is_uncertain, rows in [(False, events), (True, uncertain)]:
        for event in rows:
            try:
                a,b = event['frame_start'],event['frame_end']
                start,end = event['start_sec'],event['end_sec']
                if event['label'] not in ({'UNCERTAIN'} if is_uncertain else {'FIRE','SMOKE'}):
                    errors.append('INVALID_INTERVAL_CLASS')
                if event.get('source_sha256') != record['source_sha256']:
                    errors.append('INTERVAL_SOURCE_SHA_MISMATCH')
                if event.get('timestamp_provenance') != 'SOURCE_PTS':
                    errors.append('EVENT_TIMESTAMP_NOT_SOURCE_PTS')
                if type(a) is not int or type(b) is not int or not 0 <= a < b < len(pts):
                    errors.append('EVENT_FRAME_BOUNDS');continue
                if type(start) not in (int,float) or type(end) not in (int,float) or not all(math.isfinite(t) for t in [start,end]) or not 0 <= start < end <= duration + 1.1e-6:
                    errors.append('EVENT_TIME_BOUNDS');continue
                # Older FFprobe exports rounded seconds to six digits. This is
                # decimal quantization, not a frame-count/decode tolerance.
                if any(abs(float((pts[index]-pts[0])*base)-timestamp) > 1.1e-6 for index,timestamp in [(a,start),(b,end)]):
                    errors.append('EVENT_FRAME_SOURCE_TIME_MISMATCH')
                signature = (event['label'],a,b,start,end)
                if signature in signatures:errors.append('DUPLICATE_EVENT_INTERVAL')
                signatures.add(signature)
                event_id = event.get('event_id')
                if not event_id or event_id in identities:errors.append('MISSING_OR_DUPLICATE_EVENT_ID')
                identities.add(event_id)
            except (KeyError,TypeError,ValueError,ZeroDivisionError):
                errors.append('MALFORMED_EVENT_INTERVAL')
    for index,event in enumerate(events):
        for other in events[index+1:]:
            try:
                if event['label'] == other['label'] and max(event['start_sec'],other['start_sec']) < min(event['end_sec'],other['end_sec']):
                    errors.append('SAME_CLASS_OVERLAP_REQUIRES_REVIEW')
            except (KeyError,TypeError):
                pass
    if uncertain:
        errors.append('UNCERTAIN_INTERVALS_NOT_RESOLVED')
    return sorted(set(errors))


def event_signature(record: dict[str, Any]) -> tuple:
    """Compare semantic/source-frame boundaries without private reviewer/event UUIDs."""
    return tuple(sorted((e['label'],e['frame_start'],e['frame_end'],e['start_sec'],e['end_sec'])
                        for e in record.get('events', [])))


def local_review_issues(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose only ambiguous boundaries, never private identities or event UUIDs."""
    output = []
    fields = ['label', 'frame_start', 'frame_end', 'start_sec', 'end_sec']
    for record in records:
        events = record.get('events', [])
        pairs = []
        for index, first in enumerate(events):
            for second in events[index+1:]:
                if first['label']==second['label'] and max(first['start_sec'], second['start_sec']) < min(first['end_sec'], second['end_sec']):
                    pairs.append({'first': {k:first[k] for k in fields},
                                  'second': {k:second[k] for k in fields}})
        if pairs:
            output.append({'video_id':record['video_id'], 'source_sha256':record['source_sha256'],
                           'status':'SAME_CLASS_OVERLAP_REQUIRES_REVIEW', 'overlap_pairs':pairs,
                           'requested_action':'Confirm event ontology/boundaries only; do not redo the full pool',
                           'original_submission_modified':False})
    _, duplicates = duplicate_policy(records)
    output.extend(duplicates)
    return output


def duplicate_policy(records: list[dict[str, Any]]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Quarantine conflicting aliases; canonicalize identical submissions deterministically.

    Args:
        records: Primary submitted records bound to their current source bytes.

    Returns:
        Exclusions by video ID and a deidentified duplicate reconciliation ledger.
        Conflicting annotations are never averaged, copied or silently selected.
    """
    by_hash: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for record in records:by_hash[record['source_sha256']].append(record)
    excluded: dict[str,list[str]] = {}
    ledger = []
    for digest,group in sorted(by_hash.items()):
        if len(group) < 2:continue
        ordered = sorted(group,key=lambda r:r['video_id'])
        signatures = {(event_signature(r),
                       tuple(sorted((e['label'],e['frame_start'],e['frame_end'],e['start_sec'],e['end_sec'])
                                    for e in r.get('uncertain_intervals',[]))),
                       r.get('decision')) for r in ordered}
        if len(signatures) > 1:
            status = 'CONFLICTING_HUMAN_INTERVALS_SAME_SOURCE';canonical = None
            for record in ordered:excluded[record['video_id']] = [status]
        else:
            status = 'IDENTICAL_BOUNDARIES_CANONICAL_LOWEST_VIDEO_ID';canonical = ordered[0]['video_id']
            for record in ordered[1:]:excluded[record['video_id']] = ['DUPLICATE_SOURCE_ALIAS_EXCLUDED']
        ledger.append({'source_sha256':digest,'video_ids':[r['video_id'] for r in ordered],
                       'status':status,'canonical_video_id':canonical,
                       'intervals_by_alias':{r['video_id']:[dict(label=e['label'],frame_start=e['frame_start'],frame_end=e['frame_end'],start_sec=e['start_sec'],end_sec=e['end_sec']) for e in r['events']] for r in ordered},
                       'human_records_modified':False})
        if any(r.get('uncertain_intervals') for r in ordered):
            ledger[-1]['uncertainty_by_alias']={r['video_id']:[{k:e[k] for k in ['label','frame_start','frame_end','start_sec','end_sec']}
                for e in r.get('uncertain_intervals',[])] for r in ordered}
    return excluded,ledger
