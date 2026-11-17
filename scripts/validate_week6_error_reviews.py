#!/usr/bin/env python3
"""Error Review Validation Tool for Week 6.
Validates the human review compliance of the 60 false positive error cases.
Strictly prohibits automated status promotion and verifies that any VERIFIED
case has genuine reviewer attribution, valid ISO timestamps, and recognized root causes.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
LOG = logging.getLogger('error_review_validator')

STANDARD_TAXONOMY = {
    'VEHICLE_LIGHT',
    'REFLECTION',
    'LED_DISPLAY',
    'SUNSET_OR_SUNRISE',
    'DUST_OR_HAZE',
    'STEAM',
    'CLOUD_OR_FOG',
    'BRIGHT_OBJECT',
    'CAMERA_ARTIFACT',
    'UNKNOWN',
    'OTHER'
}

VALID_STATUSES = {
    'PENDING_REVIEW',
    'VERIFIED',
    'REJECTED',
    'INCONCLUSIVE'
}

DISALLOWED_REVIEWER_NAMES = {
    '', 'none', 'null', 'system', 'auto', 'script', 'bot', 'unassigned', 'n/a'
}


def parse_iso_timestamp(ts_str: str) -> bool:
    """Check if string is a valid ISO format date/datetime."""
    if not ts_str:
        return False
    try:
        datetime.datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        pass
    # Also support simple YYYY-MM-DD
    try:
        datetime.datetime.strptime(ts_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_error_reviews(
    manifest_csv: Path,
    output_json: Path | None = None
) -> dict[str, Any]:
    """Validate error review manifest and return compliance audit dictionary."""
    if not manifest_csv.exists():
        raise FileNotFoundError(f'Manifest CSV not found: {manifest_csv}')

    with open(manifest_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_cases = len(rows)
    pending_count = 0
    verified_count = 0
    rejected_count = 0
    inconclusive_count = 0
    invalid_count = 0

    cause_distribution: dict[str, int] = {}
    errors: list[dict[str, Any]] = []

    required_fields = [
        'case_id', 'video_id', 'timestamp', 'frame_index',
        'checkpoint_sha256', 'evidence_path', 'suspected_cause',
        'reviewer_name', 'review_timestamp', 'reviewer_status', 'review_notes'
    ]

    for idx, r in enumerate(rows, start=1):
        case_id = r.get('case_id', f'ROW_{idx}')
        missing_fields = [f for f in required_fields if f not in r]
        if missing_fields:
            errors.append({
                'case_id': case_id,
                'error': f'Missing schema fields: {missing_fields}'
            })
            invalid_count += 1
            continue

        status = (r.get('reviewer_status') or '').strip().upper()
        cause = (r.get('suspected_cause') or '').strip().upper()
        reviewer = (r.get('reviewer_name') or '').strip()
        review_ts = (r.get('review_timestamp') or '').strip()
        notes = (r.get('review_notes') or '').strip()
        evidence_path_str = r.get('evidence_path', '').strip()

        if status not in VALID_STATUSES:
            errors.append({
                'case_id': case_id,
                'error': f'Invalid status: {status}. Must be one of {list(VALID_STATUSES)}'
            })
            invalid_count += 1
            continue

        if status == 'PENDING_REVIEW':
            pending_count += 1
            continue
        elif status == 'REJECTED':
            rejected_count += 1
        elif status == 'INCONCLUSIVE':
            inconclusive_count += 1
        elif status == 'VERIFIED':
            # Check rigorous human review criteria
            case_errors = []
            if reviewer.lower() in DISALLOWED_REVIEWER_NAMES:
                case_errors.append(f'Invalid reviewer name: "{reviewer}". Genuine human reviewer required.')

            if not parse_iso_timestamp(review_ts):
                case_errors.append(f'Invalid review timestamp: "{review_ts}". Valid ISO 8601 required.')

            if cause not in STANDARD_TAXONOMY:
                case_errors.append(f'Invalid root cause: "{cause}". Must be one of {list(STANDARD_TAXONOMY)}')

            if cause in {'OTHER', 'UNKNOWN'} and len(notes) < 5:
                case_errors.append(f'Cause "{cause}" requires descriptive review notes (min 5 characters).')

            # Evidence file existence check
            if evidence_path_str and evidence_path_str != 'NOT_EXTRACTED':
                ev_path = ROOT / evidence_path_str
                if not ev_path.exists():
                    case_errors.append(f'Evidence file not found on disk: {evidence_path_str}')
            else:
                case_errors.append('Evidence frame was not extracted.')

            if case_errors:
                errors.append({
                    'case_id': case_id,
                    'error': '; '.join(case_errors)
                })
                invalid_count += 1
            else:
                verified_count += 1
                cause_distribution[cause] = cause_distribution.get(cause, 0) + 1

    meets_quota = verified_count >= 50
    blockers: list[str] = []

    if pending_count > 0:
        blockers.append(f'{pending_count}/{total_cases} error cases are still PENDING_REVIEW by human annotators.')
    if invalid_count > 0:
        blockers.append(f'{invalid_count} error review records failed schema or provenance validation.')
    if not meets_quota:
        blockers.append(f'Verified count ({verified_count}) is below minimum requirement of 50 reviewed cases.')

    if invalid_count > 0:
        validation_status = 'VALIDATION_FAILED'
    elif verified_count >= 50 and pending_count == 0:
        validation_status = 'HUMAN_REVIEW_COMPLETE'
    elif verified_count > 0:
        validation_status = 'PARTIAL_HUMAN_REVIEW'
    else:
        validation_status = 'PENDING_HUMAN_REVIEW'

    report = {
        'total_cases': total_cases,
        'pending_count': pending_count,
        'verified_count': verified_count,
        'rejected_count': rejected_count,
        'inconclusive_count': inconclusive_count,
        'invalid_count': invalid_count,
        'cause_distribution': cause_distribution,
        'validation_status': validation_status,
        'meets_minimum_50_quota': meets_quota,
        'blockers': blockers,
        'errors': errors[:50]  # truncate to first 50 errors if many
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        LOG.info('Validation report saved to %s', output_json)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate Week 6 error review manifest')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'runs/audits/week6/error_review_manifest.csv',
                        help='Path to error review manifest CSV')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/audits/week6/error_review_validation.json',
                        help='Path to output JSON report')
    args = parser.parse_args()

    report = validate_error_reviews(args.manifest, args.output)
    LOG.info('Status: %s (Verified: %d, Pending: %d, Invalid: %d)',
             report['validation_status'], report['verified_count'], report['pending_count'], report['invalid_count'])


if __name__ == '__main__':
    main()
