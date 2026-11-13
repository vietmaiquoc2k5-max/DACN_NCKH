"""Source-time sampling and immutable evidence checks for consensus experiments."""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable


def sample_indices(timestamps: list[int], time_base: str, hz: int = 5) -> list[int]:
    """Select first source frame at/after each 1/hz elapsed-time boundary.

    Sparse source frames cover several boundaries only once. No FPS estimates,
    repeated frames or interpolation are used, including for VFR inputs.
    """
    if not timestamps or hz <= 0 or any(b <= a for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError("Missing/non-monotonic source PTS or invalid sample rate")
    base=Fraction(time_base)
    if base<=0: raise ValueError("Invalid time base")
    next_time=Fraction(0); selected=[]
    for index,pts in enumerate(timestamps):
        elapsed=(pts-timestamps[0])*base
        if elapsed>=next_time:
            selected.append(index)
            next_time=Fraction((elapsed*hz).__floor__()+1,hz)
    return selected


def coverage(rows: Iterable[dict[str,Any]], indices: list[int], pts: list[int], time_base: str,
             binding: dict[str,Any]) -> dict[str,Any]:
    """Check every observed record against source reference and immutable inputs."""
    seen=[]
    for row in rows:
        if any(row.get(k)!=v for k,v in binding.items()) or row.get('processing_status')!='OK':
            raise ValueError('Frame input/run/config binding mismatch or processing failure')
        index=row['frame_index']
        if index<0 or index>=len(pts) or row['source_pts_ticks']!=pts[index] or row['time_base']!=time_base:
            raise ValueError('Frame source PTS mismatch')
        if not isinstance(row.get('detections'),list):raise ValueError('Missing observed detections list')
        seen.append(index)
    if seen!=indices:raise ValueError('Incomplete, duplicate or reordered frame coverage')
    return {'complete':True,'expected_frames':len(indices),'processed_frames':len(seen),'source_pts_verified':True}


def require_independent_test(split: dict[str,Any], lock: dict[str,Any]) -> None:
    """Never relabel a contaminated evaluation or provisional config as locked test."""
    if split.get('independent_test_valid') is not True:
        raise ValueError('INDEPENDENT_TEST_CONTAMINATION_OR_UNVERIFIED')
    if lock.get('status')!='LOCKED' or lock.get('validation_scope_hash')!=split.get('validation_scope_hash'):
        raise ValueError('OPERATING_POINT_NOT_VALIDATION_LOCKED')
