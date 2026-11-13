"""Read-only split contracts and deterministic training-only replay planning."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def split_leaks(rows: list[dict[str, Any]]) -> list[str]:
    """Return video groups crossing train/validation/test boundaries."""
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["video_id"]].add(row["split"])
    return sorted(key for key, splits in groups.items() if len(splits) > 1)


def replay_plan(vn_train: list[str], dfire_train: list[str], forbidden: set[str], seed: int = 42) -> dict[str, Any]:
    """Plan 20% D-Fire images in the combined epoch, never val/test replay.

    Args:
        vn_train: Approved Vietnamese training image identities.
        dfire_train: Approved D-Fire training image identities.
        forbidden: Validation/test image identities from either corpus.
        seed: Deterministic without-replacement replay sampling seed.

    Returns:
        A plan only; no images or labels are copied or modified.
    """
    if set(vn_train + dfire_train) & forbidden:
        raise ValueError("Validation/test leakage in replay inputs")
    if len(set(vn_train)) != len(vn_train) or len(set(dfire_train)) != len(dfire_train) or set(vn_train)&set(dfire_train):
        raise ValueError("Duplicate/overlapping training identities")
    count = math.ceil(len(vn_train) / 4)
    if not vn_train or count > len(dfire_train):
        return {"status":"BLOCKED_TRAIN_DATA", "selected":[], "fraction":None}
    selected = random.Random(seed).sample(sorted(dfire_train),count)
    return {"status":"PLANNED_NOT_TRAINED","selected":selected,"fraction":count/(len(vn_train)+count),
            "target_fraction":0.2,"unit":"image presentations per combined epoch; ceil(VN_train/4)","seed":seed}
