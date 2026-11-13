"""Inspect persisted frame evidence without treating counts as timeline proof."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def inspect_frame_jsonl(path: Path) -> dict[str, Any]:
    """Stream frame records, preserving missing fields and malformed-line evidence.

    Timestamp values are observations, not proof of source PTS or run identity.
    Both frame_idx and frame_index are accepted as explicitly persisted indices.
    """
    videos: dict[str, dict[str, Any]] = {}
    pair_hashes: dict[str, Any] = {}
    errors = []
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            hasher.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(row, dict) or not isinstance(row.get("video_id"), str) or not row["video_id"]:
                    raise ValueError("missing video_id")
            except (ValueError, UnicodeError) as error:
                errors.append({"line": number, "error": str(error)})
                continue
            state = videos.setdefault(row["video_id"], {
                "sampled_frame_records": 0, "empty_detection_records": 0,
                "frame_indices_present": True, "frame_indices_valid_and_increasing": True,
                "timestamps_present": True, "timestamps_valid_and_increasing": True,
                "first_timestamp_sec": None, "last_timestamp_sec": None,
                "last_frame_index": None,
                "run_ids": set(), "source_sha256_values": set(), "checkpoint_sha256_values": set(),
                "timestamp_sources": set(), "detector_input_configs": set(),
            })
            state["sampled_frame_records"] += 1
            pair_hasher = pair_hashes.setdefault(row["video_id"], hashlib.sha256())
            state["empty_detection_records"] += row.get("detections") == []
            index = row.get("frame_index", row.get("frame_idx"))
            if index is None:
                state["frame_indices_present"] = False
            elif (type(index) is not int or index < 0 or
                  (state["last_frame_index"] is not None and index <= state["last_frame_index"])):
                state["frame_indices_valid_and_increasing"] = False
            else:
                state["last_frame_index"] = index
            timestamp = row.get("timestamp_sec")
            if timestamp is None:
                state["timestamps_present"] = False
            elif (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or
                  (state["last_timestamp_sec"] is not None and timestamp <= state["last_timestamp_sec"])):
                state["timestamps_valid_and_increasing"] = False
            else:
                if state["first_timestamp_sec"] is None:
                    state["first_timestamp_sec"] = timestamp
                state["last_timestamp_sec"] = timestamp
            if type(index) is int and type(timestamp) in (int, float) and math.isfinite(timestamp):
                pair_hasher.update((json.dumps([index, float(timestamp)]) + "\n").encode())
            for field, target in (("run_id", "run_ids"), ("source_sha256", "source_sha256_values"),
                                  ("checkpoint_sha256", "checkpoint_sha256_values"),
                                  ("timestamp_source", "timestamp_sources"),
                                  ("detector_input_config", "detector_input_configs")):
                state[target].add(json.dumps(row.get(field), sort_keys=True))
    for video_id, state in videos.items():
        state["frame_timestamp_pairs_sha256"] = pair_hashes[video_id].hexdigest()
        for field, value in list(state.items()):
            if isinstance(value, set):
                state[field] = [json.loads(item) for item in sorted(value)]
    return {"path": str(path), "sha256": hasher.hexdigest(), "videos": videos,
            "malformed_lines": errors, "total_frame_records": sum(row["sampled_frame_records"] for row in videos.values())}


def frame_evidence_for_video(video_id: str, audits: list[dict[str, Any]], *,
                             run_id: str | None, source_sha256: str | None,
                             checkpoint_sha256: str | None, expected_samples: int,
                             detector_config: dict[str, Any] | None = None,
                             reference_timestamps: dict[int, float] | None = None) -> dict[str, Any]:
    """Bind evidence to a run and its inputs before verifying timeline coverage.

    Missing metadata or another run cannot establish a contradiction for the target
    run. A bound run with corrupt timestamps or counts is an actual failed check.
    Source PTS must match an independently decoded reference for the selected frame
    indices. Aggregate counts and a self-declared timestamp source are insufficient.
    """
    candidates, bound = [], []
    for audit in audits:
        state = audit["videos"].get(video_id)
        if state is None:
            continue
        candidates.append(audit["path"])
        if (run_id and source_sha256 and checkpoint_sha256
                and state["run_ids"] == [run_id]
                and state["source_sha256_values"] == [source_sha256]
                and state["checkpoint_sha256_values"] == [checkpoint_sha256]):
            bound.append((audit, state))
    timestamps: bool | None = None
    counts: bool | None = None
    preprocessing: bool | None = None
    if bound:
        # Multiple alternative logs require an explicit selection, never concatenation.
        states = []
        for audit, state in bound:
            counts_ok = state["sampled_frame_records"] == expected_samples
            configurations = state["detector_input_configs"]
            config_ok = (configurations == [detector_config]
                         if detector_config and None not in configurations else None)
            invalid = bool(audit["malformed_lines"]) or not state["timestamps_valid_and_increasing"] or not state["frame_indices_valid_and_increasing"]
            missing = (not state["timestamps_present"] or not state["frame_indices_present"]
                       or state["timestamp_sources"] != ["source_pts"] or config_ok is None
                       or reference_timestamps is None)
            if (reference_timestamps is not None and state["timestamps_present"]
                    and state["frame_indices_present"]):
                reference = hashlib.sha256()
                for index, timestamp in sorted(reference_timestamps.items()):
                    reference.update((json.dumps([index, float(timestamp)]) + "\n").encode())
                invalid = invalid or len(reference_timestamps) != expected_samples or reference.hexdigest() != state["frame_timestamp_pairs_sha256"]
            timeline = False if invalid or not counts_ok or config_ok is False else None if missing else True
            states.append((timeline, counts_ok, config_ok))
        if all(state == states[0] for state in states):
            timestamps, counts, preprocessing = states[0]
    return {"candidate_paths": candidates, "bound_run_evidence_count": len(bound),
            "frame_timestamps": timestamps, "sampled_frame_counts_verified": counts,
            "model_preprocessing": preprocessing,
            "full_timeline_independently_verified": timestamps is True}
