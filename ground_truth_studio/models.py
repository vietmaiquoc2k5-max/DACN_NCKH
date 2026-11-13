"""Annotation validation; draft and submitted reviews never imply verification."""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Annotation:
    event_id: str
    label: str
    start_sec: float
    end_sec: float
    frame_start: int | None
    frame_end: int | None
    timestamp_provenance: str
    source_sha256: str

    @classmethod
    def create(cls, label: str, start: dict[str, Any], end: dict[str, Any]) -> "Annotation":
        """Create a stable event identity from two deliberately marked source frames."""
        if start["source_sha256"] != end["source_sha256"]:
            raise ValueError("Marked frames belong to different source bytes")
        provenance = start["timestamp_provenance"]
        if provenance != end["timestamp_provenance"]:
            raise ValueError("Timestamp provenance differs")
        return cls(str(uuid.uuid4()), label, start["timestamp_sec"], end["timestamp_sec"],
                   start["frame_index"], end["frame_index"], provenance, start["source_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_events(events: list[dict[str, Any]], duration: float, *, submit: bool = False,
                    uncertain: bool = False) -> None:
    """Validate bounds, class, stable identity and actual-frame ordinal.

    Raises:
        ValueError: An annotation is incomplete, duplicated or outside the source.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source duration is unavailable")
    if not isinstance(events, list):
        raise ValueError("Annotation events must be a JSON list")
    identities = set()
    signatures = set()
    for event in events:
        if not isinstance(event, dict) or not {"start_sec", "end_sec", "label"}.issubset(event):
            raise ValueError("Malformed annotation event")
        try:
            if isinstance(event["start_sec"], bool) or isinstance(event["end_sec"], bool):
                raise ValueError("Boolean timestamp is invalid")
            start, end = float(event["start_sec"]), float(event["end_sec"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid onset/offset value") from error
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 1e-6:
            raise ValueError("Require 0 <= onset < offset <= source duration")
        if not isinstance(event.get("label"), str) or event["label"] not in ({"UNCERTAIN"} if uncertain else {"FIRE", "SMOKE"}):
            raise ValueError("Invalid interval class")
        if not isinstance(event.get("event_id"), str) or not event["event_id"] or event["event_id"] in identities:
            raise ValueError("Missing or duplicate event identity")
        identities.add(event["event_id"])
        signature = (event["label"], start, end)
        if signature in signatures:
            raise ValueError("Duplicate class/onset/offset interval")
        signatures.add(signature)
        a, b = event.get("frame_start"), event.get("frame_end")
        if submit and (type(a) is not int or type(b) is not int or not 0 <= a < b):
            raise ValueError("Confirm onset/offset on actual source frames before submission")
        if submit and event.get("timestamp_provenance") not in {"SOURCE_PTS", "ESTIMATED_FRAME_INDEX_OVER_FPS"}:
            raise ValueError("Timestamp provenance must be explicit")


def compare_events(first: list[dict[str, Any]], second: list[dict[str, Any]], tolerance: float = .5) -> dict[str, Any]:
    """Find one-to-one agreement without averaging boundaries or inventing events."""
    matches: dict[int, int] = {}

    def augment(i: int, visited: set[int]) -> bool:
        for j, event in enumerate(second):
            if (j not in visited and first[i]["label"] == event["label"]
                    and abs(first[i]["start_sec"] - event["start_sec"]) <= tolerance
                    and abs(first[i]["end_sec"] - event["end_sec"]) <= tolerance):
                visited.add(j)
                if j not in matches or augment(matches[j], visited):
                    matches[j] = i
                    return True
        return False

    for i in range(len(first)):
        augment(i, set())
    pairs = [{"first": i, "second": j, "onset_delta": second[j]["start_sec"] - first[i]["start_sec"],
              "offset_delta": second[j]["end_sec"] - first[i]["end_sec"]} for j, i in sorted(matches.items())]
    unmatched_first = [i for i in range(len(first)) if i not in matches.values()]
    unmatched_second = [j for j in range(len(second)) if j not in matches]
    return {"status": "AGREEMENT" if not unmatched_first and not unmatched_second else "DISPUTED",
            "tolerance_sec": tolerance, "pairs": pairs,
            "unmatched_first": unmatched_first, "unmatched_second": unmatched_second,
            "unmatched_details": [{"first": i, "second": j, "same_class": first[i]["label"] == second[j]["label"],
                                   "onset_delta": second[j]["start_sec"] - first[i]["start_sec"],
                                   "offset_delta": second[j]["end_sec"] - first[i]["end_sec"]}
                                  for i in unmatched_first for j in unmatched_second]}
