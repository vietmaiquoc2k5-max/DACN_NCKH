"""Đánh giá nghiệm thu Tuần 4 theo Event Recall cấp video và FAR trên video âm."""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger("week4_event_eval")


@dataclass(frozen=True)
class Event:
    """Một sự kiện cháy/khói với khoảng thời gian trên video."""

    video_id: str
    camera_id: str
    label: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate event identity, label, confidence and time interval."""
        if not self.video_id or not self.camera_id:
            raise ValueError("video_id và camera_id không được rỗng")
        if self.label.upper() not in {"FIRE", "SMOKE"}:
            raise ValueError(f"Nhãn không hợp lệ: {self.label}")
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError("Khoảng thời gian sự kiện không hợp lệ")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence phải thuộc [0, 1]")


@dataclass(frozen=True)
class FrameDetection:
    """Các phát hiện tại một timestamp; mỗi detection là một dictionary."""

    video_id: str
    camera_id: str
    timestamp_sec: float
    detections: tuple[dict[str, Any], ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL with line-numbered validation errors.

    Args:
        path: Path to a UTF-8 JSONL file.

    Returns:
        Parsed JSON objects.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("mỗi dòng phải là JSON object")
                records.append(item)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return records


def load_events(path: Path) -> list[Event]:
    """Load event annotations or event predictions from JSONL.

    Args:
        path: JSONL path containing event intervals.

    Returns:
        Validated event list.
    """
    events: list[Event] = []
    for row in load_jsonl(path):
        events.append(Event(
            video_id=str(row["video_id"]),
            camera_id=str(row["camera_id"]),
            label=str(row["label"]).upper(),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            confidence=float(row.get("confidence", 1.0)),
        ))
    return events


def load_frame_detections(path: Path) -> list[FrameDetection]:
    """Load frame-level detections used by confidence and k/N grid search.

    JSONL schema:
      {"video_id":"v1","camera_id":"CAM-01","timestamp_sec":1.2,
       "detections":[{"label":"FIRE","confidence":0.91,"track_id":2}]}

    Args:
        path: JSONL path containing one record per sampled frame.

    Returns:
        Frame detections sorted by video, camera and timestamp.
    """
    frames: list[FrameDetection] = []
    for row in load_jsonl(path):
        timestamp = float(row["timestamp_sec"])
        if timestamp < 0:
            raise ValueError("timestamp_sec phải >= 0")
        detections = tuple(row.get("detections", []))
        for detection in detections:
            label = str(detection.get("label", "")).upper()
            confidence = float(detection.get("confidence", 0.0))
            if label not in {"FIRE", "SMOKE"} or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Detection không hợp lệ: {detection}")
        frames.append(FrameDetection(
            str(row["video_id"]), str(row["camera_id"]), timestamp, detections
        ))
    return sorted(frames, key=lambda x: (x.video_id, x.camera_id, x.timestamp_sec))


def frame_detections_to_events(
    frames: Sequence[FrameDetection],
    confidence_threshold: float,
    window_n: int,
    confirm_k: int,
    max_event_gap_sec: float = 2.0,
) -> tuple[list[Event], list[Event]]:
    """Convert frame detections into baseline and temporal-confirmed event intervals.

    Temporal confirmation is applied independently per video/camera/class/track.
    A frame is positive when at least one detection meets the confidence threshold.
    A temporal alert starts at the timestamp where the rolling N-frame window first
    reaches K positive frames. Events are then merged when their gap is <=
    max_event_gap_sec.

    Args:
        frames: Frame-level predictions.
        confidence_threshold: Minimum confidence for an eligible detection.
        window_n: Number of recent sampled frames in the rolling window.
        confirm_k: Required positive frames within the rolling window.
        max_event_gap_sec: Maximum gap for merging alerts into one event.

    Returns:
        Tuple of (frame-level baseline events, temporal-confirmed events).
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold phải thuộc [0, 1]")
    if window_n < 1 or not 1 <= confirm_k <= window_n:
        raise ValueError("Yêu cầu 1 <= confirm_k <= window_n")
    if max_event_gap_sec < 0:
        raise ValueError("max_event_gap_sec phải >= 0")

    groups: dict[tuple[str, str, str, str], list[tuple[float, bool, float]]] = {}
    for frame in frames:
        present: dict[tuple[str, str, str, str], float] = {}
        for det in frame.detections:
            conf = float(det.get("confidence", 0.0))
            if conf < confidence_threshold:
                continue
            label = str(det["label"]).upper()
            track = str(det.get("track_id", "untracked"))
            key = (frame.video_id, frame.camera_id, label, track)
            present[key] = max(present.get(key, 0.0), conf)
        keys = set(groups) | set(present)
        for key in keys:
            groups.setdefault(key, []).append(
                (frame.timestamp_sec, key in present, present.get(key, 0.0))
            )

    baseline: list[Event] = []
    temporal: list[Event] = []
    for (video_id, camera_id, label, _track), samples in groups.items():
        samples.sort(key=lambda x: x[0])
        base_times = [(t, c) for t, positive, c in samples if positive]
        temporal_times: list[tuple[float, float]] = []
        rolling: list[tuple[float, bool, float]] = []
        was_confirmed = False
        for timestamp, positive, confidence in samples:
            rolling.append((timestamp, positive, confidence))
            rolling = rolling[-window_n:]
            confirmed = sum(1 for _, hit, _ in rolling if hit) >= confirm_k
            if confirmed and not was_confirmed:
                temporal_times.append((timestamp, max(c for _, hit, c in rolling if hit)))
            was_confirmed = confirmed

        baseline.extend(_merge_timestamps(video_id, camera_id, label, base_times, max_event_gap_sec))
        temporal.extend(_merge_timestamps(video_id, camera_id, label, temporal_times, max_event_gap_sec))

    return baseline, temporal


def _merge_timestamps(
    video_id: str,
    camera_id: str,
    label: str,
    samples: Sequence[tuple[float, float]],
    max_gap_sec: float,
) -> list[Event]:
    """Merge positive timestamps into event intervals."""
    if not samples:
        return []
    ordered = sorted(samples)
    result: list[Event] = []
    start = end = ordered[0][0]
    confidences = [ordered[0][1]]
    for timestamp, confidence in ordered[1:]:
        if timestamp - end <= max_gap_sec:
            end = timestamp
            confidences.append(confidence)
        else:
            result.append(Event(video_id, camera_id, label, start, end, max(confidences)))
            start = end = timestamp
            confidences = [confidence]
    result.append(Event(video_id, camera_id, label, start, end, max(confidences)))
    return result


def _is_match(gt: Event, pred: Event, tolerance_sec: float) -> bool:
    """Return whether a prediction overlaps a tolerance-expanded GT event."""
    return (
        gt.video_id == pred.video_id
        and gt.camera_id == pred.camera_id
        and gt.label == pred.label
        and pred.start_sec <= gt.end_sec + tolerance_sec
        and pred.end_sec >= gt.start_sec - tolerance_sec
    )


def evaluate(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    positive_video_ids: set[str],
    negative_video_ids: set[str],
    negative_hours: float,
    tolerance_sec: float = 10.0,
) -> dict[str, Any]:
    """Evaluate video-level Event Recall, event delay and negative-video FAR.

    Event Recall follows the proposal's video-level definition: a positive video
    is recalled when it contains at least one valid alert matching a GT event.
    FAR counts unmatched alert events on explicitly listed negative videos only.

    Args:
        ground_truth: Ground-truth event intervals.
        predictions: Predicted event intervals.
        positive_video_ids: All evaluated fire/smoke-positive video IDs.
        negative_video_ids: All evaluated negative video IDs.
        negative_hours: Total duration of evaluated negative videos in hours.
        tolerance_sec: Allowed temporal matching tolerance.

    Returns:
        JSON-serializable metrics and matched event pairs.
    """
    if negative_hours <= 0:
        raise ValueError("negative_hours phải > 0")
    if not positive_video_ids:
        raise ValueError("positive_video_ids không được rỗng")
    if not negative_video_ids:
        raise ValueError("negative_video_ids không được rỗng")
    if tolerance_sec < 0:
        raise ValueError("tolerance_sec phải >= 0")

    candidates = sorted(
        (abs(pred.start_sec - gt.start_sec), gi, pi)
        for gi, gt in enumerate(ground_truth)
        for pi, pred in enumerate(predictions)
        if _is_match(gt, pred, tolerance_sec)
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            pairs.append((gi, pi))

    recalled_positive_videos = {
        ground_truth[gi].video_id for gi, _ in pairs
        if ground_truth[gi].video_id in positive_video_ids
    }
    video_recall = len(recalled_positive_videos) / len(positive_video_ids)
    event_recall = len(used_gt) / len(ground_truth) if ground_truth else 0.0
    precision = len(used_pred) / len(predictions) if predictions else 0.0
    f1 = 2 * precision * event_recall / (precision + event_recall) if precision + event_recall else 0.0
    false_alarms = sum(
        1 for idx, pred in enumerate(predictions)
        if idx not in used_pred and pred.video_id in negative_video_ids
    )
    delays = [predictions[pi].start_sec - ground_truth[gi].start_sec for gi, pi in pairs]
    return {
        "counts": {
            "ground_truth_events": len(ground_truth),
            "predicted_events": len(predictions),
            "matched_events": len(pairs),
            "recalled_positive_videos": len(recalled_positive_videos),
            "positive_videos": len(positive_video_ids),
            "negative_false_alarms": false_alarms,
        },
        "metrics": {
            "video_event_recall": video_recall,
            "event_instance_recall": event_recall,
            "event_precision": precision,
            "event_f1": f1,
            "false_alarms_per_hour": false_alarms / negative_hours,
            "mean_detection_delay_sec": sum(delays) / len(delays) if delays else None,
            "negative_video_hours": negative_hours,
            "temporal_tolerance_sec": tolerance_sec,
        },
        "matches": [
            {
                "ground_truth": asdict(ground_truth[gi]),
                "prediction": asdict(predictions[pi]),
                "delay_sec": predictions[pi].start_sec - ground_truth[gi].start_sec,
            }
            for gi, pi in pairs
        ],
    }


def grid_search(
    frames: Sequence[FrameDetection],
    ground_truth: Sequence[Event],
    positive_video_ids: set[str],
    negative_video_ids: set[str],
    negative_hours: float,
    confidence_thresholds: Sequence[float],
    k_values: Sequence[int],
    n_values: Sequence[int],
    tolerance_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """Search confidence and K/N combinations, ranked by FAR then delay.

    A candidate is marked feasible only if it satisfies all three proposal
    constraints: FAR reduction >= 50% against the frame-level baseline,
    video Event Recall drop <= 0.05, and mean delay <= 10 seconds.

    Args:
        frames: Frame-level model detections.
        ground_truth: Ground-truth event intervals.
        positive_video_ids: Evaluated positive video IDs.
        negative_video_ids: Evaluated negative video IDs.
        negative_hours: Total evaluated negative video duration.
        confidence_thresholds: Confidence thresholds to test.
        k_values: K values to test.
        n_values: N values to test.
        tolerance_sec: Temporal matching tolerance.

    Returns:
        Candidate results sorted with feasible operating points first.
    """
    baseline_events, _ = frame_detections_to_events(frames, 0.0, 1, 1)
    baseline = evaluate(
        ground_truth, baseline_events, positive_video_ids, negative_video_ids,
        negative_hours, tolerance_sec
    )
    baseline_far = baseline["metrics"]["false_alarms_per_hour"]
    baseline_recall = baseline["metrics"]["video_event_recall"]
    results: list[dict[str, Any]] = []
    for confidence, n, k in itertools.product(confidence_thresholds, n_values, k_values):
        if n < 1 or k < 1 or k > n:
            continue
        _, temporal_events = frame_detections_to_events(frames, confidence, n, k)
        report = evaluate(
            ground_truth, temporal_events, positive_video_ids, negative_video_ids,
            negative_hours, tolerance_sec
        )
        metrics = report["metrics"]
        far_reduction = (
            (baseline_far - metrics["false_alarms_per_hour"]) / baseline_far
            if baseline_far > 0 else 0.0
        )
        recall_drop = baseline_recall - metrics["video_event_recall"]
        delay = metrics["mean_detection_delay_sec"]
        feasible = (
            far_reduction >= 0.50
            and recall_drop <= 0.05
            and delay is not None
            and delay <= 10.0
        )
        results.append({
            "confidence_threshold": confidence,
            "k": k,
            "n": n,
            "metrics": metrics,
            "far_reduction_vs_frame_baseline": far_reduction,
            "video_event_recall_drop": recall_drop,
            "feasible": feasible,
        })
    return sorted(
        results,
        key=lambda item: (
            not item["feasible"],
            item["metrics"]["false_alarms_per_hour"],
            item["metrics"]["mean_detection_delay_sec"]
            if item["metrics"]["mean_detection_delay_sec"] is not None else float("inf"),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run baseline evaluation and optional K/N grid search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--positive-video-ids", required=True, type=Path)
    parser.add_argument("--negative-video-ids", required=True, type=Path)
    parser.add_argument("--negative-hours", required=True, type=float)
    parser.add_argument("--confidence", nargs="+", type=float, default=[0.25, 0.35, 0.45])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--n-values", nargs="+", type=int, default=[5, 8, 10, 12])
    parser.add_argument("--tolerance-sec", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    read_ids = lambda path: {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    gt = load_events(args.ground_truth)
    frames = load_frame_detections(args.frames)
    results = grid_search(
        frames, gt, read_ids(args.positive_video_ids), read_ids(args.negative_video_ids),
        args.negative_hours, args.confidence, args.k_values, args.n_values, args.tolerance_sec
    )
    payload = {"grid_search": results, "best_candidate": results[0] if results else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved %d grid candidates to %s", len(results), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
