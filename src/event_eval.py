"""Công cụ đánh giá theo sự kiện (Event-level Evaluator) cho Tuần 4 (`src/event_eval.py`).

Định nghĩa chính thức các chỉ số nghiệm thu theo Đề cương NCKH (Mục tiêu MT3):
1. Video-level Event Recall:
   `Số video dương tính có ít nhất một alert match / Tổng số video dương tính`
2. Event-instance Recall:
   `Số sự kiện Ground Truth được khớp hợp lệ / Tổng số sự kiện Ground Truth`
3. Event Precision & F1-score:
   - `Precision = Số sự kiện dự đoán khớp GT / Tổng số sự kiện dự đoán`
   - `F1 = 2 * Precision * Event_Instance_Recall / (Precision + Event_Instance_Recall)`
4. Mean Detection Delay (Độ trễ phát hiện trung bình):
   `Trung bình (Thời điểm bắt đầu alert đầu tiên khớp GT - Thời điểm bắt đầu GT)`
5. False Alarms per Hour (FAR - Báo động giả mỗi giờ):
   `Số sự kiện cảnh báo không khớp GT trên tập video âm tính độc lập / Tổng số giờ video âm tính`

Quy tắc khoa học bắt buộc:
- Tuyệt đối không tính FAR từ video dương tính hoặc từ khoảng tiền sự kiện rồi gọi đó là FAR trên tập âm tính.
- Tuyệt đối không tự động gán FAR = 0 khi thiếu dữ liệu video âm tính.
- Nếu thiếu `negative_video_ids` hoặc `negative_hours <= 0`:
  + Mặc định (`allow_missing_negative=False`): ném ngoại lệ `ValueError` rõ ràng.
  + Khi chạy ở chế độ báo cáo thiếu dữ liệu (`allow_missing_negative=True`): đánh dấu
    `false_alarms_per_hour = None` và `status = "BLOCKED_BY_NEGATIVE_DATA"`, đồng thời kết luận
    `NO_FEASIBLE_OPERATING_POINT` cho việc nghiệm thu FAR.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

LOGGER = logging.getLogger("event_eval")


@dataclass(frozen=True)
class Event:
    """Một sự kiện cháy/khói với khoảng thời gian [start_sec, end_sec] trên một video."""

    video_id: str
    camera_id: str
    label: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của định danh, nhãn, độ tin cậy và khoảng thời gian."""
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
    """Các phát hiện tại một timestamp của một khung hình; mỗi detection là một dict."""

    video_id: str
    camera_id: str
    timestamp_sec: float
    detections: tuple[dict[str, Any], ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Đọc tệp JSONL với thông báo lỗi chi tiết kèm số dòng."""
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
    """Đọc danh sách sự kiện Ground Truth hoặc dự đoán từ tệp JSONL."""
    events: list[Event] = []
    for row in load_jsonl(path):
        events.append(Event(
            video_id=str(row["video_id"]),
            camera_id=str(row.get("camera_id", f"BENCH_{row['video_id']}")),
            label=str(row["label"]).upper(),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            confidence=float(row.get("confidence", 1.0)),
        ))
    return events


DEFAULT_GT_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth" / "events.jsonl"


def load_ground_truth_events(
    events_path: Path | str | None = None,
    catalog_videos: Sequence[dict[str, Any]] | None = None,
    video_ids: set[str] | None = None,
    only_verified: bool = True,
    strict: bool = True,
    allow_catalog_fallback: bool = False,
) -> list[Event]:
    """Nạp danh sách sự kiện Ground Truth độc lập từ tệp JSONL chuẩn.

    Quy định liêm chính khoa học:
    - Mặc định chỉ nạp các sự kiện có `verification_status == 'verified'` (only_verified=True).
    - Các bản ghi `pending_verification` KHÔNG BAO GIỜ được tham gia vào đánh giá chính thức.
    - Trong chế độ chuẩn/nghiêm ngặt (`strict=True`), nếu tệp events.jsonl bị thiếu hoặc không có
      sự kiện verified nào cho tập video yêu cầu, hàm sẽ ném ngoại lệ rõ ràng, tuyệt đối không
      âm thầm fallback sinh nhãn tự động từ catalog.
    - Fallback catalog chỉ được kích hoạt khi người dùng chỉ định rõ `allow_catalog_fallback=True`.

    Args:
        events_path: Đường dẫn tệp events.jsonl (mặc định data/ground_truth/events.jsonl).
        catalog_videos: Danh sách video từ catalog (chỉ dùng khi allow_catalog_fallback=True).
        video_ids: Bộ lọc video_id nếu chỉ muốn nạp một tập con video.
        only_verified: Nếu True (mặc định), chỉ nạp các event có verification_status == 'verified'.
        strict: Nếu True (mặc định), fail closed nếu không tìm thấy tệp hoặc không có event verified.
        allow_catalog_fallback: Nếu True, cho phép fallback tạo nhãn từ catalog khi JSONL thiếu.

    Returns:
        Danh sách Event đã được chuẩn hóa và kiểm định.
    """
    target_path = Path(events_path) if events_path is not None else DEFAULT_GT_PATH
    if target_path.exists():
        records = load_jsonl(target_path)
        events: list[Event] = []
        for line_no, row in enumerate(records, 1):
            vid = str(row.get("video_id", "")).strip()
            cam_id = str(row.get("camera_id", f"BENCH_{vid}")).strip()
            label = str(row.get("label", "")).upper()
            status = str(row.get("verification_status", "pending_verification")).lower().strip()
            annotator = str(row.get("annotator_id", "")).strip()

            valid_gt_statuses = {"cross_verified", "verified", "annotator_reviewed", "pending_verification", "disputed"}
            if status not in valid_gt_statuses:
                raise ValueError(
                    f"{target_path}: Bản ghi video {vid} có verification_status='{status}' không hợp lệ "
                    f"(chỉ chấp nhận: {sorted(list(valid_gt_statuses))})."
                )

            is_verified = status in {"cross_verified", "verified"}
            if is_verified and (not annotator or annotator.lower() in {"unspecified", "annotator_pending", "none"}):
                raise ValueError(
                    f"{target_path}: Bản ghi {status} video {vid} thiếu annotator_id hợp lệ (tìm thấy: '{annotator}')."
                )

            if video_ids is not None and vid not in video_ids:
                continue

            if only_verified and not is_verified:
                continue

            try:
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{target_path}: Dòng {line_no} video {vid} thiếu hoặc sai định dạng start_sec/end_sec: {exc}") from exc

            if start_sec < 0.0 or end_sec <= start_sec:
                raise ValueError(
                    f"{target_path}: Bản ghi video {vid} có mốc thời gian không hợp lệ: start={start_sec}, end={end_sec}"
                )

            events.append(Event(
                video_id=vid,
                camera_id=cam_id,
                label=label,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=float(row.get("confidence", 1.0)),
            ))

        if strict and only_verified and len(events) == 0 and video_ids is not None:
            raise ValueError(
                f"Không tìm thấy sự kiện Ground Truth verified nào cho tập video yêu cầu: {sorted(list(video_ids))} "
                f"trong tệp {target_path}."
            )

        return events

    # Fallback to catalog_videos chỉ khi được yêu cầu tường minh
    if allow_catalog_fallback and catalog_videos is not None:
        LOGGER.warning("CẢNH BÁO: Tệp Ground Truth độc lập không tồn tại tại %s, fallback sang catalog.", target_path)
        fallback_events: list[Event] = []
        for v in catalog_videos:
            is_pos = v.get("is_positive", True)
            if not is_pos:
                continue
            vid = str(v["video_id"])
            if video_ids is not None and vid not in video_ids:
                continue
            cam_id = str(v.get("camera_id", f"BENCH_{vid}"))
            dur = float(v.get("duration_sec", 0.0))
            f_start = float(v.get("fire_start_sec", -1.0))
            s_start = float(v.get("smoke_start_sec", -1.0))
            if f_start >= 0 and s_start >= 0:
                fallback_events.append(Event(vid, cam_id, "FIRE", f_start, dur, 1.0))
                fallback_events.append(Event(vid, cam_id, "SMOKE", s_start, dur, 1.0))
            elif f_start >= 0:
                fallback_events.append(Event(vid, cam_id, "FIRE", f_start, dur, 1.0))
            elif s_start >= 0:
                fallback_events.append(Event(vid, cam_id, "SMOKE", s_start, dur, 1.0))
        return fallback_events

    raise FileNotFoundError(
        f"Không tìm thấy tệp Ground Truth độc lập tại: {target_path}. "
        "Theo chuẩn nghiệm thu, không sử dụng silent fallback tự sinh nhãn từ catalog."
    )


def get_ground_truth_manifest(
    events_path: Path | str | None = None,
    val_video_ids: set[str] | None = None,
    test_video_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Thống kê chi tiết hiện trạng Ground Truth (verified vs pending) theo nhãn và phân chia tập dữ liệu."""
    target_path = Path(events_path) if events_path is not None else DEFAULT_GT_PATH
    if not target_path.exists():
        return {"status": "FILE_NOT_FOUND", "path": str(target_path)}

    records = load_jsonl(target_path)
    val_ids = val_video_ids or set()
    test_ids = test_video_ids or set()

    total_records = len(records)
    verified_records = [r for r in records if str(r.get("verification_status", "")).lower() in {"cross_verified", "verified"}]
    pending_records = [r for r in records if str(r.get("verification_status", "")).lower() not in {"cross_verified", "verified"}]

    return {
        "path": str(target_path),
        "total_records": total_records,
        "verified_count": len(verified_records),
        "pending_count": len(pending_records),
        "annotation_status": "COMPLETE" if len(pending_records) == 0 else "INCOMPLETE_PENDING_ANNOTATIONS",
        "verified_breakdown": {
            "fire": sum(1 for r in verified_records if str(r.get("label", "")).upper() == "FIRE"),
            "smoke": sum(1 for r in verified_records if str(r.get("label", "")).upper() == "SMOKE"),
            "validation_split": sum(1 for r in verified_records if str(r.get("video_id", "")) in val_ids),
            "test_split": sum(1 for r in verified_records if str(r.get("video_id", "")) in test_ids),
            "videos": sorted(list({str(r["video_id"]) for r in verified_records})),
        },
        "pending_breakdown": {
            "fire": sum(1 for r in pending_records if str(r.get("label", "")).upper() == "FIRE"),
            "smoke": sum(1 for r in pending_records if str(r.get("label", "")).upper() == "SMOKE"),
            "validation_split": sum(1 for r in pending_records if str(r.get("video_id", "")) in val_ids),
            "test_split": sum(1 for r in pending_records if str(r.get("video_id", "")) in test_ids),
            "videos": sorted(list({str(r["video_id"]) for r in pending_records})),
        },
    }


def load_frame_detections(path: Path) -> list[FrameDetection]:
    """Đọc danh sách phát hiện cấp khung hình (frame-level detections) từ JSONL."""
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


def extract_frame_level_alarms(
    frames: Sequence[FrameDetection],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Trích xuất danh sách cảnh báo thô cấp khung hình (Pipeline A - Frame-level baseline).

    Mỗi frame có detection FIRE/SMOKE đạt ngưỡng `confidence_threshold` tạo ra một cảnh báo
    cấp khung hình. Không dùng ByteTrack, không dùng K/N, không dùng TemporalVerifier,
    không dùng Cooldown. Dữ liệu gốc cấp khung hình được giữ nguyên vẹn.

    Args:
        frames: Chuỗi FrameDetection đầu vào.
        confidence_threshold: Ngưỡng độ tin cậy tối thiểu [0, 1].

    Returns:
        Danh sách các bản ghi cảnh báo cấp khung hình theo thứ tự thời gian.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold phải thuộc [0, 1]")

    alarms: list[dict[str, Any]] = []
    ordered_frames = sorted(frames, key=lambda f: (f.video_id, f.camera_id, f.timestamp_sec))
    for frame in ordered_frames:
        by_label: dict[str, float] = {}
        for det in frame.detections:
            conf = float(det.get("confidence", 0.0))
            if conf < confidence_threshold:
                continue
            label = str(det["label"]).upper()
            by_label[label] = max(by_label.get(label, 0.0), conf)
        for label, max_conf in sorted(by_label.items()):
            alarms.append({
                "video_id": frame.video_id,
                "camera_id": frame.camera_id,
                "timestamp_sec": float(frame.timestamp_sec),
                "label": label,
                "confidence": round(float(max_conf), 4),
            })
    return alarms


def merge_frame_alarms_to_events(
    alarms: Sequence[dict[str, Any]],
    max_event_gap_sec: float = 2.0,
) -> list[Event]:
    """Gộp các cảnh báo cấp khung hình liên tiếp thành các khoảng sự kiện Event.

    Quy tắc gộp: Các cảnh báo cùng `(video_id, camera_id, label)` có khoảng cách thời gian
    giữa hai lần xuất hiện liên tiếp `<= max_event_gap_sec` được gộp vào cùng một Event.
    """
    if max_event_gap_sec < 0:
        raise ValueError("max_event_gap_sec phải >= 0")

    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for alarm in alarms:
        key = (
            str(alarm["video_id"]),
            str(alarm["camera_id"]),
            str(alarm["label"]).upper(),
        )
        grouped.setdefault(key, []).append((float(alarm["timestamp_sec"]), float(alarm["confidence"])))

    events: list[Event] = []
    for (video_id, camera_id, label), samples in grouped.items():
        events.extend(_merge_timestamps(video_id, camera_id, label, samples, max_event_gap_sec))
    return sorted(events, key=lambda e: (e.video_id, e.camera_id, e.start_sec, e.label))


def frame_detections_to_events(
    frames: Sequence[FrameDetection],
    confidence_threshold: float,
    window_n: int,
    confirm_k: int,
    max_event_gap_sec: float = 2.0,
    feature_threshold: float = 0.0,
    cooldown_sec: float = 0.0,
) -> tuple[list[Event], list[Event]]:
    """Chuyển đổi phát hiện cấp khung hình sang (1) sự kiện baseline từng khung và (2) sự kiện xác nhận theo thời gian.

    - Baseline events (Pipeline A): cảnh báo từ từng khung hình đơn lẻ đạt `confidence_threshold`
      (không dùng track_id, không dùng K/N, không dùng Cooldown), sau đó gộp theo `max_event_gap_sec`.
    - Temporal-confirmed events (Pipeline B): áp dụng cửa sổ trượt K/N độc lập trên từng
      `(video_id, camera_id, label, track_id)`, kết hợp ngưỡng biến động `feature_threshold`
      và giãn cách `cooldown_sec`.

    Args:
        frames: Chuỗi FrameDetection đã lấy mẫu.
        confidence_threshold: Ngưỡng độ tin cậy tối thiểu [0, 1].
        window_n: Kích thước cửa sổ trượt N khung hình.
        confirm_k: Số khung hình dương tính K cần đạt trong cửa sổ N.
        max_event_gap_sec: Khoảng cách tối đa (giây) để gộp các cảnh báo thành một Event.
        feature_threshold: Ngưỡng điểm biến động đặc trưng thời gian tối thiểu.
        cooldown_sec: Thời gian Cooldown tối thiểu giữa hai lần phát cảnh báo mới.

    Returns:
        Tuple gồm `(baseline_events, temporal_confirmed_events)`.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold phải thuộc [0, 1]")
    if window_n < 1 or not 1 <= confirm_k <= window_n:
        raise ValueError("Yêu cầu 1 <= confirm_k <= window_n")
    if max_event_gap_sec < 0:
        raise ValueError("max_event_gap_sec phải >= 0")

    # 1. Pipeline A: Frame-level baseline (không dùng track_id, không dùng K/N)
    raw_alarms = extract_frame_level_alarms(frames, confidence_threshold)
    baseline = merge_frame_alarms_to_events(raw_alarms, max_event_gap_sec=max_event_gap_sec)

    # 2. Pipeline B: Temporal-confirmed theo từng (video_id, camera_id, label, track_id)
    # Nhóm các frame theo (video_id, camera_id) để cửa sổ trượt N khung hình phản ánh đúng chuỗi frame thực tế
    stream_frames: dict[tuple[str, str], list[FrameDetection]] = {}
    for frame in sorted(frames, key=lambda f: (f.video_id, f.camera_id, f.timestamp_sec)):
        stream_frames.setdefault((frame.video_id, frame.camera_id), []).append(frame)

    temporal: list[Event] = []
    for (video_id, camera_id), cam_frames in stream_frames.items():
        # Tập hợp tất cả (label, track_id) xuất hiện trong luồng này
        track_keys: set[tuple[str, str]] = set()
        per_frame_hits: list[dict[tuple[str, str], tuple[float, float]]] = []
        for frame in cam_frames:
            present: dict[tuple[str, str], tuple[float, float]] = {}
            for det in frame.detections:
                conf = float(det.get("confidence", 0.0))
                if conf < confidence_threshold:
                    continue
                label = str(det["label"]).upper()
                tid_raw = det.get("track_id", "untracked")
                # Nếu có tracking từ ByteTrack (tid_raw là int), bỏ qua các detection bị ByteTrack từ chối (<= 0)
                if isinstance(tid_raw, int) and tid_raw <= 0:
                    continue
                track = str(tid_raw)
                feat_score = float(
                    det.get(
                        "temporal_score",
                        det.get("feature_score", det.get("flicker_score", 0.0)),
                    )
                )
                key = (label, track)
                prev_conf, prev_feat = present.get(key, (0.0, 0.0))
                present[key] = (max(prev_conf, conf), max(prev_feat, feat_score))
                track_keys.add(key)
            per_frame_hits.append(present)

        # Đánh giá cửa sổ trượt K/N cho từng (label, track_id)
        label_confirmed_samples: dict[str, list[tuple[float, float]]] = {}
        for label, track in sorted(track_keys):
            rolling: list[tuple[float, bool, float, float]] = []
            was_confirmed = False
            in_active_alert = False
            last_alert_ts: float | None = None
            for frame, present in zip(cam_frames, per_frame_hits):
                hit = (label, track) in present
                conf, feat_val = present.get((label, track), (0.0, 0.0))
                rolling.append((frame.timestamp_sec, hit, conf, feat_val))
                rolling = rolling[-window_n:]

                hit_count = sum(1 for _, h, _, _ in rolling if h)
                mean_feat = (
                    sum(fv for _, h, _, fv in rolling if h) / float(hit_count)
                    if hit_count > 0 else 0.0
                )
                passes_feat = (feature_threshold <= 0.0) or (mean_feat >= feature_threshold)
                confirmed = (hit_count >= confirm_k) and passes_feat

                if confirmed:
                    if not was_confirmed:
                        if (
                            cooldown_sec <= 0.0
                            or last_alert_ts is None
                            or (frame.timestamp_sec - last_alert_ts) >= cooldown_sec
                        ):
                            in_active_alert = True
                            last_alert_ts = frame.timestamp_sec
                        else:
                            in_active_alert = False
                    if in_active_alert:
                        peak_conf = max(c for _, h, c, _ in rolling if h)
                        label_confirmed_samples.setdefault(label, []).append(
                            (frame.timestamp_sec, peak_conf)
                        )
                else:
                    in_active_alert = False
                was_confirmed = confirmed

        for label, samples in label_confirmed_samples.items():
            temporal.extend(_merge_timestamps(video_id, camera_id, label, samples, max_event_gap_sec))

    temporal.sort(key=lambda e: (e.video_id, e.camera_id, e.start_sec, e.label))
    return baseline, temporal


def _merge_timestamps(
    video_id: str,
    camera_id: str,
    label: str,
    samples: Sequence[tuple[float, float]],
    max_gap_sec: float,
) -> list[Event]:
    """Gộp các mốc thời gian dương tính liên tiếp thành khoảng sự kiện Event."""
    if not samples:
        return []
    ordered = sorted(samples, key=lambda x: x[0])
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


def is_timely_onset_match(
    gt: Event,
    pred: Event,
    tolerance_sec: float = 10.0,
    early_tolerance_sec: float | None = None,
) -> bool:
    """Kiểm tra sự kiện dự đoán có khớp kịp thời (timely detection) với thời điểm khởi phát GT hay không.

    Định nghĩa Onset Matching:
    1. Cùng video_id, camera_id và nhãn (FIRE/SMOKE).
    2. Dự đoán phải giao thoa hoặc kéo dài vào khoảng thời gian cháy/khói diễn ra: `pred.end_sec >= gt.start_sec`.
    3. Dự đoán không được bắt đầu trước GT onset vượt quá `early_tolerance_sec`:
       `pred.start_sec >= gt.start_sec - early_tolerance_sec` (mặc định early_tolerance_sec = tolerance_sec).
       Các dự đoán kết thúc trước khi GT bắt đầu hoặc bắt đầu quá sớm được coi là Early Alert / Cảnh báo sớm.
    4. Dự đoán bắt đầu không được muộn hơn thời điểm khởi phát cộng với dung sai:
       `pred.start_sec <= gt.start_sec + tolerance_sec`.
       (Dự đoán bắt đầu sau `gt.start_sec + tolerance_sec` được tính là Late Detection,
       tuyệt đối không được tính là timely detection).
    """
    if tolerance_sec < 0:
        raise ValueError("tolerance_sec phải >= 0")
    early_tol = tolerance_sec if early_tolerance_sec is None else early_tolerance_sec
    if early_tol < 0:
        raise ValueError("early_tolerance_sec phải >= 0")

    return (
        gt.video_id == pred.video_id
        and gt.camera_id == pred.camera_id
        and gt.label == pred.label
        and pred.end_sec >= gt.start_sec
        and pred.start_sec >= (gt.start_sec - early_tol)
        and pred.start_sec <= (gt.start_sec + tolerance_sec)
    )


def _is_match(gt: Event, pred: Event, tolerance_sec: float, early_tolerance_sec: float | None = None) -> bool:
    """Hàm tương thích ngược: chuyển tiếp trực tiếp sang `is_timely_onset_match`."""
    return is_timely_onset_match(gt, pred, tolerance_sec=tolerance_sec, early_tolerance_sec=early_tolerance_sec)


def max_cardinality_min_cost_matching(
    cost_matrix: list[list[float | None]],
) -> list[tuple[int, int]]:
    """Giải bài toán ghép cặp cực đại với tổng chi phí nhỏ nhất (Maximum Cardinality, Minimum Cost Bipartite Matching).

    Sử dụng thuật toán Min-Cost Max-Flow (Successive Shortest Path qua SPFA) độc lập, thuần Python:
    1. Cực đại hóa số lượng cặp ghép hợp lệ (cardinality |M|).
    2. Trong số các tập ghép có cardinality cực đại, cực tiểu hóa tổng độ trễ onset chênh lệch.
    3. Đảm bảo ánh xạ 1-1 chặt chẽ (one-to-one matching).

    Args:
        cost_matrix: Ma trận chi phí kích thước m x n, trong đó cost_matrix[i][j] là độ trễ onset (float >= 0),
                     hoặc None nếu không tồn tại cạnh hợp lệ giữa GT i và Pred j.

    Returns:
        Danh sách các cặp chỉ số (i, j) được ghép cặp tối ưu.
    """
    m = len(cost_matrix)
    if m == 0:
        return []
    n = len(cost_matrix[0])
    if n == 0:
        return []

    source = m + n
    sink = m + n + 1
    num_nodes = m + n + 2

    class FlowEdge:
        __slots__ = ("to", "cap", "cost", "rev")

        def __init__(self, to: int, cap: int, cost: float, rev: int) -> None:
            self.to = to
            self.cap = cap
            self.cost = cost
            self.rev = rev

    graph: list[list[FlowEdge]] = [[] for _ in range(num_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: float) -> None:
        graph[u].append(FlowEdge(v, cap, cost, len(graph[v])))
        graph[v].append(FlowEdge(u, 0, -cost, len(graph[u]) - 1))

    for i in range(m):
        add_edge(source, i, 1, 0.0)

    for j in range(n):
        add_edge(m + j, sink, 1, 0.0)

    BIG_REWARD = 10_000.0
    for i in range(m):
        for j in range(n):
            c = cost_matrix[i][j]
            if c is not None:
                add_edge(i, m + j, 1, float(c) - BIG_REWARD)

    flow = 0
    while True:
        dist = [float("inf")] * num_nodes
        prev_node = [-1] * num_nodes
        prev_edge = [-1] * num_nodes
        in_queue = [False] * num_nodes

        dist[source] = 0.0
        queue = [source]
        in_queue[source] = True
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            in_queue[u] = False

            for e_idx, e in enumerate(graph[u]):
                if e.cap > 0 and dist[e.to] > dist[u] + e.cost + 1e-9:
                    dist[e.to] = dist[u] + e.cost
                    prev_node[e.to] = u
                    prev_edge[e.to] = e_idx
                    if not in_queue[e.to]:
                        queue.append(e.to)
                        in_queue[e.to] = True

        if dist[sink] >= -BIG_REWARD / 2.0:
            break

        curr = sink
        while curr != source:
            p = prev_node[curr]
            e_idx = prev_edge[curr]
            graph[p][e_idx].cap -= 1
            rev_idx = graph[p][e_idx].rev
            graph[curr][rev_idx].cap += 1
            curr = p

        flow += 1

    matches: list[tuple[int, int]] = []
    for i in range(m):
        for e in graph[i]:
            if m <= e.to < m + n and e.cap == 0:
                matches.append((i, e.to - m))

    matches.sort(key=lambda pair: pair[0])
    return matches


def evaluate(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    positive_video_ids: set[str],
    negative_video_ids: set[str] | None = None,
    negative_hours: float | None = None,
    tolerance_sec: float = 10.0,
    early_tolerance_sec: float | None = None,
    allow_missing_negative: bool = False,
) -> dict[str, Any]:
    """Đánh giá Video-level Event Recall, Event-instance Recall, Precision, F1, Delay và Negative FAR.

    Định nghĩa chuẩn dựa trên Onset Matching:
    - `video_event_recall` = Số video dương tính có ít nhất 1 alert khớp kịp thời GT / Tổng số video dương tính verified.
    - `mean_detection_delay_sec` = Trung bình max(0.0, pred.start_sec - gt.start_sec) trên các cặp khớp kịp thời.
    - `false_alarms_per_hour` = Số alert event không khớp GT trên các video thuộc `negative_video_ids`
      chia cho `negative_hours`.

    Nếu thiếu `negative_video_ids` hoặc `negative_hours <= 0`:
    - Khi `allow_missing_negative=False` (mặc định): Ném `ValueError` để ngăn báo cáo FAR sai lệch.
    - Khi `allow_missing_negative=True`: Trả về `false_alarms_per_hour = None` và
      `status = "BLOCKED_BY_NEGATIVE_DATA"`, tuyệt đối không gán FAR = 0.
    """
    if not positive_video_ids:
        raise ValueError("positive_video_ids không được rỗng")
    if tolerance_sec < 0:
        raise ValueError("tolerance_sec phải >= 0")
    early_tol = tolerance_sec if early_tolerance_sec is None else early_tolerance_sec
    if early_tol < 0:
        raise ValueError("early_tolerance_sec phải >= 0")

    has_valid_negative = (
        negative_video_ids is not None
        and len(negative_video_ids) > 0
        and negative_hours is not None
        and float(negative_hours) > 0.0
    )
    if not has_valid_negative and not allow_missing_negative:
        if negative_hours is None or float(negative_hours) <= 0.0:
            raise ValueError("negative_hours phải > 0")
        raise ValueError("negative_video_ids không được rỗng")

    neg_ids: set[str] = set(negative_video_ids) if negative_video_ids else set()
    neg_hrs: float | None = float(negative_hours) if (has_valid_negative and negative_hours is not None) else None

    # Nhóm GT và Predictions theo (video_id, camera_id, label) để thực hiện Maximum Bipartite Matching
    grouped_gt: dict[tuple[str, str, str], list[tuple[int, Event]]] = {}
    for gi, gt in enumerate(ground_truth):
        key = (gt.video_id, gt.camera_id, gt.label)
        grouped_gt.setdefault(key, []).append((gi, gt))

    grouped_pred: dict[tuple[str, str, str], list[tuple[int, Event]]] = {}
    for pi, pred in enumerate(predictions):
        key = (pred.video_id, pred.camera_id, pred.label)
        grouped_pred.setdefault(key, []).append((pi, pred))

    all_keys = set(grouped_gt.keys()) | set(grouped_pred.keys())
    matched_pairs: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    for key in sorted(all_keys):
        gts = grouped_gt.get(key, [])
        preds = grouped_pred.get(key, [])
        if not gts or not preds:
            continue

        cost_matrix: list[list[float | None]] = []
        for _, gt in gts:
            row: list[float | None] = []
            for _, pred in preds:
                if is_timely_onset_match(gt, pred, tolerance_sec=tolerance_sec, early_tolerance_sec=early_tol):
                    delay = max(0.0, pred.start_sec - gt.start_sec)
                    row.append(delay)
                else:
                    row.append(None)
            cost_matrix.append(row)

        local_matches = max_cardinality_min_cost_matching(cost_matrix)
        for g_idx, p_idx in local_matches:
            gi = gts[g_idx][0]
            pi = preds[p_idx][0]
            matched_pairs.append((gi, pi))
            used_gt.add(gi)
            used_pred.add(pi)

    # Phân loại các dự đoán không được ghép cặp
    late_detections: list[dict[str, Any]] = []
    early_alerts: list[dict[str, Any]] = []
    negative_false_alarms: list[dict[str, Any]] = []
    unmatched_positive_predictions: list[dict[str, Any]] = []

    for pi, pred in enumerate(predictions):
        if pi in used_pred:
            continue
        pred_dict = asdict(pred)
        if pred.video_id in neg_ids:
            negative_false_alarms.append(pred_dict)
        elif pred.video_id in positive_video_ids:
            # Kiểm tra xem có phải late detection hay early alert đối với GT cùng video/label không
            video_label_gts = [
                gt for gt in ground_truth
                if gt.video_id == pred.video_id and gt.label == pred.label
            ]
            is_late = False
            is_early = False
            if video_label_gts:
                for gt in video_label_gts:
                    if pred.start_sec > (gt.start_sec + tolerance_sec) and pred.start_sec <= (gt.end_sec + tolerance_sec) and pred.end_sec >= gt.start_sec:
                        is_late = True
                        break
                    if pred.end_sec < gt.start_sec or pred.start_sec < (gt.start_sec - early_tol):
                        is_early = True
            if is_late:
                late_detections.append(pred_dict)
            elif is_early:
                early_alerts.append(pred_dict)
            else:
                unmatched_positive_predictions.append(pred_dict)

    recalled_positive_videos = {
        ground_truth[gi].video_id for gi, _ in matched_pairs
        if ground_truth[gi].video_id in positive_video_ids
    }
    verified_positive_video_ids = {
        gt.video_id for gt in ground_truth if gt.video_id in positive_video_ids
    }

    # Video recall được tính trên tập video có verified GT
    video_recall = (
        len(recalled_positive_videos) / len(verified_positive_video_ids)
        if verified_positive_video_ids else 0.0
    )
    split_video_recall = (
        len(recalled_positive_videos) / len(positive_video_ids)
        if positive_video_ids else 0.0
    )
    event_recall = len(matched_pairs) / len(ground_truth) if ground_truth else 0.0
    precision = len(matched_pairs) / len(predictions) if predictions else 0.0
    f1 = 2 * precision * event_recall / (precision + event_recall) if (precision + event_recall) > 0 else 0.0

    total_positive_unmatched_alerts = len(late_detections) + len(early_alerts) + len(unmatched_positive_predictions)

    if has_valid_negative and neg_hrs is not None:
        false_alarms: int | None = len(negative_false_alarms)
        far_per_hour: float | None = false_alarms / neg_hrs
        eval_status = "COMPLETE"
        far_status = "VERIFIED"
    else:
        false_alarms = None
        far_per_hour = None
        eval_status = "BLOCKED_BY_NEGATIVE_DATA"
        far_status = "UNAVAILABLE_MISSING_NEGATIVE_VIDEOS"

    delays = [max(0.0, predictions[pi].start_sec - ground_truth[gi].start_sec) for gi, pi in matched_pairs]

    unmatched_gt = [
        asdict(gt) for gi, gt in enumerate(ground_truth) if gi not in used_gt
    ]
    unmatched_preds = [
        {
            **asdict(pred),
            "is_negative_video": pred.video_id in neg_ids,
            "is_positive_video": pred.video_id in positive_video_ids,
            "is_late_detection": any(p["video_id"] == pred.video_id and p["start_sec"] == pred.start_sec for p in late_detections),
            "is_early_alert": any(p["video_id"] == pred.video_id and p["start_sec"] == pred.start_sec for p in early_alerts),
        }
        for pi, pred in enumerate(predictions) if pi not in used_pred
    ]

    return {
        "status": eval_status,
        "counts": {
            "ground_truth_events": len(ground_truth),
            "predicted_events": len(predictions),
            "matched_events": len(matched_pairs),
            "timely_detections": len(matched_pairs),
            "late_detections": len(late_detections),
            "early_alerts": len(early_alerts),
            "unmatched_gt_events": len(unmatched_gt),
            "unmatched_predicted_events": len(unmatched_preds),
            "positive_unmatched_alerts": total_positive_unmatched_alerts,
            "unmatched_positive_predictions": total_positive_unmatched_alerts,
            "recalled_positive_videos": len(recalled_positive_videos),
            "verified_positive_videos": len(verified_positive_video_ids),
            "positive_videos": len(positive_video_ids),
            "negative_videos": len(neg_ids),
            "negative_false_alarms": false_alarms,
        },
        "metrics": {
            "video_event_recall": video_recall,
            "split_video_event_recall": split_video_recall,
            "event_instance_recall": event_recall,
            "event_precision": precision,
            "event_f1": f1,
            "false_alarms_per_hour": far_per_hour,
            "far_status": far_status,
            "mean_detection_delay_sec": sum(delays) / len(delays) if delays else None,
            "negative_video_hours": neg_hrs,
            "temporal_tolerance_sec": tolerance_sec,
        },
        "targets": {
            "mean_delay_le_10s": (sum(delays) / len(delays) <= 10.0) if delays else False,
            "has_valid_negative_data": has_valid_negative,
            "overall_pass": bool(
                has_valid_negative
                and delays
                and (sum(delays) / len(delays) <= 10.0)
            ),
        },
        "matches": [
            {
                "ground_truth": asdict(ground_truth[gi]),
                "prediction": asdict(predictions[pi]),
                "delay_sec": max(0.0, predictions[pi].start_sec - ground_truth[gi].start_sec),
                "is_timely": True,
            }
            for gi, pi in matched_pairs
        ],
        "late_detections": late_detections,
        "early_alerts": early_alerts,
        "unmatched_ground_truth": unmatched_gt,
        "unmatched_predictions": unmatched_preds,
    }


def grid_search(
    frames: Sequence[FrameDetection],
    ground_truth: Sequence[Event],
    positive_video_ids: set[str],
    negative_video_ids: set[str] | None,
    negative_hours: float | None,
    confidence_thresholds: Sequence[float],
    k_values: Sequence[int],
    n_values: Sequence[int],
    tolerance_sec: float = 10.0,
    feature_thresholds: Sequence[float] = (0.0,),
    cooldown_values: Sequence[float] = (0.0,),
    max_event_gap_sec: float = 2.0,
    allow_missing_negative: bool = False,
) -> list[dict[str, Any]]:
    """Quét lưới tham số (Confidence, K, N, Feature Threshold, Cooldown) để tìm điểm vận hành tối ưu.

    Một ứng viên (candidate) chỉ được đánh dấu `feasible = True` khi có dữ liệu âm tính hợp lệ
    và thỏa mãn đồng thời cả 3 ràng buộc của Đề cương NCKH (MT3):
    1. Giảm FAR >= 50% so với Frame-level baseline.
    2. Video-level Event Recall giảm không quá 5 điểm phần trăm (<= 0.05).
    3. Mean Detection Delay <= 10.0 giây.
    """
    baselines_by_conf: dict[float, dict[str, Any]] = {}
    for conf in sorted(set(confidence_thresholds)):
        b_events, _ = frame_detections_to_events(
            frames,
            confidence_threshold=conf,
            window_n=1,
            confirm_k=1,
            max_event_gap_sec=max_event_gap_sec,
        )
        baselines_by_conf[conf] = evaluate(
            ground_truth,
            b_events,
            positive_video_ids,
            negative_video_ids,
            negative_hours,
            tolerance_sec,
            allow_missing_negative=allow_missing_negative,
        )

    results: list[dict[str, Any]] = []
    for confidence, n, k, feat_thr, cooldown in itertools.product(
        confidence_thresholds, n_values, k_values, feature_thresholds, cooldown_values
    ):
        if n < 1 or k < 1 or k > n:
            continue
        baseline = baselines_by_conf[confidence]
        baseline_far = baseline["metrics"]["false_alarms_per_hour"]
        baseline_recall = baseline["metrics"]["video_event_recall"]

        _, temporal_events = frame_detections_to_events(
            frames,
            confidence_threshold=confidence,
            window_n=n,
            confirm_k=k,
            max_event_gap_sec=max_event_gap_sec,
            feature_threshold=feat_thr,
            cooldown_sec=cooldown,
        )
        report = evaluate(
            ground_truth,
            temporal_events,
            positive_video_ids,
            negative_video_ids,
            negative_hours,
            tolerance_sec,
            allow_missing_negative=allow_missing_negative,
        )
        metrics = report["metrics"]
        cand_far = metrics["false_alarms_per_hour"]
        recall_drop = baseline_recall - metrics["video_event_recall"]
        delay = metrics["mean_detection_delay_sec"]

        if baseline_far is not None and cand_far is not None:
            far_reduction: float | None = (
                (baseline_far - cand_far) / baseline_far
                if baseline_far > 0 else 0.0
            )
            feasible = (
                far_reduction >= 0.50
                and recall_drop <= 0.05
                and delay is not None
                and delay <= 10.0
            )
            status_note = "FEASIBLE" if feasible else "INFEASIBLE_METRICS"
        else:
            far_reduction = None
            feasible = False
            status_note = "BLOCKED_BY_NEGATIVE_DATA"

        results.append({
            "confidence_threshold": confidence,
            "k": k,
            "n": n,
            "feature_threshold": feat_thr,
            "cooldown_sec": cooldown,
            "baseline_video_event_recall": baseline_recall,
            "baseline_far_per_hour": baseline_far,
            "metrics": metrics,
            "counts": report["counts"],
            "far_reduction_vs_frame_baseline": far_reduction,
            "video_event_recall_drop": recall_drop,
            "feasible": feasible,
            "status": status_note,
        })

    return sorted(
        results,
        key=lambda item: (
            not item["feasible"],
            item["metrics"]["false_alarms_per_hour"]
            if item["metrics"]["false_alarms_per_hour"] is not None else float("inf"),
            -item["metrics"]["video_event_recall"],
            item["metrics"]["mean_detection_delay_sec"]
            if item["metrics"]["mean_detection_delay_sec"] is not None else float("inf"),
        ),
    )


def select_best_operating_point(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Chọn điểm vận hành tối ưu từ kết quả grid search; báo NO_FEASIBLE_OPERATING_POINT nếu không đạt."""
    feasible_list = [c for c in candidates if c.get("feasible") is True]
    if feasible_list:
        best = feasible_list[0]
        return {
            "status": "FEASIBLE_OPERATING_POINT_FOUND",
            "feasible": True,
            "selected_operating_point": best,
            "reason": (
                f"Đạt cả 3 tiêu chí MT3: FAR giảm {best['far_reduction_vs_frame_baseline']*100:.1f}%, "
                f"Recall drop {best['video_event_recall_drop']*100:.2f} điểm %, "
                f"Mean delay {best['metrics']['mean_detection_delay_sec']:.2f}s <= 10s."
            ),
        }

    # Không có ứng viên khả thi (hoặc bị chặn bởi dữ liệu âm tính)
    blocked_by_neg = any(c.get("status") == "BLOCKED_BY_NEGATIVE_DATA" for c in candidates)
    # Chọn ứng viên có Recall cao nhất (drop <= 0.05), Delay <= 10s và ít positive_unmatched_alerts nhất làm cấu hình tạm thời
    positive_valid = [
        c for c in candidates
        if c.get("video_event_recall_drop", 1.0) <= 0.05
        and c["metrics"].get("mean_detection_delay_sec") is not None
        and c["metrics"]["mean_detection_delay_sec"] <= 10.0
    ]
    positive_valid.sort(
        key=lambda c: (
            -c["metrics"]["video_event_recall"],
            c["counts"].get("positive_unmatched_alerts", 0),
            -c["metrics"].get("event_f1", 0.0),
            c["metrics"]["mean_detection_delay_sec"],
        )
    )
    provisional = positive_valid[0] if positive_valid else (candidates[0] if candidates else None)
    reason_msg = (
        "NO_FEASIBLE_OPERATING_POINT (BLOCKED_BY_NEGATIVE_DATA: Chưa có >= 5 giờ video âm tính độc lập trên ổ đĩa "
        "để xác minh điều kiện giảm >= 50% FAR/giờ. Đã chọn cấu hình provisional tối ưu Recall và Delay trên tập Validation dương tính)."
        if blocked_by_neg
        else "NO_FEASIBLE_OPERATING_POINT (Không có tổ hợp tham số nào đồng thời đạt cả 3 ngưỡng FAR reduction >= 50%, Recall drop <= 5%, Delay <= 10s)."
    )
    return {
        "status": "NO_FEASIBLE_OPERATING_POINT",
        "feasible": False,
        "blocked_by_negative_data": blocked_by_neg,
        "selected_operating_point": None,
        "provisional_candidate_on_positive_val": provisional,
        "reason": reason_msg,
    }


def export_json(payload: Any, path: Path) -> None:
    """Xuất dữ liệu bất kỳ ra tệp JSON chuẩn UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_grid_search_csv(candidates: Sequence[dict[str, Any]], path: Path) -> None:
    """Xuất bảng kết quả quét tham số Grid Search ra tệp CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "confidence_threshold",
        "k",
        "n",
        "feature_threshold",
        "cooldown_sec",
        "baseline_video_event_recall",
        "baseline_far_per_hour",
        "video_event_recall",
        "event_instance_recall",
        "event_precision",
        "event_f1",
        "mean_detection_delay_sec",
        "false_alarms_per_hour",
        "far_reduction_vs_frame_baseline",
        "video_event_recall_drop",
        "positive_unmatched_alerts",
        "feasible",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            m = c.get("metrics", {})
            cnt = c.get("counts", {})
            writer.writerow({
                "confidence_threshold": c.get("confidence_threshold"),
                "k": c.get("k"),
                "n": c.get("n"),
                "feature_threshold": c.get("feature_threshold", 0.0),
                "cooldown_sec": c.get("cooldown_sec", 0.0),
                "baseline_video_event_recall": (
                    round(c["baseline_video_event_recall"], 4)
                    if c.get("baseline_video_event_recall") is not None else "N/A"
                ),
                "baseline_far_per_hour": (
                    round(c["baseline_far_per_hour"], 4)
                    if c.get("baseline_far_per_hour") is not None else "UNAVAILABLE"
                ),
                "video_event_recall": round(m["video_event_recall"], 4) if m.get("video_event_recall") is not None else "N/A",
                "event_instance_recall": round(m["event_instance_recall"], 4) if m.get("event_instance_recall") is not None else "N/A",
                "event_precision": round(m["event_precision"], 4) if m.get("event_precision") is not None else "N/A",
                "event_f1": round(m["event_f1"], 4) if m.get("event_f1") is not None else "N/A",
                "mean_detection_delay_sec": round(m["mean_detection_delay_sec"], 4) if m.get("mean_detection_delay_sec") is not None else "N/A",
                "false_alarms_per_hour": round(m["false_alarms_per_hour"], 4) if m.get("false_alarms_per_hour") is not None else "UNAVAILABLE",
                "far_reduction_vs_frame_baseline": (
                    round(c["far_reduction_vs_frame_baseline"], 4)
                    if c.get("far_reduction_vs_frame_baseline") is not None else "UNAVAILABLE"
                ),
                "video_event_recall_drop": round(c.get("video_event_recall_drop", 0.0), 4),
                "positive_unmatched_alerts": cnt.get("positive_unmatched_alerts", 0),
                "feasible": c.get("feasible", False),
                "status": c.get("status", ""),
            })


def export_events_csv(events: Sequence[Event], path: Path) -> None:
    """Xuất danh sách các sự kiện Event ra tệp CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_id", "camera_id", "label", "start_sec", "end_sec", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow(asdict(ev))


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy đánh giá sự kiện và quét lưới tham số từ dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--positive-video-ids", required=True, type=Path)
    parser.add_argument("--negative-video-ids", required=False, type=Path, default=None)
    parser.add_argument("--negative-hours", required=False, type=float, default=None)
    parser.add_argument("--allow-missing-negative", action="store_true")
    parser.add_argument("--confidence", nargs="+", type=float, default=[0.25, 0.35, 0.45])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--n-values", nargs="+", type=int, default=[5, 8, 10, 12])
    parser.add_argument("--feature-thresholds", nargs="+", type=float, default=[0.0])
    parser.add_argument("--cooldown-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--tolerance-sec", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    read_ids = lambda path: {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    gt = load_events(args.ground_truth)
    frames = load_frame_detections(args.frames)
    pos_ids = read_ids(args.positive_video_ids)
    neg_ids = read_ids(args.negative_video_ids) if (args.negative_video_ids and args.negative_video_ids.exists()) else set()

    results = grid_search(
        frames=frames,
        ground_truth=gt,
        positive_video_ids=pos_ids,
        negative_video_ids=neg_ids,
        negative_hours=args.negative_hours,
        confidence_thresholds=args.confidence,
        k_values=args.k_values,
        n_values=args.n_values,
        tolerance_sec=args.tolerance_sec,
        feature_thresholds=args.feature_thresholds,
        cooldown_values=args.cooldown_values,
        allow_missing_negative=args.allow_missing_negative,
    )
    selection = select_best_operating_point(results)
    payload = {
        "grid_search": results,
        "operating_point_selection": selection,
        "best_candidate": selection["selected_operating_point"],
    }
    export_json(payload, args.output)
    if args.csv_output:
        export_grid_search_csv(results, args.csv_output)
    LOGGER.info("Saved %d grid candidates to %s (status=%s)", len(results), args.output, selection["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
