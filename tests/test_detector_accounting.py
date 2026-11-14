import threading

import numpy as np

from src.detector import InferenceStatus, SharedFireSmokeDetector


class _EmptyModel:
    def predict(self, _frame, **_kwargs):
        return []


def _detector():
    detector = SharedFireSmokeDetector.__new__(SharedFireSmokeDetector)
    detector.model_loaded = True
    detector.model = _EmptyModel()
    detector.infer_lock = threading.Lock()
    detector.conf_threshold = 0.25
    detector.imgsz = 32
    detector.device = "cpu"
    detector.use_half = False
    detector.last_infer_ms = 0.0
    return detector


def test_completed_inference_with_zero_detections_is_explicit():
    result = _detector().detect_frame_result(np.zeros((32, 32, 3), dtype=np.uint8), non_blocking=True)

    assert result.status is InferenceStatus.COMPLETED
    assert result.detections == []
    assert result.inference_ms >= 0.0


def test_busy_lock_is_distinct_and_legacy_api_keeps_none_sentinel():
    detector = _detector()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detector.infer_lock.acquire()
    try:
        result = detector.detect_frame_result(frame, non_blocking=True)
        legacy_result = detector.detect_frame(frame, non_blocking=True)
    finally:
        detector.infer_lock.release()

    assert result.status is InferenceStatus.BUSY
    assert result.detections is None
    assert legacy_result == (None, 0.0)


def test_unavailable_and_invalid_input_have_distinct_statuses():
    detector = _detector()
    detector.model_loaded = False
    unavailable = detector.detect_frame_result(np.zeros((32, 32, 3), dtype=np.uint8))

    detector.model_loaded = True
    invalid = detector.detect_frame_result(None)

    assert unavailable.status is InferenceStatus.UNAVAILABLE
    assert invalid.status is InferenceStatus.INVALID_INPUT

