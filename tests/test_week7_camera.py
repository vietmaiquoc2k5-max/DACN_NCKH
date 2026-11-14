"""Unit tests for camera persistence and disconnected restoration."""
from unittest.mock import MagicMock

import numpy as np

from src.camera_manager import MultiCameraManager
from src.detector import DetectionResult, InferenceStatus
from src.temporal_verifier import DetectionBox


def _detector(monkeypatch):
    monkeypatch.setattr("src.camera_manager.ByteTrackAdapter", lambda **kwargs: MagicMock())
    detector = MagicMock()
    detector.get_model_info.return_value = {
        "loaded": True,
        "path": "mock-checkpoint",
        "device": "CPU",
        "names": {0: "SMOKE", 1: "FIRE"},
        "error": "",
    }
    return detector


def _prime_successful_temporal_pass(worker, timestamp, monkeypatch):
    """Arrange one deterministic inference pass and capture temporal inputs."""
    from unittest.mock import MagicMock

    worker.running = True
    worker.status = "CONNECTED"
    worker.frame_count = 7
    worker.latest_raw_bgr = np.full((48, 64, 3), 17, dtype=np.uint8)
    worker.latest_capture_monotonic = timestamp
    worker.infer_event.set()
    worker.features_extractor = MagicMock()
    worker.tracker.update.return_value = [DetectionBox(
        cls_id=1, label="FIRE", confidence=0.8, xyxy=(2, 2, 20, 20),
        norm_xywh=(0.2, 0.2, 0.3, 0.3), track_id=5,
    )]
    result = {
        "valid_boxes": [], "alert_level": "SAFE", "primary_label": "",
        "max_confidence": 0.0, "active_hits": 0, "window_size": 8,
        "temporal_score": 0.0, "should_trigger_db_alert": False,
        "confirmed_track_ids": [],
    }
    worker.verifier = MagicMock()
    worker.verifier.update.return_value = result
    worker.verifier.update.side_effect = lambda *args, **kwargs: (
        setattr(worker, "running", False) or result
    )
    worker.detector.detect_frame_result.return_value = DetectionResult(
        InferenceStatus.COMPLETED, [], 5000.0
    )


def test_camera_settings_restore_with_stable_id_without_opening_camera(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWATCH_RTSP_HOSTS", "camera.internal")
    monkeypatch.setenv("CAMERA_RTSP_CAM01", "rtsps://operator:private-secret@camera.internal/live")
    database = str(tmp_path / "camera.db")

    first = MultiCameraManager(db_path=database, auto_assign_videos=False, detector=_detector(monkeypatch))
    cam_id = first.add_camera(
        name="North entrance",
        source="rtsps://operator:private-secret@camera.internal/live",
        source_type="IP_CAMERA",
        camera_id="CAM-10",
        source_reference="env:CAMERA_RTSP_CAM01",
        inference_fps=7,
        exclusion_zones=[(0.1, 0.2, 0.8, 0.9)],
        auto_start=False,
    )
    assert cam_id == "CAM-10"
    first.telegram_alerts.close()

    restored = MultiCameraManager(db_path=database, auto_assign_videos=False, detector=_detector(monkeypatch))
    worker = restored.workers["CAM-10"]
    assert worker.status == "DISCONNECTED"
    assert worker.running is False
    assert worker.thread is None
    assert worker.config.inference_fps == 7
    assert worker.config.source_reference == "env:CAMERA_RTSP_CAM01"
    assert worker.config.exclusion_zones == [(0.1, 0.2, 0.8, 0.9)]
    assert "private-secret" not in worker.get_snapshot()["masked_source"]
    assert "operator" not in worker.get_snapshot()["masked_source"]
    restored.telegram_alerts.close()


def test_missing_rtsp_environment_reference_is_reported_without_reassigning_id(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWATCH_RTSP_HOSTS", "camera.internal")
    monkeypatch.setenv("CAMERA_RTSP_NOT_SET", "rtsp://camera.internal/live")
    database = str(tmp_path / "camera.db")
    first = MultiCameraManager(db_path=database, auto_assign_videos=False, detector=_detector(monkeypatch))
    first.add_camera(
        name="Loading dock",
        source="rtsp://camera.internal/live",
        source_type="IP_CAMERA",
        camera_id="CAM-14",
        source_reference="env:CAMERA_RTSP_NOT_SET",
        auto_start=False,
    )
    first.telegram_alerts.close()
    monkeypatch.delenv("CAMERA_RTSP_NOT_SET")

    restored = MultiCameraManager(db_path=database, auto_assign_videos=False, detector=_detector(monkeypatch))
    assert "CAM-14" not in restored.workers
    assert restored.camera_restore_errors[0]["camera_id"] == "CAM-14"
    assert "CAM-14" not in restored.workers
    restored.telegram_alerts.close()




def test_camera_telegram_preference_persists_without_starting_worker(tmp_path, monkeypatch):
    database = str(tmp_path / "camera-alert-preference.db")
    first = MultiCameraManager(db_path=database, auto_assign_videos=False, detector=_detector(monkeypatch))
    camera_id = first.add_camera("Entry", "0", "WEBCAM", webcam_index=0, auto_start=False)
    first.set_camera_telegram_enabled(camera_id, False)
    assert first.workers[camera_id].running is False
    first.close()

    restored = MultiCameraManager(db_path=database, auto_assign_videos=False, detector=_detector(monkeypatch))
    try:
        assert restored.workers[camera_id].config.telegram_enabled is False
        assert restored.workers[camera_id].running is False
    finally:
        restored.close()


def test_camera_inference_skips_busy_shared_detector_instead_of_queuing(tmp_path, monkeypatch):
    manager = MultiCameraManager(db_path=str(tmp_path / "nonblocking-inference.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    worker = manager.workers["CAM-01"]
    worker.running = True
    worker.status = "CONNECTED"
    worker.frame_count = 1
    worker.latest_raw_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    worker.infer_event.set()

    def finish_inference(*_args, **_kwargs):
        worker.running = False
        return DetectionResult(InferenceStatus.BUSY, None, 0.0)

    worker.detector.detect_frame_result.side_effect = finish_inference
    try:
        worker._infer_loop()
        kwargs = worker.detector.detect_frame_result.call_args.kwargs
        assert kwargs["non_blocking"] is True
        accounting = worker.get_snapshot()["inference_accounting"]
        assert accounting["inference_attempts"] == 1
        assert accounting["detector_inference_calls"] == 1
        assert accounting["inference_skipped_busy"] == 1
        assert accounting["inference_completed"] == 0
        assert accounting["detector_output_detections"] == 0
    finally:
        manager.close()


def test_camera_counts_completed_empty_inference_separately_from_busy(tmp_path, monkeypatch):
    manager = MultiCameraManager(db_path=str(tmp_path / "completed-empty-inference.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    worker = manager.workers["CAM-01"]
    worker.running = True
    worker.status = "CONNECTED"
    worker.frame_count = 1
    worker.latest_raw_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    worker.infer_event.set()

    def finish_inference(*_args, **_kwargs):
        worker.running = False
        return DetectionResult(InferenceStatus.COMPLETED, [], 1.25)

    worker.detector.detect_frame_result.side_effect = finish_inference
    try:
        worker._infer_loop()
        accounting = worker.get_snapshot()["inference_accounting"]
        assert accounting["inference_attempts"] == 1
        assert accounting["inference_completed"] == 1
        assert accounting["inference_skipped_busy"] == 0
        assert accounting["detector_output_detections"] == 0
        assert accounting["inference_frames_processed"] == 0
    finally:
        manager.close()


def test_capture_timestamp_and_frame_are_propagated_to_temporal_verifier(tmp_path, monkeypatch):
    manager = MultiCameraManager(db_path=str(tmp_path / "capture-time.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    worker = manager.workers["CAM-01"]
    _prime_successful_temporal_pass(worker, 123.5, monkeypatch)
    try:
        worker._infer_loop()
        args, kwargs = worker.verifier.update.call_args
        assert kwargs["timestamp_sec"] == 123.5
        assert worker.detector.detect_frame_result.call_args.args[0][0, 0, 0] == 17
        assert worker.features_extractor.update.call_args.kwargs["timestamp_sec"] == 123.5
        assert worker.get_snapshot()["temporal_clock_semantics"].startswith("capture_monotonic")
    finally:
        manager.close()


def test_out_of_order_capture_timestamp_is_rejected_before_inference(tmp_path, monkeypatch):
    import threading

    manager = MultiCameraManager(db_path=str(tmp_path / "out-of-order.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    worker = manager.workers["CAM-01"]
    worker.running = True
    worker.status = "CONNECTED"
    worker.frame_count = 2
    worker.latest_raw_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    worker.latest_capture_monotonic = 9.0
    worker._last_temporal_timestamp = 10.0
    worker.infer_event.set()
    rejected = threading.Event()

    class StopAfterRejection:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *_args):
            self.lock.release()
            if worker.pipeline_counters["inference_skipped_out_of_order_timestamp"]:
                worker.running = False
                rejected.set()

    worker.lock = StopAfterRejection()
    try:
        worker._infer_loop()
        assert rejected.is_set()
        assert worker.detector.detect_frame_result.call_count == 0
        assert worker.get_snapshot()["inference_accounting"]["inference_skipped_out_of_order_timestamp"] == 1
    finally:
        worker.running = False
        manager.close()


def test_missing_capture_timestamp_uses_monotonic_fallback(tmp_path, monkeypatch):
    manager = MultiCameraManager(db_path=str(tmp_path / "missing-time.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    worker = manager.workers["CAM-01"]
    _prime_successful_temporal_pass(worker, 0.0, monkeypatch)
    try:
        worker._infer_loop()
        assert worker.verifier.update.call_args.kwargs["timestamp_sec"] > 0.0
        assert worker.get_snapshot()["inference_accounting"]["temporal_timestamp_fallbacks"] == 1
    finally:
        manager.close()


def test_camera_accounts_for_detector_exceptions_and_keeps_thread_loop_safe(tmp_path, monkeypatch):
    manager = MultiCameraManager(db_path=str(tmp_path / "detector-error-inference.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    worker = manager.workers["CAM-01"]
    worker.running = True
    worker.status = "CONNECTED"
    worker.frame_count = 1
    worker.latest_raw_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    worker.infer_event.set()

    def fail_inference(*_args, **_kwargs):
        worker.running = False
        raise RuntimeError("synthetic model error")

    worker.detector.detect_frame_result.side_effect = fail_inference
    try:
        worker._infer_loop()
        accounting = worker.get_snapshot()["inference_accounting"]
        assert accounting["inference_attempts"] == 1
        assert accounting["detector_inference_calls"] == 1
        assert accounting["inference_errors"] == 1
        assert accounting["inference_completed"] == 0
    finally:
        manager.close()
