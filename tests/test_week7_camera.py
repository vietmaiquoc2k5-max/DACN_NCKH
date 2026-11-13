"""Unit tests for camera persistence and disconnected restoration."""
from unittest.mock import MagicMock

from src.camera_manager import MultiCameraManager


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
