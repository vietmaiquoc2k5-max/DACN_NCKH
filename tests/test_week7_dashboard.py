"""Dashboard/backend lifecycle unit checks; these do not launch Streamlit."""
from unittest.mock import MagicMock

from src.camera_manager import MultiCameraManager


def _detector(monkeypatch):
    monkeypatch.setattr("src.camera_manager.ByteTrackAdapter", lambda **kwargs: MagicMock())
    detector = MagicMock()
    detector.get_model_info.return_value = {
        "loaded": True, "path": "mock", "device": "CPU",
        "names": {0: "SMOKE", 1: "FIRE"}, "error": "",
    }
    return detector


def test_connecting_selected_camera_does_not_stop_hidden_running_camera(tmp_path, monkeypatch):
    manager = MultiCameraManager(db_path=str(tmp_path / "dashboard.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    try:
        first = manager.add_camera("one", "1", "WEBCAM", webcam_index=1)
        second = manager.add_camera("two", "2", "WEBCAM", webcam_index=2)
        manager.workers[first].start = MagicMock()
        manager.workers[second].start = MagicMock()
        manager.workers[first].running = True
        manager.start_monitoring([second])
        manager.workers[second].start.assert_called_once()
        assert manager.workers[first].running is True
    finally:
        manager.close()


def test_worker_start_is_only_triggered_by_explicit_connect(tmp_path, monkeypatch):
    manager = MultiCameraManager(db_path=str(tmp_path / "dashboard.db"),
                                 auto_assign_videos=False, detector=_detector(monkeypatch))
    try:
        camera_id = manager.add_camera("one", "0", "WEBCAM", webcam_index=0, auto_start=False)
        assert manager.workers[camera_id].thread is None
        assert manager.workers[camera_id].running is False
    finally:
        manager.close()
