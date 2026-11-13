"""Focused API validation tests; no camera or external service is opened."""
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from src.config import PROJECT_ROOT
from src.week7_api import _validate_source
from src.telegram_alerts import TelegramAlertDispatcher


def test_webcam_source_is_numeric_and_matches_index():
    assert _validate_source("WEBCAM", "2", 2) == "2"
    with pytest.raises(HTTPException):
        _validate_source("WEBCAM", "3", 2)


def test_local_video_must_stay_inside_project():
    with pytest.raises(HTTPException):
        _validate_source("VIDEO_TEST", str(Path(PROJECT_ROOT).parent / "outside.mp4"), 0)


def test_rtsp_requires_allowlisted_host(monkeypatch):
    monkeypatch.setenv("FIREWATCH_RTSP_HOSTS", "mediamtx,192.168.1.50")
    assert _validate_source("IP_CAMERA", "rtsp://mediamtx:8554/cam1", 0) == "rtsp://mediamtx:8554/cam1"
    with pytest.raises(HTTPException):
        _validate_source("IP_CAMERA", "rtsp://user:secret@mediamtx:8554/cam1", 0)
    with pytest.raises(HTTPException):
        _validate_source("IP_CAMERA", "rtsp://127.0.0.1/admin", 0)


def test_rtsp_secret_can_be_referenced_from_environment(monkeypatch):
    monkeypatch.setenv("FIREWATCH_RTSP_HOSTS", "camera.internal")
    monkeypatch.setenv("CAMERA_RTSP_CAM01", "rtsp://operator:secret@camera.internal/live")
    assert _validate_source("IP_CAMERA", "env:CAMERA_RTSP_CAM01", 0) == "rtsp://operator:secret@camera.internal/live"


def test_telegram_delivery_requires_api_acknowledgment(tmp_path, monkeypatch):
    import json
    import sqlite3

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"ok": True, "result": {"message_id": 123}}).encode()

    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setattr("src.telegram_alerts.urlopen", lambda *_args, **_kwargs: FakeResponse())
    snapshot = tmp_path / "snapshot.jpg"
    snapshot.write_bytes(b"test image bytes")
    dispatcher = TelegramAlertDispatcher(str(tmp_path / "events.db"))
    try:
        event = {"id": 17, "image_path": str(snapshot), "camera_name": "CAM-01", "label": "FIRE", "confidence": 0.91}
        assert dispatcher.enqueue(event)
        assert not dispatcher.enqueue(event)
        dispatcher.work.join()
        with sqlite3.connect(dispatcher.db_path) as conn:
            row = conn.execute("SELECT status,confirmed_at FROM alert_deliveries WHERE event_id=17").fetchone()
        assert row[0] == "DELIVERED"
        assert row[1]
    finally:
        dispatcher.close()

