"""Focused API validation tests; no camera or external service is opened."""
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from src.config import PROJECT_ROOT
from src.week7_api import _validate_source


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

