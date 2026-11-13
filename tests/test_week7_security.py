"""Security unit tests for camera source validation and WebSocket tickets."""
from types import SimpleNamespace
from pathlib import Path
import threading
import time

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from src.config import PROJECT_ROOT, mask_rtsp_credentials
from src.week7_api import (
    _consume_ticket,
    _issue_ticket,
    _origin_is_allowed,
    _validate_source,
    _authorized,
)


def _ticket_app():
    fake = SimpleNamespace(state=SimpleNamespace(ws_tickets={}, ws_ticket_lock=threading.Lock()))
    return fake


def test_short_lived_websocket_ticket_is_single_use_and_reconnect_requires_new_ticket():
    fake = _ticket_app()
    ticket = _issue_ticket(fake)
    assert _consume_ticket(fake, ticket)
    assert not _consume_ticket(fake, ticket)
    reconnect_ticket = _issue_ticket(fake)
    assert reconnect_ticket != ticket
    assert _consume_ticket(fake, reconnect_ticket)


def test_websocket_ticket_expiration_and_required_authentication():
    fake = _ticket_app()
    ticket = _issue_ticket(fake)
    assert not _consume_ticket(fake, ticket, now=time.monotonic() + 100)
    assert not _consume_ticket(fake, "missing-ticket")


def test_websocket_origin_is_exact_allowlist():
    allowed = {"https://monitor.example", "http://localhost:8501"}
    assert _origin_is_allowed("https://monitor.example", allowed)
    assert not _origin_is_allowed("https://evil.example", allowed)
    assert not _origin_is_allowed("", allowed)


def test_http_management_authentication_requires_the_configured_bearer_token(monkeypatch):
    monkeypatch.setenv("FIREWATCH_API_TOKEN", "test-bearer-token")
    assert _authorized("Bearer test-bearer-token") is None
    with pytest.raises(HTTPException):
        _authorized(None)
    with pytest.raises(HTTPException):
        _authorized("Bearer wrong-token")


def test_source_validation_rejects_path_traversal_and_unapproved_rtsp(monkeypatch):
    outside = Path(PROJECT_ROOT).parent / "outside.mp4"
    with pytest.raises(HTTPException):
        _validate_source("VIDEO_TEST", str(outside), 0)
    monkeypatch.setenv("FIREWATCH_RTSP_HOSTS", "camera.internal")
    with pytest.raises(HTTPException):
        _validate_source("IP_CAMERA", "rtsp://169.254.169.254/latest/meta-data", 0)


def test_all_rtsp_user_info_is_masked():
    masked = mask_rtsp_credentials("rtsps://operator:private@camera.internal/live")
    assert "operator" not in masked
    assert "private" not in masked
    assert masked == "rtsps://***@camera.internal/live"

