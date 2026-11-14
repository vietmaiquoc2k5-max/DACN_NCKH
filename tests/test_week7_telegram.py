"""Mocked tests for durable Telegram delivery; no external request is made."""
import json
import sqlite3
import threading
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs

import cv2
import numpy as np
import pytest

from src.telegram_alerts import TelegramAlertDispatcher


def _database(path):
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE alert_logs(
            id INTEGER PRIMARY KEY, timestamp TEXT, camera_name TEXT, label TEXT,
            confidence REAL, alert_level TEXT, image_path TEXT, status TEXT)""")


def _event(path, event_id=1):
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    image_path = path.parent / f"snapshot-{event_id}.jpg"
    image_path.write_bytes(encoded.tobytes())
    event = {
        "id": event_id,
        "timestamp": "2026-11-13T10:00:00+00:00",
        "event_confirmed_at": "2026-11-13T10:00:00.123456+00:00",
        "camera_name": f"CAM-{event_id:02d}",
        "camera_display_name": "Test camera",
        "label": "FIRE",
        "confidence": 0.91,
        "alert_level": "CONFIRMED",
        "image_path": str(image_path),
        "status": "UNREAD",
    }
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO alert_logs VALUES(?,?,?,?,?,?,?,?)",
                     (event_id, event["timestamp"], event["camera_name"], "FIRE", 0.91, "CONFIRMED", str(image_path), "UNREAD"))
    return event


def _telegram_env(monkeypatch, **extra):
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "unit-test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "unit-test-chat")
    for key, value in extra.items():
        monkeypatch.setenv(key, str(value))


def _rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT status,attempt_count,error_category,error_message,event_confirmed_at,queued_at,attempt_started_at,telegram_acknowledged_at FROM alert_deliveries ORDER BY event_id").fetchall()


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def _wait_for(predicate, timeout=3.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_delivery_persists_telegram_ack_and_suppresses_duplicates(tmp_path, monkeypatch):
    _telegram_env(monkeypatch)
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)
    ack = threading.Event()
    monkeypatch.setattr("src.telegram_alerts.urlopen", lambda *_a, **_k: (ack.set() or _Response({"ok": True, "result": {"message_id": 8}})))
    dispatcher = TelegramAlertDispatcher(str(db_path))
    try:
        assert dispatcher.enqueue(event)
        assert not dispatcher.enqueue(event)
        assert ack.wait(2)
        assert _wait_for(lambda: _rows(db_path)[0][0] == "DELIVERED")
        row = _rows(db_path)[0]
        assert row[1] == 1
        assert row[4] == event["event_confirmed_at"]
        assert row[5] and row[6] and row[7]
    finally:
        dispatcher.close()


def test_timeout_is_ambiguous_and_not_retried_by_default(tmp_path, monkeypatch):
    _telegram_env(monkeypatch)
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)
    calls = []

    def timeout(*_args, **_kwargs):
        calls.append(1)
        raise TimeoutError("request timed out")

    monkeypatch.setattr("src.telegram_alerts.urlopen", timeout)
    dispatcher = TelegramAlertDispatcher(str(db_path))
    try:
        dispatcher.enqueue(event)
        assert _wait_for(lambda: bool(_rows(db_path)) and _rows(db_path)[0][0] == "AMBIGUOUS")
        assert len(calls) == 1
        assert "unit-test-token" not in (_rows(db_path)[0][3] or "")
    finally:
        dispatcher.close()


def test_explicit_http_5xx_retries_then_delivers(tmp_path, monkeypatch):
    _telegram_env(monkeypatch, TELEGRAM_RETRIES=2)
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)
    calls = []

    def flaky(request, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 503, "unavailable", {}, None)
        return _Response({"ok": True, "result": {"message_id": 9}})

    monkeypatch.setattr("src.telegram_alerts.urlopen", flaky)
    dispatcher = TelegramAlertDispatcher(str(db_path))
    try:
        dispatcher.enqueue(event)
        assert _wait_for(lambda: bool(_rows(db_path)) and _rows(db_path)[0][0] == "DELIVERED", timeout=4)
        assert len(calls) == 2
        assert _rows(db_path)[0][1] == 2
    finally:
        dispatcher.close()


def test_invalid_credentials_are_terminal_and_missing_snapshot_falls_back_to_text(tmp_path, monkeypatch):
    _telegram_env(monkeypatch)
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)

    def rejected(request, **kwargs):
        raise HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("src.telegram_alerts.urlopen", rejected)
    dispatcher = TelegramAlertDispatcher(str(db_path))
    try:
        assert dispatcher.enqueue(event)
        assert _wait_for(lambda: bool(_rows(db_path)) and _rows(db_path)[0][0] == "FAILED")
        assert _rows(db_path)[0][2] == "INVALID_CREDENTIALS"
    finally:
        dispatcher.close()

    bad = dict(event, id=2, camera_name="CAM-02", confidence=0.8,
               image_path=str(tmp_path / "missing.jpg"))
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO alert_logs VALUES(?,?,?,?,?,?,?,?)", (2, "now", "CAM-02", "FIRE", .8, "CONFIRMED", bad["image_path"], "UNREAD"))

    requests = []
    def accept_text_fallback(request, **kwargs):
        requests.append(request)
        return _Response({"ok": True, "result": {"message_id": 10}})

    monkeypatch.setattr("src.telegram_alerts.urlopen", accept_text_fallback)
    dispatcher2 = TelegramAlertDispatcher(str(db_path))
    try:
        assert dispatcher2.enqueue(bad)
        assert _wait_for(lambda: _rows(db_path)[-1][0] == "DELIVERED")
        assert len(requests) == 1
        assert requests[0].full_url.endswith("/sendMessage")
        message = parse_qs(requests[0].data.decode("utf-8"))["text"][0]
        assert "event #2" in message and "CAM-02" in message and "FIRE" in message
    finally:
        dispatcher2.close()


def test_telegram_api_rejection_is_terminal(tmp_path, monkeypatch):
    _telegram_env(monkeypatch)
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)
    monkeypatch.setattr("src.telegram_alerts.urlopen",
                        lambda *_a, **_k: _Response({"ok": False, "error_code": 400}))
    dispatcher = TelegramAlertDispatcher(str(db_path))
    try:
        assert dispatcher.enqueue(event)
        assert _wait_for(lambda: bool(_rows(db_path)) and _rows(db_path)[0][0] == "FAILED")
        assert _rows(db_path)[0][2] == "TELEGRAM_REJECTED"
    finally:
        dispatcher.close()


def test_queued_delivery_recovers_after_dispatcher_restart(tmp_path, monkeypatch):
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)
    first = TelegramAlertDispatcher(str(db_path))
    assert not first.configured
    queued_at = "2026-11-13T10:00:01.000000+00:00"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""INSERT INTO alert_deliveries(delivery_id,event_id,channel,status,queued_at,
            event_confirmed_at,payload_json) VALUES('restart-id',1,'telegram','QUEUED',?,?,?)""",
            (queued_at, event["event_confirmed_at"], json.dumps(event)))
    first.close()
    _telegram_env(monkeypatch)
    ack = threading.Event()
    monkeypatch.setattr("src.telegram_alerts.urlopen", lambda *_a, **_k: (ack.set() or _Response({"ok": True, "result": {"message_id": 12}})))
    restarted = TelegramAlertDispatcher(str(db_path))
    try:
        assert ack.wait(2)
        assert _wait_for(lambda: any(row[0] == "DELIVERED" for row in _rows(db_path)))
    finally:
        restarted.close()


def test_interrupted_sending_is_marked_ambiguous_at_startup(tmp_path, monkeypatch):
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)
    first = TelegramAlertDispatcher(str(db_path))
    with sqlite3.connect(db_path) as conn:
        conn.execute("""INSERT INTO alert_deliveries(delivery_id,event_id,channel,status,queued_at,
            event_confirmed_at,payload_json,attempt_count) VALUES('interrupted',1,'telegram','SENDING',?,?,?,1)""",
            (event["timestamp"], event["event_confirmed_at"], json.dumps(event)))
    first.close()
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    restarted = TelegramAlertDispatcher(str(db_path))
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT status,error_category FROM alert_deliveries WHERE delivery_id='interrupted'").fetchone()
        assert row == ("AMBIGUOUS", "PROCESS_INTERRUPTED")
    finally:
        restarted.close()


def test_full_persistent_queue_records_failure_for_operator(tmp_path, monkeypatch):
    _telegram_env(monkeypatch, TELEGRAM_QUEUE_MAX=1)
    db_path = tmp_path / "events.db"
    _database(db_path)
    first_event = _event(db_path, 1)
    second_event = _event(db_path, 2)
    request_started = threading.Event()
    release = threading.Event()

    def delayed(_request, **_kwargs):
        request_started.set()
        release.wait(2)
        return _Response({"ok": True, "result": {"message_id": 20}})

    monkeypatch.setattr("src.telegram_alerts.urlopen", delayed)
    dispatcher = TelegramAlertDispatcher(str(db_path))
    try:
        assert dispatcher.enqueue(first_event)
        assert request_started.wait(2)
        assert not dispatcher.enqueue(second_event)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT status,error_category FROM alert_deliveries WHERE event_id=2").fetchone()
        assert row == ("FAILED", "QUEUE_FULL")
    finally:
        release.set()
        dispatcher.close()


def test_shutdown_during_active_delivery_returns_timeout_and_finishes_after_release(tmp_path, monkeypatch):
    _telegram_env(monkeypatch)
    db_path = tmp_path / "events.db"
    _database(db_path)
    event = _event(db_path)
    request_started = threading.Event()
    release = threading.Event()

    def delayed(_request, **_kwargs):
        request_started.set()
        release.wait(2)
        return _Response({"ok": True, "result": {"message_id": 21}})

    monkeypatch.setattr("src.telegram_alerts.urlopen", delayed)
    dispatcher = TelegramAlertDispatcher(str(db_path))
    assert dispatcher.enqueue(event)
    assert request_started.wait(2)
    assert dispatcher.close(timeout=0.01) is False
    release.set()
    assert _wait_for(lambda: not dispatcher.thread.is_alive())
    assert _rows(db_path)[0][0] == "DELIVERED"

