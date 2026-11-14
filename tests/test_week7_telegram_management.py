"""Telegram management tests use fake Telegram responses and isolated SQLite."""
import json
import sqlite3
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest
from fastapi import HTTPException

from src.telegram_alerts import TelegramAlertDispatcher, TelegramTestError, send_test_message
from src.week7_api import TelegramTestInput, telegram_test, telegram_status

TOKEN = "123456:abcdefghijklmnopqrstuvwxyzABCDE"


def _db(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE alert_logs(id INTEGER PRIMARY KEY, camera_name TEXT, label TEXT)")


def _response(body):
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps(body).encode()
    return Response()


def test_missing_configuration_and_invalid_chat_id_fail_before_network(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    db = tmp_path / "missing.db"
    _db(db)
    dispatcher = TelegramAlertDispatcher(str(db))
    manager = SimpleNamespace(telegram_alerts=dispatcher)
    try:
        with pytest.raises(HTTPException) as missing:
            telegram_test(TelegramTestInput(), manager=manager, _=None)
        assert missing.value.detail["code"] == "MISSING_TOKEN"
        with pytest.raises(HTTPException) as invalid:
            telegram_test(TelegramTestInput(token=TOKEN, chat_id="bad value"), manager=manager, _=None)
        assert invalid.value.detail["code"] == "INVALID_CHAT_ID"
        oversized_secret = TOKEN + ("x" * 300)
        with pytest.raises(HTTPException) as oversized:
            telegram_test(TelegramTestInput(token=oversized_secret, chat_id="-123456"), manager=manager, _=None)
        assert oversized.value.detail["code"] == "INVALID_TOKEN_FORMAT"
        assert oversized_secret not in str(oversized.value.detail)
        with pytest.raises(HTTPException) as malformed:
            telegram_test({"token": [TOKEN], "chat_id": "-123456"}, manager=manager, _=None)
        assert malformed.value.detail["code"] == "INVALID_TOKEN_FORMAT"
        assert TOKEN not in str(malformed.value.detail)
    finally:
        dispatcher.close()


def test_test_message_is_separate_from_fire_events_and_response_hides_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    db = tmp_path / "test.db"
    _db(db)
    dispatcher = TelegramAlertDispatcher(str(db))
    manager = SimpleNamespace(telegram_alerts=dispatcher)
    seen = []
    monkeypatch.setattr("src.week7_api.send_test_message",
                        lambda token, chat_id, timeout: (seen.append((token, chat_id, timeout)) or "2026-11-13T10:00:00.123456+00:00"))
    monkeypatch.setattr("src.week7_api._telegram_last_test_at", 0.0)
    try:
        result = telegram_test(TelegramTestInput(token=TOKEN, chat_id="-123456"), manager=manager, _=None)
        assert result["status"] == "DELIVERED"
        assert TOKEN not in str(result)
        assert seen == [(TOKEN, "-123456", dispatcher.timeout)]
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM alert_logs").fetchone()[0] == 0
            assert conn.execute("SELECT status FROM telegram_test_deliveries").fetchone() == ("DELIVERED",)
            database_dump = "\n".join(conn.iterdump())
            assert TOKEN not in database_dump and "-123456" not in database_dump
    finally:
        dispatcher.close()


def test_test_message_helper_verifies_bot_and_sends_labeled_message(monkeypatch):
    calls = []
    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if isinstance(request, str):
            return _response({"ok": True, "result": {"id": 123, "is_bot": True}})
        body = parse_qs(request.data.decode())['text'][0].lower()
        assert "kiểm tra kết nối" in body and "không phải cảnh báo cháy thật" in body
        return _response({"ok": True, "result": {"message_id": 7}})
    monkeypatch.setattr("src.telegram_alerts.urlopen", fake_urlopen)
    assert send_test_message(TOKEN, "@firewatch_test", timeout=2.0)
    assert len(calls) == 2 and all(timeout == 2.0 for _, timeout in calls)


def test_unexpected_test_request_error_is_sanitized(monkeypatch):
    monkeypatch.setattr("src.telegram_alerts.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(TOKEN)))
    with pytest.raises(TelegramTestError) as failure:
        send_test_message(TOKEN, "-123456", timeout=1.0)
    assert failure.value.category == "NETWORK_ERROR"
    assert TOKEN not in str(failure.value)


def test_enable_flag_persists_without_storing_credentials(tmp_path, monkeypatch):
    db = tmp_path / "settings.db"
    _db(db)
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-123456")
    dispatcher = TelegramAlertDispatcher(str(db))
    dispatcher.service_credentials_tested = True
    dispatcher.set_enabled(True)
    assert dispatcher.configured
    dispatcher.set_enabled(False)
    dispatcher.close()
    restarted = TelegramAlertDispatcher(str(db))
    try:
        assert restarted.enabled is False
        with sqlite3.connect(db) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(telegram_settings)")}
            enabled = conn.execute("SELECT enabled FROM telegram_settings WHERE setting_id=1").fetchone()
        assert "token" not in columns and "chat_id" not in columns
        assert enabled == (0,)
    finally:
        restarted.close()


def test_enable_requires_service_environment_credentials(tmp_path, monkeypatch):
    db = tmp_path / "no-creds.db"
    _db(db)
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    dispatcher = TelegramAlertDispatcher(str(db))
    try:
        with pytest.raises(ValueError, match="service environment"):
            dispatcher.set_enabled(True)
    finally:
        dispatcher.close()


def test_enable_requires_successful_service_credential_test(tmp_path, monkeypatch):
    db = tmp_path / "not-tested.db"
    _db(db)
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-123456")
    dispatcher = TelegramAlertDispatcher(str(db))
    try:
        with pytest.raises(ValueError, match="Run Test Connection"):
            dispatcher.set_enabled(True)
    finally:
        dispatcher.close()


def test_successful_service_credential_test_authorizes_enable_only_in_current_process(tmp_path, monkeypatch):
    db = tmp_path / "service-test.db"
    _db(db)
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-123456")
    dispatcher = TelegramAlertDispatcher(str(db))
    manager = SimpleNamespace(telegram_alerts=dispatcher)
    monkeypatch.setattr("src.week7_api.send_test_message", lambda *args, **kwargs: "2026-11-13T10:00:00+00:00")
    monkeypatch.setattr("src.week7_api._telegram_last_test_at", 0.0)
    try:
        telegram_test(TelegramTestInput(), manager=manager, _=None)
        assert dispatcher.service_credentials_tested is True
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT used_service_credentials FROM telegram_test_deliveries").fetchone() == (1,)
        dispatcher.set_enabled(True)
        dispatcher.set_enabled(False)
    finally:
        dispatcher.close()


def test_connection_test_is_rate_limited(tmp_path, monkeypatch):
    db = tmp_path / "rate-limit.db"
    _db(db)
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    dispatcher = TelegramAlertDispatcher(str(db))
    manager = SimpleNamespace(telegram_alerts=dispatcher)
    monkeypatch.setattr("src.week7_api.send_test_message", lambda *args, **kwargs: "2026-11-13T10:00:00+00:00")
    monkeypatch.setattr("src.week7_api._telegram_last_test_at", 0.0)
    try:
        payload = TelegramTestInput(token=TOKEN, chat_id="-123456")
        telegram_test(payload, manager=manager, _=None)
        with pytest.raises(HTTPException) as limited:
            telegram_test(payload, manager=manager, _=None)
        assert limited.value.status_code == 429
        assert limited.value.detail["code"] == "TEST_RATE_LIMITED"
        assert TOKEN not in str(limited.value.detail)
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM telegram_test_deliveries").fetchone()[0] == 1
    finally:
        dispatcher.close()


def test_status_never_returns_secret_values(tmp_path, monkeypatch):
    db = tmp_path / "status.db"
    _db(db)
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-123456")
    dispatcher = TelegramAlertDispatcher(str(db))
    manager = SimpleNamespace(telegram_alerts=dispatcher)
    try:
        response = telegram_status(manager=manager, _=None)
        assert response["state"] == "DEGRADED"
        assert TOKEN not in str(response)
        assert "-123456" not in str(response)
    finally:
        dispatcher.close()
