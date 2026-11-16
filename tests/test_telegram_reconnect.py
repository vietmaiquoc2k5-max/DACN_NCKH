from pathlib import Path
from types import SimpleNamespace

from src.telegram_alerts import TelegramAlertDispatcher
from src.week7_api import TelegramTestInput, telegram_connect


TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTuv_1234567890"
CHAT_ID = "-100123456"


def test_connect_tests_credentials_then_enables_runtime_delivery(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    db = tmp_path / "telegram.db"
    dispatcher = TelegramAlertDispatcher(str(db))
    manager = SimpleNamespace(telegram_alerts=dispatcher)
    monkeypatch.setattr("src.week7_api._telegram_last_test_at", 0.0)
    monkeypatch.setattr("src.week7_api.send_test_message", lambda *_args, **_kwargs: "2026-11-16T00:00:00+00:00")
    result = telegram_connect(TelegramTestInput(token=TOKEN, chat_id=CHAT_ID), manager=manager, _=None, __=None)
    assert result["enabled"] is True
    assert dispatcher.enabled is True
    assert dispatcher.service_credentials_tested is True
    assert dispatcher.token == TOKEN and dispatcher.chat_id == CHAT_ID
    with dispatcher._connection() as conn:
        assert conn.execute("SELECT status FROM telegram_test_deliveries").fetchone() == ("DELIVERED",)
        assert conn.execute("SELECT enabled FROM telegram_settings WHERE setting_id=1").fetchone() == (1,)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(telegram_test_deliveries)")}
        assert "token" not in columns and "chat_id" not in columns
    dispatcher.close()


def test_confirmed_camera_event_path_enqueues_telegram_and_respects_camera_setting():
    source = Path("src/camera_manager.py").read_text(encoding="utf-8")
    assert "if self.telegram_alerts is not None and self.config.telegram_enabled:" in source
    assert "self.telegram_alerts.enqueue(saved_event)" in source
    assert "self.config.telegram_enabled" in source


def test_dashboard_has_explicit_telegram_connect_form():
    source = Path("web_app.py").read_text(encoding="utf-8")
    assert '"Telegram"' in source
    assert 'telegram_connect_form' in source
    assert '"/api/telegram/connect"' in source
    assert 'type="password"' in source
