"""SQLite integrity and operator-state tests."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import numpy as np
import pytest

from alert_manager import AlertManager
from src.database import EventDatabase


def _database(tmp_path):
    manager = AlertManager(
        sound_path=str(tmp_path / "alarm.wav"),
        db_path=str(tmp_path / "events.db"),
        alerts_dir=str(tmp_path / "snapshots"),
    )
    return manager, EventDatabase(alert_manager=manager)


def test_event_insert_acknowledge_and_dismiss_are_distinct(tmp_path):
    alerts, database = _database(tmp_path)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    first = database.record_event(frame, "CAM-01", "FIRE", 0.91, event_confirmed_at="2026-11-13T10:00:00.123456+00:00")
    second = database.record_event(frame, "CAM-02", "SMOKE", 0.82)
    assert first["id"] != second["id"]
    assert database.confirm_event(first["id"], note="Verified by operator")
    assert database.dismiss_event(second["id"], note="Steam plume")
    with sqlite3.connect(alerts.db_path) as conn:
        a, b = conn.execute("SELECT status,event_state,acknowledged_at,dismissed_at,dismissal_reason FROM alert_logs ORDER BY id").fetchall()
    assert a[0:2] == ("ACKNOWLEDGED", "CONFIRMED")
    assert a[2] and a[3] is None
    assert b[0:2] == ("DISMISSED", "CONFIRMED")
    assert b[2] is None and b[3] and b[4] == "Steam plume"
    history = {row["id"]: row for row in database.query_events(include_demo=True)}
    assert history[first["id"]]["event_state"] == "CONFIRMED"
    assert history[first["id"]]["acknowledged_at"]
    assert history[second["id"]]["event_state"] == "CONFIRMED"
    assert history[second["id"]]["dismissal_reason"] == "Steam plume"


def test_concurrent_event_writes_keep_unique_ids(tmp_path):
    alerts, database = _database(tmp_path)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    def insert(index):
        return database.record_event(frame, f"CAM-{index:02d}", "FIRE", 0.7)["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(insert, range(1, 33)))
    assert len(set(ids)) == 32
    with sqlite3.connect(alerts.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM alert_logs").fetchone()[0] == 32


def test_delivery_rows_require_an_existing_event(tmp_path):
    alerts, database = _database(tmp_path)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    event = database.record_event(frame, "CAM-01", "FIRE", 0.75)
    from src.telegram_alerts import TelegramAlertDispatcher

    dispatcher = TelegramAlertDispatcher(alerts.db_path)
    try:
        with sqlite3.connect(alerts.db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO alert_deliveries(delivery_id,event_id,channel,status,queued_at) VALUES('bad','999','telegram','QUEUED','now')")
        assert event["id"] > 0
    finally:
        dispatcher.close()
