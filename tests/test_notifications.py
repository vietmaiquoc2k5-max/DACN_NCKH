import json
import sqlite3

import pytest

from src.notifications import CameraLocation, NotificationManager, NotificationQueue, WebPushNotifier


SUBSCRIPTION = {"endpoint": "https://fcm.googleapis.com/sub/1", "keys": {"p256dh": "public", "auth": "secret"}}


def test_camera_location_validates_and_builds_maps_link(tmp_path):
    location = CameraLocation("CAM-01", "Kho", "Hanoi", 21.0285, 105.8542, True)
    assert "21.0285,105.8542" in location.maps_url
    with pytest.raises(ValueError):
        CameraLocation("CAM-01", latitude=91, longitude=0)


def test_subscription_validates_persists_and_removes(tmp_path):
    queue = NotificationQueue(str(tmp_path / "n.db"))
    sid = queue.subscribe(SUBSCRIPTION)
    assert sid
    with queue.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM push_subscriptions").fetchone()[0] == 1
    queue.unsubscribe(SUBSCRIPTION["endpoint"])
    with queue.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM push_subscriptions").fetchone()[0] == 0
    with pytest.raises(ValueError):
        queue.subscribe({"endpoint": "http://not-secure", "keys": {}})


def test_queue_persists_deduplicates_event_and_location_snapshot(tmp_path):
    db = str(tmp_path / "events.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE alert_logs(id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO alert_logs(id) VALUES(7)")
    manager = NotificationManager(db)
    manager.queue.set_location(CameraLocation("CAM-01", "Warehouse", "Address", 1, 2, True))
    event = {"id": 7, "camera_name": "CAM-01", "camera_display_name": "North", "label": "FIRE", "timestamp": "2026-11-16T00:00:00Z"}
    manager.enabled = True
    manager.enqueue_event(event, "CAM-01")
    with manager.queue.connect() as conn:
        row = conn.execute("SELECT camera_id,location_name,maps_url FROM event_location_snapshots WHERE event_id=7").fetchone()
    assert row[0:2] == ("CAM-01", "Warehouse")
    assert "1.0,2.0" in row[2]
    with manager.queue.connect() as conn:
        payload = json.loads(conn.execute("SELECT payload_json FROM notification_jobs WHERE event_id=7").fetchone()[0])
    assert payload["event_id"] == 7 and payload["type"] == "FIRE"
    assert payload["location"]["maps_url"] == row[2]
    assert manager.queue.enqueue(event, {"location_name": "Warehouse"}) is False
    assert manager.queue.enqueue(event, {"location_name": "Warehouse"}) is False
    manager.close()


def test_expired_subscription_cleanup_and_bounded_retry(tmp_path, monkeypatch):
    db = str(tmp_path / "events.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE alert_logs(id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO alert_logs(id) VALUES(1)")
    monkeypatch.setenv("WEB_PUSH_ENABLED", "true")
    monkeypatch.setenv("FIREWATCH_VAPID_PUBLIC_KEY", "a" * 87)
    monkeypatch.setenv("FIREWATCH_VAPID_PRIVATE_KEY", "private")
    monkeypatch.setenv("FIREWATCH_VAPID_SUBJECT", "mailto:ops@example.test")
    monkeypatch.setenv("NOTIFICATION_RETRIES", "2")
    manager = NotificationManager(db)
    manager.queue.subscribe(SUBSCRIPTION)
    manager.queue.enqueue({"id": 1, "label": "SMOKE"})

    class Expired(Exception):
        response = type("Response", (), {"status_code": 410})()

    monkeypatch.setattr(manager.webpush, "send", lambda *_args: (_ for _ in ()).throw(Expired()))
    manager.deliver_pending()
    assert manager.queue.history()[0]["status"] == "RETRY_PENDING"
    with manager.queue.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM push_subscriptions").fetchone()[0] == 0
        conn.execute("UPDATE notification_jobs SET next_attempt_at=NULL")
    manager.deliver_pending()
    assert manager.queue.history()[0]["status"] == "FAILED"
    manager.close()


def test_vapid_requires_public_private_and_subject(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWATCH_VAPID_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("FIREWATCH_VAPID_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("FIREWATCH_VAPID_SUBJECT", raising=False)
    notifier = WebPushNotifier(NotificationQueue(str(tmp_path / "n.db")))
    assert notifier.configured is False


def test_dashboard_exposes_push_and_does_not_autosubscribe_or_connect_camera():
    dashboard = open("web_app.py", encoding="utf-8").read()
    api = open("src/week7_api.py", encoding="utf-8").read()
    assert '"Web Push"' in dashboard
    assert "Subscribe this device" in dashboard
    assert "Notification.requestPermission()" in api
    assert 'navigator.serviceWorker.register' in api
    assert "auto_assign_videos=False" in dashboard
