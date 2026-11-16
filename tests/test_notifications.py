import json
import sqlite3
from pathlib import Path

import pytest

from src.notifications import CameraLocation, NotificationManager, NotificationQueue, WebPushNotifier, validate_coordinates


SUBSCRIPTION = {"endpoint": "https://fcm.googleapis.com/sub/1", "keys": {"p256dh": "public", "auth": "secret"}}


def test_camera_location_validates_and_builds_maps_link(tmp_path):
    location = CameraLocation("CAM-01", "Kho", "Hanoi", 21.0285, 105.8542, True)
    assert "21.0285,105.8542" in location.maps_url
    with pytest.raises(ValueError):
        CameraLocation("CAM-01", latitude=91, longitude=0)


def test_optional_coordinates_sources_and_coordinate_edge_cases():
    browser = CameraLocation("CAM-01", latitude=10.2, longitude=106.1,
                             accuracy_meters=25.5, location_source="BROWSER_GEOLOCATION")
    assert browser.accuracy_meters == 25.5
    address = CameraLocation("CAM-02", address="1 Main Street", location_source="MANUAL_ADDRESS")
    assert "1+Main+Street" in address.maps_url
    assert validate_coordinates(None, None) == (None, None)
    with pytest.raises(ValueError):
        CameraLocation("CAM-01", latitude=10.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            CameraLocation("CAM-01", latitude=bad, longitude=1)
    with pytest.raises(ValueError):
        CameraLocation("CAM-01", latitude=0, longitude=0)
    deliberate = CameraLocation("CAM-01", latitude=0, longitude=0, zero_zero_confirmed=True)
    assert "0.0,0.0" in deliberate.maps_url


def test_camera_location_persists_invalidation_and_event_snapshot(tmp_path):
    db = str(tmp_path / "locations.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE alert_logs(id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO alert_logs(id) VALUES(1)")
    queue = NotificationQueue(db)
    queue.set_location(CameraLocation("CAM-01", "Lobby", "Old address", 21.0, 105.0,
                                      True, location_source="MANUAL_COORDINATES"))
    original = queue.snapshot_location(1, "CAM-01")
    assert original["location_verified"] is True
    with queue.connect() as conn:
        old = conn.execute("SELECT location_source,verified_at FROM camera_location_records WHERE camera_id='CAM-01'").fetchone()
        snap = conn.execute("SELECT latitude,longitude,location_source FROM event_location_records WHERE event_id=1").fetchone()
    assert old[0] == "MANUAL_COORDINATES" and old[1]
    queue.set_location(CameraLocation("CAM-01", "Main lobby", "New address", 21.1, 105.1,
                                      False, location_source="MAP_SELECTION"))
    updated = queue.locations()[0]
    assert updated["location_verified"] == 0
    assert updated["verified_at"] is None
    assert queue.snapshot_location(1, "CAM-01")["latitude"] == 21.0
    with queue.connect() as conn:
        assert conn.execute("SELECT latitude,longitude,location_source FROM event_location_records WHERE event_id=1").fetchone() == snap
    restarted = NotificationQueue(db)
    assert restarted.locations()[0]["latitude"] == 21.1


def test_legacy_location_tables_are_preserved_and_copied(tmp_path):
    db = str(tmp_path / "legacy.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE alert_logs(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE camera_locations(camera_id TEXT PRIMARY KEY,location_name TEXT NOT NULL DEFAULT '',address TEXT NOT NULL DEFAULT '',latitude REAL NOT NULL,longitude REAL NOT NULL,location_verified INTEGER NOT NULL DEFAULT 0,updated_at TEXT NOT NULL)")
        conn.execute("INSERT INTO camera_locations VALUES('CAM-02','Gate','Old address',10,20,1,'2026-01-01T00:00:00Z')")
    queue = NotificationQueue(db)
    assert queue.locations()[0]["latitude"] == 10
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM camera_locations").fetchone()[0] == 1


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
        row = conn.execute("SELECT camera_id,location_name,maps_url FROM event_location_records WHERE event_id=7").fetchone()
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
    assert "Đăng ký thiết bị này" in dashboard
    assert "Notification.requestPermission()" in api
    assert 'navigator.serviceWorker.register' in api
    assert "auto_assign_videos=False" in dashboard


def test_browser_geolocation_and_map_require_explicit_user_action():
    component = Path("components/camera_location/index.html").read_text(encoding="utf-8")
    dashboard = Path("web_app.py").read_text(encoding="utf-8")
    assert "navigator.geolocation.getCurrentPosition" in component
    assert "enableHighAccuracy:true" in component
    assert "timeout:15000" in component and "maximumAge:0" in component
    assert "locateButton.addEventListener('click'" in component
    assert "map.on('click'" in component and "draggable:true" in component
    assert "camera_location_map_" in dashboard
    assert "zero_zero_confirmed=zero_zero" in dashboard
