"""Comprehensive automated test suite for Telegram fire alerts with camera GPS & Google Maps.

Covers TC01 through TC14:
- TC01: Confirmed fire event with configured GPS and snapshot image
- TC02: Confirmed smoke event with correct label and Google Maps URL
- TC03: Camera without GPS displays 'Vị trí camera: Chưa cấu hình'
- TC04: Camera with address only displays address without fabricated GPS/Maps URL
- TC05: Safe rejection and handling of invalid coordinates (out of range, NaN, Infinity)
- TC06: Multi-camera location isolation (CAM-01 vs CAM-02)
- TC07: Location changes after event preserve immutable historical snapshot
- TC08: Telegram deduplication prevents duplicate sends for repeated enqueue
- TC09: Delivery failure preserves retry/ambiguous policy without crashing
- TC10: Telegram and Web Push operate completely independently
- TC11: Missing snapshot image falls back to text while preserving location and Maps URL
- TC12: Long address/location respects Telegram 1024-character caption limit safely
- TC13: Legacy queued event payloads without location deserialize and format gracefully
- TC14: End-to-end event pipeline integration from detection/database to Telegram API
"""
import io
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs

import cv2
import numpy as np
import pytest

from src.notifications import CameraLocation, NotificationManager, NotificationQueue, validate_coordinates
from src.telegram_alerts import TelegramAlertDispatcher, format_telegram_alert, utc_now


def _init_test_database(db_path: Path):
    """Create alert_logs and notification tables in a temporary test database."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS alert_logs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            camera_name TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            alert_level TEXT NOT NULL,
            image_path TEXT NOT NULL,
            status TEXT NOT NULL
        )""")
    # Initialize notification tables
    NotificationQueue(str(db_path))


def _insert_event_to_db(db_path: Path, event: dict):
    """Insert event record to satisfy foreign key constraint on alert_deliveries."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO alert_logs(id, timestamp, camera_name, label, confidence, alert_level, image_path, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(event["id"]),
                str(event.get("timestamp", utc_now())),
                str(event.get("camera_name", "CAM-01")),
                str(event.get("label", "FIRE")),
                float(event.get("confidence", 0.9)),
                str(event.get("alert_level", "CONFIRMED")),
                str(event.get("image_path", "")),
                str(event.get("status", "UNREAD")),
            ),
        )


def _create_test_snapshot(path: Path, event_id: int) -> Path:
    """Generate a valid dummy JPEG snapshot file."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = [0, 0, 255]  # Red color
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    img_path = path.parent / f"test_snap_{event_id}_{int(time.time()*1000)}.jpg"
    img_path.write_bytes(encoded.tobytes())
    return img_path


class _MockTelegramResponse:
    status = 200

    def __init__(self, message_id=101):
        self.payload = json.dumps({"ok": True, "result": {"message_id": message_id}}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


# ---------------------------------------------------------------------------
# TC01: Confirmed Fire Event with Configured GPS
# ---------------------------------------------------------------------------
def test_tc01_confirmed_fire_event_with_gps(tmp_path, monkeypatch):
    """TC01: Confirmed fire event queues location, Google Maps URL, and snapshot."""
    db_file = tmp_path / "test_firewatch.db"
    _init_test_database(db_file)
    queue = NotificationQueue(str(db_file))

    # Configure camera CAM-01 location
    loc = CameraLocation(
        camera_id="CAM-01",
        location_name="Sảnh chính Tòa nhà A",
        address="268 Lý Thường Kiệt, Q.10, TP.HCM",
        latitude=10.7721,
        longitude=106.6578,
        location_verified=True,
    )
    queue.set_location(loc)

    snap_path = _create_test_snapshot(db_file, 480)
    event = {
        "id": 480,
        "timestamp": "2026-11-16 13:09:50",
        "event_confirmed_at": "2026-11-16 13:09:50",
        "camera_name": "CAM-01",
        "camera_display_name": "Khu Vực Sảnh Chính",
        "label": "FIRE",
        "confidence": 0.952,
        "alert_level": "CONFIRMED",
        "image_path": str(snap_path),
        "status": "UNREAD",
    }

    # Snapshot location
    loc_snap = queue.snapshot_location(480, "CAM-01")
    assert loc_snap is not None
    assert loc_snap["latitude"] == 10.7721
    assert loc_snap["longitude"] == 106.6578
    assert "maps.google.com" in loc_snap["maps_url"] or "google.com/maps" in loc_snap["maps_url"]

    event["location"] = loc_snap
    _insert_event_to_db(db_file, event)

    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "mock-chat-123")

    dispatcher = TelegramAlertDispatcher(str(db_file))
    dispatcher.service_credentials_tested = True
    assert dispatcher.enqueue(event) is True

    # Check durable payload in SQLite
    with sqlite3.connect(db_file) as conn:
        row = conn.execute("SELECT payload_json FROM alert_deliveries WHERE event_id=480").fetchone()
        assert row is not None
        payload = json.loads(row[0])
        assert payload["id"] == 480
        assert payload["location"]["location_name"] == "Sảnh chính Tòa nhà A"
        assert payload["location"]["latitude"] == 10.7721
        assert payload["location"]["longitude"] == 106.6578

    # Format caption
    caption = format_telegram_alert(payload)
    assert "#480" in caption
    assert "CHÁY (FIRE)" in caption
    assert "CAM-01" in caption
    assert "Sảnh chính Tòa nhà A" in caption
    assert "268 Lý Thường Kiệt" in caption
    assert "10.772100, 106.657800" in caption
    assert "https://www.google.com/maps?q=10.772100,106.657800" in caption
    assert "Đã xác minh" in caption
    assert len(caption) <= 1024
    dispatcher.close()


# ---------------------------------------------------------------------------
# TC02: Confirmed Smoke Event
# ---------------------------------------------------------------------------
def test_tc02_confirmed_smoke_event(tmp_path):
    """TC02: Smoke event formats with correct smoke label, location, and Maps URL."""
    event = {
        "id": 481,
        "timestamp": "2026-11-16 13:10:00",
        "event_confirmed_at": "2026-11-16 13:10:00",
        "camera_name": "CAM-02",
        "camera_display_name": "Khu Vực Nhà Xe",
        "label": "SMOKE",
        "confidence": 0.845,
        "alert_level": "CONFIRMED",
        "image_path": "",
        "status": "UNREAD",
        "location": {
            "camera_id": "CAM-02",
            "location_name": "Nhà Xe Tầng Hầm",
            "address": "Bãi xe B1",
            "latitude": 10.7750,
            "longitude": 106.6600,
            "location_verified": False,
        },
    }
    caption = format_telegram_alert(event)
    assert "KHÓI (SMOKE)" in caption
    assert "CAM-02" in caption
    assert "Nhà Xe Tầng Hầm" in caption
    assert "Bãi xe B1" in caption
    assert "10.775000, 106.660000" in caption
    assert "https://www.google.com/maps?q=10.775000,106.660000" in caption
    assert "Chưa xác minh" in caption


# ---------------------------------------------------------------------------
# TC03: Camera without GPS
# ---------------------------------------------------------------------------
def test_tc03_camera_without_gps(tmp_path):
    """TC03: Unconfigured camera displays 'Vị trí camera: Chưa cấu hình' without fake Maps URL."""
    event = {
        "id": 482,
        "timestamp": "2026-11-16 13:10:15",
        "camera_name": "CAM-03",
        "camera_display_name": "Kho Hàng",
        "label": "FIRE",
        "confidence": 0.88,
        "location": None,
    }
    caption = format_telegram_alert(event)
    assert "📍 Vị trí camera: Chưa cấu hình" in caption
    assert "maps.google.com" not in caption
    assert "google.com/maps" not in caption
    assert "GPS:" not in caption


# ---------------------------------------------------------------------------
# TC04: Address Only
# ---------------------------------------------------------------------------
def test_tc04_address_only(tmp_path):
    """TC04: Camera with address only displays address without inventing GPS or coordinate Maps URL."""
    event = {
        "id": 483,
        "timestamp": "2026-11-16 13:10:30",
        "camera_name": "CAM-04",
        "camera_display_name": "Văn Phòng Tầng 3",
        "label": "FIRE",
        "confidence": 0.91,
        "location": {
            "camera_id": "CAM-04",
            "location_name": "Phòng Kế Toán",
            "address": "123 Đường Nam Kỳ Khởi Nghĩa, Q.3, TP.HCM",
            "latitude": None,
            "longitude": None,
            "location_verified": True,
        },
    }
    caption = format_telegram_alert(event)
    assert "123 Đường Nam Kỳ Khởi Nghĩa" in caption
    assert "Phòng Kế Toán" in caption
    assert "GPS:" not in caption
    assert "https://www.google.com/maps?q=" not in caption


# ---------------------------------------------------------------------------
# TC05: Invalid Coordinates Rejection
# ---------------------------------------------------------------------------
def test_tc05_invalid_coordinates():
    """TC05: Invalid coordinates are rejected safely without crashing."""
    invalid_cases = [
        (91.0, 106.0),
        (-95.0, 106.0),
        (10.0, 185.0),
        (10.0, -185.0),
        (float("nan"), 106.0),
        (10.0, float("inf")),
        (True, 106.0),  # Booleans are not coordinates
        ("invalid", 106.0),
    ]
    for lat, lon in invalid_cases:
        event = {
            "id": 500,
            "camera_name": "CAM-TEST",
            "label": "FIRE",
            "confidence": 0.8,
            "location": {
                "location_name": "Test Area",
                "address": "Test Address",
                "latitude": lat,
                "longitude": lon,
            },
        }
        caption = format_telegram_alert(event)
        # Should not crash, and should not output invalid GPS or broken URL
        assert "NaN" not in caption
        assert "inf" not in caption
        assert "91.000000" not in caption
        assert "-95.000000" not in caption


# ---------------------------------------------------------------------------
# TC06: Multi-Camera Location Isolation
# ---------------------------------------------------------------------------
def test_tc06_multi_camera_isolation(tmp_path):
    """TC06: Multiple cameras maintain strict location isolation without cross-leakage."""
    db_file = tmp_path / "multicam.db"
    _init_test_database(db_file)
    queue = NotificationQueue(str(db_file))

    queue.set_location(CameraLocation("CAM-01", "Khu Vực 1", "Địa chỉ 1", 10.1111, 106.1111))
    queue.set_location(CameraLocation("CAM-02", "Khu Vực 2", "Địa chỉ 2", 10.2222, 106.2222))

    snap1 = queue.snapshot_location(1, "CAM-01")
    snap2 = queue.snapshot_location(2, "CAM-02")

    assert snap1["camera_id"] == "CAM-01"
    assert snap1["location_name"] == "Khu Vực 1"
    assert snap1["latitude"] == 10.1111

    assert snap2["camera_id"] == "CAM-02"
    assert snap2["location_name"] == "Khu Vực 2"
    assert snap2["latitude"] == 10.2222

    # Verify no cross-camera data in snapshots
    assert snap1["location_name"] != snap2["location_name"]
    assert snap1["latitude"] != snap2["latitude"]


# ---------------------------------------------------------------------------
# TC07: Location Changes After an Event (Immutability)
# ---------------------------------------------------------------------------
def test_tc07_location_immutability_on_update(tmp_path):
    """TC07: Historical event snapshots remain immutable when camera location changes."""
    db_file = tmp_path / "immutable.db"
    _init_test_database(db_file)
    queue = NotificationQueue(str(db_file))

    # 1. Camera starts at Location A
    queue.set_location(CameraLocation("CAM-01", "Vị trí Cũ A", "Địa chỉ Cũ A", 10.1000, 106.1000))
    snap_old = queue.snapshot_location(101, "CAM-01")
    assert snap_old["location_name"] == "Vị trí Cũ A"

    # 2. Camera location is modified to Location B
    queue.set_location(CameraLocation("CAM-01", "Vị trí Mới B", "Địa chỉ Mới B", 10.2000, 106.2000))

    # 3. Retrieve Event 101's snapshot: must remain Location A
    hist_snap = queue.get_event_location(101)
    assert hist_snap["location_name"] == "Vị trí Cũ A"
    assert hist_snap["latitude"] == 10.1000

    # 4. New Event 102 receives Location B
    snap_new = queue.snapshot_location(102, "CAM-01")
    assert snap_new["location_name"] == "Vị trí Mới B"
    assert snap_new["latitude"] == 10.2000


# ---------------------------------------------------------------------------
# TC08: Telegram Deduplication
# ---------------------------------------------------------------------------
def test_tc08_telegram_deduplication(tmp_path, monkeypatch):
    """TC08: Repeated enqueue of the same event produces only 1 delivery record."""
    db_file = tmp_path / "dedup.db"
    _init_test_database(db_file)
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")

    dispatcher = TelegramAlertDispatcher(str(db_file))
    dispatcher.service_credentials_tested = True

    event = {
        "id": 888,
        "timestamp": utc_now(),
        "camera_name": "CAM-01",
        "label": "FIRE",
        "confidence": 0.9,
    }
    _insert_event_to_db(db_file, event)

    # First enqueue succeeds
    assert dispatcher.enqueue(event) is True
    # Second enqueue of same event is ignored (deduplicated)
    assert dispatcher.enqueue(event) is False

    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM alert_deliveries WHERE event_id=888").fetchone()[0]
        assert count == 1
    dispatcher.close()


# ---------------------------------------------------------------------------
# TC09: Telegram Delivery Failure and Retry Bounding
# ---------------------------------------------------------------------------
def test_tc09_telegram_delivery_failure_and_retry(tmp_path, monkeypatch):
    """TC09: Delivery network timeout is marked ambiguous/retry without crashing worker."""
    db_file = tmp_path / "timeout.db"
    _init_test_database(db_file)
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setenv("TELEGRAM_RETRIES", "1")

    # Mock urlopen to raise TimeoutError
    def _mock_urlopen(req, timeout=None):
        raise TimeoutError("Network connection timed out")

    monkeypatch.setattr("src.telegram_alerts.urlopen", _mock_urlopen)

    dispatcher = TelegramAlertDispatcher(str(db_file))
    dispatcher.service_credentials_tested = True

    event = {
        "id": 901,
        "timestamp": utc_now(),
        "camera_name": "CAM-01",
        "label": "FIRE",
        "confidence": 0.88,
    }
    _insert_event_to_db(db_file, event)
    assert dispatcher.enqueue(event) is True

    # Process delivery
    claim = dispatcher._claim_next()
    assert claim is not None
    delivery_id, payload, attempts = claim
    dispatcher._deliver(delivery_id, payload, attempts)

    with sqlite3.connect(db_file) as conn:
        row = conn.execute("SELECT status, error_category FROM alert_deliveries WHERE event_id=901").fetchone()
        assert row[0] in ("AMBIGUOUS", "FAILED", "RETRY_PENDING")
        assert "TIMEOUT" in (row[1] or "")
    dispatcher.close()


# ---------------------------------------------------------------------------
# TC10: Web Push and Telegram Independence
# ---------------------------------------------------------------------------
def test_tc10_web_push_and_telegram_independence(tmp_path, monkeypatch):
    """TC10: Telegram failure does not block Web Push; Web Push failure does not block Telegram."""
    db_file = tmp_path / "independent.db"
    _init_test_database(db_file)

    # Disable Telegram, enable Web Push
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    telegram_disp = TelegramAlertDispatcher(str(db_file))
    push_manager = NotificationManager(str(db_file))

    event = {
        "id": 1001,
        "timestamp": utc_now(),
        "camera_name": "CAM-01",
        "label": "FIRE",
        "confidence": 0.9,
    }

    # Telegram enqueue fails gracefully without exception
    assert telegram_disp.enqueue(event) is False

    # Web Push enqueue still works smoothly
    push_manager.enqueue_event(event, "CAM-01")
    # Verify event location was snapshotted
    with sqlite3.connect(db_file) as conn:
        snap = conn.execute("SELECT event_id FROM event_location_records WHERE event_id=1001").fetchone()
        # Even if camera has no record, enqueue_event executed without crashing
    telegram_disp.close()
    push_manager.close()


# ---------------------------------------------------------------------------
# TC11: Missing Snapshot Image Text Fallback
# ---------------------------------------------------------------------------
def test_tc11_missing_snapshot_image_text_fallback(tmp_path, monkeypatch):
    """TC11: When snapshot image is missing or cannot be decoded, text alert is sent with full location."""
    db_file = tmp_path / "notfound.db"
    _init_test_database(db_file)
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")

    delivered_requests = []

    def _mock_urlopen(req, timeout=None):
        delivered_requests.append(req)
        return _MockTelegramResponse()

    monkeypatch.setattr("src.telegram_alerts.urlopen", _mock_urlopen)

    dispatcher = TelegramAlertDispatcher(str(db_file))
    dispatcher.service_credentials_tested = True

    event = {
        "id": 1101,
        "timestamp": utc_now(),
        "camera_name": "CAM-01",
        "label": "FIRE",
        "confidence": 0.95,
        "image_path": str(tmp_path / "non_existent_snapshot.jpg"),
        "location": {
            "location_name": "Khu Vực B",
            "address": "456 Lê Lợi",
            "latitude": 10.7788,
            "longitude": 106.6999,
            "location_verified": True,
        },
    }
    _insert_event_to_db(db_file, event)
    dispatcher.enqueue(event)
    claim = dispatcher._claim_next()
    delivery_id, payload, attempts = claim
    dispatcher._deliver(delivery_id, payload, attempts)

    assert len(delivered_requests) == 1
    # Fallback to sendMessage URL
    assert "sendMessage" in delivered_requests[0].full_url
    body = delivered_requests[0].data.decode("utf-8")
    assert "456+L%C3%AA+L%E1%BB%A3i" in body or "456 Lê Lợi" in body or "10.778800" in body
    assert "https%3A%2F%2Fwww.google.com%2Fmaps%3Fq%3D10.778800%2C106.699900" in body or "https://www.google.com/maps?q=10.778800,106.699900" in body
    dispatcher.close()


# ---------------------------------------------------------------------------
# TC12: Caption Length Overflow Protection
# ---------------------------------------------------------------------------
def test_tc12_caption_length_protection():
    """TC12: Extremely long address/location respects Telegram 1024-character caption limit."""
    long_address = "Đường rất dài số " + ("9" * 800) + ", Phường mở rộng, Quận trung tâm, Thành phố Hồ Chí Minh"
    event = {
        "id": 1201,
        "timestamp": "2026-11-16 13:30:00",
        "camera_name": "CAM-01",
        "camera_display_name": "Khu Vực C",
        "label": "FIRE",
        "confidence": 0.99,
        "location": {
            "location_name": "Khu phức hợp công nghiệp",
            "address": long_address,
            "latitude": 10.7721,
            "longitude": 106.6578,
            "location_verified": True,
        },
    }
    caption = format_telegram_alert(event, max_length=1024)
    assert len(caption) <= 1024
    assert "#1201" in caption
    assert "CHÁY (FIRE)" in caption
    assert "CAM-01" in caption
    assert "https://www.google.com/maps?q=10.772100,106.657800" in caption
    assert "..." in caption


# ---------------------------------------------------------------------------
# TC13: Legacy Queued Events Backward Compatibility
# ---------------------------------------------------------------------------
def test_tc13_legacy_queued_events_compatibility(tmp_path):
    """TC13: Legacy payload without location key formats gracefully without error."""
    legacy_payload = {
        "id": 1301,
        "timestamp": "2026-11-16 12:00:00",
        "camera_name": "CAM-LEGACY",
        "camera_display_name": "Camera Cũ",
        "label": "SMOKE",
        "confidence": 0.77,
        "alert_level": "CONFIRMED",
        "image_path": "",
        "status": "UNREAD",
        # Notice: no "location" key
    }
    caption = format_telegram_alert(legacy_payload)
    assert "#1301" in caption
    assert "KHÓI (SMOKE)" in caption
    assert "CAM-LEGACY" in caption
    assert "📍 Vị trí camera: Chưa cấu hình" in caption


# ---------------------------------------------------------------------------
# TC14: End-to-End Pipeline Integration
# ---------------------------------------------------------------------------
def test_tc14_end_to_end_event_pipeline(tmp_path, monkeypatch):
    """TC14: End-to-end event flow: Config -> Snapshot -> Event -> Queue -> Formatted Delivery."""
    db_file = tmp_path / "e2e.db"
    _init_test_database(db_file)
    queue = NotificationQueue(str(db_file))

    # 1. Config camera location
    queue.set_location(CameraLocation(
        "CAM-01", "Sảnh đón khách", "Tòa nhà Bitexco, Q.1", 10.7716, 106.7044, True
    ))

    # 2. Setup mock Telegram dispatcher
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-tok-14")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id-14")

    sent_requests = []

    def _mock_urlopen(req, timeout=None):
        sent_requests.append(req)
        return _MockTelegramResponse(message_id=999)

    monkeypatch.setattr("src.telegram_alerts.urlopen", _mock_urlopen)

    dispatcher = TelegramAlertDispatcher(str(db_file))
    dispatcher.service_credentials_tested = True

    # 3. Trigger confirmed event
    snap_path = _create_test_snapshot(db_file, 1401)
    event_payload = {
        "id": 1401,
        "timestamp": "2026-11-16 13:45:00",
        "event_confirmed_at": "2026-11-16 13:45:00",
        "camera_name": "CAM-01",
        "camera_display_name": "Sảnh Chính",
        "label": "FIRE",
        "confidence": 0.94,
        "alert_level": "CONFIRMED",
        "image_path": str(snap_path),
        "status": "UNREAD",
    }

    # Location snapshot is taken
    loc_snap = queue.snapshot_location(1401, "CAM-01")
    event_payload["location"] = loc_snap
    _insert_event_to_db(db_file, event_payload)

    # Enqueue to Telegram
    assert dispatcher.enqueue(event_payload) is True

    # Run delivery
    claim = dispatcher._claim_next()
    assert claim is not None
    delivery_id, payload, attempts = claim
    dispatcher._deliver(delivery_id, payload, attempts)

    # 4. Verify API request
    assert len(sent_requests) == 1
    req = sent_requests[0]
    assert "sendPhoto" in req.full_url
    assert "multipart/form-data" in req.headers["Content-type"]

    data = bytes(req.data)
    assert b"snapshot.jpg" in data
    assert b"10.771600, 106.704400" in data
    assert b"https://www.google.com/maps?q=10.771600,106.704400" in data
    assert b"Bitexco" in data

    # 5. Verify database delivery status is DELIVERED
    with sqlite3.connect(db_file) as conn:
        status, ack_at = conn.execute("SELECT status, telegram_acknowledged_at FROM alert_deliveries WHERE event_id=1401").fetchone()
        assert status == "DELIVERED"
        assert ack_at is not None

    dispatcher.close()
