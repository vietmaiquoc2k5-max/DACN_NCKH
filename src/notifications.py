"""Provider independent persistent notifications and camera location records.

Web Push is opt-in per browser subscription. Delivery status means the push
service accepted the request; it does not prove that a device displayed it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def validate_coordinates(latitude: Any, longitude: Any):
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude must be numeric")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Coordinates are outside valid latitude/longitude ranges")
    return lat, lon


class CameraLocation:
    def __init__(self, camera_id: str, location_name: str = "", address: str = "",
                 latitude: Any = None, longitude: Any = None, location_verified: bool = False):
        if not camera_id or len(camera_id) > 120:
            raise ValueError("camera_id is required")
        self.camera_id = camera_id
        self.location_name = location_name.strip()
        self.address = address.strip()
        self.latitude, self.longitude = validate_coordinates(latitude, longitude)
        self.location_verified = bool(location_verified)

    @property
    def maps_url(self) -> str:
        return "https://www.google.com/maps/search/?api=1&query={},{}".format(self.latitude, self.longitude)


class NotificationQueue:
    """SQLite backed deduplicated queue using additive tables only."""
    def __init__(self, db_path: str, max_attempts: int = 4):
        self.db_path = str(db_path)
        self.max_attempts = max(1, min(8, int(max_attempts)))
        self._init()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init(self):
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS push_subscriptions (subscription_id TEXT PRIMARY KEY, endpoint TEXT NOT NULL UNIQUE, subscription_json TEXT NOT NULL, created_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)")
            conn.execute("CREATE TABLE IF NOT EXISTS camera_locations (camera_id TEXT PRIMARY KEY, location_name TEXT NOT NULL DEFAULT '', address TEXT NOT NULL DEFAULT '', latitude REAL NOT NULL, longitude REAL NOT NULL, location_verified INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS event_location_snapshots (event_id INTEGER PRIMARY KEY, camera_id TEXT NOT NULL, location_name TEXT NOT NULL, address TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL, location_verified INTEGER NOT NULL, maps_url TEXT NOT NULL, FOREIGN KEY(event_id) REFERENCES alert_logs(id))")
            conn.execute("CREATE TABLE IF NOT EXISTS notification_jobs (job_id TEXT PRIMARY KEY, event_id INTEGER NOT NULL, provider TEXT NOT NULL, status TEXT NOT NULL, payload_json TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error_message TEXT, UNIQUE(event_id,provider))")
            conn.execute("CREATE TABLE IF NOT EXISTS notification_settings (setting_id INTEGER PRIMARY KEY CHECK(setting_id=1), enabled INTEGER NOT NULL, updated_at TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notification_jobs_pending ON notification_jobs(status,next_attempt_at,created_at)")

    def set_location(self, location: CameraLocation):
        with self.connect() as conn:
            conn.execute("INSERT INTO camera_locations VALUES(?,?,?,?,?,?,?) ON CONFLICT(camera_id) DO UPDATE SET location_name=excluded.location_name,address=excluded.address,latitude=excluded.latitude,longitude=excluded.longitude,location_verified=excluded.location_verified,updated_at=excluded.updated_at", (location.camera_id, location.location_name, location.address, location.latitude, location.longitude, int(location.location_verified), _now()))

    def locations(self):
        with self.connect() as conn:
            return [dict(zip(("camera_id", "location_name", "address", "latitude", "longitude", "location_verified", "updated_at"), row)) for row in conn.execute("SELECT camera_id,location_name,address,latitude,longitude,location_verified,updated_at FROM camera_locations ORDER BY camera_id")]

    def snapshot_location(self, event_id: int, camera_id: str):
        with self.connect() as conn:
            row = conn.execute("SELECT location_name,address,latitude,longitude,location_verified FROM camera_locations WHERE camera_id=?", (camera_id,)).fetchone()
            if not row:
                return None
            maps = "https://www.google.com/maps/search/?api=1&query={},{}".format(row[2], row[3])
            conn.execute("INSERT OR IGNORE INTO event_location_snapshots VALUES(?,?,?,?,?,?,?,?)", (event_id, camera_id) + row + (maps,))
            return {"camera_id": camera_id, "location_name": row[0], "address": row[1], "latitude": row[2], "longitude": row[3], "location_verified": bool(row[4]), "maps_url": maps}

    @staticmethod
    def validate_subscription(data: Any):
        if not isinstance(data, dict) or not isinstance(data.get("endpoint"), str):
            raise ValueError("Invalid push subscription")
        endpoint = data["endpoint"]
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or len(endpoint) > 4096:
            raise ValueError("Push endpoint must be a valid HTTPS URL")
        host = parsed.hostname.lower().rstrip(".")
        allowed_hosts = {item.strip().lower().rstrip(".") for item in os.getenv(
            "FIREWATCH_PUSH_ALLOWED_HOSTS",
            "fcm.googleapis.com,updates.push.services.mozilla.com,web.push.apple.com,notify.windows.com",
        ).split(",") if item.strip()}
        if host not in allowed_hosts and not any(host.endswith("." + suffix) for suffix in allowed_hosts):
            raise ValueError("Push endpoint host is not an approved browser push service")
        keys = data.get("keys")
        if not isinstance(keys, dict) or not all(isinstance(keys.get(k), str) and keys[k] for k in ("p256dh", "auth")):
            raise ValueError("Push subscription keys are missing")
        return endpoint

    def subscribe(self, data: Any):
        endpoint = self.validate_subscription(data)
        sid = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("INSERT INTO push_subscriptions(subscription_id,endpoint,subscription_json,created_at,active) VALUES(?,?,?,?,1) ON CONFLICT(endpoint) DO UPDATE SET subscription_json=excluded.subscription_json,active=1", (sid, endpoint, json.dumps(data), _now()))
            row = conn.execute("SELECT subscription_id FROM push_subscriptions WHERE endpoint=?", (endpoint,)).fetchone()
            return row[0]

    def unsubscribe(self, endpoint: str):
        with self.connect() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint=?", (endpoint,))

    def enqueue(self, event: Dict[str, Any], location: Optional[Dict[str, Any]] = None,
                ready: bool = True):
        if not event or not event.get("id"):
            return False
        payload = {"event_id": int(event["id"]), "type": str(event.get("label", "FIRE")).upper(),
                   "camera_name": str(event.get("camera_display_name") or event.get("camera_name") or "Camera"),
                   "camera_id": str(event.get("camera_name", "")), "confirmed_at": str(event.get("event_confirmed_at") or event.get("timestamp") or _now()),
                   "location": location or {}}
        with self.connect() as conn:
            state = "QUEUED" if ready else "FAILED"
            error = None if ready else "VAPID_NOT_CONFIGURED"
            cur = conn.execute("INSERT OR IGNORE INTO notification_jobs(job_id,event_id,provider,status,payload_json,created_at,updated_at,error_message) VALUES(?,?,'webpush',?,?,?,?,?)", (uuid.uuid4().hex, int(event["id"]), state, json.dumps(payload), _now(), _now(), error))
        return cur.rowcount == 1

    def history(self, limit: int = 100):
        with self.connect() as conn:
            return [dict(zip(("job_id", "event_id", "status", "attempt_count", "created_at", "updated_at", "error_message"), r)) for r in conn.execute("SELECT job_id,event_id,status,attempt_count,created_at,updated_at,error_message FROM notification_jobs ORDER BY created_at DESC LIMIT ?", (limit,))]


class WebPushNotifier:
    def __init__(self, queue: NotificationQueue):
        self.queue = queue
        self.public_key = os.getenv("FIREWATCH_VAPID_PUBLIC_KEY", "")
        self.private_key = os.getenv("FIREWATCH_VAPID_PRIVATE_KEY", "")
        self.subject = os.getenv("FIREWATCH_VAPID_SUBJECT", "")
        public_key_valid = bool(re.fullmatch(r"[A-Za-z0-9_-]{80,100}", self.public_key))
        self.configured = bool(public_key_valid and self.private_key and
                               (self.subject.startswith("mailto:") or self.subject.startswith("https://")))

    def send(self, subscription: Dict[str, Any], payload: Dict[str, Any]):
        if not self.configured:
            raise RuntimeError("VAPID configuration is incomplete")
        from pywebpush import webpush
        return webpush(subscription_info=subscription, data=json.dumps(payload), vapid_private_key=self.private_key,
                       vapid_claims={"sub": self.subject}, ttl=60)


class EmailNotifier:
    """Optional SMTP provider interface; disabled unless explicitly configured."""
    def send(self, payload: Dict[str, Any]):
        raise RuntimeError("SMTP fallback is optional and not configured")


class NotificationManager:
    def __init__(self, db_path: str):
        self.queue = NotificationQueue(db_path, int(os.getenv("NOTIFICATION_RETRIES", "4")))
        self.webpush = WebPushNotifier(self.queue)
        self.email = EmailNotifier()
        self.enabled = os.getenv("WEB_PUSH_ENABLED", "false").lower() in ("1", "true", "yes", "on")
        with self.queue.connect() as conn:
            if self.enabled:
                conn.execute("INSERT OR IGNORE INTO notification_settings VALUES(1,1,?)", (_now(),))
            saved = conn.execute("SELECT enabled FROM notification_settings WHERE setting_id=1").fetchone()
            if saved is not None:
                self.enabled = bool(saved[0])
        self._stop = threading.Event()
        self._thread = None
        if self.enabled and self.webpush.configured:
            self._thread = threading.Thread(target=self._run, name="web-push-notifications", daemon=True)
            self._thread.start()

    def enqueue_event(self, event: Dict[str, Any], camera_id: str):
        location = self.queue.snapshot_location(int(event["id"]), camera_id)
        if self.enabled:
            self.queue.enqueue(event, location, ready=self.webpush.configured)

    def set_enabled(self, enabled: bool):
        if enabled and not self.webpush.configured:
            raise ValueError("Configure VAPID public/private keys and mailto subject first")
        self.enabled = bool(enabled)
        with self.queue.connect() as conn:
            conn.execute("INSERT INTO notification_settings VALUES(1,?,?) ON CONFLICT(setting_id) DO UPDATE SET enabled=excluded.enabled,updated_at=excluded.updated_at", (int(self.enabled), _now()))
        if self.enabled and (self._thread is None or not self._thread.is_alive()):
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="web-push-notifications", daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.deliver_pending()
            except Exception:
                LOGGER.exception("Notification queue worker failed")
            self._stop.wait(0.5)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def deliver_pending(self):
        if not self.enabled or not self.webpush.configured:
            return
        with self.queue.connect() as conn:
            job = conn.execute("SELECT job_id,event_id,payload_json,attempt_count FROM notification_jobs WHERE provider='webpush' AND status IN ('QUEUED','RETRY_PENDING') AND (next_attempt_at IS NULL OR next_attempt_at<=?) ORDER BY created_at LIMIT 1", (_now(),)).fetchone()
            if not job:
                return
            conn.execute("UPDATE notification_jobs SET status='SENDING',attempt_count=attempt_count+1,updated_at=? WHERE job_id=?", (_now(), job[0]))
            subs = conn.execute("SELECT subscription_id,subscription_json FROM push_subscriptions WHERE active=1").fetchall()
        payload = json.loads(job[2])
        errors = []
        delivered = 0
        for sid, raw in subs:
            try:
                self.webpush.send(json.loads(raw), payload)
                delivered += 1
            except Exception as exc:
                status = getattr(exc, "response", None)
                code = getattr(status, "status_code", None)
                if code in (404, 410):
                    with self.queue.connect() as conn:
                        conn.execute("DELETE FROM push_subscriptions WHERE subscription_id=?", (sid,))
                errors.append(type(exc).__name__)
                LOGGER.warning("Web Push failed for subscription %s (%s)", sid, type(exc).__name__)
        tries = job[3] + 1
        state = "ACCEPTED_BY_PUSH_SERVICE" if delivered else ("RETRY_PENDING" if tries < self.queue.max_attempts else "FAILED")
        retry = datetime.fromtimestamp(time.time() + min(300, 2 ** tries), timezone.utc).isoformat() if state == "RETRY_PENDING" else None
        with self.queue.connect() as conn:
            conn.execute("UPDATE notification_jobs SET status=?,next_attempt_at=?,updated_at=?,error_message=? WHERE job_id=?", (state, retry, _now(), ",".join(errors)[:300] or None, job[0]))

    def send_test(self):
        if not self.webpush.configured:
            raise ValueError("VAPID configuration is incomplete")
        with self.queue.connect() as conn:
            subs = conn.execute("SELECT subscription_id,subscription_json FROM push_subscriptions WHERE active=1").fetchall()
            if not subs:
                raise ValueError("No browser subscriptions are registered")
            payload = {"event_id": "TEST", "type": "TEST", "camera_name": "FireWatch", "confirmed_at": _now(), "location": {"location_name": "Test notification", "maps_url": "/"}}
            job_id = uuid.uuid4().hex
            conn.execute("INSERT INTO notification_jobs(job_id,event_id,provider,status,payload_json,created_at,updated_at) VALUES(?,0,'webpush_test','SENDING',?,?,?)", (job_id, json.dumps(payload), _now(), _now()))
        delivered, errors = 0, []
        for sid, raw in subs:
            try:
                self.webpush.send(json.loads(raw), payload)
                delivered += 1
            except Exception as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                if code in (404, 410):
                    self.queue.unsubscribe(json.loads(raw).get("endpoint", ""))
                errors.append(type(exc).__name__)
                LOGGER.warning("Web Push test failed for subscription %s (%s)", sid, type(exc).__name__)
        state = "ACCEPTED_BY_PUSH_SERVICE" if delivered else "FAILED"
        with self.queue.connect() as conn:
            conn.execute("UPDATE notification_jobs SET status=?,updated_at=?,error_message=? WHERE job_id=?", (state, _now(), ",".join(errors)[:300] or None, job_id))
        return {"job_id": job_id, "status": state, "accepted_subscriptions": delivered}
