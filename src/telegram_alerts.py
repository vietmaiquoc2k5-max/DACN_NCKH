"""Asynchronous Telegram photo alerts with durable delivery state."""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


class TelegramAlertDispatcher:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = os.getenv("ALERTS_ENABLED", "false").lower() in ("1", "true", "yes", "on")
        self.timeout = max(1.0, float(os.getenv("TELEGRAM_TIMEOUT_SEC", "8")))
        self.retries = max(0, min(8, int(os.getenv("TELEGRAM_RETRIES", "3"))))
        self.work: queue.Queue = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self._initialize()
        self.thread = None
        if self.configured:
            self.thread = threading.Thread(target=self._run, name="telegram-alerts", daemon=True)
            self.thread.start()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _initialize(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS alert_deliveries (
                delivery_id TEXT PRIMARY KEY, event_id INTEGER NOT NULL, channel TEXT NOT NULL,
                status TEXT NOT NULL, queued_at TEXT NOT NULL, sent_at TEXT, confirmed_at TEXT,
                error_message TEXT, UNIQUE(event_id, channel))""")

    @property
    def configured(self):
        return bool(self.enabled and self.token and self.chat_id)

    def enqueue(self, event: dict) -> bool:
        if not self.configured or not event.get("id"):
            return False
        event_id = int(event["id"])
        queued_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        delivery_id = uuid.uuid4().hex
        with self._connect() as conn:
            cursor = conn.execute("""INSERT OR IGNORE INTO alert_deliveries
                (delivery_id,event_id,channel,status,queued_at) VALUES (?,?,'telegram','QUEUED',?)""",
                (delivery_id, event_id, queued_at))
            if cursor.rowcount == 0:
                return False
        try:
            self.work.put_nowait((delivery_id, dict(event)))
        except queue.Full:
            self._update(delivery_id, "FAILED", error="Telegram queue is full")
            return False
        return True

    def _update(self, delivery_id: str, status: str, sent_at: str = "", error: str = ""):
        with self._connect() as conn:
            conn.execute("UPDATE alert_deliveries SET status=?, sent_at=?, confirmed_at=?, error_message=? WHERE delivery_id=?",
                         (status, sent_at or None, sent_at or None, error[:500] or None, delivery_id))

    def _run(self):
        while not self.stop_event.is_set():
            try:
                delivery_id, event = self.work.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._deliver(delivery_id, event)
            finally:
                self.work.task_done()

    def _deliver(self, delivery_id: str, event: dict):
        image = Path(str(event.get("image_path") or ""))
        if not image.is_file():
            self._update(delivery_id, "FAILED", error="Snapshot file is missing")
            return
        text = (f"Firewatch event #{event['id']}\n"
                f"Camera: {event.get('camera_name', 'unknown')} / {event.get('camera_display_name', 'unknown')}\n"
                f"Class: {event.get('label', 'unknown')} ({float(event.get('confidence', 0)):.1%})\n"
                f"Confirmed: {event.get('timestamp', 'unknown')}\n"
                f"Status: {event.get('status', 'UNREAD')}")
        boundary = "----firewatch" + uuid.uuid4().hex
        fields = [("chat_id", self.chat_id), ("caption", text)]
        body = bytearray()
        for key, value in fields:
            body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"snapshot.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode())
        body.extend(image.read_bytes())
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        request = Request(f"https://api.telegram.org/bot{self.token}/sendPhoto", data=bytes(body),
                          headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        last_error = "delivery failed"
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and payload.get("ok") and payload.get("result", {}).get("message_id"):
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                    self._update(delivery_id, "DELIVERED", sent_at=stamp)
                    return
                last_error = "Telegram did not acknowledge the message"
            except Exception as exc:
                # urllib exceptions can embed the request URL, which contains the bot token.
                status = getattr(exc, "code", None)
                last_error = f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")
            if attempt < self.retries:
                self.stop_event.wait(min(8.0, 0.5 * (2 ** attempt)))
        self._update(delivery_id, "FAILED", error=last_error)

    def close(self, timeout: float = 3.0):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(0.0, timeout))

