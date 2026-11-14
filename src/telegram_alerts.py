"""Durable, asynchronous Telegram photo delivery with explicit ambiguity handling."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
import threading
import time
import uuid
from urllib.parse import urlencode
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class TelegramTestError(RuntimeError):
    """Sanitized test-message error with a stable operator-facing category."""

    def __init__(self, category: str, message: str, ambiguous: bool = False):
        super().__init__(message)
        self.category = category
        self.ambiguous = ambiguous


def send_test_message(token: str, chat_id: str, timeout: float = 8.0) -> str:
    """Verify bot identity and send a labeled non-event test message.

    Credentials are accepted only for this call and are never persisted or logged.
    Returns the UTC timestamp of Telegram's successful API acknowledgment.
    """
    safe_token = str(token or "").strip()
    safe_chat_id = str(chat_id or "").strip()
    if not safe_token:
        raise TelegramTestError("MISSING_TOKEN", "Bot token is not configured")
    if not safe_chat_id:
        raise TelegramTestError("MISSING_CHAT_ID", "Chat ID is not configured")
    try:
        with urlopen(f"https://api.telegram.org/bot{safe_token}/getMe", timeout=timeout) as response:
            bot_payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(bot_payload, dict) or bot_payload.get("ok") is not True:
            raise TelegramTestError("INVALID_TOKEN", "Telegram could not authenticate this bot token")
    except HTTPError as exc:
        category = "INVALID_TOKEN" if exc.code in (401, 403) else "TELEGRAM_HTTP_ERROR"
        raise TelegramTestError(category, f"Telegram getMe returned HTTP {exc.code}") from None
    except (TimeoutError, URLError, OSError) as exc:
        raise TelegramTestError("NETWORK_TIMEOUT" if "timeout" in type(exc).__name__.lower() else "NETWORK_ERROR",
                                "Telegram API request failed; check network access and timeout") from None
    except TelegramTestError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise TelegramTestError("INVALID_RESPONSE", "Telegram returned an unreadable response") from None
    except Exception:
        raise TelegramTestError("NETWORK_ERROR", "Telegram API request failed; check network access and timeout") from None

    timestamp = datetime.now(timezone.utc)
    message = (
        "FIREWATCH — KIỂM TRA KẾT NỐI\n\n"
        "Bot Telegram đã gửi thành công tin nhắn kiểm tra của FireWatch.\n\n"
        "Loại: TIN NHẮN KIỂM TRA\n"
        "Nguồn: Bảng điều khiển FireWatch\n"
        f"Thời gian: {timestamp.isoformat(timespec='seconds')}\n\n"
        "Đây không phải cảnh báo cháy thật."
    )
    body = urlencode({"chat_id": safe_chat_id, "text": message}).encode("utf-8")
    request = Request(f"https://api.telegram.org/bot{safe_token}/sendMessage", data=body,
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or payload.get("ok") is not True or not isinstance(result, dict) or not isinstance(result.get("message_id"), int):
            raise TelegramTestError("DESTINATION_REJECTED", "Bot is valid, but Telegram did not acknowledge the destination")
    except HTTPError as exc:
        category = "DESTINATION_REJECTED" if exc.code in (400, 403) else "TELEGRAM_HTTP_ERROR"
        raise TelegramTestError(category, f"Telegram sendMessage returned HTTP {exc.code}") from None
    except (TimeoutError, URLError, OSError) as exc:
        raise TelegramTestError("NETWORK_TIMEOUT" if "timeout" in type(exc).__name__.lower() else "NETWORK_ERROR",
                                "Telegram message outcome is unknown; check the destination before retrying",
                                ambiguous=True) from None
    except TelegramTestError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise TelegramTestError("INVALID_RESPONSE", "Telegram returned an unreadable response") from None
    except Exception:
        raise TelegramTestError("NETWORK_ERROR", "Telegram message outcome is unknown; check the destination before retrying",
                                ambiguous=True) from None
    return utc_now()


def utc_now() -> str:
    """Return a UTC timestamp with microsecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class TelegramAlertDispatcher:
    """Persist and deliver confirmed snapshots without claiming exactly-once delivery.

    Timeouts have an ambiguous outcome: Telegram may have accepted the request even
    though its response did not reach this process. Such attempts become AMBIGUOUS
    and are not retried unless TELEGRAM_RETRY_AMBIGUOUS=true is explicitly set.
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = os.getenv("ALERTS_ENABLED", "false").lower() in ("1", "true", "yes", "on")
        self.timeout = max(1.0, float(os.getenv("TELEGRAM_TIMEOUT_SEC", "8")))
        self.retries = max(0, min(8, int(os.getenv("TELEGRAM_RETRIES", "3"))))
        self.max_pending = max(1, int(os.getenv("TELEGRAM_QUEUE_MAX", "256")))
        self.retry_ambiguous = os.getenv("TELEGRAM_RETRY_AMBIGUOUS", "false").lower() in ("1", "true", "yes", "on")
        self.poll_sec = max(0.05, float(os.getenv("TELEGRAM_QUEUE_POLL_SEC", "0.5")))
        # A successful test of the currently loaded service credentials is
        # process-local so old test rows cannot authorize a later credential set.
        self.service_credentials_tested = False
        self.stop_event = threading.Event()
        self.wakeup = threading.Event()
        self.thread_lock = threading.Lock()
        self.accepting = True
        self.thread: threading.Thread | None = None
        self._initialize()
        if self.configured:
            self._start_worker()

    @property
    def configured(self) -> bool:
        """Whether Telegram delivery is enabled and credentials are present."""
        return bool(self.enabled and self.token and self.chat_id)

    def _connect(self) -> sqlite3.Connection:
        """Create a short-lived SQLite connection with safe concurrency defaults."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _connection(self):
        """Own one short-lived connection, including transaction and close."""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        """Create or migrate the delivery table and recover interrupted work."""
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS alert_deliveries (
                delivery_id TEXT PRIMARY KEY,
                event_id INTEGER NOT NULL REFERENCES alert_logs(id),
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                event_confirmed_at TEXT,
                attempt_started_at TEXT,
                telegram_acknowledged_at TEXT,
                next_attempt_at TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL DEFAULT '{}',
                error_category TEXT,
                error_message TEXT,
                UNIQUE(event_id, channel))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS telegram_settings (
                setting_id INTEGER PRIMARY KEY CHECK(setting_id=1),
                enabled INTEGER NOT NULL,
                updated_at TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS telegram_test_deliveries (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                requested_at TEXT NOT NULL,
                acknowledged_at TEXT,
                status TEXT NOT NULL,
                error_category TEXT,
                used_service_credentials INTEGER NOT NULL DEFAULT 0)""")
            test_columns = {row[1] for row in conn.execute("PRAGMA table_info(telegram_test_deliveries)")}
            if "used_service_credentials" not in test_columns:
                conn.execute("ALTER TABLE telegram_test_deliveries ADD COLUMN used_service_credentials INTEGER NOT NULL DEFAULT 0")
            # Do not turn an initial environment-default false into a durable
            # operator override. The first explicit UI action creates the row.
            if self.enabled:
                conn.execute("INSERT OR IGNORE INTO telegram_settings(setting_id,enabled,updated_at) VALUES(1,1,?)",
                             (utc_now(),))
            persisted_enabled = conn.execute(
                "SELECT enabled FROM telegram_settings WHERE setting_id=1"
            ).fetchone()
            if persisted_enabled is not None:
                self.enabled = bool(persisted_enabled[0])
            columns = {row[1] for row in conn.execute("PRAGMA table_info(alert_deliveries)")}
            additions = {
                "event_confirmed_at": "TEXT",
                "attempt_started_at": "TEXT",
                "telegram_acknowledged_at": "TEXT",
                "next_attempt_at": "TEXT",
                "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "payload_json": "TEXT NOT NULL DEFAULT '{}'",
                "error_category": "TEXT",
            }
            for column, sql_type in additions.items():
                if column not in columns:
                    conn.execute(f"ALTER TABLE alert_deliveries ADD COLUMN {column} {sql_type}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_delivery_pending ON alert_deliveries(status,next_attempt_at,queued_at)")
            self._recover_interrupted(conn)

    def _start_worker(self) -> None:
        """Start the single delivery worker if it is not already running."""
        with self.thread_lock:
            if not self.accepting:
                raise RuntimeError("Telegram dispatcher is shutting down")
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="telegram-alerts", daemon=True)
                self.thread.start()
            self.wakeup.set()

    def set_enabled(self, enabled: bool) -> None:
        """Persist the nonsecret enabled setting and update the one dispatcher."""
        enabled = bool(enabled)
        if enabled and not (self.token and self.chat_id):
            raise ValueError("Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the service environment first")
        if enabled and not self.service_credentials_tested:
            raise ValueError("Run Test Connection using the service environment credentials before enabling alerts")
        if not self.accepting:
            raise RuntimeError("Telegram dispatcher is shutting down")
        with self._connection() as conn:
            conn.execute("""INSERT INTO telegram_settings(setting_id,enabled,updated_at) VALUES(1,?,?)
                ON CONFLICT(setting_id) DO UPDATE SET enabled=excluded.enabled,updated_at=excluded.updated_at""",
                         (int(enabled), utc_now()))
        self.enabled = enabled
        if enabled:
            self._start_worker()
        else:
            self.wakeup.set()

    def _recover_interrupted(self, conn: sqlite3.Connection) -> None:
        """Recover queued rows; mark interrupted network attempts as ambiguous."""
        now = utc_now()
        conn.execute("""UPDATE alert_deliveries SET status='AMBIGUOUS', error_category='PROCESS_INTERRUPTED',
            error_message='Process stopped during a Telegram request; automatic retry withheld', next_attempt_at=NULL
            WHERE status='SENDING'""")
        if self.retry_ambiguous:
            conn.execute("""UPDATE alert_deliveries SET status='RETRY_PENDING', error_category='AMBIGUOUS_RETRY_OPT_IN',
                error_message='Retrying an ambiguous outcome by explicit operator policy', next_attempt_at=?
                WHERE status='AMBIGUOUS'""", (now,))
        pending = conn.execute("SELECT delivery_id,event_id,payload_json FROM alert_deliveries WHERE status IN ('QUEUED','RETRY_PENDING')").fetchall()
        for delivery_id, event_id, payload_json in pending:
            try:
                payload = json.loads(payload_json or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if not payload:
                row = conn.execute("SELECT timestamp,camera_name,label,confidence,alert_level,image_path,status FROM alert_logs WHERE id=?", (event_id,)).fetchone()
                if row is None:
                    conn.execute("UPDATE alert_deliveries SET status='FAILED',error_category='EVENT_MISSING',error_message='Referenced event no longer exists' WHERE delivery_id=?", (delivery_id,))
                    continue
                keys = ("timestamp", "camera_name", "label", "confidence", "alert_level", "image_path", "status")
                payload = dict(zip(keys, row))
                payload["id"] = event_id
                conn.execute("UPDATE alert_deliveries SET payload_json=?,event_confirmed_at=COALESCE(event_confirmed_at,?) WHERE delivery_id=?",
                             (json.dumps(payload, separators=(",", ":")), payload.get("timestamp"), delivery_id))

    def enqueue(self, event: dict) -> bool:
        """Persist a confirmed event, or persist a visible failure.

        Returns true only when accepted into the durable pending queue. Duplicate
        event/channel rows are ignored without creating another Telegram send.
        """
        if not self.configured or not self.accepting or not event or not event.get("id"):
            return False
        event_id = int(event["id"])
        image_path = Path(str(event.get("image_path") or ""))
        payload = {
            "id": event_id,
            "timestamp": str(event.get("timestamp", "")),
            "camera_name": str(event.get("camera_name", "unknown")),
            "camera_display_name": str(event.get("camera_display_name", "unknown")),
            "label": str(event.get("label", "unknown")),
            "confidence": float(event.get("confidence", 0.0)),
            "alert_level": str(event.get("alert_level", "CONFIRMED")),
            "image_path": str(image_path),
            "status": str(event.get("status", "UNREAD")),
        }
        confirmed_at = str(event.get("event_confirmed_at") or payload["timestamp"] or utc_now())
        queued_at = utc_now()
        delivery_id = uuid.uuid4().hex
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            pending_count = conn.execute("SELECT COUNT(*) FROM alert_deliveries WHERE status IN ('QUEUED','SENDING','RETRY_PENDING')").fetchone()[0]
            if pending_count >= self.max_pending:
                conn.execute("""INSERT OR IGNORE INTO alert_deliveries
                    (delivery_id,event_id,channel,status,queued_at,event_confirmed_at,payload_json,error_category,error_message)
                    VALUES(?,?,'telegram','FAILED',?,?,?,'QUEUE_FULL','Persistent Telegram queue capacity reached')""",
                    (delivery_id, event_id, queued_at, confirmed_at, json.dumps(payload, separators=(",", ":"))))
                return False
            cursor = conn.execute("""INSERT OR IGNORE INTO alert_deliveries
                (delivery_id,event_id,channel,status,queued_at,event_confirmed_at,payload_json)
                VALUES(?,?,'telegram','QUEUED',?,?,?)""",
                (delivery_id, event_id, queued_at, confirmed_at, json.dumps(payload, separators=(",", ":"))))
            if cursor.rowcount == 0:
                return False
        self.wakeup.set()
        return True

    def _insert_failure(self, event: dict, category: str, message: str) -> None:
        """Persist a preflight failure when the referenced event exists."""
        if not event.get("id"):
            LOGGER.error("Telegram preflight failed without an event id (%s)", category)
            return
        payload = {"id": int(event["id"]), "image_path": str(event.get("image_path") or "")}
        with self._connection() as conn:
            conn.execute("""INSERT OR IGNORE INTO alert_deliveries
                (delivery_id,event_id,channel,status,queued_at,event_confirmed_at,payload_json,error_category,error_message)
                VALUES(?,?,'telegram','FAILED',?,?,?,?,?)""",
                (uuid.uuid4().hex, int(event["id"]), utc_now(),
                 str(event.get("event_confirmed_at") or event.get("timestamp") or utc_now()),
                 json.dumps(payload), category, message))

    @staticmethod
    def _read_snapshot(path: Path) -> bytes | None:
        """Read a decodable snapshot once, or return None for text fallback."""
        try:
            content = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            return content if image is not None else None
        except (OSError, cv2.error):
            return None

    def _claim_next(self) -> tuple[str, dict, int] | None:
        """Atomically claim one eligible delivery from the persistent queue."""
        now = utc_now()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""SELECT delivery_id,payload_json,attempt_count FROM alert_deliveries
                WHERE status IN ('QUEUED','RETRY_PENDING') AND (next_attempt_at IS NULL OR next_attempt_at<=?)
                ORDER BY queued_at LIMIT 1""", (now,)).fetchone()
            if row is None:
                return None
            delivery_id, payload_json, attempts = row
            try:
                payload = json.loads(payload_json)
            except (TypeError, json.JSONDecodeError):
                conn.execute("UPDATE alert_deliveries SET status='FAILED',error_category='PAYLOAD_INVALID',error_message='Stored delivery payload is invalid' WHERE delivery_id=?", (delivery_id,))
                return None
            attempts = int(attempts or 0) + 1
            conn.execute("""UPDATE alert_deliveries SET status='SENDING',attempt_count=?,attempt_started_at=?,
                next_attempt_at=NULL,error_category=NULL,error_message=NULL WHERE delivery_id=?""",
                (attempts, now, delivery_id))
            return delivery_id, payload, attempts

    def _run(self) -> None:
        """Poll the bounded SQLite queue and deliver off the inference threads."""
        while not self.stop_event.is_set():
            claim = self._claim_next() if self.configured else None
            if claim is None:
                self.wakeup.wait(self.poll_sec)
                self.wakeup.clear()
                continue
            delivery_id, payload, attempts = claim
            try:
                self._deliver(delivery_id, payload, attempts)
            except Exception as exc:
                # Exception strings and tracebacks can contain the bot token URL.
                LOGGER.error("Unhandled Telegram worker error for delivery %s (%s)",
                             delivery_id, type(exc).__name__)
                self._update(delivery_id, "FAILED", error_category="WORKER_ERROR",
                             error_message="Unhandled delivery worker error")

    def _deliver(self, delivery_id: str, event: dict, attempt: int) -> None:
        """Send an event snapshot, falling back to text when it is unavailable."""
        image = Path(str(event.get("image_path") or ""))
        snapshot = self._read_snapshot(image)
        raw_label = str(event.get("label", "unknown")).upper()
        event_label = {"FIRE": "CHÁY (FIRE)", "SMOKE": "KHÓI (SMOKE)"}.get(raw_label, raw_label)
        caption = (f"🔥 FIREWATCH — CẢNH BÁO ĐÃ XÁC NHẬN #{event['id']}\n"
                   f"Camera: {event.get('camera_name', 'unknown')} / {event.get('camera_display_name', 'unknown')}\n"
                   f"Loại sự kiện: {event_label}\n"
                   f"Độ tin cậy: {float(event.get('confidence', 0)):.1%}\n"
                   f"Thời điểm xác nhận: {event.get('event_confirmed_at') or event.get('timestamp', 'unknown')}\n"
                   "Trạng thái: ĐÃ XÁC NHẬN")
        if snapshot is not None:
            boundary = "----firewatch" + uuid.uuid4().hex
            body = bytearray()
            for key, value in (("chat_id", self.chat_id), ("caption", caption)):
                body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
            body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"snapshot.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode())
            body.extend(snapshot)
            body.extend(f"\r\n--{boundary}--\r\n".encode())
            request = Request(f"https://api.telegram.org/bot{self.token}/sendPhoto", data=bytes(body),
                              headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        else:
            # The confirmed event remains useful when its snapshot was not saved,
            # was removed, or could not be decoded. Never lose the alert for that reason.
            body = urlencode({"chat_id": self.chat_id, "text": caption}).encode("utf-8")
            request = Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=body,
                              headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status_code = int(response.status)
                response_text = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code in (401, 403):
                self._update(delivery_id, "FAILED", error_category="INVALID_CREDENTIALS",
                             error_message=f"Telegram HTTP {exc.code}")
            elif exc.code == 429 or exc.code >= 500:
                self._retry_or_fail(delivery_id, attempt, "HTTP_TRANSIENT", f"Telegram HTTP {exc.code}")
            else:
                self._update(delivery_id, "FAILED", error_category="HTTP_REJECTED",
                             error_message=f"Telegram HTTP {exc.code}")
            return
        except (TimeoutError, URLError, OSError) as exc:
            category = "NETWORK_TIMEOUT" if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower() else "NETWORK_OUTCOME_UNKNOWN"
            if self.retry_ambiguous and attempt <= self.retries:
                self._retry_or_fail(delivery_id, attempt, "AMBIGUOUS_RETRY_OPT_IN", category, ambiguous=True)
            else:
                self._update(delivery_id, "AMBIGUOUS", error_category=category,
                             error_message="Telegram outcome is unknown; automatic retry withheld to avoid duplicates")
            return
        except Exception as exc:
            self._update(delivery_id, "FAILED", error_category="REQUEST_ERROR",
                         error_message=type(exc).__name__)
            return
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            self._update(delivery_id, "AMBIGUOUS", error_category="ACK_INVALID",
                         error_message="Telegram response could not be parsed; automatic retry withheld")
            return
        message_id = payload.get("result", {}).get("message_id") if isinstance(payload, dict) else None
        if status_code == 200 and isinstance(payload, dict) and payload.get("ok") is True and isinstance(message_id, int):
            self._update(delivery_id, "DELIVERED", telegram_acknowledged_at=utc_now())
            return
        api_code = payload.get("error_code") if isinstance(payload, dict) else None
        message = f"Telegram API rejected delivery (code {api_code})" if api_code else "Telegram API rejected delivery"
        self._update(delivery_id, "FAILED", error_category="TELEGRAM_REJECTED", error_message=message)

    def _retry_or_fail(self, delivery_id: str, attempt: int, category: str, message: str,
                       ambiguous: bool = False) -> None:
        """Schedule bounded exponential retry or mark terminal failure."""
        if attempt > self.retries:
            self._update(delivery_id, "FAILED", error_category=category, error_message=message)
            return
        delay = min(300.0, 0.5 * (2 ** max(0, attempt - 1)))
        retry_at = datetime.fromtimestamp(time.time() + delay, timezone.utc).isoformat(timespec="microseconds")
        status = "RETRY_PENDING"
        if ambiguous and not self.retry_ambiguous:
            status, retry_at = "AMBIGUOUS", ""
        self._update(delivery_id, status, error_category=category, error_message=message,
                     next_attempt_at=retry_at)
        if status == "RETRY_PENDING":
            self.wakeup.set()

    def _update(self, delivery_id: str, status: str, *, telegram_acknowledged_at: str = "",
                error_category: str = "", error_message: str = "", next_attempt_at: str = "") -> None:
        """Update one delivery state atomically."""
        with self._connection() as conn:
            conn.execute("""UPDATE alert_deliveries SET status=?,telegram_acknowledged_at=?,
                error_category=?,error_message=?,next_attempt_at=? WHERE delivery_id=?""",
                (status, telegram_acknowledged_at or None, error_category or None,
                 error_message[:500] or None, next_attempt_at or None, delivery_id))

    def close(self, timeout: float = 10.0) -> bool:
        """Stop accepting work and wait boundedly for an active request to finish.

        Pending work remains in SQLite for startup recovery. Return false if a network
        request did not finish within the shutdown timeout.
        """
        self.accepting = False
        self.stop_event.set()
        self.wakeup.set()
        if self.thread is None:
            return True
        self.thread.join(timeout=max(0.0, timeout))
        return not self.thread.is_alive()

