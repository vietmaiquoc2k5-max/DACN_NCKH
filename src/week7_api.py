"""Week 7 HTTP API for the existing Firewatch camera and event pipeline.

The API deliberately creates no camera connections during startup. Management
routes require FIREWATCH_API_TOKEN, and RTSP destinations must be allowlisted.
"""
from __future__ import annotations

import os
import secrets
import time
import asyncio
import re
import sqlite3
from contextlib import closing
import json
import threading
import atexit
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.camera_manager import MultiCameraManager
from src.config import PROJECT_ROOT, mask_rtsp_credentials
from src.telegram_alerts import TelegramTestError, send_test_message


class CameraInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_type: str = Field(pattern="^(WEBCAM|VIDEO_TEST|IP_CAMERA)$")
    source: str = Field(min_length=1, max_length=2048)
    webcam_index: int = Field(default=0, ge=0, le=64)
    enabled: bool = True
    auto_reconnect: bool = True
    inference_fps: int = Field(default=8, ge=1, le=60)
    telegram_enabled: bool = True
    exclusion_zones: list[tuple[float, float, float, float]] = Field(default_factory=list, max_length=64)


class DismissInput(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class TelegramTestInput(BaseModel):
    # Length checks happen in the handler so Pydantic's validation response
    # cannot echo a rejected secret back to the caller.
    token: Optional[str] = None
    chat_id: Optional[str] = None


def _manager(request: Request) -> MultiCameraManager:
    manager = getattr(request.app.state, "manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail={"code": "NOT_READY", "message": "Camera service is starting"})
    return manager


def _same_dashboard_instance(
    request: Request,
    instance_id: Optional[str] = Header(default=None, alias="X-FireWatch-Instance-ID"),
) -> None:
    """Reject dashboard management requests routed to another FireWatch process."""
    if instance_id is None:
        return
    manager = getattr(request.app.state, "manager", None)
    expected = getattr(manager, "_fw_api_instance_id", "") if manager is not None else ""
    if not expected or not secrets.compare_digest(instance_id, expected):
        raise HTTPException(409, detail={"code": "BACKEND_INSTANCE_MISMATCH",
                                         "message": "Dashboard and management API belong to different FireWatch instances"})


def _authorized(authorization: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("FIREWATCH_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail={"code": "AUTH_NOT_CONFIGURED", "message": "Set FIREWATCH_API_TOKEN before enabling management API"})
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "A valid bearer token is required"})


def _validate_source(source_type: str, source: str, webcam_index: int) -> str:
    value = source.strip()
    if source_type == "WEBCAM":
        if not value.isdigit() or int(value) != webcam_index:
            raise HTTPException(422, detail={"code": "INVALID_SOURCE", "message": "Webcam source must match webcam_index"})
        return str(webcam_index)
    if source_type == "VIDEO_TEST":
        path = Path(value)
        resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            raise HTTPException(422, detail={"code": "PATH_NOT_ALLOWED", "message": "Video files must be within the project directory"})
        if not resolved.is_file():
            raise HTTPException(422, detail={"code": "SOURCE_NOT_FOUND", "message": "Video file does not exist"})
        return str(resolved)
    is_env_reference = value.startswith("env:")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        raise HTTPException(422, detail={"code": "INVALID_RTSP", "message": "Malformed RTSP URL"})
    if is_env_reference:
        variable = value[4:]
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", variable):
            raise HTTPException(422, detail={"code": "INVALID_SECRET_REFERENCE", "message": "Secret reference must use env:VARIABLE_NAME"})
        value = os.getenv(variable, "").strip()
        if not value:
            raise HTTPException(422, detail={"code": "SECRET_NOT_CONFIGURED", "message": "The referenced RTSP environment variable is empty"})
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
        except ValueError:
            raise HTTPException(422, detail={"code": "INVALID_RTSP", "message": "Malformed RTSP URL"})
    if parsed.scheme not in ("rtsp", "rtsps") or not hostname:
        raise HTTPException(422, detail={"code": "INVALID_RTSP", "message": "Use an RTSP or RTSPS URL"})
    if not is_env_reference and (parsed.username is not None or parsed.password is not None):
        raise HTTPException(422, detail={"code": "INLINE_CREDENTIALS_NOT_ALLOWED", "message": "Put authenticated RTSP URLs in an environment variable and use source env:VARIABLE_NAME"})
    allowlist = {host.strip().lower() for host in os.getenv("FIREWATCH_RTSP_HOSTS", "").split(",") if host.strip()}
    if hostname.lower() not in allowlist:
        raise HTTPException(422, detail={"code": "RTSP_HOST_NOT_ALLOWED", "message": "RTSP host is not present in FIREWATCH_RTSP_HOSTS"})
    return value


def _initialize_camera_config(db_path: str):
    with closing(sqlite3.connect(db_path, timeout=10)) as conn, conn:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("""CREATE TABLE IF NOT EXISTS cameras (
            camera_id TEXT PRIMARY KEY, name TEXT NOT NULL, source_type TEXT NOT NULL,
            source TEXT NOT NULL, webcam_index INTEGER NOT NULL DEFAULT 0,
            enabled INTEGER NOT NULL DEFAULT 1, auto_reconnect INTEGER NOT NULL DEFAULT 1,
            inference_fps INTEGER NOT NULL DEFAULT 8, exclusion_zones TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")


def _save_camera_config(manager: MultiCameraManager, camera_id: str, input_data: CameraInput):
    with manager.telegram_alerts._connect() as conn:
        conn.execute("""INSERT INTO cameras(camera_id,name,source_type,source,webcam_index,enabled,
            auto_reconnect,inference_fps,exclusion_zones,telegram_enabled,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name,source_type=excluded.source_type,
            source=excluded.source,webcam_index=excluded.webcam_index,enabled=excluded.enabled,
            auto_reconnect=excluded.auto_reconnect,inference_fps=excluded.inference_fps,
            exclusion_zones=excluded.exclusion_zones,telegram_enabled=excluded.telegram_enabled,
            updated_at=CURRENT_TIMESTAMP""",
            (camera_id, input_data.name, input_data.source_type, input_data.source,
             input_data.webcam_index, int(input_data.enabled), int(input_data.auto_reconnect),
             input_data.inference_fps, json.dumps(input_data.exclusion_zones), int(input_data.telegram_enabled)))


def _restore_camera_configs(manager: MultiCameraManager):
    with manager.telegram_alerts._connect() as conn:
        rows = conn.execute("SELECT camera_id,name,source_type,source,webcam_index,enabled,auto_reconnect,inference_fps,exclusion_zones FROM cameras ORDER BY camera_id").fetchall()
    errors = []
    for camera_id, name, source_type, source, webcam_index, enabled, reconnect, fps, zones_json in rows:
        try:
            actual_source = _validate_source(source_type, source, webcam_index)
            zones = json.loads(zones_json or "[]")
            if camera_id in manager.workers:
                manager.update_camera_source(camera_id, name, actual_source, source_type, webcam_index,
                                             bool(enabled), bool(reconnect), auto_restart=False)
                manager.workers[camera_id].config.inference_fps = int(fps)
                manager.update_camera_roi(camera_id, zones)
            else:
                restored_id = manager.add_camera(name, actual_source, source_type, webcam_index,
                                                 bool(enabled), bool(reconnect), zones,
                                                 auto_start=False, inference_fps=int(fps),
                                                 camera_id=camera_id)
                if restored_id != camera_id:
                    raise RuntimeError(f"Saved camera id {camera_id} restored as {restored_id}")
        except Exception as exc:
            safe_error = mask_rtsp_credentials(str(exc))
            manager.add_log(f"Camera configuration {camera_id} was not restored: {safe_error}")
            errors.append({"camera_id": camera_id, "error": safe_error})
    return errors


def _public_camera(snapshot: dict[str, Any]) -> dict[str, Any]:
    # Frames are available only through the live WebSocket to keep list payloads small.
    return {key: value for key, value in snapshot.items() if key not in ("frame_rgb", "preview_rgb", "jpeg_b64")}


@asynccontextmanager
async def _lifespan(application: FastAPI):
    manager = getattr(application.state, "injected_manager", None)
    owns_manager = manager is None
    if owns_manager:
        db_path = os.getenv("FIREWATCH_DB_PATH", "alerts.db")
        model_path = os.getenv("FIREWATCH_MODEL_PATH", "best.pt")
        manager = MultiCameraManager(db_path=db_path, model_path=model_path, auto_assign_videos=False)
    # This is intentionally idempotent: a shared Streamlit manager is authoritative.
    manager.stop_monitoring()
    application.state.camera_restore_errors = manager.camera_restore_errors
    application.state.manager = manager
    try:
        yield
    finally:
        if owns_manager:
            manager.stop_monitoring()
            manager.telegram_alerts.close()
        application.state.manager = None


app = FastAPI(title="Firewatch Week 7 API", version="1.0.0", lifespan=_lifespan)
app.state.ws_tickets = {}
app.state.ws_ticket_lock = threading.Lock()
_telegram_test_lock = threading.Lock()
_telegram_last_test_at = 0.0


def start_embedded_api(manager: MultiCameraManager, host: str = "0.0.0.0", port: int | None = None):
    """Serve the API beside Streamlit using the same manager and detector instance."""
    if getattr(app.state, "embedded_thread", None) is not None and app.state.embedded_thread.is_alive():
        previous_manager = getattr(app.state, "manager", None)
        if previous_manager is not None and previous_manager is not manager:
            previous_manager.stop_monitoring()
            previous_manager.telegram_alerts.close()
        app.state.injected_manager = manager
        app.state.manager = manager
        app.state.instance_id = getattr(manager, "_fw_api_instance_id", "")
        return app.state.embedded_server
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to expose the shared camera API; install requirements.txt") from exc
    listen_port = int(port or os.getenv("FIREWATCH_API_PORT", "8000"))
    app.state.injected_manager = manager
    app.state.instance_id = getattr(manager, "_fw_api_instance_id", "")
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=listen_port, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name="firewatch-api", daemon=True)
    app.state.embedded_server = server
    app.state.embedded_thread = thread
    thread.start()
    deadline = time.monotonic() + 3.0
    while thread.is_alive() and not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=1.0)
        raise RuntimeError(f"Shared API did not start on port {listen_port}; check the port and API settings")

    def stop_server():
        server.should_exit = True
        thread.join(timeout=5.0)
        active_manager = getattr(app.state, "manager", None) or manager
        active_manager.close()

    atexit.register(stop_server)
    return server


def _allowed_websocket_origins() -> set[str]:
    """Read exact browser origins permitted to open the camera WebSocket."""
    return {item.strip().rstrip("/") for item in os.getenv("FIREWATCH_ALLOWED_ORIGINS", "").split(",") if item.strip()}


def _issue_ticket(application: FastAPI, lifetime_sec: float = 45.0) -> str:
    """Create and store a random single-use camera WebSocket ticket."""
    ticket = secrets.token_urlsafe(32)
    with application.state.ws_ticket_lock:
        now = time.monotonic()
        application.state.ws_tickets = {
            key: expiry for key, expiry in application.state.ws_tickets.items() if expiry > now
        }
        application.state.ws_tickets[ticket] = now + lifetime_sec
    return ticket


def _consume_ticket(application: FastAPI, ticket: str, now: float | None = None) -> bool:
    """Consume a WebSocket ticket exactly once and reject expired values."""
    with application.state.ws_ticket_lock:
        expires_at = application.state.ws_tickets.pop(ticket, None)
    return expires_at is not None and expires_at > (time.monotonic() if now is None else now)


def _origin_is_allowed(origin: str, allowed: set[str]) -> bool:
    """Check a browser Origin against the exact configured allowlist."""
    return bool(origin) and origin.rstrip("/") in allowed


@app.post("/api/ws-tickets")
def create_websocket_ticket(request: Request, _: None = Depends(_authorized)):
    """Issue a short-lived, single-use ticket scoped to opening the camera feed."""
    ticket = _issue_ticket(request.app)
    return {"ticket": ticket, "expires_in_seconds": 45, "scope": "camera_stream"}


@app.get("/health")
def health(request: Request):
    manager = getattr(request.app.state, "manager", None)
    if manager is None:
        return {"status": "starting", "model": None}
    model_info = manager.detector.get_model_info()
    return {"status": "ok" if model_info.get("loaded") else "degraded", "model": model_info}


@app.get("/api/cameras")
def list_cameras(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    return [_public_camera(item) for item in manager.get_all_snapshots()]


@app.get("/api/cameras/config-errors")
def camera_restore_errors(request: Request, _: None = Depends(_authorized)):
    manager = getattr(request.app.state, "manager", None)
    return manager.camera_restore_errors if manager is not None else getattr(request.app.state, "camera_restore_errors", [])


@app.post("/api/cameras", status_code=201)
def add_camera(payload: CameraInput, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    source = _validate_source(payload.source_type, payload.source, payload.webcam_index)
    cam_id = manager.add_camera(
        name=payload.name, source=source, source_type=payload.source_type,
        webcam_index=payload.webcam_index, enabled=payload.enabled,
        auto_reconnect=payload.auto_reconnect, exclusion_zones=payload.exclusion_zones,
        auto_start=False, inference_fps=payload.inference_fps,
        source_reference=payload.source,
        telegram_enabled=payload.telegram_enabled,
    )
    _save_camera_config(manager, cam_id, payload)
    return {"camera_id": cam_id, "status": "DISCONNECTED", "source": mask_rtsp_credentials(source)}


@app.put("/api/cameras/{camera_id}")
def edit_camera(camera_id: str, payload: CameraInput, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    if camera_id not in manager.workers:
        raise HTTPException(404, detail={"code": "CAMERA_NOT_FOUND", "message": "Unknown camera id"})
    source = _validate_source(payload.source_type, payload.source, payload.webcam_index)
    manager.update_camera_source(camera_id, payload.name, source, payload.source_type,
                                 payload.webcam_index, payload.enabled, payload.auto_reconnect,
                                 auto_restart=False, source_reference=payload.source)
    manager.update_camera_roi(camera_id, payload.exclusion_zones)
    manager.workers[camera_id].config.inference_fps = payload.inference_fps
    manager.set_camera_telegram_enabled(camera_id, payload.telegram_enabled)
    _save_camera_config(manager, camera_id, payload)
    return _public_camera(manager.workers[camera_id].get_snapshot())


@app.delete("/api/cameras/{camera_id}")
def remove_camera(camera_id: str, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    if camera_id not in manager.workers:
        raise HTTPException(404, detail={"code": "CAMERA_NOT_FOUND", "message": "Unknown camera id"})
    manager.stop_camera(camera_id)
    if not manager.remove_camera(camera_id):
        raise HTTPException(409, detail={"code": "CAMERA_REQUIRED", "message": "At least one camera slot must remain"})
    with manager.telegram_alerts._connect() as conn:
        conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
    return {"camera_id": camera_id, "removed": True}


@app.post("/api/cameras/{camera_id}/connect")
def connect_camera(camera_id: str, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    if camera_id not in manager.workers:
        raise HTTPException(404, detail={"code": "CAMERA_NOT_FOUND", "message": "Unknown camera id"})
    worker = manager.workers[camera_id]
    if not worker.config.source.strip():
        raise HTTPException(409, detail={"code": "SOURCE_UNASSIGNED", "message": "Configure a source before connecting"})
    manager.start_camera(camera_id)
    if worker.status == "ERROR" and not worker.running:
        raise HTTPException(503, detail={"code": "CAMERA_START_FAILED", "message": worker.error_detail})
    return _public_camera(worker.get_snapshot())


@app.post("/api/cameras/{camera_id}/disconnect")
def disconnect_camera(camera_id: str, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    if camera_id not in manager.workers:
        raise HTTPException(404, detail={"code": "CAMERA_NOT_FOUND", "message": "Unknown camera id"})
    manager.stop_camera(camera_id)
    return _public_camera(manager.workers[camera_id].get_snapshot())


@app.post("/api/cameras/{camera_id}/reconnect")
def reconnect_camera(camera_id: str, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    if camera_id not in manager.workers:
        raise HTTPException(404, detail={"code": "CAMERA_NOT_FOUND", "message": "Unknown camera id"})
    manager.reconnect_camera(camera_id)
    return _public_camera(manager.workers[camera_id].get_snapshot())


@app.get("/api/events")
def list_events(limit: int = Query(100, ge=1, le=1000), camera: str = "", event_type: str = "", status: str = "", manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    allowed_statuses = {"", "UNREAD", "ACKNOWLEDGED", "DISMISSED"}
    normalized_status = status.upper()
    if normalized_status not in allowed_statuses:
        raise HTTPException(422, detail={"code": "INVALID_STATUS", "message": "Unsupported event status"})
    clauses = ["COALESCE(is_demo, 0) = 0"]
    params: list[Any] = []
    if camera:
        clauses.append("camera_name = ?")
        params.append(camera)
    if event_type:
        clauses.append("UPPER(label) LIKE ?")
        params.append(f"%{event_type.upper()}%")
    if normalized_status:
        clauses.append("status = ?")
        params.append(normalized_status)
    params.append(limit)
    with manager.telegram_alerts._connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id,timestamp,camera_name,label,confidence,alert_level,image_path,status,event_state,event_confirmed_at,acknowledged_at,dismissed_at,dismissal_reason,action_taken,operator_note FROM alert_logs WHERE " + " AND ".join(clauses) + " ORDER BY id DESC LIMIT ?", params).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/events/{event_id}/snapshot")
def event_snapshot(event_id: int, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    with manager.telegram_alerts._connect() as conn:
        row = conn.execute("SELECT image_path FROM alert_logs WHERE id = ?", (event_id,)).fetchone()
    if not row or not row[0]:
        raise HTTPException(404, detail={"code": "SNAPSHOT_NOT_FOUND", "message": "Event snapshot is unavailable"})
    root = Path(manager.alert_manager.alerts_dir).resolve()
    candidate = Path(row[0]).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(404, detail={"code": "SNAPSHOT_NOT_FOUND", "message": "Event snapshot is unavailable"})
    if not candidate.is_file():
        raise HTTPException(404, detail={"code": "SNAPSHOT_NOT_FOUND", "message": "Event snapshot is unavailable"})
    return FileResponse(candidate, media_type="image/jpeg", filename=f"event-{event_id}.jpg")


@app.post("/api/events/{event_id}/acknowledge")
def acknowledge_event(event_id: int, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    with manager.telegram_alerts._connect() as conn:
        if not conn.execute("SELECT 1 FROM alert_logs WHERE id = ?", (event_id,)).fetchone():
            raise HTTPException(404, detail={"code": "EVENT_NOT_FOUND", "message": "Unknown event id"})
    if not manager.alert_manager.update_alert_action(event_id, action_taken="Acknowledged via API", operator_note="", status="ACKNOWLEDGED"):
        raise HTTPException(404, detail={"code": "EVENT_NOT_FOUND", "message": "Event could not be acknowledged"})
    return {"event_id": event_id, "status": "ACKNOWLEDGED"}


@app.post("/api/events/{event_id}/dismiss")
def dismiss_event(event_id: int, payload: DismissInput, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    with manager.telegram_alerts._connect() as conn:
        if not conn.execute("SELECT 1 FROM alert_logs WHERE id = ?", (event_id,)).fetchone():
            raise HTTPException(404, detail={"code": "EVENT_NOT_FOUND", "message": "Unknown event id"})
    if not manager.alert_manager.update_alert_action(event_id, action_taken="Dismissed", operator_note=payload.reason, status="DISMISSED"):
        raise HTTPException(404, detail={"code": "EVENT_NOT_FOUND", "message": "Event could not be dismissed"})
    return {"event_id": event_id, "status": "DISMISSED"}


@app.get("/api/performance")
def performance(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    snapshots = manager.get_all_snapshots()
    result = manager.get_system_metrics(snapshots)
    result["model"] = manager.detector.get_model_info()
    result["camera_metrics"] = [{
        "camera_id": s["cam_id"],
        "capture_fps": s["capture_fps"],
        "inference_fps": s["inference_fps"],
        "display_fps": s["display_fps"],
        "inference_latency_ms": s["infer_ms"],
        "end_to_end_latency_ms": s["end_to_end_latency_ms"],
        "captured_frames": s["captured_frames"],
        "processed_frames": s["processed_frames"],
        "dropped_frames": s["dropped_frames"],
        "queue_depth": s["queue_depth"],
        "displayed_frames": s["displayed_frames"],
        "status": s["status"],
    } for s in snapshots]
    return result


@app.get("/api/telegram/status")
def telegram_status(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    dispatcher = manager.telegram_alerts
    with dispatcher._connect() as conn:
        row = conn.execute("""SELECT
            SUM(CASE WHEN status IN ('QUEUED','SENDING','RETRY_PENDING') THEN 1 ELSE 0 END),
            SUM(CASE WHEN status IN ('FAILED','AMBIGUOUS') THEN 1 ELSE 0 END),
            MAX(telegram_acknowledged_at),
            MAX(CASE WHEN status IN ('FAILED','AMBIGUOUS') THEN attempt_started_at END)
            FROM alert_deliveries""").fetchone()
        latest_test = conn.execute("""SELECT status,acknowledged_at,error_category,used_service_credentials
            FROM telegram_test_deliveries ORDER BY test_id DESC LIMIT 1""").fetchone()
    configured = bool(dispatcher.token and dispatcher.chat_id)
    enabled = bool(dispatcher.enabled)
    latest_ack = row[2]
    if not configured:
        state = "NOT CONFIGURED"
    elif not enabled:
        state = "DISABLED"
    elif row[3] and (not latest_ack or row[3] > latest_ack):
        state = "ERROR"
    elif latest_ack:
        state = "READY"
    else:
        state = "DEGRADED"
    return {"state": state, "enabled": enabled, "credentials_configured": configured,
            "worker_running": bool(dispatcher.thread and dispatcher.thread.is_alive()),
            "service_credentials_tested": bool(dispatcher.service_credentials_tested),
            "last_test": ({"status": latest_test[0], "acknowledged_at": latest_test[1],
                           "error_category": latest_test[2],
                           "used_service_credentials": bool(latest_test[3])} if latest_test else None),
            "pending_count": int(row[0] or 0), "failed_count": int(row[1] or 0),
            "last_api_acknowledged_at": latest_ack}


@app.post("/api/telegram/test")
def telegram_test(payload: Any = Body(default=None), manager: MultiCameraManager = Depends(_manager),
                  _: None = Depends(_authorized), __: None = Depends(_same_dashboard_instance)):
    global _telegram_last_test_at
    if isinstance(payload, TelegramTestInput):
        payload = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise HTTPException(422, detail={"code": "INVALID_TEST_REQUEST", "message": "Test credentials must be provided as text fields"})
    raw_token, raw_chat_id = payload.get("token"), payload.get("chat_id")
    if raw_token is not None and not isinstance(raw_token, str):
        raise HTTPException(422, detail={"code": "INVALID_TOKEN_FORMAT", "message": "Bot token format is invalid"})
    if raw_chat_id is not None and not isinstance(raw_chat_id, str):
        raise HTTPException(422, detail={"code": "INVALID_CHAT_ID", "message": "Chat ID must be a numeric ID or a public @channel username"})
    token = (raw_token or manager.telegram_alerts.token).strip()
    chat_id = (raw_chat_id or manager.telegram_alerts.chat_id).strip()
    if not token:
        raise HTTPException(422, detail={"code": "MISSING_TOKEN", "message": "Set the bot token in the password field or service environment"})
    if len(token) > 256 or not re.fullmatch(r"[0-9]{5,}:[A-Za-z0-9_-]{20,}", token):
        raise HTTPException(422, detail={"code": "INVALID_TOKEN_FORMAT", "message": "Bot token format is invalid"})
    if not chat_id:
        raise HTTPException(422, detail={"code": "MISSING_CHAT_ID", "message": "Set the Chat ID in the form or service environment"})
    if len(chat_id) > 128 or not re.fullmatch(r"-?[0-9]{1,24}|@[A-Za-z0-9_]{5,32}", chat_id):
        raise HTTPException(422, detail={"code": "INVALID_CHAT_ID", "message": "Chat ID must be a numeric ID or a public @channel username"})
    now = time.monotonic()
    global_lock = _telegram_test_lock
    with global_lock:
        if now - _telegram_last_test_at < 30.0:
            retry_after = max(1, int(30.0 - (now - _telegram_last_test_at)))
            raise HTTPException(429, detail={"code": "TEST_RATE_LIMITED", "retry_after_seconds": retry_after,
                                             "message": "Wait before sending another test message"})
        _telegram_last_test_at = now
    requested_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    used_service_credentials = int(
        token == manager.telegram_alerts.token and chat_id == manager.telegram_alerts.chat_id
    )
    with manager.telegram_alerts._connect() as conn:
        cursor = conn.execute("INSERT INTO telegram_test_deliveries(requested_at,status,used_service_credentials) VALUES(?, 'SENDING', ?)",
                              (requested_at, used_service_credentials))
        test_id = cursor.lastrowid
    try:
        ack_at = send_test_message(token, chat_id, timeout=manager.telegram_alerts.timeout)
    except TelegramTestError as exc:
        with manager.telegram_alerts._connect() as conn:
            conn.execute("UPDATE telegram_test_deliveries SET status=?,error_category=? WHERE test_id=?",
                         ("AMBIGUOUS" if exc.ambiguous else "FAILED", exc.category, test_id))
        raise HTTPException(502, detail={"code": exc.category, "message": str(exc), "test_id": test_id}) from None
    with manager.telegram_alerts._connect() as conn:
        conn.execute("UPDATE telegram_test_deliveries SET status='DELIVERED',acknowledged_at=? WHERE test_id=?",
                     (ack_at, test_id))
    if used_service_credentials:
        manager.telegram_alerts.service_credentials_tested = True
        if manager.telegram_alerts.enabled:
            try:
                manager.telegram_alerts._start_worker()
            except RuntimeError as exc:
                raise HTTPException(503, detail={"code": "DISPATCHER_UNAVAILABLE",
                                                 "message": str(exc)}) from None
    return {"test_id": test_id, "status": "DELIVERED", "acknowledged_at": ack_at,
            "message": "Telegram acknowledged the labeled test message; it is not a fire event."}


@app.post("/api/telegram/enable")
def telegram_enable(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized),
                    __: None = Depends(_same_dashboard_instance)):
    try:
        manager.telegram_alerts.set_enabled(True)
    except ValueError as exc:
        raise HTTPException(409, detail={"code": "CREDENTIALS_NOT_CONFIGURED", "message": str(exc)}) from None
    except RuntimeError as exc:
        raise HTTPException(503, detail={"code": "DISPATCHER_UNAVAILABLE", "message": str(exc)}) from None
    return {"enabled": True, "credentials_configured": bool(manager.telegram_alerts.token and manager.telegram_alerts.chat_id)}


@app.post("/api/telegram/disable")
def telegram_disable(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized),
                     __: None = Depends(_same_dashboard_instance)):
    manager.telegram_alerts.set_enabled(False)
    return {"enabled": False}


@app.get("/api/telegram/deliveries")
def telegram_deliveries(limit: int = Query(100, ge=1, le=1000), status: str = "",
                        manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    states = {"QUEUED", "SENDING", "RETRY_PENDING", "DELIVERED", "FAILED", "AMBIGUOUS"}
    normalized = status.upper().strip()
    if normalized and normalized not in states:
        raise HTTPException(422, detail={"code": "INVALID_STATUS", "message": "Unsupported Telegram delivery status"})
    query = """SELECT d.event_id,e.camera_name,e.label,d.event_confirmed_at,d.queued_at,
        d.attempt_started_at,d.telegram_acknowledged_at,d.status,d.attempt_count,d.error_category,d.error_message
        FROM alert_deliveries d JOIN alert_logs e ON e.id=d.event_id"""
    params: list[Any] = []
    if normalized:
        query += " WHERE d.status=?"
        params.append(normalized)
    query += " ORDER BY d.queued_at DESC LIMIT ?"
    params.append(limit)
    with manager.telegram_alerts._connect() as conn:
        rows = conn.execute(query, params).fetchall()
    columns = ("event_id", "camera", "event_type", "event_confirmed_at", "queued_at",
               "attempt_started_at", "acknowledged_at", "status", "attempt_count",
               "error_category", "error_message")
    result = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            confirmed = datetime.fromisoformat(item["event_confirmed_at"])
            acknowledged = datetime.fromisoformat(item["acknowledged_at"]) if item["acknowledged_at"] else None
            item["latency_ms"] = round((acknowledged - confirmed).total_seconds() * 1000, 3) if acknowledged else None
        except (TypeError, ValueError):
            item["latency_ms"] = None
        result.append(item)
    return result


@app.get("/api/alerts/status")
def alert_status(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    status = telegram_status(manager)
    return {"telegram_configured": status["credentials_configured"], "enabled": status["enabled"],
            "delivery": "active" if manager.telegram_alerts.configured else "disabled", **status}


@app.get("/api/alerts/deliveries")
def alert_deliveries(limit: int = Query(100, ge=1, le=1000), manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    with manager.telegram_alerts._connect() as conn:
        rows = conn.execute("""SELECT delivery_id,event_id,channel,status,queued_at,event_confirmed_at,
            attempt_started_at,telegram_acknowledged_at,next_attempt_at,attempt_count,error_category,error_message
            FROM alert_deliveries ORDER BY queued_at DESC LIMIT ?""", (limit,)).fetchall()
    columns = ("delivery_id", "event_id", "channel", "status", "queued_at", "event_confirmed_at",
               "attempt_started_at", "telegram_acknowledged_at", "next_attempt_at", "attempt_count",
               "error_category", "error_message")
    return [dict(zip(columns, row)) for row in rows]


@app.websocket("/ws/cameras")
async def camera_updates(websocket: WebSocket):
    origin = websocket.headers.get("origin", "").rstrip("/")
    allowed_origins = _allowed_websocket_origins()
    if not _origin_is_allowed(origin, allowed_origins):
        await websocket.close(code=4403)
        return
    ticket = websocket.query_params.get("ticket", "")
    if not _consume_ticket(websocket.app, ticket):
        await websocket.close(code=4401)
        return
    manager = getattr(websocket.app.state, "manager", None)
    if manager is None:
        await websocket.close(code=1013)
        return
    await websocket.accept()
    client_id = secrets.token_urlsafe(12)
    try:
        while True:
            snapshots = manager.get_all_snapshots()
            cameras = []
            for snapshot in snapshots:
                item = _public_camera(snapshot)
                item["frame_id"] = manager.workers[snapshot["cam_id"]].frame_count
                item["jpeg_b64"] = snapshot.get("jpeg_b64", "")
                cameras.append(item)
            await websocket.send_json({"timestamp": time.time(), "cameras": cameras})
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            displayed = message.get("displayed", {}) if isinstance(message, dict) else {}
            if isinstance(displayed, dict):
                for camera_id, frame_id in displayed.items():
                    worker = manager.workers.get(str(camera_id))
                    if worker is not None:
                        try:
                            worker.record_displayed_frame(client_id, int(frame_id))
                        except (TypeError, ValueError):
                            continue
    except WebSocketDisconnect:
        pass
    finally:
        for worker in manager.workers.values():
            worker.remove_display_client(client_id)

