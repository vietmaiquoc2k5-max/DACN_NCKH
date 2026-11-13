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
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.camera_manager import MultiCameraManager
from src.config import PROJECT_ROOT, mask_rtsp_credentials


class CameraInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_type: str = Field(pattern="^(WEBCAM|VIDEO_TEST|IP_CAMERA)$")
    source: str = Field(min_length=1, max_length=2048)
    webcam_index: int = Field(default=0, ge=0, le=64)
    enabled: bool = True
    auto_reconnect: bool = True
    inference_fps: int = Field(default=8, ge=1, le=60)
    exclusion_zones: list[tuple[float, float, float, float]] = Field(default_factory=list, max_length=64)


class DismissInput(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


def _manager(request: Request) -> MultiCameraManager:
    manager = getattr(request.app.state, "manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail={"code": "NOT_READY", "message": "Camera service is starting"})
    return manager


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
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        raise HTTPException(422, detail={"code": "INVALID_RTSP", "message": "Malformed RTSP URL"})
    is_env_reference = False
    if not parsed.scheme and value.startswith("env:"):
        is_env_reference = True
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
    with sqlite3.connect(db_path, timeout=10) as conn:
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
            auto_reconnect,inference_fps,exclusion_zones,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name,source_type=excluded.source_type,
            source=excluded.source,webcam_index=excluded.webcam_index,enabled=excluded.enabled,
            auto_reconnect=excluded.auto_reconnect,inference_fps=excluded.inference_fps,
            exclusion_zones=excluded.exclusion_zones,updated_at=CURRENT_TIMESTAMP""",
            (camera_id, input_data.name, input_data.source_type, input_data.source,
             input_data.webcam_index, int(input_data.enabled), int(input_data.auto_reconnect),
             input_data.inference_fps, json.dumps(input_data.exclusion_zones)))


def _restore_camera_configs(manager: MultiCameraManager):
    with manager.telegram_alerts._connect() as conn:
        rows = conn.execute("SELECT camera_id,name,source_type,source,webcam_index,enabled,auto_reconnect,inference_fps,exclusion_zones FROM cameras ORDER BY camera_id").fetchall()
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
                                                 auto_start=False, inference_fps=int(fps))
                if restored_id != camera_id:
                    manager.add_log(f"Saved camera id {camera_id} restored as {restored_id}")
        except Exception as exc:
            manager.add_log(f"Camera configuration {camera_id} was not restored ({type(exc).__name__})")


def _public_camera(snapshot: dict[str, Any]) -> dict[str, Any]:
    # Frames are available only through the live WebSocket to keep list payloads small.
    return {key: value for key, value in snapshot.items() if key not in ("frame_rgb", "preview_rgb", "jpeg_b64")}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    db_path = os.getenv("FIREWATCH_DB_PATH", "alerts.db")
    model_path = os.getenv("FIREWATCH_MODEL_PATH", "best.pt")
    manager = MultiCameraManager(db_path=db_path, model_path=model_path, auto_assign_videos=False)
    _initialize_camera_config(db_path)
    _restore_camera_configs(manager)
    # Preserve manual-connect behavior even if future defaults change.
    manager.stop_monitoring()
    app.state.manager = manager
    try:
        yield
    finally:
        manager.stop_monitoring()
        manager.telegram_alerts.close()
        app.state.manager = None


app = FastAPI(title="Firewatch Week 7 API", version="1.0.0", lifespan=_lifespan)


@app.get("/health")
def health(request: Request):
    manager = getattr(request.app.state, "manager", None)
    return {"status": "ok" if manager else "starting", "model": manager.detector.get_model_info() if manager else None}


@app.get("/api/cameras")
def list_cameras(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    return [_public_camera(item) for item in manager.get_all_snapshots()]


@app.post("/api/cameras", status_code=201)
def add_camera(payload: CameraInput, manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    source = _validate_source(payload.source_type, payload.source, payload.webcam_index)
    cam_id = manager.add_camera(
        name=payload.name, source=source, source_type=payload.source_type,
        webcam_index=payload.webcam_index, enabled=payload.enabled,
        auto_reconnect=payload.auto_reconnect, exclusion_zones=payload.exclusion_zones,
        auto_start=False, inference_fps=payload.inference_fps,
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
                                 auto_restart=False)
    manager.update_camera_roi(camera_id, payload.exclusion_zones)
    manager.workers[camera_id].config.inference_fps = payload.inference_fps
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
        rows = conn.execute("SELECT id,timestamp,camera_name,label,confidence,alert_level,image_path,status,action_taken,operator_note FROM alert_logs WHERE " + " AND ".join(clauses) + " ORDER BY id DESC LIMIT ?", params).fetchall()
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
    result["camera_metrics"] = [{"camera_id": s["cam_id"], "capture_fps": s["capture_fps"], "inference_fps": s["inference_fps"], "inference_latency_ms": s["infer_ms"], "status": s["status"]} for s in snapshots]
    return result


@app.get("/api/alerts/status")
def alert_status(manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled = os.getenv("ALERTS_ENABLED", "false").lower() in ("1", "true", "yes", "on")
    return {"telegram_configured": bool(token and chat), "enabled": enabled,
            "delivery": "active" if manager.telegram_alerts.configured else "disabled"}


@app.get("/api/alerts/deliveries")
def alert_deliveries(limit: int = Query(100, ge=1, le=1000), manager: MultiCameraManager = Depends(_manager), _: None = Depends(_authorized)):
    with manager.telegram_alerts._connect() as conn:
        rows = conn.execute("SELECT delivery_id,event_id,channel,status,queued_at,sent_at,confirmed_at,error_message FROM alert_deliveries ORDER BY queued_at DESC LIMIT ?", (limit,)).fetchall()
    columns = ("delivery_id", "event_id", "channel", "status", "queued_at", "sent_at", "confirmed_at", "error_message")
    return [dict(zip(columns, row)) for row in rows]


@app.websocket("/ws/cameras")
async def camera_updates(websocket: WebSocket):
    expected = os.getenv("FIREWATCH_API_TOKEN", "")
    supplied = websocket.query_params.get("token", "")
    if not expected or not secrets.compare_digest(supplied, expected):
        await websocket.close(code=4401)
        return
    manager = getattr(websocket.app.state, "manager", None)
    if manager is None:
        await websocket.close(code=1013)
        return
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"timestamp": time.time(), "cameras": [_public_camera(s) for s in manager.get_all_snapshots()]})
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return

