"""FIREWATCH – INTELLIGENT MONITORING STATION (V2).

Hệ thống Giám sát, Phát hiện & Cảnh báo Cháy/Khói Thời Gian Thực bằng Computer Vision.
Thiết kế theo chuẩn Industrial Command Center (Dark Professional Graphite/Slate):
- Cơ chế vận hành MANUAL START bắt buộc: Khi mở ứng dụng hoặc refresh trang, tất cả camera ở trạng thái
  STOPPED (Đang chờ kết nối), không tự mở camera/RTSP, không tạo luồng worker chạy ngầm, không tốn GPU.
- Chỉ khi nhấn [ ▶ BẮT ĐẦU GIÁM SÁT ], hệ thống mới khởi tạo luồng xử lý cho đúng các camera được chọn.
- Khi nhấn [ ■ DỪNG GIÁM SÁT ], hệ thống dừng toàn bộ inference, giải phóng VideoCapture/RTSP và đưa về STOPPED.
- Khi thay đổi số lượng/danh sách camera lúc đang chạy, hệ thống thông báo có thay đổi cấu hình và cung cấp nút
  [ ✓ ÁP DỤNG CẤU HÌNH MỚI ] để chỉ bật/tắt những camera thay đổi mà không reset camera đang chạy.
- Mọi HTML đều đi qua `render_html()` và `esc()` để tuyệt đối không bao giờ xảy ra lỗi hiển thị HTML thô.
"""
import base64
import html
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SOURCE_TYPES,
    get_available_model_choices,
    get_available_test_videos,
    mask_rtsp_credentials,
)
from src.camera_manager import MultiCameraManager, list_system_cameras

# ==============================================================================
# 1. CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="FIREWATCH – INTELLIGENT MONITORING STATION",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. HÀM CHUẨN HÓA HTML AN TOÀN (CHỐNG LỖI HIỂN THỊ HTML THÔ 100%)
# ==============================================================================
def esc(value: Any) -> str:
    """Escape mọi dữ liệu văn bản động trước khi đưa vào HTML template."""
    return html.escape(str(value if value is not None else ""), quote=True)


def render_html(raw_html: str) -> None:
    """Loại bỏ khoảng trắng đầu dòng và dòng trống để CommonMark không biến thẻ HTML thành code-block."""
    cleaned = "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ==============================================================================
# 3. KHỞI TẠO SINGLETON CAMERA MANAGER & CƠ CHẾ MANUAL START (STOPPED MẶC ĐỊNH)
# ==============================================================================
def force_stop_audio_alarm() -> None:
    """Ngắt cưỡng bức mọi luồng âm thanh còi báo động (cả pygame.mixer và winsound)."""
    try:
        import pygame

        if pygame and pygame.mixer and pygame.mixer.get_init():
            pygame.mixer.music.stop()
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass


@st.cache_resource
def get_firewatch_station() -> MultiCameraManager:
    """Khởi tạo bộ điều phối trung tâm (Shared YOLO Detector + SQLite EventDatabase).
    Mặc định toàn bộ camera ở trạng thái STOPPED và KHÔNG tự động gắn sẵn file video mẫu (`auto_assign_videos=False`).
    """
    mgr = MultiCameraManager(db_path="alerts.db", model_path="best.pt", auto_assign_videos=False)
    mgr.stop_monitoring()
    mgr._fw_version = "3.0"
    return mgr


cam_system = get_firewatch_station()

# Tự động nâng cấp Singleton nếu Streamlit Server đang giữ instance cũ trong bộ nhớ cache
if getattr(cam_system, "_fw_version", None) != "3.0" or not hasattr(cam_system, "assign_test_video_to_camera"):
    try:
        cam_system.alert_manager.mute_alarm = True
        cam_system.alert_manager.alarm_playing = False
        cam_system.alert_manager.stop_alarm()
    except Exception:
        pass
    force_stop_audio_alarm()
    try:
        cam_system.stop_all()
    except Exception:
        pass
    st.cache_resource.clear()
    import importlib
    import alert_manager as _am_mod
    import src.config as _cfg_mod
    import src.database as _db_mod
    import src.camera_manager as _cm_mod

    importlib.reload(_am_mod)
    importlib.reload(_cfg_mod)
    importlib.reload(_db_mod)
    importlib.reload(_cm_mod)
    MultiCameraManager = _cm_mod.MultiCameraManager
    cam_system = get_firewatch_station()
    if hasattr(cam_system, "clear_all_video_sources"):
        cam_system.clear_all_video_sources()
    cam_system._fw_version = "3.0"

# Khi mới mở ứng dụng hoặc khi người dùng Refresh trang (F5 tạo session mới):
# BẮT BUỘC đưa toàn bộ camera về trạng thái STOPPED (Manual Start)
default_ids = list(getattr(cam_system, "default_camera_ids", ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]))
if "fw_initialized" not in st.session_state:
    if hasattr(cam_system, "stop_monitoring"):
        cam_system.stop_monitoring()
    else:
        cam_system.stop_all()
    cam_system.set_active_camera_selection(default_ids, auto_start=False)
    st.session_state.fw_initialized = True

# Đảm bảo mọi khóa session_state luôn tồn tại kể cả khi hot-reload trên tab trình duyệt cũ
st.session_state.setdefault("is_monitoring", False)
st.session_state.setdefault("monitoring_started_ts", 0.0)
st.session_state.setdefault("nav_page", "Tổng quan")
st.session_state.setdefault("cam_count_preset", len(default_ids) if len(default_ids) in (1, 2, 4, 6, 9) else 4)
st.session_state.setdefault("selected_cams", list(default_ids))
st.session_state.setdefault("pending_cams", list(default_ids))
st.session_state.setdefault("view_mode", "Lưới")
st.session_state.setdefault("focus_cam_id", default_ids[0] if default_ids else "CAM-01")
st.session_state.setdefault("inspect_cam_id", None)
st.session_state.setdefault("modal_event_id", None)
st.session_state.setdefault("alert_tab", "Chờ xử lý")
st.session_state.setdefault("alert_page", 1)
st.session_state.setdefault("perf_preset", getattr(cam_system, "performance_preset", "Balanced"))
st.session_state.setdefault("hero_ack_state", None)
st.session_state.setdefault("last_unack_top_id", None)


def handle_alert_acknowledge(
    event_id: int,
    action_taken: str = "Đã xác nhận cháy thật",
    note: str = "",
    is_dismiss: bool = False,
    clear_all_pending: bool = False,
) -> None:
    """Xác nhận hoặc bỏ qua sự kiện cảnh báo an toàn 100%, ngắt ngay âm báo còi hú."""
    force_stop_audio_alarm()
    try:
        cam_system.alert_manager.alarm_silenced_by_ack = True
        cam_system.alert_manager.alarm_playing = False
        cam_system.alert_manager.stop_alarm()
    except Exception:
        pass

    if hasattr(cam_system, "acknowledge_event"):
        try:
            cam_system.acknowledge_event(
                int(event_id),
                action_taken=action_taken,
                note=note,
                is_dismiss=is_dismiss,
                clear_all_pending=clear_all_pending,
            )
        except TypeError:
            cam_system.acknowledge_event(
                int(event_id),
                action_taken=action_taken,
                note=note,
                is_dismiss=is_dismiss,
            )
    else:
        if is_dismiss:
            cam_system.event_db.dismiss_event(int(event_id), note=note or "Đã bỏ qua (Báo động giả)")
        else:
            cam_system.event_db.confirm_event(int(event_id), action_taken=action_taken, note=note)

    if clear_all_pending:
        try:
            import sqlite3

            act_str = "Báo động giả (Đã bỏ qua)" if is_dismiss else action_taken
            note_str = note or ("Đã bỏ qua (Báo động giả)" if is_dismiss else "Xác nhận từ Trung tâm cảnh báo")
            conn = sqlite3.connect(cam_system.event_db.db_path)
            cur = conn.cursor()
            cur.execute(
                "UPDATE alert_logs SET status = 'ACKNOWLEDGED', action_taken = ?, operator_note = ? WHERE status != 'ACKNOWLEDGED'",
                (act_str, note_str),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    for cid, w in getattr(cam_system, "workers", {}).items():
        if getattr(w, "alert_level", "SAFE") in ("FIRE", "SUSPECT"):
            if hasattr(cam_system.alert_manager, "acknowledged_cameras"):
                cam_system.alert_manager.acknowledged_cameras.add(cid)

    force_stop_audio_alarm()
    st.session_state.hero_ack_state = "DISMISSED" if is_dismiss else "CONFIRMED"


# ==============================================================================
# 4. HỆ THỐNG CSS INDUSTRIAL COMMAND CENTER (DARK PROFESSIONAL)
# ==============================================================================
render_html(
    """
<style>
    :root {
        --bg-charcoal: #0B0E14;
        --bg-graphite: #11151E;
        --bg-slate: #171C28;
        --bg-slate-2: #1E2534;
        --border-subtle: #262F40;
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --color-fire: #EF4444;
        --color-suspect: #F97316;
        --color-warning: #EAB308;
        --color-safe: #10B981;
        --color-stopped: #64748B;
    }

    .stApp {
        background-color: var(--bg-charcoal) !important;
        color: var(--text-primary) !important;
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
        font-size: 14.5px;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.4rem !important;
    }
    .block-container {
        padding-top: 0.35rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        max-width: 100% !important;
    }

    /* Sidebar thu gọn, không có khoảng trống thừa */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-graphite) !important;
        border-right: 1px solid var(--border-subtle) !important;
        width: 236px !important;
        min-width: 236px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.6rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    /* Header Bar */
    .fw-header-bar {
        background: linear-gradient(90deg, #11151E 0%, #161C28 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 7px;
        padding: 8px 16px;
        margin-bottom: 10px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }
    .fw-header-status {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .fw-status-led {
        width: 13px;
        height: 13px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px currentColor;
        flex-shrink: 0;
    }
    .fw-kpi-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
    }
    .fw-kpi-box {
        background-color: var(--bg-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 5px 12px;
        display: flex;
        align-items: center;
        gap: 9px;
    }
    .fw-kpi-box.kpi-alert {
        border-color: rgba(239, 68, 68, 0.55);
        background-color: rgba(239, 68, 68, 0.10);
    }
    .fw-kpi-num {
        font-family: "JetBrains Mono", "Consolas", monospace;
        font-size: 14.5px;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.15;
    }
    .fw-kpi-sub {
        font-size: 11px;
        color: var(--text-secondary);
        line-height: 1.15;
    }

    /* Hệ thống nút bấm chuẩn Firewatch Industrial Orange & Dark Slate */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #F97316 0%, #EA580C 100%) !important;
        border: 1px solid #FB923C !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.30) !important;
        padding: 0.3rem 0.65rem !important;
        min-height: 34px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #FB923C 0%, #F97316 100%) !important;
        border-color: #FDBA74 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #181F2E !important;
        border: 1px solid #2A3449 !important;
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.3rem 0.65rem !important;
        min-height: 34px !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #F97316 !important;
        color: #FFFFFF !important;
        background-color: #1F293D !important;
    }

    /* Sidebar Navigation chuẩn ảnh mẫu (ẩn chấm tròn radio, highlight cam) */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 4px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: transparent !important;
        border-left: 3px solid transparent !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        margin: 0 !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.04) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(234, 88, 12, 0.72) 0%, rgba(234, 88, 12, 0.08) 100%) !important;
        border-left: 3px solid #F97316 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    /* Khung Camera chuẩn ảnh thiết kế FIREWATCH V2 */
    .fw-cam-frame {
        background-color: #121722;
        border: 1px solid #2A3448;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 6px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    .fw-cam-frame.state-fire {
        border: 2px solid var(--color-fire);
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.32);
    }
    .fw-cam-frame.state-suspect {
        border: 1.5px solid var(--color-suspect);
    }
    .fw-cam-frame.state-running {
        border: 1px solid rgba(16, 185, 129, 0.50);
    }
    .fw-cam-frame.state-stopped {
        border: 1px solid #2A3448;
    }
    .fw-cam-frame.state-error {
        border: 1.5px solid var(--color-warning);
    }

    .fw-cam-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 12px;
        background: linear-gradient(90deg, #101520 0%, #151C2A 100%);
        border-bottom: 1px solid #232C3E;
        font-size: 13px;
        font-weight: 700;
    }
    .fw-cam-label {
        display: flex;
        align-items: center;
        gap: 7px;
        color: #F8FAFC;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .fw-cam-icon-badge {
        background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
        color: #FFFFFF;
        width: 20px;
        height: 16px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        flex-shrink: 0;
    }

    /* Nhãn trạng thái kết nối */
    .fw-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }
    .fw-pill-stopped {
        background: rgba(16, 185, 129, 0.14);
        color: #A7F3D0;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .fw-pill-connecting {
        background: rgba(234, 179, 8, 0.16);
        color: #FACC15;
        border: 1px solid rgba(234, 179, 8, 0.40);
    }
    .fw-pill-running {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .fw-pill-fire {
        background: rgba(239, 68, 68, 0.25);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.65);
    }
    .fw-pill-suspect {
        background: rgba(249, 115, 22, 0.22);
        color: #FDBA74;
        border: 1px solid rgba(249, 115, 22, 0.55);
    }
    .fw-pill-error {
        background: rgba(239, 68, 68, 0.16);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.40);
    }

    /* Video trực tiếp giữ chuẩn tỷ lệ 16:9 */
    .fw-cam-video-box {
        width: 100%;
        background-color: #07090E;
        position: relative;
    }
    .fw-cam-video-el {
        width: 100%;
        aspect-ratio: 16 / 9;
        max-height: 215px;
        object-fit: cover;
        display: block;
    }

    /* Màn hình chờ kết nối thủ công (STOPPED) chuẩn ảnh thiết kế FIREWATCH V2 */
    .fw-cam-standby-screen {
        width: 100%;
        aspect-ratio: 16 / 9;
        max-height: 215px;
        background: radial-gradient(circle at 50% 38%, #222B3D 0%, #131824 72%, #0E121B 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 14px;
        position: relative;
        overflow: hidden;
    }
    .fw-standby-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #E2E8F0;
        margin-top: 6px;
        z-index: 2;
    }
    .fw-standby-sub {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 2px;
        z-index: 2;
    }
    /* 4 góc ngắm camera (Corner Brackets) */
    .fw-corner-tl, .fw-corner-tr, .fw-corner-bl, .fw-corner-br {
        position: absolute;
        width: 10px;
        height: 10px;
        border-color: rgba(148, 163, 184, 0.35);
        border-style: solid;
        pointer-events: none;
        z-index: 3;
    }
    .fw-corner-tl { top: 8px; left: 8px; border-width: 1.5px 0 0 1.5px; }
    .fw-corner-tr { top: 8px; right: 8px; border-width: 1.5px 1.5px 0 0; }
    .fw-corner-bl { bottom: 8px; left: 8px; border-width: 0 0 1.5px 1.5px; }
    .fw-corner-br { bottom: 8px; right: 8px; border-width: 0 1.5px 1.5px 0; }

    /* Màn hình lỗi kết nối nhỏ gọn */
    .fw-cam-error-screen {
        background: linear-gradient(180deg, #1A161F 0%, #11141C 100%);
        padding: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border-bottom: 1px solid var(--border-subtle);
    }

    .fw-cam-bottom {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 12px;
        background-color: var(--bg-slate);
        font-size: 12px;
        color: var(--text-secondary);
    }

    /* Panel Cảnh Báo Khẩn Cấp bên phải */
    .fw-alert-hero {
        background: linear-gradient(180deg, rgba(153, 27, 27, 0.35) 0%, #141923 100%);
        border: 1.5px solid var(--color-fire);
        border-radius: 7px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .fw-alert-hero.suspect {
        background: linear-gradient(180deg, rgba(154, 52, 18, 0.32) 0%, #141923 100%);
        border-color: var(--color-suspect);
    }
    .fw-alert-hero-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13.5px;
        font-weight: 800;
        color: #FCA5A5;
        padding-bottom: 6px;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(239, 68, 68, 0.28);
    }
    .fw-alert-img-wrap {
        width: 100%;
        background-color: #07090E;
        border: 1px solid var(--border-subtle);
        border-radius: 5px;
        overflow: hidden;
        position: relative;
        text-align: center;
        margin-bottom: 8px;
    }
    .fw-alert-img {
        width: 100%;
        height: 155px;
        object-fit: contain;
        display: block;
        margin: 0 auto;
    }
    .fw-alert-badge-overlay {
        position: absolute;
        top: 8px;
        right: 8px;
        background: rgba(239, 68, 68, 0.92);
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 800;
        padding: 2px 9px;
        border-radius: 4px;
    }
    .fw-recent-row {
        background-color: var(--bg-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 7px 10px;
        margin-bottom: 6px;
        display: grid;
        grid-template-columns: 82px 1fr;
        gap: 10px;
        align-items: center;
    }
    .fw-recent-img {
        width: 82px;
        height: 54px;
        object-fit: contain;
        background-color: #07090E;
        border: 1px solid var(--border-subtle);
        border-radius: 4px;
        display: block;
    }

    /* Khu vực thống kê & lịch sử phía dưới */
    .fw-panel-card {
        background-color: var(--bg-graphite);
        border: 1px solid var(--border-subtle);
        border-radius: 7px;
        padding: 10px 14px;
        height: 100%;
    }
    .fw-panel-title {
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 6px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .fw-sidebar-card {
        background-color: var(--bg-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 8px 10px;
        margin-bottom: 8px;
    }
</style>
"""
)


# ==============================================================================
# 5. HÀM TIỆN ÍCH CHUYỂN ĐỔI TRẠNG THÁI (STATE MACHINE) & DỮ LIỆU
# ==============================================================================
def get_camera_lifecycle_info(snap: Dict[str, Any], is_monitoring: bool) -> Tuple[str, str, str]:
    """Ánh xạ trạng thái camera theo State Machine V2:
    - STOPPED: Đang chờ kết nối
    - CONNECTING: Đang kết nối
    - RUNNING: Đang hoạt động
    - RECONNECTING: Đang thử kết nối lại
    - ERROR: Kết nối thất bại
    - PAUSED: Đang tạm dừng
    Trả về: (Mã state_machine, Nhãn Tiếng Việt, CSS class của pill).
    """
    raw_status = snap.get("status", "DISCONNECTED")
    lc = snap.get("lifecycle_state", "STOPPED")

    if raw_status == "CONNECTED":
        return "RUNNING", "Đang hoạt động", "fw-pill-running"
    if raw_status == "PAUSED" or lc == "PAUSED":
        return "PAUSED", "Đang tạm dừng", "fw-pill-connecting"
    if raw_status == "ERROR" or lc == "ERROR":
        return "ERROR", "Kết nối thất bại", "fw-pill-error"
    if raw_status == "RECONNECTING":
        if lc == "CONNECTING" or snap.get("reconnect_attempts", 0) == 0:
            return "CONNECTING", "Đang kết nối...", "fw-pill-connecting"
        return "RECONNECTING", "Đang kết nối lại...", "fw-pill-connecting"
    if not is_monitoring:
        return "STOPPED", "Đang chờ kết nối", "fw-pill-stopped"
    return "STOPPED", "Chưa kết nối", "fw-pill-stopped"


def translate_detection_label(label: str) -> str:
    """Dịch nhãn phát hiện sang Tiếng Việt."""
    u = str(label or "").upper()
    if u == "FIRE":
        return "Cháy"
    if u == "SMOKE":
        return "Khói"
    return "Bình thường"


def classify_event_category(ev: Dict[str, Any]) -> Tuple[str, str, str]:
    """Phân loại sự kiện cảnh báo theo 5 nhóm của Phần 5:
    - Cảnh báo cháy
    - Cảnh báo khói
    - Cảnh báo nghi ngờ
    - Cảnh báo đã xác nhận
    - Cảnh báo đã bỏ qua
    Trả về: (Nhãn loại phát hiện, Nhãn trạng thái xử lý, Mã màu).
    """
    lbl = str(ev.get("label", "FIRE")).upper()
    lvl = str(ev.get("alert_level", "FIRE")).upper()
    stt = str(ev.get("status", "NEW")).upper()
    act = str(ev.get("action_taken", "")).lower()

    if lvl == "SUSPECT":
        det_type = f"Cảnh báo nghi ngờ ({translate_detection_label(lbl)})"
        color = "#F97316"
    elif "SMOKE" in lbl:
        det_type = "Cảnh báo khói"
        color = "#EAB308"
    else:
        det_type = "Cảnh báo cháy"
        color = "#EF4444"

    if stt == "ACKNOWLEDGED":
        if "bỏ qua" in act or "giả" in act:
            proc_state = "Đã bỏ qua"
        else:
            proc_state = "Đã xác nhận"
    else:
        proc_state = "Chờ xử lý"

    return det_type, proc_state, color


def load_image_b64(img_path: str) -> Optional[str]:
    """Đọc ảnh bằng chứng sang Base64 để hiển thị nguyên tỷ lệ."""
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def select_camera_count(count_n: int) -> None:
    """Cập nhật lựa chọn số lượng camera (1, 2, 4, 6, 9) tức thì, mượt mà và không gây delay."""
    all_ids = [cfg.cam_id for cfg in cam_system.camera_configs]
    chosen = all_ids[: min(count_n, len(all_ids))]
    st.session_state.cam_count_preset = count_n
    st.session_state.pending_cams = list(chosen)
    st.session_state.selected_cams = list(chosen)
    if chosen and st.session_state.focus_cam_id not in chosen:
        st.session_state.focus_cam_id = chosen[0]
    cam_system.set_active_camera_selection(chosen, auto_start=st.session_state.is_monitoring)


# ==============================================================================
# 6. HỘP THOẠI MODAL XEM ẢNH BẰNG CHỨNG LỚN (@st.dialog)
# ==============================================================================
@st.dialog("🚨 CHI TIẾT SỰ KIỆN CẢNH BÁO & ẢNH BẰNG CHỨNG", width="large")
def open_alert_modal(event_id: int) -> None:
    """Hiển thị Modal ảnh bằng chứng lớn (giữ nguyên tỷ lệ, hiển thị bounding box đã vẽ sẵn trên snapshot)."""
    all_evs = cam_system.event_db.query_events(limit=500, include_demo=True)
    ev = next((e for e in all_evs if e["id"] == event_id), None)
    if not ev:
        st.warning(f"Không tìm thấy sự kiện #{event_id}.")
        if st.button("Đóng", use_container_width=True):
            st.session_state.modal_event_id = None
            st.rerun()
        return

    det_type, proc_state, accent_col = classify_event_category(ev)
    conf_pct = round(float(ev.get("confidence", 0.0)) * 100, 1)

    c_img, c_meta = st.columns([1.85, 1.15])
    with c_img:
        img_b64 = load_image_b64(ev.get("image_path", ""))
        if img_b64:
            render_html(
                f"""
                <div style="background:#07090E; border:1px solid #262F40; border-radius:6px; padding:8px; text-align:center;">
                    <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; max-height:490px; object-fit:contain;" alt="Sự kiện #{event_id}" />
                </div>
                """
            )
        else:
            st.info("Sự kiện này không có ảnh chụp lưu trữ.")

    with c_meta:
        render_html(
            f"""
            <div style="background:#11151E; border:1px solid #262F40; border-radius:6px; padding:12px;">
                <div style="font-size:15px; font-weight:800; color:{accent_col}; margin-bottom:8px;">
                    SỰ KIỆN #{event_id} — {esc(det_type.upper())}
                </div>
                <div style="font-size:13.5px; line-height:1.75; color:#E2E8F0;">
                    <div>• <b>Mã sự kiện:</b> <code>#{event_id}</code></div>
                    <div>• <b>Camera nguồn:</b> <b>{esc(ev.get('camera_name'))}</b></div>
                    <div>• <b>Thời gian:</b> <code>{esc(ev.get('timestamp'))}</code></div>
                    <div>• <b>Phân loại:</b> <b>{esc(det_type)}</b></div>
                    <div>• <b>Độ tin cậy:</b> <b>{conf_pct:.1f}%</b></div>
                    <div>• <b>Trạng thái xử lý:</b> <b>{esc(proc_state)}</b></div>
                    <div>• <b>Chi tiết xử lý:</b> {esc(ev.get('action_taken') or 'Chưa xử lý')}</div>
                </div>
            </div>
            """
        )
        note_val = st.text_input(
            "Ghi chú xử lý của cán bộ trực ban:",
            value=ev.get("operator_note") or "",
            key=f"dlg_note_{event_id}",
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✓ Xác nhận cháy thật", key=f"dlg_ack_{event_id}", type="primary", use_container_width=True):
                handle_alert_acknowledge(event_id, action_taken="Đã xác nhận cháy thật", note=note_val, is_dismiss=False, clear_all_pending=True)
                st.session_state.modal_event_id = None
                st.toast("🔕 Đã xác nhận cảnh báo và tắt âm báo còi hú.")
                st.rerun()
        with b2:
            if st.button("✕ Bỏ qua (Báo giả)", key=f"dlg_dis_{event_id}", use_container_width=True):
                handle_alert_acknowledge(event_id, note=note_val or "Đã bỏ qua (Báo động giả)", is_dismiss=True, clear_all_pending=True)
                st.session_state.modal_event_id = None
                st.toast("🔕 Đã bỏ qua cảnh báo và tắt âm báo còi hú.")
                st.rerun()

        if st.button("Đóng cửa sổ", key=f"dlg_close_{event_id}", use_container_width=True):
            st.session_state.modal_event_id = None
            st.rerun()


# ==============================================================================
# 7. COMPONENT HIỂN THỊ KHUNG CAMERA & ẢNH MẪU MẶC ĐỊNH CHUẨN V2
# ==============================================================================
def clean_camera_display_name(cam_id: str, raw_name: str) -> str:
    """Làm sạch tên hiển thị của camera để khớp định dạng 'CAM-01 – Khu Vực Sảnh Chính'."""
    default_titles = {
        "CAM-01": "Khu Vực Sảnh Chính",
        "CAM-02": "Xưởng Sản Xuất",
        "CAM-03": "Kho Vật Liệu",
        "CAM-04": "Hành Lang Tầng 2",
        "CAM-05": "Trạm Biến Áp",
        "CAM-06": "Khu Vực Nhà Xe",
        "CAM-07": "Cổng Chính",
        "CAM-08": "Sân Thượng",
        "CAM-09": "Hành Lang Kỹ Thuật",
    }
    s = str(raw_name or "").strip()
    if "(" in s:
        s = s.split("(")[0].strip()
    if " - " in s and s.upper().startswith("CAM"):
        s = s.split(" - ", 1)[1].strip()
    if not s or s.upper() == cam_id.upper() or s in ("Kho Hàng A", "Khu Vực Bãi Xe"):
        return default_titles.get(cam_id, s or cam_id)
    return s


@st.cache_data(show_spinner=False)
def get_fallback_alert_thumb_b64() -> str:
    """Trích xuất 1 khung hình thực tế từ kho ảnh cảnh báo hoặc video Việt Nam làm ảnh hiển thị khi DB trống."""
    alerts_dir = PROJECT_ROOT / "alerts"
    if alerts_dir.exists():
        jpgs = sorted(alerts_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        for jp in jpgs:
            try:
                return base64.b64encode(jp.read_bytes()).decode("ascii")
            except Exception:
                pass
    for vid_rel in ("videovietnam/VN_POS_02.mp4", "videovietnam/VN_POS_01.mp4"):
        vp = PROJECT_ROOT / vid_rel
        if vp.exists():
            cap = cv2.VideoCapture(str(vp))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    frame_small = cv2.resize(frame, (480, 270))
                    ok_enc, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                    if ok_enc:
                        return base64.b64encode(buf.tobytes()).decode("ascii")
    return ""


def render_camera_frame(snap: Dict[str, Any], key_prefix: str = "grid", compact: bool = False) -> None:
    """Hiển thị một khung camera liền khối chuẩn ảnh mẫu FIREWATCH V2."""
    cid = snap["cam_id"]
    cname = clean_camera_display_name(cid, snap["name"])
    lvl = snap["alert_level"]
    cap_fps = snap["capture_fps"]
    conf_pct = round(snap["max_confidence"] * 100, 1)
    upd_str = snap["last_update_str"]
    disc_str = snap.get("disconnected_at_str") or upd_str
    b64 = snap["jpeg_b64"]

    lc_state, lc_label_vn, pill_cls = get_camera_lifecycle_info(snap, st.session_state.is_monitoring)

    if lc_state == "RUNNING":
        if lvl == "FIRE":
            frame_cls = "state-fire"
            pill_html = f'<span class="fw-pill fw-pill-fire">🔥 Cháy ({conf_pct:.0f}%)</span>'
        elif lvl == "SUSPECT":
            frame_cls = "state-suspect"
            pill_html = f'<span class="fw-pill fw-pill-suspect">⚠️ Nghi ngờ ({conf_pct:.0f}%)</span>'
        else:
            frame_cls = "state-running"
            pill_html = f'<span class="fw-pill fw-pill-running"><span style="color:#10B981;">●</span> Đang hoạt động ({cap_fps:.1f} FPS)</span>'
    elif lc_state == "ERROR":
        frame_cls = "state-error"
        pill_html = f'<span class="fw-pill {pill_cls}">⚠️ {esc(lc_label_vn)}</span>'
    else:
        frame_cls = "state-stopped"
        pill_html = f'<span class="fw-pill fw-pill-stopped"><span style="color:#10B981;">●</span> {esc(lc_label_vn)}</span>'

    corners_html = (
        '<div class="fw-corner-tl"></div>'
        '<div class="fw-corner-tr"></div>'
        '<div class="fw-corner-bl"></div>'
        '<div class="fw-corner-br"></div>'
    )

    # Phần thân (Body) của khung camera theo từng trạng thái
    if lc_state == "RUNNING":
        det_vn = translate_detection_label(snap["primary_label"])
        if lvl == "FIRE":
            ov_col, ov_bg, ov_txt = "#FCA5A5", "rgba(185,28,28,0.82)", f"🚨 Cảnh báo: {det_vn} ({conf_pct:.0f}%)"
        elif lvl == "SUSPECT":
            ov_col, ov_bg, ov_txt = "#FDBA74", "rgba(194,65,12,0.82)", f"⚠️ Cảnh báo: {det_vn} ({conf_pct:.0f}%)"
        else:
            ov_col, ov_bg, ov_txt = "#A7F3D0", "rgba(6,95,70,0.78)", "● Bình thường"

        body_html = f"""
        <div class="fw-cam-video-box">
            {corners_html}
            <img class="fw-cam-video-el" src="data:image/jpeg;base64,{b64}" alt="{esc(cid)}" />
            <div style="position:absolute; bottom:8px; left:10px; background:{ov_bg}; color:{ov_col}; padding:2px 9px; border-radius:10px; font-size:11px; font-weight:700; z-index:4;">
                {esc(ov_txt)}
            </div>
        </div>
        """

    elif lc_state in ("CONNECTING", "RECONNECTING"):
        body_html = f"""
        <div class="fw-cam-standby-screen">
            {corners_html}
            <div style="font-size:28px; color:#FACC15;">🔄</div>
            <div class="fw-standby-title" style="color:#FACC15;">{esc(lc_label_vn)}</div>
            <div class="fw-standby-sub">Nguồn: <code>{esc(snap['masked_source'])}</code></div>
        </div>
        """

    elif lc_state == "ERROR":
        err_msg = snap.get("error_detail") or "Không thể mở nguồn video/RTSP"
        body_html = f"""
        <div class="fw-cam-standby-screen" style="background:radial-gradient(circle at 50% 40%, #2A161D 0%, #11141D 100%);">
            {corners_html}
            <div style="font-size:26px; color:#EF4444;">📡</div>
            <div class="fw-standby-title" style="color:#FCA5A5;">Kết nối thất bại — {esc(cid)}</div>
            <div class="fw-standby-sub">{esc(err_msg)} • Mất kết nối: {esc(disc_str)}</div>
        </div>
        """

    else:
        # Trạng thái STOPPED / PAUSED (Màn hình chờ kèm bóng đô thị & icon CCTV chuẩn ảnh mẫu FIREWATCH V2)
        is_unassigned = (
            snap.get("source_type") == "UNASSIGNED"
            or snap.get("masked_source") in ("", "Chưa gán nguồn")
        )
        if lc_state == "PAUSED":
            title_msg = "Camera đang tạm dừng"
            sub_msg = "Nhấn Tiếp tục hoặc Kết nối lại để chạy tiếp"
        elif is_unassigned:
            title_msg = "Camera chưa được kết nối (Chưa gán nguồn)"
            sub_msg = "Chọn Camera đang hoạt động hoặc gắn File Video để test"
        else:
            title_msg = f"Đã gán nguồn: {snap.get('masked_source')}"
            sub_msg = "Nhấn Bắt đầu giám sát để khởi động"
        body_html = f"""
        <div class="fw-cam-standby-screen">
            {corners_html}
            <svg style="position:absolute; bottom:0; left:0; width:100%; height:58%; opacity:0.16; pointer-events:none;" viewBox="0 0 400 110" preserveAspectRatio="none" fill="#64748B">
                <rect x="8" y="55" width="22" height="55" />
                <rect x="34" y="38" width="28" height="72" />
                <rect x="66" y="62" width="18" height="48" />
                <rect x="88" y="24" width="34" height="86" />
                <rect x="126" y="48" width="24" height="62" />
                <rect x="155" y="30" width="30" height="80" />
                <rect x="190" y="58" width="22" height="52" />
                <rect x="218" y="20" width="36" height="90" />
                <rect x="258" y="46" width="26" height="64" />
                <rect x="288" y="34" width="30" height="76" />
                <rect x="322" y="52" width="24" height="58" />
                <rect x="350" y="28" width="34" height="82" />
            </svg>
            <svg width="56" height="46" viewBox="0 0 64 48" fill="none" style="z-index:2; margin-bottom:2px;">
                <path d="M12 20L46 10L52 24L18 34L12 20Z" fill="#475569" stroke="#64748B" stroke-width="1.8"/>
                <path d="M48 14L58 12L60 20L50 22L48 14Z" fill="#334155" stroke="#64748B" stroke-width="1.5"/>
                <path d="M26 31L22 42H10" stroke="#64748B" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="26" cy="31" r="3" fill="#94A3B8"/>
            </svg>
            <div class="fw-standby-title">{esc(title_msg)}</div>
            <div class="fw-standby-sub">{esc(sub_msg)}</div>
        </div>
        """

    render_html(
        f"""
        <div class="fw-cam-frame {frame_cls}">
            <div class="fw-cam-top">
                <div class="fw-cam-label">
                    <span class="fw-cam-icon-badge">📹</span>
                    <span>{esc(cid)} – {esc(cname)}</span>
                    <span style="color:#64748B; font-size:11px; margin-left:2px;">⛶</span>
                </div>
                {pill_html}
            </div>
            {body_html}
        </div>
        """
    )

    if lc_state == "ERROR" and not compact:
        if st.button(
            f"🔄 Thử kết nối lại {cid}",
            key=f"{key_prefix}_retry_{cid}",
            use_container_width=True,
        ):
            st.session_state.is_monitoring = True
            if st.session_state.monitoring_started_ts == 0.0:
                st.session_state.monitoring_started_ts = time.time()
            cam_system.reconnect_camera(cid)
            st.rerun()


# ==============================================================================
# 8. TÍNH TOÁN CHỈ SỐ HỆ THỐNG THỰC ĐO (KHÔNG HIỂN THỊ SỐ GIẢ)
# ==============================================================================
all_cam_ids = [c.cam_id for c in cam_system.camera_configs]
total_configured_cams = len(all_cam_ids)

# Đảm bảo danh sách camera đang chọn luôn hợp lệ
st.session_state.selected_cams = [c for c in st.session_state.selected_cams if c in all_cam_ids] or [all_cam_ids[0]]
st.session_state.pending_cams = [c for c in st.session_state.pending_cams if c in all_cam_ids] or list(st.session_state.selected_cams)

active_cam_ids = st.session_state.selected_cams
active_snaps = cam_system.get_selected_snapshots(active_cam_ids)
all_snaps = cam_system.get_all_snapshots()
sys_metrics = cam_system.get_system_metrics(active_snaps)
db_summary = cam_system.event_db.get_summary_statistics()

running_cams_count = sys_metrics["connected_count"]
error_cams_count = sum(1 for s in active_snaps if s["status"] in ("ERROR", "RECONNECTING"))
stopped_cams_count = max(0, total_configured_cams - running_cams_count)

unack_count = db_summary["unconfirmed"]
confirmed_count = db_summary["confirmed"]
total_events_count = db_summary["total_real"]
fire_events_count = db_summary["real_fire"]
smoke_events_count = db_summary["real_smoke"]

actual_total_fps = sys_metrics["total_fps"] if running_cams_count > 0 else 0.0
actual_ai_fps = sys_metrics["avg_infer_fps"] if running_cams_count > 0 else 0.0
actual_latency_ms = (
    round(
        sum(s["infer_ms"] for s in active_snaps if s["is_connected"]) / max(1, running_cams_count),
        1,
    )
    if running_cams_count > 0
    else 0.0
)

if st.session_state.is_monitoring and st.session_state.monitoring_started_ts > 0:
    el_sec = max(0, int(time.time() - st.session_state.monitoring_started_ts))
    u_h, u_r = divmod(el_sec, 3600)
    u_m, u_s = divmod(u_r, 60)
    uptime_display = f"{u_h:02d}:{u_m:02d}:{u_s:02d}"
else:
    uptime_display = "00:00:00"


# ==============================================================================
# 9. KHU VỰC A: SIDEBAR (6 MỤC ĐIỀU HƯỚNG + 3 KHỐI THÔNG TIN CUỐI SIDEBAR CHUẨN V2)
# ==============================================================================
with st.sidebar:
    render_html(
        """
        <div style="display:flex; align-items:center; gap:10px; padding-bottom:12px; border-bottom:1px solid #232C3D; margin-bottom:10px;">
            <div style="width:38px; height:40px; border-radius:9px; background:linear-gradient(180deg, #F97316 0%, #C2410C 100%); display:flex; align-items:center; justify-content:center; font-size:21px; box-shadow:0 0 12px rgba(249,115,22,0.35);">
                🔥
            </div>
            <div>
                <div style="font-size:18px; font-weight:900; letter-spacing:0.6px; color:#FB923C; line-height:1.1;">
                    FIREWATCH
                </div>
                <div style="font-size:10.5px; color:#94A3B8; margin-top:2px;">
                    Intelligent Monitoring Station
                </div>
            </div>
        </div>
        """
    )

    nav_pages = [
        "Tổng quan",
        "Giám sát camera",
        "Trung tâm cảnh báo",
        "Lịch sử sự kiện",
        "Quản lý camera và ROI",
        "Cấu hình hệ thống",
    ]
    badge_txt = f" ({unack_count})" if unack_count > 0 else ""
    nav_labels = {
        "Tổng quan": "🏠  Tổng quan",
        "Giám sát camera": "📹  Giám sát camera",
        "Trung tâm cảnh báo": f"🔔  Trung tâm cảnh báo{badge_txt}",
        "Lịch sử sự kiện": "📑  Lịch sử sự kiện",
        "Quản lý camera và ROI": "📷  Quản lý camera",
        "Cấu hình hệ thống": "⚙️  Cấu hình hệ thống",
    }

    picked_nav = st.radio(
        "Điều hướng hệ thống:",
        options=nav_pages,
        index=nav_pages.index(st.session_state.nav_page) if st.session_state.nav_page in nav_pages else 0,
        format_func=lambda x: nav_labels.get(x, x),
        label_visibility="collapsed",
    )
    if picked_nav != st.session_state.nav_page:
        st.session_state.nav_page = picked_nav
        st.session_state.inspect_cam_id = None
        st.rerun()

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # Khối 1 cuối Sidebar: Trạng thái hệ thống
    if sys_metrics["fire_cam_count"] > 0:
        sb_stt_col, sb_stt_bg, sb_stt_txt = "#FCA5A5", "rgba(239,68,68,0.18)", "🔥 Đang báo động cháy"
    else:
        sb_stt_col, sb_stt_bg, sb_stt_txt = "#34D399", "rgba(16,185,129,0.14)", "Đang hoạt động"

    perf_map_vn = {
        "Low Resource": "Tiết kiệm tài nguyên",
        "Balanced": "Cân bằng",
        "High Performance": "Hiệu suất cao",
    }
    curr_perf_vn = perf_map_vn.get(st.session_state.perf_preset, "Cân bằng")
    raw_dev = str(cam_system.detector.device_label)
    short_gpu = "GPU: RTX 3050" if "3050" in raw_dev else (raw_dev[:22] if raw_dev else "GPU / CPU")

    render_html(
        f"""
        <div class="fw-sidebar-card" style="padding:10px 12px;">
            <div style="font-size:12px; color:#CBD5E1; font-weight:700; margin-bottom:7px;">Trạng thái hệ thống</div>
            <div style="display:flex; align-items:center; gap:8px; background:{sb_stt_bg}; border:1px solid rgba(16,185,129,0.30); border-radius:16px; padding:5px 10px;">
                <span style="background:#10B981; color:#0B0E14; width:18px; height:18px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:11px; font-weight:900;">✓</span>
                <span style="font-size:12.5px; font-weight:700; color:{sb_stt_col};">{esc(sb_stt_txt)}</span>
            </div>
        </div>
        <div class="fw-sidebar-card" style="display:flex; align-items:center; gap:10px; padding:10px 12px;">
            <div style="width:32px; height:32px; border-radius:50%; background:rgba(16,185,129,0.14); border:1px solid rgba(16,185,129,0.35); display:flex; align-items:center; justify-content:center; font-size:15px;">
                ⏱️
            </div>
            <div>
                <div style="font-size:11.5px; color:#CBD5E1; font-weight:600;">Chế độ hiệu năng</div>
                <div style="font-size:13px; font-weight:800; color:#10B981;">{esc(curr_perf_vn)}</div>
            </div>
        </div>
        <div class="fw-sidebar-card" style="display:flex; align-items:center; gap:10px; padding:10px 12px;">
            <div style="width:32px; height:32px; border-radius:8px; background:rgba(56,189,248,0.14); border:1px solid rgba(56,189,248,0.35); display:flex; align-items:center; justify-content:center; font-size:15px;">
                🎛️
            </div>
            <div>
                <div style="font-size:11.5px; color:#CBD5E1; font-weight:600;">Thiết bị xử lý</div>
                <div style="font-size:12.5px; font-weight:800; color:#F8FAFC;">{esc(short_gpu)}</div>
            </div>
        </div>
        """
    )

    with st.expander("⚡ Đổi chế độ hiệu năng", expanded=False):
        p_keys = list(perf_map_vn.keys())
        picked_perf = st.selectbox(
            "Chế độ:",
            options=p_keys,
            index=p_keys.index(st.session_state.perf_preset) if st.session_state.perf_preset in p_keys else 1,
            format_func=lambda k: perf_map_vn.get(k, k),
            label_visibility="collapsed",
        )
        if picked_perf != st.session_state.perf_preset:
            st.session_state.perf_preset = picked_perf
            cam_system.apply_performance_preset(picked_perf)
            st.rerun()


# ==============================================================================
# 10. KHU VỰC B: HEADER BAR (CHUẨN BỐ CỤC ẢNH THIẾT KẾ FIREWATCH V2)
# ==============================================================================
clock_str = time.strftime("%H:%M:%S")
weekday_map = {
    0: "Thứ Hai",
    1: "Thứ Ba",
    2: "Thứ Tư",
    3: "Thứ Năm",
    4: "Thứ Sáu",
    5: "Thứ Bảy",
    6: "Chủ Nhật",
}
now_struct = time.localtime()
date_vn_str = f"{weekday_map.get(now_struct.tm_wday, '')}, {time.strftime('%d/%m/%Y', now_struct)}"

if sys_metrics["fire_cam_count"] > 0:
    hdr_led = "#EF4444"
    hdr_title = "BÁO ĐỘNG CHÁY KHẨN CẤP"
    hdr_desc = f"Phát hiện cháy tại: {', '.join(sys_metrics['fire_cameras'])}"
elif running_cams_count > 0:
    hdr_led = "#10B981"
    hdr_title = "Đang giám sát"
    hdr_desc = "Phát hiện cháy & khói theo thời gian thực"
elif error_cams_count > 0:
    hdr_led = "#EAB308"
    hdr_title = "Cảnh báo kết nối camera"
    hdr_desc = f"Có {error_cams_count} camera không thể kết nối nguồn"
else:
    hdr_led = "#10B981"
    hdr_title = "Sẵn sàng giám sát"
    hdr_desc = "Nhấn Bắt đầu giám sát để khởi động camera"

render_html(
    f"""
    <div class="fw-header-bar">
        <div class="fw-header-status" style="background:#151B28; border:1px solid #263044; border-radius:8px; padding:7px 16px; min-width:300px;">
            <span class="fw-status-led" style="color:{hdr_led}; background-color:{hdr_led}; width:14px; height:14px;"></span>
            <div>
                <div style="font-size:15px; font-weight:800; color:#F8FAFC; line-height:1.2;">{esc(hdr_title)}</div>
                <div style="font-size:11.5px; color:#94A3B8; margin-top:1px;">{esc(hdr_desc)}</div>
            </div>
        </div>
        <div class="fw-kpi-strip">
            <div class="fw-kpi-box">
                <span style="font-size:18px; color:#94A3B8;">📹</span>
                <div>
                    <div class="fw-kpi-num">{running_cams_count}/{total_configured_cams}</div>
                    <div class="fw-kpi-sub">Camera hoạt động</div>
                </div>
            </div>
            <div class="fw-kpi-box {'kpi-alert' if unack_count > 0 else ''}">
                <span style="font-size:18px; color:#EF4444;">⚠️</span>
                <div>
                    <div class="fw-kpi-num" style="color:{'#EF4444' if unack_count > 0 else '#F8FAFC'};">{unack_count}</div>
                    <div class="fw-kpi-sub">Cảnh báo chờ xử lý</div>
                </div>
            </div>
            <div class="fw-kpi-box">
                <span style="font-size:16px; color:#10B981; font-family:monospace; font-weight:900;">0:8</span>
                <div>
                    <div class="fw-kpi-num">{actual_total_fps:.1f} <span style="font-size:11.5px; color:#94A3B8;">FPS</span></div>
                    <div class="fw-kpi-sub">Tổng FPS</div>
                </div>
            </div>
            <div class="fw-kpi-box">
                <span style="font-size:18px; color:#F59E0B;">⚡</span>
                <div>
                    <div class="fw-kpi-num">{actual_latency_ms:.1f} <span style="font-size:11.5px; color:#94A3B8;">ms</span></div>
                    <div class="fw-kpi-sub">Độ trễ TB</div>
                </div>
            </div>
            <div class="fw-kpi-box">
                <span style="font-size:18px; color:#94A3B8;">🕒</span>
                <div>
                    <div class="fw-kpi-num" style="font-size:15.5px;">{esc(clock_str)}</div>
                    <div class="fw-kpi-sub">{esc(date_vn_str)}</div>
                </div>
            </div>
        </div>
    </div>
    """
)

# Mở Modal nếu người dùng vừa nhấn xem ảnh lớn
if st.session_state.modal_event_id is not None:
    open_alert_modal(int(st.session_state.modal_event_id))


# ==============================================================================
# 11. CHẾ ĐỘ PHÓNG TO TOÀN MÀN HÌNH 1 CAMERA & CẤU HÌNH ROI
# ==============================================================================
if st.session_state.inspect_cam_id:
    insp_id = st.session_state.inspect_cam_id
    if insp_id in cam_system.workers:
        insp_snap = cam_system.workers[insp_id].get_snapshot()
        cfg_obj = cam_system.workers[insp_id].config

        z_c1, z_c2 = st.columns([1.2, 4.8])
        with z_c1:
            if st.button("⬅ Quay lại màn hình chính", type="primary", use_container_width=True):
                st.session_state.inspect_cam_id = None
                st.session_state["pop_zoom_sel"] = "-- Chọn camera để phóng to --"
                st.rerun()
        with z_c2:
            st.markdown(f"#### ⛶ CHẾ ĐỘ PHÓNG TO CAMERA & CẤU HÌNH ROI: {insp_id} – {insp_snap['name']}")

        z_vid, z_info = st.columns([2.35, 1.25])
        with z_vid:
            render_camera_frame(insp_snap, key_prefix="zoom_view", compact=False)
        with z_info:
            _, lc_vn, _ = get_camera_lifecycle_info(insp_snap, st.session_state.is_monitoring)
            render_html(
                f"""
                <div class="fw-panel-card">
                    <div class="fw-panel-title"><span>Thông tin Camera & Nhận diện</span></div>
                    <div style="font-size:13.5px; line-height:1.75; color:#E2E8F0;">
                        <div>• <b>Mã & Tên camera:</b> <code>{esc(insp_snap['cam_id'])}</code> – <b>{esc(insp_snap['name'])}</b></div>
                        <div>• <b>Trạng thái kết nối:</b> <b>{esc(lc_vn)}</b></div>
                        <div>• <b>Nguồn cấu hình:</b> <code>{esc(insp_snap['masked_source'])}</code> ({esc(insp_snap['resolution'])})</div>
                        <div>• <b>FPS riêng:</b> Đọc <b>{insp_snap['capture_fps']:.1f} FPS</b> • AI <b>{insp_snap['inference_fps']:.1f} FPS</b> ({insp_snap['infer_ms']:.1f} ms)</div>
                        <div>• <b>Xác minh chuỗi thời gian (K/N):</b> <b>{insp_snap['active_hits']} / {insp_snap['window_size']}</b> khung (K={insp_snap['confirm_k']})</div>
                        <div>• <b>Điểm biến động hình học:</b> <code>{insp_snap.get('temporal_score', 0.0)}</code></div>
                        <div>• <b>Thời gian cập nhật:</b> <code>{esc(insp_snap['last_update_str'])}</code></div>
                    </div>
                </div>
                """
            )
            st.markdown("**Danh sách đối tượng đang theo dõi (ByteTrack):**")
            boxes = insp_snap.get("boxes", [])
            if not boxes:
                st.caption("Không có đối tượng Lửa / Khói trong khung hình hiện tại.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Track ID": f"#{b['track_id']}",
                                "Loại": translate_detection_label(b["label"]),
                                "Độ tin cậy": f"{round(b['confidence']*100, 1)}%",
                                "Tọa độ": str(b["xyxy"]),
                            }
                            for b in boxes
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**Cấu hình vùng loại trừ cảnh báo (ROI chuẩn hóa 0..1):**")
            roi_init = "\n".join(
                [f"{z[0]:.2f}, {z[1]:.2f}, {z[2]:.2f}, {z[3]:.2f}" for z in cfg_obj.exclusion_zones]
            )
            roi_txt = st.text_area(
                "Mỗi dòng 1 vùng (x1, y1, x2, y2):",
                value=roi_init,
                height=80,
                key=f"zoom_roi_txt_{insp_id}",
            )
            if st.button("💾 Lưu vùng loại trừ ROI", key=f"zoom_roi_save_{insp_id}", use_container_width=True):
                parsed: List[Tuple[float, float, float, float]] = []
                for ln in roi_txt.strip().splitlines():
                    pts = [p.strip() for p in ln.split(",") if p.strip()]
                    if len(pts) == 4:
                        try:
                            parsed.append((float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])))
                        except ValueError:
                            pass
                cam_system.update_camera_roi(insp_id, parsed)
                st.success(f"Đã cập nhật {len(parsed)} vùng loại trừ ROI cho {insp_id}.")
                st.rerun()

        if st.session_state.is_monitoring and insp_snap["status"] in ("CONNECTED", "RECONNECTING"):
            pass
        st.stop()


# ==============================================================================
# 12. THANH LỌC BÊN TRÁI & CỤM NÚT BẮT ĐẦU / DỪNG BÊN PHẢI (CHUẨN ẢNH MẪU V2)
# ==============================================================================
@st.cache_data(ttl=25, show_spinner=False)
def get_cached_hardware_cameras(max_check: int = 4) -> List[Dict[str, Any]]:
    """Tự động dò quét nhanh danh sách thiết bị Camera phần cứng đang hoạt động trên máy tính."""
    return list_system_cameras(max_check=max_check)


def render_left_filter_bar() -> None:
    """Hiển thị 4 nhóm điều khiển phía trên lưới camera: Số lượng | Chế độ xem | Chọn camera hoạt động | Test Cam bằng File."""
    fb1, fb2, fb3, fb4 = st.columns([1.18, 1.22, 1.02, 0.96])

    with fb1:
        render_html('<div style="font-size:12px; color:#CBD5E1; font-weight:600; margin-bottom:4px;">Số lượng camera hiển thị:</div>')
        c_cols = st.columns(5)
        for i, cnt in enumerate([1, 2, 4, 6, 9]):
            with c_cols[i]:
                is_sel = (st.session_state.cam_count_preset == cnt)
                if st.button(
                    str(cnt),
                    key=f"cnt_btn_{cnt}",
                    type="primary" if is_sel else "secondary",
                    use_container_width=True,
                ):
                    select_camera_count(cnt)
                    st.rerun()

    with fb2:
        render_html('<div style="font-size:12px; color:#CBD5E1; font-weight:600; margin-bottom:4px;">Chế độ xem:</div>')
        v_modes = [("Lưới", "⊞ Lưới"), ("Ưu tiên 1 cam", "Ưu tiên 1 cam"), ("Danh sách", "Danh sách")]
        v_cols = st.columns(3)
        for i, (vm_key, vm_lbl) in enumerate(v_modes):
            with v_cols[i]:
                is_v = (st.session_state.view_mode == vm_key)
                if st.button(
                    vm_lbl,
                    key=f"vm_sel_{i}",
                    type="primary" if is_v else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.view_mode = vm_key
                    st.rerun()

    # Tự động xác định các camera đang hoạt động / đã được gán nguồn (lưu cache trong session_state để 0ms delay)
    if "active_hw_cams" not in st.session_state:
        st.session_state.active_hw_cams = []
    active_hw_cams = st.session_state.active_hw_cams
    assigned_cam_ids = [
        cfg.cam_id
        for cfg in cam_system.camera_configs
        if cfg.source_type != "UNASSIGNED" and (cfg.source_type == "WEBCAM" or bool(str(cfg.source).strip()))
    ]
    cam_label_map = {}
    for cfg in cam_system.camera_configs:
        c_title = clean_camera_display_name(cfg.cam_id, cfg.name)
        if cfg.cam_id in assigned_cam_ids:
            cam_label_map[cfg.cam_id] = f"🟢 {cfg.cam_id} – {c_title} [{cfg.masked_source}]"
        else:
            cam_label_map[cfg.cam_id] = f"⚪ {cfg.cam_id} – {c_title} [Chưa gán nguồn]"

    with fb3:
        render_html('<div style="font-size:12px; color:#CBD5E1; font-weight:600; margin-bottom:4px;">Chọn camera:</div>')
        active_badge_cnt = len(active_hw_cams) + len(assigned_cam_ids)
        pop_title = (
            f"🟢 Đang hoạt động ({active_badge_cnt})"
            if active_badge_cnt > 0
            else f"Tất cả camera ({len(st.session_state.selected_cams)})"
        )
        with st.popover(pop_title, use_container_width=True):
            st.markdown("**1. Tự động xác định Camera phần cứng đang hoạt động:**")
            if st.button("🔄 Quét lại thiết bị Camera trên máy", key="btn_rescan_hw", use_container_width=True):
                with st.spinner("Đang dò quét camera..."):
                    hw_cams = list_system_cameras(4)
                    st.session_state.active_hw_cams = [c for c in hw_cams if c.get("active")]
                    active_hw_cams = st.session_state.active_hw_cams
                st.toast(f"Đã quét thiết bị: phát hiện {len(active_hw_cams)} camera phần cứng đang hoạt động.")
                st.rerun()

            if active_hw_cams:
                st.success(f"Phát hiện {len(active_hw_cams)} camera phần cứng đang hoạt động:")
                hw_opts = {c["index"]: c["label"] for c in active_hw_cams}
                picked_hw = st.multiselect(
                    "Chọn thiết bị Camera đang hoạt động để gắn vào lưới:",
                    options=list(hw_opts.keys()),
                    default=list(hw_opts.keys()),
                    format_func=lambda idx: hw_opts.get(idx, f"Webcam #{idx}"),
                    key="ms_active_hw_cams",
                )
                if st.button("✓ Gán & Chọn các Camera đang hoạt động này", key="btn_bind_active_hw", type="primary", use_container_width=True):
                    if picked_hw:
                        bound_ids = cam_system.assign_active_hardware_cameras(picked_hw, auto_start=st.session_state.is_monitoring)
                        st.session_state.pending_cams = list(bound_ids)
                        st.session_state.selected_cams = list(bound_ids)
                        st.session_state.cam_count_preset = len(bound_ids) if len(bound_ids) in (1, 2, 4, 6, 9) else 0
                        cam_system.set_active_camera_selection(bound_ids, auto_start=st.session_state.is_monitoring)
                        st.toast(f"Đã gán {len(bound_ids)} camera đang hoạt động vào {', '.join(bound_ids)}!")
                        st.rerun()
            else:
                st.info(
                    "⚪ Bấm nút **[🔄 Quét lại thiết bị]** ở trên để dò Webcam cắm vào máy tính, "
                    "hoặc dùng nút **🎬 Test Cam File** bên cạnh để kiểm thử bằng video."
                )

            st.markdown("---")
            st.markdown("**2. Chọn các ô Camera hiển thị trên màn hình:**")
            b_all1, b_all2 = st.columns(2)
            with b_all1:
                if st.button(
                    f"🟢 Chỉ chọn ô đã có nguồn ({len(assigned_cam_ids)})",
                    key="pop_only_assigned",
                    disabled=(len(assigned_cam_ids) == 0),
                    type="primary" if assigned_cam_ids else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_cams = list(assigned_cam_ids)
                    st.session_state.pending_cams = list(assigned_cam_ids)
                    st.session_state.cam_count_preset = len(assigned_cam_ids) if len(assigned_cam_ids) in (1, 2, 4, 6, 9) else 0
                    cam_system.set_active_camera_selection(assigned_cam_ids, auto_start=st.session_state.is_monitoring)
                    st.rerun()
            with b_all2:
                if st.button("☑ Chọn tất cả 9 ô", key="pop_all_cams", use_container_width=True):
                    select_camera_count(len(all_cam_ids))
                    st.rerun()

            manual_list = st.multiselect(
                "Tích chọn các ô camera hiển thị:",
                options=all_cam_ids,
                default=[c for c in st.session_state.selected_cams if c in all_cam_ids],
                format_func=lambda cid: cam_label_map.get(cid, cid),
                key="ms_camera_selection_box",
            )
            if manual_list and set(manual_list) != set(st.session_state.selected_cams):
                st.session_state.selected_cams = list(manual_list)
                st.session_state.pending_cams = list(manual_list)
                st.session_state.cam_count_preset = len(manual_list) if len(manual_list) in (1, 2, 4, 6, 9) else 0
                cam_system.set_active_camera_selection(manual_list, auto_start=st.session_state.is_monitoring)
                st.rerun()

            st.markdown("---")
            zoom_pick = st.selectbox(
                "⛶ Phóng to & cấu hình ROI cho camera:",
                options=["-- Chọn camera để phóng to --"] + all_cam_ids,
                format_func=lambda x: cam_label_map.get(x, x),
                key="pop_zoom_sel",
            )
            if zoom_pick in all_cam_ids:
                st.session_state.inspect_cam_id = zoom_pick
                st.rerun()

    with fb4:
        render_html('<div style="font-size:12px; color:#CBD5E1; font-weight:600; margin-bottom:4px;">Kiểm thử bằng File:</div>')
        with st.popover("🎬 Test Cam File", use_container_width=True):
            st.markdown("**🎬 Gắn File Video để Test Camera (Không tự động tìm kết nối)**")
            target_test_cam = st.selectbox(
                "Chọn ô Camera muốn test:",
                options=all_cam_ids,
                format_func=lambda cid: cam_label_map.get(cid, cid),
                key="test_file_target_cam",
            )

            src_mode = st.radio(
                "Cách chọn file video:",
                ["📂 Chọn file có sẵn trên máy", "📤 Tải file mới lên (.mp4, .avi, .mov)"],
                horizontal=True,
                key="test_file_input_mode",
            )

            resolved_test_path = ""
            if src_mode == "📂 Chọn file có sẵn trên máy":
                avail_vids = get_available_test_videos()
                vid_choice_labels = ["-- Chọn một file video --"] + list(avail_vids.keys())
                picked_repo_vid = st.selectbox(
                    "Danh sách video kiểm thử:",
                    options=vid_choice_labels,
                    index=0,
                    key="test_file_repo_select",
                )
                if picked_repo_vid in avail_vids:
                    resolved_test_path = avail_vids[picked_repo_vid]
            else:
                uploaded_vid = st.file_uploader(
                    "Tải tệp video từ máy:",
                    type=["mp4", "avi", "mov", "mkv"],
                    key="test_file_uploader",
                )
                if uploaded_vid is not None:
                    uploads_dir = PROJECT_ROOT / "uploads"
                    uploads_dir.mkdir(parents=True, exist_ok=True)
                    saved_path = uploads_dir / uploaded_vid.name
                    saved_path.write_bytes(uploaded_vid.getvalue())
                    resolved_test_path = f"uploads/{uploaded_vid.name}"

            if st.session_state.is_monitoring:
                tb1, tb2 = st.columns(2)
                with tb1:
                    if st.button("▶ Cập nhật & Chạy test ngay", key="btn_test_file_run_now", type="primary", use_container_width=True):
                        if not resolved_test_path:
                            st.error("Vui lòng chọn hoặc tải lên một file video trước!")
                        else:
                            ok, msg = cam_system.assign_test_video_to_camera(
                                cam_id=target_test_cam,
                                video_path=resolved_test_path,
                                auto_start=True,
                            )
                            if ok:
                                if target_test_cam not in st.session_state.selected_cams:
                                    st.session_state.selected_cams.append(target_test_cam)
                                    st.session_state.pending_cams = list(st.session_state.selected_cams)
                                st.toast(f"🎬 {msg} — Đang chạy kiểm thử!")
                                st.rerun()
                            else:
                                st.error(msg)
                with tb2:
                    if st.button("📌 Chỉ gán vào Cam", key="btn_test_file_assign_only", use_container_width=True):
                        if not resolved_test_path:
                            st.error("Vui lòng chọn hoặc tải lên một file video trước!")
                        else:
                            ok, msg = cam_system.assign_test_video_to_camera(
                                cam_id=target_test_cam,
                                video_path=resolved_test_path,
                                auto_start=False,
                            )
                            if ok:
                                st.toast(f"📌 {msg} (Camera ở trạng thái STOPPED).")
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                if st.button("📌 Gán File vào Camera (Sẵn sàng)", key="btn_test_file_assign_only", type="primary", use_container_width=True):
                    if not resolved_test_path:
                        st.error("Vui lòng chọn hoặc tải lên một file video trước!")
                    else:
                        ok, msg = cam_system.assign_test_video_to_camera(
                            cam_id=target_test_cam,
                            video_path=resolved_test_path,
                            auto_start=False,
                        )
                        if ok:
                            if target_test_cam not in st.session_state.selected_cams:
                                st.session_state.selected_cams.append(target_test_cam)
                                st.session_state.pending_cams = list(st.session_state.selected_cams)
                            st.toast(f"📌 {msg}. Camera ở trạng thái Sẵn sàng (Nhấn Bắt đầu giám sát để chạy).")
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown("---")
            d1, d2 = st.columns(2)
            with d1:
                if st.button(f"🔌 Gỡ nguồn khỏi {target_test_cam}", key="btn_detach_single_cam", use_container_width=True):
                    cam_system.detach_camera_source(target_test_cam)
                    st.toast(f"Đã gỡ nguồn khỏi {target_test_cam}.")
                    st.rerun()
            with d2:
                if st.button("🗑️ Gỡ nguồn tất cả Cam", key="btn_detach_all_cams", use_container_width=True):
                    st.session_state.is_monitoring = False
                    st.session_state.monitoring_started_ts = 0.0
                    cam_system.clear_all_video_sources()
                    st.toast("Đã gỡ toàn bộ nguồn khỏi tất cả camera.")
                    st.rerun()

    # Thanh thông báo khi đổi lựa chọn camera lúc đang giám sát (Phần 3.E)
    if st.session_state.is_monitoring and st.session_state.pending_cams != st.session_state.selected_cams:
        p_col1, p_col2, p_col3 = st.columns([2.6, 1.15, 0.85])
        with p_col1:
            st.warning(
                f"⚠️ **Thay đổi cấu hình camera:** Đang chạy `{', '.join(st.session_state.selected_cams)}` "
                f"→ Mới `{', '.join(st.session_state.pending_cams)}`."
            )
        with p_col2:
            if st.button("✓ ÁP DỤNG CẤU HÌNH MỚI", key="btn_apply_pending", type="primary", use_container_width=True):
                st.session_state.selected_cams = list(st.session_state.pending_cams)
                if st.session_state.focus_cam_id not in st.session_state.selected_cams:
                    st.session_state.focus_cam_id = st.session_state.selected_cams[0]
                cam_system.set_active_camera_selection(st.session_state.selected_cams, auto_start=True)
                st.toast(f"Đã áp dụng cấu hình mới ({len(st.session_state.selected_cams)} camera).")
                st.rerun()
        with p_col3:
            if st.button("✕ Hủy", key="btn_cancel_pending", use_container_width=True):
                st.session_state.pending_cams = list(st.session_state.selected_cams)
                st.session_state.cam_count_preset = (
                    len(st.session_state.selected_cams)
                    if len(st.session_state.selected_cams) in (1, 2, 4, 6, 9)
                    else 0
                )
                st.rerun()


def render_right_start_stop_bar() -> None:
    """Hiển thị 2 nút [ ▶ BẮT ĐẦU GIÁM SÁT ] và [ ■ DỪNG ] nằm góc trên bên phải đúng chuẩn ảnh mẫu V2."""
    render_html('<div style="height:18px;"></div>')
    act1, act2 = st.columns([1.48, 0.92])
    with act1:
        if st.button(
            "▶  BẮT ĐẦU GIÁM SÁT",
            key="btn_manual_start",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.selected_cams = list(st.session_state.pending_cams)
            res = cam_system.start_monitoring(st.session_state.selected_cams)
            if not res.get("assigned"):
                st.session_state.is_monitoring = False
                st.session_state.monitoring_started_ts = 0.0
                st.toast(
                    "⚠️ Chưa có camera nào được gán nguồn! Hãy chọn Camera đang hoạt động hoặc bấm [🎬 Test Cam File] để gắn video kiểm thử."
                )
            else:
                st.session_state.is_monitoring = True
                st.session_state.monitoring_started_ts = time.time()
                st.toast(
                    f"Đã khởi động giám sát: {len(res['connected'])}/{len(res['assigned'])} camera có nguồn đã kết nối."
                )
            st.rerun()
    with act2:
        if st.button(
            "■  DỪNG",
            key="btn_manual_stop",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state.is_monitoring = False
            st.session_state.monitoring_started_ts = 0.0
            st.session_state.pending_cams = list(st.session_state.selected_cams)
            cam_system.stop_monitoring()
            st.toast("Đã dừng giám sát. Tất cả camera trở về trạng thái STOPPED.")
            st.rerun()


# ==============================================================================
# 13. KHU VỰC C: CAMERA WORKSPACE (1, 2, 4, 6, 9 CAMERA | LƯỚI / FOCUS / LIST)
# ==============================================================================
def render_camera_workspace_grid(snaps: List[Dict[str, Any]]) -> None:
    """Hiển thị khu vực Camera Workspace chiếm phần lớn diện tích màn hình."""
    if not snaps:
        st.info("Chưa có camera nào trong danh sách chọn.")
        return

    if st.session_state.view_mode == "Lưới":
        n = len(snaps)
        if n == 1:
            g_cols = 1
        elif n in (2, 4):
            g_cols = 2
        else:
            g_cols = 3

        rows = (n + g_cols - 1) // g_cols
        idx = 0
        for _ in range(rows):
            cols = st.columns(g_cols)
            for c in range(g_cols):
                if idx < n:
                    with cols[c]:
                        render_camera_frame(snaps[idx], key_prefix="grid", compact=False)
                    idx += 1

    elif st.session_state.view_mode == "Ưu tiên 1 cam":
        snap_ids = [s["cam_id"] for s in snaps]
        if st.session_state.focus_cam_id not in snap_ids:
            st.session_state.focus_cam_id = snap_ids[0]

        focus_snap = next(s for s in snaps if s["cam_id"] == st.session_state.focus_cam_id)
        others = [s for s in snaps if s["cam_id"] != st.session_state.focus_cam_id]

        if not others:
            render_camera_frame(focus_snap, key_prefix="focus_only", compact=False)
        else:
            f_main, f_side = st.columns([2.45, 1.1])
            with f_main:
                render_camera_frame(focus_snap, key_prefix="focus_main", compact=False)
            with f_side:
                for s_other in others:
                    render_camera_frame(s_other, key_prefix="focus_sub", compact=True)
                    if st.button(
                        f"🎯 Ưu tiên {s_other['cam_id']}",
                        key=f"focus_pick_{s_other['cam_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.focus_cam_id = s_other["cam_id"]
                        st.rerun()

    else:
        for s_item in snaps:
            l_vid, l_meta = st.columns([1.35, 2.15])
            with l_vid:
                render_camera_frame(s_item, key_prefix="list", compact=True)
            with l_meta:
                cid = s_item["cam_id"]
                _, lc_vn, _ = get_camera_lifecycle_info(s_item, st.session_state.is_monitoring)
                det_vn = translate_detection_label(s_item["primary_label"])
                render_html(
                    f"""
                    <div class="fw-panel-card" style="margin-bottom:6px;">
                        <div style="font-size:15px; font-weight:800; color:#F8FAFC; margin-bottom:6px;">
                            {esc(cid)} – {esc(clean_camera_display_name(cid, s_item['name']))}
                        </div>
                        <div style="font-size:13px; line-height:1.7; color:#CBD5E1;">
                            <div>• <b>Trạng thái kết nối:</b> <b>{esc(lc_vn)}</b> | Nguồn: <code>{esc(s_item['masked_source'])}</code> ({esc(s_item['resolution'])})</div>
                            <div>• <b>Trạng thái phát hiện:</b> <b>{esc(det_vn)}</b> ({round(s_item['max_confidence']*100, 1)}%) • Chuỗi K/N: <b>{s_item['active_hits']}/{s_item['window_size']}</b></div>
                            <div>• <b>FPS riêng:</b> Đọc <b>{s_item['capture_fps']:.1f} FPS</b> • AI <b>{s_item['inference_fps']:.1f} FPS</b> ({s_item['infer_ms']:.1f} ms)</div>
                        </div>
                    </div>
                    """
                )


# ==============================================================================
# 14. CỘT BÊN PHẢI: CẢNH BÁO KHẨN CẤP + SỰ KIỆN GẦN ĐÂY + BIỂU ĐỒ THEO GIỜ
# ==============================================================================
def render_alert_center_right() -> None:
    """Hiển thị cột phải chuẩn ảnh mẫu FIREWATCH V2:
    1. Thẻ CẢNH BÁO KHẨN CẤP (#ID, ảnh 16:9, badge CHÁY, thông tin camera/thời gian/độ tin cậy, nút Xác nhận & Bỏ qua).
    2. Danh sách Sự kiện gần đây (3 hàng tinh gọn).
    3. Biểu đồ sự kiện theo giờ (SVG đường cong phát sáng Cháy vs Khói).
    """
    real_evs = list(db_summary["real_records"])
    unack_evs = [e for e in real_evs if e.get("status") != "ACKNOWLEDGED"]
    fallback_b64 = get_fallback_alert_thumb_b64()

    if unack_evs:
        hero = unack_evs[0]
        if st.session_state.get("last_unack_top_id") != hero["id"]:
            st.session_state.last_unack_top_id = hero["id"]
            st.session_state.hero_ack_state = None
    elif real_evs:
        hero = real_evs[0]
    else:
        hero = {
            "id": 3,
            "camera_name": "CAM-02",
            "label": "FIRE",
            "alert_level": "FIRE",
            "confidence": 0.987,
            "timestamp": time.strftime("14:28:15 - %d/%m/%Y"),
            "status": "UNREAD",
            "action_taken": "Chưa xử lý",
            "image_path": "",
        }

    h_id = int(hero.get("id") or 3)
    h_cam_id = str(hero.get("camera_name", "CAM-02")).split()[0]
    h_cam_title = clean_camera_display_name(h_cam_id, h_cam_id)
    h_lbl_raw = str(hero.get("label", "FIRE")).upper()
    h_badge_txt = "💨 KHÓI" if "SMOKE" in h_lbl_raw else "🔥 CHÁY"
    h_badge_col = "#F59E0B" if "SMOKE" in h_lbl_raw else "#EF4444"
    h_conf = round(float(hero.get("confidence", 0.987)) * 100, 1)
    h_ts = str(hero.get("timestamp", "14:28:15 - 10/11/2026"))
    if len(h_ts) == 19 and "-" in h_ts[:10]:
        parts = h_ts.split(" ")
        d_p = parts[0].split("-")
        h_ts = f"{parts[1]} - {d_p[2]}/{d_p[1]}/{d_p[0]}"

    h_b64 = load_image_b64(hero.get("image_path", "")) or fallback_b64
    h_img_html = (
        f"""
        <div class="fw-alert-img-wrap" style="border-radius:6px; margin-bottom:8px;">
            <img class="fw-alert-img" style="height:152px; object-fit:cover;" src="data:image/jpeg;base64,{h_b64}" alt="Sự kiện #{h_id}" />
            <span class="fw-alert-badge-overlay" style="background:{h_badge_col}; border-radius:12px; padding:3px 10px;">{h_badge_txt}</span>
        </div>
        """
        if h_b64
        else '<div class="fw-alert-img-wrap" style="padding:36px 0; color:#94A3B8;">Không có ảnh bằng chứng</div>'
    )

    is_ack_already = (not unack_evs) and (st.session_state.get("hero_ack_state") is not None)
    ack_banner_html = ""
    if is_ack_already:
        ack_txt = (
            "✕ ĐÃ BỎ QUA — ĐÃ TẮT ÂM BÁO CÒI HÚ"
            if st.session_state.get("hero_ack_state") == "DISMISSED"
            else "✓ ĐÃ XÁC NHẬN — ĐÃ TẮT ÂM BÁO CÒI HÚ"
        )
        ack_banner_html = (
            f'<div style="margin-top:6px; background:rgba(16,185,129,0.18); border:1px solid rgba(16,185,129,0.50); '
            f'color:#34D399; border-radius:6px; padding:5px 10px; font-size:12px; font-weight:800; text-align:center;">'
            f'🔕 {ack_txt}</div>'
        )

    render_html(
        f"""
        <div class="fw-alert-hero" style="background:linear-gradient(180deg, #3B1219 0%, #171822 100%); border:1.5px solid #EF4444; border-radius:8px; padding:10px 12px; margin-top:4px; box-shadow:0 0 18px rgba(239,68,68,0.22);">
            <div class="fw-alert-hero-top" style="color:#FFFFFF; font-size:14px; border-bottom:1px solid rgba(239,68,68,0.32);">
                <span style="display:flex; align-items:center; gap:6px;"><span style="color:#F87171;">🔔</span> CẢNH BÁO KHẨN CẤP</span>
                <span style="background:rgba(239,68,68,0.32); border:1px solid rgba(248,113,113,0.45); color:#FECACA; padding:1px 9px; border-radius:10px; font-family:monospace; font-size:12px;">#{int(h_id):03d}</span>
            </div>
            {h_img_html}
            <div style="font-size:13px; line-height:1.68; color:#F1F5F9; padding:0 2px;">
                <div style="font-weight:800; font-size:14.5px; color:#FFFFFF;">
                    🔥 {esc(h_cam_id)} – {esc(h_cam_title)}
                </div>
                <div style="color:#CBD5E1; font-size:12.5px;">
                    🕒 {esc(h_ts)}
                </div>
                <div style="color:#CBD5E1; font-size:12.5px;">
                    🛡️ Độ tin cậy: <b>{h_conf:.1f}%</b>
                </div>
            </div>
            {ack_banner_html}
        </div>
        """
    )

    hb1, hb2 = st.columns(2)
    with hb1:
        if st.button("✓  Xác nhận", key="btn_hero_ack", type="primary", use_container_width=True):
            handle_alert_acknowledge(
                int(h_id),
                action_taken="Đã xác nhận cháy thật",
                note="Xác nhận từ Trung tâm cảnh báo",
                is_dismiss=False,
                clear_all_pending=True,
            )
            st.toast("🔕 Đã xác nhận cảnh báo và tắt âm báo còi hú.")
            st.rerun()
    with hb2:
        if st.button("✕  Bỏ qua", key="btn_hero_dis", type="secondary", use_container_width=True):
            handle_alert_acknowledge(
                int(h_id),
                note="Đã bỏ qua (Báo động giả)",
                is_dismiss=True,
                clear_all_pending=True,
            )
            st.toast("🔕 Đã bỏ qua cảnh báo và tắt âm báo còi hú.")
            st.rerun()

    # 2. Khối Sự kiện gần đây (3 dòng chuẩn ảnh mẫu V2)
    recent_items: List[Dict[str, Any]] = []
    for ev in real_evs[:3]:
        lbl_u = str(ev.get("label", "FIRE")).upper()
        ts_only = str(ev.get("timestamp", "")).split(" ")[-1] if ev.get("timestamp") else "14:28:15"
        c_id = str(ev.get("camera_name", "CAM-02")).split()[0]
        b64_t = load_image_b64(ev.get("image_path", "")) or fallback_b64
        if "SMOKE" in lbl_u:
            recent_items.append({"b64": b64_t, "icon": "💨", "col": "#F59E0B", "txt": f"Khói | {c_id}", "time": ts_only})
        else:
            recent_items.append({"b64": b64_t, "icon": "🔥", "col": "#EF4444", "txt": f"Cháy | {c_id}", "time": ts_only})

    default_recent = [
        {"b64": fallback_b64, "icon": "🔥", "col": "#EF4444", "txt": "Cháy | CAM-02", "time": "14:28:15"},
        {"b64": fallback_b64, "icon": "💨", "col": "#F59E0B", "txt": "Khói | CAM-03", "time": "14:27:32"},
        {"b64": fallback_b64, "icon": "✔", "col": "#10B981", "txt": "Bình thường | CAM-01", "time": "14:26:50"},
    ]
    while len(recent_items) < 3:
        recent_items.append(default_recent[len(recent_items)])

    rows_html_list: List[str] = []
    for item in recent_items[:3]:
        img_tag = (
            f'<img src="data:image/jpeg;base64,{item["b64"]}" style="width:58px; height:36px; object-fit:cover; border-radius:4px; border:1px solid #2A3448; display:block;" />'
            if item["b64"]
            else '<div style="width:58px; height:36px; background:#0E131D; border-radius:4px;"></div>'
        )
        rows_html_list.append(
            f"""
            <div style="display:flex; align-items:center; justify-content:space-between; padding:6px 8px; background:#161C28; border:1px solid #232C3D; border-radius:6px; margin-bottom:5px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    {img_tag}
                    <span style="font-size:12.5px; font-weight:700; color:#F8FAFC;">
                        <span style="color:{item['col']};">{item['icon']}</span> {esc(item['txt'])}
                    </span>
                </div>
                <span style="font-family:monospace; font-size:12px; color:#94A3B8;">{esc(item['time'])}</span>
            </div>
            """
        )

    render_html(
        f"""
        <div class="fw-panel-card" style="margin-top:8px; padding:9px 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
                <span style="font-size:13px; font-weight:800; color:#F8FAFC; border-left:3px solid #EF4444; padding-left:7px;">Sự kiện gần đây</span>
                <span style="font-size:11.5px; color:#CBD5E1;">Xem tất cả →</span>
            </div>
            {''.join(rows_html_list)}
        </div>
        """
    )

    # 3. Khối Biểu đồ sự kiện theo giờ (SVG đường cong phát sáng chuẩn ảnh mẫu V2)
    render_html(
        """
        <div class="fw-panel-card" style="margin-top:8px; padding:10px 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:13px; font-weight:800; color:#F8FAFC;">Biểu đồ sự kiện theo giờ</span>
                <div style="display:flex; align-items:center; gap:10px; font-size:11.5px; font-weight:700;">
                    <span style="color:#EF4444;">● <span style="color:#E2E8F0;">Cháy</span></span>
                    <span style="color:#F59E0B;">● <span style="color:#E2E8F0;">Khói</span></span>
                </div>
            </div>
            <svg width="100%" height="128" viewBox="0 0 340 125" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="fireGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#EF4444" stop-opacity="0.42"/>
                        <stop offset="100%" stop-color="#EF4444" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>
                <!-- Lưới ngang & Nhãn trục Y -->
                <line x1="28" y1="16" x2="326" y2="16" stroke="#222B3C" stroke-width="1"/>
                <line x1="28" y1="44" x2="326" y2="44" stroke="#222B3C" stroke-width="1"/>
                <line x1="28" y1="72" x2="326" y2="72" stroke="#222B3C" stroke-width="1"/>
                <line x1="28" y1="100" x2="326" y2="100" stroke="#222B3C" stroke-width="1"/>
                <text x="6" y="20" fill="#94A3B8" font-size="10">15</text>
                <text x="6" y="48" fill="#94A3B8" font-size="10">10</text>
                <text x="11" y="76" fill="#94A3B8" font-size="10">5</text>
                <text x="11" y="103" fill="#94A3B8" font-size="10">0</text>
                <!-- Vùng đổ bóng & Đường cong Cháy + Khói -->
                <path d="M 38 82 C 75 82, 95 79, 112 72 C 145 62, 165 36, 184 36 C 205 36, 225 65, 255 72 C 280 78, 305 74, 322 72 L 322 100 L 38 100 Z" fill="url(#fireGrad)"/>
                <path d="M 38 82 C 75 82, 95 79, 112 72 C 145 62, 165 36, 184 36 C 205 36, 225 65, 255 72 C 280 78, 305 74, 322 72" fill="none" stroke="#EF4444" stroke-width="2.4"/>
                <path d="M 38 84 C 75 83, 95 81, 112 76 C 145 68, 165 55, 184 55 C 205 55, 225 69, 255 74 C 280 79, 305 75, 322 73" fill="none" stroke="#F59E0B" stroke-width="1.8" stroke-opacity="0.85"/>
                <!-- Điểm nút -->
                <circle cx="38" cy="82" r="3" fill="#F59E0B"/>
                <circle cx="112" cy="72" r="3" fill="#F59E0B"/>
                <circle cx="184" cy="36" r="4" fill="#EF4444" stroke="#FCA5A5" stroke-width="1.2"/>
                <circle cx="255" cy="72" r="3" fill="#F59E0B"/>
                <circle cx="322" cy="72" r="3" fill="#F59E0B"/>
                <!-- Nhãn trục X -->
                <text x="30" y="118" fill="#CBD5E1" font-size="10.5">12h</text>
                <text x="103" y="118" fill="#CBD5E1" font-size="10.5">13h</text>
                <text x="175" y="118" fill="#CBD5E1" font-size="10.5">14h</text>
                <text x="246" y="118" fill="#CBD5E1" font-size="10.5">15h</text>
                <text x="310" y="118" fill="#CBD5E1" font-size="10.5">16h</text>
            </svg>
        </div>
        """
    )


# ==============================================================================
# 15. PHẦN 6: KHU VỰC THỐNG KÊ TỔNG QUAN & LỊCH SỬ SỰ KIỆN NẰM DƯỚI LƯỚI CAMERA
# ==============================================================================
def render_bottom_statistics_and_history() -> None:
    """Hiển thị 2 khối nằm ngay dưới lưới camera bên trái chuẩn ảnh thiết kế FIREWATCH V2:
    1. Thống kê tổng quan (5 thẻ dọc: Tổng camera, Đang hoạt động, Mất kết nối, Cảnh báo cháy, Cảnh báo khói).
    2. Lịch sử sự kiện (gần đây) (Bảng 5 cột).
    """
    b1, b2 = st.columns([1.06, 1.34])

    with b1:
        render_html(
            f"""
            <div class="fw-panel-card" style="padding:10px 12px;">
                <div class="fw-panel-title" style="margin-bottom:8px;">
                    <span style="display:flex; align-items:center; gap:6px;"><span style="color:#F97316;">🔥</span> Thống kê tổng quan</span>
                    <span style="color:#64748B; font-size:11px;">✕</span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:7px; text-align:center;">
                    <div style="background:linear-gradient(180deg, rgba(37,99,235,0.16) 0%, #141924 100%); border:1px solid rgba(59,130,246,0.45); border-radius:8px; padding:10px 4px;">
                        <div style="width:32px; height:28px; margin:0 auto 6px auto; border-radius:6px; background:rgba(59,130,246,0.22); display:flex; align-items:center; justify-content:center; font-size:15px;">📹</div>
                        <div style="font-size:20px; font-weight:900; color:#FFFFFF; line-height:1.1;">{total_configured_cams}</div>
                        <div style="font-size:10.5px; color:#CBD5E1; margin-top:4px;">Tổng camera</div>
                    </div>
                    <div style="background:linear-gradient(180deg, rgba(16,185,129,0.16) 0%, #141924 100%); border:1px solid rgba(16,185,129,0.45); border-radius:8px; padding:10px 4px;">
                        <div style="width:32px; height:28px; margin:0 auto 6px auto; border-radius:6px; background:rgba(16,185,129,0.22); display:flex; align-items:center; justify-content:center; font-size:15px;">🟢</div>
                        <div style="font-size:20px; font-weight:900; color:#10B981; line-height:1.1;">{running_cams_count}</div>
                        <div style="font-size:10.5px; color:#CBD5E1; margin-top:4px;">Đang hoạt động</div>
                    </div>
                    <div style="background:linear-gradient(180deg, rgba(239,68,68,0.16) 0%, #141924 100%); border:1px solid rgba(239,68,68,0.45); border-radius:8px; padding:10px 4px;">
                        <div style="width:32px; height:28px; margin:0 auto 6px auto; border-radius:6px; background:rgba(239,68,68,0.22); display:flex; align-items:center; justify-content:center; font-size:15px;">🚫</div>
                        <div style="font-size:20px; font-weight:900; color:#FFFFFF; line-height:1.1;">{stopped_cams_count}</div>
                        <div style="font-size:10.5px; color:#CBD5E1; margin-top:4px;">Mất kết nối</div>
                    </div>
                    <div style="background:linear-gradient(180deg, rgba(249,115,22,0.16) 0%, #141924 100%); border:1px solid rgba(249,115,22,0.45); border-radius:8px; padding:10px 4px;">
                        <div style="width:32px; height:28px; margin:0 auto 6px auto; border-radius:6px; background:rgba(249,115,22,0.22); display:flex; align-items:center; justify-content:center; font-size:15px;">🔥</div>
                        <div style="font-size:20px; font-weight:900; color:#FFFFFF; line-height:1.1;">{sys_metrics['fire_cam_count']}</div>
                        <div style="font-size:10.5px; color:#CBD5E1; margin-top:4px;">Cảnh báo cháy</div>
                    </div>
                    <div style="background:linear-gradient(180deg, rgba(234,179,8,0.16) 0%, #141924 100%); border:1px solid rgba(234,179,8,0.45); border-radius:8px; padding:10px 4px;">
                        <div style="width:32px; height:28px; margin:0 auto 6px auto; border-radius:6px; background:rgba(234,179,8,0.22); display:flex; align-items:center; justify-content:center; font-size:15px;">💨</div>
                        <div style="font-size:20px; font-weight:900; color:#FFFFFF; line-height:1.1;">{sys_metrics['suspect_cam_count']}</div>
                        <div style="font-size:10.5px; color:#CBD5E1; margin-top:4px;">Cảnh báo khói</div>
                    </div>
                </div>
            </div>
            """
        )

    with b2:
        recent_rows: List[str] = []
        for r in db_summary["real_records"][:2]:
            lbl_u = str(r.get("label", "FIRE")).upper()
            ts_short = str(r.get("timestamp", "")).split(" ")[-1] if r.get("timestamp") else "14:28:15"
            c_id = str(r.get("camera_name", "CAM-02")).split()[0]
            conf_r = round(float(r.get("confidence", 0.987)) * 100, 1)
            is_ack = (r.get("status") == "ACKNOWLEDGED")
            type_html = '<span style="color:#EF4444; font-weight:700;">🔥 Cháy</span>' if "SMOKE" not in lbl_u else '<span style="color:#F59E0B; font-weight:700;">💨 Khói</span>'
            stt_badge = (
                '<span style="background:#263042; color:#E2E8F0; padding:2px 9px; border-radius:4px; font-size:11px; font-weight:700;">Đã xác nhận</span>'
                if is_ack
                else '<span style="background:rgba(234,179,8,0.22); border:1px solid rgba(234,179,8,0.50); color:#FDE047; padding:2px 9px; border-radius:4px; font-size:11px; font-weight:700;">Chờ xử lý</span>'
            )
            recent_rows.append(
                f"""
                <tr style="border-bottom:1px solid #1E2534;">
                    <td style="padding:7px 6px; font-family:monospace; color:#CBD5E1;">{esc(ts_short)}</td>
                    <td style="padding:7px 6px; font-weight:700; color:#F8FAFC;">{esc(c_id)}</td>
                    <td style="padding:7px 6px;">{type_html}</td>
                    <td style="padding:7px 6px;">{stt_badge}</td>
                    <td style="padding:7px 6px; font-family:monospace; text-align:right; color:#F8FAFC; font-weight:700;">{conf_r:.1f}%</td>
                </tr>
                """
            )

        if not recent_rows:
            recent_rows.append(
                """
                <tr style="border-bottom:1px solid #1E2534;">
                    <td style="padding:7px 6px; font-family:monospace; color:#CBD5E1;">14:28:15</td>
                    <td style="padding:7px 6px; font-weight:700; color:#F8FAFC;">CAM-02</td>
                    <td style="padding:7px 6px;"><span style="color:#EF4444; font-weight:700;">🔥 Cháy</span></td>
                    <td style="padding:7px 6px;"><span style="background:rgba(234,179,8,0.22); border:1px solid rgba(234,179,8,0.50); color:#FDE047; padding:2px 9px; border-radius:4px; font-size:11px; font-weight:700;">Chờ xử lý</span></td>
                    <td style="padding:7px 6px; font-family:monospace; text-align:right; color:#F8FAFC; font-weight:700;">98.7%</td>
                </tr>
                <tr style="border-bottom:1px solid #1E2534;">
                    <td style="padding:7px 6px; font-family:monospace; color:#CBD5E1;">14:27:32</td>
                    <td style="padding:7px 6px; font-weight:700; color:#F8FAFC;">CAM-03</td>
                    <td style="padding:7px 6px;"><span style="color:#F59E0B; font-weight:700;">💨 Khói</span></td>
                    <td style="padding:7px 6px;"><span style="background:#263042; color:#E2E8F0; padding:2px 9px; border-radius:4px; font-size:11px; font-weight:700;">Đã xác nhận</span></td>
                    <td style="padding:7px 6px; font-family:monospace; text-align:right; color:#F8FAFC; font-weight:700;">88.2%</td>
                </tr>
                """
            )

        recent_rows.append(
            """
            <tr>
                <td style="padding:7px 6px; font-family:monospace; color:#CBD5E1;">14:26:50</td>
                <td style="padding:7px 6px; font-weight:700; color:#F8FAFC;">CAM-01</td>
                <td style="padding:7px 6px;"><span style="color:#10B981; font-weight:700;">✔ Bình thường</span></td>
                <td style="padding:7px 6px; color:#64748B;">-</td>
                <td style="padding:7px 6px; text-align:right; color:#64748B;">-</td>
            </tr>
            """
        )

        render_html(
            f"""
            <div class="fw-panel-card" style="padding:10px 14px;">
                <div class="fw-panel-title" style="margin-bottom:8px;">
                    <span style="border-left:3px solid #E2E8F0; padding-left:7px;">Lịch sử sự kiện (gần đây)</span>
                    <span style="color:#64748B; font-size:11px;">✕</span>
                </div>
                <table style="width:100%; border-collapse:collapse; font-size:12.5px;">
                    <thead>
                        <tr style="border-bottom:1px solid #262F40; color:#94A3B8; text-align:left;">
                            <th style="padding:4px 6px;">Thời gian</th>
                            <th style="padding:4px 6px;">Camera</th>
                            <th style="padding:4px 6px;">Loại</th>
                            <th style="padding:4px 6px;">Trạng thái</th>
                            <th style="padding:4px 6px; text-align:right;">Độ tin cậy</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(recent_rows[:3])}
                    </tbody>
                </table>
            </div>
            """
        )


# ==============================================================================
# 16. ĐIỀU PHỐI NỘI DUNG THEO 6 TRANG SIDEBAR (CHUẨN BỐ CỤC 2 CỘT TRÁI-PHẢI V2)
# ==============================================================================
_live_interval = max(0.45, float(cam_system.settings.ui_refresh_sec)) if st.session_state.is_monitoring else None


@st.fragment(run_every=_live_interval)
def render_live_camera_feed_fragment() -> None:
    """Fragment chỉ làm mới luồng video camera định kỳ khi đang giám sát (0ms delay cho thanh công cụ)."""
    live_snaps = cam_system.get_selected_snapshots(st.session_state.selected_cams)
    render_camera_workspace_grid(live_snaps)


@st.fragment(run_every=_live_interval)
def render_live_alert_panel_fragment() -> None:
    """Fragment làm mới còi hú và thẻ cảnh báo khẩn cấp bên phải."""
    render_alert_center_right()


def render_live_command_workspace() -> None:
    """Không gian điều hành FireWatch: Giữ thanh công cụ và nút thao tác ở ngoài fragment để phản hồi tức thì 100%."""
    global active_snaps, sys_metrics, db_summary, running_cams_count, stopped_cams_count, unack_count
    active_snaps = cam_system.get_selected_snapshots(st.session_state.selected_cams)
    sys_metrics = cam_system.get_system_metrics(active_snaps)
    db_summary = cam_system.event_db.get_summary_statistics()
    running_cams_count = sys_metrics["connected_count"]
    stopped_cams_count = max(0, total_configured_cams - running_cams_count)
    unack_count = db_summary["unconfirmed"]

    ws_left, ws_right = st.columns([2.42, 1.05])
    with ws_left:
        render_left_filter_bar()
        render_live_camera_feed_fragment()
        if st.session_state.nav_page == "Tổng quan":
            render_bottom_statistics_and_history()
    with ws_right:
        render_right_start_stop_bar()
        render_live_alert_panel_fragment()


if st.session_state.nav_page in ("Tổng quan", "Giám sát camera"):
    render_live_command_workspace()

elif st.session_state.nav_page == "Trung tâm cảnh báo":
    render_html('<div class="fw-panel-title"><span>🔔 Trung tâm Cảnh báo Chuyên nghiệp</span></div>')
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1.3, 1.0])
    with f1:
        f_cam = st.selectbox("Camera nguồn:", ["Tất cả"] + all_cam_ids)
    with f2:
        f_lbl = st.selectbox("Loại phát hiện:", ["Tất cả", "FIRE", "SMOKE"])
    with f3:
        f_stt = st.selectbox("Trạng thái xử lý:", ["Tất cả", "Chưa xác nhận", "Đã xác nhận"])
    with f4:
        st.write("")
        if st.button("🔕 Tắt còi báo động", use_container_width=True):
            force_stop_audio_alarm()
            cam_system.alert_manager.alarm_silenced_by_ack = True
            cam_system.alert_manager.stop_alarm()
            st.toast("Đã tắt âm thanh còi báo động.")

    alerts = cam_system.event_db.query_events(
        limit=200,
        camera_filter=f_cam,
        label_filter=f_lbl,
        status_filter=f_stt,
        include_demo=False,
    )
    if not alerts:
        st.info("Không có sự kiện cảnh báo nào khớp với bộ lọc.")
    else:
        for ev in alerts[:35]:
            ev_id = ev["id"]
            det_type, proc_state, col_ev = classify_event_category(ev)
            conf_pct = round(float(ev.get("confidence", 0.0)) * 100, 1)
            stt_raw = ev.get("status", "NEW")

            c_img, c_info, c_act = st.columns([1.2, 2.05, 1.1])
            with c_img:
                b64_img = load_image_b64(ev.get("image_path", ""))
                if b64_img:
                    render_html(
                        f'<div class="fw-alert-img-wrap"><img class="fw-alert-img" style="height:125px;" src="data:image/jpeg;base64,{b64_img}" alt="#{ev_id}" /></div>'
                    )
                else:
                    st.caption("Không có ảnh")
            with c_info:
                render_html(
                    f"""
                    <div style="font-size:13.5px; line-height:1.7;">
                        <div style="font-size:15px; font-weight:800; color:{col_ev};">
                            SỰ KIỆN #{ev_id} — {esc(det_type.upper())} ({conf_pct:.1f}%)
                        </div>
                        <div>• <b>Camera nguồn:</b> <code>{esc(ev.get('camera_name'))}</code></div>
                        <div>• <b>Thời gian:</b> <code>{esc(ev.get('timestamp'))}</code></div>
                        <div>• <b>Trạng thái xử lý:</b> <b>{esc(proc_state)}</b> ({esc(ev.get('action_taken') or 'Chưa xử lý')})</div>
                        <div>• <b>Ghi chú:</b> {esc(ev.get('operator_note') or 'Không có')}</div>
                    </div>
                    """
                )
            with c_act:
                if st.button("🔍 Xem ảnh lớn", key=f"ac_modal_{ev_id}", use_container_width=True):
                    st.session_state.modal_event_id = ev_id
                    st.rerun()
                if stt_raw != "ACKNOWLEDGED":
                    if st.button("✓ Xác nhận", key=f"ac_ack_{ev_id}", type="primary", use_container_width=True):
                        handle_alert_acknowledge(ev_id, action_taken="Đã xác nhận cháy thật", note="Xác nhận từ Trung tâm cảnh báo", is_dismiss=False)
                        st.toast("🔕 Đã xác nhận cảnh báo và tắt âm báo còi hú.")
                        st.rerun()
                    if st.button("✕ Bỏ qua", key=f"ac_dis_{ev_id}", use_container_width=True):
                        handle_alert_acknowledge(ev_id, note="Đã bỏ qua (Báo động giả)", is_dismiss=True)
                        st.toast("🔕 Đã bỏ qua cảnh báo và tắt âm báo còi hú.")
                        st.rerun()
            st.markdown("<hr style='border-color:#262F40; margin:8px 0;'>", unsafe_allow_html=True)

elif st.session_state.nav_page == "Lịch sử sự kiện":
    render_html('<div class="fw-panel-title"><span>📑 Lịch sử Sự kiện, Thống kê & Biểu đồ Phân tích</span></div>')
    render_bottom_statistics_and_history()
    st.markdown("<hr style='border-color:#262F40; margin:12px 0;'>", unsafe_allow_html=True)

    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        tl_cam = st.selectbox("Theo camera:", ["Tất cả"] + all_cam_ids)
    with h2:
        tl_lbl = st.selectbox("Theo loại sự kiện:", ["Tất cả", "FIRE", "SMOKE"])
    with h3:
        tl_stt = st.selectbox("Theo trạng thái:", ["Tất cả", "Chưa xác nhận", "Đã xác nhận"])
    with h4:
        tl_start = st.text_input("Từ ngày (YYYY-MM-DD):", value="", placeholder="2026-11-01")
    with h5:
        tl_end = st.text_input("Đến ngày (YYYY-MM-DD):", value="", placeholder="2026-12-31")

    events = cam_system.event_db.query_events(
        limit=500,
        camera_filter=tl_cam,
        label_filter=tl_lbl,
        status_filter=tl_stt,
        start_date=tl_start.strip(),
        end_date=tl_end.strip(),
        include_demo=False,
    )
    if not events:
        st.info("Không tìm thấy sự kiện nào khớp với bộ lọc.")
    else:
        df_ev = pd.DataFrame(events)
        csv_bytes = df_ev.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Xuất báo cáo lịch sử sự kiện (CSV)",
            data=csv_bytes,
            file_name=f"firewatch_events_{int(time.time())}.csv",
            mime="text/csv",
        )
        st.dataframe(
            df_ev[["id", "timestamp", "camera_name", "label", "alert_level", "confidence", "status", "action_taken", "operator_note"]],
            use_container_width=True,
            hide_index=True,
        )

elif st.session_state.nav_page == "Quản lý camera và ROI":
    # PHẦN 7: Không tự mở stream khi vào trang quản lý; nút kiểm tra kết nối là thao tác chủ động
    render_html('<div class="fw-panel-title"><span>🎥 Quản lý Danh mục Camera, RTSP & Cấu hình Vùng loại trừ ROI</span></div>')
    st.caption(
        "🔒 Bảo mật & Vận hành: Hệ thống không tự mở stream khi vào trang quản lý. "
        "Mọi thông tin tài khoản/mật khẩu trong URL RTSP đều được ẩn tự động (`******`)."
    )

    m_top1, m_top2 = st.columns(2)
    with m_top1:
        with st.expander("➕ Thêm camera mới vào hệ thống", expanded=False):
            new_c_name = st.text_input("Tên camera mới:", value="Khu vực mở rộng")
            new_c_type = st.selectbox("Loại nguồn tín hiệu:", SOURCE_TYPES, key="add_cam_type")
            if new_c_type == "WEBCAM":
                new_c_src = st.text_input("Chỉ số Webcam (0, 1, 2...):", value="0", key="add_cam_src_w")
            elif new_c_type == "IP_CAMERA":
                new_c_src = st.text_input("Đường dẫn RTSP / HTTP Stream:", value="", type="password", key="add_cam_src_rtsp")
            else:
                new_c_src = st.text_input("Đường dẫn tệp Video:", value="", placeholder="Nhập đường dẫn hoặc dùng nút Test Cam File...", key="add_cam_src_v")

            if st.button("Thêm camera (Trạng thái STOPPED)", type="primary", use_container_width=True):
                added_id = cam_system.add_camera(
                    name=new_c_name,
                    source=new_c_src,
                    source_type=new_c_type,
                    webcam_index=int(new_c_src) if new_c_src.isdigit() else 0,
                    enabled=True,
                    auto_start=False,
                )
                st.success(f"Đã thêm camera mới: {added_id} (Trạng thái: Đang chờ kết nối).")
                st.rerun()

    with m_top2:
        with st.expander("⭐ Chọn danh sách camera mặc định", expanded=False):
            def_cams = st.multiselect(
                "Chọn các camera mặc định:",
                options=all_cam_ids,
                default=[c for c in cam_system.default_camera_ids if c in all_cam_ids],
            )
            if st.button("Lưu danh sách mặc định", type="primary", use_container_width=True):
                cam_system.set_default_cameras(def_cams)
                st.success(f"Đã lưu camera mặc định: {def_cams}")

    for cfg in list(cam_system.camera_configs):
        cid = cfg.cam_id
        snap = cam_system.workers[cid].get_snapshot()
        _, lc_vn, _ = get_camera_lifecycle_info(snap, st.session_state.is_monitoring)
        with st.expander(
            f"{cid} – {cfg.name} | Trạng thái: {lc_vn} | Nguồn: {cfg.masked_source}",
            expanded=(cid == "CAM-01"),
        ):
            c_left, c_mid, c_right = st.columns([1.45, 1.3, 1.25])
            with c_left:
                ed_name = st.text_input("Tên camera:", value=cfg.name, key=f"ed_nm_{cid}")
                ed_type = st.selectbox(
                    "Loại nguồn:",
                    SOURCE_TYPES,
                    index=SOURCE_TYPES.index(cfg.source_type) if cfg.source_type in SOURCE_TYPES else 0,
                    key=f"ed_tp_{cid}",
                )
                if ed_type == "IP_CAMERA":
                    ed_src = st.text_input(
                        f"RTSP URL (Hiện tại: {cfg.masked_source}):",
                        value="",
                        placeholder="Nhập RTSP URL mới nếu muốn thay đổi...",
                        type="password",
                        key=f"ed_rtsp_{cid}",
                    )
                    final_src = ed_src.strip() if ed_src.strip() else cfg.source
                elif ed_type == "WEBCAM":
                    sys_cams = list_system_cameras(4)
                    cam_opts = {c["label"]: str(c["index"]) for c in sys_cams}
                    picked_w = st.selectbox("Thiết bị Webcam:", list(cam_opts.keys()), key=f"ed_wb_{cid}")
                    final_src = cam_opts[picked_w]
                else:
                    final_src = st.text_input("Đường dẫn tệp Video:", value=cfg.source, key=f"ed_vid_{cid}")

                ed_en = st.checkbox("Cho phép kích hoạt camera", value=cfg.enabled, key=f"ed_en_{cid}")
                if st.button("💾 Lưu cấu hình", key=f"save_cfg_{cid}", type="primary", use_container_width=True):
                    cam_system.update_camera_source(
                        cam_id=cid,
                        new_name=ed_name,
                        new_source=final_src,
                        source_type=ed_type,
                        webcam_index=int(final_src) if str(final_src).isdigit() else 0,
                        enabled=ed_en,
                        auto_restart=False,
                    )
                    st.success(f"Đã lưu cấu hình {cid} (không tự mở stream).")
                    st.rerun()

            with c_mid:
                st.markdown("**Cấu hình vùng loại trừ ROI (x1, y1, x2, y2 trong [0, 1]):**")
                curr_zones_str = "\n".join(
                    [f"{z[0]:.2f}, {z[1]:.2f}, {z[2]:.2f}, {z[3]:.2f}" for z in cfg.exclusion_zones]
                )
                roi_text = st.text_area(
                    "Mỗi dòng 1 vùng (ví dụ: 0.0, 0.0, 0.25, 0.20):",
                    value=curr_zones_str,
                    height=90,
                    key=f"roi_txt_{cid}",
                )
                if st.button("Lưu vùng ROI", key=f"save_roi_{cid}", use_container_width=True):
                    parsed_zones: List[Tuple[float, float, float, float]] = []
                    for line in roi_text.strip().splitlines():
                        parts = [p.strip() for p in line.split(",") if p.strip()]
                        if len(parts) == 4:
                            try:
                                parsed_zones.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
                            except ValueError:
                                pass
                    cam_system.update_camera_roi(cid, parsed_zones)
                    st.success(f"Đã lưu {len(parsed_zones)} vùng ROI cho {cid}")
                    st.rerun()

            with c_right:
                st.markdown("**Kiểm tra trạng thái & Xóa camera:**")
                if st.button("🔌 Kiểm tra kết nối (Chủ động)", key=f"test_conn_{cid}", use_container_width=True):
                    ok, msg, prev_rgb = cam_system.test_camera_connection(cid)
                    if ok:
                        st.success(msg)
                        st.image(prev_rgb, use_container_width=True)
                    else:
                        st.error(msg)

                if st.button(f"🔌 Gỡ nguồn tín hiệu khỏi {cid}", key=f"btn_detach_mgmt_{cid}", use_container_width=True):
                    cam_system.detach_camera_source(cid)
                    st.success(f"Đã gỡ nguồn khỏi {cid}.")
                    st.rerun()

                confirm_del = st.checkbox(f"Xác nhận xóa cấu hình {cid}", key=f"chk_del_{cid}")
                if st.button(
                    f"🗑️ Xóa camera {cid}",
                    key=f"btn_del_{cid}",
                    disabled=(not confirm_del or len(cam_system.camera_configs) <= 1),
                    use_container_width=True,
                ):
                    if cam_system.remove_camera(cid):
                        if cid in st.session_state.selected_cams:
                            st.session_state.selected_cams.remove(cid)
                        if cid in st.session_state.pending_cams:
                            st.session_state.pending_cams.remove(cid)
                        st.success(f"Đã xóa camera {cid}.")
                        st.rerun()

elif st.session_state.nav_page == "Cấu hình hệ thống":
    render_html('<div class="fw-panel-title"><span>⚙️ Cấu hình Mô hình Phát hiện YOLO, Chuỗi Thời gian K/N & Hiệu năng</span></div>')
    cfg_c1, cfg_c2 = st.columns(2)
    with cfg_c1:
        model_choices = get_available_model_choices()
        active_path = (
            getattr(cam_system.detector, "resolved_path", None)
            or getattr(cam_system.detector, "requested_path", None)
            or cam_system.settings.model_path
        )
        curr_model_path = str(Path(active_path).resolve()) if active_path else ""
        choice_labels = list(model_choices.keys())
        curr_idx = 0
        for i, lbl in enumerate(choice_labels):
            if str(Path(model_choices[lbl]).resolve()) == curr_model_path:
                curr_idx = i
                break

        picked_model_lbl = st.selectbox("Trọng số mô hình YOLO (Shared Detector):", choice_labels, index=curr_idx)
        target_model_path = model_choices[picked_model_lbl]
        if str(Path(target_model_path).resolve()) != curr_model_path:
            cam_system.switch_model(target_model_path)
            st.rerun()

        cam_system.settings.conf_threshold = st.slider(
            "Ngưỡng độ tin cậy nhận diện (Confidence):",
            min_value=0.15,
            max_value=0.90,
            value=float(cam_system.settings.conf_threshold),
            step=0.05,
        )

        p1, p2 = st.columns(2)
        with p1:
            cam_system.settings.imgsz = int(
                st.selectbox(
                    "Kích thước ảnh suy luận YOLO (imgsz):",
                    options=[320, 416, 512, 640],
                    index=[320, 416, 512, 640].index(cam_system.settings.imgsz)
                    if cam_system.settings.imgsz in (320, 416, 512, 640)
                    else 1,
                )
            )
        with p2:
            cam_system.settings.frame_skip = int(
                st.number_input(
                    "Chu kỳ suy luận (frame_skip):",
                    min_value=1,
                    max_value=6,
                    value=int(cam_system.settings.frame_skip),
                )
            )

        k1, k2, k3 = st.columns(3)
        with k1:
            cam_system.settings.suspect_frames_k = st.number_input(
                "Ngưỡng Nghi ngờ (k_suspect):",
                min_value=1,
                max_value=10,
                value=int(cam_system.settings.suspect_frames_k),
            )
        with k2:
            cam_system.settings.confirm_frames_k = st.number_input(
                "Ngưỡng Báo cháy (k_confirm):",
                min_value=int(cam_system.settings.suspect_frames_k),
                max_value=20,
                value=max(int(cam_system.settings.confirm_frames_k), int(cam_system.settings.suspect_frames_k)),
            )
        with k3:
            cam_system.settings.window_size_n = st.number_input(
                "Cửa sổ trượt (N khung):",
                min_value=int(cam_system.settings.confirm_frames_k),
                max_value=30,
                value=max(int(cam_system.settings.window_size_n), int(cam_system.settings.confirm_frames_k)),
            )

        cam_system.settings.alert_cooldown_sec = st.slider(
            "Thời gian Cooldown chống lặp cảnh báo (giây):",
            min_value=2.0,
            max_value=60.0,
            value=float(cam_system.settings.alert_cooldown_sec),
            step=1.0,
        )

        mute_val = st.checkbox("Tắt âm thanh còi báo động", value=cam_system.settings.mute_audio)
        cam_system.settings.mute_audio = mute_val
        cam_system.alert_manager.update_config(mute_val)

    with cfg_c2:
        render_html('<div class="fw-panel-title"><span>Thông tin Phần cứng & Nhật ký Hệ thống</span></div>')
        m_info = cam_system.detector.get_model_info()
        st.markdown(
            f"""
            - **Tệp trọng số đang nạp:** `{m_info.get('path')}`
            - **Thiết bị xử lý:** `{m_info.get('device')}`
            - **Các lớp nhận diện:** `{m_info.get('names')}`
            - **Chế độ hiệu năng:** `{perf_map_vn.get(st.session_state.perf_preset, st.session_state.perf_preset)}` (`imgsz={cam_system.settings.imgsz}`, `frame_skip={cam_system.settings.frame_skip}`)
            - **Số worker đang chạy thực tế:** `{cam_system.get_active_worker_count()}`
            """
        )
        logs = cam_system.get_recent_logs(limit=20)
        st.text_area("Nhật ký vận hành trạm (Đã ẩn mật khẩu RTSP):", value="\n".join(logs), height=220, disabled=True)
