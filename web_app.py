"""Ứng dụng Web Dashboard Giám Sát & Cảnh Báo Cháy Đa Luồng Camera AI (Redesigned).

Phong cách thiết kế: Professional CCTV Control Room (Dark Mode: #0B1120).
Bảo toàn 100% kiến trúc:
CameraStreamWorker -> SharedFireSmokeDetector -> ByteTrackAdapter -> TemporalFeatureExtractor -> TemporalVerifier -> EventDatabase -> Dashboard

Tính năng chính:
- Dynamic Camera Selection: 1 camera (SINGLE), 4 camera (QUAD), 6 camera (HEXA), 9 camera (FULL), hoặc tùy chọn thủ công bất kỳ (CAM-01 .. CAM-09).
- Quản lý tài nguyên nghiêm ngặt: Camera không được chọn sẽ dừng hoàn toàn worker, giải phóng VideoCapture, không tốn GPU/CPU.
- Hiển thị đầy đủ thông số thời gian thực: Bounding Box, Track ID, FIRE/SMOKE, Confidence, Capture FPS, Inference FPS, Trạng thái kết nối.
- Bảng điều khiển cảnh báo (Alert Panel), Lịch sử sự kiện (Event History) với bộ lọc đa chiều và xem chi tiết snapshot.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CANDIDATE_MODEL_PATHS,
    DEFAULT_USERS,
    OPERATING_MODES,
    SOURCE_TYPES,
    get_available_model_choices,
    mask_rtsp_credentials,
)
from src.camera_manager import MultiCameraManager, list_system_cameras

# ==============================================================================
# 1. CẤU HÌNH TRANG & THEME PROFESSIONAL CCTV CONTROL ROOM (#0B1120)
# ==============================================================================
st.set_page_config(
    page_title="FIRE DETECTION SYSTEM - CCTV CONTROL ROOM",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Nền chính và màu chữ tổng thể */
    .stApp {
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar styling: #111827 */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Control Room Top Header */
    .control-room-header {
        background: linear-gradient(90deg, #111827 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid #334155;
        border-left: 6px solid #3B82F6;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #F8FAFC;
    }
    .header-metrics {
        display: flex;
        gap: 18px;
        align-items: center;
        font-size: 13px;
        color: #94A3B8;
    }
    .metric-badge {
        background: #0B1120;
        border: 1px solid #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        color: #38BDF8;
    }

    /* KPI Summary Cards: #1E293B */
    .kpi-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .kpi-lbl {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 2px;
    }
    .kpi-val {
        font-size: 22px;
        font-weight: 800;
        color: #F8FAFC;
    }

    /* Camera Card Grid */
    .cam-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 12px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35);
    }
    .cam-card-top {
        background: #0F172A;
        padding: 6px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #334155;
        font-size: 12px;
        font-weight: 700;
    }
    .cam-card-btm {
        background: #0B1120;
        padding: 6px 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: #94A3B8;
        border-top: 1px solid #1E293B;
    }

    /* Status Badges */
    .badge-safe {
        background: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        border: 1px solid #22C55E;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-fire {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 800;
        animation: blink 1s infinite;
    }
    .badge-smoke {
        background: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-conn-err {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
    }
    .badge-standby {
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid #64748B;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
    }

    /* Alerts Panel */
    .alert-item {
        background: #1E293B;
        border-left: 4px solid #EF4444;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        font-size: 12px;
    }
    .alert-item.smoke {
        border-left-color: #F59E0B;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. KHỞI TẠO SINGLETON MULTI-CAMERA MANAGER
# ==============================================================================
@st.cache_resource
def get_control_room_system() -> MultiCameraManager:
    """Khởi tạo duy nhất 1 bộ điều phối trung tâm MultiCameraManager."""
    return MultiCameraManager(db_path="alerts.db", model_path="best.pt")


cam_system = get_control_room_system()

# Khởi tạo trạng thái phiên làm việc Streamlit
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "QUAD"  # "SINGLE" | "QUAD" | "HEXA" | "FULL" | "CUSTOM"
if "selected_cams" not in st.session_state:
    st.session_state.selected_cams = ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]
if "expanded_cam" not in st.session_state:
    st.session_state.expanded_cam = None
if "live_stream_active" not in st.session_state:
    st.session_state.live_stream_active = True
if "perf_mode" not in st.session_state:
    st.session_state.perf_mode = "Balanced"


# ==============================================================================
# 3. SIDEBAR: ĐIỀU HƯỚNG, CẤU HÌNH CAMERA ĐỘNG & THAM SỐ PHÁT HIỆN
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-size:28px;">🔥</span>
            <div>
                <div style="color:#F8FAFC; font-weight:800; font-size:16px;">FIRE & SMOKE SENTINEL</div>
                <div style="color:#3B82F6; font-size:11px; font-weight:700;">CCTV CONTROL ROOM PRO</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- NHÓM 3: NAVIGATION ---
    st.markdown("### 📌 ĐIỀU HƯỚNG TRUNG TÂM")
    nav_view = st.radio(
        "Chọn chế độ xem:",
        [
            "📺 Live Monitoring",
            "📋 Event History",
            "⚙️ Camera Management",
            "⚡ System Performance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # --- NHÓM 1: DYNAMIC CAMERA CONFIGURATION ---
    st.markdown("### 📹 CẤU HÌNH CAMERA ĐỘNG")
    
    # 1. Chọn chế độ lưới động (Preset Grid Modes)
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("1 Cam", use_container_width=True, type="primary" if st.session_state.active_mode == "SINGLE" else "secondary"):
            st.session_state.active_mode = "SINGLE"
            st.session_state.selected_cams = ["CAM-01"]
            cam_system.set_active_camera_selection(st.session_state.selected_cams, auto_start=True)
            st.rerun()
    with preset_cols[1]:
        if st.button("4 Cam", use_container_width=True, type="primary" if st.session_state.active_mode == "QUAD" else "secondary"):
            st.session_state.active_mode = "QUAD"
            st.session_state.selected_cams = ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]
            cam_system.set_active_camera_selection(st.session_state.selected_cams, auto_start=True)
            st.rerun()
    with preset_cols[2]:
        if st.button("6 Cam", use_container_width=True, type="primary" if st.session_state.active_mode == "HEXA" else "secondary"):
            st.session_state.active_mode = "HEXA"
            st.session_state.selected_cams = [f"CAM-0{i}" for i in range(1, 7)]
            cam_system.set_active_camera_selection(st.session_state.selected_cams, auto_start=True)
            st.rerun()
    with preset_cols[3]:
        if st.button("9 Cam", use_container_width=True, type="primary" if st.session_state.active_mode == "FULL" else "secondary"):
            st.session_state.active_mode = "FULL"
            st.session_state.selected_cams = [f"CAM-0{i}" for i in range(1, 10)]
            cam_system.set_active_camera_selection(st.session_state.selected_cams, auto_start=True)
            st.rerun()

    # 2. Chọn camera cụ thể (CAM-01 .. CAM-09)
    all_cam_ids = [c.cam_id for c in cam_system.camera_configs]
    custom_cams = st.multiselect(
        "Chọn danh sách camera giám sát:",
        options=all_cam_ids,
        default=st.session_state.selected_cams,
        key="ms_cams",
    )
    if custom_cams != st.session_state.selected_cams and custom_cams:
        st.session_state.selected_cams = custom_cams
        st.session_state.active_mode = "CUSTOM"
        cam_system.set_active_camera_selection(custom_cams, auto_start=True)
        st.rerun()

    # 3. Nút điều khiển Start / Stop
    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("▶ Start Monitoring", type="primary", use_container_width=True):
            cam_system.start_all(only_enabled=True)
            st.session_state.live_stream_active = True
            st.rerun()
    with c_btn2:
        if st.button("⏹ Stop Monitoring", use_container_width=True):
            cam_system.stop_all()
            st.session_state.live_stream_active = False
            st.rerun()

    st.markdown("---")

    # --- NHÓM 2: DETECTION SETTINGS ---
    st.markdown("### 🧠 DETECTION & TEMPORAL SETTINGS")

    # Chọn mô hình đã huấn luyện (MT2)
    model_choices = get_available_model_choices()
    active_path = getattr(cam_system.detector, "resolved_path", None) or getattr(cam_system.detector, "requested_path", None) or cam_system.settings.model_path
    curr_model_path = str(Path(active_path).resolve()) if active_path else ""
    curr_idx = 0
    choice_labels = list(model_choices.keys())
    for idx_m, lbl_m in enumerate(choice_labels):
        if str(Path(model_choices[lbl_m]).resolve()) == curr_model_path:
            curr_idx = idx_m
            break
    chosen_label = st.selectbox("Checkpoint Model:", choice_labels, index=curr_idx)
    target_path = model_choices[chosen_label]
    if str(Path(target_path).resolve()) != curr_model_path:
        cam_system.switch_model(target_path)
        st.rerun()

    cam_system.settings.conf_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.15,
        max_value=0.90,
        value=float(cam_system.settings.conf_threshold),
        step=0.05,
    )

    col_k, col_n = st.columns(2)
    with col_k:
        cam_system.settings.confirm_frames_k = st.number_input(
            "Xác nhận K:",
            min_value=2,
            max_value=15,
            value=int(cam_system.settings.confirm_frames_k),
            step=1,
        )
    with col_n:
        cam_system.settings.window_size_n = st.number_input(
            "Cửa sổ N:",
            min_value=int(cam_system.settings.confirm_frames_k),
            max_value=25,
            value=max(int(cam_system.settings.window_size_n), int(cam_system.settings.confirm_frames_k)),
            step=1,
        )

    # Preset Performance
    st.markdown("### ⚡ FPS OPTIMIZATION PRESET")
    perf_choice = st.radio(
        "Chế độ tài nguyên:",
        ["Low Resource", "Balanced", "High Performance"],
        index=["Low Resource", "Balanced", "High Performance"].index(st.session_state.perf_mode),
        horizontal=True,
    )
    if perf_choice != st.session_state.perf_mode:
        st.session_state.perf_mode = perf_choice
        if perf_choice == "Low Resource":
            cam_system.settings.imgsz = 320
            cam_system.settings.capture_width = 380
            cam_system.settings.frame_skip = 3
            cam_system.settings.jpeg_quality = 65
        elif perf_choice == "Balanced":
            cam_system.settings.imgsz = 416
            cam_system.settings.capture_width = 480
            cam_system.settings.frame_skip = 2
            cam_system.settings.jpeg_quality = 72
        else:
            cam_system.settings.imgsz = 640
            cam_system.settings.capture_width = 640
            cam_system.settings.frame_skip = 1
            cam_system.settings.jpeg_quality = 85
        st.rerun()

    mute_box = st.checkbox("🔕 Mute Alarm Sound", value=cam_system.settings.mute_audio)
    cam_system.settings.mute_audio = mute_box
    cam_system.alert_manager.update_config(mute_box)


# ==============================================================================
# 4. TOP CONTROL ROOM HEADER & KPI BANNER
# ==============================================================================
active_cam_ids = st.session_state.selected_cams
snaps = cam_system.get_selected_snapshots(active_cam_ids)
sys_metrics = cam_system.get_system_metrics(snaps)

# Lấy trạng thái hệ thống
sys_status_str = "NORMAL"
sys_status_color = "#22C55E"
if sys_metrics["overall_level"] == "FIRE":
    sys_status_str = "FIRE ALARM DETECTED"
    sys_status_color = "#EF4444"
elif sys_metrics["overall_level"] == "SUSPECT":
    sys_status_str = "SUSPECT ACTIVITY"
    sys_status_color = "#F59E0B"
elif sys_metrics["overall_level"] == "CONN_ERROR":
    sys_status_str = "CAMERA SIGNAL LOSS"
    sys_status_color = "#EF4444"

cur_time_str = time.strftime("%H:%M:%S | %d/%m/%Y")

st.markdown(
    f"""
    <div class="control-room-header">
        <div>
            <div class="header-title">🔥 FIRE DETECTION SYSTEM — CCTV CONTROL ROOM</div>
            <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">
                ByteTrack Real-Time Tracking & Temporal Sliding Window (k={cam_system.settings.confirm_frames_k}/N={cam_system.settings.window_size_n})
            </div>
        </div>
        <div class="header-metrics">
            <div>STATUS: <span style="color:{sys_status_color}; font-weight:800;">{sys_status_str}</span></div>
            <div>ACTIVE: <span class="metric-badge">{len(active_cam_ids)} CAM</span></div>
            <div>TOTAL CAPTURE: <span class="metric-badge">{sys_metrics['total_fps']} FPS</span></div>
            <div>AVG YOLO: <span class="metric-badge">{sys_metrics['avg_infer_fps']} FPS</span></div>
            <div>TIME: <span style="color:#E2E8F0; font-weight:600;">{cur_time_str}</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Thẻ KPI Summary
db_stats = cam_system.event_db.get_summary_statistics()
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
with kpi_c1:
    st.markdown(f'<div class="kpi-box"><div class="kpi-lbl">TỔNG CAMERA CHỌN</div><div class="kpi-val" style="color:#3B82F6;">{len(active_cam_ids)}</div></div>', unsafe_allow_html=True)
with kpi_c2:
    st.markdown(f'<div class="kpi-box"><div class="kpi-lbl">CAMERA ĐANG CHẠY</div><div class="kpi-val" style="color:#22C55E;">{sys_metrics["connected_count"]}</div></div>', unsafe_allow_html=True)
with kpi_c3:
    st.markdown(f'<div class="kpi-box"><div class="kpi-lbl">CẢNH BÁO FIRE</div><div class="kpi-val" style="color:#EF4444;">{sys_metrics["fire_cam_count"]}</div></div>', unsafe_allow_html=True)
with kpi_c4:
    st.markdown(f'<div class="kpi-box"><div class="kpi-lbl">NGHI NGỜ SMOKE</div><div class="kpi-val" style="color:#F59E0B;">{sys_metrics["suspect_cam_count"]}</div></div>', unsafe_allow_html=True)
with kpi_c5:
    st.markdown(f'<div class="kpi-box"><div class="kpi-lbl">FPS ĐỌC TRUNG BÌNH</div><div class="kpi-val" style="color:#38BDF8;">{sys_metrics["avg_fps"]}</div></div>', unsafe_allow_html=True)
with kpi_c6:
    st.markdown(f'<div class="kpi-box"><div class="kpi-lbl">SỰ KIỆN LỊCH SỬ</div><div class="kpi-val" style="color:#A855F7;">{db_stats["total_real"]}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# 5. NỘI DUNG CHÍNH THEO ĐIỀU HƯỚNG
# ==============================================================================

if nav_view == "📺 Live Monitoring":
    col_grid, col_alert = st.columns([3.3, 1.2])

    with col_grid:
        # Nếu có camera đang được phóng to
        exp_id = st.session_state.expanded_cam
        if exp_id and exp_id in [s["cam_id"] for s in snaps]:
            exp_snap = next(s for s in snaps if s["cam_id"] == exp_id)
            st.markdown(f"### 🔍 CAMERA PHÓNG TO: {exp_id} ({exp_snap['name']})")
            if st.button("⬅️ Quay lại chế độ lưới", type="secondary"):
                st.session_state.expanded_cam = None
                st.rerun()

            img_b64 = exp_snap["jpeg_b64"]
            st.markdown(
                f"""
                <div class="cam-card">
                    <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px; display:block;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Layout lưới động tùy theo số lượng camera được chọn
            n_cams = len(snaps)
            if n_cams == 1:
                grid_cols = 1
            elif n_cams <= 4:
                grid_cols = 2
            else:
                grid_cols = 3

            rows = (n_cams + grid_cols - 1) // grid_cols
            cam_idx = 0
            for r in range(rows):
                cols = st.columns(grid_cols)
                for c in range(grid_cols):
                    if cam_idx < n_cams:
                        snap_item = snaps[cam_idx]
                        cid = snap_item["cam_id"]
                        cname = snap_item["name"]
                        stt = snap_item["status"]
                        lvl = snap_item["alert_level"]
                        cap_fps = snap_item["capture_fps"]
                        inf_fps = snap_item["inference_fps"]
                        inf_ms = snap_item["infer_ms"]
                        b64 = snap_item["jpeg_b64"]
                        boxes_n = snap_item["box_count"]

                        # Badge trạng thái
                        if lvl == "FIRE":
                            badge_html = '<span class="badge-fire">🚨 FIRE ALARM</span>'
                        elif lvl == "SUSPECT":
                            badge_html = '<span class="badge-smoke">⚠️ SUSPECT</span>'
                        elif stt == "CONNECTED":
                            badge_html = '<span class="badge-safe">🟢 CONNECTED</span>'
                        elif stt == "RECONNECTING":
                            badge_html = '<span class="badge-conn-err">🔄 RECONNECT</span>'
                        elif stt == "ERROR":
                            badge_html = '<span class="badge-conn-err">🔴 ERROR</span>'
                        else:
                            badge_html = '<span class="badge-standby">⚪ STANDBY</span>'

                        with cols[c]:
                            st.markdown(
                                f"""
                                <div class="cam-card">
                                    <div class="cam-card-top">
                                        <span>📹 {cid}: {cname[:22]}</span>
                                        {badge_html}
                                    </div>
                                    <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:auto; display:block; aspect-ratio:16/9; object-fit:cover;" />
                                    <div class="cam-card-btm">
                                        <span>Cap: <b>{cap_fps:.1f} FPS</b> | YOLO: <b>{inf_fps:.1f} FPS</b> ({inf_ms:.1f}ms)</span>
                                        <span>Tracks: <b>{boxes_n}</b></span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            # Các nút điều khiển camera
                            c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)
                            with c_ctrl1:
                                if st.button("🔍", key=f"zoom_{cid}", help="Phóng to camera này", use_container_width=True):
                                    st.session_state.expanded_cam = cid
                                    st.rerun()
                            with c_ctrl2:
                                if st.button("🔌", key=f"start_{cid}", help="Bắt đầu / Kết nối lại", use_container_width=True):
                                    cam_system.start_camera(cid)
                                    st.rerun()
                            with c_ctrl3:
                                if st.button("⏸", key=f"pause_{cid}", help="Tạm dừng / Tiếp tục", use_container_width=True):
                                    cam_system.pause_camera(cid)
                                    st.rerun()
                            with c_ctrl4:
                                if st.button("⏹", key=f"stop_{cid}", help="Ngắt kết nối", use_container_width=True):
                                    cam_system.stop_camera(cid)
                                    st.rerun()

                        cam_idx += 1

    with col_alert:
        st.markdown("### 🚨 REAL-TIME ALERTS PANEL")
        recent_evs = cam_system.event_db.query_events(limit=6, include_demo=False)
        if not recent_evs:
            st.info("Chưa có cảnh báo cháy/khói nào được kích hoạt.")
        else:
            for ev in recent_evs:
                is_fire = "FIRE" in str(ev.get("label", "")).upper()
                c_border = "#EF4444" if is_fire else "#F59E0B"
                cls_type = "" if is_fire else "smoke"
                st.markdown(
                    f"""
                    <div class="alert-item {cls_type}" style="border-left-color:{c_border};">
                        <div style="display:flex; justify-content:space-between; font-weight:800; color:{c_border};">
                            <span>#{ev['id']} • {ev.get('label')} ({int(ev.get('confidence', 0)*100)}%)</span>
                            <span>{ev.get('camera_name')}</span>
                        </div>
                        <div style="color:#94A3B8; font-size:11px; margin-top:2px;">
                            Thời gian: <code>{ev.get('timestamp')}</code><br>
                            Mức: <b>{ev.get('alert_level')}</b> | Trạng thái: <b>{ev.get('status')}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                img_p = ev.get("image_path", "")
                if img_p and os.path.exists(img_p):
                    st.image(img_p, use_container_width=True)
                if st.button("✅ Xác nhận đã xử lý", key=f"ack_{ev['id']}", use_container_width=True):
                    cam_system.event_db.confirm_event(ev["id"], action_taken="Đã kiểm tra hiện trường", note="Xác nhận từ Dashboard")
                    st.success(f"Đã xác nhận sự kiện #{ev['id']}")
                    st.rerun()

    # Tự động refresh luồng
    if st.session_state.live_stream_active and sys_metrics["connected_count"] > 0:
        time.sleep(max(0.25, float(cam_system.settings.ui_refresh_sec)))
        st.rerun()


elif nav_view == "📋 Event History":
    st.markdown("## 📋 LỊCH SỬ CẢNH BÁO CHÁY & KHÓI (SQLITE DATABASE)")
    
    # Bộ lọc
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        f_cam = st.selectbox("Lọc theo Camera:", ["Tất cả"] + all_cam_ids)
    with f_c2:
        f_label = st.selectbox("Lọc theo Loại:", ["Tất cả", "FIRE", "SMOKE"])
    with f_c3:
        f_status = st.selectbox("Trạng thái:", ["Tất cả", "Chưa xác nhận", "Đã xác nhận"])
    with f_c4:
        f_limit = st.slider("Số lượng bản ghi:", min_value=10, max_value=200, value=50)

    events = cam_system.event_db.query_events(
        limit=f_limit,
        camera_filter=f_cam,
        label_filter=f_label,
        status_filter=f_status,
        include_demo=False,
    )

    if not events:
        st.info("Không tìm thấy sự kiện nào thỏa mãn điều kiện lọc.")
    else:
        df = pd.DataFrame(events)
        cols_show = ["id", "timestamp", "camera_name", "label", "confidence", "alert_level", "status", "action_taken"]
        st.dataframe(df[[c for c in cols_show if c in df.columns]], use_container_width=True)

        st.markdown("### 🖼️ XEM CHI TIẾT BẰNG CHỨNG SỰ KIỆN")
        selected_id = st.selectbox("Chọn ID sự kiện để xem ảnh snapshot:", [e["id"] for e in events])
        matched_ev = next(e for e in events if e["id"] == selected_id)
        if matched_ev.get("image_path") and os.path.exists(matched_ev["image_path"]):
            st.image(matched_ev["image_path"], caption=f"Snapshot Sự Kiện #{selected_id} ({matched_ev.get('camera_name')})", width=680)
        else:
            st.warning("Không tìm thấy file ảnh snapshot của sự kiện này.")


elif nav_view == "⚙️ Camera Management":
    st.markdown("## ⚙️ QUẢN LÝ & CẤU HÌNH TỪNG CAMERA")
    st.info("🔒 Mật khẩu RTSP được che hoàn toàn (`******`) sau khi lưu và không bao giờ xuất hiện trong log.")

    for cfg in cam_system.camera_configs:
        cid = cfg.cam_id
        w = cam_system.workers[cid]
        with st.expander(f"📹 {cid}: {cfg.name} (Loại: {cfg.source_type} | Nguồn: {cfg.masked_source} | Bật: {cfg.enabled})", expanded=(cid in st.session_state.selected_cams[:2])):
            col1, col2 = st.columns([2, 1])
            with col1:
                n_name = st.text_input("Tên hiển thị:", value=cfg.name, key=f"nm_{cid}")
                n_type = st.selectbox("Loại nguồn:", SOURCE_TYPES, index=SOURCE_TYPES.index(cfg.source_type), key=f"st_{cid}")
                if n_type == "VIDEO_TEST":
                    n_src = st.text_input("Đường dẫn file video:", value=cfg.source, key=f"src_{cid}")
                elif n_type == "WEBCAM":
                    sys_cams = list_system_cameras(4)
                    c_map = {c["label"]: c["index"] for c in sys_cams}
                    c_pick = st.selectbox("Chọn Webcam máy tính:", list(c_map.keys()), key=f"wpick_{cid}")
                    n_src = str(c_map[c_pick])
                else:
                    n_src = st.text_input("Địa chỉ RTSP (Password sẽ ẩn):", value=cfg.source, type="password", key=f"rtsp_{cid}")

                if st.button("💾 Lưu cấu hình camera", key=f"save_{cid}", type="primary"):
                    cam_system.update_camera_source(
                        cam_id=cid,
                        new_name=n_name,
                        new_source=n_src,
                        source_type=n_type,
                        webcam_index=int(n_src) if n_src.isdigit() else 0,
                    )
                    st.success(f"Đã cập nhật {cid}")
                    st.rerun()

            with col2:
                st.markdown("**Kiểm tra tín hiệu xem trước:**")
                if st.button("🔌 Test Connection", key=f"tst_{cid}"):
                    ok, msg, prev_img = cam_system.test_camera_connection(cid)
                    if ok and prev_img is not None:
                        st.success(msg)
                        st.image(prev_img, width=280)
                    else:
                        st.error(msg)


elif nav_view == "⚡ System Performance":
    st.markdown("## ⚡ HIỆU NĂNG SUY LUẬN & TẢI PHẦN CỨNG TOÀN HỆ THỐNG")
    perf_c1, perf_c2 = st.columns(2)
    with perf_c1:
        st.markdown("### 📊 THÔNG TIN THIẾT BỊ SUY LUẬN")
        m_info = cam_system.detector.get_model_info()
        st.markdown(
            f"""
            - **Mô hình AI:** `{m_info.get('path')}`
            - **Thiết bị:** `{m_info.get('device', 'CPU')}` (CUDA FP16: `{m_info.get('has_cuda')}`)
            - **Các lớp nhận diện:** `{m_info.get('names')}`
            - **Kích thước ảnh vào (imgsz):** `{cam_system.settings.imgsz}x{cam_system.settings.imgsz}`
            - **FPS Đọc tổng:** `{sys_metrics['total_fps']} FPS`
            - **FPS YOLO trung bình:** `{sys_metrics['avg_infer_fps']} FPS`
            """
        )
    with perf_c2:
        st.markdown("### 📋 NHẬT KÝ HỆ THỐNG GẦN ĐÂY")
        logs = cam_system.get_recent_logs(limit=15)
        st.text_area("Recent Logs:", value="\n".join(logs), height=240, disabled=True)
