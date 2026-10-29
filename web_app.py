"""Ứng dụng Web Dashboard Phát hiện và Cảnh báo Cháy 9 Camera (3x3 Grid) bằng Python & Streamlit.

Tận dụng và mở rộng mã nguồn hiện có của dự án:
- Giao diện Dark Mode chuyên nghiệp cho Trung tâm Giám sát PCCC.
- Bố cục 9 Camera (3x3), mỗi camera có worker thread độc lập, trạng thái kết nối, FPS thực tế,
  thời gian cập nhật và các nút Bắt đầu / Tạm dừng / Kết nối lại / Dừng riêng biệt.
- Tích hợp mô hình YOLO (`best.pt` với 2 lớp chuẩn `0: smoke` -> `SMOKE`, `1: fire` -> `FIRE`).
- Xác minh qua chuỗi khung hình liên tiếp (`TemporalVerifier` k/N khung hình + Cooldown chống báo lặp).
- Lưu trữ SQLite (`alerts.db`), lọc theo ngày/camera/loại sự kiện, xem ảnh bằng chứng và xuất CSV.
- Thống kê trực quan (Plotly) phân biệt rõ dữ liệu vận hành thực tế và dữ liệu kiểm thử/demo.
"""
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.config import (
    PROJECT_ROOT,
    CANDIDATE_MODEL_PATHS,
    DEFAULT_USERS,
)
from src.camera_manager import MultiCameraManager

# ==============================================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN DARK MODE PCCC CHUYÊN NGHIỆP
# ==============================================================================
st.set_page_config(
    page_title="TRUNG TÂM GIÁM SÁT & CẢNH BÁO CHÁY PCCC - 9 CAMERA AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #070B14;
        color: #F1F5F9;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    div[data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }
    .pccc-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid #334155;
        border-left: 5px solid #EF4444;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 14px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 21px;
        font-weight: 800;
        color: #38BDF8;
    }
    .cam-card-header {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px 8px 0 0;
        padding: 6px 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
        font-weight: 700;
    }
    .cam-card-footer {
        background: #0B1120;
        border: 1px solid #1E293B;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 6px 10px;
        margin-bottom: 6px;
        font-size: 11.5px;
        color: #CBD5E1;
    }
    .badge-online {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
    }
    .badge-paused {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid #F59E0B;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
    }
    .badge-disconnected {
        background: rgba(239, 68, 68, 0.22);
        color: #FCA5A5;
        border: 1px solid #EF4444;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
    }
    .badge-stopped {
        background: rgba(100, 116, 139, 0.2);
        color: #94A3B8;
        border: 1px solid #475569;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
    }
    .banner-safe {
        background: linear-gradient(90deg, #064E3B 0%, #047857 100%);
        border: 1px solid #10B981;
        border-radius: 10px;
        padding: 10px 16px;
        color: #ECFDF5;
        font-weight: 700;
        text-align: center;
        margin-bottom: 12px;
    }
    .banner-suspect {
        background: linear-gradient(90deg, #78350F 0%, #B45309 100%);
        border: 1px solid #F59E0B;
        border-radius: 10px;
        padding: 10px 16px;
        color: #FEF3C7;
        font-weight: 700;
        text-align: center;
        margin-bottom: 12px;
    }
    .banner-fire {
        background: linear-gradient(90deg, #7F1D1D 0%, #DC2626 100%);
        border: 2px solid #F87171;
        border-radius: 10px;
        padding: 12px 16px;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. KHỞI TẠO SINGLETON QUẢN LÝ 9 CAMERA (TRÁNH LẶP VÒNG ĐỌC KHI RERUN)
# ==============================================================================
@st.cache_resource
def get_multi_camera_system() -> MultiCameraManager:
    """Khởi tạo duy nhất một MultiCameraManager xuyên suốt vòng đời Streamlit server."""
    return MultiCameraManager(db_path="alerts.db", model_path="best.pt")


cam_system = get_multi_camera_system()

if "logged_user" not in st.session_state:
    st.session_state.logged_user = DEFAULT_USERS["admin"]
if "live_refresh" not in st.session_state:
    st.session_state.live_refresh = True


def list_local_videos() -> List[str]:
    """Liệt kê tất cả các file video thực tế có sẵn trong dự án để người dùng chọn nhanh."""
    vids: List[str] = []
    for folder in ["videovietnam", "FireSmokeLabeler/videos"]:
        p = PROJECT_ROOT / folder
        if p.exists():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                    vids.append(f.relative_to(PROJECT_ROOT).as_posix())
    return vids


# ==============================================================================
# 3. THANH BÊN (SIDEBAR) ĐIỀU HƯỚNG & TRẠNG THÁI MODEL / HỆ THỐNG
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="font-size:32px;">🔥</span>
            <div>
                <div style="color:#38BDF8; font-weight:800; font-size:17px;">FIRE SENTINEL PCCC</div>
                <div style="color:#94A3B8; font-size:11px; font-weight:600;">9-CAMERA AI COMMAND CENTER</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_info = cam_system.detector.get_model_info()
    if model_info["loaded"]:
        st.success(
            f"✅ **Model đang dùng:** `{Path(model_info['path']).name}`  \n"
            f"• Lớp: `{model_info['names']}`"
        )
    else:
        st.error(f"❌ **LỖI MODEL:** {model_info['error']}")

    st.markdown("---")
    nav_tab = st.radio(
        "CHỨC NĂNG TRUNG TÂM:",
        [
            "🎥 Giám Sát 9 Camera (3×3)",
            "📋 Lịch Sử Sự Kiện & CSV",
            "📊 Thống Kê & Biểu Đồ",
            "⚙️ Cấu Hình Camera & Model",
        ],
    )

    st.markdown("---")
    st.markdown("#### ⚡ ĐIỀU KHIỂN NHANH 9 CAMERA")
    c_sb1, c_sb2 = st.columns(2)
    with c_sb1:
        if st.button("▶ Chạy 4 Cam", use_container_width=True, help="Chạy 4 camera đầu (Tiết kiệm CPU/GPU)"):
            for i, cfg in enumerate(cam_system.camera_configs):
                if i < 4:
                    cam_system.start_camera(cfg.cam_id)
                else:
                    cam_system.stop_camera(cfg.cam_id)
            st.rerun()
    with c_sb2:
        if st.button("▶ Chạy Cả 9", type="primary", use_container_width=True, help="Bật đồng thời toàn bộ 9 camera (3x3)"):
            cam_system.start_all(only_enabled=False)
            st.rerun()

    c_sb3, c_sb4 = st.columns(2)
    with c_sb3:
        if st.button("⏸ Tạm Dừng", use_container_width=True):
            cam_system.pause_all()
            st.rerun()
    with c_sb4:
        if st.button("■ Dừng Tất Cả", use_container_width=True):
            cam_system.stop_all()
            st.rerun()

    if st.button("🔄 Kết Nối Lại Tất Cả Camera", use_container_width=True):
        cam_system.reconnect_all()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🔊 ÂM THANH & TỐI ƯU PHẦN CỨNG")
    mute_val = st.checkbox("🔕 Tắt còi báo động âm thanh", value=cam_system.settings.mute_audio)
    cam_system.settings.mute_audio = mute_val
    cam_system.alert_manager.update_config(mute_val)
    if st.button("🔕 Ngắt Còi Khẩn Cấp Ngay", use_container_width=True):
        cam_system.alert_manager.stop_alarm()

    cam_system.settings.target_fps_per_cam = st.slider(
        "Giới hạn FPS tối đa / Camera:", min_value=2, max_value=25, value=cam_system.settings.target_fps_per_cam
    )
    cam_system.settings.frame_skip = st.slider(
        "Chu kỳ suy luận AI (1 = Mọi frame, 2 = Cách 1 frame):",
        min_value=1,
        max_value=4,
        value=cam_system.settings.frame_skip,
    )
    cam_system.settings.conf_threshold = st.slider(
        "Ngưỡng độ tin cậy (Confidence):",
        min_value=0.15,
        max_value=0.90,
        value=float(cam_system.settings.conf_threshold),
        step=0.05,
    )
    cam_system.settings.confirm_frames_k = st.slider(
        f"Số frame liên tiếp xác nhận cháy (k/{cam_system.settings.window_size_n}):",
        min_value=2,
        max_value=cam_system.settings.window_size_n,
        value=cam_system.settings.confirm_frames_k,
    )

    st.session_state.live_refresh = st.checkbox(
        "🔄 Tự động làm mới hình ảnh trực tiếp", value=st.session_state.live_refresh
    )


# ==============================================================================
# 4. HÀM HIỂN THỊ LƯỚI 9 CAMERA (3x3) & KHU VỰC CẢNH BÁO TRỰC TIẾP
# ==============================================================================
def status_badge_html(status: str) -> str:
    if status == "ONLINE":
        return '<span class="badge-online">🟢 ONLINE</span>'
    if status == "PAUSED":
        return '<span class="badge-paused">🟡 PAUSED</span>'
    if status in ("DISCONNECTED", "ERROR"):
        return f'<span class="badge-disconnected">🔴 {status}</span>'
    if status == "CONNECTING":
        return '<span class="badge-paused">🔵 CONNECTING</span>'
    return '<span class="badge-stopped">⚪ STANDBY</span>'


def render_monitoring_dashboard():
    """Hiển thị KPI, Cảnh báo khẩn cấp, Lưới 9 Camera (3x3) và Panel Cảnh báo Trực tiếp."""
    sys_metrics = cam_system.get_system_metrics()
    db_stats = cam_system.event_db.get_summary_statistics()
    model_info = cam_system.detector.get_model_info()

    # Cảnh báo rõ ràng nếu không tìm thấy model hợp lệ
    if not model_info["loaded"]:
        st.error(
            f"🚨 **KHÔNG TÌM THẤY MÔ HÌNH PHÁT HIỆN CHÁY/KHÓI HỢP LỆ:** {model_info['error']}  \n"
            f"👉 Hướng dẫn: Vui lòng chuyển sang tab **⚙️ Cấu Hình Camera & Model** để chọn đường dẫn tới `best.pt`."
        )

    # Hàng KPI thực tế
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        lvl = sys_metrics["overall_level"]
        lvl_txt = "🚨 BÁO CHÁY" if lvl == "FIRE" else ("⚠️ NGHI NGỜ" if lvl == "SUSPECT" else "🟢 AN TOÀN")
        lvl_col = "#EF4444" if lvl == "FIRE" else ("#F59E0B" if lvl == "SUSPECT" else "#10B981")
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">TRẠNG THÁI PCCC</div>'
            f'<div class="kpi-value" style="color:{lvl_col};">{lvl_txt}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">CAMERA HOẠT ĐỘNG</div>'
            f'<div class="kpi-value">{sys_metrics["online_count"]} / {sys_metrics["total_cameras"]} ONLINE</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">FPS THỰC TẾ (TB / TỔNG)</div>'
            f'<div class="kpi-value" style="color:#38BDF8;">{sys_metrics["avg_fps"]} / {sys_metrics["total_fps"]} FPS</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">SỰ KIỆN THỰC TẾ (CHƯA XỬ LÝ)</div>'
            f'<div class="kpi-value" style="color:#F59E0B;">{db_stats["total_real"]} ({db_stats["unconfirmed"]} mới)</div></div>',
            unsafe_allow_html=True,
        )
    with k5:
        m_name = Path(model_info["path"]).name if model_info["loaded"] else "CHƯA NẠP"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">MODEL HIỆN HÀNH</div>'
            f'<div class="kpi-value" style="color:#A855F7; font-size:17px;">{m_name}</div></div>',
            unsafe_allow_html=True,
        )

    # Banner trạng thái toàn cục
    if sys_metrics["overall_level"] == "FIRE":
        cams_str = ", ".join(sys_metrics["fire_cameras"])
        st.markdown(
            f'<div class="banner-fire">🚨 CẢNH BÁO KHẨN CẤP: PHÁT HIỆN NGUY CƠ CHÁY/KHÓI TẠI [{cams_str}]! '
            f'YÊU CẦU KIỂM TRA XÁC NHẬN NGAY! 🚨</div>',
            unsafe_allow_html=True,
        )
    elif sys_metrics["overall_level"] == "SUSPECT":
        cams_str = ", ".join(sys_metrics["suspect_cameras"])
        st.markdown(
            f'<div class="banner-suspect">⚠️ CẢNH BÁO SỚM (NGHI NGỜ): Phát hiện dấu hiệu lửa/khói tại [{cams_str}], '
            f'đang xác minh chuỗi khung hình k/N...</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-safe">🛡️ HỆ THỐNG GIÁM SÁT 9 CAMERA ĐANG HOẠT ĐỘNG — CHƯA PHÁT HIỆN NGUY CƠ CHÁY</div>',
            unsafe_allow_html=True,
        )

    # Chia bố cục: Trái (Lưới 3x3 = 9 Camera), Phải (Khu vực Cảnh báo Trực tiếp & Nhật ký)
    col_grid, col_alert = st.columns([3.1, 1.15])

    snapshots = cam_system.get_all_snapshots()

    with col_grid:
        # Hiển thị 9 camera theo đúng bố cục 3 hàng x 3 cột
        for row_idx in range(3):
            row_cols = st.columns(3)
            for col_idx in range(3):
                cam_idx = row_idx * 3 + col_idx
                snap = snapshots[cam_idx]
                cid = snap["cam_id"]

                with row_cols[col_idx]:
                    badge = status_badge_html(snap["status"])
                    lvl_border = (
                        "border-color:#EF4444;"
                        if snap["alert_level"] == "FIRE"
                        else ("border-color:#F59E0B;" if snap["alert_level"] == "SUSPECT" else "")
                    )
                    st.markdown(
                        f'<div class="cam-card-header" style="{lvl_border}">'
                        f'<span>📹 {snap["name"][:26]}</span>{badge}</div>',
                        unsafe_allow_html=True,
                    )

                    st.image(snap["frame_rgb"], channels="RGB", use_container_width=True)

                    det_txt = (
                        f'<b style="color:#EF4444;">🔥 {snap["primary_label"]} ({int(snap["max_confidence"]*100)}%)</b>'
                        if snap["primary_label"]
                        else '<span style="color:#10B981;">An toàn</span>'
                    )
                    st.markdown(
                        f'<div class="cam-card-footer" style="{lvl_border}">'
                        f'⚡ <b>FPS:</b> {snap["fps"]:.1f} &nbsp;|&nbsp; '
                        f'🕒 <b>Cập nhật:</b> {snap["last_update_str"]} &nbsp;|&nbsp; '
                        f'🎯 <b>k/N:</b> {snap["active_hits"]}/{snap["window_size"]}<br>'
                        f'🔍 <b>Trạng thái AI:</b> {det_txt}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Hàng nút điều khiển riêng cho từng camera: Bắt đầu, Tạm dừng, Kết nối lại, Dừng
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        if st.button("▶", key=f"start_{cid}", use_container_width=True, help=f"Bắt đầu {cid}"):
                            cam_system.start_camera(cid)
                            st.rerun()
                    with b2:
                        if st.button("⏸", key=f"pause_{cid}", use_container_width=True, help=f"Tạm dừng / Tiếp tục {cid}"):
                            cam_system.pause_camera(cid)
                            st.rerun()
                    with b3:
                        if st.button("🔄", key=f"rec_{cid}", use_container_width=True, help=f"Kết nối lại {cid}"):
                            cam_system.reconnect_camera(cid)
                            st.rerun()
                    with b4:
                        if st.button("■", key=f"stop_{cid}", use_container_width=True, help=f"Dừng {cid}"):
                            cam_system.stop_camera(cid)
                            st.rerun()

    with col_alert:
        st.markdown("### 🚨 KHU VỰC CẢNH BÁO")
        recent_events = cam_system.event_db.query_events(limit=8, include_demo=False)

        if not recent_events:
            st.info("Chưa có sự kiện cảnh báo thực tế nào được ghi nhận.")
        else:
            latest_ev = recent_events[0]
            is_ack = latest_ev.get("status") == "ACKNOWLEDGED"
            ack_badge = (
                "✅ ĐÃ XÁC NHẬN" if is_ack else "🔴 CHƯA XÁC NHẬN (CẦN XỬ LÝ)"
            )
            lbl_color = "#EF4444" if "FIRE" in latest_ev["label"].upper() else "#F59E0B"

            st.markdown(
                f"""
                <div style="background:#1E1B2E; border:2px solid {lbl_color}; border-radius:10px; padding:12px; margin-bottom:10px;">
                    <div style="font-size:13px; font-weight:800; color:{lbl_color};">
                        🔔 SỰ KIỆN #{latest_ev['id']} — MỨC: {latest_ev.get('alert_level', 'FIRE')}
                    </div>
                    <div style="font-size:12px; color:#E2E8F0; margin-top:4px;">
                        • <b>Camera:</b> <code>{latest_ev.get('camera_name', 'CAM-01')}</code><br>
                        • <b>Thời gian:</b> <code>{latest_ev['timestamp']}</code><br>
                        • <b>Đối tượng:</b> <b style="color:{lbl_color};">{latest_ev['label']}</b><br>
                        • <b>Độ tin cậy:</b> <b>{latest_ev['confidence']*100:.1f}%</b><br>
                        • <b>Trạng thái:</b> <b>{ack_badge}</b><br>
                        • <b>Xử lý:</b> {latest_ev.get('action_taken', 'Chưa xử lý')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            img_p = latest_ev.get("image_path", "")
            if img_p and os.path.exists(img_p):
                st.image(img_p, caption=f"Bằng chứng #{latest_ev['id']} ({latest_ev.get('camera_name')})", use_container_width=True)

            ca1, ca2 = st.columns(2)
            with ca1:
                if st.button("✅ Xác nhận cháy", key=f"ack_live_{latest_ev['id']}", type="primary", use_container_width=True):
                    cam_system.event_db.confirm_event(
                        latest_ev["id"],
                        action_taken="Đã xác nhận nguy cơ cháy",
                        note="Xác nhận trực tiếp từ bảng điều khiển",
                    )
                    cam_system.alert_manager.stop_alarm()
                    st.rerun()
            with ca2:
                if st.button("❎ Báo động giả", key=f"dismiss_live_{latest_ev['id']}", use_container_width=True):
                    cam_system.event_db.dismiss_event(
                        latest_ev["id"],
                        note="Đánh dấu báo động giả từ bảng điều khiển",
                    )
                    cam_system.alert_manager.stop_alarm()
                    st.rerun()

            if len(recent_events) > 1:
                st.markdown("##### 📌 Các cảnh báo gần đây:")
                for ev in recent_events[1:5]:
                    st_icon = "✅" if ev.get("status") == "ACKNOWLEDGED" else "🔴"
                    st.caption(
                        f"{st_icon} **#{ev['id']}** | `{ev.get('camera_name')}` | **{ev['label']}** "
                        f"({ev['confidence']*100:.0f}%) | {ev['timestamp'][11:]}"
                    )

        st.markdown("---")
        st.markdown("### 📜 NHẬT KÝ HỆ THỐNG")
        logs_text = "\n".join(cam_system.get_recent_logs(limit=18))
        st.code(logs_text if logs_text else "Chưa có nhật ký.", language="text")


# ==============================================================================
# 5. ĐIỀU HƯỚNG CÁC CHẾ ĐỘ MÀN HÌNH
# ==============================================================================
st.markdown(
    """
    <div class="pccc-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:20px; font-weight:800; color:#F8FAFC;">
                    🔥 HỆ THỐNG GIÁM SÁT & CẢNH BÁO CHÁY SỚM ĐA LUỒNG (9 CAMERA 3×3)
                </span>
                <div style="font-size:12px; color:#94A3B8;">
                    Computer Vision YOLOv8/YOLO11 • Xác minh chuỗi thời gian k/N khung hình • Cơ sở dữ liệu sự kiện SQLite
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if nav_tab == "🎥 Giám Sát 9 Camera (3×3)":
    if hasattr(st, "fragment") and st.session_state.live_refresh:
        @st.fragment(run_every=0.8)
        def _auto_live_fragment():
            render_monitoring_dashboard()

        _auto_live_fragment()
    else:
        render_monitoring_dashboard()
        if st.session_state.live_refresh and cam_system.get_system_metrics()["online_count"] > 0:
            time.sleep(0.8)
            st.rerun()

elif nav_tab == "📋 Lịch Sử Sự Kiện & CSV":
    st.subheader("📋 Lịch Sử Sự Kiện Cảnh Báo Cháy & Quản Lý Bằng Chứng (SQLite)")

    f1, f2, f3, f4, f5 = st.columns([1.2, 1.2, 1.1, 1.1, 1.2])
    with f1:
        start_d = st.date_input("Từ ngày:", value=date.today() - timedelta(days=30))
    with f2:
        end_d = st.date_input("Đến ngày:", value=date.today())
    with f3:
        cam_options = ["Tất cả"] + [c.cam_id for c in cam_system.camera_configs]
        sel_cam = st.selectbox("Camera:", cam_options)
    with f4:
        sel_label = st.selectbox("Loại đối tượng:", ["Tất cả", "FIRE", "SMOKE"])
    with f5:
        sel_status = st.selectbox("Trạng thái:", ["Tất cả", "Chưa xác nhận", "Đã xác nhận"])

    col_s1, col_s2, col_s3 = st.columns([2.5, 1.2, 1.3])
    with col_s1:
        search_kw = st.text_input("🔍 Tìm kiếm theo từ khóa / ghi chú:", placeholder="Nhập ID, camera, ghi chú...")
    with col_s2:
        show_demo = st.checkbox("Hiển thị cả dữ liệu Demo/Test", value=False)
    with col_s3:
        if st.button("🗑️ Xóa toàn bộ lịch sử CSDL", use_container_width=True):
            cam_system.event_db.clear_all()
            st.success("Đã xóa toàn bộ lịch sử cảnh báo trong SQLite.")
            st.rerun()

    events = cam_system.event_db.query_events(
        limit=1000,
        search_query=search_kw,
        camera_filter=sel_cam,
        label_filter=sel_label,
        status_filter=sel_status,
        start_date=str(start_d),
        end_date=str(end_d),
        include_demo=show_demo,
    )

    if not events:
        st.info("Không tìm thấy sự kiện nào khớp với bộ lọc hiện tại.")
    else:
        df_ev = pd.DataFrame(events)
        df_ev["confidence_pct"] = (df_ev["confidence"] * 100).round(1).astype(str) + "%"
        df_ev["status_vi"] = df_ev["status"].apply(
            lambda x: "✅ Đã xác nhận" if x == "ACKNOWLEDGED" else "🔴 Chưa xác nhận"
        )
        df_ev["data_type"] = df_ev["is_demo"].apply(lambda x: "Demo/Test" if x else "Thực tế")

        # Nút xuất CSV và tải trực tiếp
        csv_export_path = f"lich_su_su_kien_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        ce1, ce2 = st.columns([1.2, 3])
        with ce1:
            csv_bytes = df_ev[
                [
                    "id",
                    "timestamp",
                    "camera_name",
                    "label",
                    "alert_level",
                    "confidence_pct",
                    "status_vi",
                    "action_taken",
                    "operator_note",
                    "image_path",
                    "data_type",
                ]
            ].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Tải Xuống File CSV",
                data=csv_bytes,
                file_name=csv_export_path,
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

        st.dataframe(
            df_ev[
                [
                    "id",
                    "timestamp",
                    "camera_name",
                    "label",
                    "alert_level",
                    "confidence_pct",
                    "status_vi",
                    "action_taken",
                    "operator_note",
                    "image_path",
                    "data_type",
                ]
            ].rename(
                columns={
                    "id": "ID",
                    "timestamp": "Thời Gian",
                    "camera_name": "Camera",
                    "label": "Loại Phát Hiện",
                    "alert_level": "Mức Cảnh Báo",
                    "confidence_pct": "Độ Tin Cậy",
                    "status_vi": "Trạng Thái",
                    "action_taken": "Xử Lý Ca Trực",
                    "operator_note": "Ghi Chú",
                    "image_path": "Đường Dẫn Ảnh",
                    "data_type": "Nguồn Dữ Liệu",
                }
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### 📸 Xem Ảnh Bằng Chứng & Xác Nhận Xử Lý Sự Kiện")
        selected_ev_id = st.selectbox(
            "Chọn ID sự kiện cần kiểm tra / cập nhật:",
            options=[e["id"] for e in events],
            format_func=lambda x: f"Sự kiện #{x}",
        )
        sel_record = next((e for e in events if e["id"] == selected_ev_id), None)
        if sel_record:
            c_img, c_act = st.columns([1.5, 1.2])
            with c_img:
                p_img = sel_record.get("image_path", "")
                if p_img and os.path.exists(p_img):
                    st.image(
                        p_img,
                        caption=f"Ảnh bằng chứng #{sel_record['id']} | {sel_record['camera_name']} | {sel_record['timestamp']}",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Không tìm thấy file ảnh tại đường dẫn: `{p_img}`")

            with c_act:
                st.markdown(
                    f"**ID Sự kiện:** `#{sel_record['id']}`  \n"
                    f"**Camera phát hiện:** `{sel_record.get('camera_name', 'CAM-01')}`  \n"
                    f"**Thời gian:** `{sel_record['timestamp']}`  \n"
                    f"**Loại đối tượng:** **{sel_record['label']}** (`{sel_record['confidence']*100:.1f}%`)  \n"
                    f"**Phân loại dữ liệu:** `{'Dữ liệu Demo/Test' if sel_record.get('is_demo') else 'Dữ liệu Thực tế'}`"
                )
                with st.form(f"form_ev_{sel_record['id']}"):
                    act_choice = st.selectbox(
                        "Kết quả xác minh:",
                        [
                            "Đã xác nhận cháy thật - Đã xử lý",
                            "Báo động giả (Ánh đèn / Nhiễu)",
                            "Nghiêm trọng - Đã báo lực lượng PCCC 114",
                            "Chưa xử lý",
                        ],
                    )
                    note_inp = st.text_input("Ghi chú ca trực:", value=sel_record.get("operator_note", ""))
                    sub_btn = st.form_submit_button("💾 Cập Nhật Trạng Thái Sự Kiện", type="primary")
                    if sub_btn:
                        new_st = "UNREAD" if act_choice == "Chưa xử lý" else "ACKNOWLEDGED"
                        cam_system.event_db.alert_manager.update_alert_action(
                            sel_record["id"], action_taken=act_choice, operator_note=note_inp, status=new_st
                        )
                        st.success(f"Đã cập nhật sự kiện #{sel_record['id']}!")
                        st.rerun()

                if st.button(f"🗑️ Xóa sự kiện #{sel_record['id']}", use_container_width=True):
                    cam_system.event_db.delete_event(sel_record["id"])
                    st.rerun()

elif nav_tab == "📊 Thống Kê & Biểu Đồ":
    st.subheader("📊 Thống Kê Hệ Thống & Phân Tích Dữ Liệu Thực Tế")

    sys_m = cam_system.get_system_metrics()
    stats = cam_system.event_db.get_summary_statistics()
    snaps = cam_system.get_all_snapshots()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tổng Sự Kiện Thực Tế", f"{stats['total_real']} sự kiện", f"Demo/Test: {stats['total_demo']}")
    s2.metric("Phân Bố FIRE / SMOKE (Thực tế)", f"{stats['real_fire']} FIRE / {stats['real_smoke']} SMOKE")
    s3.metric("Số Camera Đang Hoạt Động", f"{sys_m['online_count']} / {sys_m['total_cameras']}")
    s4.metric("FPS Thực Tế (Trung bình / Tổng)", f"{sys_m['avg_fps']} / {sys_m['total_fps']} FPS")

    st.markdown("---")
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### 📈 Hiệu Năng FPS Thực Tế Theo Từng Camera")
        df_fps = pd.DataFrame(
            [
                {
                    "Camera": s["cam_id"],
                    "Tên Camera": s["name"],
                    "FPS Thực Tế": s["fps"],
                    "Trạng Thái": s["status"],
                }
                for s in snaps
            ]
        )
        if HAS_PLOTLY:
            fig_fps = px.bar(
                df_fps,
                x="Camera",
                y="FPS Thực Tế",
                color="Trạng Thái",
                hover_data=["Tên Camera"],
                color_discrete_map={
                    "ONLINE": "#10B981",
                    "PAUSED": "#F59E0B",
                    "DISCONNECTED": "#EF4444",
                    "STOPPED": "#64748B",
                },
                template="plotly_dark",
            )
            fig_fps.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_fps, use_container_width=True)
        else:
            st.bar_chart(df_fps.set_index("Camera")["FPS Thực Tế"])

    with c_chart2:
        st.markdown("#### 🔍 Phân Biệt Dữ Liệu Thực Tế vs. Dữ Liệu Demo/Kiểm Thử")
        df_split = pd.DataFrame(
            [
                {"Phân Loại": "Sự kiện Thực tế (Lửa - FIRE)", "Số lượng": stats["real_fire"]},
                {"Phân Loại": "Sự kiện Thực tế (Khói - SMOKE)", "Số lượng": stats["real_smoke"]},
                {"Phân Loại": "Dữ liệu Demo / Kiểm thử", "Số lượng": stats["total_demo"]},
            ]
        )
        if HAS_PLOTLY:
            fig_pie = px.pie(
                df_split,
                names="Phân Loại",
                values="Số lượng",
                hole=0.42,
                color="Phân Loại",
                color_discrete_map={
                    "Sự kiện Thực tế (Lửa - FIRE)": "#EF4444",
                    "Sự kiện Thực tế (Khói - SMOKE)": "#F59E0B",
                    "Dữ liệu Demo / Kiểm thử": "#64748B",
                },
                template="plotly_dark",
            )
            fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.dataframe(df_split, use_container_width=True)

    st.markdown("#### 🕒 Biểu Đồ Sự Kiện Thực Tế Theo Thời Gian & Theo Camera")
    real_recs = stats["real_records"]
    if not real_recs:
        st.info("Chưa có dữ liệu sự kiện thực tế để vẽ biểu đồ chuỗi thời gian.")
    else:
        df_real = pd.DataFrame(real_recs)
        df_real["hour_slot"] = df_real["timestamp"].str.slice(0, 13) + ":00"
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            df_by_time = df_real.groupby(["hour_slot", "label"]).size().reset_index(name="Số sự kiện")
            if HAS_PLOTLY:
                fig_time = px.bar(
                    df_by_time,
                    x="hour_slot",
                    y="Số sự kiện",
                    color="label",
                    title="Số lượng sự kiện thực tế theo khung giờ",
                    color_discrete_map={"FIRE": "#EF4444", "SMOKE": "#F59E0B"},
                    template="plotly_dark",
                )
                fig_time.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.dataframe(df_by_time, use_container_width=True)

        with c_t2:
            df_by_cam = df_real.groupby(["camera_name", "label"]).size().reset_index(name="Số sự kiện")
            if HAS_PLOTLY:
                fig_cam = px.bar(
                    df_by_cam,
                    x="camera_name",
                    y="Số sự kiện",
                    color="label",
                    title="Thống kê sự kiện theo từng Camera",
                    color_discrete_map={"FIRE": "#EF4444", "SMOKE": "#F59E0B"},
                    template="plotly_dark",
                )
                fig_cam.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_cam, use_container_width=True)
            else:
                st.dataframe(df_by_cam, use_container_width=True)

elif nav_tab == "⚙️ Cấu Hình Camera & Model":
    st.subheader("⚙️ Cấu Hình Mô Hình YOLO & 9 Nguồn Camera Giám Sát")

    # 1. Cấu hình Model
    st.markdown("### 1. Tích Hợp & Kiểm Tra Mô Hình YOLO (`best.pt`)")
    m_info = cam_system.detector.get_model_info()
    c_m1, c_m2 = st.columns([2.5, 1.5])
    with c_m1:
        avail_models = [p for p in CANDIDATE_MODEL_PATHS if os.path.exists(p)]
        custom_model_path = st.text_input(
            "Đường dẫn file trọng số mô hình (.pt):",
            value=m_info["path"] if m_info["loaded"] else "best.pt",
        )
        if avail_models:
            st.caption("Các model đã huấn luyện tìm thấy trong dự án: " + " | ".join(f"`{Path(p).as_posix()}`" for p in avail_models))
    with c_m2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Nạp & Kiểm Tra Model", type="primary", use_container_width=True):
            ok, msg = cam_system.detector.load_model(custom_model_path.strip())
            if ok:
                cam_system.add_log(msg)
                st.success(msg)
            else:
                cam_system.add_log(f"Lỗi nạp model: {msg}")
                st.error(msg)

    # 2. Cấu hình 9 Camera
    st.markdown("---")
    st.markdown("### 2. Cấu Hình Nguồn Cho 9 Camera (Webcam / Video Cục Bộ / RTSP)")
    local_vids = list_local_videos()
    st.caption(
        "💡 **Hướng dẫn:** Bạn có thể nhập `"
        "0` (Webcam), đường dẫn file video cục bộ (VD: `videovietnam/VN_POS_01.mp4`), "
        "hoặc đường dẫn luồng `rtsp://user:pass@ip:port/stream`."
    )

    for idx, cfg in enumerate(cam_system.camera_configs):
        with st.expander(f"📹 {cfg.cam_id} — {cfg.name} (Nguồn hiện tại: `{cfg.source}`)", expanded=(idx < 2)):
            cc1, cc2, cc3, cc4 = st.columns([1.4, 1.4, 1.8, 0.8])
            with cc1:
                new_name = st.text_input("Tên hiển thị:", value=cfg.name, key=f"cfg_name_{cfg.cam_id}")
            with cc2:
                quick_pick = st.selectbox(
                    "Chọn nhanh video có sẵn:",
                    options=["-- Tự nhập đường dẫn / RTSP --", "0 (Webcam 0)"] + local_vids,
                    key=f"cfg_pick_{cfg.cam_id}",
                )
            with cc3:
                default_src = (
                    "0"
                    if quick_pick == "0 (Webcam 0)"
                    else (quick_pick if quick_pick != "-- Tự nhập đường dẫn / RTSP --" else cfg.source)
                )
                new_src = st.text_input(
                    "Đường dẫn Nguồn (File / RTSP / Webcam ID):",
                    value=default_src,
                    key=f"cfg_src_{cfg.cam_id}",
                )
            with cc4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("💾 Lưu", key=f"cfg_save_{cfg.cam_id}", type="primary", use_container_width=True):
                    cam_system.update_camera_source(cfg.cam_id, new_name, new_src, auto_restart=True)
                    st.success(f"Đã lưu {cfg.cam_id}!")
                    st.rerun()
