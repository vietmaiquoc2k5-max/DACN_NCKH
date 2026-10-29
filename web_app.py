"""Ứng dụng Web Dashboard Phát hiện và Cảnh báo Cháy 9 Camera (3x3 Grid) & Camera Thực Tế.

Đáp ứng toàn bộ yêu cầu Mục 1 -> Mục 10:
- Hỗ trợ 4 chế độ hoạt động: `VIDEO_TEST`, `WEBCAM`, `IP_CAMERA`, `MULTI_CAMERA` (tối đa 9 camera 3x3).
- Giao diện cấu hình từng camera với đầy đủ: Tên camera, Loại nguồn (`VIDEO_TEST`, `WEBCAM`, `IP_CAMERA`),
  chỉ số Webcam, trình nhập RTSP bảo mật (ẩn mật khẩu sau khi lưu, không ghi mật khẩu vào log),
  bật/tắt từng camera độc lập, nút Kết nối / Ngắt kết nối / Kiểm tra kết nối / Lưu cấu hình và Ảnh xem trước.
- Hiển thị trạng thái chuẩn: `CONNECTED`, `DISCONNECTED`, `RECONNECTING`, `ERROR`, `PAUSED`.
- Hiển thị riêng biệt **FPS đọc khung hình (`Capture FPS`)** và **FPS suy luận model (`YOLO FPS`)**.
- Khi camera mất tín hiệu, hiển thị cảnh báo lỗi kết nối riêng biệt, không hiển thị trạng thái an toàn giả tạo.
"""
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.config import (
    PROJECT_ROOT,
    CANDIDATE_MODEL_PATHS,
    DEFAULT_USERS,
    OPERATING_MODES,
    SOURCE_TYPES,
    mask_rtsp_credentials,
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
        padding: 12px 18px;
        margin-bottom: 12px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 20px;
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
        font-size: 11px;
        color: #CBD5E1;
    }
    .badge-connected {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10.5px;
    }
    .badge-reconnecting {
        background: rgba(56, 189, 248, 0.18);
        color: #38BDF8;
        border: 1px solid #0284C7;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10.5px;
    }
    .badge-paused {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid #F59E0B;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10.5px;
    }
    .badge-error {
        background: rgba(239, 68, 68, 0.22);
        color: #FCA5A5;
        border: 1px solid #EF4444;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10.5px;
    }
    .badge-disconnected {
        background: rgba(100, 116, 139, 0.22);
        color: #94A3B8;
        border: 1px solid #475569;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10.5px;
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
    .banner-standby {
        background: linear-gradient(90deg, #1E293B 0%, #334155 100%);
        border: 1px solid #64748B;
        border-radius: 10px;
        padding: 10px 16px;
        color: #E2E8F0;
        font-weight: 700;
        text-align: center;
        margin-bottom: 12px;
    }
    .banner-conn-error {
        background: linear-gradient(90deg, #7C2D12 0%, #9A3412 100%);
        border: 1px solid #F97316;
        border-radius: 10px;
        padding: 10px 16px;
        color: #FFEDD5;
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
if "test_preview_results" not in st.session_state:
    st.session_state.test_preview_results = {}  # cam_id -> (ok, msg, rgb_img)


def list_local_videos() -> List[str]:
    """Liệt kê tất cả các file video kiểm thử có sẵn trong dự án."""
    vids: List[str] = []
    for folder in ["videovietnam", "FireSmokeLabeler/videos"]:
        p = PROJECT_ROOT / folder
        if p.exists():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                    vids.append(f.relative_to(PROJECT_ROOT).as_posix())
    return vids


def status_badge_html(status: str) -> str:
    """Trả về HTML badge cho trạng thái CONNECTED / DISCONNECTED / RECONNECTING / ERROR / PAUSED."""
    if status == "CONNECTED":
        return '<span class="badge-connected">🟢 CONNECTED</span>'
    if status == "RECONNECTING":
        return '<span class="badge-reconnecting">🔄 RECONNECTING</span>'
    if status == "PAUSED":
        return '<span class="badge-paused">🟡 PAUSED</span>'
    if status == "ERROR":
        return '<span class="badge-error">🔴 ERROR</span>'
    return '<span class="badge-disconnected">⚪ DISCONNECTED</span>'


# ==============================================================================
# 3. THANH BÊN (SIDEBAR) ĐIỀU HƯỚNG, CHẾ ĐỘ HOẠT ĐỘNG & TỐI ƯU PHẦN CỨNG
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="font-size:32px;">🔥</span>
            <div>
                <div style="color:#38BDF8; font-weight:800; font-size:17px;">FIRE SENTINEL PCCC</div>
                <div style="color:#94A3B8; font-size:11px; font-weight:600;">REAL-TIME CAMERA & VIDEO AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_info = cam_system.detector.get_model_info()
    if model_info["loaded"]:
        dev_str = model_info.get("device", "CPU")
        dev_icon = "🚀" if model_info.get("has_cuda") else "💻"
        st.success(
            f"✅ **Model YOLO:** `{Path(model_info['path']).name}`  \n"
            f"• Thiết bị: `{dev_icon} {dev_str}`  \n"
            f"• Lớp: `{model_info['names']}`"
        )
        if not model_info.get("has_cuda"):
            st.caption("💡 *Mẹo: Chạy bằng `.venv\\Scripts\\python.exe -m streamlit run web_app.py` (hoặc `chay_giao_dien_web.bat`) để bật GPU CUDA RTX 3050 FP16.*")
    else:
        st.error(f"❌ **LỖI MODEL:** {model_info['error']}")

    st.markdown("---")
    nav_tab = st.radio(
        "CHỨC NĂNG TRUNG TÂM:",
        [
            "🎥 Giám Sát 9 Camera (3×3)",
            "⚙️ Cấu Hình Nguồn Camera & RTSP",
            "📋 Lịch Sử Sự Kiện & CSV",
            "📊 Thống Kê & Biểu Đồ",
        ],
    )

    st.markdown("---")
    st.markdown("#### 🎛️ CHẾ ĐỘ HOẠT ĐỘNG")
    mode_keys = list(OPERATING_MODES.keys())
    curr_mode_idx = mode_keys.index(cam_system.settings.operating_mode) if cam_system.settings.operating_mode in mode_keys else 0
    selected_mode = st.selectbox(
        "Chế độ vận hành:",
        options=mode_keys,
        index=curr_mode_idx,
        format_func=lambda k: f"{k} — {OPERATING_MODES[k]}",
    )
    cam_system.settings.operating_mode = selected_mode

    st.markdown("#### ⚡ ĐIỀU KHIỂN TOÀN HỆ THỐNG")
    c_sb1, c_sb2 = st.columns(2)
    with c_sb1:
        if st.button("🔌 Kết Nối Các Cam Đã Bật", use_container_width=True, help="Kết nối tất cả các camera đang được tick Bật (Enabled)"):
            cam_system.start_all(only_enabled=True)
            st.rerun()
    with c_sb2:
        if st.button("🔌 Kết Nối Cả 9 Cam", type="primary", use_container_width=True, help="Bật và kết nối đồng thời cả 9 camera (3x3)"):
            cam_system.start_all(only_enabled=False)
            st.rerun()

    c_sb3, c_sb4 = st.columns(2)
    with c_sb3:
        if st.button("⏸ Tạm Dừng Tất Cả", use_container_width=True):
            cam_system.pause_all()
            st.rerun()
    with c_sb4:
        if st.button("⏹ Ngắt Kết Nối Tất Cả", use_container_width=True):
            cam_system.stop_all()
            st.rerun()

    if st.button("🔄 Thử Kết Nối Lại Tất Cả (Reconnect)", use_container_width=True):
        cam_system.reconnect_all()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🚀 TỐI ƯU FPS & HIỆU NĂNG CPU/GPU")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("⚡ Chế Độ Siêu Mượt", use_container_width=True, help="Tối ưu FPS cao nhất cho nhiều camera (imgsz=320, skip=3, 25 FPS)"):
            cam_system.settings.imgsz = 320
            cam_system.settings.capture_width = 480
            cam_system.settings.display_width = 380
            cam_system.settings.frame_skip = 3
            cam_system.settings.target_fps_per_cam = 25
            cam_system.settings.jpeg_quality = 68
            cam_system.settings.ui_refresh_sec = 0.35
            st.rerun()
    with p_col2:
        if st.button("⚖️ Chế Độ Cân Bằng", use_container_width=True, help="Cân bằng giữa độ nét và tốc độ (imgsz=416, skip=2, 24 FPS)"):
            cam_system.settings.imgsz = 416
            cam_system.settings.capture_width = 480
            cam_system.settings.display_width = 420
            cam_system.settings.frame_skip = 2
            cam_system.settings.target_fps_per_cam = 24
            cam_system.settings.jpeg_quality = 72
            cam_system.settings.ui_refresh_sec = 0.4
            st.rerun()

    mute_val = st.checkbox("🔕 Tắt còi báo động âm thanh", value=cam_system.settings.mute_audio)
    cam_system.settings.mute_audio = mute_val
    cam_system.alert_manager.update_config(mute_val)
    if st.button("🔕 Ngắt Còi Khẩn Cấp Ngay", use_container_width=True):
        cam_system.alert_manager.stop_alarm()

    cam_system.settings.imgsz = st.select_slider(
        "Kích thước suy luận YOLO (imgsz - nhỏ hơn = FPS cao hơn):",
        options=[320, 416, 480, 640],
        value=cam_system.settings.imgsz if cam_system.settings.imgsz in [320, 416, 480, 640] else 416,
    )
    cam_system.detector.imgsz = cam_system.settings.imgsz

    cam_system.settings.capture_width = st.select_slider(
        "Độ phân giải xử lý khung hình (Width):",
        options=[320, 480, 640, 800, 960],
        value=cam_system.settings.capture_width if cam_system.settings.capture_width in [320, 480, 640, 800, 960] else 480,
    )
    cam_system.settings.target_fps_per_cam = st.slider(
        "Giới hạn FPS đọc tối đa / Camera:", min_value=5, max_value=30, value=cam_system.settings.target_fps_per_cam
    )
    cam_system.settings.frame_skip = st.slider(
        "Chu kỳ suy luận YOLO (1 = Mọi frame, 2-3 = Mượt gấp 2-3 lần):",
        min_value=1,
        max_value=6,
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
    cam_system.settings.ui_refresh_sec = st.select_slider(
        "Chu kỳ làm mới giao diện (giây):",
        options=[0.25, 0.35, 0.4, 0.5, 0.8, 1.0],
        value=cam_system.settings.ui_refresh_sec if cam_system.settings.ui_refresh_sec in [0.25, 0.35, 0.4, 0.5, 0.8, 1.0] else 0.4,
    )

    st.session_state.live_refresh = st.checkbox(
        "🔄 Tự động cập nhật luồng trực tiếp", value=st.session_state.live_refresh
    )


# ==============================================================================
# 4. KHUNG CẤU HÌNH NGUỒN CAMERA NHANH THEO CHẾ ĐỘ HOẠT ĐỘNG (MỤC 10.A)
# ==============================================================================
def render_quick_mode_selector():
    """Hiển thị thanh chuyển đổi nhanh giữa VIDEO_TEST, WEBCAM, IP_CAMERA và MULTI_CAMERA."""
    mode = cam_system.settings.operating_mode
    local_vids = list_local_videos()

    with st.expander(
        f"🎛️ CẤU HÌNH NHANH NGUỒN CAMERA — CHẾ ĐỘ HIỆN TẠI: [{mode}] ({OPERATING_MODES.get(mode, '')})",
        expanded=(mode != "MULTI_CAMERA"),
    ):
        if mode == "VIDEO_TEST":
            c1, c2, c3, c4 = st.columns([1.2, 2.2, 1.8, 1.0])
            with c1:
                target_cam = st.selectbox("Gán vào ô Camera:", [c.cam_id for c in cam_system.camera_configs], key="qm_vid_cam")
            with c2:
                chosen_vid = st.selectbox("Chọn video kiểm thử có sẵn:", local_vids if local_vids else ["videovietnam/VN_POS_01.mp4"], key="qm_vid_pick")
            with c3:
                custom_vid = st.text_input("Hoặc nhập đường dẫn file video trên máy:", value=chosen_vid, key="qm_vid_path")
            with c4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔌 Kết nối Video", type="primary", use_container_width=True, key="qm_vid_btn"):
                    cam_system.update_camera_source(
                        target_cam,
                        f"{target_cam} - Video Kiểm Thử",
                        custom_vid.strip(),
                        source_type="VIDEO_TEST",
                        enabled=True,
                        auto_restart=True,
                    )
                    st.rerun()

        elif mode == "WEBCAM":
            c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.8, 1.2])
            with c1:
                target_cam = st.selectbox("Gán vào ô Camera:", [c.cam_id for c in cam_system.camera_configs], key="qm_web_cam")
            with c2:
                web_idx = st.number_input("Chỉ số thiết bị Webcam (0, 1, 2...):", min_value=0, max_value=10, value=0, step=1, key="qm_web_idx")
            with c3:
                cam_label = st.text_input("Tên hiển thị Camera:", value=f"{target_cam} - Webcam Thực Tế #{web_idx}", key="qm_web_name")
            with c4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔌 Kết nối Webcam", type="primary", use_container_width=True, key="qm_web_btn"):
                    cam_system.update_camera_source(
                        target_cam,
                        cam_label,
                        str(int(web_idx)),
                        source_type="WEBCAM",
                        webcam_index=int(web_idx),
                        enabled=True,
                        auto_restart=True,
                    )
                    st.rerun()

        elif mode == "IP_CAMERA":
            st.caption("🔒 **Bảo mật RTSP:** Mật khẩu camera sẽ tự động được ẩn (`******`) sau khi lưu và không bao giờ ghi vào log.")
            c1, c2, c3, c4 = st.columns([1.1, 1.4, 2.4, 1.0])
            with c1:
                target_cam = st.selectbox("Gán vào ô Camera:", [c.cam_id for c in cam_system.camera_configs], key="qm_ip_cam")
            with c2:
                ip_name = st.text_input("Tên Camera IP:", value=f"{target_cam} - Camera IP Giám Sát", key="qm_ip_name")
            with c3:
                rtsp_input = st.text_input(
                    "Địa chỉ RTSP (VD: rtsp://user:pass@192.168.1.100:554/stream1):",
                    type="password",
                    placeholder="rtsp://username:password@ip:554/Streaming/Channels/101",
                    key="qm_ip_url",
                )
            with c4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔌 Kết nối RTSP", type="primary", use_container_width=True, key="qm_ip_btn"):
                    if not rtsp_input.strip():
                        st.warning("Vui lòng nhập địa chỉ RTSP hợp lệ!")
                    else:
                        cam_system.update_camera_source(
                            target_cam,
                            ip_name,
                            rtsp_input.strip(),
                            source_type="IP_CAMERA",
                            enabled=True,
                            auto_restart=True,
                        )
                        st.rerun()
        else:
            st.info(
                "Đang ở chế độ **MULTI_CAMERA (Giám sát đa nguồn 9 camera 3×3)**. "
                "Bạn có thể điều khiển trực tiếp từng ô camera bên dưới hoặc vào tab **⚙️ Cấu Hình Nguồn Camera & RTSP** để thiết lập chi tiết từng luồng."
            )


# ==============================================================================
# 5. HÀM HIỂN THỊ LƯỚI 9 CAMERA (3x3) & KHU VỰC CẢNH BÁO TRỰC TIẾP
# ==============================================================================
def render_monitoring_dashboard():
    """Hiển thị KPI, Cảnh báo khẩn cấp, Lưới 9 Camera (3x3) và Panel Cảnh báo Trực tiếp."""
    snapshots = cam_system.get_all_snapshots()
    sys_metrics = cam_system.get_system_metrics(snapshots=snapshots)
    db_stats = cam_system.event_db.get_summary_statistics()
    model_info = cam_system.detector.get_model_info()

    if not model_info["loaded"]:
        st.error(
            f"🚨 **KHÔNG TÌM THẤY MÔ HÌNH PHÁT HIỆN CHÁY/KHÓI HỢP LỆ:** {model_info['error']}  \n"
            f"👉 Hướng dẫn: Vui lòng chuyển sang tab **⚙️ Cấu Hình Nguồn Camera & RTSP** để chọn đường dẫn tới `best.pt`."
        )

    # Hàng KPI thực tế (Phân biệt rõ trạng thái cháy, lỗi kết nối, camera thật vs video kiểm thử, Capture FPS vs YOLO FPS)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        lvl = sys_metrics["overall_level"]
        if lvl == "FIRE":
            lvl_txt, lvl_col = "🚨 BÁO CHÁY", "#EF4444"
        elif lvl == "SUSPECT":
            lvl_txt, lvl_col = "⚠️ NGHI NGỜ", "#F59E0B"
        elif lvl == "CONN_ERROR":
            lvl_txt, lvl_col = "🔌 LỖI KẾT NỐI", "#F97316"
        elif lvl == "STANDBY":
            lvl_txt, lvl_col = "⚪ CHỜ KẾT NỐI", "#94A3B8"
        else:
            lvl_txt, lvl_col = "🟢 ĐANG GIÁM SÁT", "#10B981"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">TRẠNG THÁI HỆ THỐNG</div>'
            f'<div class="kpi-value" style="color:{lvl_col};">{lvl_txt}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">CAMERA KẾT NỐI (THẬT / VIDEO)</div>'
            f'<div class="kpi-value">{sys_metrics["connected_count"]}/9 '
            f'<span style="font-size:13px; color:#94A3B8;">({sys_metrics["real_cam_count"]} Cam, {sys_metrics["test_vid_count"]} Video)</span></div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">FPS ĐỌC LUỒNG / FPS YOLO</div>'
            f'<div class="kpi-value" style="color:#38BDF8;">{sys_metrics["avg_fps"]} / {sys_metrics["avg_infer_fps"]} FPS</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">SỰ KIỆN THỰC TẾ (CHƯA XỬ LÝ)</div>'
            f'<div class="kpi-value" style="color:#F59E0B;">{db_stats["total_real"]} ({db_stats["unconfirmed"]} mới)</div></div>',
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-title">CHẾ ĐỘ HOẠT ĐỘNG</div>'
            f'<div class="kpi-value" style="color:#A855F7; font-size:16px;">{sys_metrics["operating_mode"]}</div></div>',
            unsafe_allow_html=True,
        )

    # Banner trạng thái toàn cục (Đảm bảo Mục 10.E: Khi mất tín hiệu hiển thị cảnh báo lỗi kết nối riêng, KHÔNG hiển thị an toàn)
    overall = sys_metrics["overall_level"]
    if overall == "FIRE":
        cams_str = ", ".join(sys_metrics["fire_cameras"])
        st.markdown(
            f'<div class="banner-fire">🚨 CẢNH BÁO KHẨN CẤP: PHÁT HIỆN LỬA/KHÓI TẠI [{cams_str}]! '
            f'YÊU CẦU KIỂM TRA XÁC NHẬN NGAY! 🚨</div>',
            unsafe_allow_html=True,
        )
    elif overall == "SUSPECT":
        cams_str = ", ".join(sys_metrics["suspect_cameras"])
        st.markdown(
            f'<div class="banner-suspect">⚠️ CẢNH BÁO SỚM (NGHI NGỜ): Phát hiện dấu hiệu lửa/khói tại [{cams_str}], '
            f'đang xác minh chuỗi khung hình k/N...</div>',
            unsafe_allow_html=True,
        )
    elif overall == "CONN_ERROR":
        err_cams = ", ".join(sys_metrics["error_cameras"])
        st.markdown(
            f'<div class="banner-conn-error">🔌 CẢNH BÁO LỖI KẾT NỐI CAMERA: Luồng [{err_cams}] đang mất tín hiệu hoặc đang thử kết nối lại (RECONNECTING / ERROR). '
            f'Không thể khẳng định an toàn tại khu vực mất tín hiệu!</div>',
            unsafe_allow_html=True,
        )
    elif overall == "STANDBY":
        st.markdown(
            '<div class="banner-standby">⚪ CHƯA CÓ CAMERA NÀO ĐANG KẾT NỐI — Nhấn "🔌 Kết nối" trên từng camera hoặc trên thanh công cụ để bắt đầu giám sát.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner-safe">🛡️ ĐANG GIÁM SÁT TRỰC TIẾP ({sys_metrics["connected_count"]} LUỒNG CONNECTED) — '
            f'CHƯA PHÁT HIỆN NGUY CƠ CHÁY TRÊN CÁC CAMERA ĐANG HOẠT ĐỘNG</div>',
            unsafe_allow_html=True,
        )

    col_grid, col_alert = st.columns([3.1, 1.15])

    with col_grid:
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
                        else (
                            "border-color:#F59E0B;"
                            if snap["alert_level"] == "SUSPECT"
                            else ("border-color:#F97316;" if snap["status"] in ("ERROR", "RECONNECTING") else "")
                        )
                    )
                    src_tag = (
                        "🎥 VIDEO_TEST"
                        if snap["source_type"] == "VIDEO_TEST"
                        else ("📷 WEBCAM" if snap["source_type"] == "WEBCAM" else "🌐 IP_CAMERA")
                    )

                    if snap["status"] in ("ERROR", "RECONNECTING"):
                        det_txt = f'<b style="color:#F97316;">⚠️ Lỗi tín hiệu ({snap["error_detail"][:28]})</b>'
                    elif snap["status"] != "CONNECTED":
                        det_txt = '<span style="color:#94A3B8;">Chưa giám sát (Ngắt kết nối)</span>'
                    elif snap["primary_label"]:
                        det_txt = f'<b style="color:#EF4444;">🔥 {snap["primary_label"]} ({int(snap["max_confidence"]*100)}%)</b>'
                    else:
                        det_txt = '<span style="color:#10B981;">Đang giám sát (Không có lửa/khói)</span>'

                    # Kết xuất trực tiếp thẻ Camera + ảnh JPEG Base64 (đã nén bằng C++ trong luồng nền)
                    # giúp loại bỏ hoàn toàn độ trễ nén PNG của st.image() và chống nháy hình
                    st.markdown(
                        f'<div class="cam-card-header" style="{lvl_border}">'
                        f'<span>{snap["name"][:24]} <small style="color:#94A3B8;">[{src_tag}]</small></span>{badge}</div>'
                        f'<div style="background:#050811; border-left:1px solid #1E293B; border-right:1px solid #1E293B; {lvl_border} text-align:center;">'
                        f'<img src="data:image/jpeg;base64,{snap["jpeg_b64"]}" '
                        f'style="width:100%; height:195px; object-fit:contain; display:block;" />'
                        f'</div>'
                        f'<div class="cam-card-footer" style="{lvl_border}">'
                        f'📹 <b>FPS đọc:</b> {snap["capture_fps"]:.1f} &nbsp;|&nbsp; '
                        f'🧠 <b>FPS YOLO:</b> {snap["inference_fps"]:.1f} &nbsp;|&nbsp; '
                        f'🕒 {snap["last_update_str"]}<br>'
                        f'🎯 <b>Xác minh k/N:</b> {snap["active_hits"]}/{snap["window_size"]} &nbsp;|&nbsp; '
                        f'{det_txt}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        if st.button("🔌", key=f"start_{cid}", use_container_width=True, help=f"Kết nối {cid}"):
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
                        if st.button("⏹", key=f"stop_{cid}", use_container_width=True, help=f"Ngắt kết nối {cid}"):
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
            ack_badge = "✅ ĐÃ XÁC NHẬN" if is_ack else "🔴 CHƯA XÁC NHẬN (CẦN XỬ LÝ)"
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
        st.markdown("### 📜 NHẬT KÝ HỆ THỐNG (ĐÃ ẨN MẬT KHẨU RTSP)")
        logs_text = "\n".join(cam_system.get_recent_logs(limit=16))
        st.code(logs_text if logs_text else "Chưa có nhật ký.", language="text")


# ==============================================================================
# 6. ĐIỀU HƯỚNG CÁC CHẾ ĐỘ MÀN HÌNH
# ==============================================================================
st.markdown(
    """
    <div class="pccc-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:19px; font-weight:800; color:#F8FAFC;">
                    🔥 HỆ THỐNG GIÁM SÁT & CẢNH BÁO CHÁY SỚM ĐA LUỒNG (9 CAMERA 3×3 & CAMERA THỰC TẾ)
                </span>
                <div style="font-size:11.5px; color:#94A3B8;">
                    Hỗ trợ Video Kiểm Thử (VIDEO_TEST) • Webcam USB (WEBCAM) • Camera Giám Sát IP/RTSP (IP_CAMERA) • Đa Luồng 9 Camera (MULTI_CAMERA)
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if nav_tab == "🎥 Giám Sát 9 Camera (3×3)":
    render_quick_mode_selector()

    refresh_interval = max(0.25, float(cam_system.settings.ui_refresh_sec))
    if hasattr(st, "fragment") and st.session_state.live_refresh:
        @st.fragment(run_every=refresh_interval)
        def _auto_live_fragment():
            render_monitoring_dashboard()

        _auto_live_fragment()
    else:
        render_monitoring_dashboard()
        if st.session_state.live_refresh and cam_system.get_system_metrics()["connected_count"] > 0:
            time.sleep(refresh_interval)
            st.rerun()

    st.caption(
        "⚠️ **Khuyến cáo vận hành PCCC:** Hệ thống ứng dụng thị giác máy tính (YOLO + xác minh thời gian) để hỗ trợ phát hiện sớm lửa và khói; "
        "không tuyên bố bảo đảm phát hiện 100% mọi đám cháy trong mọi điều kiện che khuất/thời tiết. Cần kiểm thử thực tế tại hiện trường và kết hợp hệ thống báo cháy chuyên dụng."
    )

elif nav_tab == "⚙️ Cấu Hình Nguồn Camera & RTSP":
    st.subheader("⚙️ Cấu Hình Chi Tiết 9 Camera Thực Tế (VIDEO_TEST / WEBCAM / IP_CAMERA) & Model YOLO")

    # 1. Cấu hình Model YOLO
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

    # 2. Cấu hình từng camera trong 9 camera (Mục 10.B)
    st.markdown("---")
    st.markdown("### 2. Quản Lý & Kiểm Tra Kết Nối Từng Camera (Tối Đa 9 Camera)")
    st.info(
        "🔒 **Chính sách bảo mật RTSP:** Mật khẩu trong URL RTSP được nhập qua trường ẩn (`type='password'`), "
        "tự động che bằng `******` sau khi lưu cấu hình và tuyệt đối không ghi vào log hệ thống hay GitHub."
    )

    local_vids = list_local_videos()
    snapshots_map = {s["cam_id"]: s for s in cam_system.get_all_snapshots()}

    for idx, cfg in enumerate(cam_system.camera_configs):
        cid = cfg.cam_id
        snap = snapshots_map[cid]
        badge = status_badge_html(snap["status"])

        with st.expander(
            f"📹 {cid} — {cfg.name} | Loại: [{cfg.source_type}] | Nguồn: {cfg.masked_source} | Trạng thái: {snap['status']}",
            expanded=(idx < 2),
        ):
            st.markdown(f"**Trạng thái hiện tại:** {badge} &nbsp;|&nbsp; **Nguồn đã lưu (ẩn mật khẩu):** `{cfg.masked_source}`", unsafe_allow_html=True)

            col_cfg, col_prev = st.columns([2.2, 1.2])
            with col_cfg:
                r1_c1, r1_c2, r1_c3 = st.columns([1.5, 1.2, 1.0])
                with r1_c1:
                    new_name = st.text_input("Tên Camera:", value=cfg.name, key=f"cname_{cid}")
                with r1_c2:
                    stype_idx = SOURCE_TYPES.index(cfg.source_type) if cfg.source_type in SOURCE_TYPES else 0
                    new_stype = st.selectbox(
                        "Loại nguồn đầu vào:",
                        options=SOURCE_TYPES,
                        index=stype_idx,
                        key=f"cstype_{cid}",
                    )
                with r1_c3:
                    new_enabled = st.checkbox("Bật camera này", value=cfg.enabled, key=f"cenab_{cid}")
                    new_autorec = st.checkbox("Tự kết nối lại", value=cfg.auto_reconnect, key=f"crec_{cid}")

                # Hiển thị ô nhập nguồn tương ứng với loại nguồn đã chọn
                new_source_val = cfg.source
                new_webcam_idx = cfg.webcam_index

                if new_stype == "VIDEO_TEST":
                    vc1, vc2 = st.columns([1.4, 1.8])
                    with vc1:
                        pick_v = st.selectbox(
                            "Chọn từ kho video kiểm thử:",
                            options=["-- Giữ nguyên / Tự nhập --"] + local_vids,
                            key=f"cpickv_{cid}",
                        )
                    with vc2:
                        init_v = cfg.source if cfg.source_type == "VIDEO_TEST" else (local_vids[0] if local_vids else "")
                        if pick_v != "-- Giữ nguyên / Tự nhập --":
                            init_v = pick_v
                        new_source_val = st.text_input(
                            "Đường dẫn file video kiểm thử (.mp4, .avi):",
                            value=init_v,
                            key=f"cvidpath_{cid}",
                        )

                elif new_stype == "WEBCAM":
                    new_webcam_idx = int(
                        st.number_input(
                            "Chỉ số thiết bị Webcam máy tính (0 = Webcam mặc định, 1 = Webcam USB...):",
                            min_value=0,
                            max_value=10,
                            value=int(cfg.webcam_index),
                            step=1,
                            key=f"cwebidx_{cid}",
                        )
                    )
                    new_source_val = str(new_webcam_idx)

                elif new_stype == "IP_CAMERA":
                    st.caption(f"URL RTSP đang lưu (đã che mật khẩu): `{cfg.masked_source}`")
                    rtsp_entered = st.text_input(
                        "Nhập địa chỉ RTSP mới (Mật khẩu được ẩn hoàn toàn):",
                        value="",
                        type="password",
                        placeholder="rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101 (Để trống nếu giữ nguyên)",
                        key=f"crtsp_{cid}",
                    )
                    new_source_val = rtsp_entered.strip() if rtsp_entered.strip() else cfg.source

                # 4 nút chức năng chuẩn theo Mục 10.B: Lưu cấu hình, Kiểm tra kết nối, Kết nối, Ngắt kết nối
                nb1, nb2, nb3, nb4 = st.columns(4)
                with nb1:
                    if st.button("💾 Lưu cấu hình", key=f"btn_save_{cid}", type="primary", use_container_width=True):
                        cam_system.update_camera_source(
                            cam_id=cid,
                            new_name=new_name,
                            new_source=new_source_val,
                            source_type=new_stype,
                            webcam_index=new_webcam_idx,
                            enabled=new_enabled,
                            auto_reconnect=new_autorec,
                            auto_restart=False,
                        )
                        st.success(f"Đã lưu cấu hình {cid} ({mask_rtsp_credentials(new_source_val)})!")
                        st.rerun()

                with nb2:
                    if st.button("🩺 Kiểm tra kết nối", key=f"btn_test_{cid}", use_container_width=True):
                        cam_system.update_camera_source(
                            cam_id=cid,
                            new_name=new_name,
                            new_source=new_source_val,
                            source_type=new_stype,
                            webcam_index=new_webcam_idx,
                            enabled=new_enabled,
                            auto_reconnect=new_autorec,
                            auto_restart=False,
                        )
                        ok, t_msg, prev_img = cam_system.test_camera_connection(cid)
                        st.session_state.test_preview_results[cid] = (ok, t_msg, prev_img)
                        st.rerun()

                with nb3:
                    if st.button("🔌 Kết nối", key=f"btn_conn_{cid}", use_container_width=True):
                        cam_system.update_camera_source(
                            cam_id=cid,
                            new_name=new_name,
                            new_source=new_source_val,
                            source_type=new_stype,
                            webcam_index=new_webcam_idx,
                            enabled=True,
                            auto_reconnect=new_autorec,
                            auto_restart=True,
                        )
                        st.rerun()

                with nb4:
                    if st.button("⏹ Ngắt kết nối", key=f"btn_disc_{cid}", use_container_width=True):
                        cam_system.stop_camera(cid)
                        st.rerun()

                if cid in st.session_state.test_preview_results:
                    t_ok, t_msg, _ = st.session_state.test_preview_results[cid]
                    if t_ok:
                        st.success(f"✅ {t_msg}")
                    else:
                        st.error(f"❌ {t_msg}")

            with col_prev:
                st.markdown("**🖼️ Hình ảnh xem trước (Preview):**")
                if snap["status"] == "CONNECTED":
                    st.image(snap["frame_rgb"], caption=f"Đang kết nối trực tiếp ({snap['resolution']})", channels="RGB", use_container_width=True)
                elif cid in st.session_state.test_preview_results:
                    _, _, prev_rgb = st.session_state.test_preview_results[cid]
                    st.image(prev_rgb, caption=f"Ảnh kiểm tra kết nối {cid}", channels="RGB", use_container_width=True)
                elif snap["preview_rgb"] is not None:
                    st.image(snap["preview_rgb"], caption=f"Ảnh xem trước gần nhất {cid}", channels="RGB", use_container_width=True)
                else:
                    st.image(snap["frame_rgb"], caption="Nhấn '🩺 Kiểm tra kết nối' hoặc '🔌 Kết nối' để xem trước", channels="RGB", use_container_width=True)

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

        csv_export_path = f"lich_su_su_kien_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        ce1, _ = st.columns([1.2, 3])
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
    s3.metric("Camera Đang Kết Nối", f"{sys_m['connected_count']} / {sys_m['total_cameras']}")
    s4.metric("FPS Đọc Luồng / FPS YOLO", f"{sys_m['avg_fps']} / {sys_m['avg_infer_fps']} FPS")

    st.markdown("---")
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### 📈 Hiệu Năng FPS Đọc Luồng & FPS Suy Luận YOLO Theo Camera")
        df_fps = pd.DataFrame(
            [
                {
                    "Camera": s["cam_id"],
                    "Tên Camera": s["name"],
                    "FPS Đọc Khung Hình": s["capture_fps"],
                    "FPS Suy Luận YOLO": s["inference_fps"],
                    "Trạng Thái": s["status"],
                }
                for s in snaps
            ]
        )
        if HAS_PLOTLY:
            fig_fps = px.bar(
                df_fps,
                x="Camera",
                y=["FPS Đọc Khung Hình", "FPS Suy Luận YOLO"],
                barmode="group",
                hover_data=["Tên Camera", "Trạng Thái"],
                template="plotly_dark",
            )
            fig_fps.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_fps, use_container_width=True)
        else:
            st.dataframe(df_fps, use_container_width=True)

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
