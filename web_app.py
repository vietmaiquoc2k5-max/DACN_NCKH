import os
import cv2
import time
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Import Central Modules
from alert_manager import AlertManager
from detector import FireDetector

# 1. Page Configuration (Premium Dark look)
st.set_page_config(
    page_title="AI Fire Sentinel Web Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling override for custom items
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    div[data-testid="stSidebar"] {
        background-color: #1A202C;
    }
    .stButton>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Session State Initialization
if "alert_manager" not in st.session_state:
    st.session_state.alert_manager = AlertManager()
if "detector" not in st.session_state:
    st.session_state.detector = FireDetector()
if "running" not in st.session_state:
    st.session_state.running = False
if "conf_history" not in st.session_state:
    st.session_state.conf_history = [0.0] * 50
if "latest_alert" not in st.session_state:
    st.session_state.latest_alert = None

# Thread-safe global log list using Streamlit cache_resource
@st.cache_resource
def get_detector_logs():
    return []

detector_logs = get_detector_logs()

def append_web_log(message):
    """Callback to store system log lines safely in a cached list."""
    timestamp = time.strftime("[%H:%M:%S]")
    log_line = f"{timestamp} {message}"
    detector_logs.append(log_line)
    if len(detector_logs) > 40:
        detector_logs.pop(0)

# Connect log callback to detector
st.session_state.detector.log_callback = append_web_log

# 3. Sidebar Layout (Configurations)
st.sidebar.markdown("# ⚙️ CẤU HÌNH HỆ THỐNG")

# Input Camera Source selector
source_option = st.sidebar.selectbox(
    "Nguồn Camera Đầu Vào:",
    ["Webcam 0", "Webcam 1", "File Video", "IP Camera RTSP"]
)

source_input = ""
if source_option == "File Video":
    source_input = st.sidebar.text_input("Đường dẫn file video cục bộ:")
elif source_option == "IP Camera RTSP":
    source_input = st.sidebar.text_input("Nhập link RTSP của Camera (hoặc http video url):")

# Telegram bot configurations
st.sidebar.markdown("### 🤖 Cấu Hình Telegram Bot")
telegram_token = st.sidebar.text_input("Telegram Token Bot:", type="password", value=os.environ.get("TELEGRAM_TOKEN", ""))
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID:", value=os.environ.get("TELEGRAM_CHAT_ID", ""))

# Sound setting
mute_alarm = st.sidebar.checkbox("Tắt âm thanh báo động còi hú", value=False)

# Update config values in real-time
st.session_state.alert_manager.update_config(telegram_token, telegram_chat_id, mute_alarm)

# Test telegram button
if st.sidebar.button("🧪 Gửi Thử Tin Nhắn Tele", width="stretch"):
    if telegram_token and telegram_chat_id:
        append_web_log("Đang gửi thử nghiệm thông báo tới Telegram...")
        success, error_msg = st.session_state.alert_manager.send_telegram_alert_sync(
            None, 
            message="🧪 Kiểm thử hệ thống Web: Cấu hình Telegram hoạt động chính xác! ✅"
        )
        if success:
            st.sidebar.success("Đã gửi tin nhắn test! Hãy kiểm tra Telegram.")
        else:
            append_web_log(f"Lỗi Telegram: {error_msg}")
            st.sidebar.error(f"Gửi thất bại: {error_msg}")
    else:
        st.sidebar.warning("Vui lòng điền đủ Token và Chat ID.")

# Parameter adjustment sliders
st.sidebar.markdown("### 🎛️ Tinh Chỉnh Ngưỡng AI")
conf_threshold = st.sidebar.slider("Ngưỡng nhận diện (Confidence):", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
st.session_state.detector.conf_threshold = conf_threshold

consecutive_limit = st.sidebar.slider("Độ trễ cảnh báo (Số frames liên tiếp):", min_value=2, max_value=30, value=10, step=1)
st.session_state.detector.consecutive_frames_limit = consecutive_limit

# Control buttons (Start / Stop)
st.sidebar.markdown("---")
col_start, col_stop = st.sidebar.columns(2)

with col_start:
    if st.button("▶ BẮT ĐẦU", width="stretch", type="primary"):
        st.session_state.running = True

with col_stop:
    if st.button("■ DỪNG", width="stretch"):
        st.session_state.running = False
        st.session_state.detector.stop()
        st.session_state.alert_manager.stop_alarm()

# Sidebar system status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("#### Trạng thái hoạt động:")
cam_status = "Đang chạy" if st.session_state.running else "Đã dừng"
st.sidebar.info(f"Camera: {cam_status}")
tele_status = "Sẵn sàng" if (telegram_token and telegram_chat_id) else "Chưa cấu hình"
st.sidebar.warning(f"Telegram: {tele_status}")

# 4. Main Page View Grid (2 Columns layout)
st.title("🔥 Sentinel AI - Web Dashboard Giám Sát Cháy")

col_main, col_details = st.columns([2, 1])

# Left column: Video stream & Matplotlib chart
with col_main:
    # Alert Status Banner
    banner_placeholder = st.empty()
    if st.session_state.running:
        banner_placeholder.success("💚 ĐANG GIÁM SÁT - HOẠT ĐỘNG AN TOÀN")
    else:
        banner_placeholder.subheader("⚪ Hệ thống đang tắt")
        
    # Main video frame container
    video_placeholder = st.empty()
    video_placeholder.info("Màn hình Camera livestream sẽ hiển thị tại đây khi bạn bấm nút '▶ BẮT ĐẦU'.")
    
    # Live confidence chart
    st.markdown("### 📈 Đồ thị biến thiên độ tin cậy nhận diện")
    chart_placeholder = st.empty()
    # Initial empty chart
    chart_placeholder.line_chart(st.session_state.conf_history)

# Right column: Smart History snapshot & system log
with col_details:
    st.markdown("### 🚨 CẢNH BÁO MỚI NHẤT")
    thumb_placeholder = st.empty()
    thumb_placeholder.info("Chưa có ảnh chụp đám cháy lịch sử.")
    
    meta_placeholder = st.empty()
    
    # System logs
    st.markdown("### 📋 NHẬT KÝ HOẠT ĐỘNG")
    logs_placeholder = st.empty()
    logs_placeholder.code("\n".join(detector_logs[::-1]), language="text")

# 5. Core Video Loops (Triggered if running is True)
if st.session_state.running:
    # Resolve source
    source = "0"
    if source_option == "Webcam 1":
        source = "1"
    elif source_option in ["File Video", "IP Camera RTSP"] and source_input:
        source = source_input
        
    # Start detector thread if not already running
    if not st.session_state.detector.running:
        st.session_state.detector.start(source, st.session_state.alert_manager)
        append_web_log("Bắt đầu giám sát luồng camera...")
        
    # Running loops update loop
    while st.session_state.running:
        # Check camera connection state
        if not st.session_state.detector.running:
            banner_placeholder.error("❌ LỖI KẾT NỐI CAMERA HOẶC HẾT VIDEO")
            st.session_state.running = False
            break
            
        # 1. Fetch latest data from background thread
        frame = st.session_state.detector.latest_annotated_frame
        conf = st.session_state.detector.latest_conf
        
        # 2. Update Live Stream Image
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", width="stretch")
            
        # 3. Update Live Chart
        st.session_state.conf_history.append(conf)
        st.session_state.conf_history.pop(0)
        chart_placeholder.line_chart(st.session_state.conf_history)
        
        # 4. Check if alert is actively triggered to update Smart History
        is_fire_detected = st.session_state.detector.consecutive_fire_frames >= st.session_state.detector.consecutive_frames_limit
        if is_fire_detected:
            # Change banner to red
            banner_placeholder.error("🚨🚨 CẢNH BÁO: PHÁT HIỆN CÓ CHÁY/KHÓI! 🚨🚨")
            
            # Store latest alert frame in memory
            if frame is not None:
                st.session_state.latest_alert = {
                    "image": rgb_frame,
                    "time": time.strftime("%H:%M:%S"),
                    "conf": int(conf * 100)
                }
        else:
            banner_placeholder.success("💚 ĐANG GIÁM SÁT - HOẠT ĐỘNG AN TOÀN")
            
        # Update Smart History UI
        if st.session_state.latest_alert is not None:
            thumb_placeholder.image(st.session_state.latest_alert["image"], width="stretch")
            meta_placeholder.markdown(
                f"**Thời gian phát hiện:** `{st.session_state.latest_alert['time']}`  \n"
                f"**Độ tin cậy nhận diện:** `{st.session_state.latest_alert['conf']}%`"
            )
            
        # 5. Update Logs (latest first)
        logs_placeholder.code(
            "\n".join(detector_logs[::-1]), 
            language="text"
        )
        
        # Lock FPS to ~25-30 frames per second
        time.sleep(0.035)
