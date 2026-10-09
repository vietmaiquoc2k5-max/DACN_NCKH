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

# 1. Page Configuration
st.set_page_config(
    page_title="AI FIRE SENTINEL PRO - Đăng Nhập & Cảnh Báo Cháy",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS Styling
st.markdown("""
<style>
    .stApp {
        background-color: #090D16;
        color: #F8FAFC;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    div[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .kpi-title {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 800;
        color: #00F2FE;
    }
    .kpi-safe {
        color: #10B981;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
    .safe-banner {
        background: linear-gradient(90deg, #065F46 0%, #047857 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 14px 20px;
        color: #ECFDF5;
        font-weight: bold;
        font-size: 16px;
        text-align: center;
        margin-bottom: 15px;
    }
    .danger-banner {
        background: linear-gradient(90deg, #991B1B 0%, #DC2626 100%);
        border: 1px solid #F43F5E;
        border-radius: 12px;
        padding: 14px 20px;
        color: #FFF1F2;
        font-weight: bold;
        font-size: 16px;
        text-align: center;
        margin-bottom: 15px;
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
if "logged_user" not in st.session_state:
    st.session_state.logged_user = None

ACCOUNTS = {
    "user": {"password": "user123", "role": "USER", "name": "Nhân Viên Bảo Vệ"},
    "admin": {"password": "admin123", "role": "ADMIN", "name": "Quản Trị Viên Hệ Thống"}
}

@st.cache_resource
def get_detector_logs():
    return []

detector_logs = get_detector_logs()

def append_web_log(message):
    timestamp = time.strftime("[%H:%M:%S]")
    log_line = f"{timestamp} {message}"
    detector_logs.append(log_line)
    if len(detector_logs) > 50:
        detector_logs.pop(0)

st.session_state.detector.log_callback = append_web_log

# ==============================================================================
# 3. MANDATORY LOGIN GATE
# ==============================================================================
if st.session_state.logged_user is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c_left, c_mid, c_right = st.columns([1, 1.2, 1])
    
    with c_mid:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 50px;">🔥</span>
            <h2 style="color: #00F2FE; margin: 5px 0 0 0;">FIRE SENTINEL AI</h2>
            <p style="color: #64748B; font-size: 13px; font-weight: bold;">CỔNG ĐĂNG NHẬP XÁC THỰC AN NINH</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("mandatory_login_form"):
            user_input = st.text_input("Tên đăng nhập:", placeholder="user hoặc admin")
            pass_input = st.text_input("Mật khẩu:", type="password", placeholder="mật khẩu...")
            submit_login = st.form_submit_button("🔓 ĐĂNG NHẬP HỆ THỐNG", use_container_width=True)
            
            if submit_login:
                u = user_input.strip()
                p = pass_input.strip()
                if u in ACCOUNTS and ACCOUNTS[u]["password"] == p:
                    st.session_state.logged_user = {
                        "username": u,
                        "role": ACCOUNTS[u]["role"],
                        "name": ACCOUNTS[u]["name"]
                    }
                    st.success(f"Đăng nhập thành công! Xin chào {ACCOUNTS[u]['name']}.")
                    st.rerun()
                else:
                    st.error("❌ Sai Tên đăng nhập hoặc Mật khẩu!")
                    
        st.info("""
        💡 **TÀI KHOẢN TRUY CẬP MẶC ĐỊNH:**  
        • **Người dùng (Bảo vệ):** User: `user` | Pass: `user123`  
        • **Quản trị viên (Admin):** User: `admin` | Pass: `admin123`
        """)
        
    st.stop()

# ==============================================================================
# 4. MAIN DASHBOARD
# ==============================================================================
curr_user = st.session_state.logged_user if st.session_state.logged_user else {"username": "guest", "role": "GUEST", "name": "Guest"}

st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
    <span style="font-size: 32px;">🔥</span>
    <div>
        <h2 style="margin: 0; color: #00F2FE; font-size: 20px;">FIRE SENTINEL</h2>
        <span style="color: #64748B; font-size: 11px; font-weight: bold;">AI VISION PRO DASHBOARD</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
if curr_user.get("role") == "ADMIN":
    st.sidebar.success(f"🔑 ADMIN: {curr_user.get('username')} ({curr_user.get('name')})")
else:
    st.sidebar.info(f"👤 USER: {curr_user.get('username')} ({curr_user.get('name')})")

if st.sidebar.button("🔒 Đăng Xuất Hệ Thống", use_container_width=True):
    st.session_state.running = False
    st.session_state.detector.stop()
    st.session_state.alert_manager.stop_alarm()
    st.session_state.logged_user = None
    st.rerun()

st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "MENU ĐIỀU HÀNH:",
    ["👁️ Giao Diện Người Dùng", "📋 Lịch Sử Cảnh Báo CSDL", "⚙️ Admin Quản Lý"]
)

mute_alarm = st.sidebar.checkbox("🔕 Tắt âm thanh báo động còi hú", value=False)
st.session_state.alert_manager.update_config(mute_alarm)

if st.sidebar.button("📁 Mở thư mục ảnh cảnh báo", use_container_width=True):
    alerts_dir = os.path.abspath("alerts")
    if not os.path.exists(alerts_dir):
        os.makedirs(alerts_dir)
    try:
        os.startfile(alerts_dir)
    except Exception:
        import subprocess
        subprocess.Popen(f'explorer "{alerts_dir}"')

st.sidebar.markdown("---")
st.sidebar.markdown("#### TRẠNG THÁI HỆ THỐNG:")
cam_status = "🟢 Đang giám sát" if st.session_state.running else "🔴 Đã dừng"
st.sidebar.info(f"Camera: {cam_status}")
st.sidebar.success("Kênh thông báo: CSDL Nội Bộ SQLite")

# ==============================================================================
# TAB 1: USER MONITORING VIEW
# ==============================================================================
if app_mode == "👁️ Giao Diện Người Dùng":
    st.title("👁️ Màn Hình Giám Sát Cảnh Báo Cháy Thời Gian Thực")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        status_text = "🟢 AN TOÀN" if not st.session_state.detector.alert_callback else ("🚨 CẢNH BÁO" if st.session_state.running else "⚪ TẮT")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">TRẠNG THÁI GIÁM SÁT</div>
            <div class="kpi-value {'kpi-safe' if st.session_state.running else ''}">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        latest_c = int(st.session_state.detector.latest_conf * 100)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ĐỘ TIN CẬY AI KHUNG HÌNH</div>
            <div class="kpi-value">{latest_c}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        alert_total = len(st.session_state.alert_manager.get_all_alerts(limit=500))
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">TỔNG VỤ CẢNH BÁO CSDL</div>
            <div class="kpi-value" style="color: #F59E0B;">{alert_total} VỤ</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">MÔ HÌNH NHẬN DIỆN</div>
            <div class="kpi-value" style="color: #A855F7;">YOLOv8 AI</div>
        </div>
        """, unsafe_allow_html=True)
    with st.expander("📷 TÙY CHỈNH NGUỒN CAMERA ĐẦU VÀO (CHO BẢO VỆ / USER)", expanded=True):
        col_src1, col_src2 = st.columns([1, 2])
        with col_src1:
            user_src_opt = st.selectbox(
                "📹 Chọn Nguồn Camera:",
                ["Webcam 0", "Webcam 1", "File Video", "IP Camera RTSP"],
                key="user_cam_source_choice"
            )
        with col_src2:
            user_src_path = ""
            if user_src_opt == "File Video":
                user_src_path = st.text_input("Đường dẫn file video cục bộ (VD: video.mp4):", key="user_vid_input")
            elif user_src_opt == "IP Camera RTSP":
                user_src_path = st.text_input("URL luồng IP Camera (rtsp://...):", key="user_rtsp_input")
            else:
                st.caption(f"Đang chọn thiết bị mặc định: **{user_src_opt}**")

    # Resolve camera source selected by user
    resolved_cam = "0"
    if user_src_opt == "Webcam 1":
        resolved_cam = "1"
    elif user_src_opt in ["File Video", "IP Camera RTSP"] and user_src_path.strip():
        resolved_cam = user_src_path.strip()
    st.session_state.user_cam_source = resolved_cam

    c_start, c_stop, c_space = st.columns([1, 1, 3])
    with c_start:
        if st.button("▶ BẮT ĐẦU GIÁM SÁT", type="primary", use_container_width=True):
            st.session_state.running = True
    with c_stop:
        if st.button("■ DỪNG HỆ THỐNG", use_container_width=True):
            st.session_state.running = False
            st.session_state.detector.stop()
            st.session_state.alert_manager.stop_alarm()

    col_main, col_details = st.columns([2.5, 1])

    with col_main:
        banner_placeholder = st.empty()
        if st.session_state.running:
            banner_placeholder.markdown('<div class="safe-banner">💚 ĐANG GIÁM SÁT - HOẠT ĐỘNG AN TOÀN</div>', unsafe_allow_html=True)
        else:
            banner_placeholder.info("⚪ Hệ thống đang tắt. Nhấn '▶ BẮT ĐẦU GIÁM SÁT' để kết nối camera.")

        video_placeholder = st.empty()
        video_placeholder.info("Màn hình Livestream Camera sẽ hiển thị tại đây khi hệ thống hoạt động.")

        st.markdown("### 📈 Đồ thị biến thiên độ tin cậy AI thời gian thực")
        chart_placeholder = st.empty()
        chart_placeholder.line_chart(st.session_state.conf_history)

    with col_details:
        st.markdown("### 🚨 CẢNH BÁO MỚI NHẤT")
        thumb_placeholder = st.empty()
        thumb_placeholder.info("Chưa phát hiện sự cố cháy nào.")
        meta_placeholder = st.empty()

        st.markdown("### 📋 NHẬT KÝ THÔNG BÁO TỨC THỜI")
        logs_placeholder = st.empty()
        logs_placeholder.code("\n".join(detector_logs[::-1]), language="text")

    if st.session_state.running:
        source = getattr(st.session_state, "user_cam_source", "0")
        if not st.session_state.detector.running:
            st.session_state.detector.start(source, st.session_state.alert_manager)
            append_web_log(f"Bắt đầu luồng nhận diện camera ({source})...")

        while st.session_state.running:
            if not st.session_state.detector.running:
                banner_placeholder.error("❌ LỖI KẾT NỐI CAMERA HOẶC HẾT LUỒNG VIDEO")
                st.session_state.running = False
                break

            frame = st.session_state.detector.latest_annotated_frame
            conf = st.session_state.detector.latest_conf

            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            st.session_state.conf_history.append(conf)
            st.session_state.conf_history.pop(0)
            chart_placeholder.line_chart(st.session_state.conf_history)

            is_fire_detected = st.session_state.detector.consecutive_fire_frames >= st.session_state.detector.consecutive_frames_limit
            if is_fire_detected:
                banner_placeholder.markdown('<div class="danger-banner">🚨🚨 CẢNH BÁO NGUY HIỂM: PHÁT HIỆN CÓ CHÁY/KHÓI! 🚨🚨</div>', unsafe_allow_html=True)
                if frame is not None:
                    st.session_state.latest_alert = {
                        "image": rgb_frame,
                        "time": time.strftime("%H:%M:%S"),
                        "conf": int(conf * 100)
                    }
            else:
                banner_placeholder.markdown('<div class="safe-banner">💚 ĐANG GIÁM SÁT - HOẠT ĐỘNG AN TOÀN</div>', unsafe_allow_html=True)

            if st.session_state.latest_alert is not None:
                thumb_placeholder.image(st.session_state.latest_alert["image"], use_container_width=True)
                meta_placeholder.markdown(
                    f"**Thời gian:** `{st.session_state.latest_alert['time']}`  \n"
                    f"**Độ tin cậy:** `{st.session_state.latest_alert['conf']}%`"
                )

            logs_placeholder.code("\n".join(detector_logs[::-1]), language="text")
            time.sleep(0.035)

# ==============================================================================
# TAB 2: LỊCH SỬ CẢNH BÁO CSDL (YÊU CẦU QUYỀN ADMIN)
# ==============================================================================
elif app_mode == "📋 Lịch Sử Cảnh Báo CSDL":
    st.title("📋 Trung Tâm Quản Lý Lịch Sử Cảnh Báo (CSDL SQLite)")
    
    if curr_user.get("role") != "ADMIN":
        st.warning("🔒 Khóa Bảo Mật: Trang này yêu cầu quyền ADMIN Quản Trị.")
        st.info(f"Tài khoản hiện tại của bạn là **{curr_user.get('username')}** ({curr_user.get('name')}). Vui lòng đăng xuất và đăng nhập tài khoản Admin (`admin` / `admin123`) để truy cập.")
    else:
        alerts = st.session_state.alert_manager.get_all_alerts(limit=200)
        
        col_filter, col_act1, col_act2 = st.columns([3, 1, 1])
        with col_filter:
            search_q = st.text_input("🔍 Tìm kiếm trong CSDL (Từ khóa / Thời gian / Loại):", "")
            if search_q:
                alerts = st.session_state.alert_manager.get_all_alerts(limit=200, search_query=search_q)

        with col_act1:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Xuất File CSV", use_container_width=True):
                csv_path = f"Bao_Cao_Canh_Bao_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                success, msg = st.session_state.alert_manager.export_csv(csv_path)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        with col_act2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Xóa Tất Cả CSDL", type="primary", use_container_width=True):
                st.session_state.alert_manager.clear_all_alerts()
                st.rerun()

        if alerts:
            df = pd.DataFrame(alerts)
            df_display = df.rename(columns={
                "id": "ID",
                "timestamp": "Thời Gian",
                "label": "Loại Cảnh Báo",
                "confidence": "Độ Tin Cậy",
                "action_taken": "Kết Quả Xử Lý Ca Trực",
                "operator_note": "Ghi Chú Chi Tiết",
                "image_path": "Đường Dẫn Ảnh Snapshot"
            })
            
            st.dataframe(df_display, use_container_width=True)

            st.markdown("### 📸 Xem Chi Tiết Ảnh & Cập Nhật Kết Quả Xử Lý Ca Trực")
            selected_id = st.selectbox("Chọn ID vụ cháy để cập nhật / xem ảnh:", df["id"].tolist())
            
            if selected_id:
                record = next((a for a in alerts if a["id"] == selected_id), None)
                if record:
                    c_img, c_form = st.columns([1.5, 1])
                    with c_img:
                        if record["image_path"] and os.path.exists(record["image_path"]):
                            img = Image.open(record["image_path"])
                            st.image(img, caption=f"Ảnh Snapshot vụ cháy ID #{record['id']} - {record['timestamp']}", use_container_width=True)
                        else:
                            st.warning("Không có file ảnh snapshot.")
                    
                    with c_form:
                        st.markdown(f"**Vụ Cháy ID:** `#{record['id']}`  \n**Thời Gian:** `{record['timestamp']}`  \n**Loại:** `{record['label']}` (`{int(record['confidence']*100)}%`)")
                        
                        with st.form(f"update_action_form_{selected_id}"):
                            action_sel = st.selectbox(
                                "Ghi nhận kết quả ca trực:",
                                ["Chưa xử lý", "Báo động giả (Khói nhẹ/Nhiễu)", "Đã dập tắt tại chỗ", "Nghiêm trọng / Đã báo 114"],
                                index=0
                            )
                            note_text = st.text_input("Ghi chú bổ sung:", value=record.get("operator_note", ""))
                            save_btn = st.form_submit_button("💾 Lưu Nhật Ký Xử Lý", type="primary")
                            
                            if save_btn:
                                st.session_state.alert_manager.update_alert_action(selected_id, action_sel, note_text)
                                st.success(f"Đã lưu kết quả vụ việc #{selected_id}!")
                                st.rerun()

                    if st.button(f"🗑️ Xóa bản ghi ID #{selected_id}"):
                        st.session_state.alert_manager.delete_alert(selected_id)
                        st.success(f"Đã xóa bản ghi #{selected_id}")
                        st.rerun()
        else:
            st.info("Chưa có bản ghi cảnh báo nào trong Cơ sở dữ liệu.")

# ==============================================================================
# TAB 3: ADMIN QUẢN LÝ (YÊU CẦU QUYỀN ADMIN)
# ==============================================================================
elif app_mode == "⚙️ Admin Quản Lý":
    st.title("⚙️ Admin - Cấu Hình & Quản Trị Hệ Thống")
    
    if curr_user.get("role") != "ADMIN":
        st.warning("🔒 Khóa Bảo Mật: Trang này yêu cầu quyền ADMIN Quản Trị.")
        st.info(f"Tài khoản hiện tại của bạn là **{curr_user.get('username')}** ({curr_user.get('name')}). Vui lòng đăng xuất và đăng nhập tài khoản Admin (`admin` / `admin123`) để truy cập.")
    else:
        st.markdown("### 1. Cấu Hình Nguồn Camera Đầu Vào")
        source_option = st.selectbox(
            "Nguồn Camera:",
            ["Webcam 0", "Webcam 1", "File Video", "IP Camera RTSP"]
        )

        source_input = ""
        if source_option == "File Video":
            source_input = st.text_input("Đường dẫn file video cục bộ:")
        elif source_option == "IP Camera RTSP":
            source_input = st.text_input("Nhập link RTSP / HTTP Stream URL:")

        resolved_src = "0"
        if source_option == "Webcam 1":
            resolved_src = "1"
        elif source_option in ["File Video", "IP Camera RTSP"] and source_input:
            resolved_src = source_input
            
        st.session_state.admin_source = resolved_src

        st.markdown("---")
        st.markdown("### 2. Tinh Chỉnh Tham Số Thuật Toán AI")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            conf_thresh = st.slider("Ngưỡng nhận diện (Confidence):", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
            st.session_state.detector.conf_threshold = conf_thresh
        with col_p2:
            sens_limit = st.slider("Độ trễ chống báo giả (Số frames liên tiếp):", min_value=2, max_value=30, value=10, step=1)
            st.session_state.detector.consecutive_frames_limit = sens_limit

        cooldown_val = st.slider("Thời gian giãn cách tự động lưu ảnh cảnh báo (Giây):", min_value=3, max_value=60, value=10, step=1)
        st.session_state.detector.alert_cooldown = cooldown_val

        st.markdown("---")
        st.markdown("### 3. Quản Lý Âm Thanh & Báo Động Khẩn Cấp")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔊 Thử Âm Thanh Còi Hú", use_container_width=True):
                st.session_state.alert_manager.play_alarm()
        with c_btn2:
            if st.button("🔕 Tắt Còi Hú Khẩn Cấp", type="primary", use_container_width=True):
                st.session_state.alert_manager.stop_alarm()

        st.markdown("---")
        st.markdown("### 4. Điều Khiển Bắt Đầu / Dừng Hệ Thống")
        c_adm_start, c_adm_stop = st.columns(2)
        with c_adm_start:
            if st.button("▶ BẮT ĐẦU GIÁM SÁT (ADMIN)", type="primary", use_container_width=True):
                st.session_state.running = True
                st.success("Đã bật hệ thống giám sát!")
        with c_adm_stop:
            if st.button("■ DỪNG HỆ THỐNG (ADMIN)", use_container_width=True):
                st.session_state.running = False
                st.session_state.detector.stop()
                st.session_state.alert_manager.stop_alarm()
                st.info("Đã dừng hệ thống giám sát!")
