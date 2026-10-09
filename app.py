import os
import cv2
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from PIL import Image, ImageTk

# Import matplotlib for live chart rendering
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import custom modules
from alert_manager import AlertManager
from detector import FireDetector

# Set CustomTkinter theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FireAlarmApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("🔥 AI FIRE SENTINEL PRO - Đăng Nhập & Giám Sát Cảnh Báo Cháy")
        self.geometry("1360x820")
        self.minsize(1150, 700)
        self.configure(fg_color="#090D16")  # Deep Cyber Charcoal
        
        # Initialize Core Modules
        self.alert_manager = AlertManager()
        self.detector = FireDetector()
        
        # User Accounts Database
        self.accounts = {
            "user": {"password": "user123", "role": "USER", "name": "Nhân Viên Bảo Vệ"},
            "admin": {"password": "admin123", "role": "ADMIN", "name": "Quản Trị Viên Hệ Thống"}
        }
        
        # Session state
        self.current_user_role = None
        self.current_username = ""
        
        # Connect callbacks
        self.detector.log_callback = self.add_log
        self.detector.status_callback = self.on_detector_status_change
        self.detector.alert_callback = self.on_alert_state_change
        self.alert_manager.notification_callback = self.on_new_in_app_notification
        
        # GUI state & KPI counters
        self.is_flashing = False
        self.alert_active = False
        self.current_tab = "user"
        self.popup_window = None
        self.start_timestamp = time.time()
        self.today_alert_count = 0
        
        # Matplotlib data
        self.conf_history = [0.0] * 50
        
        # Hook window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create Primary Mandatory Login Screen Container
        self.create_mandatory_login_screen()
        
        # Set up Main Dashboard Layout
        self.grid_columnconfigure(0, weight=0, minsize=260)  # Left panel (Sidebar)
        self.grid_columnconfigure(1, weight=1)                # Main Content Area
        self.grid_rowconfigure(0, weight=1)
        
        self.create_sidebar()
        self.create_user_view()     # Tab 1: User View
        self.create_history_view()  # Tab 2: History DB View
        self.create_admin_view()    # Tab 3: Admin Controls View
        
        # Initially hide main dashboard, show login screen mandatory!
        self.show_mandatory_login_screen()
        
        # Load default settings
        self.load_defaults()
        
        # Start GUI polling loops
        self.update_video_loop()
        self.update_chart_loop()
        self.update_uptime_loop()

    # -------------------------------------------------------------
    # 0. PRIMARY MANDATORY LOGIN SCREEN
    # -------------------------------------------------------------
    def create_mandatory_login_screen(self):
        """Creates the full-screen mandatory login UI overlay."""
        self.login_view = ctk.CTkFrame(self, fg_color="#090D16", corner_radius=0)
        
        login_card = ctk.CTkFrame(self.login_view, width=440, height=520, fg_color="#111726", corner_radius=20, border_width=1, border_color="#1E293B")
        login_card.pack(expand=True)
        login_card.pack_propagate(False)
        
        ctk.CTkLabel(login_card, text="🔥", font=ctk.CTkFont(size=48)).pack(pady=(30, 0))
        ctk.CTkLabel(
            login_card, 
            text="FIRE SENTINEL AI", 
            font=ctk.CTkFont(size=22, weight="bold", family="Segoe UI"),
            text_color="#00F2FE"
        ).pack(pady=(5, 0))
        
        ctk.CTkLabel(
            login_card, 
            text="CỔNG ĐĂNG NHẬP XÁC THỰC AN NINH", 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748B"
        ).pack(pady=(2, 20))
        
        form_box = ctk.CTkFrame(login_card, fg_color="transparent")
        form_box.pack(fill="x", padx=35)
        
        ctk.CTkLabel(form_box, text="Tên đăng nhập:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#CBD5E1").pack(anchor="w", pady=(0, 4))
        self.login_user_entry = ctk.CTkEntry(form_box, placeholder_text="Nhập user hoặc admin...", height=40, corner_radius=10, font=ctk.CTkFont(size=13))
        self.login_user_entry.pack(fill="x", pady=(0, 12))
        
        ctk.CTkLabel(form_box, text="Mật khẩu:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#CBD5E1").pack(anchor="w", pady=(0, 4))
        self.login_pass_entry = ctk.CTkEntry(form_box, placeholder_text="Nhập mật khẩu...", show="*", height=40, corner_radius=10, font=ctk.CTkFont(size=13))
        self.login_pass_entry.pack(fill="x", pady=(0, 8))
        self.login_pass_entry.bind("<Return>", lambda e: self.process_primary_login())
        
        self.lbl_login_error = ctk.CTkLabel(form_box, text="", text_color="#EF4444", font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_login_error.pack(pady=(2, 8))
        
        btn_login = ctk.CTkButton(
            form_box, 
            text="🔓 ĐĂNG NHẬP HỆ THỐNG", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7", 
            hover_color="#0369A1",
            height=44,
            corner_radius=10,
            command=self.process_primary_login
        )
        btn_login.pack(fill="x", pady=(5, 15))
        
        note_box = ctk.CTkFrame(login_card, fg_color="#0F172A", corner_radius=10)
        note_box.pack(fill="x", padx=35, pady=(0, 20))
        ctk.CTkLabel(
            note_box, 
            text="💡 TÀI KHOẢN MẶC ĐỊNH:\n• User:  user  | Pass: user123\n• Admin: admin | Pass: admin123",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8",
            justify="left"
        ).pack(padx=12, pady=8)

    def show_mandatory_login_screen(self):
        self.sidebar.grid_forget()
        self.user_view.grid_forget()
        self.history_view.grid_forget()
        self.admin_view.grid_forget()
        self.login_view.pack(fill="both", expand=True)
        self.login_user_entry.focus()

    def process_primary_login(self):
        u = self.login_user_entry.get().strip()
        p = self.login_pass_entry.get().strip()
        
        if u in self.accounts and self.accounts[u]["password"] == p:
            role = self.accounts[u]["role"]
            self.current_user_role = role
            self.current_username = u
            self.is_admin_logged_in = (role == "ADMIN")
            
            self.login_view.pack_forget()
            self.lbl_login_error.configure(text="")
            self.login_pass_entry.delete(0, tk.END)
            
            self.sidebar.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
            self.update_role_ui()
            self.show_tab("user")
            self.add_log(f"🔑 Đã đăng nhập thành công với tài khoản [{u.upper()}] ({self.accounts[u]['name']})!")
        else:
            self.lbl_login_error.configure(text="❌ Sai Tên đăng nhập hoặc Mật khẩu!")

    def process_logout(self):
        if messagebox.askyesno("Đăng Xuất", "Bạn có chắc chắn muốn đăng xuất khỏi hệ thống không?"):
            self.detector.stop()
            self.alert_manager.stop_alarm()
            self.current_user_role = None
            self.current_username = ""
            self.is_admin_logged_in = False
            self.show_mandatory_login_screen()

    # -------------------------------------------------------------
    # 1. SIDEBAR & NAVIGATION
    # -------------------------------------------------------------
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=18, fg_color="#111726", border_width=1, border_color="#1E293B")
        
        logo_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_box.pack(fill="x", padx=15, pady=(25, 15))
        
        logo_icon = ctk.CTkLabel(logo_box, text="🔥", font=ctk.CTkFont(size=32))
        logo_icon.pack(side="left", padx=(0, 10))
        
        title_text_box = ctk.CTkFrame(logo_box, fg_color="transparent")
        title_text_box.pack(side="left", fill="both")
        
        ctk.CTkLabel(
            title_text_box, 
            text="FIRE SENTINEL", 
            font=ctk.CTkFont(size=18, weight="bold", family="Segoe UI"),
            text_color="#00F2FE"
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            title_text_box, 
            text="AI VISION PRO v2.5", 
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748B"
        ).pack(anchor="w")
        
        self.role_card = ctk.CTkFrame(self.sidebar, fg_color="#182234", corner_radius=12, border_width=1, border_color="#334155")
        self.role_card.pack(fill="x", padx=15, pady=(0, 15))
        
        self.lbl_role_status = ctk.CTkLabel(
            self.role_card, 
            text="👤 TK: USER", 
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38BDF8"
        )
        self.lbl_role_status.pack(pady=(10, 2))
        
        self.btn_auth_action = ctk.CTkButton(
            self.role_card, 
            text="🔒 Đăng Xuất", 
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#EF4444", 
            hover_color="#DC2626",
            height=30,
            corner_radius=8,
            command=self.process_logout
        )
        self.btn_auth_action.pack(pady=(4, 10), padx=12, fill="x")
        
        ctk.CTkLabel(
            self.sidebar, 
            text="MENU QUẢN TRỊ", 
            text_color="#475569", 
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=20, pady=(5, 5))
        
        self.btn_tab_user = ctk.CTkButton(
            self.sidebar, 
            text="👁️  GIÁM SÁT NGƯỜI DÙNG", 
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#EA580C",
            text_color="#FFFFFF",
            hover_color="#C2410C",
            height=42,
            corner_radius=10,
            anchor="w",
            command=lambda: self.show_tab("user")
        )
        self.btn_tab_user.pack(fill="x", padx=15, pady=4)
        
        self.btn_tab_history = ctk.CTkButton(
            self.sidebar, 
            text="📋  LỊCH SỬ CẢNH BÁO 🔒", 
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            text_color="#94A3B8",
            hover_color="#1E293B",
            height=42,
            corner_radius=10,
            anchor="w",
            command=lambda: self.show_tab("history")
        )
        self.btn_tab_history.pack(fill="x", padx=15, pady=4)

        self.btn_tab_admin = ctk.CTkButton(
            self.sidebar, 
            text="⚙️  ADMIN CẤU HÌNH 🔒", 
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            text_color="#94A3B8",
            hover_color="#1E293B",
            height=42,
            corner_radius=10,
            anchor="w",
            command=lambda: self.show_tab("admin")
        )
        self.btn_tab_admin.pack(fill="x", padx=15, pady=4)
        
        ctk.CTkLabel(
            self.sidebar, 
            text="THAO TÁC NHANH", 
            text_color="#475569", 
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))
        
        self.btn_mute_emergency = ctk.CTkButton(
            self.sidebar, 
            text="🔕  TẮT CÒI KHẨN CẤP", 
            fg_color="#DC2626", 
            hover_color="#B91C1C",
            font=ctk.CTkFont(weight="bold", size=12),
            height=38,
            corner_radius=10,
            command=self.toggle_mute
        )
        self.btn_mute_emergency.pack(fill="x", padx=15, pady=4)
        
        self.btn_open_alerts = ctk.CTkButton(
            self.sidebar, 
            text="📁  THƯ MỤC THƯ VIỆN", 
            fg_color="#334155", 
            hover_color="#475569",
            font=ctk.CTkFont(weight="bold", size=12),
            height=38,
            corner_radius=10,
            command=self.open_alerts_folder
        )
        self.btn_open_alerts.pack(fill="x", padx=15, pady=4)
        
        status_frame = ctk.CTkFrame(self.sidebar, fg_color="#0F172A", corner_radius=12, border_width=1, border_color="#1E293B")
        status_frame.pack(fill="x", side="bottom", padx=15, pady=15)
        
        ctk.CTkLabel(
            status_frame, 
            text="THÔNG SỐ HỆ THỐNG", 
            text_color="#64748B",
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(anchor="w", padx=12, pady=(10, 4))
        
        cam_status_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        cam_status_frame.pack(fill="x", padx=12, pady=2)
        self.led_cam = ctk.CTkLabel(cam_status_frame, text="●", text_color="#EF4444", font=ctk.CTkFont(size=16))
        self.led_cam.pack(side="left")
        self.lbl_cam_status = ctk.CTkLabel(cam_status_frame, text="Camera: Ngoại tuyến", font=ctk.CTkFont(size=11), text_color="#94A3B8")
        self.lbl_cam_status.pack(side="left", padx=6)
        
        uptime_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        uptime_frame.pack(fill="x", padx=12, pady=(2, 10))
        self.led_uptime = ctk.CTkLabel(uptime_frame, text="⏱️", font=ctk.CTkFont(size=12))
        self.led_uptime.pack(side="left")
        self.lbl_uptime_val = ctk.CTkLabel(uptime_frame, text="Thời gian: 00:00:00", font=ctk.CTkFont(size=11), text_color="#94A3B8")
        self.lbl_uptime_val.pack(side="left", padx=6)

    def update_role_ui(self):
        if self.current_user_role == "ADMIN":
            self.lbl_role_status.configure(text=f"🔑 ADMIN ({self.current_username})", text_color="#10B981")
            self.btn_tab_history.configure(text="📋  LỊCH SỬ CẢNH BÁO", text_color="#F8FAFC")
            self.btn_tab_admin.configure(text="⚙️  ADMIN CẤU HÌNH", text_color="#F8FAFC")
        else:
            self.lbl_role_status.configure(text=f"👤 USER ({self.current_username})", text_color="#38BDF8")
            self.btn_tab_history.configure(text="📋  LỊCH SỬ CẢNH BÁO 🔒", text_color="#94A3B8")
            self.btn_tab_admin.configure(text="⚙️  ADMIN CẤU HÌNH 🔒", text_color="#94A3B8")

    # -------------------------------------------------------------
    # 2. TAB 1: USER VIEW (DASHBOARD CÓ CHO USER CHỌN CAMERA)
    # -------------------------------------------------------------
    def create_user_view(self):
        """Creates the User Monitoring Dashboard View."""
        self.user_view = ctk.CTkFrame(self, fg_color="transparent")
        
        self.user_view.grid_columnconfigure(0, weight=3)
        self.user_view.grid_columnconfigure(1, weight=1, minsize=340)
        self.user_view.grid_rowconfigure(0, weight=1)
        
        left_box = ctk.CTkFrame(self.user_view, fg_color="transparent")
        left_box.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        left_box.grid_rowconfigure(0, weight=0)
        left_box.grid_rowconfigure(1, weight=0)
        left_box.grid_rowconfigure(2, weight=10)
        left_box.grid_rowconfigure(3, weight=4)
        left_box.grid_columnconfigure(0, weight=1)
        
        # 1. KPI Stats Metrics Row
        kpi_row = ctk.CTkFrame(left_box, fg_color="transparent")
        kpi_row.grid(row=0, column=0, padx=0, pady=(0, 10), sticky="ew")
        kpi_row.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        card1 = ctk.CTkFrame(kpi_row, fg_color="#1E293B", corner_radius=12, border_width=1, border_color="#334155")
        card1.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkLabel(card1, text="TRẠNG THÁI GIÁM SÁT", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B").pack(anchor="w", padx=12, pady=(10, 0))
        self.kpi_status_val = ctk.CTkLabel(card1, text="AN TOÀN", font=ctk.CTkFont(size=15, weight="bold"), text_color="#10B981")
        self.kpi_status_val.pack(anchor="w", padx=12, pady=(0, 10))
        
        card2 = ctk.CTkFrame(kpi_row, fg_color="#1E293B", corner_radius=12, border_width=1, border_color="#334155")
        card2.grid(row=0, column=1, padx=3, sticky="ew")
        ctk.CTkLabel(card2, text="ĐỘ TIN CẬY AI KHUNG HÌNH", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B").pack(anchor="w", padx=12, pady=(10, 0))
        self.kpi_conf_val = ctk.CTkLabel(card2, text="0.00%", font=ctk.CTkFont(size=15, weight="bold"), text_color="#38BDF8")
        self.kpi_conf_val.pack(anchor="w", padx=12, pady=(0, 10))
        
        card3 = ctk.CTkFrame(kpi_row, fg_color="#1E293B", corner_radius=12, border_width=1, border_color="#334155")
        card3.grid(row=0, column=2, padx=3, sticky="ew")
        ctk.CTkLabel(card3, text="SỐ CẢNH BÁO TỔNG CỘNG", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B").pack(anchor="w", padx=12, pady=(10, 0))
        self.kpi_alerts_val = ctk.CTkLabel(card3, text="0 VỤ", font=ctk.CTkFont(size=15, weight="bold"), text_color="#F59E0B")
        self.kpi_alerts_val.pack(anchor="w", padx=12, pady=(0, 10))
        
        card4 = ctk.CTkFrame(kpi_row, fg_color="#1E293B", corner_radius=12, border_width=1, border_color="#334155")
        card4.grid(row=0, column=3, padx=(6, 0), sticky="ew")
        ctk.CTkLabel(card4, text="MÔ HÌNH NHẬN DIỆN", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B").pack(anchor="w", padx=12, pady=(10, 0))
        self.kpi_model_val = ctk.CTkLabel(card4, text="YOLOv8 Engine", font=ctk.CTkFont(size=15, weight="bold"), text_color="#A855F7")
        self.kpi_model_val.pack(anchor="w", padx=12, pady=(0, 10))
        
        # 2. Status Banner
        self.banner = ctk.CTkFrame(left_box, height=48, corner_radius=12, fg_color="#1E293B", border_width=1, border_color="#334155")
        self.banner.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        
        self.banner_container = ctk.CTkFrame(self.banner, fg_color="transparent")
        self.banner_container.pack(expand=True, pady=8)
        
        self.status_led = ctk.CTkLabel(self.banner_container, text="●", text_color="#64748B", font=ctk.CTkFont(size=24))
        self.status_led.pack(side="left", padx=(0, 10))
        
        self.banner_label = ctk.CTkLabel(
            self.banner_container, 
            text="HỆ THỐNG GIÁM SÁT ĐANG TẮT", 
            text_color="#F8FAFC",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.banner_label.pack(side="left")
        
        # 3. Video Container Screen
        self.video_container = ctk.CTkFrame(left_box, fg_color="#0F172A", corner_radius=14, border_width=1, border_color="#1E293B")
        self.video_container.grid(row=2, column=0, padx=0, pady=4, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_container, text="Không có tín hiệu camera (Nhấn ▶ BẮT ĐẦU từ bảng điều khiển)", font=ctk.CTkFont(size=13), text_color="#64748B")
        self.video_label.pack(expand=True, fill="both")
        
        # 4. Live Matplotlib Chart
        self.chart_container = ctk.CTkFrame(left_box, fg_color="#1E293B", corner_radius=14, border_width=1, border_color="#334155")
        self.chart_container.grid(row=3, column=0, padx=0, pady=(10, 0), sticky="nsew")
        
        self.fig = Figure(figsize=(6, 2.0), dpi=100, facecolor='#1E293B')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0F172A')
        self.ax.spines['bottom'].set_color('#334155')
        self.ax.spines['top'].set_color('#1E293B')
        self.ax.spines['left'].set_color('#334155')
        self.ax.spines['right'].set_color('#1E293B')
        self.ax.tick_params(colors='#94A3B8', labelsize=8)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xlim(0, 50)
        self.ax.set_ylabel("Độ tin cậy AI", color="#94A3B8", fontsize=8)
        self.ax.set_title("Biểu đồ nhận diện Lửa/Khói theo thời gian thực", color="#F8FAFC", fontsize=9, pad=5)
        
        self.plot_line, = self.ax.plot(self.conf_history, color='#00F2FE', linewidth=2)
        self.threshold_line = self.ax.axhline(y=0.4, color='#EF4444', linestyle='--', linewidth=1.5)
        self.ax.grid(True, color='#1E293B', linestyle=':', alpha=0.6)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=5)
        
        # Right Panel
        right_box = ctk.CTkFrame(self.user_view, fg_color="transparent")
        right_box.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        right_box.grid_rowconfigure(0, weight=0)
        right_box.grid_rowconfigure(1, weight=0)
        right_box.grid_rowconfigure(2, weight=1)
        right_box.grid_columnconfigure(0, weight=1)
        
        # Quick Controls Card (CHO NGUỜI DÙNG CHỌN CAMERA TRỰC TIẾP)
        ctrl_card = ctk.CTkFrame(right_box, corner_radius=14, fg_color="#1E293B", border_width=1, border_color="#334155")
        ctrl_card.grid(row=0, column=0, padx=0, pady=(0, 10), sticky="ew")
        
        ctk.CTkLabel(ctrl_card, text="🎮 BẢNG ĐIỀU KHIỂN & CHỌN CAMERA", font=ctk.CTkFont(size=13, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=15, pady=(12, 4))
        
        # Camera Source selector directly for User
        ctk.CTkLabel(ctrl_card, text="📹 Chọn Nguồn Camera Giám Sát:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#CBD5E1").pack(anchor="w", padx=15, pady=(2, 2))
        
        self.user_source_var = ctk.StringVar(value="Webcam 0")
        self.user_source_menu = ctk.CTkOptionMenu(
            ctrl_card, 
            values=["Webcam 0", "Webcam 1", "File Video", "IP Camera RTSP"],
            variable=self.user_source_var,
            height=34,
            corner_radius=8,
            fg_color="#0284C7",
            command=self.on_user_source_change
        )
        self.user_source_menu.pack(fill="x", padx=15, pady=(0, 4))
        
        self.user_source_input_frame = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        self.user_source_input_frame.pack(fill="x", padx=15, pady=2)
        
        self.user_source_entry_label = ctk.CTkLabel(self.user_source_input_frame, text="Đường dẫn file / RTSP:")
        self.user_source_entry = ctk.CTkEntry(self.user_source_input_frame, placeholder_text="Nhập file hoặc link RTSP...", height=32, corner_radius=6)
        self.btn_user_browse = ctk.CTkButton(self.user_source_input_frame, text="Chọn File", width=70, height=32, corner_radius=6, command=self.browse_user_video_file)
        
        btn_user_action = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        btn_user_action.pack(fill="x", padx=15, pady=(6, 12))
        
        self.btn_user_start = ctk.CTkButton(
            btn_user_action, 
            text="▶  BẮT ĐẦU GIÁM SÁT", 
            fg_color="#10B981", 
            hover_color="#059669",
            font=ctk.CTkFont(weight="bold"),
            height=40,
            corner_radius=10,
            command=self.start_monitoring
        )
        self.btn_user_start.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self.btn_user_stop = ctk.CTkButton(
            btn_user_action, 
            text="■  DỪNG GIÁM SÁT", 
            fg_color="#EF4444", 
            hover_color="#DC2626",
            font=ctk.CTkFont(weight="bold"),
            height=40,
            corner_radius=10,
            state="disabled",
            command=self.stop_monitoring
        )
        self.btn_user_stop.pack(side="right", fill="x", expand=True, padx=(4, 0))
        
        # Latest Snapshot Card
        self.history_frame = ctk.CTkFrame(right_box, corner_radius=14, fg_color="#1E293B", border_width=1, border_color="#334155")
        self.history_frame.grid(row=1, column=0, padx=0, pady=5, sticky="ew")
        
        ctk.CTkLabel(
            self.history_frame, 
            text="🚨 CẢNH BÁO MỚI NHẤT (NỘI BỘ)", 
            text_color="#F43F5E",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(12, 5))
        
        self.thumb_container = ctk.CTkFrame(self.history_frame, width=260, height=155, fg_color="#0F172A", corner_radius=10)
        self.thumb_container.pack(padx=15, pady=5)
        self.thumb_container.pack_propagate(False)
        
        self.thumb_label = ctk.CTkLabel(self.thumb_container, text="Chưa phát hiện sự cố cháy", text_color="#64748B", font=ctk.CTkFont(size=12))
        self.thumb_label.pack(expand=True, fill="both")
        
        self.lbl_alert_time = ctk.CTkLabel(self.history_frame, text="Thời gian: --:--:--", font=ctk.CTkFont(size=12), text_color="#CBD5E1")
        self.lbl_alert_time.pack(anchor="w", padx=15, pady=1)
        
        self.lbl_alert_conf = ctk.CTkLabel(self.history_frame, text="Độ tin cậy: --%", font=ctk.CTkFont(size=12), text_color="#CBD5E1")
        self.lbl_alert_conf.pack(anchor="w", padx=15, pady=(1, 12))
        
        # Live System Text Logs
        self.log_frame = ctk.CTkFrame(right_box, corner_radius=14, fg_color="#1E293B", border_width=1, border_color="#334155")
        self.log_frame.grid(row=2, column=0, padx=0, pady=(10, 0), sticky="nsew")
        self.log_frame.grid_rowconfigure(0, weight=0)
        self.log_frame.grid_rowconfigure(1, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            self.log_frame, 
            text="📋 NHẬT KÝ THÔNG BÁO TỨC THỜI", 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#38BDF8"
        ).grid(row=0, column=0, padx=15, pady=(12, 4), sticky="w")
        
        self.log_text = ctk.CTkTextbox(self.log_frame, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#0F172A", text_color="#CBD5E1")
        self.log_text.grid(row=1, column=0, padx=12, pady=(4, 12), sticky="nsew")
        self.log_text.configure(state="disabled")

    def on_user_source_change(self, value):
        """Dynamic camera input field handler for User screen."""
        self.user_source_entry.pack_forget()
        self.btn_user_browse.pack_forget()
        
        if value == "File Video":
            self.btn_user_browse.pack(side="right", padx=(4, 0))
            self.user_source_entry.pack(fill="x", side="left", expand=True)
        elif value == "IP Camera RTSP":
            self.user_source_entry.pack(fill="x")

    def browse_user_video_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.user_source_entry.delete(0, tk.END)
            self.user_source_entry.insert(0, filepath)

    # -------------------------------------------------------------
    # 3. TAB 2: ALERT HISTORY DATABASE VIEW (CẦN QUYỀN ADMIN)
    # -------------------------------------------------------------
    def create_history_view(self):
        """Creates the Alert History Database view."""
        self.history_view = ctk.CTkFrame(self, corner_radius=16, fg_color="#1E293B", border_width=1, border_color="#334155")
        self.history_view.grid_rowconfigure(0, weight=0)
        self.history_view.grid_rowconfigure(1, weight=0)
        self.history_view.grid_rowconfigure(2, weight=1)
        self.history_view.grid_columnconfigure(0, weight=1)
        
        # Header
        ctk.CTkLabel(
            self.history_view, 
            text="📋 CƠ SỞ DỮ LIỆU LỊCH SỬ CẢNH BÁO CHÁY (QUẢN TRỊ)", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#38BDF8"
        ).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        toolbar = ctk.CTkFrame(self.history_view, fg_color="transparent")
        toolbar.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        
        self.search_entry = ctk.CTkEntry(toolbar, placeholder_text="Tìm kiếm theo từ khóa / mốc thời gian...", width=320, height=36, corner_radius=8)
        self.search_entry.pack(side="left", padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self.refresh_history_table())
        
        btn_search = ctk.CTkButton(toolbar, text="🔍 Tìm Kiếm", width=100, height=36, corner_radius=8, fg_color="#0284C7", hover_color="#0369A1", command=self.refresh_history_table)
        btn_search.pack(side="left", padx=(0, 10))
        
        btn_refresh = ctk.CTkButton(toolbar, text="🔄 Làm Mới", width=100, height=36, corner_radius=8, fg_color="#475569", hover_color="#334155", command=self.refresh_history_table)
        btn_refresh.pack(side="left", padx=(0, 10))
        
        btn_export = ctk.CTkButton(toolbar, text="📥 Xuất Báo Cáo CSV", height=36, corner_radius=8, fg_color="#10B981", hover_color="#059669", command=self.export_history_csv)
        btn_export.pack(side="right", padx=(10, 0))
        
        btn_clear = ctk.CTkButton(toolbar, text="🗑️ Xóa Lịch Sử", height=36, corner_radius=8, fg_color="#EF4444", hover_color="#DC2626", command=self.clear_history_db)
        btn_clear.pack(side="right", padx=0)
        
        content_split = ctk.CTkFrame(self.history_view, fg_color="transparent")
        content_split.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="nsew")
        content_split.grid_columnconfigure(0, weight=3)
        content_split.grid_columnconfigure(1, weight=1, minsize=320)
        content_split.grid_rowconfigure(0, weight=1)
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", 
                        background="#0F172A", 
                        foreground="#F8FAFC", 
                        fieldbackground="#0F172A", 
                        rowheight=30,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading", 
                        background="#1E293B", 
                        foreground="#38BDF8", 
                        font=("Segoe UI", 11, "bold"))
        style.map("Treeview", background=[("selected", "#EA580C")])
        
        tree_frame = ctk.CTkFrame(content_split, fg_color="#0F172A", corner_radius=12, border_width=1, border_color="#334155")
        tree_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        self.tree = ttk.Treeview(
            tree_frame, 
            columns=("id", "timestamp", "label", "conf", "action", "image"), 
            show="headings",
            selectmode="browse"
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("timestamp", text="Thời Gian")
        self.tree.heading("label", text="Loại Cảnh Báo")
        self.tree.heading("conf", text="Độ Tin Cậy")
        self.tree.heading("action", text="Kết Quả Xử Lý Ca Trực")
        self.tree.heading("image", text="Đường Dẫn Ảnh Snapshot")
        
        self.tree.column("id", width=45, anchor="center")
        self.tree.column("timestamp", width=140, anchor="center")
        self.tree.column("label", width=100, anchor="center")
        self.tree.column("conf", width=90, anchor="center")
        self.tree.column("action", width=160, anchor="center")
        self.tree.column("image", width=200, anchor="w")
        
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        preview_box = ctk.CTkFrame(content_split, corner_radius=12, fg_color="#0F172A", border_width=1, border_color="#334155")
        preview_box.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        
        ctk.CTkLabel(preview_box, text="📸 XEM ẢNH NỘI BỘ", font=ctk.CTkFont(size=14, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=15, pady=(15, 10))
        
        self.preview_img_container = ctk.CTkFrame(preview_box, width=280, height=180, fg_color="#1E293B")
        self.preview_img_container.pack(padx=15, pady=5)
        self.preview_img_container.pack_propagate(False)
        
        self.preview_img_label = ctk.CTkLabel(self.preview_img_container, text="Chọn bản ghi để xem ảnh", text_color="#64748B")
        self.preview_img_label.pack(expand=True, fill="both")
        
        self.lbl_preview_info = ctk.CTkLabel(preview_box, text="Thông tin chi tiết...", font=ctk.CTkFont(size=12), text_color="#CBD5E1", justify="left")
        self.lbl_preview_info.pack(anchor="w", padx=15, pady=10)
        
        btn_del_single = ctk.CTkButton(
            preview_box, 
            text="🗑️ Xóa Bản Ghi Này", 
            fg_color="#EF4444", 
            hover_color="#DC2626",
            height=36,
            corner_radius=8,
            command=self.delete_selected_history_record
        )
        btn_del_single.pack(fill="x", padx=15, side="bottom", pady=15)

    # -------------------------------------------------------------
    # 4. TAB 3: ADMIN MANAGEMENT VIEW (CẦN QUYỀN ADMIN)
    # -------------------------------------------------------------
    def create_admin_view(self):
        """Creates the Admin Configuration and Control View."""
        self.admin_view = ctk.CTkFrame(self, corner_radius=16, fg_color="#1E293B", border_width=1, border_color="#334155")
        self.admin_view.grid_rowconfigure(0, weight=0)
        self.admin_view.grid_rowconfigure(1, weight=1)
        self.admin_view.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            self.admin_view, 
            text="⚙️ ADMIN - CẤU HÌNH & QUẢN TRỊ HỆ THỐNG", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#38BDF8"
        ).grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        form_frame = ctk.CTkFrame(self.admin_view, fg_color="transparent")
        form_frame.grid(row=1, column=0, padx=40, pady=10, sticky="nsew")
        
        # 1. Camera Source Config
        ctk.CTkLabel(form_frame, text="1. Nguồn Camera Đầu Vào Mặc Định:", font=ctk.CTkFont(weight="bold", size=14), text_color="#F8FAFC").pack(anchor="w", pady=(10, 5))
        self.source_var = ctk.StringVar(value="Webcam 0")
        self.source_menu = ctk.CTkOptionMenu(
            form_frame, 
            values=["Webcam 0", "Webcam 1", "File Video", "IP Camera RTSP"],
            variable=self.source_var,
            height=36,
            corner_radius=8,
            fg_color="#0284C7",
            command=self.on_source_change
        )
        self.source_menu.pack(fill="x", pady=5)
        
        self.source_input_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        self.source_input_frame.pack(fill="x", pady=5)
        
        self.source_entry_label = ctk.CTkLabel(self.source_input_frame, text="Đường dẫn / URL RTSP:")
        self.source_entry = ctk.CTkEntry(self.source_input_frame, placeholder_text="Nhập đường dẫn file hoặc link RTSP...", height=36, corner_radius=8)
        self.btn_browse = ctk.CTkButton(self.source_input_frame, text="Chọn File Video", height=36, corner_radius=8, command=self.browse_video_file)
        
        # 2. AI Hyperparameter Sliders
        ctk.CTkLabel(form_frame, text="2. Tinh Chỉnh Tham Số AI Nhận Diện:", font=ctk.CTkFont(weight="bold", size=14), text_color="#F8FAFC").pack(anchor="w", pady=(20, 5))
        
        slider_row = ctk.CTkFrame(form_frame, fg_color="transparent")
        slider_row.pack(fill="x", pady=5)
        ctk.CTkLabel(slider_row, text="Ngưỡng độ tin cậy (Conf threshold):").pack(side="left")
        self.conf_slider_val = ctk.CTkLabel(slider_row, text="40%", font=ctk.CTkFont(weight="bold"), text_color="#38BDF8")
        self.conf_slider_val.pack(side="right")
        
        self.conf_slider = ctk.CTkSlider(form_frame, from_=0.1, to=1.0, number_of_steps=18, command=self.on_conf_slider_change)
        self.conf_slider.set(0.4)
        self.conf_slider.pack(fill="x", pady=2)
        
        sens_row = ctk.CTkFrame(form_frame, fg_color="transparent")
        sens_row.pack(fill="x", pady=5)
        ctk.CTkLabel(sens_row, text="Độ trễ chống báo giả (Số frames xuất hiện liên tục):").pack(side="left")
        self.sensitivity_val = ctk.CTkLabel(sens_row, text="10 frames", font=ctk.CTkFont(weight="bold"), text_color="#38BDF8")
        self.sensitivity_val.pack(side="right")
        
        self.sensitivity_slider = ctk.CTkSlider(form_frame, from_=2, to=30, number_of_steps=28, command=self.on_sensitivity_slider_change)
        self.sensitivity_slider.set(10)
        self.sensitivity_slider.pack(fill="x", pady=2)
        
        # 3. Audio & In-App Notification Settings
        ctk.CTkLabel(form_frame, text="3. Cấu Hình Cảnh Báo Nội Bộ & Còi Hú:", font=ctk.CTkFont(weight="bold", size=14), text_color="#F8FAFC").pack(anchor="w", pady=(20, 5))
        
        self.mute_var = ctk.BooleanVar(value=False)
        self.mute_check = ctk.CTkCheckBox(
            form_frame, 
            text="Tắt âm thanh báo động còi hú", 
            variable=self.mute_var,
            command=self.on_mute_change
        )
        self.mute_check.pack(anchor="w", pady=5)
        
        # 4. Action Buttons (Start/Stop)
        ctk.CTkLabel(form_frame, text="4. Điều Khiển Hệ Thống:", font=ctk.CTkFont(weight="bold", size=14), text_color="#F8FAFC").pack(anchor="w", pady=(20, 5))
        
        btn_admin_action = ctk.CTkFrame(form_frame, fg_color="transparent")
        btn_admin_action.pack(fill="x", pady=10)
        
        self.btn_admin_start = ctk.CTkButton(
            btn_admin_action, 
            text="▶ BẮT ĐẦU GIÁM SÁT", 
            fg_color="#10B981", 
            hover_color="#059669",
            font=ctk.CTkFont(weight="bold", size=14),
            height=45,
            corner_radius=10,
            command=self.start_monitoring
        )
        self.btn_admin_start.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_admin_stop = ctk.CTkButton(
            btn_admin_action, 
            text="■ DỪNG HỆ THỐNG", 
            fg_color="#EF4444", 
            hover_color="#DC2626",
            font=ctk.CTkFont(weight="bold", size=14),
            height=45,
            corner_radius=10,
            state="disabled",
            command=self.stop_monitoring
        )
        self.btn_admin_stop.pack(side="right", fill="x", expand=True, padx=(10, 0))

    # -------------------------------------------------------------
    # NAVIGATION & TAB LOGIC
    # -------------------------------------------------------------
    def show_tab(self, tab_name):
        if tab_name in ["history", "admin"] and self.current_user_role != "ADMIN":
            messagebox.showwarning("Truy Cập Bị Khóa", f"Trang [{tab_name.upper()}] yêu cầu tài khoản ADMIN. Vui lòng đăng xuất và đăng nhập tài khoản Admin!")
            return
            
        self.current_tab = tab_name
        
        self.btn_tab_user.configure(fg_color="transparent")
        self.btn_tab_history.configure(fg_color="transparent")
        self.btn_tab_admin.configure(fg_color="transparent")
        
        self.user_view.grid_forget()
        self.history_view.grid_forget()
        self.admin_view.grid_forget()
        
        if tab_name == "user":
            self.btn_tab_user.configure(fg_color="#EA580C")
            self.user_view.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        elif tab_name == "history":
            self.btn_tab_history.configure(fg_color="#EA580C")
            self.history_view.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
            self.refresh_history_table()
        elif tab_name == "admin":
            self.btn_tab_admin.configure(fg_color="#EA580C")
            self.admin_view.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")

    def load_defaults(self):
        self.on_source_change("Webcam 0")
        self.on_user_source_change("Webcam 0")

    def toggle_mute(self):
        new_state = not self.mute_var.get()
        self.mute_var.set(new_state)
        self.on_mute_change()

    def on_mute_change(self):
        self.alert_manager.mute_alarm = self.mute_var.get()
        self.add_log(f"Đã {'tắt' if self.mute_var.get() else 'mở'} âm thanh còi hú.")
        if self.mute_var.get():
            self.btn_mute_emergency.configure(text="🔊  BẬT LẠI CÒI HÚ", fg_color="#10B981", hover_color="#059669")
        else:
            self.btn_mute_emergency.configure(text="🔕  TẮT CÒI KHẨN CẤP", fg_color="#DC2626", hover_color="#B91C1C")

    def on_source_change(self, value):
        self.source_entry_label.pack_forget()
        self.source_entry.pack_forget()
        self.btn_browse.pack_forget()
        
        if value == "File Video":
            self.source_entry_label.configure(text="Đường dẫn file video:")
            self.source_entry_label.pack(anchor="w", pady=2)
            self.btn_browse.pack(side="right", padx=(5, 0))
            self.source_entry.pack(fill="x", side="left", expand=True)
        elif value == "IP Camera RTSP":
            self.source_entry_label.configure(text="Nhập link RTSP của Camera:")
            self.source_entry_label.pack(anchor="w", pady=2)
            self.source_entry.pack(fill="x")

    def browse_video_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, filepath)

    def on_conf_slider_change(self, val):
        self.conf_slider_val.configure(text=f"{int(float(val)*100)}%")
        self.detector.conf_threshold = float(val)
        self.threshold_line.set_ydata([float(val), float(val)])
        self.canvas.draw_idle()

    def on_sensitivity_slider_change(self, val):
        self.sensitivity_val.configure(text=f"{int(float(val))} frames")
        self.detector.consecutive_frames_limit = int(float(val))

    def add_log(self, text):
        timestamp = time.strftime("[%H:%M:%S]")
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"{timestamp} {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def open_alerts_folder(self):
        alerts_dir = os.path.abspath("alerts")
        if not os.path.exists(alerts_dir):
            os.makedirs(alerts_dir)
        try:
            os.startfile(alerts_dir)
            self.add_log("Đã mở thư mục ảnh cảnh báo.")
        except Exception:
            import subprocess
            subprocess.Popen(f'explorer "{alerts_dir}"')

    # -------------------------------------------------------------
    # IN-APP POPUP NOTIFICATION
    # -------------------------------------------------------------
    def on_new_in_app_notification(self, notification_dict):
        self.today_alert_count += 1
        self.kpi_alerts_val.configure(text=f"{self.today_alert_count} VỤ")
        self.add_log(f"🔥 CẢNH BÁO MỚI (ID #{notification_dict['id']}): {notification_dict['label']} ({int(notification_dict['confidence']*100)}%)")
        self.show_in_app_popup(notification_dict)

    def show_in_app_popup(self, noti):
        try:
            if self.popup_window and self.popup_window.winfo_exists():
                self.popup_window.destroy()
                
            popup = ctk.CTkToplevel(self)
            self.popup_window = popup
            popup.title("🚨 CẢNH BÁO CHÁY TỨC THỜI (XÁC NHẬN NỘI BỘ)")
            popup.geometry("480x420")
            popup.configure(fg_color="#0F172A")
            popup.attributes("-topmost", True)
            popup.transient(self)
            
            card = ctk.CTkFrame(popup, fg_color="#881337", corner_radius=14, border_width=1, border_color="#F43F5E")
            card.pack(fill="both", expand=True, padx=12, pady=12)
            
            ctk.CTkLabel(card, text="🚨 KÍCH HOẠT CẢNH BÁO CHÁY! 🚨", text_color="#FFFFFF", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(12, 2))
            ctk.CTkLabel(card, text=f"Vị trí: Camera chính | Loại: {noti['label']} ({int(noti['confidence']*100)}%)", text_color="#FECDD3", font=ctk.CTkFont(size=12)).pack(pady=1)
            
            if noti['image_path'] and os.path.exists(noti['image_path']):
                try:
                    img = Image.open(noti['image_path'])
                    img.thumbnail((360, 130))
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                    img_lbl = ctk.CTkLabel(card, image=ctk_img, text="")
                    img_lbl.pack(pady=6)
                except Exception:
                    pass
            
            ctk.CTkLabel(card, text="✍️ CHỌN KẾT QUẢ XỬ LÝ CA TRỰC:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=15, pady=(4, 2))
            
            entry_note = ctk.CTkEntry(card, placeholder_text="Ghi chú thêm (VD: khói bếp, cháy nhỏ...)...", height=32, corner_radius=6)
            entry_note.pack(fill="x", padx=15, pady=(0, 8))
            
            btn_box = ctk.CTkFrame(card, fg_color="transparent")
            btn_box.pack(fill="x", padx=15, pady=(0, 12))
            
            def handle_action_submit(action_text):
                note = entry_note.get().strip()
                self.alert_manager.update_alert_action(noti["id"], action_text, note)
                self.alert_manager.stop_alarm()
                self.add_log(f"📝 Ca trực đã xác nhận vụ việc #{noti['id']}: [{action_text}] - {note}")
                popup.destroy()
                if hasattr(self, "tree"):
                    self.refresh_history_table()
            
            btn_false = ctk.CTkButton(
                btn_box, 
                text="🟡 Báo Động Giả", 
                fg_color="#D97706", 
                hover_color="#B45309",
                font=ctk.CTkFont(size=11, weight="bold"),
                height=34,
                corner_radius=6,
                command=lambda: handle_action_submit("Báo động giả (Khói nhẹ/Nhiễu)")
            )
            btn_false.pack(side="left", fill="x", expand=True, padx=(0, 3))
            
            btn_fixed = ctk.CTkButton(
                btn_box, 
                text="🟢 Đã Dập Tắt", 
                fg_color="#059669", 
                hover_color="#047857",
                font=ctk.CTkFont(size=11, weight="bold"),
                height=34,
                corner_radius=6,
                command=lambda: handle_action_submit("Đã dập tắt tại chỗ")
            )
            btn_fixed.pack(side="left", fill="x", expand=True, padx=3)
            
            btn_danger = ctk.CTkButton(
                btn_box, 
                text="🔴 Cháy Lớn / 114", 
                fg_color="#DC2626", 
                hover_color="#B91C1C",
                font=ctk.CTkFont(size=11, weight="bold"),
                height=34,
                corner_radius=6,
                command=lambda: handle_action_submit("Nghiêm trọng / Đã báo 114")
            )
            btn_danger.pack(side="right", fill="x", expand=True, padx=(3, 0))
            
        except Exception as e:
            print(f"Error showing popup: {e}")

    # -------------------------------------------------------------
    # HISTORY DATABASE ACTIONS
    # -------------------------------------------------------------
    def refresh_history_table(self):
        if not hasattr(self, "tree"):
            return
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        query = self.search_entry.get().strip() if hasattr(self, "search_entry") else ""
        alerts = self.alert_manager.get_all_alerts(limit=100, search_query=query)
        
        for a in alerts:
            self.tree.insert("", "end", iid=str(a["id"]), values=(
                a["id"],
                a["timestamp"],
                a["label"],
                f"{int(a['confidence']*100)}%",
                a.get("action_taken", "Chưa xử lý"),
                a["image_path"]
            ))

    def on_tree_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        item_id = selected[0]
        item = self.tree.item(item_id)
        vals = item["values"]
        
        img_path = vals[5] if len(vals) > 5 else vals[4]
        self.lbl_preview_info.configure(text=f"ID: #{vals[0]}\nThời gian: {vals[1]}\nLoại: {vals[2]} ({vals[3]})\nKết quả: {vals[4]}")
        
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img.thumbnail((260, 160))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self.preview_img_label.configure(image=ctk_img, text="")
                self.preview_img_label._image = ctk_img
            except Exception as e:
                self.preview_img_label.configure(image="", text="Không đọc được ảnh")
        else:
            self.preview_img_label.configure(image="", text="Không có file ảnh")

    def export_history_csv(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"Bao_Cao_Canh_Bao_{time.strftime('%Y%m%d')}.csv"
        )
        if filepath:
            success, msg = self.alert_manager.export_csv(filepath)
            if success:
                messagebox.showinfo("Thành Công", msg)
            else:
                messagebox.showerror("Thất Bại", msg)

    def delete_selected_history_record(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Thông Báo", "Vui lòng chọn bản ghi trong bảng để xóa.")
            return
        alert_id = int(selected[0])
        if messagebox.askyesno("Xác Nhận", f"Bạn có chắc muốn xóa bản ghi #{alert_id}?"):
            self.alert_manager.delete_alert(alert_id)
            self.refresh_history_table()

    def clear_history_db(self):
        if messagebox.askyesno("CẢNH BÁO", "Bạn có chắc chắn muốn xóa TOÀN BỘ lịch sử và ảnh cảnh báo không?"):
            self.alert_manager.clear_all_alerts()
            self.refresh_history_table()
            messagebox.showinfo("Thành Công", "Đã xóa toàn bộ lịch sử CSDL.")

    # -------------------------------------------------------------
    # DETECTOR START / STOP CONTROL
    # -------------------------------------------------------------
    def start_monitoring(self):
        # Prefer camera selected directly by User on screen if set
        if hasattr(self, "user_source_var"):
            src_option = self.user_source_var.get()
            if src_option == "Webcam 0":
                source = "0"
            elif src_option == "Webcam 1":
                source = "1"
            elif src_option in ["File Video", "IP Camera RTSP"]:
                source = self.user_source_entry.get().strip()
                if not source:
                    messagebox.showerror("Lỗi", "Vui lòng cung cấp đường dẫn video hoặc link RTSP.")
                    return
            else:
                source = "0"
        else:
            source = "0"
            
        self.btn_user_start.configure(state="disabled")
        self.btn_user_stop.configure(state="normal")
        self.btn_admin_start.configure(state="disabled")
        self.btn_admin_stop.configure(state="normal")
        
        self.detector.start(source, self.alert_manager)

    def stop_monitoring(self):
        self.btn_user_stop.configure(state="disabled")
        self.btn_admin_stop.configure(state="disabled")
        self.detector.stop()
        self.btn_user_start.configure(state="normal")
        self.btn_admin_start.configure(state="normal")

    def on_detector_status_change(self, status):
        if status == "Running":
            self.banner_label.configure(text="💚 ĐANG GIÁM SÁT - HOẠT ĐỘNG AN TOÀN")
            self.banner.configure(fg_color="#065F46", border_color="#10B981")
            self.status_led.configure(text_color="#34D399")
            self.led_cam.configure(text_color="#10B981")
            self.lbl_cam_status.configure(text="Camera: Đang kết nối", text_color="#34D399")
            self.kpi_status_val.configure(text="ĐANG GIÁM SÁT", text_color="#10B981")
        elif status == "Stopped":
            self.banner_label.configure(text="HỆ THỐNG GIÁM SÁT ĐANG TẮT")
            self.banner.configure(fg_color="#1E293B", border_color="#334155")
            self.status_led.configure(text_color="#64748B")
            self.video_label.configure(image="", text="Không có tín hiệu camera")
            self.led_cam.configure(text_color="#EF4444")
            self.lbl_cam_status.configure(text="Camera: Ngoại tuyến", text_color="#94A3B8")
            self.kpi_status_val.configure(text="TẮT", text_color="#94A3B8")
        elif "Error" in status:
            self.banner_label.configure(text="❌ LỖI KẾT NỐI CAMERA ❌")
            self.banner.configure(fg_color="#991B1B", border_color="#EF4444")
            self.status_led.configure(text_color="#F87171")
            self.led_cam.configure(text_color="#EF4444")
            self.lbl_cam_status.configure(text="Camera: Lỗi kết nối", text_color="#EF4444")
            self.kpi_status_val.configure(text="LỖI CAMERA", text_color="#EF4444")
            self.stop_monitoring()

    def on_alert_state_change(self, is_active):
        self.alert_active = is_active
        if not is_active:
            self.banner_label.configure(
                text="💚 ĐANG GIÁM SÁT - HOẠT ĐỘNG AN TOÀN" if self.detector.running else "HỆ THỐNG GIÁM SÁT ĐANG TẮT"
            )
            self.banner.configure(
                fg_color="#065F46" if self.detector.running else "#1E293B",
                border_color="#10B981" if self.detector.running else "#334155"
            )
            if self.detector.running:
                self.status_led.configure(text_color="#34D399")
                self.kpi_status_val.configure(text="AN TOÀN", text_color="#10B981")
            else:
                self.status_led.configure(text_color="#64748B")
                self.kpi_status_val.configure(text="TẮT", text_color="#94A3B8")
        else:
            self.kpi_status_val.configure(text="🚨 NGUY HIỂM", text_color="#EF4444")
            if self.detector.running and self.detector.latest_annotated_frame is not None:
                self.update_smart_history_thumbnail(self.detector.latest_annotated_frame, self.detector.latest_conf)

    def update_smart_history_thumbnail(self, frame, conf_val):
        try:
            h, w = frame.shape[:2]
            target_w = 260
            target_h = int((target_w / w) * h)
            if target_h > 155:
                target_h = 155
                target_w = int((target_h / h) * w)
                
            frame_resized = cv2.resize(frame, (target_w, target_h))
            cv2_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_rgb)
            
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            
            self.thumb_label.configure(image=ctk_img, text="")
            self.thumb_label._image = ctk_img
            
            self.lbl_alert_time.configure(text=f"Thời gian: {time.strftime('%H:%M:%S')}")
            self.lbl_alert_conf.configure(text=f"Độ tin cậy: {int(conf_val * 100)}%")
        except Exception as e:
            print(f"Error drawing thumbnail: {e}")

    def flash_red_banner(self):
        if self.alert_active and self.detector.running:
            self.is_flashing = not self.is_flashing
            if self.is_flashing:
                self.banner_label.configure(text="🚨 CẢNH BÁO NGUY HIỂM: PHÁT HIỆN CÓ CHÁY/KHÓI! 🚨")
                self.banner.configure(fg_color="#991B1B", border_color="#F43F5E") 
                self.status_led.configure(text_color="#FB7185") 
            else:
                self.banner_label.configure(text="🚨 CẢNH BÁO NGUY HIỂM: PHÁT HIỆN CÓ CHÁY/KHÓI! 🚨")
                self.banner.configure(fg_color="#1E293B", border_color="#991B1B") 
                self.status_led.configure(text_color="#1E293B") 
        self.after(500, self.flash_red_banner)

    def update_video_loop(self):
        if self.detector.running and self.current_tab == "user" and self.current_user_role is not None:
            frame = self.detector.latest_annotated_frame
            if frame is not None:
                try:
                    h, w = frame.shape[:2]
                    target_w = 680
                    target_h = int((target_w / w) * h)
                    
                    frame_resized = cv2.resize(frame, (target_w, target_h))
                    cv2_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cv2_rgb)
                    
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
                    self.video_label.configure(image=ctk_img, text="")
                    self.video_label._image = ctk_img
                except Exception as e:
                    print(f"Error drawing video frame: {e}")
                    
        self.after(33, self.update_video_loop)

    def update_chart_loop(self):
        if self.detector.running and self.current_tab == "user" and self.current_user_role is not None:
            current_conf = self.detector.latest_conf
            self.kpi_conf_val.configure(text=f"{int(current_conf * 100)}%")
            self.conf_history.append(current_conf)
            self.conf_history.pop(0)
            
            try:
                self.plot_line.set_ydata(self.conf_history)
                self.canvas.draw_idle()
            except Exception as e:
                print(f"Error updating chart: {e}")
                
        self.after(100, self.update_chart_loop)

    def update_uptime_loop(self):
        elapsed = int(time.time() - self.start_timestamp)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        self.lbl_uptime_val.configure(text=f"Thời gian: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.after(1000, self.update_uptime_loop)

    def on_closing(self):
        if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn thoát ứng dụng không?"):
            self.detector.stop()
            self.alert_manager.stop_alarm()
            self.destroy()

if __name__ == "__main__":
    app = FireAlarmApp()
    app.after(500, app.flash_red_banner)
    app.mainloop()
