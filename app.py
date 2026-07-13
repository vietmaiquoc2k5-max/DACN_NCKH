import os
import cv2
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox
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
        
        # Configure window
        self.title("🔥 AI Fire Sentinel - Dashboard Giám Sát Cảnh Báo Cháy Chuyên Nghiệp")
        self.geometry("1200x700")
        self.minsize(1100, 650)
        
        # Initialize Core Modules
        self.alert_manager = AlertManager()
        self.detector = FireDetector()
        
        # Connect callbacks from Detector to GUI
        self.detector.log_callback = self.add_log
        self.detector.status_callback = self.on_detector_status_change
        self.detector.alert_callback = self.on_alert_state_change
        
        # GUI state
        self.is_flashing = False
        self.alert_active = False
        self.current_tab = "monitor"
        
        # Matplotlib data
        self.conf_history = [0.0] * 50
        
        # Hook window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set up grid layout (1 row, 3 columns)
        self.grid_columnconfigure(0, weight=0, minsize=240)  # Left panel (Sidebar)
        self.grid_columnconfigure(1, weight=3)                # Center panel (Main View)
        self.grid_columnconfigure(2, weight=1, minsize=300)  # Right panel (Details & Controls)
        self.grid_rowconfigure(0, weight=1)
        
        # Create UI components
        self.create_sidebar()
        self.create_main_panel()    # Center column
        self.create_right_panel()   # Right column
        self.create_settings_panel() # Tab Settings (hidden initially)
        
        # Set default tab
        self.show_tab("monitor")
        
        # Load saved settings
        self.load_defaults()
        
        # Start GUI polling loops
        self.update_video_loop()
        self.update_chart_loop()
        
        self.add_log("Dashboard giám sát đã khởi động thành công!")

    def create_sidebar(self):
        """Creates the left sidebar menu."""
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=15)
        self.sidebar.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Logo / Title
        logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="🔥 FIRE SENTINEL AI", 
            font=ctk.CTkFont(size=18, weight="bold", family="Helvetica")
        )
        logo_label.pack(padx=10, pady=(25, 20))
        
        # Tab Buttons
        self.btn_tab_monitor = ctk.CTkButton(
            self.sidebar, 
            text="📺 GIÁM SÁT TRỰC TIẾP", 
            font=ctk.CTkFont(weight="bold"),
            fg_color="transparent",
            text_color="#FFFFFF",
            hover_color="#1F2530",
            height=40,
            command=lambda: self.show_tab("monitor")
        )
        self.btn_tab_monitor.pack(fill="x", padx=15, pady=5)
        
        self.btn_tab_settings = ctk.CTkButton(
            self.sidebar, 
            text="⚙️ CẤU HÌNH HỆ THỐNG", 
            font=ctk.CTkFont(weight="bold"),
            fg_color="transparent",
            text_color="#FFFFFF",
            hover_color="#1F2530",
            height=40,
            command=lambda: self.show_tab("settings")
        )
        self.btn_tab_settings.pack(fill="x", padx=15, pady=5)
        
        # Open local alerts folder
        self.btn_open_alerts = ctk.CTkButton(
            self.sidebar, 
            text="📁 THƯ MỤC CẢNH BÁO", 
            fg_color="#D35400", 
            hover_color="#A04000",
            font=ctk.CTkFont(weight="bold", size=12),
            height=35,
            command=self.open_alerts_folder
        )
        self.btn_open_alerts.pack(fill="x", padx=15, pady=(20, 5))
        
        # System status indicator panel at the bottom
        status_frame = ctk.CTkFrame(self.sidebar, fg_color="#1A202C", corner_radius=10)
        status_frame.pack(fill="x", side="bottom", padx=15, pady=20)
        
        ctk.CTkLabel(
            status_frame, 
            text="TRẠNG THÁI HỆ THỐNG", 
            text_color="#718096",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Camera Connection status
        cam_status_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        cam_status_frame.pack(fill="x", padx=10, pady=3)
        self.led_cam = ctk.CTkLabel(cam_status_frame, text="●", text_color="#E74C3C", font=ctk.CTkFont(size=20))
        self.led_cam.pack(side="left")
        self.lbl_cam_status = ctk.CTkLabel(cam_status_frame, text="Camera: Off", font=ctk.CTkFont(size=12))
        self.lbl_cam_status.pack(side="left", padx=5)
        
        # Telegram Bot configuration status
        tele_status_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        tele_status_frame.pack(fill="x", padx=10, pady=(3, 10))
        self.led_tele = ctk.CTkLabel(tele_status_frame, text="●", text_color="#F1C40F", font=ctk.CTkFont(size=20))
        self.led_tele.pack(side="left")
        self.lbl_tele_status = ctk.CTkLabel(tele_status_frame, text="Telegram: Chưa có", font=ctk.CTkFont(size=12))
        self.lbl_tele_status.pack(side="left", padx=5)

    def create_main_panel(self):
        """Creates the center column containing video stream and Matplotlib chart."""
        self.main_panel = ctk.CTkFrame(self, fg_color="transparent")
        # Gridded dynamically in show_tab()
        self.main_panel.grid_rowconfigure(0, weight=0)  # Status banner
        self.main_panel.grid_rowconfigure(1, weight=10) # Video feed
        self.main_panel.grid_rowconfigure(2, weight=4)  # Matplotlib Chart
        self.main_panel.grid_columnconfigure(0, weight=1)
        
        # 1. Alert Status Banner with LED Indicator
        self.banner = ctk.CTkFrame(self.main_panel, height=50, corner_radius=10, fg_color="#1E2530")
        self.banner.grid(row=0, column=0, padx=5, pady=(0, 10), sticky="ew")
        
        self.banner_container = ctk.CTkFrame(self.banner, fg_color="transparent")
        self.banner_container.pack(expand=True, pady=10)
        
        self.status_led = ctk.CTkLabel(
            self.banner_container, 
            text="●", 
            text_color="#95A5A6", 
            font=ctk.CTkFont(size=26)
        )
        self.status_led.pack(side="left", padx=(0, 10))
        
        self.banner_label = ctk.CTkLabel(
            self.banner_container, 
            text="HỆ THỐNG ĐANG TẮT", 
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.banner_label.pack(side="left")
        
        # 2. Video Feed Frame
        self.video_container = ctk.CTkFrame(self.main_panel, fg_color="#111625", corner_radius=10)
        self.video_container.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_container, text="Không có tín hiệu camera", font=ctk.CTkFont(size=14))
        self.video_label.pack(expand=True, fill="both")
        
        # 3. Live Data Chart (Matplotlib)
        self.chart_container = ctk.CTkFrame(self.main_panel, fg_color="#1E2530", corner_radius=10)
        self.chart_container.grid(row=2, column=0, padx=5, pady=(10, 0), sticky="nsew")
        
        # Create matplotlib figure with dark background
        self.fig = Figure(figsize=(6, 2.0), dpi=100, facecolor='#1E2530')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1E2530')
        self.ax.spines['bottom'].set_color('#7F8C8D')
        self.ax.spines['top'].set_color('#1E2530')
        self.ax.spines['left'].set_color('#7F8C8D')
        self.ax.spines['right'].set_color('#1E2530')
        self.ax.tick_params(colors='#BDC3C7', labelsize=8)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xlim(0, 50)
        self.ax.set_ylabel("Độ tin cậy", color="#BDC3C7", fontsize=8)
        self.ax.set_title("Độ biến thiên nhận diện Lửa/Khói", color="#FFFFFF", fontsize=9, pad=5)
        
        # Set up plot lines
        self.plot_line, = self.ax.plot(self.conf_history, color='#FF5722', linewidth=2)
        self.threshold_line = self.ax.axhline(y=0.4, color='#E74C3C', linestyle='--', linewidth=1.5)
        
        # Grid lines
        self.ax.grid(True, color='#2C3E50', linestyle=':', alpha=0.5)
        self.fig.tight_layout()
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=5)

    def create_right_panel(self):
        """Creates the right-side control and log details panel."""
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        # Gridded dynamically in show_tab()
        self.right_panel.grid_rowconfigure(0, weight=0)  # Smart History
        self.right_panel.grid_rowconfigure(1, weight=0)  # Custom threshold controls
        self.right_panel.grid_rowconfigure(2, weight=1)  # Summary logs
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # 1. Smart History Panel (Alert Snapshots)
        self.history_frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        self.history_frame.grid(row=0, column=0, padx=5, pady=(0, 10), sticky="ew")
        
        ctk.CTkLabel(
            self.history_frame, 
            text="🚨 CẢNH BÁO MỚI NHẤT", 
            text_color="#E74C3C",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        # Thumbnail container
        self.thumb_container = ctk.CTkFrame(self.history_frame, width=220, height=140, fg_color="#1A202C")
        self.thumb_container.pack(padx=15, pady=5)
        self.thumb_container.pack_propagate(False)
        
        self.thumb_label = ctk.CTkLabel(self.thumb_container, text="Chưa có ảnh chụp", text_color="#4A5568", font=ctk.CTkFont(size=12))
        self.thumb_label.pack(expand=True, fill="both")
        
        # Alert Metadata
        self.lbl_alert_time = ctk.CTkLabel(self.history_frame, text="Thời gian: --:--:--", font=ctk.CTkFont(size=12))
        self.lbl_alert_time.pack(anchor="w", padx=15, pady=1)
        
        self.lbl_alert_conf = ctk.CTkLabel(self.history_frame, text="Độ tin cậy: --%", font=ctk.CTkFont(size=12))
        self.lbl_alert_conf.pack(anchor="w", padx=15, pady=(1, 10))
        
        # 2. Configurations & Sliders
        self.controls_frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        self.controls_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(
            self.controls_frame, 
            text="🎛️ ĐIỀU CHỈNH NGƯỠNG", 
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=15, pady=(10, 2))
        
        # Threshold Slider
        self.slider_label = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.slider_label.pack(fill="x", padx=15, pady=(5, 0))
        ctk.CTkLabel(self.slider_label, text="Độ tin cậy (Conf):", font=ctk.CTkFont(size=11)).pack(side="left")
        self.conf_slider_val = ctk.CTkLabel(self.slider_label, text="40%", font=ctk.CTkFont(size=11, weight="bold"))
        self.conf_slider_val.pack(side="right")
        
        self.conf_slider = ctk.CTkSlider(self.controls_frame, from_=0.1, to=1.0, number_of_steps=18, command=self.on_conf_slider_change)
        self.conf_slider.set(0.4)
        self.conf_slider.pack(fill="x", padx=15, pady=2)
        
        # Sensitivity Slider
        self.sens_label = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.sens_label.pack(fill="x", padx=15, pady=(5, 0))
        ctk.CTkLabel(self.sens_label, text="Độ trễ cảnh báo:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.sensitivity_val = ctk.CTkLabel(self.sens_label, text="10 frames", font=ctk.CTkFont(size=11, weight="bold"))
        self.sensitivity_val.pack(side="right")
        
        self.sensitivity_slider = ctk.CTkSlider(self.controls_frame, from_=2, to=30, number_of_steps=28, command=self.on_sensitivity_slider_change)
        self.sensitivity_slider.set(10)
        self.sensitivity_slider.pack(fill="x", padx=15, pady=2)
        
        # Mute control
        self.mute_var = ctk.BooleanVar(value=False)
        self.mute_check = ctk.CTkCheckBox(
            self.controls_frame, 
            text="Tắt âm thanh báo động", 
            font=ctk.CTkFont(size=11),
            variable=self.mute_var,
            command=self.on_mute_change
        )
        self.mute_check.pack(anchor="w", padx=15, pady=(10, 15))
        
        # 3. Text Log Output
        self.log_frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        self.log_frame.grid(row=2, column=0, padx=5, pady=(10, 0), sticky="nsew")
        self.log_frame.grid_rowconfigure(0, weight=0)
        self.log_frame.grid_rowconfigure(1, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            self.log_frame, 
            text="📋 NHẬT KÝ HỆ THỐNG", 
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, padx=15, pady=(10, 2), sticky="w")
        
        self.log_text = ctk.CTkTextbox(self.log_frame, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_text.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        self.log_text.configure(state="disabled")

    def create_settings_panel(self):
        """Creates the configurations Tab layout (centered when open)."""
        self.settings_panel = ctk.CTkFrame(self, corner_radius=15)
        # Gridded dynamically in show_tab()
        self.settings_panel.grid_rowconfigure(0, weight=0)
        self.settings_panel.grid_rowconfigure(1, weight=1)
        self.settings_panel.grid_columnconfigure(0, weight=1)
        
        # Panel Title
        ctk.CTkLabel(
            self.settings_panel, 
            text="⚙️ CẤU HÌNH HỆ THỐNG AI FIRE SENTINEL", 
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        form_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        form_frame.grid(row=1, column=0, padx=40, pady=10, sticky="nsew")
        
        # 1. Video stream source config
        ctk.CTkLabel(form_frame, text="Nguồn Camera Đầu Vào:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5))
        self.source_var = ctk.StringVar(value="Webcam 0")
        self.source_menu = ctk.CTkOptionMenu(
            form_frame, 
            values=["Webcam 0", "Webcam 1", "File Video", "IP Camera RTSP"],
            variable=self.source_var,
            command=self.on_source_change
        )
        self.source_menu.pack(fill="x", pady=5)
        
        # File / URL input fields
        self.source_input_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        self.source_input_frame.pack(fill="x", pady=5)
        
        self.source_entry_label = ctk.CTkLabel(self.source_input_frame, text="Đường dẫn / URL RTSP:")
        self.source_entry = ctk.CTkEntry(self.source_input_frame, placeholder_text="Nhập link RTSP hoặc link luồng phát...")
        self.btn_browse = ctk.CTkButton(self.source_input_frame, text="Chọn File Video", command=self.browse_video_file)
        
        # 2. Telegram API Config
        ctk.CTkLabel(form_frame, text="Cấu Hình Telegram Bot Cảnh Báo:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(20, 5))
        
        self.token_entry = ctk.CTkEntry(form_frame, placeholder_text="Telegram Token Bot", show="*")
        self.token_entry.pack(fill="x", pady=5)
        
        self.chat_id_entry = ctk.CTkEntry(form_frame, placeholder_text="Telegram Chat ID")
        self.chat_id_entry.pack(fill="x", pady=5)
        
        # Button to test Telegram Integration
        self.btn_test_tele = ctk.CTkButton(
            form_frame, 
            text="Gửi Thử Tin Nhắn Tele", 
            fg_color="#E67E22", 
            hover_color="#D35400",
            font=ctk.CTkFont(weight="bold"),
            command=self.test_telegram
        )
        self.btn_test_tele.pack(fill="x", pady=10)
        
        # Action Buttons (START/STOP) in settings too for convenience
        btn_action_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        btn_action_frame.pack(fill="x", side="bottom", pady=20)
        
        self.btn_start = ctk.CTkButton(
            btn_action_frame, 
            text="▶ BẮT ĐẦU GIÁM SÁT", 
            fg_color="#2ECC71", 
            hover_color="#27AE60",
            font=ctk.CTkFont(weight="bold", size=14),
            height=45,
            command=self.start_monitoring
        )
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_stop = ctk.CTkButton(
            btn_action_frame, 
            text="■ DỪNG HỆ THỐNG", 
            fg_color="#E74C3C", 
            hover_color="#C0392B",
            font=ctk.CTkFont(weight="bold", size=14),
            height=45,
            state="disabled",
            command=self.stop_monitoring
        )
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(10, 0))

    def show_tab(self, tab_name):
        """Switches display content based on active sidebar tab."""
        self.current_tab = tab_name
        
        if tab_name == "monitor":
            self.btn_tab_monitor.configure(fg_color="#D35400") # Dark orange highlight
            self.btn_tab_settings.configure(fg_color="transparent")
            
            # Hide settings, show main & right panels
            self.settings_panel.grid_forget()
            self.main_panel.grid(row=0, column=1, padx=(0, 10), pady=15, sticky="nsew")
            self.right_panel.grid(row=0, column=2, padx=(0, 15), pady=15, sticky="nsew")
        elif tab_name == "settings":
            self.btn_tab_monitor.configure(fg_color="transparent")
            self.btn_tab_settings.configure(fg_color="#D35400")
            
            # Hide main & right panels, show settings panel
            self.main_panel.grid_forget()
            self.right_panel.grid_forget()
            self.settings_panel.grid(row=0, column=1, columnspan=2, padx=(0, 15), pady=15, sticky="nsew")

    def load_defaults(self):
        """Loads default config values and checks configuration status."""
        self.token_entry.insert(0, os.environ.get("TELEGRAM_TOKEN", ""))
        self.chat_id_entry.insert(0, os.environ.get("TELEGRAM_CHAT_ID", ""))
        self.on_source_change("Webcam 0")
        self.update_telegram_status_led()

    def update_telegram_status_led(self):
        """Update Telegram status LED based on config entries."""
        token = self.token_entry.get().strip()
        chat_id = self.chat_id_entry.get().strip()
        if token and chat_id:
            self.led_tele.configure(text_color="#2ECC71") # Green
            self.lbl_tele_status.configure(text="Telegram: Sẵn sàng")
        else:
            self.led_tele.configure(text_color="#F1C40F") # Yellow
            self.lbl_tele_status.configure(text="Telegram: Chưa có")

    def on_source_change(self, value):
        """Show/hide custom URL/file inputs depending on camera selection."""
        self.source_entry_label.pack_forget()
        self.source_entry.pack_forget()
        self.btn_browse.pack_forget()
        
        if value == "File Video":
            self.source_entry_label.configure(text="Đường dẫn file video:")
            self.source_entry_label.pack(anchor="w", pady=2)
            # Pack the button FIRST so it claims the right-side space
            self.btn_browse.pack(side="right", padx=(5, 0))
            # Pack the entry SECOND so it fills the remaining space on the left
            self.source_entry.pack(fill="x", side="left", expand=True)
        elif value == "IP Camera RTSP":
            self.source_entry_label.configure(text="Nhập link RTSP của Camera:")
            self.source_entry_label.pack(anchor="w", pady=2)
            self.source_entry.pack(fill="x")

    def browse_video_file(self):
        """File open selector dialog."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, filepath)

    def on_conf_slider_change(self, val):
        self.conf_slider_val.configure(text=f"{int(float(val)*100)}%")
        self.detector.conf_threshold = float(val)
        # Update threshold line in matplotlib chart dynamically (axhline has size 2)
        self.threshold_line.set_ydata([float(val), float(val)])
        self.canvas.draw_idle()

    def on_sensitivity_slider_change(self, val):
        self.sensitivity_val.configure(text=f"{int(float(val))} frames")
        self.detector.consecutive_frames_limit = int(float(val))

    def on_mute_change(self):
        self.alert_manager.mute_alarm = self.mute_var.get()
        self.add_log(f"Đã {'tắt' if self.mute_var.get() else 'mở'} âm thanh báo động.")

    def add_log(self, text):
        """Appends a new line to console screen."""
        timestamp = time.strftime("[%H:%M:%S]")
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"{timestamp} {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def test_telegram(self):
        """Trigger testing alert on Telegram."""
        token = self.token_entry.get().strip()
        chat_id = self.chat_id_entry.get().strip()
        
        if not token or not chat_id:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ Telegram Token và Chat ID.")
            return
            
        self.alert_manager.update_config(token, chat_id, self.mute_var.get())
        self.update_telegram_status_led()
        self.add_log("Đang gửi thử nghiệm thông báo tới Telegram...")
        
        # Run synchronously for test alerts to give direct user feedback on correctness of token/id
        success, error_msg = self.alert_manager.send_telegram_alert_sync(
            None, 
            message="🧪 Kiểm thử hệ thống: Cấu hình Telegram của bạn hoạt động chính xác! ✅"
        )
        if success:
            messagebox.showinfo("Thành công", "Đã gửi tin nhắn test. Vui lòng kiểm tra Telegram của bạn.")
        else:
            self.add_log(f"Lỗi Telegram: {error_msg}")
            messagebox.showerror("Thất bại", f"Lỗi gửi thử nghiệm:\n{error_msg}")

    def open_alerts_folder(self):
        """Open local folder holding alert images."""
        alerts_dir = os.path.abspath("alerts")
        if not os.path.exists(alerts_dir):
            os.makedirs(alerts_dir)
        try:
            os.startfile(alerts_dir)
            self.add_log("Đã mở thư mục ảnh cảnh báo.")
        except Exception as e:
            try:
                import subprocess
                subprocess.Popen(f'explorer "{alerts_dir}"')
                self.add_log("Đã mở thư mục ảnh cảnh báo.")
            except Exception as ex:
                self.add_log(f"Không thể mở thư mục ảnh: {ex}")

    def start_monitoring(self):
        """Start the video AI thread."""
        token = self.token_entry.get().strip()
        chat_id = self.chat_id_entry.get().strip()
        
        self.alert_manager.update_config(token, chat_id, self.mute_var.get())
        self.update_telegram_status_led()
        
        # Parse video source
        src_option = self.source_var.get()
        if src_option == "Webcam 0":
            source = "0"
        elif src_option == "Webcam 1":
            source = "1"
        elif src_option in ["File Video", "IP Camera RTSP"]:
            source = self.source_entry.get().strip()
            if not source:
                messagebox.showerror("Lỗi", "Vui lòng cung cấp đường dẫn video hoặc link RTSP.")
                return
        else:
            source = "0"
            
        # Start detector
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.detector.start(source, self.alert_manager)

    def stop_monitoring(self):
        """Stop the video AI thread."""
        self.btn_stop.configure(state="disabled")
        self.detector.stop()
        self.btn_start.configure(state="normal")

    def on_detector_status_change(self, status):
        """Callback showing connection and state change of YOLO engine."""
        if status == "Running":
            self.banner_label.configure(text="💚 ĐANG GIÁM SÁT - HOẠT ĐỘNG 💚")
            self.banner.configure(fg_color="#27AE60")
            self.status_led.configure(text_color="#2ECC71") # Green LED
            
            self.led_cam.configure(text_color="#2ECC71")     # Sidebar status
            self.lbl_cam_status.configure(text="Camera: Connected")
        elif status == "Stopped":
            self.banner_label.configure(text="HỆ THỐNG ĐANG TẮT")
            self.banner.configure(fg_color="#1E2530")
            self.status_led.configure(text_color="#95A5A6") # Gray LED
            self.video_label.configure(image="", text="Không có tín hiệu camera")
            
            self.led_cam.configure(text_color="#E74C3C")     # Sidebar status
            self.lbl_cam_status.configure(text="Camera: Off")
        elif "Error" in status:
            self.banner_label.configure(text="❌ LỖI KẾT NỐI CAMERA ❌")
            self.banner.configure(fg_color="#C0392B")
            self.status_led.configure(text_color="#E74C3C") # Red LED
            
            self.led_cam.configure(text_color="#E74C3C")     # Sidebar status
            self.lbl_cam_status.configure(text="Camera: Error")
            self.stop_monitoring()

    def on_alert_state_change(self, is_active):
        """Callback showing if a fire alert condition is active."""
        self.alert_active = is_active
        if not is_active:
            self.banner_label.configure(
                text="💚 ĐANG GIÁM SÁT - AN TOÀN 💚" if self.detector.running else "HỆ THỐNG ĐANG TẮT"
            )
            self.banner.configure(
                fg_color="#27AE60" if self.detector.running else "#1E2530"
            )
            if self.detector.running:
                self.status_led.configure(text_color="#2ECC71")
            else:
                self.status_led.configure(text_color="#95A5A6")
        else:
            # When alert is active, update the Smart History thumbnail frame on right panel
            if self.detector.running and self.detector.latest_annotated_frame is not None:
                self.update_smart_history_thumbnail(self.detector.latest_annotated_frame, self.detector.latest_conf)

    def update_smart_history_thumbnail(self, frame, conf_val):
        """Updates the thumbnail widget on right panel with the current warning frame."""
        try:
            h, w = frame.shape[:2]
            # Target thumbnail size: width=220, height=140
            target_w = 220
            target_h = int((target_w / w) * h)
            if target_h > 140:
                target_h = 140
                target_w = int((target_h / h) * w)
                
            frame_resized = cv2.resize(frame, (target_w, target_h))
            cv2_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_rgb)
            
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            
            # Update Thumbnail image label
            self.thumb_label.configure(image=ctk_img, text="")
            self.thumb_label._image = ctk_img
            
            # Update labels
            self.lbl_alert_time.configure(text=f"Thời gian: {time.strftime('%H:%M:%S')}")
            self.lbl_alert_conf.configure(text=f"Độ tin cậy: {int(conf_val * 100)}%")
        except Exception as e:
            print(f"Error drawing smart history thumbnail: {e}")

    def flash_red_banner(self):
        """Siren visual flashing effect (banner and LED)."""
        if self.alert_active and self.detector.running:
            self.is_flashing = not self.is_flashing
            if self.is_flashing:
                self.banner_label.configure(text="🚨 CẢNH BÁO: PHÁT HIỆN CÓ CHÁY/KHÓI! 🚨")
                self.banner.configure(fg_color="#C0392B") 
                self.status_led.configure(text_color="#E74C3C") 
            else:
                self.banner_label.configure(text="🚨 CẢNH BÁO: PHÁT HIỆN CÓ CHÁY/KHÓI! 🚨")
                self.banner.configure(fg_color="#1E2530") 
                self.status_led.configure(text_color="#1E2530") 
        # Loop flashing
        self.after(500, self.flash_red_banner)

    def update_video_loop(self):
        """Fetches the latest camera frame and shows it on screen."""
        if self.detector.running and self.current_tab == "monitor":
            frame = self.detector.latest_annotated_frame
            if frame is not None:
                try:
                    h, w = frame.shape[:2]
                    # Target dimension: width=640
                    target_w = 640
                    target_h = int((target_w / w) * h)
                    
                    frame_resized = cv2.resize(frame, (target_w, target_h))
                    cv2_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cv2_rgb)
                    
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
                    self.video_label.configure(image=ctk_img, text="")
                    self.video_label._image = ctk_img
                except Exception as e:
                    print(f"Error drawing frame in GUI: {e}")
                    
        # Update every 33ms (~30 FPS)
        self.after(33, self.update_video_loop)

    def update_chart_loop(self):
        """Updates the dynamic Matplotlib chart with current detector confidence levels."""
        if self.detector.running and self.current_tab == "monitor":
            # Fetch latest confidence value
            current_conf = self.detector.latest_conf
            
            # Slide history queue
            self.conf_history.append(current_conf)
            self.conf_history.pop(0)
            
            try:
                # Update data on the line plot (very lightweight)
                self.plot_line.set_ydata(self.conf_history)
                self.canvas.draw_idle()
            except Exception as e:
                print(f"Error updating Matplotlib chart: {e}")
                
        # Update chart every 100ms (~10 FPS is plenty for graph visualization)
        self.after(100, self.update_chart_loop)

    def on_closing(self):
        """Ensures all loops and streams stop on window close."""
        if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn thoát ứng dụng không?"):
            self.detector.stop()
            self.alert_manager.stop_alarm()
            self.destroy()

if __name__ == "__main__":
    app = FireAlarmApp()
    # Trigger banner flashing thread
    app.after(500, app.flash_red_banner)
    app.mainloop()
