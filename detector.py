import cv2
import threading
import time
from ultralytics import YOLO

class FireDetector:
    def __init__(self, model_path="best.pt", fallback_model_path="yolov8n.pt"):
        self.model_path = model_path
        self.fallback_model_path = fallback_model_path
        self.model = None
        self.running = False
        self.thread = None
        
        # Thread communication
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_conf = 0.0
        
        # Configuration parameters
        self.conf_threshold = 0.4
        self.consecutive_frames_limit = 10
        self.telegram_cooldown = 60 # seconds between telegram photos
        
        # Detection state
        self.consecutive_fire_frames = 0
        self.last_telegram_time = 0
        
        # Status callbacks for GUI updates
        self.status_callback = None # func(status_text)
        self.alert_callback = None  # func(is_alert_active)
        self.log_callback = None    # func(log_message)
        
        # Load the model
        self.load_model()

    def load_model(self):
        """Loads the YOLO model (either custom best.pt or fallback yolov8n.pt)."""
        try:
            # Try to load custom model first
            import os
            if os.path.exists(self.model_path):
                self._log(f"Đang tải mô hình đã huấn luyện: {self.model_path}...")
                self.model = YOLO(self.model_path)
                self._log("Tải thành công mô hình custom best.pt!")
            else:
                self._log(f"Không tìm thấy '{self.model_path}'. Đang tải mô hình mặc định: {self.fallback_model_path}...")
                self.model = YOLO(self.fallback_model_path)
                self._log(f"Tải thành công mô hình mặc định {self.fallback_model_path} (Dùng để chạy thử nghiệm)!")
        except Exception as e:
            self._log(f"Lỗi khi tải mô hình YOLO: {str(e)}")
            self.model = None

    def _log(self, text):
        """Helper to send log message to GUI."""
        if self.log_callback:
            self.log_callback(text)
        else:
            print(text)

    def start(self, source, alert_manager):
        """Starts the detection thread."""
        if self.running:
            return
            
        if self.model is None:
            self.load_model()
            if self.model is None:
                self._log("Không thể khởi động: Chưa tải được mô hình YOLO.")
                return

        self.running = True
        self.thread = threading.Thread(
            target=self._detection_loop,
            args=(source, alert_manager),
            daemon=True
        )
        self.thread.start()
        self._log(f"Đã bắt đầu tiến trình nhận diện với nguồn đầu vào: {source}")

    def stop(self):
        """Stops the detection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_conf = 0.0
        self.consecutive_fire_frames = 0
        self._log("Đã dừng tiến trình nhận diện.")

    def _detection_loop(self, source, alert_manager):
        """Internal detection loop running in a background thread."""
        import sys
        # Convert source to int if it's a single digit (Webcam ID)
        if isinstance(source, str) and source.isdigit():
            cap_source = int(source)
        else:
            cap_source = source
            
        if isinstance(cap_source, int) and sys.platform == "win32":
            # DirectShow (CAP_DSHOW) is usually faster and more reliable for webcams on Windows
            cap = cv2.VideoCapture(cap_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to default backend if DirectShow fails
                cap = cv2.VideoCapture(cap_source)
        else:
            cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            self._log(f"Lỗi: Không thể kết nối với nguồn video '{source}'. Vui lòng kiểm tra xem camera đã được kết nối chưa, có bị ứng dụng khác chiếm giữ không, hoặc đổi nguồn đầu vào.")
            if self.status_callback:
                self.status_callback("Error Opening Source")
            self.running = False
            return

        if self.status_callback:
            self.status_callback("Running")

        # Determine class indices for fire and smoke
        # COCO standard names don't have fire/smoke, so if they are missing, we enter SIMULATION mode.
        model_names = self.model.names
        fire_smoke_indices = []
        is_simulation_mode = True
        
        for idx, name in model_names.items():
            if name.lower() in ["fire", "smoke", "chay", "khoi", "lửa", "khói"]:
                fire_smoke_indices.append(idx)
                is_simulation_mode = False

        if is_simulation_mode:
            self._log("⚠️ Chế độ Thử nghiệm: Không tìm thấy nhãn 'fire'/'smoke' trong mô hình. Nhận diện mọi vật thể (như person, cell phone...) để thử nghiệm còi/Telegram.")
        else:
            self._log(f"Đã cấu hình nhận diện nhãn: {[model_names[i] for i in fire_smoke_indices]}")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self._log("Hết luồng video hoặc mất kết nối camera.")
                break

            self.latest_frame = frame.copy()
            
            # Run YOLO inference on the frame
            # verbose=False reduces terminal output
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Track if fire/smoke (or any object in simulation mode) is detected in current frame
            fire_detected_in_frame = False
            highest_conf = 0.0
            detected_label = ""
            
            # Standard annotation plot
            annotated_frame = results[0].plot()
            
            # Analyze results
            boxes = results[0].boxes
            for box in boxes:
                class_idx = int(box.cls[0])
                conf = float(box.conf[0])
                
                if is_simulation_mode:
                    # In simulation mode, any detection triggers alert
                    fire_detected_in_frame = True
                    if conf > highest_conf:
                        highest_conf = conf
                        detected_label = model_names[class_idx]
                else:
                    # In normal mode, check if class matches fire/smoke
                    if class_idx in fire_smoke_indices:
                        fire_detected_in_frame = True
                        if conf > highest_conf:
                            highest_conf = conf
                            detected_label = model_names[class_idx]
            
            # Handle consecutive frames trigger to prevent false alerts
            if fire_detected_in_frame:
                self.consecutive_fire_frames += 1
                if self.consecutive_fire_frames >= self.consecutive_frames_limit:
                    # Trigger Alarm
                    alert_manager.play_alarm()
                    if self.alert_callback:
                        self.alert_callback(True)
                        
                    # Handle Telegram Alert Cooldown
                    current_time = time.time()
                    if current_time - self.last_telegram_time > self.telegram_cooldown:
                        self.last_telegram_time = current_time
                        msg = f"🚨 CẢNH BÁO: Phát hiện {detected_label.upper()}! Độ tin cậy: {highest_conf:.2f}"
                        if is_simulation_mode:
                            msg = f"🚨 [THỬ NGHIỆM] Cảnh báo giả lập: Nhận diện '{detected_label}'! Độ tin cậy: {highest_conf:.2f}"
                        
                        self._log(f"🔥 KÍCH HOẠT CẢNH BÁO: {msg}")
                        # Send the annotated frame to Telegram
                        alert_manager.send_telegram_alert(annotated_frame, message=msg)
            else:
                # Cool down slow-down
                if self.consecutive_fire_frames > 0:
                    self.consecutive_fire_frames -= 1
                else:
                    alert_manager.stop_alarm()
                    if self.alert_callback:
                        self.alert_callback(False)

            # Store the frame and confidence for GUI fetching
            self.latest_annotated_frame = annotated_frame
            self.latest_conf = highest_conf
            
            # Cap the loop execution speed to ~30 FPS
            time.sleep(0.01)

        # Cleanup
        cap.release()
        alert_manager.stop_alarm()
        self.latest_conf = 0.0
        if self.alert_callback:
            self.alert_callback(False)
        if self.status_callback:
            self.status_callback("Stopped")
        self.running = False
