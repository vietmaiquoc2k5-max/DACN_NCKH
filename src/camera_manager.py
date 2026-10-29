"""Module quản lý đa luồng 9 Camera (3x3 Grid) thời gian thực (`src/camera_manager.py`).

Đáp ứng toàn bộ yêu cầu Mục 5 (Xử lý nhiều camera):
- Hỗ trợ tối đa 9 nguồn camera (Webcam, Video cục bộ, RTSP).
- Mỗi luồng có worker thread và trạng thái riêng (`ONLINE`, `PAUSED`, `DISCONNECTED`, `STOPPED`, `CONNECTING`).
- Không để lỗi hoặc mất kết nối ở một camera làm treo các camera còn lại.
- Có cơ chế giải phóng tài nguyên (`cap.release()`) và kết nối lại (`reconnect`).
- Tích hợp `TemporalVerifier` riêng cho từng camera và chia sẻ duy nhất một `SharedFireSmokeDetector`
  để tránh nạp lại model YOLO nhiều lần.
"""
import os
import sys
import time
import threading
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from src.config import (
    CameraConfig,
    SystemSettings,
    PROJECT_ROOT,
    get_default_9_cameras,
)
from src.detector import SharedFireSmokeDetector
from src.temporal_verifier import TemporalVerifier, DetectionBox
from src.database import EventDatabase
from alert_manager import AlertManager


def create_status_placeholder_rgb(
    cam_name: str,
    status: str,
    detail: str = "",
    width: int = 480,
    height: int = 270,
) -> np.ndarray:
    """Tạo khung hình RGB thông báo trạng thái rõ ràng khi camera đang tắt hoặc mất kết nối."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (15, 23, 42)  # Dark slate background (#0F172A)

    # Khung viền theo trạng thái
    if status in ("DISCONNECTED", "ERROR"):
        border_color = (239, 68, 68)    # Đỏ cảnh báo mất kết nối
        title_text = "CAMERA DISCONNECTED"
    elif status == "PAUSED":
        border_color = (245, 158, 11)   # Vàng tạm dừng
        title_text = "STREAM PAUSED"
    elif status == "CONNECTING":
        border_color = (56, 189, 248)   # Xanh dương đang kết nối
        title_text = "CONNECTING STREAM..."
    else:
        border_color = (71, 85, 105)    # Xám khi đã dừng
        title_text = "CAMERA STANDBY"

    cv2.rectangle(img, (2, 2), (width - 3, height - 3), border_color, 2)

    # Tên camera góc trên
    cv2.putText(
        img,
        cam_name[:36],
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    # Trạng thái chính giữa
    (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    tx = max(16, (width - tw) // 2)
    ty = height // 2 - 6
    cv2.putText(
        img,
        title_text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        border_color,
        2,
        cv2.LINE_AA,
    )

    if detail:
        sub = detail[:52]
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        sx = max(16, (width - sw) // 2)
        cv2.putText(
            img,
            sub,
            (sx, ty + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (203, 213, 225),
            1,
            cv2.LINE_AA,
        )

    time_str = time.strftime("%H:%M:%S")
    cv2.putText(
        img,
        f"Updated: {time_str}",
        (16, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )
    return img


class CameraStreamWorker:
    """Quản lý vòng đời đọc luồng video, suy luận YOLO và xác minh thời gian cho 1 camera."""

    def __init__(
        self,
        config: CameraConfig,
        detector: SharedFireSmokeDetector,
        event_db: EventDatabase,
        settings: SystemSettings,
        log_callback=None,
    ):
        self.config = config
        self.detector = detector
        self.event_db = event_db
        self.settings = settings
        self.log_callback = log_callback

        self.verifier = TemporalVerifier(
            window_size_n=settings.window_size_n,
            confirm_frames_k=settings.confirm_frames_k,
            suspect_frames_k=settings.suspect_frames_k,
            cooldown_sec=settings.alert_cooldown_sec,
            exclusion_zones=config.exclusion_zones,
        )

        self.status: str = "STOPPED"
        self.error_detail: str = "Nhấn Bắt đầu để kết nối"
        self.running: bool = False
        self.paused: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Chỉ số thời gian thực của luồng
        self.fps: float = 0.0
        self.infer_ms: float = 0.0
        self.last_update_ts: float = 0.0
        self.last_update_str: str = "--:--:--"
        self.frame_count: int = 0

        # Trạng thái phát hiện hiện tại
        self.alert_level: str = "SAFE"  # "SAFE" | "SUSPECT" | "FIRE"
        self.primary_label: str = ""
        self.max_confidence: float = 0.0
        self.active_hits: int = 0
        self.window_size: int = settings.window_size_n
        self.last_boxes: List[DetectionBox] = []

        self.latest_rgb_frame: np.ndarray = create_status_placeholder_rgb(
            self.config.name, "STOPPED", self.error_detail
        )

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(f"[{self.config.cam_id}] {msg}")

    def start(self):
        """Khởi chạy worker thread cho camera này nếu chưa chạy."""
        with self.lock:
            if self.running and self.thread and self.thread.is_alive():
                self.paused = False
                if self.status == "PAUSED":
                    self.status = "ONLINE"
                return
            self.running = True
            self.paused = False
            self.status = "CONNECTING"
            self.error_detail = f"Đang mở nguồn: {self.config.source}"
            self.latest_rgb_frame = create_status_placeholder_rgb(
                self.config.name, "CONNECTING", self.error_detail
            )
            self.verifier.reset()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
        self._log(f"Bắt đầu kết nối nguồn '{self.config.source}'")

    def pause(self):
        """Tạm dừng xử lý luồng camera (giữ kết nối hoặc tạm nghỉ đọc frame)."""
        with self.lock:
            if self.running:
                self.paused = not self.paused
                self.status = "PAUSED" if self.paused else "ONLINE"
                if self.paused:
                    self.fps = 0.0
                    self.latest_rgb_frame = create_status_placeholder_rgb(
                        self.config.name, "PAUSED", "Đã tạm dừng bởi người vận hành"
                    )
        self._log("Tạm dừng luồng" if self.paused else "Tiếp tục luồng")

    def stop(self):
        """Dừng hoàn toàn luồng camera và giải phóng tài nguyên."""
        self.running = False
        self.paused = False
        t = self.thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.5)
        with self.lock:
            self.status = "STOPPED"
            self.fps = 0.0
            self.infer_ms = 0.0
            self.alert_level = "SAFE"
            self.primary_label = ""
            self.max_confidence = 0.0
            self.active_hits = 0
            self.error_detail = "Đã dừng giám sát"
            self.latest_rgb_frame = create_status_placeholder_rgb(
                self.config.name, "STOPPED", self.error_detail
            )
        self._log("Đã dừng và giải phóng tài nguyên camera.")

    def reconnect(self):
        """Ngắt kết nối hiện tại và khởi tạo lại luồng camera."""
        self._log("Đang thực hiện kết nối lại (Reconnect)...")
        self.stop()
        time.sleep(0.1)
        self.start()

    def _open_capture(self) -> Tuple[Optional[cv2.VideoCapture], bool]:
        """Mở nguồn video (Webcam, File cục bộ hoặc RTSP) một cách an toàn."""
        raw_src = str(self.config.source).strip()
        if not raw_src:
            self.error_detail = "Chưa cấu hình nguồn camera"
            return None, False

        is_file = False
        if raw_src.isdigit():
            cap_src: Any = int(raw_src)
            if sys.platform == "win32":
                cap = cv2.VideoCapture(cap_src, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(cap_src)
            else:
                cap = cv2.VideoCapture(cap_src)
        elif raw_src.lower().startswith(("rtsp://", "http://", "https://", "rtmp://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
            cap = cv2.VideoCapture(raw_src, cv2.CAP_FFMPEG)
        else:
            is_file = True
            p = Path(raw_src)
            if not p.is_absolute():
                p = PROJECT_ROOT / raw_src
            if not p.exists():
                self.error_detail = f"Không tìm thấy file: {raw_src}"
                return None, True
            cap = cv2.VideoCapture(str(p))

        if not cap or not cap.isOpened():
            self.error_detail = f"Không thể mở nguồn: {raw_src}"
            if cap:
                cap.release()
            return None, is_file

        return cap, is_file

    def _run_loop(self):
        """Vòng lặp đọc frame và nhận diện YOLO chạy trong thread riêng biệt."""
        cap, is_file = self._open_capture()
        if cap is None:
            with self.lock:
                self.status = "DISCONNECTED"
                self.running = False
                self.fps = 0.0
                self.latest_rgb_frame = create_status_placeholder_rgb(
                    self.config.name, "DISCONNECTED", self.error_detail
                )
            self._log(f"Mất kết nối: {self.error_detail}")
            return

        with self.lock:
            self.status = "ONLINE"
            self.error_detail = ""

        fps_window = deque(maxlen=20)
        last_tick = time.perf_counter()

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    last_tick = time.perf_counter()
                    continue

                loop_start = time.perf_counter()
                ret, frame = cap.read()

                if not ret or frame is None:
                    if is_file and self.config.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()

                    if not ret or frame is None:
                        with self.lock:
                            self.status = "DISCONNECTED"
                            self.fps = 0.0
                            self.error_detail = "Mất tín hiệu từ nguồn video/camera"
                            self.latest_rgb_frame = create_status_placeholder_rgb(
                                self.config.name, "DISCONNECTED", self.error_detail
                            )
                            self.running = False
                        self._log("Mất tín hiệu hoặc kết thúc luồng video.")
                        break

                self.frame_count += 1

                # Cập nhật tham số động của TemporalVerifier từ SystemSettings
                self.verifier.update_params(
                    window_size_n=self.settings.window_size_n,
                    confirm_frames_k=self.settings.confirm_frames_k,
                    suspect_frames_k=self.settings.suspect_frames_k,
                    cooldown_sec=self.settings.alert_cooldown_sec,
                )

                # Hỗ trợ frame_skip để giảm tải phần cứng khi chạy đồng thời 9 luồng
                skip_n = max(1, int(self.settings.frame_skip))
                if (self.frame_count % skip_n == 0) or (self.frame_count == 1):
                    raw_boxes, infer_ms = self.detector.detect_frame(
                        frame, conf_threshold=self.settings.conf_threshold
                    )
                    ver_res = self.verifier.update(raw_boxes)
                    valid_boxes = ver_res["valid_boxes"]
                    self.last_boxes = valid_boxes
                else:
                    infer_ms = self.infer_ms
                    valid_boxes = self.last_boxes
                    ver_res = {
                        "valid_boxes": valid_boxes,
                        "active_hits": sum(self.verifier.frame_history),
                        "window_size": self.verifier.window_size_n,
                        "alert_level": self.verifier.current_level,
                        "should_trigger_db_alert": False,
                        "primary_label": self.primary_label,
                        "max_confidence": self.max_confidence,
                    }

                # Tính FPS thực tế
                now_perf = time.perf_counter()
                dt = max(1e-4, now_perf - last_tick)
                last_tick = now_perf
                fps_window.append(1.0 / dt)
                actual_fps = float(sum(fps_window) / len(fps_window))

                # Vẽ bounding box, nhãn FIRE/SMOKE và confidence trực tiếp lên frame
                annotated_bgr = SharedFireSmokeDetector.draw_annotations(
                    frame=frame,
                    boxes=valid_boxes,
                    cam_name=self.config.name,
                    alert_level=ver_res["alert_level"],
                    fps=actual_fps,
                    exclusion_zones=self.config.exclusion_zones,
                )

                # Nếu sự kiện cháy/khói mới được xác nhận qua k/N khung hình -> lưu vào SQLite & phát cảnh báo
                if ver_res["should_trigger_db_alert"]:
                    lbl = ver_res["primary_label"] or "FIRE"
                    conf_val = ver_res["max_confidence"]
                    self.event_db.record_event(
                        frame=annotated_bgr,
                        camera_name=self.config.cam_id,
                        label=lbl,
                        confidence=conf_val,
                        alert_level=ver_res["alert_level"],
                        is_demo=0,
                    )
                    if not self.settings.mute_audio:
                        self.event_db.alert_manager.play_alarm()
                    self._log(
                        f"🚨 XÁC NHẬN SỰ KIỆN [{lbl}] ({int(conf_val*100)}%) "
                        f"sau {ver_res['active_hits']}/{ver_res['window_size']} khung!"
                    )

                # Thu phóng khung hình hiển thị theo display_width để render 9 ô 3x3 mượt mà
                disp_w = max(320, int(self.settings.display_width))
                h, w = annotated_bgr.shape[:2]
                if w > disp_w:
                    disp_h = max(180, int(h * (disp_w / float(w))))
                    disp_bgr = cv2.resize(annotated_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                else:
                    disp_bgr = annotated_bgr

                rgb_img = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
                now_str = time.strftime("%H:%M:%S")

                with self.lock:
                    self.status = "ONLINE"
                    self.fps = round(actual_fps, 1)
                    self.infer_ms = round(infer_ms, 1)
                    self.last_update_ts = time.time()
                    self.last_update_str = now_str
                    self.alert_level = ver_res["alert_level"]
                    self.primary_label = ver_res["primary_label"]
                    self.max_confidence = ver_res["max_confidence"]
                    self.active_hits = ver_res["active_hits"]
                    self.window_size = ver_res["window_size"]
                    self.latest_rgb_frame = rgb_img

                # Điều tiết tốc độ đọc theo target_fps_per_cam
                target_fps = max(1, int(self.settings.target_fps_per_cam))
                target_period = 1.0 / float(target_fps)
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.005, target_period - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            with self.lock:
                self.status = "ERROR"
                self.fps = 0.0
                self.error_detail = f"Lỗi luồng: {str(e)[:45]}"
                self.latest_rgb_frame = create_status_placeholder_rgb(
                    self.config.name, "ERROR", self.error_detail
                )
                self.running = False
            self._log(f"Ngoại lệ trong luồng camera: {e}")
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def get_snapshot(self) -> Dict[str, Any]:
        """Lấy bản sao trạng thái hiện tại của camera một cách an toàn đa luồng."""
        with self.lock:
            return {
                "cam_id": self.config.cam_id,
                "name": self.config.name,
                "source": self.config.source,
                "enabled": self.config.enabled,
                "status": self.status,
                "error_detail": self.error_detail,
                "fps": self.fps,
                "infer_ms": self.infer_ms,
                "last_update_str": self.last_update_str,
                "alert_level": self.alert_level,
                "primary_label": self.primary_label,
                "max_confidence": self.max_confidence,
                "active_hits": self.active_hits,
                "window_size": self.window_size,
                "confirm_k": self.settings.confirm_frames_k,
                "frame_rgb": self.latest_rgb_frame,
                "box_count": len(self.last_boxes),
            }


class MultiCameraManager:
    """Bộ điều phối trung tâm quản lý 9 luồng camera (bố cục 3x3), Detector dùng chung và SQLite."""

    def __init__(self, db_path: str = "alerts.db", model_path: str = "best.pt"):
        self.settings = SystemSettings(model_path=model_path)
        self.alert_manager = AlertManager(db_path=db_path)
        self.event_db = EventDatabase(alert_manager=self.alert_manager, db_path=db_path)
        self.detector = SharedFireSmokeDetector(
            model_path=self.settings.model_path,
            conf_threshold=self.settings.conf_threshold,
            imgsz=self.settings.imgsz,
        )

        self.logs: deque = deque(maxlen=80)
        self.log_lock = threading.Lock()

        self.camera_configs: List[CameraConfig] = get_default_9_cameras()
        self.workers: Dict[str, CameraStreamWorker] = {}
        for cfg in self.camera_configs:
            self.workers[cfg.cam_id] = CameraStreamWorker(
                config=cfg,
                detector=self.detector,
                event_db=self.event_db,
                settings=self.settings,
                log_callback=self.add_log,
            )

        self.add_log("Khởi tạo trung tâm điều hành 9 Camera (3x3 Grid) thành công.")
        model_info = self.detector.get_model_info()
        if model_info["loaded"]:
            self.add_log(f"Đã nạp model YOLO: {model_info['path']} | Lớp: {model_info['names']}")
        else:
            self.add_log(f"⚠️ CẢNH BÁO MODEL: {model_info['error']}")

    def add_log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        with self.log_lock:
            self.logs.appendleft(f"[{ts}] {message}")

    def get_recent_logs(self, limit: int = 25) -> List[str]:
        with self.log_lock:
            return list(self.logs)[:limit]

    def start_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].config.enabled = True
            self.workers[cam_id].start()

    def pause_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].pause()

    def stop_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].stop()

    def reconnect_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].reconnect()

    def start_all(self, only_enabled: bool = False):
        """Khởi chạy đồng loạt các camera."""
        for cid, w in self.workers.items():
            if not only_enabled or w.config.enabled:
                w.config.enabled = True
                w.start()

    def pause_all(self):
        """Tạm dừng đồng loạt các camera đang chạy."""
        for w in self.workers.values():
            if w.running and not w.paused:
                w.pause()

    def stop_all(self):
        """Dừng toàn bộ 9 camera và tắt còi báo động."""
        for w in self.workers.values():
            w.stop()
        self.alert_manager.stop_alarm()

    def reconnect_all(self):
        """Kết nối lại tất cả các camera đã bật hoặc đang bị mất kết nối."""
        for w in self.workers.values():
            if w.config.enabled or w.status in ("DISCONNECTED", "ERROR", "ONLINE"):
                w.reconnect()

    def update_camera_source(self, cam_id: str, new_name: str, new_source: str, auto_restart: bool = True):
        """Cập nhật tên và nguồn cho một camera cụ thể."""
        if cam_id not in self.workers:
            return
        w = self.workers[cam_id]
        was_running = w.running
        w.stop()
        w.config.name = new_name.strip() or cam_id
        w.config.source = new_source.strip()
        w.latest_rgb_frame = create_status_placeholder_rgb(
            w.config.name, "STOPPED", f"Nguồn mới: {w.config.source}"
        )
        self.add_log(f"Đã cập nhật cấu hình {cam_id}: '{w.config.name}' -> '{w.config.source}'")
        if auto_restart and was_running:
            w.start()

    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """Lấy trạng thái thời gian thực của cả 9 camera theo đúng thứ tự CAM-01..CAM-09."""
        return [self.workers[cfg.cam_id].get_snapshot() for cfg in self.camera_configs]

    def get_system_metrics(self) -> Dict[str, Any]:
        """Tổng hợp chỉ số vận hành thực tế của toàn hệ thống 9 camera."""
        snaps = self.get_all_snapshots()
        online_cams = [s for s in snaps if s["status"] == "ONLINE"]
        disconnected_cams = [s for s in snaps if s["status"] in ("DISCONNECTED", "ERROR")]
        fire_cams = [s for s in snaps if s["alert_level"] == "FIRE"]
        suspect_cams = [s for s in snaps if s["alert_level"] == "SUSPECT"]

        total_fps = sum(s["fps"] for s in online_cams)
        avg_fps = (total_fps / len(online_cams)) if online_cams else 0.0

        overall_level = "SAFE"
        if fire_cams:
            overall_level = "FIRE"
        elif suspect_cams:
            overall_level = "SUSPECT"

        return {
            "total_cameras": len(snaps),
            "online_count": len(online_cams),
            "disconnected_count": len(disconnected_cams),
            "fire_cam_count": len(fire_cams),
            "suspect_cam_count": len(suspect_cams),
            "total_fps": round(total_fps, 1),
            "avg_fps": round(avg_fps, 1),
            "overall_level": overall_level,
            "fire_cameras": [s["cam_id"] for s in fire_cams],
            "suspect_cameras": [s["cam_id"] for s in suspect_cams],
        }
