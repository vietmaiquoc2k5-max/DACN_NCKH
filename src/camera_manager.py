"""Module quản lý đa luồng 9 Camera (3x3 Grid) & Camera Thực Tế (`src/camera_manager.py`).

Đáp ứng toàn bộ yêu cầu Mục 5, Mục 10 & Tối ưu hóa FPS / Chống giật lag:
- Hỗ trợ 4 chế độ hoạt động: `VIDEO_TEST`, `WEBCAM`, `IP_CAMERA`, `MULTI_CAMERA` (tối đa 9 camera).
- Hiển thị chuẩn xác trạng thái: `CONNECTED`, `DISCONNECTED`, `RECONNECTING`, `PAUSED`, `ERROR`.
- Có timeout, xử lý lỗi và cơ chế tự động kết nối lại (`RECONNECTING`) khi camera mất tín hiệu.
- Giải phóng `VideoCapture` (`cap.release()`) an toàn khi ngắt kết nối hoặc dừng ứng dụng.
- Hiển thị riêng biệt **FPS đọc khung hình (`capture_fps`)** và **FPS suy luận model (`inference_fps`)**.
- Nén khung hình JPEG bằng C++ OpenCV (`cv2.imencode`) ngay trong luồng nền để loại bỏ hoàn toàn
  độ trễ nén PNG trên luồng giao diện chính của Streamlit.
- Hỗ trợ suy luận không chặn (`non_blocking`) giúp luồng đọc video đạt 20-30 FPS mượt mà ngay cả khi
  bật đồng thời nhiều camera.
"""
import os
import sys
import time
import base64
import re
import json
import sqlite3
import logging
from contextlib import closing
import threading
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from src.config import (
    CameraConfig,
    SystemSettings,
    PROJECT_ROOT,
    get_default_9_cameras,
    mask_rtsp_credentials,
    infer_source_type,
)
from src.detector import SharedFireSmokeDetector
from src.temporal_verifier import TemporalVerifier, DetectionBox
from src.bytetrack_adapter import ByteTrackAdapter
from src.temporal_features import TemporalFeatureExtractor
from src.database import EventDatabase
from alert_manager import AlertManager
from src.telegram_alerts import TelegramAlertDispatcher, utc_now

LOGGER = logging.getLogger(__name__)


def encode_bgr_to_jpeg_b64(bgr_img: np.ndarray, quality: int = 72) -> str:
    """Nén nhanh khung hình BGR sang chuỗi Base64 JPEG bằng C++ OpenCV (< 0.5 ms)."""
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_rgb_to_jpeg_b64(rgb_img: np.ndarray, quality: int = 72) -> str:
    """Nén nhanh khung hình RGB sang chuỗi Base64 JPEG."""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return encode_bgr_to_jpeg_b64(bgr, quality=quality)


def create_status_placeholder_rgb(
    cam_name: str,
    status: str,
    detail: str = "",
    width: int = 420,
    height: int = 236,
) -> np.ndarray:
    """Tạo khung hình RGB thông báo trạng thái rõ ràng khi camera ngắt kết nối, đang kết nối lại hoặc lỗi."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (15, 23, 42)  # Dark slate background (#0F172A)

    if status == "ERROR":
        border_color = (239, 68, 68)    # Đỏ lỗi
        title_text = "CONNECTION ERROR"
    elif status == "RECONNECTING":
        border_color = (56, 189, 248)   # Xanh dương đang kết nối lại
        title_text = "RECONNECTING..."
    elif status == "DISCONNECTED":
        border_color = (249, 115, 22)   # Cam đỏ mất kết nối / đã ngắt
        title_text = "DISCONNECTED"
    elif status == "PAUSED":
        border_color = (245, 158, 11)   # Vàng tạm dừng
        title_text = "STREAM PAUSED"
    else:
        border_color = (100, 116, 139)
        title_text = str(status).upper()

    cv2.rectangle(img, (2, 2), (width - 3, height - 3), border_color, 2)

    cv2.putText(
        img,
        cam_name[:34],
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    (tw, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    tx = max(14, (width - tw) // 2)
    ty = height // 2 - 8
    cv2.putText(
        img,
        title_text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        border_color,
        2,
        cv2.LINE_AA,
    )

    if detail:
        safe_detail = mask_rtsp_credentials(detail)[:50]
        (sw, _), _ = cv2.getTextSize(safe_detail, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        sx = max(14, (width - sw) // 2)
        cv2.putText(
            img,
            safe_detail,
            (sx, ty + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (203, 213, 225),
            1,
            cv2.LINE_AA,
        )

    time_str = time.strftime("%H:%M:%S")
    cv2.putText(
        img,
        f"Updated: {time_str}",
        (14, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )
    return img


def list_system_cameras(max_check: int = 4) -> List[Dict[str, Any]]:
    """Tự động dò quét nhanh danh sách các camera phần cứng và phần mềm camera (Iriun, EOS, USB UVC...) trên máy tính."""
    cams = []
    device_hints = {
        0: "Webcam #0 - Camera Tích Hợp / Chính",
        1: "Webcam #1 - USB2.0 HD UVC WebCam",
        2: "Webcam #2 - Camera Phụ / USB",
        3: "Webcam #3 - Iriun / Virtual Camera",
    }
    for idx in range(max_check):
        cap = None
        is_ok = False
        res_str = ""
        backend_used = "CAP_DSHOW"
        # Ưu tiên CAP_DSHOW trên Windows để dò thiết bị nhanh (~0.05s/index)
        backends = [("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", None))]
        if idx == 0:
            backends.append(("CAP_MSMF", getattr(cv2, "CAP_MSMF", None)))
        for backend_name, b_val in backends:
            if b_val is None:
                continue
            try:
                cap = cv2.VideoCapture(idx, b_val)
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        res_str = f"{w}x{h}"
                        backend_used = backend_name
                        is_ok = True
                        break
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

        name = device_hints.get(idx, f"Webcam #{idx}")
        if is_ok:
            cams.append({
                "index": idx,
                "label": f"🟢 {name} [{res_str}] (Đang hoạt động)",
                "backend": backend_used,
                "resolution": res_str,
                "active": True,
            })
        else:
            cams.append({
                "index": idx,
                "label": f"⚪ {name} (Không có tín hiệu)",
                "backend": "CAP_DSHOW",
                "resolution": "N/A",
                "active": False,
            })
    return cams


def open_opencv_capture(
    source: str,
    source_type: str = "VIDEO_TEST",
    webcam_index: int = 0,
    capture_width: int = 480,
    preferred_backend: Optional[str] = None,
) -> Tuple[Optional[cv2.VideoCapture], bool, str]:
    """Mở nguồn video/webcam/RTSP bằng OpenCV VideoCapture với cấu hình giảm trễ (buffer=1).

    Hỗ trợ tự động fallback giữa CAP_MSMF, CAP_ANY, CAP_DSHOW để tương thích mọi phần mềm camera.

    Returns:
        (cap, is_file_source, error_message_masked)
    """
    raw_src = str(source).strip()
    s_type = source_type or infer_source_type(raw_src)

    if s_type == "WEBCAM" or raw_src.isdigit():
        idx = int(raw_src) if raw_src.isdigit() else int(webcam_index)
        try:
            cap = None
            # Ưu tiên các backend an toàn trên Windows: CAP_MSMF -> CAP_ANY -> CAP_DSHOW
            backend_candidates = []
            if preferred_backend == "CAP_DSHOW" and hasattr(cv2, "CAP_DSHOW"):
                backend_candidates.append(cv2.CAP_DSHOW)
            if hasattr(cv2, "CAP_MSMF"):
                backend_candidates.append(cv2.CAP_MSMF)
            backend_candidates.append(cv2.CAP_ANY)
            if hasattr(cv2, "CAP_DSHOW"):
                backend_candidates.append(cv2.CAP_DSHOW)

            last_err = ""
            for b in backend_candidates:
                try:
                    cap = cv2.VideoCapture(idx, b)
                    if cap is not None and cap.isOpened():
                        ret, test_f = cap.read()
                        if ret and test_f is not None:
                            break
                        else:
                            cap.release()
                            cap = None
                except Exception as ex:
                    last_err = str(ex)
                    if cap:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    cap = None

            if cap is None or not cap.isOpened():
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                return None, False, f"Không tìm thấy tín hiệu hình ảnh từ Webcam chỉ số #{idx} ({last_err or 'Thiết bị đang bận hoặc chưa bật'})"

            if capture_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(capture_width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(int(capture_width * 9 / 16)))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, False, ""
        except Exception as e:
            return None, False, f"Lỗi mở Webcam #{idx}: {e}"

    if s_type == "IP_CAMERA" or raw_src.lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://")):
        masked = mask_rtsp_credentials(raw_src)
        if not raw_src:
            return None, False, "Chưa nhập địa chỉ RTSP/IP Camera"
        try:
            # Thiết lập timeout 5 giây và tắt bộ đệm (fflags;nobuffer) để giảm độ trễ RTSP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;5000000|rw_timeout;5000000"
            )
            cap = cv2.VideoCapture(raw_src, cv2.CAP_FFMPEG)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                return None, False, f"Không thể kết nối luồng RTSP: {masked}"
            return cap, False, ""
        except Exception as e:
            return None, False, f"Lỗi kết nối RTSP ({masked}): {mask_rtsp_credentials(str(e))}"

    # Mặc định: VIDEO_TEST (file video cục bộ)
    if not raw_src:
        return None, True, "Chưa chọn đường dẫn file video kiểm thử"
    p = Path(raw_src)
    if not p.is_absolute():
        p = PROJECT_ROOT / raw_src
    if not p.exists() or not p.is_file():
        return None, True, f"Không tìm thấy file video: {raw_src}"

    try:
        cap = cv2.VideoCapture(str(p))
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None, True, f"Không thể giải mã file video: {raw_src}"
        return cap, True, ""
    except Exception as e:
        return None, True, f"Lỗi mở file video '{raw_src}': {e}"


class CameraStreamWorker:
    """Quản lý vòng đời đọc luồng video/camera, suy luận YOLO, tự kết nối lại và xác minh thời gian."""

    def __init__(
        self,
        config: CameraConfig,
        detector: SharedFireSmokeDetector,
        event_db: EventDatabase,
        settings: SystemSettings,
        log_callback=None,
        telegram_alerts: Optional[TelegramAlertDispatcher] = None,
    ):
        self.config = config
        self.detector = detector
        self.event_db = event_db
        self.telegram_alerts = telegram_alerts
        self.settings = settings
        self.log_callback = log_callback

        self.verifier = TemporalVerifier(
            window_size_n=settings.window_size_n,
            confirm_frames_k=settings.confirm_frames_k,
            suspect_frames_k=settings.suspect_frames_k,
            cooldown_sec=settings.alert_cooldown_sec,
            exclusion_zones=config.exclusion_zones,
            feature_threshold=getattr(settings, "feature_threshold", 0.0),
        )
        self.tracker = ByteTrackAdapter(
            settings=getattr(settings, "bytetrack", None),
            per_class=True,
        )
        self.features_extractor = TemporalFeatureExtractor(history_size=12)
        self.feature_extractor = self.features_extractor

        # Trạng thái chuẩn: "CONNECTED" | "DISCONNECTED" | "RECONNECTING" | "ERROR" | "PAUSED"
        # State Machine V2: "STOPPED" | "CONNECTING" | "RUNNING" | "RECONNECTING" | "ERROR" | "PAUSED"
        self.status: str = "DISCONNECTED"
        self.lifecycle_state: str = "STOPPED"
        self.error_detail: str = "Camera chưa được kết nối. Nhấn Bắt đầu giám sát để khởi động."
        self.running: bool = False
        self.paused: bool = False
        self.thread: Optional[threading.Thread] = None
        self.infer_thread: Optional[threading.Thread] = None
        self.infer_event = threading.Event()
        self.stop_event = threading.Event()
        self.lifecycle_lock = threading.RLock()
        self.lock = threading.Lock()
        self.pipeline_counters = {"detector_inference_calls": 0, "detector_output_detections": 0, "tracker_calls": 0, "tracker_input_detections": 0, "tracker_output_tracks": 0, "temporal_verifier_calls": 0, "temporal_input_tracks_detections": 0, "temporal_confirmation_outputs": 0, "event_evaluation_calls": 0, "confirmed_events": 0}

        # Chỉ số FPS đọc khung hình và FPS suy luận model riêng biệt
        self.capture_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.display_fps: float = 0.0
        self.infer_ms: float = 0.0
        self.end_to_end_latency_ms: float = 0.0
        self.inference_frames: int = 0
        self.dropped_frames: int = 0
        self.queue_depth: int = 0
        self._inference_ticks = deque(maxlen=50)
        self._display_ticks = deque(maxlen=100)
        self._displayed_by_client: Dict[str, int] = {}
        self.displayed_frames = 0
        self.last_update_ts: float = 0.0
        self.last_update_str: str = "--:--:--"
        self.disconnected_at_str: str = time.strftime("%H:%M:%S")
        self.frame_count: int = 0
        self.last_inferred_frame_id: int = 0
        self.reconnect_attempts: int = 0
        self.resolution_str: str = "N/A"
        self.latest_raw_bgr: Optional[np.ndarray] = None
        self.latest_capture_monotonic: float = 0.0

        # Trạng thái phát hiện hiện tại
        self.alert_level: str = "SAFE"  # "SAFE" | "SUSPECT" | "FIRE" | "CONN_ERROR"
        self.primary_label: str = ""
        self.max_confidence: float = 0.0
        self.active_hits: int = 0
        self.window_size: int = settings.window_size_n
        self.temporal_score: float = 0.0
        self.last_boxes: List[DetectionBox] = []

        init_rgb = create_status_placeholder_rgb(
            self.config.name, "DISCONNECTED", self.error_detail
        )
        self.latest_rgb_frame: np.ndarray = init_rgb
        self.latest_jpeg_b64: str = encode_rgb_to_jpeg_b64(init_rgb, quality=settings.jpeg_quality)
        self.preview_rgb_frame: Optional[np.ndarray] = None

    def _set_placeholder(self, status: str, detail: str):
        """Cập nhật cả khung hình RGB và chuỗi JPEG Base64 cho trạng thái thông báo."""
        if status != "CONNECTED":
            self.disconnected_at_str = time.strftime("%H:%M:%S")
        rgb = create_status_placeholder_rgb(self.config.name, status, detail)
        self.latest_rgb_frame = rgb
        self.latest_jpeg_b64 = encode_rgb_to_jpeg_b64(rgb, quality=self.settings.jpeg_quality)

    def _log(self, msg: str):
        safe_msg = mask_rtsp_credentials(msg)
        if self.log_callback:
            self.log_callback(f"[{self.config.cam_id}] {safe_msg}")

    def start(self):
        """Serialize lifecycle operations so manual reconnect cannot duplicate workers."""
        with self.lifecycle_lock:
            self._start()

    def _start(self, reconnecting: bool = False):
        """Khởi chạy kết nối camera (tách biệt luồng đọc khung hình và luồng suy luận YOLO)."""
        with self.lock:
            if getattr(self.detector, "model_loaded", True) is False:
                detail = getattr(self.detector, "error_message", "") or "A valid fire/smoke checkpoint is not loaded."
                self.running = False
                self.status = "ERROR"
                self.lifecycle_state = "ERROR"
                self.error_detail = str(detail)
                self._set_placeholder("ERROR", self.error_detail)
                self._log(f"Camera connect rejected: {self.error_detail}")
                return
            if self.config.source_type == "UNASSIGNED" or (
                self.config.source_type != "WEBCAM" and not str(self.config.source).strip()
            ):
                self.running = False
                self.paused = False
                self.status = "DISCONNECTED"
                self.lifecycle_state = "STOPPED"
                self.error_detail = "Chưa gán nguồn camera. Hãy chọn Camera đang hoạt động hoặc gắn File Video để test."
                self._set_placeholder("DISCONNECTED", self.error_detail)
                return
            if self.running and self.thread and self.thread.is_alive():
                self.paused = False
                if self.status == "PAUSED":
                    self.status = "CONNECTED"
                    self.lifecycle_state = "RUNNING"
                return
            if any(t and t.is_alive() for t in (self.thread, self.infer_thread)):
                raise RuntimeError("Previous camera threads have not stopped; refusing duplicate workers")
            self.stop_event.clear()
            self.running = True
            self.paused = False
            self.reconnect_attempts = 0
            self.frame_count = 0
            self.last_inferred_frame_id = 0
            self.inference_frames = 0
            self.dropped_frames = 0
            self.queue_depth = 0
            self.inference_fps = 0.0
            self.infer_ms = 0.0
            self.end_to_end_latency_ms = 0.0
            self._inference_ticks.clear()
            self.latest_raw_bgr = None
            self.status = "RECONNECTING" if reconnecting else "CONNECTING"
            self.lifecycle_state = self.status
            self.error_detail = f"Đang kết nối tới {self.config.masked_source}..."
            self._set_placeholder("RECONNECTING", self.error_detail)
            self.verifier.reset()
            self.tracker.reset()
            self.features_extractor.reset()
            self.infer_event.clear()
            self.thread = threading.Thread(
                target=self._run_loop, name=f"firewatch-capture-{self.config.cam_id}", daemon=True
            )
            self.infer_thread = threading.Thread(
                target=self._infer_loop, name=f"firewatch-inference-{self.config.cam_id}", daemon=True
            )
            self.thread.start()
            self.infer_thread.start()
        self._log(f"Đang kết nối nguồn [{self.config.source_type}]: {self.config.masked_source}")

    def pause(self):
        """Tạm dừng / tiếp tục luồng camera."""
        with self.lock:
            if self.running:
                self.paused = not self.paused
                self.status = "PAUSED" if self.paused else "CONNECTED"
                self.lifecycle_state = "PAUSED" if self.paused else "RUNNING"
                if self.paused:
                    self.capture_fps = 0.0
                    self.inference_fps = 0.0
                    self._set_placeholder("PAUSED", "Đã tạm dừng bởi người vận hành")
        self._log("Tạm dừng luồng" if self.paused else "Tiếp tục luồng")

    def stop(self):
        """Stop both threads before allowing the next generation to start."""
        with self.lifecycle_lock:
            self._stop()

    def _stop(self):
        """Ngắt kết nối hoàn toàn luồng camera và giải phóng VideoCapture, đưa về trạng thái STOPPED."""
        shutdown_started = time.monotonic()
        self.running = False
        self.stop_event.set()
        self.paused = False
        self.infer_event.set()
        t = self.thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.5)
        it = self.infer_thread
        if it and it.is_alive() and it is not threading.current_thread():
            it.join(timeout=1.0)
        alive_threads = [thread for thread in (t, it) if thread and thread.is_alive()]
        if alive_threads:
            LOGGER.error(
                "camera_shutdown_timeout camera_id=%s source_type=%s status=%s elapsed_seconds=%.3f "
                "alive_threads=%s",
                self.config.cam_id,
                self.config.source_type,
                self.status,
                time.monotonic() - shutdown_started,
                [f"{thread.name}(ident={thread.ident})" for thread in alive_threads],
            )
            raise RuntimeError("Camera shutdown timed out; retained thread handles for safe retry")
        self.thread = None
        self.infer_thread = None
        with self.lock:
            self.status = "DISCONNECTED"
            self.lifecycle_state = "STOPPED"
            self.capture_fps = 0.0
            self.inference_fps = 0.0
            self.infer_ms = 0.0
            self.latest_raw_bgr = None
            self.alert_level = "SAFE"
            self.primary_label = ""
            self.max_confidence = 0.0
            self.active_hits = 0
            self.error_detail = "Camera chưa được kết nối. Nhấn Bắt đầu giám sát để khởi động."
            self._set_placeholder("DISCONNECTED", self.error_detail)
            self.verifier.reset()
            self.tracker.reset()
            self.features_extractor.reset()
        self._log("Đã ngắt kết nối và giải phóng VideoCapture (STOPPED).")

    def reconnect(self):
        """Ngắt kết nối hiện tại và thực hiện kết nối lại ngay lập tức."""
        self._log(f"Kết nối lại thủ công ({self.config.masked_source})...")
        with self.lifecycle_lock:
            self.stop()
            self._start(reconnecting=True)

    def test_connection(self) -> Tuple[bool, str, np.ndarray]:
        """Kiểm tra kết nối thực tế tới nguồn camera và trả về hình ảnh xem trước (Preview)."""
        with self.lock:
            if self.running and self.status == "CONNECTED" and self.preview_rgb_frame is not None:
                msg = f"Kết nối thành công ({self.config.masked_source} | Độ phân giải: {self.resolution_str})"
                return True, msg, self.preview_rgb_frame

        cap, _, err = open_opencv_capture(
            source=self.config.source,
            source_type=self.config.source_type,
            webcam_index=self.config.webcam_index,
            capture_width=self.settings.capture_width,
        )
        if cap is None:
            err_frame = create_status_placeholder_rgb(self.config.name, "ERROR", err)
            with self.lock:
                if not self.running:
                    self.status = "ERROR"
                    self.error_detail = err
                    self.latest_rgb_frame = err_frame
                    self.latest_jpeg_b64 = encode_rgb_to_jpeg_b64(err_frame, quality=self.settings.jpeg_quality)
            self._log(f"Kiểm tra kết nối thất bại: {err}")
            return False, err, err_frame

        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                err_msg = f"Mở được nguồn {self.config.masked_source} nhưng không đọc được khung hình."
                err_frame = create_status_placeholder_rgb(self.config.name, "ERROR", err_msg)
                self._log(err_msg)
                return False, err_msg, err_frame

            h, w = frame.shape[:2]
            res_str = f"{w}x{h}"

            # Thu gọn ảnh trước khi suy luận thử để phản hồi tức thì
            max_dim = max(320, int(self.settings.capture_width))
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                proc = cv2.resize(frame, (max(160, int(w * scale)), max(160, int(h * scale))), interpolation=cv2.INTER_LINEAR)
            else:
                proc = frame.copy()

            boxes, infer_ms = self.detector.detect_frame(
                proc,
                conf_threshold=self.settings.conf_threshold,
                imgsz=self.settings.imgsz,
                non_blocking=False,
            )
            boxes = boxes or []
            annotated = SharedFireSmokeDetector.draw_annotations(
                frame=proc,
                boxes=boxes,
                cam_name=self.config.name,
                alert_level="FIRE" if any(b.label == "FIRE" for b in boxes) else "SAFE",
                fps=1000.0 / infer_ms if infer_ms > 0 else 0.0,
                exclusion_zones=self.config.exclusion_zones,
                inplace=True,
            )
            rgb_prev = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            jpeg_b64 = encode_bgr_to_jpeg_b64(annotated, quality=self.settings.jpeg_quality)
            with self.lock:
                self.preview_rgb_frame = rgb_prev
                self.resolution_str = res_str
                if not self.running:
                    self.latest_rgb_frame = rgb_prev
                    self.latest_jpeg_b64 = jpeg_b64
            ok_msg = (
                f"Kết nối thành công tới {self.config.masked_source} "
                f"(Độ phân giải: {res_str}, Phát hiện thử: {len(boxes)} đối tượng, {infer_ms:.1f} ms)."
            )
            self._log(ok_msg)
            return True, ok_msg, rgb_prev
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _infer_loop(self):
        """Luồng suy luận YOLO bất đồng bộ: chạy song song với luồng đọc video để video luôn đạt 24-30 FPS."""
        last_inference_at = 0.0
        while self.running:
            self.infer_event.wait(timeout=0.05)
            self.infer_event.clear()
            if not self.running:
                break
            if self.paused or self.status != "CONNECTED":
                time.sleep(0.05)
                continue

            with self.lock:
                raw_frame = self.latest_raw_bgr
                fid = self.frame_count
                captured_at = self.latest_capture_monotonic

            if raw_frame is None or fid <= self.last_inferred_frame_id:
                continue

            self.last_inferred_frame_id = fid
            min_interval = 1.0 / max(1, min(60, int(self.config.inference_fps)))
            now_perf = time.perf_counter()
            if now_perf - last_inference_at < min_interval:
                continue
            last_inference_at = now_perf
            infer_input = raw_frame.copy()

            self.verifier.update_params(
                window_size_n=self.settings.window_size_n,
                confirm_frames_k=self.settings.confirm_frames_k,
                suspect_frames_k=self.settings.suspect_frames_k,
                cooldown_sec=self.settings.alert_cooldown_sec,
                feature_threshold=getattr(self.settings, "feature_threshold", 0.0),
            )

            raw_boxes, new_infer_ms = self.detector.detect_frame(
                infer_input,
                conf_threshold=self.settings.conf_threshold,
                imgsz=self.settings.imgsz,
                non_blocking=True,
            )
            with self.lock:
                self.pipeline_counters["detector_inference_calls"] += 1
                self.pipeline_counters["detector_output_detections"] += len(raw_boxes or [])
            if raw_boxes is None or not self.running:
                continue
            processed_at = time.perf_counter()
            with self.lock:
                self.inference_frames += 1
                self._inference_ticks.append(processed_at)
                self.inference_fps = (
                    round((len(self._inference_ticks) - 1) / max(1e-6, self._inference_ticks[-1] - self._inference_ticks[0]), 2)
                    if len(self._inference_ticks) > 1 else 0.0
                )
                self.end_to_end_latency_ms = round(max(0.0, processed_at - captured_at) * 1000.0, 2)
                self.dropped_frames = max(0, self.frame_count - self.inference_frames)
                self.queue_depth = min(1, max(0, self.frame_count - fid))

            # 1. ByteTrack Association (gán track_id ổn định 2 giai đoạn)
            tracked_boxes = self.tracker.update(raw_boxes, frame=infer_input)
            with self.lock:
                self.pipeline_counters["tracker_calls"] += 1
                self.pipeline_counters["tracker_input_detections"] += len(raw_boxes)
                self.pipeline_counters["tracker_output_tracks"] += sum(1 for b in tracked_boxes if b.track_id > 0)

            # 2. Cập nhật Temporal Visual Features cho từng track
            now_ts = time.time()
            feature_map = {}
            for b in tracked_boxes:
                if b.track_id > 0:
                    try:
                        feat_res = self.features_extractor.update(
                            track_id=b.track_id,
                            frame_bgr=infer_input,
                            xyxy=b.xyxy,
                            timestamp_sec=now_ts,
                        )
                        feature_map[b.track_id] = feat_res
                    except Exception:
                        pass

            # 3. TemporalVerifier (K/N cửa sổ trượt, temporal features, exclusion zones, cooldown)
            ver_res = self.verifier.update(
                tracked_boxes,
                tracks_assigned=True,
                feature_map=feature_map,
                timestamp_sec=now_ts,
            )
            with self.lock:
                self.pipeline_counters["temporal_verifier_calls"] += 1
                self.pipeline_counters["temporal_input_tracks_detections"] += len(tracked_boxes)
                self.pipeline_counters["temporal_confirmation_outputs"] += len(ver_res.get("confirmed_track_ids", []))
                self.pipeline_counters["event_evaluation_calls"] += 1
            valid_boxes = ver_res["valid_boxes"]
            with self.lock:
                self.last_boxes = valid_boxes
                self.infer_ms = round(new_infer_ms, 1)
                if self.status == "CONNECTED":
                    self.alert_level = ver_res["alert_level"]
                self.primary_label = ver_res["primary_label"]
                self.max_confidence = ver_res["max_confidence"]
                self.active_hits = ver_res["active_hits"]
                self.window_size = ver_res["window_size"]
                self.temporal_score = round(float(ver_res.get("temporal_score", 0.0)), 3)
                curr_cap_fps = self.capture_fps

            if ver_res["should_trigger_db_alert"]:
                with self.lock:
                    self.pipeline_counters["confirmed_events"] += 1
                lbl = ver_res["primary_label"] or "FIRE"
                conf_val = ver_res["max_confidence"]
                snap_bgr = SharedFireSmokeDetector.draw_annotations(
                    frame=infer_input,
                    boxes=valid_boxes,
                    cam_name=f"{self.config.cam_id} ({self.config.name})",
                    alert_level=ver_res["alert_level"],
                    fps=curr_cap_fps,
                    exclusion_zones=self.config.exclusion_zones,
                    inplace=True,
                )
                saved_event = self.event_db.record_event(
                    frame=snap_bgr,
                    camera_name=self.config.cam_id,
                    label=lbl,
                    confidence=conf_val,
                    alert_level=ver_res["alert_level"],
                    is_demo=0,
                    event_confirmed_at=utc_now(),
                )
                if self.telegram_alerts is not None and self.config.telegram_enabled:
                    saved_event["camera_display_name"] = self.config.name
                    self.telegram_alerts.enqueue(saved_event)
                if not self.settings.mute_audio and self.config.cam_id not in self.event_db.alert_manager.acknowledged_cameras:
                    self.event_db.alert_manager.play_alarm(camera_name=self.config.cam_id)
                self._log(
                    f"🚨 CẢNH BÁO [{lbl}] ({int(conf_val*100)}%) tại {self.config.name} "
                    f"sau {ver_res['active_hits']}/{ver_res['window_size']} khung!"
                )

    def _run_loop(self):
        """Vòng lặp đọc frame tốc độ cao (24-30 FPS), hoàn toàn không bị chặn bởi thời gian suy luận YOLO."""
        cap: Optional[cv2.VideoCapture] = None
        is_file: bool = False
        capture_ticks = deque(maxlen=25)

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.08)
                    continue

                # 1. Mở kết nối (hoặc tự kết nối lại khi mất tín hiệu)
                if cap is None or not cap.isOpened():
                    cap, is_file, err_msg = open_opencv_capture(
                        source=self.config.source,
                        source_type=self.config.source_type,
                        webcam_index=self.config.webcam_index,
                        capture_width=self.settings.capture_width,
                    )
                    if cap is None:
                        if is_file:
                            # Nguồn file video: nếu lỗi hoặc không tìm thấy, dừng ngay về ERROR/STOPPED, không lặp tự tìm kết nối
                            with self.lock:
                                self.status = "ERROR"
                                self.lifecycle_state = "ERROR"
                                self.capture_fps = 0.0
                                self.inference_fps = 0.0
                                self.alert_level = "CONN_ERROR"
                                self.error_detail = err_msg or "Không thể đọc tệp video kiểm thử"
                                self._set_placeholder("ERROR", self.error_detail)
                                self.running = False
                            self._log(f"Lỗi đọc tệp video kiểm thử: {err_msg}")
                            break

                        self.reconnect_attempts += 1
                        max_att = max(1, int(self.config.max_reconnect_attempts))
                        if self.config.auto_reconnect and self.reconnect_attempts <= max_att:
                            with self.lock:
                                self.status = "RECONNECTING"
                                self.lifecycle_state = "RECONNECTING"
                                self.capture_fps = 0.0
                                self.inference_fps = 0.0
                                self.alert_level = "CONN_ERROR"
                                self.error_detail = (
                                    f"Mất kết nối. Đang thử lại ({self.reconnect_attempts}/{max_att}): {err_msg}"
                                )
                                self._set_placeholder(
                                    "RECONNECTING",
                                    f"Attempt {self.reconnect_attempts}/{max_att}: {err_msg}",
                                )
                            self._log(
                                f"Thử kết nối lại lần {self.reconnect_attempts}/{max_att} sau "
                                f"{self.config.reconnect_interval_sec}s ({err_msg})"
                            )
                            self.stop_event.wait(self.config.reconnect_interval_sec)
                            continue
                        else:
                            with self.lock:
                                self.status = "ERROR"
                                self.lifecycle_state = "ERROR"
                                self.capture_fps = 0.0
                                self.inference_fps = 0.0
                                self.alert_level = "CONN_ERROR"
                                self.error_detail = err_msg
                                self._set_placeholder("ERROR", err_msg)
                                self.running = False
                            self._log(f"Ngừng kết nối do lỗi: {err_msg}")
                            break

                    self.reconnect_attempts = 0
                    self.verifier.reset()
                    if hasattr(self, "tracker") and self.tracker:
                        self.tracker.reset()
                    if hasattr(self, "feature_extractor") and self.feature_extractor:
                        self.feature_extractor.reset()
                    with self.lock:
                        self.status = "CONNECTED"
                        self.lifecycle_state = "RUNNING"
                        self.error_detail = ""
                        if self.alert_level == "CONN_ERROR":
                            self.alert_level = "SAFE"
                    self._log(f"Đã kết nối thành công ({self.config.masked_source}).")

                # 2. Đọc khung hình từ VideoCapture
                loop_start = time.perf_counter()
                ret, frame = cap.read()

                if not ret or frame is None:
                    if is_file and self.config.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()

                    if not ret or frame is None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None

                        if self.config.auto_reconnect and not is_file:
                            self.verifier.reset()
                            with self.lock:
                                self.status = "RECONNECTING"
                                self.capture_fps = 0.0
                                self.inference_fps = 0.0
                                self.alert_level = "CONN_ERROR"
                                self.error_detail = "Mất tín hiệu từ camera, đang tự động kết nối lại..."
                                self._set_placeholder("RECONNECTING", self.error_detail)
                            self._log("Mất tín hiệu hình ảnh. Chuyển sang trạng thái RECONNECTING...")
                            self.stop_event.wait(1.0)
                            continue
                        else:
                            with self.lock:
                                self.status = "DISCONNECTED"
                                self.capture_fps = 0.0
                                self.inference_fps = 0.0
                                self.alert_level = "CONN_ERROR"
                                self.error_detail = "Mất tín hiệu hoặc đã kết thúc video"
                                self._set_placeholder("DISCONNECTED", self.error_detail)
                                self.running = False
                            self._log("Mất tín hiệu hoặc kết thúc luồng video.")
                            break

                next_frame_count = self.frame_count + 1
                captured_at = time.perf_counter()
                fh, fw = frame.shape[:2]
                if next_frame_count == 1 or next_frame_count % 30 == 0:
                    self.resolution_str = f"{fw}x{fh}"

                # Thu phóng khung hình xử lý cực nhanh bằng INTER_LINEAR, giới hạn cạnh dài nhất theo capture_width
                cap_max = max(320, int(self.settings.capture_width))
                longest = max(fw, fh)
                if longest > cap_max:
                    scale = cap_max / float(longest)
                    new_w = max(160, int(fw * scale))
                    new_h = max(120, int(fh * scale))
                    proc_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    proc_frame = frame.copy()

                # Gửi khung hình mới cho luồng _infer_loop theo chu kỳ frame_skip mà không chặn luồng hiển thị
                skip_n = max(1, int(self.settings.frame_skip))
                with self.lock:
                    self.frame_count = next_frame_count
                    self.latest_raw_bgr = proc_frame
                    self.latest_capture_monotonic = captured_at
                    self.queue_depth = min(1, max(0, self.frame_count - self.last_inferred_frame_id))
                    valid_boxes = list(self.last_boxes)
                    curr_alert_level = self.alert_level if self.alert_level != "CONN_ERROR" else "SAFE"

                if (next_frame_count % skip_n == 0) or (next_frame_count == 1):
                    self.infer_event.set()

                # Tính FPS đọc khung hình (capture_fps)
                now_perf = time.perf_counter()
                capture_ticks.append(now_perf)
                actual_capture_fps = (
                    (len(capture_ticks) - 1) / max(1e-6, capture_ticks[-1] - capture_ticks[0])
                    if len(capture_ticks) > 1 else 0.0
                )

                # Vẽ bounding box mới nhất lên bản sao hiển thị
                disp_canvas = proc_frame.copy()
                annotated_bgr = SharedFireSmokeDetector.draw_annotations(
                    frame=disp_canvas,
                    boxes=valid_boxes,
                    cam_name=f"{self.config.cam_id} ({self.config.name})",
                    alert_level=curr_alert_level,
                    fps=actual_capture_fps,
                    exclusion_zones=self.config.exclusion_zones,
                    inplace=True,
                )

                disp_w = max(320, int(self.settings.display_width))
                ah, aw = annotated_bgr.shape[:2]
                if aw > disp_w:
                    disp_h = max(180, int(ah * (disp_w / float(aw))))
                    disp_bgr = cv2.resize(annotated_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                else:
                    disp_bgr = annotated_bgr

                jpeg_b64 = encode_bgr_to_jpeg_b64(disp_bgr, quality=self.settings.jpeg_quality)
                rgb_img = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
                now_str = time.strftime("%H:%M:%S")

                with self.lock:
                    self.status = "CONNECTED"
                    self.lifecycle_state = "RUNNING"
                    self.capture_fps = round(actual_capture_fps, 1)
                    self.last_update_ts = time.time()
                    self.last_update_str = now_str
                    self.latest_rgb_frame = rgb_img
                    self.latest_jpeg_b64 = jpeg_b64
                    self.preview_rgb_frame = rgb_img

                # Điều tiết tốc độ đọc theo target_fps_per_cam (tránh Windows timer 15.6ms penalty khi đã đủ thời gian)
                target_fps = max(1, int(self.settings.target_fps_per_cam))
                target_period = 1.0 / float(target_fps)
                elapsed = time.perf_counter() - loop_start
                remain = target_period - elapsed
                if remain > 0.002:
                    time.sleep(remain)
                else:
                    time.sleep(0.0005)

        except Exception as e:
            safe_err = mask_rtsp_credentials(str(e))[:50]
            with self.lock:
                self.status = "ERROR"
                self.capture_fps = 0.0
                self.inference_fps = 0.0
                self.alert_level = "CONN_ERROR"
                self.error_detail = f"Lỗi luồng: {safe_err}"
                self._set_placeholder("ERROR", self.error_detail)
                self.running = False
            self._log(f"Ngoại lệ trong luồng camera: {safe_err}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def get_snapshot(self) -> Dict[str, Any]:
        """Lấy bản sao trạng thái hiện tại của camera một cách an toàn đa luồng (đã ẩn mật khẩu RTSP)."""
        with self.lock:
            return {
                "cam_id": self.config.cam_id,
                "name": self.config.name,
                "source_type": self.config.source_type,
                "masked_source": self.config.masked_source,
                "webcam_index": self.config.webcam_index,
                "enabled": self.config.enabled,
                "telegram_enabled": self.config.telegram_enabled,
                "inference_target_fps": self.config.inference_fps,
                "auto_reconnect": self.config.auto_reconnect,
                "status": self.status,  # "CONNECTED" | "DISCONNECTED" | "RECONNECTING" | "ERROR" | "PAUSED"
                "lifecycle_state": self.lifecycle_state,  # "STOPPED" | "CONNECTING" | "RUNNING" | "RECONNECTING" | "ERROR" | "PAUSED"
                "is_connected": (self.status == "CONNECTED"),
                "error_detail": mask_rtsp_credentials(self.error_detail),
                "fps": self.capture_fps,
                "capture_fps": self.capture_fps,
                "inference_fps": self.inference_fps,
                "display_fps": self.display_fps,
                "infer_ms": self.infer_ms,
                "end_to_end_latency_ms": self.end_to_end_latency_ms,
                "captured_frames": self.frame_count,
                "processed_frames": self.inference_frames,
                "dropped_frames": self.dropped_frames,
                "queue_depth": self.queue_depth,
                "displayed_frames": self.displayed_frames,
                "resolution": self.resolution_str,
                "last_update_str": self.last_update_str,
                "disconnected_at_str": self.disconnected_at_str,
                "alert_level": self.alert_level,
                "primary_label": self.primary_label,
                "max_confidence": self.max_confidence,
                "active_hits": self.active_hits,
                "window_size": self.window_size,
                "confirm_k": self.settings.confirm_frames_k,
                "frame_rgb": self.latest_rgb_frame,
                "jpeg_b64": self.latest_jpeg_b64,
                "preview_rgb": self.preview_rgb_frame,
                "box_count": len(self.last_boxes),
                "boxes": [
                    {
                        "label": b.label,
                        "confidence": round(float(b.confidence), 3),
                        "track_id": int(b.track_id),
                        "xyxy": tuple(b.xyxy),
                    }
                    for b in self.last_boxes
                ],
                "temporal_score": self.temporal_score,
                "exclusion_zones": list(self.config.exclusion_zones),
                "reconnect_attempts": self.reconnect_attempts,
            }

    def record_displayed_frame(self, client_id: str, frame_id: int) -> None:
        """Record client display acknowledgment without doing image encoding."""
        with self.lock:
            previous = self._displayed_by_client.get(client_id, 0)
            if frame_id > previous:
                self._displayed_by_client[client_id] = frame_id
                self._display_ticks.append(time.perf_counter())
                self.displayed_frames += 1
                if len(self._display_ticks) > 1:
                    elapsed = self._display_ticks[-1] - self._display_ticks[0]
                    self.display_fps = round((len(self._display_ticks) - 1) / max(1e-6, elapsed), 2)

    def remove_display_client(self, client_id: str) -> None:
        """Release per-client display accounting after a WebSocket closes."""
        with self.lock:
            self._displayed_by_client.pop(client_id, None)


class MultiCameraManager:
    """Bộ điều phối trung tâm quản lý 9 luồng camera (bố cục 3x3), Detector dùng chung và SQLite."""

    def __init__(
        self,
        db_path: str = "alerts.db",
        model_path: str = "best.pt",
        auto_assign_videos: bool = True,
        detector: Optional[SharedFireSmokeDetector] = None,
    ):
        self.start_time = time.time()
        self.performance_preset = "Balanced"
        self.default_camera_ids: List[str] = ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]
        self.settings = SystemSettings(model_path=model_path)
        self.alert_manager = AlertManager(db_path=db_path)
        self.event_db = EventDatabase(alert_manager=self.alert_manager, db_path=db_path)
        self.telegram_alerts = TelegramAlertDispatcher(db_path=db_path)
        self.detector = detector or SharedFireSmokeDetector(
            model_path=self.settings.model_path,
            conf_threshold=self.settings.conf_threshold,
            imgsz=self.settings.imgsz,
        )

        self.logs: deque = deque(maxlen=100)
        self.log_lock = threading.Lock()

        self.camera_configs: List[CameraConfig] = get_default_9_cameras(
            auto_assign_videos=auto_assign_videos
        )
        self.camera_restore_errors: List[Dict[str, str]] = []
        self._initialize_camera_config_table()
        self._restore_camera_configs()
        self.workers: Dict[str, CameraStreamWorker] = {}
        for cfg in self.camera_configs:
            self.workers[cfg.cam_id] = CameraStreamWorker(
                config=cfg,
                detector=self.detector,
                event_db=self.event_db,
                settings=self.settings,
                log_callback=self.add_log,
                telegram_alerts=self.telegram_alerts,
            )

        self.add_log("Khởi tạo trung tâm điều hành 9 Camera (3x3 Grid) thành công.")
        model_info = self.detector.get_model_info()
        if model_info["loaded"]:
            self.add_log(
                f"Đã nạp model YOLO: {model_info['path']} trên {model_info.get('device', 'CPU')} | Lớp: {model_info['names']}"
            )
        else:
            self.add_log(f"⚠️ CẢNH BÁO MODEL: {model_info['error']}")

    def _initialize_camera_config_table(self) -> None:
        """Create the additive camera configuration table without touching event history."""
        with closing(sqlite3.connect(self.event_db.db_path, timeout=10)) as conn, conn:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY, name TEXT NOT NULL, source_type TEXT NOT NULL,
                source TEXT NOT NULL, webcam_index INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1, auto_reconnect INTEGER NOT NULL DEFAULT 1,
                inference_fps INTEGER NOT NULL DEFAULT 8, exclusion_zones TEXT NOT NULL DEFAULT '[]',
                telegram_enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def _resolve_persisted_source(self, source_type: str, source_reference: str, webcam_index: int) -> str:
        """Resolve a safe source reference only in memory when restoring a camera."""
        if source_type == "WEBCAM":
            return str(webcam_index)
        if source_type == "IP_CAMERA" and source_reference.startswith("env:"):
            variable = source_reference[4:]
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", variable):
                raise ValueError("Invalid environment reference (expected env:VARIABLE_NAME)")
            resolved = os.getenv(variable, "").strip()
            if not resolved:
                raise ValueError(f"Environment reference {variable} is not configured")
            parsed = urlsplit(resolved)
            allowed = {host.strip().lower() for host in os.getenv("FIREWATCH_RTSP_HOSTS", "").split(",") if host.strip()}
            if parsed.scheme not in ("rtsp", "rtsps") or not parsed.hostname or parsed.hostname.lower() not in allowed:
                raise ValueError("Referenced RTSP URL is invalid or its hostname is not allowlisted")
            return resolved
        if source_type == "IP_CAMERA":
            parsed = urlsplit(source_reference)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("Inline RTSP credentials are forbidden; use env:VARIABLE_NAME")
            allowed = {host.strip().lower() for host in os.getenv("FIREWATCH_RTSP_HOSTS", "").split(",") if host.strip()}
            if parsed.scheme not in ("rtsp", "rtsps") or not parsed.hostname:
                raise ValueError("IP camera source must be an RTSP or RTSPS URL")
            if parsed.hostname.lower() not in allowed:
                raise ValueError("RTSP host is not present in FIREWATCH_RTSP_HOSTS")
        return source_reference

    def _restore_camera_configs(self) -> None:
        """Restore persisted camera settings while leaving all capture workers stopped."""
        with closing(sqlite3.connect(self.event_db.db_path, timeout=10)) as conn, conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(cameras)")}
            if "telegram_enabled" not in columns:
                conn.execute("ALTER TABLE cameras ADD COLUMN telegram_enabled INTEGER NOT NULL DEFAULT 1")
            rows = conn.execute("""SELECT camera_id,name,source_type,source,webcam_index,enabled,
                auto_reconnect,inference_fps,exclusion_zones,telegram_enabled FROM cameras ORDER BY camera_id""").fetchall()
        by_id = {cfg.cam_id: cfg for cfg in self.camera_configs}
        for row in rows:
            camera_id, name, source_type, source_ref, device, enabled, reconnect, fps, zones_json, telegram_enabled = row
            try:
                if not re.fullmatch(r"CAM-\d{2,}", str(camera_id)):
                    raise ValueError("Invalid camera id; expected CAM- followed by at least two digits")
                zones = json.loads(zones_json or "[]")
                if not isinstance(zones, list):
                    raise ValueError("Exclusion zones must be a list")
                resolved = self._resolve_persisted_source(source_type, source_ref, int(device))
                cfg = by_id.get(camera_id)
                if cfg is None:
                    cfg = CameraConfig(cam_id=camera_id, name=name, source=resolved, source_reference=source_ref,
                                       source_type=source_type, webcam_index=int(device), enabled=bool(enabled),
                                       inference_fps=int(fps), exclusion_zones=[tuple(z) for z in zones],
                                       telegram_enabled=bool(telegram_enabled))
                    self.camera_configs.append(cfg)
                    by_id[camera_id] = cfg
                else:
                    cfg.name = name
                    cfg.source = resolved
                    cfg.source_reference = source_ref
                    cfg.source_type = source_type
                    cfg.webcam_index = int(device)
                    cfg.enabled = bool(enabled)
                    cfg.auto_reconnect = bool(reconnect)
                    cfg.inference_fps = max(1, min(60, int(fps)))
                    cfg.exclusion_zones = [tuple(z) for z in zones]
                    cfg.telegram_enabled = bool(telegram_enabled)
            except Exception as exc:
                safe_error = mask_rtsp_credentials(str(exc))
                self.camera_restore_errors.append({"camera_id": str(camera_id), "error": safe_error})
                self.add_log(f"Camera configuration {camera_id} restore failed: {safe_error}")

    def _persist_camera_config(self, cfg: CameraConfig) -> bool:
        """Persist only the safe source reference, never a resolved credential URL."""
        source_ref = cfg.source_reference or cfg.source
        if cfg.source_type == "IP_CAMERA":
            parsed = urlsplit(source_ref) if not source_ref.startswith("env:") else None
            if parsed and (parsed.username is not None or parsed.password is not None):
                self.add_log(f"{cfg.cam_id} settings were not persisted: authenticated RTSP sources require an environment reference.")
                return False
        try:
            with closing(sqlite3.connect(self.event_db.db_path, timeout=10)) as conn, conn:
                conn.execute("PRAGMA busy_timeout=10000")
                conn.execute("""INSERT INTO cameras(camera_id,name,source_type,source,webcam_index,enabled,
                    auto_reconnect,inference_fps,exclusion_zones,telegram_enabled,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                    ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name,source_type=excluded.source_type,
                    source=excluded.source,webcam_index=excluded.webcam_index,enabled=excluded.enabled,
                    auto_reconnect=excluded.auto_reconnect,inference_fps=excluded.inference_fps,
                    exclusion_zones=excluded.exclusion_zones,telegram_enabled=excluded.telegram_enabled,
                    updated_at=CURRENT_TIMESTAMP""",
                    (cfg.cam_id,cfg.name,cfg.source_type,source_ref,cfg.webcam_index,int(cfg.enabled),
                     int(cfg.auto_reconnect),cfg.inference_fps,json.dumps(cfg.exclusion_zones),
                     int(cfg.telegram_enabled)))
            return True
        except sqlite3.Error as exc:
            self.add_log(f"Camera configuration {cfg.cam_id} could not be persisted ({type(exc).__name__}).")
            return False

    def add_log(self, message: str):
        """Ghi nhật ký hệ thống, tự động che mật khẩu RTSP nếu có."""
        ts = time.strftime("%H:%M:%S")
        safe_message = mask_rtsp_credentials(message)
        with self.log_lock:
            self.logs.appendleft(f"[{ts}] {safe_message}")

    def get_recent_logs(self, limit: int = 25) -> List[str]:
        with self.log_lock:
            return list(self.logs)[:limit]

    def switch_model(self, model_path: str) -> Dict[str, Any]:
        """Chuyển đổi trực tiếp mô hình YOLO đang sử dụng cho toàn bộ 9 luồng camera."""
        self.settings.model_path = model_path
        ok = self.detector.load_model(model_path)
        info = self.detector.get_model_info()
        if ok:
            self.add_log(
                f"Đã chuyển sang model: {info['path']} trên {info.get('device', 'CPU')}"
            )
        else:
            self.add_log(f"⚠️ Lỗi khi nạp model '{model_path}': {info.get('error')}")
        return info

    def set_camera_enabled(self, cam_id: str, enabled: bool):
        """Bật hoặc tắt quyền kích hoạt của một camera độc lập."""
        if cam_id in self.workers:
            w = self.workers[cam_id]
            w.config.enabled = bool(enabled)
            if not enabled and w.running:
                w.stop()
            self._persist_camera_config(w.config)
            self.add_log(f"Đã chuyển trạng thái kích hoạt {cam_id} -> {'BẬT' if enabled else 'TẮT'}.")

    def set_camera_telegram_enabled(self, cam_id: str, enabled: bool) -> None:
        """Set the per-camera notification policy without changing detection."""
        if cam_id not in self.workers:
            raise ValueError(f"Unknown camera id: {cam_id}")
        config = self.workers[cam_id].config
        config.telegram_enabled = bool(enabled)
        self._persist_camera_config(config)

    def update_camera_inference_fps(self, cam_id: str, inference_fps: int) -> None:
        """Persist the per-camera inference scheduling cap."""
        if cam_id not in self.workers:
            raise ValueError(f"Unknown camera id: {cam_id}")
        fps = int(inference_fps)
        if not 1 <= fps <= 60:
            raise ValueError("inference_fps must be between 1 and 60")
        worker = self.workers[cam_id]
        worker.config.inference_fps = fps
        self._persist_camera_config(worker.config)

    def start_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].config.enabled = True
            self._persist_camera_config(self.workers[cam_id].config)
            self.workers[cam_id].start()

    def pause_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].pause()

    def stop_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].stop()

    def reconnect_camera(self, cam_id: str):
        if cam_id in self.workers:
            self.alert_manager.reset_alarm_acknowledgment(cam_id)
            self.workers[cam_id].reconnect()

    def test_camera_connection(self, cam_id: str) -> Tuple[bool, str, np.ndarray]:
        """Kiểm tra kết nối thực tế và trả về ảnh xem trước cho một camera."""
        if cam_id not in self.workers:
            empty = create_status_placeholder_rgb(cam_id, "ERROR", "Camera ID không hợp lệ")
            return False, "Camera ID không hợp lệ", empty
        return self.workers[cam_id].test_connection()

    def start_all(self, only_enabled: bool = False):
        """Khởi chạy đồng loạt các camera (hoặc chỉ các camera đang được bật enabled)."""
        for w in self.workers.values():
            if not only_enabled or w.config.enabled:
                w.config.enabled = True
                w.start()

    def pause_all(self):
        """Tạm dừng đồng loạt các camera đang chạy."""
        for w in self.workers.values():
            if w.running and not w.paused:
                w.pause()

    def stop_all(self):
        """Ngắt kết nối toàn bộ 9 camera và tắt còi báo động."""
        for w in self.workers.values():
            w.stop()
        self.alert_manager.stop_alarm()

    def reconnect_all(self):
        """Kết nối lại tất cả các camera đang được bật (enabled)."""
        for w in self.workers.values():
            if w.config.enabled or w.status in ("CONNECTED", "RECONNECTING", "ERROR"):
                w.reconnect()

    def update_camera_source(
        self,
        cam_id: str,
        new_name: str,
        new_source: str,
        source_type: Optional[str] = None,
        webcam_index: int = 0,
        enabled: Optional[bool] = None,
        auto_reconnect: Optional[bool] = None,
        auto_restart: bool = False,
        source_reference: Optional[str] = None,
    ):
        """Lưu cấu hình mới cho một camera (che mật khẩu RTSP trong log)."""
        if cam_id not in self.workers:
            return
        w = self.workers[cam_id]
        was_running = w.running
        w.stop()

        w.config.name = new_name.strip() or cam_id
        if source_type in ("VIDEO_TEST", "WEBCAM", "IP_CAMERA"):
            w.config.source_type = source_type
        else:
            w.config.source_type = infer_source_type(new_source)

        safe_reference = str(source_reference if source_reference is not None else new_source).strip()
        if w.config.source_type == "WEBCAM":
            w.config.webcam_index = int(webcam_index)
            w.config.source = str(webcam_index)
            w.config.source_reference = str(webcam_index)
        else:
            w.config.source = self._resolve_persisted_source(w.config.source_type, safe_reference, int(webcam_index))
            w.config.source_reference = safe_reference

        if enabled is not None:
            w.config.enabled = bool(enabled)
        if auto_reconnect is not None:
            w.config.auto_reconnect = bool(auto_reconnect)

        with w.lock:
            w._set_placeholder("DISCONNECTED", f"Đã lưu cấu hình: {w.config.masked_source}")
        self.add_log(
            f"Đã lưu cấu hình {cam_id} [{w.config.source_type}]: '{w.config.name}' -> '{w.config.masked_source}'"
        )
        self._persist_camera_config(w.config)
        if auto_restart:
            w.start()
        else:
            with w.lock:
                w.status = "DISCONNECTED"
                w.lifecycle_state = "STOPPED"
                w.error_detail = (
                    f"Đã gán nguồn: {w.config.masked_source}. Nhấn Bắt đầu giám sát để khởi động."
                    if w.config.source_type != "UNASSIGNED" and bool(str(w.config.source).strip())
                    else "Chưa gán nguồn camera. Hãy chọn Camera đang hoạt động hoặc gắn File Video để test."
                )
                w._set_placeholder("DISCONNECTED", w.error_detail)

    def apply_operating_mode_preset(
        self,
        mode: str,
        primary_source: str = "",
        webcam_index: int = 0,
        auto_connect: bool = True,
    ):
        """Chuyển đổi nhanh chế độ hoạt động (`VIDEO_TEST`, `WEBCAM`, `IP_CAMERA`, `MULTI_CAMERA`)."""
        self.settings.operating_mode = mode
        if mode == "VIDEO_TEST":
            src = primary_source.strip() or "videovietnam/VN_POS_01.mp4"
            self.update_camera_source(
                "CAM-01",
                "CAM 01 - Video Kiểm Thử",
                src,
                source_type="VIDEO_TEST",
                enabled=True,
                auto_restart=auto_connect,
            )
        elif mode == "WEBCAM":
            self.update_camera_source(
                "CAM-01",
                f"CAM 01 - Webcam Máy Tính #{webcam_index}",
                str(webcam_index),
                source_type="WEBCAM",
                webcam_index=webcam_index,
                enabled=True,
                auto_restart=auto_connect,
            )
        elif mode == "IP_CAMERA":
            if primary_source.strip():
                self.update_camera_source(
                    "CAM-01",
                    "CAM 01 - Camera Giám Sát IP/RTSP",
                    primary_source.strip(),
                    source_type="IP_CAMERA",
                    enabled=True,
                    auto_restart=auto_connect,
                )
        self.add_log(f"Đã chuyển chế độ hoạt động hệ thống sang: {mode}")

    def set_active_camera_selection(self, selected_cam_ids: List[str], auto_start: bool = True):
        """Áp dụng danh sách camera được chọn (Dynamic Camera Selection: 1, 2, 4, 6, 9 hoặc custom).

        Quy tắc quản lý nghiêm ngặt:
        1. Camera không được chọn: dừng worker, giải phóng VideoCapture, không tốn GPU/CPU.
        2. Camera được thêm: khởi tạo và chạy worker.
        3. Camera giữ nguyên: tiếp tục chạy bình thường, KHÔNG reset tracker hay verifier.
        """
        target_ids = set(selected_cam_ids)
        newly_started: List[CameraStreamWorker] = []
        for cfg in self.camera_configs:
            cid = cfg.cam_id
            w = self.workers[cid]
            if cid in target_ids:
                cfg.enabled = True
                w.config.enabled = True
                # Chỉ khởi chạy khi camera đã có nguồn hợp lệ và không ở trạng thái ERROR
                has_source = (w.config.source_type != "UNASSIGNED") and (
                    w.config.source_type == "WEBCAM" or bool(str(w.config.source).strip())
                )
                if auto_start and has_source and not w.running and w.status != "ERROR":
                    w.start()
                    newly_started.append(w)
            else:
                cfg.enabled = False
                w.config.enabled = False
                if w.running:
                    w.stop()
            self._persist_camera_config(cfg)

        # Khởi chạy bất đồng bộ hoàn toàn không block luồng Streamlit (0ms delay)
        self.add_log(f"Đã cập nhật danh sách camera giám sát: {sorted(list(target_ids))}")

    def apply_performance_preset(self, preset: str):
        """Áp dụng chế độ hiệu năng thực tế (Low Resource / Balanced / High Performance)."""
        mode = str(preset).strip()
        if mode == "Low Resource":
            self.settings.imgsz = 320
            self.settings.capture_width = 380
            self.settings.display_width = 360
            self.settings.frame_skip = 3
            self.settings.target_fps_per_cam = 15
            self.settings.jpeg_quality = 65
            self.settings.ui_refresh_sec = 0.5
        elif mode == "High Performance":
            self.settings.imgsz = 640
            self.settings.capture_width = 640
            self.settings.display_width = 540
            self.settings.frame_skip = 1
            self.settings.target_fps_per_cam = 30
            self.settings.jpeg_quality = 82
            self.settings.ui_refresh_sec = 0.3
        else:
            mode = "Balanced"
            self.settings.imgsz = 416
            self.settings.capture_width = 480
            self.settings.display_width = 420
            self.settings.frame_skip = 2
            self.settings.target_fps_per_cam = 24
            self.settings.jpeg_quality = 72
            self.settings.ui_refresh_sec = 0.4
        self.performance_preset = mode
        self.add_log(
            f"Đã áp dụng chế độ hiệu năng '{mode}' (imgsz={self.settings.imgsz}, skip={self.settings.frame_skip}, target_fps={self.settings.target_fps_per_cam})"
        )

    def add_camera(
        self,
        name: str,
        source: str,
        source_type: Optional[str] = None,
        webcam_index: int = 0,
        enabled: bool = True,
        auto_reconnect: bool = True,
        exclusion_zones: Optional[List[Tuple[float, float, float, float]]] = None,
        auto_start: bool = False,
        inference_fps: int = 24,
        camera_id: Optional[str] = None,
        source_reference: Optional[str] = None,
        telegram_enabled: bool = True,
    ) -> str:
        """Thêm một camera mới vào hệ thống và khởi tạo CameraStreamWorker độc lập."""
        if camera_id is not None:
            if not re.fullmatch(r"CAM-\d{2,}", str(camera_id)):
                raise ValueError("camera_id must match CAM- followed by at least two digits")
            if camera_id in self.workers:
                raise ValueError(f"Camera id already exists: {camera_id}")
            cam_id = str(camera_id)
        else:
            next_num = 1
            while f"CAM-{next_num:02d}" in self.workers:
                next_num += 1
            cam_id = f"CAM-{next_num:02d}"
        s_type = source_type if source_type in ("VIDEO_TEST", "WEBCAM", "IP_CAMERA") else infer_source_type(source)
        safe_reference = str(source_reference if source_reference is not None else source).strip()
        src_str = str(webcam_index) if s_type == "WEBCAM" else self._resolve_persisted_source(s_type, safe_reference, webcam_index)

        cfg = CameraConfig(
            cam_id=cam_id,
            name=name.strip() or cam_id,
            source=src_str,
            source_reference=str(webcam_index) if s_type == "WEBCAM" else safe_reference,
            telegram_enabled=bool(telegram_enabled),
            source_type=s_type,
            webcam_index=int(webcam_index) if s_type == "WEBCAM" else 0,
            enabled=bool(enabled),
            inference_fps=max(1, min(60, int(inference_fps))),
            loop_video=True,
            auto_reconnect=bool(auto_reconnect),
            exclusion_zones=list(exclusion_zones) if exclusion_zones else [],
        )
        self.camera_configs.append(cfg)
        worker = CameraStreamWorker(
            config=cfg,
            detector=self.detector,
            event_db=self.event_db,
            settings=self.settings,
            log_callback=self.add_log,
            telegram_alerts=self.telegram_alerts,
        )
        self.workers[cam_id] = worker
        self._persist_camera_config(cfg)
        self.add_log(f"Đã thêm camera mới {cam_id} ({cfg.name}) -> {cfg.masked_source}")
        if auto_start and cfg.enabled:
            worker.start()
        return cam_id

    def remove_camera(self, cam_id: str) -> bool:
        """Xóa cấu hình một camera theo xác nhận của người vận hành (giữ tối thiểu 1 camera)."""
        if cam_id not in self.workers or len(self.camera_configs) <= 1:
            return False
        w = self.workers.pop(cam_id)
        if w.running:
            w.stop()
        self.camera_configs = [c for c in self.camera_configs if c.cam_id != cam_id]
        with closing(sqlite3.connect(self.event_db.db_path, timeout=10)) as conn, conn:
            conn.execute("DELETE FROM cameras WHERE camera_id=?", (cam_id,))
        if cam_id in self.default_camera_ids:
            self.default_camera_ids = [c for c in self.default_camera_ids if c != cam_id]
            if not self.default_camera_ids and self.camera_configs:
                self.default_camera_ids = [self.camera_configs[0].cam_id]
        self.add_log(f"Đã xóa cấu hình camera {cam_id}.")
        return True

    def start_monitoring(self, selected_cam_ids: List[str]) -> Dict[str, Any]:
        """Bắt đầu giám sát thủ công (MANUAL START): chỉ khởi chạy các camera đã được gán nguồn hợp lệ trong danh sách chọn."""
        self.alert_manager.reset_alarm_acknowledgment()
        target_ids = [cid for cid in selected_cam_ids if cid in self.workers]
        assigned_ids: List[str] = []
        unassigned_ids: List[str] = []
        for cid in target_ids:
            w = self.workers[cid]
            has_src = w.config.source_type != "UNASSIGNED" and (
                w.config.source_type == "WEBCAM" or bool(str(w.config.source).strip())
            )
            if not has_src:
                unassigned_ids.append(cid)
                continue
            assigned_ids.append(cid)
            # Nếu camera trước đó bị ERROR, cho phép thử kết nối lại khi người dùng chủ động nhấn Bắt đầu giám sát
            if w.status == "ERROR" and not w.running:
                w.status = "DISCONNECTED"
                w.lifecycle_state = "STOPPED"
        # Connecting a chosen camera must not stop already-running cameras that
        # are hidden by the current dashboard layout or another browser session.
        for cid in assigned_ids:
            worker = self.workers[cid]
            worker.config.enabled = True
            self._persist_camera_config(worker.config)
            if worker.status == "ERROR" and not worker.running:
                worker.status = "DISCONNECTED"
                worker.lifecycle_state = "STOPPED"
            worker.start()
        snaps = self.get_selected_snapshots(target_ids)
        return {
            "requested": target_ids,
            "assigned": assigned_ids,
            "unassigned": unassigned_ids,
            "connected": [s["cam_id"] for s in snaps if s["is_connected"]],
            "failed": [s["cam_id"] for s in snaps if s["status"] == "ERROR"],
        }

    def detect_available_cameras(self, max_check: int = 4) -> Dict[str, Any]:
        """Tự động dò quét các camera phần cứng đang hoạt động trên máy tính và các luồng RTSP/Video đã gán."""
        sys_cams = list_system_cameras(max_check=max_check)
        active_hw = [c for c in sys_cams if c.get("active")]
        assigned_cams: List[Dict[str, Any]] = []
        for cfg in self.camera_configs:
            has_src = cfg.source_type != "UNASSIGNED" and (
                cfg.source_type == "WEBCAM" or bool(str(cfg.source).strip())
            )
            if has_src:
                assigned_cams.append(
                    {
                        "cam_id": cfg.cam_id,
                        "name": cfg.name,
                        "source_type": cfg.source_type,
                        "masked_source": cfg.masked_source,
                    }
                )
        return {
            "active_hardware": active_hw,
            "all_hardware": sys_cams,
            "assigned_cameras": assigned_cams,
            "assigned_camera_ids": [c["cam_id"] for c in assigned_cams],
        }

    def assign_active_hardware_cameras(
        self, webcam_indices: List[int], auto_start: bool = False
    ) -> List[str]:
        """Gán danh sách các thiết bị Webcam đang hoạt động được người dùng chọn vào các ô CAM-01, CAM-02..."""
        assigned_ids: List[str] = []
        for i, w_idx in enumerate(webcam_indices):
            if i >= len(self.camera_configs):
                break
            cid = self.camera_configs[i].cam_id
            cname = self.camera_configs[i].name
            self.update_camera_source(
                cam_id=cid,
                new_name=cname,
                new_source=str(int(w_idx)),
                source_type="WEBCAM",
                webcam_index=int(w_idx),
                enabled=True,
                auto_restart=auto_start,
            )
            assigned_ids.append(cid)
        return assigned_ids

    def assign_test_video_to_camera(
        self,
        cam_id: str,
        video_path: str,
        custom_name: Optional[str] = None,
        auto_start: bool = False,
    ) -> Tuple[bool, str]:
        """Gắn một file video vào một ô camera cụ thể để chạy kiểm thử (Test Cam bằng File)."""
        if cam_id not in self.workers:
            return False, f"Không tìm thấy ô camera {cam_id}"
        raw_p = str(video_path or "").strip()
        if not raw_p:
            return False, "Chưa chọn file video kiểm thử"
        p = Path(raw_p)
        if not p.is_absolute():
            p = PROJECT_ROOT / raw_p
        if not p.exists() or not p.is_file():
            return False, f"Không tìm thấy file video: {raw_p}"

        cap = cv2.VideoCapture(str(p))
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return False, f"Không thể đọc định dạng file video: {p.name}"
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return False, f"File video '{p.name}' không có khung hình hợp lệ"

        w = self.workers[cam_id]
        target_name = custom_name.strip() if custom_name and custom_name.strip() else w.config.name
        self.update_camera_source(
            cam_id=cam_id,
            new_name=target_name,
            new_source=str(raw_p),
            source_type="VIDEO_TEST",
            enabled=True,
            auto_restart=auto_start,
        )
        return True, f"Đã gắn file video '{p.name}' vào {cam_id} ({target_name})"

    def detach_camera_source(self, cam_id: str) -> bool:
        """Gỡ nguồn video/camera khỏi một ô camera, đưa về trạng thái trống ban đầu (UNASSIGNED)."""
        if cam_id not in self.workers:
            return False
        w = self.workers[cam_id]
        w.stop()
        with w.lock:
            w.config.source = ""
            w.config.source_reference = ""
            w.config.source_type = "UNASSIGNED"
            w.status = "DISCONNECTED"
            w.lifecycle_state = "STOPPED"
            w.error_detail = "Chưa gán nguồn camera. Hãy chọn Camera đang hoạt động hoặc gắn File Video để test."
            w._set_placeholder("DISCONNECTED", w.error_detail)
        with closing(sqlite3.connect(self.event_db.db_path, timeout=10)) as conn, conn:
            conn.execute("DELETE FROM cameras WHERE camera_id=?", (cam_id,))
        self.add_log(f"Đã gỡ nguồn khỏi {cam_id} (Trạng thái: Chưa gán nguồn).")
        return True

    def clear_all_video_sources(self):
        """Gỡ toàn bộ các nguồn file video mẫu khỏi tất cả 9 camera để không tự động phát video."""
        for cid, w in self.workers.items():
            if w.config.source_type in ("VIDEO_TEST", "UNASSIGNED"):
                self.detach_camera_source(cid)

    def stop_monitoring(self):
        """Dừng giám sát toàn bộ camera, giải phóng VideoCapture, đóng RTSP và đưa tất cả về STOPPED."""
        self.stop_all()
        self.alert_manager.reset_alarm_acknowledgment()
        self.add_log("Đã dừng giám sát toàn bộ camera (Tất cả trở về trạng thái STOPPED).")

    def close(self) -> None:
        """Release camera workers and background delivery resources."""
        self.stop_all()
        self.telegram_alerts.close(timeout=10.0)

    def acknowledge_event(
        self,
        event_id: int,
        action_taken: str = "Đã xác nhận cháy thật",
        note: str = "",
        is_dismiss: bool = False,
        clear_all_pending: bool = False,
    ) -> bool:
        """Xác nhận hoặc bỏ qua sự kiện cảnh báo và tắt ngay âm báo còi hú."""
        if is_dismiss:
            ok = self.event_db.dismiss_event(event_id, note=note or "Đã bỏ qua (Báo động giả)")
            if clear_all_pending and hasattr(self.event_db, "acknowledge_all_pending"):
                self.event_db.acknowledge_all_pending(
                    action_taken="Báo động giả (Đã bỏ qua)",
                    note=note or "Đã bỏ qua (Báo động giả)",
                )
        else:
            ok = self.event_db.confirm_event(event_id, action_taken=action_taken, note=note)
            if clear_all_pending and hasattr(self.event_db, "acknowledge_all_pending"):
                self.event_db.acknowledge_all_pending(
                    action_taken=action_taken,
                    note=note or "Xác nhận từ Trung tâm cảnh báo",
                )

        # Đánh dấu các camera đang trong trạng thái cảnh báo đã được trực ban tiếp nhận để không tự bật lại còi
        for cid, w in self.workers.items():
            if w.alert_level in ("FIRE", "SUSPECT"):
                self.alert_manager.acknowledged_cameras.add(cid)

        self.alert_manager.alarm_silenced_by_ack = True
        self.alert_manager.stop_alarm()
        self.add_log(f"🔕 Đã {'bỏ qua' if is_dismiss else 'xác nhận'} sự kiện #{event_id} và tắt âm báo còi hú.")
        return ok

    def get_active_worker_count(self) -> int:
        """Trả về số lượng worker camera đang thực sự chạy luồng nền."""
        return sum(1 for w in self.workers.values() if w.running and w.thread is not None and w.thread.is_alive())

    def update_camera_roi(self, cam_id: str, exclusion_zones: List[Tuple[float, float, float, float]]):
        """Cập nhật vùng loại trừ ROI chuẩn hóa (x1, y1, x2, y2) trong [0, 1] cho một camera."""
        if cam_id not in self.workers:
            return
        clean_zones: List[Tuple[float, float, float, float]] = []
        for z in exclusion_zones:
            if len(z) == 4:
                x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in z]
                if x2 > x1 and y2 > y1:
                    clean_zones.append((round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)))
        w = self.workers[cam_id]
        with w.lock:
            w.config.exclusion_zones = list(clean_zones)
            w.verifier.exclusion_zones = list(clean_zones)
        self.add_log(f"Đã cập nhật {len(clean_zones)} vùng loại trừ ROI cho {cam_id}.")
        self._persist_camera_config(w.config)

    def set_default_cameras(self, cam_ids: List[str]):
        """Thiết lập danh sách camera mặc định khi khởi động trung tâm chỉ huy."""
        valid = [cid for cid in cam_ids if cid in self.workers]
        if valid:
            self.default_camera_ids = valid
            self.add_log(f"Đã đặt camera mặc định: {valid}")

    def get_uptime_str(self) -> str:
        """Trả về chuỗi thời gian hoạt động của hệ thống định dạng HH:MM:SS."""
        elapsed = max(0, int(time.time() - self.start_time))
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    def get_selected_snapshots(self, selected_cam_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Lấy trạng thái chỉ của các camera đang được chọn."""
        if selected_cam_ids is None:
            return [w.get_snapshot() for w in self.workers.values() if w.config.enabled]
        s_set = set(selected_cam_ids)
        return [self.workers[cfg.cam_id].get_snapshot() for cfg in self.camera_configs if cfg.cam_id in s_set]

    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """Lấy trạng thái thời gian thực của cả 9 camera theo đúng thứ tự CAM-01..CAM-09."""
        return [self.workers[cfg.cam_id].get_snapshot() for cfg in self.camera_configs]

    def get_system_metrics(self, snapshots: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Tổng hợp chỉ số vận hành thực tế của toàn hệ thống 9 camera."""
        snaps = snapshots if snapshots is not None else self.get_all_snapshots()
        connected_cams = [s for s in snaps if s["status"] == "CONNECTED"]
        reconnecting_cams = [s for s in snaps if s["status"] == "RECONNECTING"]
        error_cams = [s for s in snaps if s["status"] == "ERROR"]
        disconnected_cams = [s for s in snaps if s["status"] in ("DISCONNECTED", "ERROR", "RECONNECTING")]

        fire_cams = [s for s in connected_cams if s["alert_level"] == "FIRE"]
        suspect_cams = [s for s in connected_cams if s["alert_level"] == "SUSPECT"]

        total_cap_fps = sum(s["capture_fps"] for s in connected_cams)
        avg_cap_fps = (total_cap_fps / len(connected_cams)) if connected_cams else 0.0
        avg_inf_fps = (
            sum(s["inference_fps"] for s in connected_cams) / len(connected_cams)
            if connected_cams
            else 0.0
        )

        real_cam_count = sum(
            1 for s in connected_cams if s["source_type"] in ("WEBCAM", "IP_CAMERA")
        )
        test_vid_count = sum(
            1 for s in connected_cams if s["source_type"] == "VIDEO_TEST"
        )

        if fire_cams:
            overall_level = "FIRE"
        elif suspect_cams:
            overall_level = "SUSPECT"
        elif error_cams or reconnecting_cams:
            overall_level = "CONN_ERROR"
        elif len(connected_cams) == 0:
            overall_level = "STANDBY"
        else:
            overall_level = "SAFE"

        return {
            "operating_mode": self.settings.operating_mode,
            "total_cameras": len(snaps),
            "online_count": len(connected_cams),
            "connected_count": len(connected_cams),
            "reconnecting_count": len(reconnecting_cams),
            "error_count": len(error_cams),
            "disconnected_count": len(disconnected_cams),
            "real_cam_count": real_cam_count,
            "test_vid_count": test_vid_count,
            "fire_cam_count": len(fire_cams),
            "suspect_cam_count": len(suspect_cams),
            "total_fps": round(total_cap_fps, 1),
            "avg_fps": round(avg_cap_fps, 1),
            "avg_infer_fps": round(avg_inf_fps, 1),
            "overall_level": overall_level,
            "fire_cameras": [s["cam_id"] for s in fire_cams],
            "suspect_cameras": [s["cam_id"] for s in suspect_cams],
            "error_cameras": [s["cam_id"] for s in error_cams + reconnecting_cams],
        }
