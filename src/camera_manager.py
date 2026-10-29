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
    mask_rtsp_credentials,
    infer_source_type,
)
from src.detector import SharedFireSmokeDetector
from src.temporal_verifier import TemporalVerifier, DetectionBox
from src.database import EventDatabase
from alert_manager import AlertManager


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


def open_opencv_capture(
    source: str,
    source_type: str = "VIDEO_TEST",
    webcam_index: int = 0,
    capture_width: int = 480,
) -> Tuple[Optional[cv2.VideoCapture], bool, str]:
    """Mở nguồn video/webcam/RTSP bằng OpenCV VideoCapture với cấu hình giảm trễ (buffer=1).

    Returns:
        (cap, is_file_source, error_message_masked)
    """
    raw_src = str(source).strip()
    s_type = source_type or infer_source_type(raw_src)

    if s_type == "WEBCAM" or raw_src.isdigit():
        idx = int(raw_src) if raw_src.isdigit() else int(webcam_index)
        try:
            if sys.platform == "win32":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)

            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                return None, False, f"Không tìm thấy thiết bị Webcam chỉ số #{idx}"

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

        # Trạng thái chuẩn: "CONNECTED" | "DISCONNECTED" | "RECONNECTING" | "ERROR" | "PAUSED"
        self.status: str = "DISCONNECTED"
        self.error_detail: str = "Chưa kết nối (Nhấn Kết nối để bắt đầu)"
        self.running: bool = False
        self.paused: bool = False
        self.thread: Optional[threading.Thread] = None
        self.infer_thread: Optional[threading.Thread] = None
        self.infer_event = threading.Event()
        self.lock = threading.Lock()

        # Chỉ số FPS đọc khung hình và FPS suy luận model riêng biệt
        self.capture_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.infer_ms: float = 0.0
        self.last_update_ts: float = 0.0
        self.last_update_str: str = "--:--:--"
        self.frame_count: int = 0
        self.last_inferred_frame_id: int = 0
        self.reconnect_attempts: int = 0
        self.resolution_str: str = "N/A"
        self.latest_raw_bgr: Optional[np.ndarray] = None

        # Trạng thái phát hiện hiện tại
        self.alert_level: str = "SAFE"  # "SAFE" | "SUSPECT" | "FIRE" | "CONN_ERROR"
        self.primary_label: str = ""
        self.max_confidence: float = 0.0
        self.active_hits: int = 0
        self.window_size: int = settings.window_size_n
        self.last_boxes: List[DetectionBox] = []

        init_rgb = create_status_placeholder_rgb(
            self.config.name, "DISCONNECTED", self.error_detail
        )
        self.latest_rgb_frame: np.ndarray = init_rgb
        self.latest_jpeg_b64: str = encode_rgb_to_jpeg_b64(init_rgb, quality=settings.jpeg_quality)
        self.preview_rgb_frame: Optional[np.ndarray] = None

    def _set_placeholder(self, status: str, detail: str):
        """Cập nhật cả khung hình RGB và chuỗi JPEG Base64 cho trạng thái thông báo."""
        rgb = create_status_placeholder_rgb(self.config.name, status, detail)
        self.latest_rgb_frame = rgb
        self.latest_jpeg_b64 = encode_rgb_to_jpeg_b64(rgb, quality=self.settings.jpeg_quality)

    def _log(self, msg: str):
        safe_msg = mask_rtsp_credentials(msg)
        if self.log_callback:
            self.log_callback(f"[{self.config.cam_id}] {safe_msg}")

    def start(self):
        """Khởi chạy kết nối camera (tách biệt luồng đọc khung hình và luồng suy luận YOLO)."""
        with self.lock:
            if self.running and self.thread and self.thread.is_alive():
                self.paused = False
                if self.status == "PAUSED":
                    self.status = "CONNECTED"
                return
            self.running = True
            self.paused = False
            self.reconnect_attempts = 0
            self.frame_count = 0
            self.last_inferred_frame_id = 0
            self.latest_raw_bgr = None
            self.status = "RECONNECTING"
            self.error_detail = f"Đang kết nối tới {self.config.masked_source}..."
            self._set_placeholder("RECONNECTING", self.error_detail)
            self.verifier.reset()
            self.infer_event.clear()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
            self.thread.start()
            self.infer_thread.start()
        self._log(f"Đang kết nối nguồn [{self.config.source_type}]: {self.config.masked_source}")

    def pause(self):
        """Tạm dừng / tiếp tục luồng camera."""
        with self.lock:
            if self.running:
                self.paused = not self.paused
                self.status = "PAUSED" if self.paused else "CONNECTED"
                if self.paused:
                    self.capture_fps = 0.0
                    self.inference_fps = 0.0
                    self._set_placeholder("PAUSED", "Đã tạm dừng bởi người vận hành")
        self._log("Tạm dừng luồng" if self.paused else "Tiếp tục luồng")

    def stop(self):
        """Ngắt kết nối hoàn toàn luồng camera và giải phóng VideoCapture."""
        self.running = False
        self.paused = False
        self.infer_event.set()
        t = self.thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.5)
        it = self.infer_thread
        if it and it.is_alive() and it is not threading.current_thread():
            it.join(timeout=1.0)
        with self.lock:
            self.status = "DISCONNECTED"
            self.capture_fps = 0.0
            self.inference_fps = 0.0
            self.infer_ms = 0.0
            self.latest_raw_bgr = None
            self.alert_level = "SAFE"
            self.primary_label = ""
            self.max_confidence = 0.0
            self.active_hits = 0
            self.error_detail = "Đã ngắt kết nối"
            self._set_placeholder("DISCONNECTED", self.error_detail)
        self._log("Đã ngắt kết nối và giải phóng VideoCapture.")

    def reconnect(self):
        """Ngắt kết nối hiện tại và thực hiện kết nối lại ngay lập tức."""
        self._log(f"Kết nối lại thủ công ({self.config.masked_source})...")
        self.stop()
        time.sleep(0.05)
        self.start()

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
        infer_fps_win = deque(maxlen=15)
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

            if raw_frame is None or fid <= self.last_inferred_frame_id:
                continue

            self.last_inferred_frame_id = fid
            infer_input = raw_frame.copy()

            self.verifier.update_params(
                window_size_n=self.settings.window_size_n,
                confirm_frames_k=self.settings.confirm_frames_k,
                suspect_frames_k=self.settings.suspect_frames_k,
                cooldown_sec=self.settings.alert_cooldown_sec,
            )

            raw_boxes, new_infer_ms = self.detector.detect_frame(
                infer_input,
                conf_threshold=self.settings.conf_threshold,
                imgsz=self.settings.imgsz,
                non_blocking=False,
            )
            if raw_boxes is None or not self.running:
                continue

            ver_res = self.verifier.update(raw_boxes)
            valid_boxes = ver_res["valid_boxes"]
            if new_infer_ms > 0:
                infer_fps_win.append(1000.0 / new_infer_ms)
            actual_infer_fps = float(sum(infer_fps_win) / len(infer_fps_win)) if infer_fps_win else 0.0

            with self.lock:
                self.last_boxes = valid_boxes
                self.infer_ms = round(new_infer_ms, 1)
                self.inference_fps = round(actual_infer_fps, 1)
                if self.status == "CONNECTED":
                    self.alert_level = ver_res["alert_level"]
                self.primary_label = ver_res["primary_label"]
                self.max_confidence = ver_res["max_confidence"]
                self.active_hits = ver_res["active_hits"]
                self.window_size = ver_res["window_size"]
                curr_cap_fps = self.capture_fps

            if ver_res["should_trigger_db_alert"]:
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
                self.event_db.record_event(
                    frame=snap_bgr,
                    camera_name=self.config.cam_id,
                    label=lbl,
                    confidence=conf_val,
                    alert_level=ver_res["alert_level"],
                    is_demo=0,
                )
                if not self.settings.mute_audio:
                    self.event_db.alert_manager.play_alarm()
                self._log(
                    f"🚨 CẢNH BÁO [{lbl}] ({int(conf_val*100)}%) tại {self.config.name} "
                    f"sau {ver_res['active_hits']}/{ver_res['window_size']} khung!"
                )

    def _run_loop(self):
        """Vòng lặp đọc frame tốc độ cao (24-30 FPS), hoàn toàn không bị chặn bởi thời gian suy luận YOLO."""
        cap: Optional[cv2.VideoCapture] = None
        is_file: bool = False
        capture_fps_win = deque(maxlen=25)
        last_tick = time.perf_counter()

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.08)
                    last_tick = time.perf_counter()
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
                        self.reconnect_attempts += 1
                        max_att = max(1, int(self.config.max_reconnect_attempts))
                        if self.config.auto_reconnect and self.reconnect_attempts <= max_att:
                            with self.lock:
                                self.status = "RECONNECTING"
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
                            time.sleep(self.config.reconnect_interval_sec)
                            continue
                        else:
                            with self.lock:
                                self.status = "ERROR"
                                self.capture_fps = 0.0
                                self.inference_fps = 0.0
                                self.alert_level = "CONN_ERROR"
                                self.error_detail = err_msg
                                self._set_placeholder("ERROR", err_msg)
                                self.running = False
                            self._log(f"Ngừng kết nối do lỗi: {err_msg}")
                            break

                    self.reconnect_attempts = 0
                    with self.lock:
                        self.status = "CONNECTED"
                        self.error_detail = ""
                        if self.alert_level == "CONN_ERROR":
                            self.alert_level = "SAFE"
                    self._log(f"Đã kết nối thành công ({self.config.masked_source}).")
                    last_tick = time.perf_counter()

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
                            time.sleep(1.0)
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

                self.frame_count += 1
                fh, fw = frame.shape[:2]
                if self.frame_count == 1 or self.frame_count % 30 == 0:
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
                    self.latest_raw_bgr = proc_frame
                    valid_boxes = list(self.last_boxes)
                    curr_alert_level = self.alert_level if self.alert_level != "CONN_ERROR" else "SAFE"

                if (self.frame_count % skip_n == 0) or (self.frame_count == 1):
                    self.infer_event.set()

                # Tính FPS đọc khung hình (capture_fps)
                now_perf = time.perf_counter()
                dt = max(1e-4, now_perf - last_tick)
                last_tick = now_perf
                capture_fps_win.append(1.0 / dt)
                actual_capture_fps = float(sum(capture_fps_win) / len(capture_fps_win))

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
                "auto_reconnect": self.config.auto_reconnect,
                "status": self.status,  # "CONNECTED" | "DISCONNECTED" | "RECONNECTING" | "ERROR" | "PAUSED"
                "is_connected": (self.status == "CONNECTED"),
                "error_detail": mask_rtsp_credentials(self.error_detail),
                "fps": self.capture_fps,
                "capture_fps": self.capture_fps,
                "inference_fps": self.inference_fps,
                "infer_ms": self.infer_ms,
                "resolution": self.resolution_str,
                "last_update_str": self.last_update_str,
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
                "reconnect_attempts": self.reconnect_attempts,
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

        self.logs: deque = deque(maxlen=100)
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
            self.add_log(
                f"Đã nạp model YOLO: {model_info['path']} trên {model_info.get('device', 'CPU')} | Lớp: {model_info['names']}"
            )
        else:
            self.add_log(f"⚠️ CẢNH BÁO MODEL: {model_info['error']}")

    def add_log(self, message: str):
        """Ghi nhật ký hệ thống, tự động che mật khẩu RTSP nếu có."""
        ts = time.strftime("%H:%M:%S")
        safe_message = mask_rtsp_credentials(message)
        with self.log_lock:
            self.logs.appendleft(f"[{ts}] {safe_message}")

    def get_recent_logs(self, limit: int = 25) -> List[str]:
        with self.log_lock:
            return list(self.logs)[:limit]

    def set_camera_enabled(self, cam_id: str, enabled: bool):
        """Bật hoặc tắt quyền kích hoạt của một camera độc lập."""
        if cam_id in self.workers:
            w = self.workers[cam_id]
            w.config.enabled = bool(enabled)
            if not enabled and w.running:
                w.stop()
            self.add_log(f"Đã chuyển trạng thái kích hoạt {cam_id} -> {'BẬT' if enabled else 'TẮT'}.")

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

        if w.config.source_type == "WEBCAM":
            w.config.webcam_index = int(webcam_index)
            w.config.source = str(webcam_index)
        else:
            w.config.source = new_source.strip()

        if enabled is not None:
            w.config.enabled = bool(enabled)
        if auto_reconnect is not None:
            w.config.auto_reconnect = bool(auto_reconnect)

        with w.lock:
            w._set_placeholder("DISCONNECTED", f"Đã lưu cấu hình: {w.config.masked_source}")
        self.add_log(
            f"Đã lưu cấu hình {cam_id} [{w.config.source_type}]: '{w.config.name}' -> '{w.config.masked_source}'"
        )
        if auto_restart or (was_running and w.config.enabled):
            w.start()

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
