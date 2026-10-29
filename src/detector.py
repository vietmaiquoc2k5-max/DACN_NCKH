"""Module tích hợp mô hình YOLO phát hiện Lửa (FIRE) và Khói (SMOKE) cho hệ thống đa luồng.

Đáp ứng nghiêm ngặt yêu cầu Mục 4 & Tối ưu hóa tốc độ thời gian thực:
- Ưu tiên sử dụng `best.pt` đã huấn luyện trong dự án.
- Không tự ý sử dụng model pretrained COCO (`yolov8n.pt`) để giả lập kết quả phát hiện cháy thật.
- Kiểm tra chính xác tên lớp (`0: smoke` -> `SMOKE`, `1: fire` -> `FIRE`).
- Tự động kích hoạt GPU CUDA + FP16 (`half=True`) nếu có GPU NVIDIA, hoặc tối ưu luồng CPU.
- Hỗ trợ cơ chế suy luận không chặn (`non_blocking`) giúp luồng đọc video luôn mượt (20-30 FPS)
  ngay cả khi chạy đồng thời nhiều camera.
"""
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import CANDIDATE_MODEL_PATHS, CLASS_COLORS_BGR, PROJECT_ROOT
from src.temporal_verifier import DetectionBox


class SharedFireSmokeDetector:
    """Bộ phát hiện Lửa & Khói dùng chung an toàn đa luồng cho tối đa 9 camera."""

    def __init__(self, model_path: str = "best.pt", conf_threshold: float = 0.35, imgsz: int = 416):
        self.requested_path = model_path
        self.resolved_path: str = ""
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)

        # Tự động phát hiện GPU CUDA & kích hoạt FP16 để tăng tốc 2.5x - 4x
        self.has_cuda: bool = bool(torch.cuda.is_available())
        self.device: str = "cuda:0" if self.has_cuda else "cpu"
        self.use_half: bool = self.has_cuda
        if self.has_cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                self.device_label = f"GPU: {gpu_name} (FP16)"
                torch.backends.cudnn.benchmark = True
            except Exception:
                self.device_label = "GPU CUDA (FP16)"
        else:
            cpu_threads = max(2, min(8, (os.cpu_count() or 4) // 2))
            try:
                torch.set_num_threads(cpu_threads)
            except Exception:
                pass
            self.device_label = f"CPU ({cpu_threads} threads)"

        self.model: Optional[YOLO] = None
        self.model_loaded: bool = False
        self.error_message: str = ""
        self.class_map: Dict[int, str] = {}  # {0: 'SMOKE', 1: 'FIRE'}
        self.infer_lock = threading.Lock()
        self.last_infer_ms: float = 0.0

        self.load_model(model_path)

    def _resolve_model_path(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Tìm kiếm đường dẫn file trọng số model đã huấn luyện của dự án."""
        search_list = []
        if custom_path:
            p = Path(custom_path)
            search_list.append(str(p))
            if not p.is_absolute():
                search_list.append(str(PROJECT_ROOT / custom_path))
        for cp in CANDIDATE_MODEL_PATHS:
            if cp not in search_list:
                search_list.append(cp)

        for candidate in search_list:
            if candidate and os.path.exists(candidate) and os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    def load_model(self, custom_path: Optional[str] = None) -> Tuple[bool, str]:
        """Tải, xác thực và khởi động (warmup) mô hình YOLO chuyên dụng phát hiện Lửa & Khói."""
        with self.infer_lock:
            target = custom_path if custom_path else self.requested_path
            self.requested_path = target
            found = self._resolve_model_path(target)

            if not found:
                self.model = None
                self.model_loaded = False
                self.class_map = {}
                self.error_message = (
                    f"Không tìm thấy file mô hình '{target}'. Vui lòng đặt file 'best.pt' tại thư mục gốc "
                    f"({PROJECT_ROOT}) hoặc cấu hình đường dẫn hợp lệ trong mục Cấu hình Model."
                )
                return False, self.error_message

            try:
                loaded_model = YOLO(found)
                raw_names = loaded_model.names or {}

                # Kiểm tra chặt chẽ các lớp của model có phải là fire/smoke hay không
                mapped: Dict[int, str] = {}
                for cid, cname in raw_names.items():
                    low = str(cname).strip().lower()
                    if low in ("fire", "lua", "lửa", "chay", "cháy"):
                        mapped[int(cid)] = "FIRE"
                    elif low in ("smoke", "khoi", "khói"):
                        mapped[int(cid)] = "SMOKE"

                if not mapped:
                    self.model = None
                    self.model_loaded = False
                    self.class_map = {}
                    self.error_message = (
                        f"Mô hình '{found}' có các lớp {raw_names} không phải là mô hình phát hiện Lửa/Khói "
                        f"(`fire` / `smoke`). Hệ thống từ chối dùng model không hợp lệ để tránh báo cáo sai."
                    )
                    return False, self.error_message

                # Fuse Conv+BN và warmup sẵn trên thiết bị đích để không bị trễ ở khung hình đầu tiên
                try:
                    loaded_model.fuse()
                except Exception:
                    pass

                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                try:
                    predict_kwargs: Dict[str, Any] = {
                        "conf": 0.5,
                        "imgsz": 320,
                        "device": self.device,
                        "verbose": False,
                    }
                    if self.use_half:
                        predict_kwargs["half"] = True
                    loaded_model.predict(dummy, **predict_kwargs)
                except Exception:
                    if self.has_cuda:
                        self.use_half = False

                self.model = loaded_model
                self.resolved_path = found
                self.class_map = mapped
                self.model_loaded = True
                self.error_message = ""
                return True, f"Đã nạp thành công mô hình: {found} trên {self.device_label} (Lớp: {self.class_map})"

            except Exception as e:
                self.model = None
                self.model_loaded = False
                self.class_map = {}
                self.error_message = f"Lỗi khi nạp mô hình '{found}': {e}"
                return False, self.error_message

    def get_model_info(self) -> Dict[str, Any]:
        """Trả về thông tin trạng thái của mô hình và phần cứng hiện tại."""
        return {
            "loaded": self.model_loaded,
            "requested_path": self.requested_path,
            "path": self.resolved_path or self.requested_path,
            "names": self.class_map,
            "raw_names": self.model.names if self.model is not None else {},
            "device": self.device_label,
            "has_cuda": self.has_cuda,
            "error": self.error_message,
            "last_infer_ms": round(self.last_infer_ms, 2),
        }

    def detect_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
        non_blocking: bool = False,
    ) -> Tuple[Optional[List[DetectionBox]], float]:
        """Chạy suy luận YOLO trên một khung hình.

        Nếu `non_blocking=True` và model đang bận xử lý luồng camera khác, trả về `(None, 0.0)`
        ngay lập tức để luồng đọc camera hiện tại không bị khựng hình.
        """
        if not self.model_loaded or self.model is None or frame is None:
            return [], 0.0

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return [], 0.0

        conf_val = float(conf_threshold) if conf_threshold is not None else self.conf_threshold
        run_imgsz = int(imgsz) if imgsz is not None else self.imgsz

        if non_blocking:
            acquired = self.infer_lock.acquire(blocking=True, timeout=0.004)
            if not acquired:
                return None, 0.0
        else:
            self.infer_lock.acquire()

        try:
            t0 = time.perf_counter()
            predict_kwargs: Dict[str, Any] = {
                "conf": conf_val,
                "imgsz": run_imgsz,
                "device": self.device,
                "max_det": 25,
                "verbose": False,
            }
            if self.use_half:
                predict_kwargs["half"] = True
            with torch.inference_mode():
                results = self.model.predict(frame, **predict_kwargs)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            self.last_infer_ms = infer_ms
        finally:
            self.infer_lock.release()

        detections: List[DetectionBox] = []
        if not results:
            return detections, infer_ms

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections, infer_ms

        # Chuyển toàn bộ tensor sang CPU/numpy trong 1 lần gọi duy nhất thay vì lặp từng box.cpu()
        xyxy_arr = boxes.xyxy.detach().cpu().numpy()
        conf_arr = boxes.conf.detach().cpu().numpy()
        cls_arr = boxes.cls.detach().cpu().numpy()

        inv_w = 1.0 / float(w)
        inv_h = 1.0 / float(h)

        for i in range(len(xyxy_arr)):
            cls_id = int(cls_arr[i])
            if cls_id not in self.class_map:
                continue

            label = self.class_map[cls_id]  # "FIRE" hoặc "SMOKE"
            conf = float(conf_arr[i])
            coords = xyxy_arr[i]
            x1 = max(0, int(coords[0]))
            y1 = max(0, int(coords[1]))
            x2 = min(w - 1, int(coords[2]))
            y2 = min(h - 1, int(coords[3]))
            if x2 <= x1 or y2 <= y1:
                continue

            bw = (x2 - x1) * inv_w
            bh = (y2 - y1) * inv_h
            xc = (x1 + x2) * 0.5 * inv_w
            yc = (y1 + y2) * 0.5 * inv_h

            detections.append(
                DetectionBox(
                    cls_id=cls_id,
                    label=label,
                    confidence=conf,
                    xyxy=(x1, y1, x2, y2),
                    norm_xywh=(xc, yc, bw, bh),
                )
            )

        return detections, infer_ms

    @staticmethod
    def draw_annotations(
        frame: np.ndarray,
        boxes: List[DetectionBox],
        cam_name: str = "",
        alert_level: str = "SAFE",
        fps: float = 0.0,
        exclusion_zones: Optional[List[Tuple[float, float, float, float]]] = None,
        inplace: bool = False,
    ) -> np.ndarray:
        """Vẽ bounding box, nhãn FIRE/SMOKE, confidence và trạng thái lên khung hình."""
        annotated = frame if inplace else frame.copy()
        h, w = annotated.shape[:2]

        # Vẽ các vùng loại trừ (Exclusion Zones) nếu có
        if exclusion_zones:
            for rx1, ry1, rx2, ry2 in exclusion_zones:
                px1, py1 = int(rx1 * w), int(ry1 * h)
                px2, py2 = int(rx2 * w), int(ry2 * h)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (128, 128, 128), 1)
                cv2.putText(
                    annotated,
                    "ROI MASK",
                    (px1 + 4, max(14, py1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 180),
                    1,
                )

        # Vẽ từng bounding box phát hiện (kèm tên camera, loại đối tượng FIRE/SMOKE và confidence)
        short_cam = cam_name.split("-")[0].strip() if "-" in cam_name else cam_name[:10]
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            color = CLASS_COLORS_BGR.get(box.label, (0, 0, 255))
            thickness = 2 if max(w, h) < 800 else 3

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            trk_str = f"#{box.track_id} " if box.track_id > 0 else ""
            cam_prefix = f"[{short_cam}] " if short_cam else ""
            caption = f"{cam_prefix}{trk_str}{box.label} {int(round(box.confidence * 100))}%"

            font_scale = 0.50 if max(w, h) < 800 else 0.65
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            lbl_y1 = max(0, y1 - th - baseline - 6)
            lbl_y2 = y1 if y1 - th - baseline - 6 >= 0 else min(h, y1 + th + baseline + 6)
            lbl_x2 = min(w, x1 + tw + 10)

            cv2.rectangle(annotated, (x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
            text_y = lbl_y2 - baseline - 3 if y1 - th - baseline - 6 >= 0 else lbl_y2 - baseline - 2
            cv2.putText(
                annotated,
                caption,
                (x1 + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Vẽ header mờ hiển thị tên camera trên góc trái trên của khung hình
        if cam_name:
            overlay_text = f"{cam_name[:32]}"
            cv2.rectangle(annotated, (6, 6), (min(w - 6, 12 + len(overlay_text) * 8), 26), (15, 23, 42), -1)
            cv2.putText(
                annotated,
                overlay_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (226, 232, 240),
                1,
                cv2.LINE_AA,
            )

        # Vẽ viền cảnh báo trên khung hình khi có SUSPECT hoặc FIRE
        if alert_level == "FIRE":
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (36, 36, 239), 5)
        elif alert_level == "SUSPECT":
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (11, 158, 245), 3)

        return annotated
