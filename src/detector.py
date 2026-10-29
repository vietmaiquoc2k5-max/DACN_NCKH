"""Module tích hợp mô hình YOLO phát hiện Lửa (FIRE) và Khói (SMOKE) cho hệ thống đa luồng.

Đáp ứng nghiêm ngặt yêu cầu Mục 4:
- Ưu tiên sử dụng `best.pt` đã huấn luyện trong dự án.
- Không tự ý sử dụng model pretrained COCO (`yolov8n.pt`) để giả lập kết quả phát hiện cháy thật.
- Kiểm tra chính xác tên lớp (`0: smoke` -> `SMOKE`, `1: fire` -> `FIRE`).
- Vẽ bounding box, nhãn `FIRE` / `SMOKE` và confidence trực tiếp trên hình ảnh với màu phân biệt.
- Đo thời gian suy luận thực tế (ms) và FPS xử lý thực tế.
- Chia sẻ một instance model duy nhất với khóa đồng bộ (Thread-safe Lock) để tiết kiệm RAM/VRAM khi chạy 9 camera.
"""
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import CANDIDATE_MODEL_PATHS, CLASS_COLORS_BGR, PROJECT_ROOT
from src.temporal_verifier import DetectionBox


class SharedFireSmokeDetector:
    """Bộ phát hiện Lửa & Khói dùng chung an toàn đa luồng cho tối đa 9 camera."""

    def __init__(self, model_path: str = "best.pt", conf_threshold: float = 0.35, imgsz: int = 640):
        self.requested_path = model_path
        self.resolved_path: str = ""
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)

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
        """Tải và xác thực mô hình YOLO chuyên dụng phát hiện Lửa & Khói."""
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

                self.model = loaded_model
                self.resolved_path = found
                self.class_map = mapped
                self.model_loaded = True
                self.error_message = ""
                return True, f"Đã nạp thành công mô hình: {found} (Lớp: {self.class_map})"

            except Exception as e:
                self.model = None
                self.model_loaded = False
                self.class_map = {}
                self.error_message = f"Lỗi khi nạp mô hình '{found}': {e}"
                return False, self.error_message

    def get_model_info(self) -> Dict[str, Any]:
        """Trả về thông tin trạng thái của mô hình hiện tại."""
        return {
            "loaded": self.model_loaded,
            "requested_path": self.requested_path,
            "path": self.resolved_path or self.requested_path,
            "names": self.class_map,
            "raw_names": self.model.names if self.model is not None else {},
            "error": self.error_message,
            "last_infer_ms": round(self.last_infer_ms, 2),
        }

    def detect_frame(
        self, frame: np.ndarray, conf_threshold: Optional[float] = None
    ) -> Tuple[List[DetectionBox], float]:
        """Chạy suy luận YOLO trên một khung hình, trả về danh sách DetectionBox và thời gian xử lý (ms)."""
        if not self.model_loaded or self.model is None or frame is None:
            return [], 0.0

        conf_val = float(conf_threshold) if conf_threshold is not None else self.conf_threshold
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return [], 0.0

        t0 = time.perf_counter()
        with self.infer_lock:
            results = self.model.predict(
                frame,
                conf=conf_val,
                imgsz=self.imgsz,
                verbose=False,
            )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self.last_infer_ms = infer_ms

        detections: List[DetectionBox] = []
        if not results:
            return detections, infer_ms

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections, infer_ms

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.class_map:
                continue

            label = self.class_map[cls_id]  # "FIRE" hoặc "SMOKE"
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = (
                max(0, int(coords[0])),
                max(0, int(coords[1])),
                min(w - 1, int(coords[2])),
                min(h - 1, int(coords[3])),
            )
            if x2 <= x1 or y2 <= y1:
                continue

            bw = (x2 - x1) / float(w)
            bh = (y2 - y1) / float(h)
            xc = (x1 + x2) / (2.0 * float(w))
            yc = (y1 + y2) / (2.0 * float(h))

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
    ) -> np.ndarray:
        """Vẽ bounding box, nhãn FIRE/SMOKE, confidence và trạng thái lên khung hình."""
        annotated = frame.copy()
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

        # Vẽ từng bounding box phát hiện
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            color = CLASS_COLORS_BGR.get(box.label, (0, 0, 255))
            thickness = 2 if max(w, h) < 800 else 3

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            trk_str = f"#{box.track_id} " if box.track_id > 0 else ""
            caption = f"{trk_str}{box.label} {int(round(box.confidence * 100))}%"

            font_scale = 0.55 if max(w, h) < 800 else 0.7
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

        # Vẽ viền cảnh báo trên khung hình khi có SUSPECT hoặc FIRE
        if alert_level == "FIRE":
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (36, 36, 239), 6)
        elif alert_level == "SUSPECT":
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (11, 158, 245), 4)

        return annotated

