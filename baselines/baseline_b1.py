"""Baseline 1 (B1): Bộ phát hiện cháy/khói dựa trên Luật màu (HSV/YCbCr) kết hợp Trừ nền (MOG2).

Theo Đề cương NCKH (Mục tiêu MT2):
- Cơ chế: Kết hợp luật màu không gian màu HSV & YCbCr (đặc trưng sắc độ của ngọn lửa và khói xám)
  với thuật toán trừ nền thích nghi GMM (Gaussian Mixture Model / MOG2) để tách chuyển động.
- Mục đích: Làm mốc tham chiếu truyền thống (Traditional Baseline) để so sánh khoa học
  với các mô hình Deep Learning (YOLOv8, YOLO11, RT-DETR).

Tài liệu tham khảo lý thuyết:
1. T. H. Chen, P. H. Wu, and Y. C. Chiou, "An early fire-detection method based on image processing," ICIP 2004.
2. T. Celik and H. Demirel, "Fire detection in video sequences using a novel color model," IVC 2007.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class RuleBasedFireSmokeDetector:
    """Bộ phát hiện đám cháy và khói truyền thống bằng Luật màu & Trừ nền (Baseline B1)."""

    def __init__(
        self,
        use_motion: bool = True,
        min_area: int = 120,
        history: int = 500,
        var_threshold: float = 25.0,
        imgsz: Optional[int] = None,
    ):
        """
        Khởi tạo detector B1.
        :param use_motion: Bật trừ nền MOG2 (áp dụng cho video/RTSP stream).
        :param min_area: Diện tích tối thiểu của contour (pixel) để loại bỏ nhiễu hạt.
        :param history: Số khung hình dùng để ước lượng nền trong MOG2.
        :param var_threshold: Ngưỡng phương sai Mahalanobis trong MOG2.
        :param imgsz: Kích thước cạnh dài để resize tăng tốc (ví dụ 640). Mặc định None (giữ nguyên).
        """
        self.use_motion = use_motion
        self.min_area = min_area
        self.imgsz = imgsz

        # Khởi tạo thuật toán trừ nền Gaussian Mixture Model (MOG2)
        if self.use_motion:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=False,
            )
        else:
            self.bg_subtractor = None

        # Kernel cho các phép toán hình thái học (Morphological Operations)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def reset_background(self):
        """Khởi tạo lại mô hình nền (dùng khi chuyển camera hoặc đổi video)."""
        if self.bg_subtractor is not None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=25.0, detectShadows=False
            )

    def extract_fire_color_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Trích xuất mặt nạ màu Lửa dựa trên kết hợp 3 không gian màu: RGB, YCbCr và HSV.
        """
        b, g, r = cv2.split(bgr_frame)
        h, w = bgr_frame.shape[:2]

        # 1. Luật màu không gian RGB (Celik et al.): R > G > B và R > R_mean
        r_mean = np.mean(r)
        rgb_fire = (r > g) & (g > b) & (r > r_mean) & (r > 120)

        # 2. Luật màu không gian YCbCr (Chen et al.):
        # Y (độ sáng), Cb (độ lệch xanh lam), Cr (độ lệch đỏ)
        # Ngọn lửa: Y > Cb, Cr > Cb, và khoảng cách |Cr - Cb| đủ lớn
        ycbcr = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        ycbcr_fire = (y > cb) & (cr > cb) & (cr >= 135) & (cb <= 125)

        # 3. Luật màu không gian HSV:
        # Hue của lửa nằm trong dải Đỏ - Cam - Vàng ([0, 35] hoặc [170, 180])
        # Saturation và Value cao (màu bão hòa và sáng rực)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        hsv_fire = ((hue <= 35) | (hue >= 170)) & (sat >= 70) & (val >= 110)

        # Kết hợp các luật màu (AND logic)
        fire_mask = (rgb_fire & ycbcr_fire & hsv_fire).astype(np.uint8) * 255

        # Lọc hình thái học để nối liền các mảnh lửa và khử nhiễu
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, self.kernel_open)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, self.kernel_close)
        return fire_mask

    def extract_smoke_color_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Trích xuất mặt nạ màu Khói dựa trên tính chất xám phân tán (Gray Dispersion).
        Khói có độ bão hòa màu rất thấp (R ~ G ~ B) và độ sáng từ tối đến vừa.
        """
        b, g, r = cv2.split(bgr_frame.astype(np.int16))

        # 1. Độ lệch chuẩn giữa các kênh màu nhỏ (|R-G| < 20, |G-B| < 20, |R-B| < 20)
        diff_rg = np.abs(r - g)
        diff_gb = np.abs(g - b)
        diff_rb = np.abs(r - b)
        color_neutral = (diff_rg <= 22) & (diff_gb <= 22) & (diff_rb <= 22)

        # 2. Không gian HSV: Saturation thấp (nhạt màu) và Value trong dải khói [60, 225]
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)
        hsv_smoke = (sat <= 45) & (val >= 65) & (val <= 230)

        smoke_mask = (color_neutral & hsv_smoke).astype(np.uint8) * 255

        # Lọc hình thái học
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, self.kernel_open)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, self.kernel_close)
        return smoke_mask

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Thực hiện phát hiện đám cháy và khói trên khung hình.
        :param frame: Ảnh đầu vào BGR.
        :return: Danh sách các phát hiện:
                 [{"box": [x1, y1, x2, y2], "class_id": int, "class_name": str, "conf": float}]
                 (class_id: 0: smoke, 1: fire)
        """
        detections = []
        orig_h, orig_w = frame.shape[:2]

        # Tùy chọn resize để đạt tốc độ thời gian thực trên CPU
        if self.imgsz and max(orig_h, orig_w) > self.imgsz:
            scale = self.imgsz / max(orig_h, orig_w)
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)
            proc_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            sx = orig_w / target_w
            sy = orig_h / target_h
        else:
            proc_frame = frame
            sx = 1.0
            sy = 1.0

        h, w = proc_frame.shape[:2]

        # 1. Trừ nền chuyển động (nếu bật chế độ video/stream)
        if self.use_motion and self.bg_subtractor is not None:
            fg_mask = self.bg_subtractor.apply(proc_frame)
            # Loại bỏ bóng mờ (giá trị 127 trong MOG2)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        else:
            fg_mask = np.full((h, w), 255, dtype=np.uint8)

        # 2. Trích xuất mặt nạ màu Lửa & Khói
        fire_color = self.extract_fire_color_mask(proc_frame)
        smoke_color = self.extract_smoke_color_mask(proc_frame)

        # 3. Kết hợp với mặt nạ chuyển động: Cháy/Khói thực tế phải đang chuyển động/biến động
        # (Nếu use_motion=True thì loại trừ các vật thể đỏ tĩnh như biển quảng cáo)
        if self.use_motion:
            # Ngọn lửa chuyển động rõ rệt
            final_fire_mask = cv2.bitwise_and(fire_color, fg_mask)
            # Khói có thể khuếch tán chậm, giãn nở mặt nạ chuyển động
            fg_dilated = cv2.dilate(fg_mask, self.kernel_close, iterations=1)
            final_smoke_mask = cv2.bitwise_and(smoke_color, fg_dilated)
        else:
            final_fire_mask = fire_color
            final_smoke_mask = smoke_color

        # 4. Trích xuất Bounding Box cho LỬA (Class 1)
        fire_contours, _ = cv2.findContours(final_fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in fire_contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Tính độ tự tin heuristic: tỷ lệ pixel lửa thực sự trong hộp bao
                roi_mask = final_fire_mask[y:y+bh, x:x+bw]
                density = cv2.countNonZero(roi_mask) / (bw * bh)
                conf = min(0.95, round(0.40 + 0.55 * density, 2))

                # Scale tọa độ về ảnh gốc
                rx1, ry1 = int(x * sx), int(y * sy)
                rx2, ry2 = int((x + bw) * sx), int((y + bh) * sy)

                detections.append({
                    "box": [rx1, ry1, rx2, ry2],
                    "class_id": 1,
                    "class_name": "fire",
                    "conf": conf,
                    "area": area * sx * sy,
                })

        # 5. Trích xuất Bounding Box cho KHÓI (Class 0)
        smoke_contours, _ = cv2.findContours(final_smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in smoke_contours:
            area = cv2.contourArea(cnt)
            # Khói thường có diện tích bốc lên lớn hơn lửa
            if area >= self.min_area * 1.5:
                x, y, bw, bh = cv2.boundingRect(cnt)
                roi_mask = final_smoke_mask[y:y+bh, x:x+bw]
                density = cv2.countNonZero(roi_mask) / (bw * bh)
                conf = min(0.90, round(0.35 + 0.50 * density, 2))

                # Scale tọa độ về ảnh gốc
                rx1, ry1 = int(x * sx), int(y * sy)
                rx2, ry2 = int((x + bw) * sx), int((y + bh) * sy)

                detections.append({
                    "box": [rx1, ry1, rx2, ry2],
                    "class_id": 0,
                    "class_name": "smoke",
                    "conf": conf,
                    "area": area * sx * sy,
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Vẽ bounding box và nhãn của Baseline B1 lên khung hình."""
        display = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det["class_id"]
            conf = det["conf"]

            color = (0, 69, 255) if cls_id == 1 else (255, 191, 0) # Fire: Đỏ cam, Smoke: Xanh lam
            label = f"B1-{det['class_name'].upper()}: {conf:.2f}"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return display
