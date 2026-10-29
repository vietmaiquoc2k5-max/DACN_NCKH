"""Baseline B2: Phân loại khung hình phát hiện cháy & khói bằng mạng MobileNetV3-Small.

Thuộc đề cương NCKH: MT2 - Bộ phát hiện và các Baseline so sánh.
Đặc trưng:
- Kiến trúc nhẹ (MobileNetV3-Small ~2.54M tham số).
- Phân loại cấp độ khung hình (Frame-level Multi-label Classification: Smoke, Fire).
- Dùng làm mốc so sánh (Baseline B2) với:
  + Baseline B1 (Luật màu HSV/YCbCr + Trừ nền MOG2).
  + Các mạng phát hiện đối tượng YOLOv8, YOLO11, RT-DETR.
"""
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class MobileNetV3FireClassifier(nn.Module):
    """Mạng MobileNetV3-Small tùy biến cho bài toán nhận diện khói & lửa cấp độ khung hình."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """Khởi tạo mô hình.
        
        Args:
            num_classes: 2 lớp độc lập (0: smoke, 1: fire).
            pretrained: Sử dụng trọng số ImageNet tiền huấn luyện.
        """
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Lấy kích thước feature in của classifier gốc (thường là 1024)
        in_features = self.backbone.classifier[0].in_features
        # Thay thế classifier head bằng cấu trúc tùy chỉnh cho 2 lớp
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass trả về raw logits kích thước (B, num_classes)."""
        return self.backbone(x)


class FireSmokeClassifierB2:
    """Wrapper quản lý nạp mô hình, tiền xử lý, suy luận và huấn luyện cho Baseline B2."""

    CLASS_NAMES = {0: "smoke", 1: "fire"}

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_thresh: float = 0.5
    ):
        """Khởi tạo Classifier.
        
        Args:
            model_path: Đường dẫn tới file trọng số .pt (nếu có).
            device: 'cuda', 'cpu' hoặc None (tự động chọn).
            conf_thresh: Ngưỡng xác suất phân loại (mặc định 0.5).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.conf_thresh = conf_thresh
        self.model = MobileNetV3FireClassifier(num_classes=2, pretrained=True)

        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            print(f"✅ Đã tải trọng số Baseline B2 từ: {model_path}")

        self.model.to(self.device)
        self.model.eval()

        # Chuẩn hóa ảnh đầu vào ImageNet tiêu chuẩn 224x224
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Dự đoán nhãn khói và lửa trên 1 khung hình BGR (từ OpenCV).
        
        Returns:
            Dict chứa xác suất khói, lửa, trạng thái phát hiện và độ trễ inference (ms).
        """
        t_start = time.perf_counter()

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            # Áp dụng Sigmoid cho bài toán phân loại đa nhãn (Multi-label)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        smoke_prob = float(probs[0])
        fire_prob = float(probs[1])

        has_smoke = smoke_prob >= self.conf_thresh
        has_fire = fire_prob >= self.conf_thresh

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if has_fire and has_smoke:
            label = "Fire & Smoke"
            status = "FIRE_ALARM"
        elif has_fire:
            label = "Fire"
            status = "FIRE_ALARM"
        elif has_smoke:
            label = "Smoke"
            status = "SMOKE_ALARM"
        else:
            label = "Normal"
            status = "NORMAL"

        return {
            "label": label,
            "status": status,
            "has_smoke": has_smoke,
            "has_fire": has_fire,
            "smoke_conf": smoke_prob,
            "fire_conf": fire_prob,
            "latency_ms": latency_ms,
            "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0
        }

    def annotate_frame(self, frame_bgr: np.ndarray, pred: Dict[str, Any]) -> np.ndarray:
        """Vẽ thông tin dự đoán của Baseline B2 lên khung hình."""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        color = (0, 255, 0) # Normal: Xanh lá
        if pred["status"] == "FIRE_ALARM":
            color = (0, 0, 255) # Đỏ
        elif pred["status"] == "SMOKE_ALARM":
            color = (0, 165, 255) # Cam

        # Banner trạng thái trên cùng
        banner_h = 75
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (30, 30, 30), -1)

        text_title = f"[Baseline B2 - MobileNetV3] Status: {pred['label']}"
        cv2.putText(annotated, text_title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        text_details = (
            f"Smoke: {pred['smoke_conf']*100:.1f}% | "
            f"Fire: {pred['fire_conf']*100:.1f}% | "
            f"Latency: {pred['latency_ms']:.1f}ms ({pred['fps']:.1f} FPS)"
        )
        cv2.putText(annotated, text_details, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Viền cảnh báo nếu có cháy hoặc khói
        if pred["status"] != "NORMAL":
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, 4)

        return annotated


class DFireClassificationDataset(torch.utils.data.Dataset):
    """Dataset đọc dữ liệu từ thư mục chuẩn D-Fire để huấn luyện/đánh giá phân loại cấp khung hình."""

    def __init__(self, images_dir: Path, labels_dir: Path, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.jpeg"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_files[idx]
        stem = img_path.stem
        label_file = self.labels_dir / f"{stem}.txt"

        # Đọc nhãn ground truth: 0: smoke, 1: fire
        # Multi-label vector: [has_smoke, has_fire]
        target = torch.zeros(2, dtype=torch.float32)

        if label_file.exists():
            try:
                content = label_file.read_text(encoding="utf-8", errors="ignore").strip()
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            target[0] = 1.0  # smoke
                        elif cls_id == 1:
                            target[1] = 1.0  # fire
            except Exception:
                pass

        # Đọc ảnh bằng PIL để tương thích torchvision transform
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, target, stem
