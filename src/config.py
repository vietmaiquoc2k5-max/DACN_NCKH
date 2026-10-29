"""Cấu hình trung tâm cho ứng dụng Web Dashboard Phát hiện & Cảnh báo Cháy 9 Camera."""
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Danh sách các chế độ vận hành của hệ thống
OPERATING_MODES: Dict[str, str] = {
    "MULTI_CAMERA": "Nhiều Camera (Lưới 3×3 - Giám sát đồng thời)",
    "VIDEO_TEST": "Video Kiểm Thử (VIDEO_TEST)",
    "WEBCAM": "Camera Máy Tính (WEBCAM)",
    "IP_CAMERA": "Camera Giám Sát IP/RTSP (IP_CAMERA)",
}

# Các loại nguồn đầu vào cho từng camera
SOURCE_TYPES: List[str] = ["VIDEO_TEST", "WEBCAM", "IP_CAMERA"]

# Danh sách các đường dẫn model đã huấn luyện của dự án (ưu tiên best.pt)
CANDIDATE_MODEL_PATHS: List[str] = [
    str(PROJECT_ROOT / "best.pt"),
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed123" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed456" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "FireSmokeLabeler" / "model" / "best.pt"),
    str(PROJECT_ROOT / "models" / "fire_smoke_v2.pt"),
]

# Chuẩn hóa tên lớp theo dataset thực tế của dự án (0: smoke, 1: fire)
EXPECTED_CLASSES: Dict[int, str] = {
    0: "SMOKE",
    1: "FIRE",
}

# Màu sắc BGR (cho OpenCV) và HEX (cho giao diện Streamlit)
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "FIRE": (36, 36, 239),    # Đỏ tươi nổi bật (#EF2424)
    "SMOKE": (11, 158, 245),  # Cam vàng cảnh báo khói (#F59E0B)
}

CLASS_COLORS_HEX: Dict[str, str] = {
    "FIRE": "#EF4444",
    "SMOKE": "#F59E0B",
}

# Tài khoản đăng nhập (lấy từ biến môi trường nếu có, không hardcode thông tin nhạy cảm thực tế)
DEFAULT_USERS: Dict[str, Dict[str, str]] = {
    "admin": {
        "password": os.environ.get("FIRE_ADMIN_PASSWORD", "admin123"),
        "role": "ADMIN",
        "name": "Quản Trị Viên PCCC",
    },
    "user": {
        "password": os.environ.get("FIRE_USER_PASSWORD", "user123"),
        "role": "USER",
        "name": "Cán Bộ Trực Giám Sát",
    },
}


def mask_rtsp_credentials(source: str) -> str:
    """Ẩn mật khẩu xác thực trong URL RTSP/HTTP để không bao giờ lộ trên giao diện hoặc trong log.

    Ví dụ:
      'rtsp://admin:MySecret123@192.168.1.50:554/Streaming/Channels/101'
      -> 'rtsp://admin:******@192.168.1.50:554/Streaming/Channels/101'
    """
    if not source:
        return ""
    s = str(source).strip()
    return re.sub(r"((?:rtsp|rtsps|http|https|rtmp)://[^:/@\s]+:)([^@\s]+)(@)", r"\1******\3", s, flags=re.IGNORECASE)


def infer_source_type(source: str) -> str:
    """Tự động suy luận loại nguồn đầu vào từ chuỗi cấu hình."""
    s = str(source).strip()
    if s.isdigit():
        return "WEBCAM"
    if s.lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://")):
        return "IP_CAMERA"
    return "VIDEO_TEST"


@dataclass
class CameraConfig:
    """Cấu hình cho từng luồng camera trong lưới 3x3 (9 camera)."""
    cam_id: str
    name: str
    source: str
    source_type: str = "VIDEO_TEST"  # "VIDEO_TEST" | "WEBCAM" | "IP_CAMERA"
    webcam_index: int = 0
    enabled: bool = True
    loop_video: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_interval_sec: float = 2.5
    exclusion_zones: List[Tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def masked_source(self) -> str:
        """Trả về chuỗi nguồn đã ẩn mật khẩu RTSP (an toàn để hiển thị và ghi log)."""
        if self.source_type == "WEBCAM" or str(self.source).strip().isdigit():
            return f"Webcam #{self.webcam_index}"
        return mask_rtsp_credentials(self.source)


def get_default_9_cameras() -> List[CameraConfig]:
    """Khởi tạo cấu hình mặc định cho 9 camera (3x3) sử dụng các video thực tế có sẵn trong repo."""
    candidate_videos = [
        ("CAM-01", "CAM 01 - Kho Hàng A (VN_POS_01)", "videovietnam/VN_POS_01.mp4"),
        ("CAM-02", "CAM 02 - Xưởng Sản Xuất (VN_POS_02)", "videovietnam/VN_POS_02.mp4"),
        ("CAM-03", "CAM 03 - Khu Vực Bãi Xe (VN_POS_03)", "videovietnam/VN_POS_03.mp4"),
        ("CAM-04", "CAM 04 - Hành Lang Tầng 2 (VN_POS_04)", "videovietnam/VN_POS_04.mp4"),
        ("CAM-05", "CAM 05 - Trạm Biến Áp (VN_POS_05)", "videovietnam/VN_POS_05.mp4"),
        ("CAM-06", "CAM 06 - Nhà Kho B (VN_POS_06)", "videovietnam/VN_POS_06.mp4"),
        ("CAM-07", "CAM 07 - Cổng Chính (VN_POS_07)", "videovietnam/VN_POS_07.mp4"),
        ("CAM-08", "CAM 08 - Sân Thượng (VN_POS_08)", "videovietnam/VN_POS_08.mp4"),
        ("CAM-09", "CAM 09 - Giao Thông (VN_NEG_01)", "videovietnam/VN_NEG_01.mp4"),
    ]

    fallback_videos = [
        "FireSmokeLabeler/videos/testpos01.817.avi",
        "FireSmokeLabeler/videos/posVideo1.868.avi",
        "FireSmokeLabeler/videos/posVideo2.871.avi",
        "FireSmokeLabeler/videos/posVideo3.872.avi",
        "FireSmokeLabeler/videos/posVideo4.873.avi",
        "FireSmokeLabeler/videos/posVideo5.874.avi",
        "FireSmokeLabeler/videos/posVideo6.875.avi",
        "FireSmokeLabeler/videos/posVideo7.876.avi",
        "FireSmokeLabeler/videos/posVideo8.877.avi",
    ]

    configs: List[CameraConfig] = []
    for idx, (cid, cname, rel_path) in enumerate(candidate_videos):
        full_p = PROJECT_ROOT / rel_path
        if full_p.exists():
            src = rel_path
        else:
            fb_rel = fallback_videos[idx % len(fallback_videos)]
            fb_full = PROJECT_ROOT / fb_rel
            src = fb_rel if fb_full.exists() else "0"
        s_type = infer_source_type(src)
        configs.append(
            CameraConfig(
                cam_id=cid,
                name=cname,
                source=src,
                source_type=s_type,
                webcam_index=int(src) if src.isdigit() else 0,
                enabled=(idx < 4),
                loop_video=True,
                auto_reconnect=True,
            )
        )
    return configs


@dataclass
class SystemSettings:
    """Tham số vận hành và tối ưu hóa phần cứng toàn hệ thống."""
    operating_mode: str = "MULTI_CAMERA"  # "MULTI_CAMERA" | "VIDEO_TEST" | "WEBCAM" | "IP_CAMERA"
    model_path: str = "best.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 416
    capture_width: int = 480
    target_fps_per_cam: int = 24
    display_width: int = 420
    jpeg_quality: int = 72
    ui_refresh_sec: float = 0.4
    frame_skip: int = 2
    window_size_n: int = 10
    confirm_frames_k: int = 4
    suspect_frames_k: int = 2
    alert_cooldown_sec: float = 12.0
    mute_audio: bool = False
