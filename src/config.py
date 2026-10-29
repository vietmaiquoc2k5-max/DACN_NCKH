"""Cấu hình trung tâm cho ứng dụng Web Dashboard Phát hiện & Cảnh báo Cháy 9 Camera."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


@dataclass
class CameraConfig:
    """Cấu hình cho từng luồng camera trong lưới 3x3 (9 camera)."""
    cam_id: str
    name: str
    source: str
    enabled: bool = True
    loop_video: bool = True
    exclusion_zones: List[Tuple[float, float, float, float]] = field(default_factory=list)


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
        configs.append(
            CameraConfig(
                cam_id=cid,
                name=cname,
                source=src,
                enabled=(idx < 4),  # Mặc định bật sẵn 4 luồng đầu theo MT5 để tiết kiệm tài nguyên, có thể bật cả 9
                loop_video=True,
            )
        )
    return configs


@dataclass
class SystemSettings:
    """Tham số vận hành và tối ưu hóa phần cứng toàn hệ thống."""
    model_path: str = "best.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640
    target_fps_per_cam: int = 12
    display_width: int = 480
    frame_skip: int = 1
    window_size_n: int = 10
    confirm_frames_k: int = 4
    suspect_frames_k: int = 2
    alert_cooldown_sec: float = 12.0
    mute_audio: bool = False

