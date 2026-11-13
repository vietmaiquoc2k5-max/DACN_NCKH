"""Cấu hình trung tâm cho ứng dụng Web Dashboard Phát hiện & Cảnh báo Cháy 9 Camera."""
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional

_orig_path_resolve = Path.resolve


def _safe_path_resolve(self: Path, strict: bool = False) -> Path:
    try:
        return _orig_path_resolve(self, strict=strict)
    except PermissionError:
        return Path(os.path.abspath(os.fspath(self)))


Path.resolve = _safe_path_resolve  # type: ignore[method-assign]

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent

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
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolo11n_seed123" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolo11n_seed456" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed123" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed42" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks" / "yolov8n_seed456" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "FireSmokeLabeler" / "model" / "best.pt"),
    str(PROJECT_ROOT / "models" / "fire_smoke_v2.pt"),
]


def get_available_model_choices() -> Dict[str, str]:
    """Liệt kê tất cả các checkpoint mô hình có sẵn trong dự án (bao gồm các lượt benchmark MT2)."""
    choices: Dict[str, str] = {}
    root_best = PROJECT_ROOT / "best.pt"
    if root_best.exists():
        choices["best.pt (Model Mặc Định)"] = str(root_best)

    bench_dir = PROJECT_ROOT / "runs" / "detect" / "runs" / "benchmarks"
    if bench_dir.exists():
        for run_dir in sorted(bench_dir.iterdir()):
            w_path = run_dir / "weights" / "best.pt"
            if w_path.exists():
                choices[f"{run_dir.name} (Benchmark MT2)"] = str(w_path)

    if not choices:
        choices["best.pt"] = str(root_best)
    return choices

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
    """Ẩn toàn bộ thông tin xác thực trong URL để không lộ tài khoản trong giao diện hoặc log.

    Ví dụ:
      'rtsp://admin:MySecret123@192.168.1.50:554/Streaming/Channels/101'
      -> 'rtsp://***@192.168.1.50:554/Streaming/Channels/101'
    """
    if not source:
        return ""
    s = str(source).strip()
    return re.sub(
        r"((?:rtsp|rtsps|http|https|rtmp)://)[^/@\s]+@",
        r"\1***@",
        s,
        flags=re.IGNORECASE,
    )


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
    inference_fps: int = 24
    loop_video: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_interval_sec: float = 2.5
    exclusion_zones: List[Tuple[float, float, float, float]] = field(default_factory=list)
    source_reference: str = ""
    telegram_enabled: bool = True

    @property
    def masked_source(self) -> str:
        """Trả về chuỗi nguồn đã ẩn mật khẩu RTSP (an toàn để hiển thị và ghi log)."""
        if not str(self.source).strip() or self.source_type == "UNASSIGNED":
            return "Chưa gán nguồn"
        if self.source_type == "WEBCAM" or str(self.source).strip().isdigit():
            return f"Webcam #{self.webcam_index}"
        return mask_rtsp_credentials(self.source)


def get_available_test_videos() -> Dict[str, str]:
    """Liệt kê danh sách các file video kiểm thử có sẵn trên ổ đĩa để người dùng chủ động chọn khi test cam."""
    videos: Dict[str, str] = {}
    vn_dir = PROJECT_ROOT / "videovietnam"
    if vn_dir.exists():
        for p in sorted(vn_dir.glob("*.mp4")):
            rel = f"videovietnam/{p.name}"
            videos[f"🇻🇳 {p.name} ({rel})"] = rel

    pos_dir = PROJECT_ROOT / "VN_POS_21"
    if pos_dir.exists():
        for p in sorted(pos_dir.glob("*.mp4")):
            rel = f"VN_POS_21/{p.name}"
            videos[f"🔥 {p.name} ({rel})"] = rel

    neg_dir = PROJECT_ROOT / "VN_NEG_05"
    if neg_dir.exists():
        for p in sorted(neg_dir.glob("*.mp4")):
            rel = f"VN_NEG_05/{p.name}"
            videos[f"🛡️ {p.name} ({rel})"] = rel

    ds_neg_dir = PROJECT_ROOT / "datasets" / "negative_videos"
    if ds_neg_dir.exists():
        for p in sorted(ds_neg_dir.glob("*.mp4")):
            rel = f"datasets/negative_videos/{p.name}"
            videos[f"🛡️ {p.name} ({rel})"] = rel

    raw_neg_dir = PROJECT_ROOT / "negative_videos"
    if raw_neg_dir.exists():
        for p in sorted(raw_neg_dir.glob("*.mp4")):
            rel = f"negative_videos/{p.name}"
            prefix = "🔥" if "POS" in p.name.upper() else "🛡️"
            videos[f"{prefix} {p.name} ({rel})"] = rel

    fsl_dir = PROJECT_ROOT / "FireSmokeLabeler" / "videos"
    if fsl_dir.exists():
        for p in sorted(fsl_dir.iterdir()):
            if p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv"):
                rel = f"FireSmokeLabeler/videos/{p.name}"
                videos[f"🎞️ {p.name} ({rel})"] = rel

    uploads_dir = PROJECT_ROOT / "uploads"
    if uploads_dir.exists():
        for p in sorted(uploads_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                rel = f"uploads/{p.name}"
                videos[f"📤 [Đã tải lên] {p.name}"] = rel

    return videos


def get_default_9_cameras(auto_assign_videos: bool = True) -> List[CameraConfig]:
    """Khởi tạo cấu hình mặc định cho 9 camera (3x3).
    - Khi `auto_assign_videos=False` (dùng trên giao diện Web Dashboard): KHÔNG tự động gắn sẵn file video vào camera,
      mọi camera khởi tạo ở trạng thái trống (`source=""`, `source_type="UNASSIGNED"`) để người dùng tự chọn camera đang hoạt động
      hoặc chủ động gắn file video qua tính năng Test Cam bằng File.
    """
    candidate_videos = [
        ("CAM-01", "Khu Vực Sảnh Chính", "videovietnam/VN_POS_01.mp4"),
        ("CAM-02", "Xưởng Sản Xuất", "videovietnam/VN_POS_02.mp4"),
        ("CAM-03", "Kho Vật Liệu", "videovietnam/VN_POS_03.mp4"),
        ("CAM-04", "Hành Lang Tầng 2", "videovietnam/VN_POS_04.mp4"),
        ("CAM-05", "Trạm Biến Áp", "videovietnam/VN_POS_05.mp4"),
        ("CAM-06", "Khu Vực Nhà Xe", "videovietnam/VN_POS_06.mp4"),
        ("CAM-07", "Cổng Chính", "videovietnam/VN_POS_07.mp4"),
        ("CAM-08", "Sân Thượng", "videovietnam/VN_POS_08.mp4"),
        ("CAM-09", "Hành Lang Kỹ Thuật", "videovietnam/VN_NEG_01.mp4"),
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
        if not auto_assign_videos:
            src = ""
            s_type = "UNASSIGNED"
        else:
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
class ByteTrackSettings:
    """Cấu hình tham số cho thuật toán theo dõi ByteTrack chính thức."""
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.35
    track_buffer: int = 15
    match_thresh: float = 0.8
    fuse_score: bool = False


@dataclass
class SystemSettings:
    """Tham số vận hành và tối ưu hóa phần cứng toàn hệ thống (Đồng bộ với Operating Point chuẩn hóa)."""
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
    window_size_n: int = 8
    confirm_frames_k: int = 2
    suspect_frames_k: int = 1
    alert_cooldown_sec: float = 5.0
    feature_threshold: float = 0.02
    mute_audio: bool = False
    bytetrack: ByteTrackSettings = field(default_factory=ByteTrackSettings)
    per_class_settings: Optional[Dict[str, Dict[str, Any]]] = None

