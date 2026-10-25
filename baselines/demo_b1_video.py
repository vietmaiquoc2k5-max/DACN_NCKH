"""Chạy thử nghiệm Baseline B1 (Luật màu + Trừ nền MOG2) trên video thực tế.

Hiển thị hoặc lưu video gồm:
- Khung hình gốc có vẽ Bounding Box phát hiện đám cháy / khói.
- Cửa sổ mặt nạ chuyển động MOG2.
- Cửa sổ mặt nạ màu sắc (HSV/YCbCr).
- Thống kê FPS thời gian thực.
"""
import sys
import time
import argparse
from pathlib import Path
import cv2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines.baseline_b1 import RuleBasedFireSmokeDetector


def run_video(video_path: str, output_path: str = None, display: bool = True, max_frames: int = 500, imgsz: int = 640):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print("\n" + "=" * 70)
    print("🎥 CHẠY THỬ NGHIỆM BASELINE B1 TRÊN VIDEO")
    print("=" * 70)
    print(f"• File video: {video_path}")
    print(f"• Độ phân giải: {w}x{h} | FPS: {fps:.1f} | Tổng khung hình: {total_frames}")
    if imgsz:
        print(f"• Kích thước xử lý (imgsz): {imgsz}px")

    detector = RuleBasedFireSmokeDetector(use_motion=True, min_area=150, imgsz=imgsz)

    writer = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))
        print(f"• Xuất video kết quả tại: {out_p.resolve()}")

    frame_count = 0
    total_detect_time = 0.0
    detected_fire_frames = 0
    detected_smoke_frames = 0

    win_name = "Baseline B1 - Fire & Smoke Detection (MOG2 + Color Rules)"
    if display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 960, 540)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break

        frame_count += 1
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        total_detect_time += dt

        has_fire = any(d["class_id"] == 1 for d in detections)
        has_smoke = any(d["class_id"] == 0 for d in detections)
        if has_fire:
            detected_fire_frames += 1
        if has_smoke:
            detected_smoke_frames += 1

        annotated = detector.draw_detections(frame, detections)

        # Vẽ thanh thông tin FPS
        cur_fps = 1000.0 / dt if dt > 0 else 0
        info_text = f"Frame: {frame_count} | Latency: {dt:.1f}ms ({cur_fps:.1f} FPS) | Fire: {has_fire} | Smoke: {has_smoke}"
        cv2.rectangle(annotated, (0, 0), (w, 35), (20, 20, 20), -1)
        cv2.putText(annotated, info_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(annotated)

        if display:
            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    avg_latency = total_detect_time / frame_count if frame_count > 0 else 0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("-" * 70)
    print(f"✅ Đã xử lý {frame_count} khung hình.")
    print(f"• Độ trễ trung bình: {avg_latency:.2f} ms")
    print(f"• Tốc độ trung bình: {avg_fps:.1f} FPS")
    print(f"• Khung hình phát hiện Lửa: {detected_fire_frames}/{frame_count} ({detected_fire_frames/frame_count*100:.1f}%)")
    print(f"• Khung hình phát hiện Khói: {detected_smoke_frames}/{frame_count} ({detected_smoke_frames/frame_count*100:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm Baseline B1 trên video")
    parser.add_argument("--video", default="videovietnam/VN_POS_03.mp4", help="Đường dẫn file video")
    parser.add_argument("--output", default=None, help="Đường dẫn lưu file video kết quả")
    parser.add_argument("--no-display", action="store_true", help="Không mở cửa sổ GUI")
    parser.add_argument("--max-frames", type=int, default=300, help="Số khung hình tối đa cần test")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước cạnh dài resize tăng tốc (mặc định 640)")
    args = parser.parse_args()

    run_video(args.video, output_path=args.output, display=not args.no_display, max_frames=args.max_frames, imgsz=args.imgsz)
