import os
import sys
import cv2
import pandas as pd
from pathlib import Path

# Đảm bảo in UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH = Path("data/video_dataset_list.csv")

def tag_video_timestamps(prefix=None):
    if not CSV_PATH.exists():
        print(f"❌ Không tìm thấy file {CSV_PATH}. Hãy chạy scripts/index_test_videos.py trước.")
        return

    df = pd.read_csv(CSV_PATH)
    pos_mask = df["category"].str.contains("pos")
    if prefix:
        pos_mask &= df["video_id"].astype(str).str.startswith(prefix)
    pos_indices = df[pos_mask].index.tolist()

    print("\n" + "="*70)
    print("⏱️ CÔNG CỤ ĐÁNH DẤU MỐC THỜI GIAN VIDEO CHÁY/KHÓI (TIMESTAMPS TAGGER)")
    print("="*70)
    print("Phím tắt điều khiển khi xem video:")
    print("  • [Space] : Tạm dừng / Tiếp tục phát")
    print("  • [S]     : Đánh dấu GIÂY BẮT ĐẦU CÓ KHÓI tại khung hình hiện tại")
    print("  • [F]     : Đánh dấu GIÂY BẮT ĐẦU CÓ LỬA tại khung hình hiện tại")
    print("  • [R]     : Tua lại từ đầu video")
    print("  • [N / Enter] : Lưu mốc video hiện tại & chuyển sang video kế tiếp")
    print("  • [Q / ESC]   : Lưu toàn bộ tiến độ và thoát")
    print("="*70 + "\n")

    for idx in pos_indices:
        row = df.loc[idx]
        v_path = Path(row["relative_path"])
        if not v_path.exists():
            print(f"⚠️ Bỏ qua {row['file_name']}: Không tìm thấy file tại {v_path}")
            continue

        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur_sec = total_frames / fps if fps > 0 else 0

        cur_smoke_sec = row.get("smoke_start_sec", -1)
        cur_fire_sec = row.get("fire_start_sec", -1)

        print(f"\n▶️ Đang xem: [{row['video_id']}] {row['file_name']} | Độ dài: {dur_sec:.1f}s | FPS: {fps:.1f}")
        print(f"   Mốc hiện tại -> Smoke: {cur_smoke_sec}s | Fire: {cur_fire_sec}s")

        paused = False
        win_name = f"Timestamp Tagger - [{row['video_id']}] {row['file_name']}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 500)

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Hết video, tua lại đầu
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            cur_sec = round(cur_frame / fps, 2) if fps > 0 else 0

            # Vẽ thanh trạng thái
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, h - 70), (w, h), (15, 23, 42), -1)
            display_frame = cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0)

            info_text = f"Time: {cur_sec:.1f}s / {dur_sec:.1f}s  [Frame: {cur_frame}/{total_frames}]"
            tag_text = f"Tagged -> Smoke: {cur_smoke_sec}s | Fire: {cur_fire_sec}s"
            help_text = "[S]: Tag Smoke | [F]: Tag Fire | [Space]: Pause | [Enter]: Next"

            cv2.putText(display_frame, info_text, (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 1, cv2.LINE_AA)
            cv2.putText(display_frame, tag_text, (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)
            cv2.putText(display_frame, help_text, (w - 380, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            if paused:
                cv2.putText(display_frame, "|| PAUSED", (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            cv2.imshow(win_name, display_frame)

            wait_time = 0 if paused else int(1000 / fps)
            key = cv2.waitKey(max(1, wait_time)) & 0xFF

            if key == ord(' '):
                paused = not paused
            elif key in [ord('s'), ord('S')]:
                cur_smoke_sec = round(cur_sec, 1)
                print(f"  🏷️ Đã gán Khói (Smoke) bắt đầu tại: {cur_smoke_sec}s")
            elif key in [ord('f'), ord('F')]:
                cur_fire_sec = round(cur_sec, 1)
                print(f"  🏷️ Đã gán Lửa (Fire) bắt đầu tại: {cur_fire_sec}s")
            elif key in [ord('r'), ord('R')]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key in [13, ord('n'), ord('N')]: # Enter or N
                break
            elif key in [27, ord('q'), ord('Q')]: # ESC or Q
                cap.release()
                cv2.destroyAllWindows()
                df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                print(f"\n💾 Đã lưu tiến độ mốc thời gian vào {CSV_PATH}!")
                return

        cap.release()
        cv2.destroyWindow(win_name)

        # Cập nhật vào DataFrame
        df.loc[idx, "smoke_start_sec"] = cur_smoke_sec
        df.loc[idx, "fire_start_sec"] = cur_fire_sec
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"  ✅ Đã cập nhật [{row['video_id']}]: Smoke={cur_smoke_sec}s, Fire={cur_fire_sec}s")

    cv2.destroyAllWindows()
    print(f"\n🎉 HOÀN TẤT GÁN MỐC THỜI GIAN TOÀN BỘ VIDEO! File đã lưu tại: {CSV_PATH}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None, help="Chỉ gán mốc cho video_id bắt đầu bằng chuỗi này, vd. VN_")
    tag_video_timestamps(ap.parse_args().prefix)
