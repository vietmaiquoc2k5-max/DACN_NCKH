import time
import os
import sys
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import SystemSettings
from src.camera_manager import MultiCameraManager

def benchmark():
    process = psutil.Process(os.getpid())
    print("--- BENCHMARK DYNAMIC CAMERA SYSTEM ---")
    manager = MultiCameraManager(db_path="benchmark_alerts.db", model_path="best.pt")
    manager.settings.performance_mode = "Balanced"
    manager.start_all()
    time.sleep(2) # Cho workers nạp

    test_modes = [
        ("SINGLE (1 CAM)", ["CAM-01"]),
        ("QUAD (4 CAMS)", ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]),
        ("HEXA (6 CAMS)", ["CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"]),
        ("FULL (9 CAMS)", [f"CAM-{i:02d}" for i in range(1, 10)])
    ]

    results = []

    for name, cams in test_modes:
        manager.set_active_camera_selection(cams, auto_start=True)
        # Chờ ổn định 5 giây
        time.sleep(5)
        
        # Lấy snapshots và stats
        snapshots = manager.get_selected_snapshots(cams)
        cap_fps_list = [s.get("capture_fps", 0.0) for s in snapshots if s.get("is_connected")]
        inf_fps_list = [s.get("inference_fps", 0.0) for s in snapshots if s.get("is_connected")]
        lat_list = [s.get("infer_ms", 0.0) for s in snapshots if s.get("is_connected")]
        
        avg_cap_fps = sum(cap_fps_list) / len(cap_fps_list) if cap_fps_list else 0.0
        avg_inf_fps = sum(inf_fps_list) / len(inf_fps_list) if inf_fps_list else 0.0
        avg_lat = sum(lat_list) / len(lat_list) if lat_list else 0.0
        tot_cap_fps = sum(cap_fps_list)
        
        mem_info = process.memory_info()
        ram_mb = mem_info.rss / (1024 * 1024)
        
        res = {
            "mode": name,
            "cams_count": len(cams),
            "avg_cap_fps": round(avg_cap_fps, 1),
            "tot_cap_fps": round(tot_cap_fps, 1),
            "avg_inf_fps": round(avg_inf_fps, 1),
            "avg_lat_ms": round(avg_lat, 1),
            "ram_mb": round(ram_mb, 1)
        }
        results.append(res)
        print(f"[{name}] Cams: {len(cams)} | Avg Cap FPS: {avg_cap_fps:.1f} | Tot Cap FPS: {tot_cap_fps:.1f} | Avg Infer FPS: {avg_inf_fps:.1f} | Latency: {avg_lat:.1f}ms | RAM: {ram_mb:.1f}MB")

    manager.stop_all()
    print("\nSUMMARY TABLE:")
    print("| Chế độ | Số camera | Capture FPS TB / Cam | Tổng Capture FPS | Inference FPS TB | Latency TB (ms) | RAM (MB) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['mode']} | {r['cams_count']} | {r['avg_cap_fps']} | {r['tot_cap_fps']} | {r['avg_inf_fps']} | {r['avg_lat_ms']} | {r['ram_mb']} |")

if __name__ == "__main__":
    benchmark()
