import time
import os
import sys
import psutil
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.camera_manager import MultiCameraManager


def benchmark():
    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)
    print("--- BENCHMARK FIREWATCH COMMAND CENTER: 1 / 2 / 4 / 6 / 9 CAMERAS ---")
    manager = MultiCameraManager(db_path="benchmark_alerts.db", model_path="best.pt")
    manager.settings.mute_audio = True
    manager.alert_manager.update_config(True)
    manager.apply_performance_preset("Balanced")

    det_info = manager.detector.get_model_info()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {det_info.get('device')} ({gpu_name}) | FP16: {det_info.get('use_fp16')}")

    test_modes = [
        ("SINGLE (1 CAM)", ["CAM-01"]),
        ("DUAL (2 CAMS)", ["CAM-01", "CAM-02"]),
        ("QUAD (4 CAMS)", ["CAM-01", "CAM-02", "CAM-03", "CAM-04"]),
        ("HEXA (6 CAMS)", ["CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"]),
        ("FULL (9 CAMS)", [f"CAM-{i:02d}" for i in range(1, 10)]),
    ]

    results = []

    # 0. Đo trạng thái khởi động mặc định STOPPED (Manual Start V2: 0 camera kết nối)
    manager.stop_monitoring()
    manager.set_active_camera_selection(["CAM-01", "CAM-02", "CAM-03", "CAM-04"], auto_start=False)
    psutil.cpu_percent(interval=None)
    time.sleep(1.5)
    stopped_cpu = psutil.cpu_percent(interval=0.5)
    stopped_ram = process.memory_info().rss / (1024 * 1024)
    stopped_gpu = torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    results.append({
        "mode": "STOPPED (STANDBY 0 CAM)",
        "cams_count": 0,
        "avg_cap_fps": 0.0,
        "tot_cap_fps": 0.0,
        "avg_inf_fps": 0.0,
        "tot_inf_fps": 0.0,
        "avg_lat_ms": 0.0,
        "cpu_pct": round(stopped_cpu, 1),
        "ram_mb": round(stopped_ram, 1),
        "gpu_mb": round(stopped_gpu, 1),
    })
    print(
        f"[STOPPED (STANDBY 0 CAM)] Active Workers: {manager.get_active_worker_count()} | "
        f"CPU: {stopped_cpu:.1f}% | RAM: {stopped_ram:.1f}MB | GPU VRAM: {stopped_gpu:.1f}MB"
    )

    for name, cams in test_modes:
        manager.set_active_camera_selection(cams, auto_start=True)
        # Chờ luồng ổn định 3.5 giây và đo CPU trong khoảng thời gian chạy
        psutil.cpu_percent(interval=None)
        time.sleep(3.5)
        cpu_pct = psutil.cpu_percent(interval=0.5)

        snapshots = manager.get_selected_snapshots(cams)
        cap_fps_list = [s.get("capture_fps", 0.0) for s in snapshots if s.get("is_connected")]
        inf_fps_list = [s.get("inference_fps", 0.0) for s in snapshots if s.get("is_connected")]
        lat_list = [s.get("infer_ms", 0.0) for s in snapshots if s.get("is_connected")]

        avg_cap_fps = sum(cap_fps_list) / len(cap_fps_list) if cap_fps_list else 0.0
        avg_inf_fps = sum(inf_fps_list) / len(inf_fps_list) if inf_fps_list else 0.0
        avg_lat = sum(lat_list) / len(lat_list) if lat_list else 0.0
        tot_cap_fps = sum(cap_fps_list)
        tot_inf_fps = sum(inf_fps_list)

        mem_info = process.memory_info()
        ram_mb = mem_info.rss / (1024 * 1024)

        gpu_mb = 0.0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)

        res = {
            "mode": name,
            "cams_count": len(cams),
            "avg_cap_fps": round(avg_cap_fps, 1),
            "tot_cap_fps": round(tot_cap_fps, 1),
            "avg_inf_fps": round(avg_inf_fps, 1),
            "tot_inf_fps": round(tot_inf_fps, 1),
            "avg_lat_ms": round(avg_lat, 1),
            "cpu_pct": round(cpu_pct, 1),
            "ram_mb": round(ram_mb, 1),
            "gpu_mb": round(gpu_mb, 1),
        }
        results.append(res)
        print(
            f"[{name}] Cams: {len(cams)} | Avg Cap FPS: {avg_cap_fps:.1f} | "
            f"Tot Cap FPS: {tot_cap_fps:.1f} | Avg Infer FPS: {avg_inf_fps:.1f} | "
            f"Tot Infer FPS: {tot_inf_fps:.1f} | Latency: {avg_lat:.1f}ms | "
            f"CPU: {cpu_pct:.1f}% | RAM: {ram_mb:.1f}MB | GPU VRAM: {gpu_mb:.1f}MB"
        )

    manager.stop_all()
    if os.path.exists("benchmark_alerts.db"):
        try:
            os.remove("benchmark_alerts.db")
        except OSError:
            pass

    print("\nSUMMARY TABLE:")
    print("| Chế độ | Số camera | Capture FPS TB/Cam | Tổng Capture FPS | Inference FPS TB/Cam | Tổng AI FPS | Latency TB (ms) | CPU (%) | RAM (MB) | GPU VRAM (MB) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['mode']} | {r['cams_count']} | {r['avg_cap_fps']} | {r['tot_cap_fps']} | "
            f"{r['avg_inf_fps']} | {r['tot_inf_fps']} | {r['avg_lat_ms']} | "
            f"{r['cpu_pct']}% | {r['ram_mb']} | {r['gpu_mb']} |"
        )


if __name__ == "__main__":
    benchmark()
