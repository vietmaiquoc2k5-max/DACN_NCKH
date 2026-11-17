#!/usr/bin/env python3
"""Export and Benchmark Infrastructure for Week 6.
Handles runtime inspection, dependency verification, model export, and latency/FPS benchmarking.
Separates model-only from end-to-end benchmarking; never fakes unavailable runtimes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
LOG = logging.getLogger('export_benchmark')


def inspect_environment() -> dict[str, Any]:
    """Inspect all execution runtimes and deployment packages without modifying environment."""
    env_info: dict[str, Any] = {
        'os': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cpu_name': platform.processor() or 'x86_64',
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    }

    backends: dict[str, Any] = {}

    # 1. PyTorch native (CUDA & CPU)
    backends['pytorch_cuda'] = {
        'available': torch.cuda.is_available(),
        'status': 'AVAILABLE' if torch.cuda.is_available() else 'UNAVAILABLE_NO_CUDA',
        'version': torch.__version__
    }
    backends['pytorch_cpu'] = {
        'available': True,
        'status': 'AVAILABLE',
        'version': torch.__version__
    }

    # 2. ONNX
    try:
        import onnx
        backends['onnx'] = {
            'available': True,
            'status': 'AVAILABLE',
            'version': getattr(onnx, '__version__', 'unknown')
        }
    except ImportError:
        backends['onnx'] = {
            'available': False,
            'status': 'MISSING_PACKAGE',
            'install_command': 'pip install onnx>=1.14.0'
        }

    # 3. ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        backends['onnxruntime'] = {
            'available': True,
            'status': 'AVAILABLE',
            'version': getattr(ort, '__version__', 'unknown'),
            'providers': providers,
            'has_cuda_provider': 'CUDAExecutionProvider' in providers
        }
    except ImportError:
        backends['onnxruntime'] = {
            'available': False,
            'status': 'MISSING_PACKAGE',
            'install_command': 'pip install onnxruntime-gpu (or onnxruntime for CPU-only)'
        }

    # 4. OpenVINO
    try:
        import openvino as ov
        backends['openvino'] = {
            'available': True,
            'status': 'AVAILABLE',
            'version': getattr(ov, '__version__', 'unknown')
        }
    except ImportError:
        backends['openvino'] = {
            'available': False,
            'status': 'MISSING_PACKAGE',
            'install_command': 'pip install openvino>=2023.2.0'
        }

    # 5. TensorRT
    try:
        import tensorrt as trt
        backends['tensorrt'] = {
            'available': True,
            'status': 'AVAILABLE',
            'version': getattr(trt, '__version__', 'unknown')
        }
    except ImportError:
        backends['tensorrt'] = {
            'available': False,
            'status': 'MISSING_PACKAGE',
            'install_command': 'pip install tensorrt (requires NVIDIA TensorRT CUDA libraries)'
        }

    return {
        'environment': env_info,
        'backends': backends
    }


def benchmark_pytorch_runtime(
    model_path: Path,
    device: str = 'cuda:0',
    half: bool = True,
    imgsz: int = 640,
    warmup_frames: int = 50,
    measured_frames: int = 500,
    batch_size: int = 1
) -> dict[str, Any]:
    """Measure model-only latency vs end-to-end latency with real timing and memory tracking."""
    is_cuda = 'cuda' in device and torch.cuda.is_available()
    dev = torch.device(device if is_cuda else 'cpu')
    use_half = half and is_cuda

    LOG.info('Benchmarking PyTorch: device=%s, half=%s, imgsz=%d, frames=%d', dev, use_half, imgsz, measured_frames)
    
    yolo_model = YOLO(str(model_path))
    nn_model = yolo_model.model.to(dev)
    if use_half:
        nn_model.half()
    nn_model.eval()

    # Pre-allocate dummy input tensor for Model-Only benchmark
    dtype = torch.float16 if use_half else torch.float32
    dummy_input = torch.randn(batch_size, 3, imgsz, imgsz, device=dev, dtype=dtype)

    # 1. Warmup Model-Only
    with torch.no_grad():
        for _ in range(warmup_frames):
            _ = nn_model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()

    # 2. Benchmark Model-Only Latency
    model_latencies = []
    process = psutil.Process(os.getpid())
    cpu_usages = []
    
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for i in range(measured_frames):
            if i % 100 == 0:
                cpu_usages.append(process.cpu_percent())
            t0 = time.perf_counter()
            _ = nn_model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            model_latencies.append((t1 - t0) * 1000.0)  # ms

    vram_peak_mb = (torch.cuda.max_memory_allocated(dev) / (1024**2)) if is_cuda else 0.0
    ram_mb = process.memory_info().rss / (1024**2)

    model_latencies_np = np.array(model_latencies)
    mean_lat = float(np.mean(model_latencies_np))
    p50_lat = float(np.median(model_latencies_np))
    p95_lat = float(np.percentile(model_latencies_np, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    # 3. End-to-End Latency Benchmark (Preprocess + Model + Postprocess NMS via Ultralytics predict)
    dummy_numpy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    e2e_latencies = []

    # Warmup E2E
    for _ in range(min(warmup_frames, 20)):
        _ = yolo_model.predict(dummy_numpy_img, device=device, half=use_half, imgsz=imgsz, verbose=False)
        if is_cuda:
            torch.cuda.synchronize()

    for _ in range(min(measured_frames, 200)):
        t0 = time.perf_counter()
        _ = yolo_model.predict(dummy_numpy_img, device=device, half=use_half, imgsz=imgsz, verbose=False)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_latencies.append((t1 - t0) * 1000.0)

    e2e_latencies_np = np.array(e2e_latencies)
    e2e_mean_lat = float(np.mean(e2e_latencies_np))
    e2e_fps = 1000.0 / e2e_mean_lat if e2e_mean_lat > 0 else 0.0

    return {
        'runtime': f"pytorch_{'cuda' if is_cuda else 'cpu'}",
        'device': str(dev),
        'precision': 'FP16' if use_half else 'FP32',
        'batch_size': batch_size,
        'imgsz': imgsz,
        'warmup_frames': warmup_frames,
        'measured_frames': measured_frames,
        'model_only': {
            'fps': round(fps, 2),
            'mean_latency_ms': round(mean_lat, 3),
            'p50_latency_ms': round(p50_lat, 3),
            'p95_latency_ms': round(p95_lat, 3)
        },
        'end_to_end': {
            'fps': round(e2e_fps, 2),
            'mean_latency_ms': round(e2e_mean_lat, 3),
            'p50_latency_ms': round(float(np.median(e2e_latencies_np)), 3),
            'p95_latency_ms': round(float(np.percentile(e2e_latencies_np, 95)), 3)
        },
        'resources': {
            'ram_usage_mb': round(ram_mb, 2),
            'peak_vram_mb': round(vram_peak_mb, 2),
            'cpu_usage_percent': round(float(np.mean(cpu_usages)) if cpu_usages else 0.0, 1)
        }
    }


def export_and_benchmark(
    checkpoint_path: Path,
    output_dir: Path,
    warmup: int = 50,
    frames: int = 500,
    imgsz: int = 640
) -> dict[str, Any]:
    """Execute complete dependency inspection, available exports and benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = inspect_environment()

    # Save dependency report
    dep_path = output_dir / 'dependency_report.json'
    dep_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    LOG.info('Environment inspection saved to %s', dep_path)

    benchmarks: dict[str, Any] = {}

    # Benchmark PyTorch GPU if available
    if report['backends']['pytorch_cuda']['available']:
        try:
            benchmarks['pytorch_cuda_fp16'] = benchmark_pytorch_runtime(
                checkpoint_path, device='cuda:0', half=True, imgsz=imgsz,
                warmup_frames=warmup, measured_frames=frames
            )
        except Exception as err:
            LOG.error('PyTorch CUDA benchmark failed: %s', err)
            benchmarks['pytorch_cuda_fp16'] = {'error': str(err)}

    # Benchmark PyTorch CPU
    try:
        benchmarks['pytorch_cpu_fp32'] = benchmark_pytorch_runtime(
            checkpoint_path, device='cpu', half=False, imgsz=imgsz,
            warmup_frames=min(warmup, 20), measured_frames=min(frames, 100)
        )
    except Exception as err:
        LOG.error('PyTorch CPU benchmark failed: %s', err)
        benchmarks['pytorch_cpu_fp32'] = {'error': str(err)}

    # Export ONNX if available
    if report['backends']['onnx']['available']:
        LOG.info('ONNX is installed. Attempting model export...')
        try:
            m = YOLO(str(checkpoint_path))
            onnx_path = m.export(format='onnx', imgsz=imgsz, dynamic=False, opset=12)
            benchmarks['onnx_export'] = {
                'status': 'EXPORTED',
                'exported_path': str(onnx_path),
                'file_size_bytes': Path(onnx_path).stat().st_size
            }
        except Exception as err:
            LOG.error('ONNX export failed: %s', err)
            benchmarks['onnx_export'] = {'status': 'FAILED', 'error': str(err)}
    else:
        benchmarks['onnx_export'] = {
            'status': 'BLOCKED_DEPENDENCY_UNAVAILABLE',
            'missing': 'onnx'
        }

    # OpenVINO & TensorRT status
    for b_name in ['openvino', 'tensorrt']:
        if not report['backends'][b_name]['available']:
            benchmarks[f'{b_name}_export'] = {
                'status': 'BLOCKED_DEPENDENCY_UNAVAILABLE',
                'missing': b_name
            }

    final_payload = {
        'model_path': str(checkpoint_path),
        'model_size_bytes': checkpoint_path.stat().st_size,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': report['environment'],
        'backend_status': report['backends'],
        'benchmarks': benchmarks
    }

    out_file = output_dir / 'benchmark_results.json'
    out_file.write_text(json.dumps(final_payload, indent=2), encoding='utf-8')
    LOG.info('Benchmark results saved to %s', out_file)
    return final_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ROOT / 'runs/benchmarks/week5_domain_adaptation/final_gpu_handoff/checkpoints/strategy_c_best.pt'
    )
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/benchmarks/week6_export_benchmarks')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--inspect-only', action='store_true', help='Only inspect environment and output dependency report')
    args = parser.parse_args()

    if args.inspect_only:
        report = inspect_environment()
        print(json.dumps(report, indent=2))
        return 0

    try:
        export_and_benchmark(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            warmup=args.warmup,
            frames=args.frames,
            imgsz=args.imgsz
        )
        return 0
    except Exception as err:
        LOG.error('Export and benchmark runner failed: %s', err)
        return 1


if __name__ == '__main__':
    sys.exit(main())
