"""Opt-in hardware throughput check; no configured cap is treated as a result."""
import os
import time

import pytest

from src.camera_manager import MultiCameraManager


pytestmark = [pytest.mark.hardware]


@pytest.mark.skipif(os.getenv("FIREWATCH_RUN_PERFORMANCE") != "1",
                    reason="requires explicit camera/model hardware benchmark opt-in")
@pytest.mark.parametrize("camera_count", [1, 4, 9])
def test_measured_camera_inference_throughput(tmp_path, camera_count):
    sources = [os.getenv(f"FIREWATCH_PERF_SOURCE_{i}", "") for i in range(1, camera_count + 1)]
    assert all(sources), f"Configure {camera_count} FIREWATCH_PERF_SOURCE_n inputs"
    manager = MultiCameraManager(
        db_path=str(tmp_path / f"performance-{camera_count}.db"),
        model_path=os.getenv("FIREWATCH_MODEL_PATH", "best.pt"),
        auto_assign_videos=False,
    )
    try:
        ids = [manager.add_camera(f"Perf {i}", source, source_type="IP_CAMERA",
                                  source_reference=source, auto_start=False, inference_fps=60)
               for i, source in enumerate(sources, 1)]
        for camera_id in ids:
            manager.start_camera(camera_id)
        time.sleep(120)
        result = {s["cam_id"]: s for s in manager.get_all_snapshots()}
        assert all(result[camera_id]["processed_frames"] > 0 for camera_id in ids)
        # Report observed rates for the operator without imposing hardware-specific pass claims.
        observed = {camera_id: result[camera_id]["inference_fps"] for camera_id in ids}
        print(f"Measured inference FPS for {camera_count} cameras: {observed}")
    finally:
        manager.close()
