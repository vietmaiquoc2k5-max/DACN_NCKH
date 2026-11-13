"""Opt-in external RTSP integration test (not a mocked replacement)."""
import os
import time

import pytest

from src.camera_manager import MultiCameraManager


pytestmark = [pytest.mark.integration, pytest.mark.external_service]


@pytest.mark.skipif(os.getenv("FIREWATCH_RUN_RTSP_INTEGRATION") != "1",
                    reason="requires explicit four-stream RTSP integration opt-in")
def test_four_rtsp_streams_decode_and_run_detector():
    sources = [os.getenv(f"FIREWATCH_TEST_RTSP_{index}", "") for index in range(1, 5)]
    assert all(sources), "Set FIREWATCH_TEST_RTSP_1 through _4 to authorized RTSP sources"
    manager = MultiCameraManager(
        db_path=os.getenv("FIREWATCH_DB_PATH", "runs/benchmarks/week7/integration.db"),
        model_path=os.getenv("FIREWATCH_MODEL_PATH", "best.pt"),
        auto_assign_videos=False,
    )
    try:
        camera_ids = [manager.add_camera(
            f"Integration {index}", source, source_type="IP_CAMERA",
            source_reference=source, auto_start=False, inference_fps=8,
        ) for index, source in enumerate(sources, 1)]
        for camera_id in camera_ids:
            manager.start_camera(camera_id)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            snapshots = {s["cam_id"]: s for s in manager.get_all_snapshots()}
            if all(snapshots[cid]["processed_frames"] > 0 for cid in camera_ids):
                break
            time.sleep(1)
        snapshots = {s["cam_id"]: s for s in manager.get_all_snapshots()}
        assert all(snapshots[cid]["captured_frames"] > 0 for cid in camera_ids)
        assert all(snapshots[cid]["processed_frames"] > 0 for cid in camera_ids)
        assert manager.detector.get_model_info()["loaded"]
    finally:
        manager.close()
