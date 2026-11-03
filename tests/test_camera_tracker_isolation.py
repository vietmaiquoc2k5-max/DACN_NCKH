"""Bộ kiểm thử tính cô lập tracker giữa các camera (`tests/test_camera_tracker_isolation.py`).

Kiểm tra:
1. Mỗi CameraStreamWorker sở hữu một instance ByteTrackAdapter và TemporalFeatureExtractor riêng biệt.
2. Hai camera không dùng chung tracker state hay ID counter.
3. Reset camera 1 không làm ảnh hưởng đến trạng thái tracking của camera 2.
4. Reconnect camera xóa trạng thái track cũ mà không làm gián đoạn camera khác.
5. Định dạng DetectionBox sau khi qua pipeline tương thích hoàn toàn với Dashboard.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.config import CameraConfig, SystemSettings
from src.temporal_verifier import DetectionBox
from src.camera_manager import CameraStreamWorker
from src.detector import SharedFireSmokeDetector
from src.database import EventDatabase


class TestCameraTrackerIsolation(unittest.TestCase):
    """Kiểm thử tính độc lập của tracker trên từng camera worker."""

    def setUp(self):
        self.settings = SystemSettings()
        self.mock_detector = MagicMock(spec=SharedFireSmokeDetector)
        self.mock_event_db = MagicMock(spec=EventDatabase)

        self.cfg1 = CameraConfig(
            cam_id="CAM-01",
            name="Camera Kho Hàng",
            source="FireSmokeLabeler/videos/testpos01.817.avi",
        )
        self.cfg2 = CameraConfig(
            cam_id="CAM-02",
            name="Camera Bãi Xe",
            source="FireSmokeLabeler/videos/posVideo1.868.avi",
        )

        self.worker1 = CameraStreamWorker(
            config=self.cfg1,
            detector=self.mock_detector,
            event_db=self.mock_event_db,
            settings=self.settings,
        )
        self.worker2 = CameraStreamWorker(
            config=self.cfg2,
            detector=self.mock_detector,
            event_db=self.mock_event_db,
            settings=self.settings,
        )

    def test_distinct_tracker_and_feature_instances(self):
        """1. Kiểm tra hai worker không trỏ tới cùng một instance tracker hoặc extractor."""
        self.assertIsNot(
            self.worker1.tracker,
            self.worker2.tracker,
            "Mỗi worker phải có ByteTrackAdapter riêng biệt.",
        )
        self.assertIsNot(
            self.worker1.features_extractor,
            self.worker2.features_extractor,
            "Mỗi worker phải có TemporalFeatureExtractor riêng biệt.",
        )
        self.assertIsNot(
            self.worker1.verifier,
            self.worker2.verifier,
            "Mỗi worker phải có TemporalVerifier riêng biệt.",
        )

    def test_independent_tracking_states(self):
        """2. Kiểm tra tracking state độc lập: đối tượng ở camera 1 không liên quan camera 2."""
        box_cam1 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(50, 50, 100, 100),
            norm_xywh=(0.1, 0.1, 0.1, 0.1),
        )
        box_cam2 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.91,
            xyxy=(200, 200, 260, 260),
            norm_xywh=(0.3, 0.3, 0.1, 0.1),
        )

        tracked1 = self.worker1.tracker.update([box_cam1])
        tracked2 = self.worker2.tracker.update([box_cam2])

        self.assertEqual(tracked1[0].track_id, 1)
        self.assertEqual(tracked2[0].track_id, 1)
        self.assertEqual(self.worker1.tracker.active_tracks_count(), 1)
        self.assertEqual(self.worker2.tracker.active_tracks_count(), 1)

    def test_reset_isolation(self):
        """3. Kiểm tra reset worker 1 không làm mất trạng thái tracking của worker 2."""
        box_cam1 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(50, 50, 100, 100),
            norm_xywh=(0.1, 0.1, 0.1, 0.1),
        )
        box_cam2 = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.91,
            xyxy=(200, 200, 260, 260),
            norm_xywh=(0.3, 0.3, 0.1, 0.1),
        )

        self.worker1.tracker.update([box_cam1])
        self.worker2.tracker.update([box_cam2])

        # Reset camera 1
        self.worker1.tracker.reset()
        self.worker1.verifier.reset()
        self.worker1.features_extractor.reset()

        self.assertEqual(
            self.worker1.tracker.active_tracks_count(),
            0,
            "Worker 1 phải bị xóa toàn bộ track sau reset.",
        )
        self.assertEqual(
            self.worker2.tracker.active_tracks_count(),
            1,
            "Worker 2 vẫn phải giữ nguyên 1 track đang hoạt động.",
        )

    def test_reconnect_clears_old_tracks(self):
        """4. Kiểm tra reconnect xóa sạch track cũ của camera đó."""
        box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.88,
            xyxy=(50, 50, 100, 100),
            norm_xywh=(0.1, 0.1, 0.1, 0.1),
        )
        self.worker1.tracker.update([box])
        self.assertEqual(self.worker1.tracker.active_tracks_count(), 1)

        # Mô phỏng reconnect
        self.worker1.tracker.reset()
        self.worker1.verifier.reset()
        self.worker1.features_extractor.reset()

        self.assertEqual(self.worker1.tracker.active_tracks_count(), 0)

    def test_detection_box_format_compatibility(self):
        """5. Kiểm tra định dạng DetectionBox sau khi qua tracker và verifier vẫn đúng chuẩn cho Dashboard."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw_box = DetectionBox(
            cls_id=1,
            label="FIRE",
            confidence=0.85,
            xyxy=(100, 100, 200, 200),
            norm_xywh=(0.23, 0.31, 0.15, 0.20),
        )

        # Mô phỏng pipeline trong worker
        tracked = self.worker1.tracker.update([raw_box], frame=dummy_frame)
        self.assertIsInstance(tracked, list)
        self.assertIsInstance(tracked[0], DetectionBox)
        self.assertGreater(tracked[0].track_id, 0)

        # Xác minh qua verifier
        ver_res = self.worker1.verifier.update(tracked, tracks_assigned=True)
        valid_boxes = ver_res["valid_boxes"]
        self.assertIsInstance(valid_boxes, list)
        self.assertEqual(len(valid_boxes), 1)

        box_out = valid_boxes[0]
        self.assertIsInstance(box_out, DetectionBox)
        self.assertEqual(box_out.label, "FIRE")
        self.assertEqual(box_out.cls_id, 1)
        self.assertEqual(box_out.xyxy, (100, 100, 200, 200))
        self.assertGreater(box_out.track_id, 0)

        # Vẽ thử annotations (Dashboard format)
        annotated = SharedFireSmokeDetector.draw_annotations(
            frame=dummy_frame,
            boxes=valid_boxes,
            cam_name="CAM-01",
            alert_level=ver_res["alert_level"],
            fps=25.0,
        )
        self.assertEqual(annotated.shape, dummy_frame.shape)


if __name__ == "__main__":
    unittest.main()
