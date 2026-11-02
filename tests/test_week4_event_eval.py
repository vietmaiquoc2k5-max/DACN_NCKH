"""Unit tests for the Week 4 acceptance evaluator."""
import unittest

from src.week4_event_eval import (
    Event, FrameDetection, evaluate, frame_detections_to_events, grid_search,
)


class Week4EvaluationTests(unittest.TestCase):
    """Test proposal-aligned video recall, FAR and K/N search."""

    def setUp(self) -> None:
        self.gt = [Event("pos-1", "CAM-01", "FIRE", 2.0, 8.0)]
        self.pos = {"pos-1", "pos-2"}
        self.neg = {"neg-1"}

    def test_video_level_event_recall(self) -> None:
        pred = [Event("pos-1", "CAM-01", "FIRE", 3.0, 5.0)]
        report = evaluate(self.gt, pred, self.pos, self.neg, 1.0)
        self.assertEqual(report["metrics"]["video_event_recall"], 0.5)

    def test_negative_video_false_alarm_rate(self) -> None:
        pred = [
            Event("pos-1", "CAM-01", "FIRE", 3.0, 5.0),
            Event("neg-1", "CAM-09", "FIRE", 10.0, 11.0),
            Event("pos-2", "CAM-02", "FIRE", 1.0, 1.0),
        ]
        report = evaluate(self.gt, pred, self.pos, self.neg, 2.0)
        self.assertEqual(report["counts"]["negative_false_alarms"], 1)
        self.assertEqual(report["metrics"]["false_alarms_per_hour"], 0.5)

    def test_k_n_confirmation_delays_alert(self) -> None:
        frames = [
            FrameDetection("pos-1", "CAM-01", float(t), (
                {"label": "FIRE", "confidence": 0.9, "track_id": 1},
            ) if t in (1, 2, 3) else ())
            for t in range(5)
        ]
        baseline, temporal = frame_detections_to_events(frames, 0.5, 3, 2)
        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(temporal), 1)
        self.assertGreaterEqual(temporal[0].start_sec, baseline[0].start_sec)

    def test_grid_search_returns_candidates(self) -> None:
        frames = [
            FrameDetection("pos-1", "CAM-01", float(t), (
                {"label": "FIRE", "confidence": 0.9, "track_id": 1},
            ) if t in (2, 3, 4, 5) else ())
            for t in range(8)
        ]
        rows = grid_search(
            frames, self.gt, self.pos, self.neg, 1.0,
            confidence_thresholds=[0.5], k_values=[2], n_values=[3]
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("feasible", rows[0])

    def test_invalid_k_n_rejected(self) -> None:
        frames = [FrameDetection("v", "c", 0.0, ())]
        with self.assertRaises(ValueError):
            frame_detections_to_events(frames, 0.5, 2, 3)


if __name__ == "__main__":
    unittest.main()
