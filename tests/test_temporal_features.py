"""Tests for temporal appearance and geometry features."""
import unittest
import numpy as np

from src.temporal_features import TemporalFeatureExtractor


class TemporalFeatureTests(unittest.TestCase):
    def test_area_growth_is_positive_for_expanding_box(self) -> None:
        extractor = TemporalFeatureExtractor()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        extractor.update(1, frame, (10, 10, 30, 30), 0.0)
        extractor.update(1, frame, (10, 10, 40, 40), 1.0)
        result = extractor.update(1, frame, (10, 10, 45, 45), 2.0)
        self.assertGreater(result.area_growth_rate, 0.0)
        self.assertGreater(result.smoke_expansion_score, 0.0)

    def test_flicker_descriptor_uses_roi_intensity_history(self) -> None:
        extractor = TemporalFeatureExtractor()
        outputs = []
        for timestamp, value in enumerate((20, 220, 30)):
            frame = np.full((50, 50, 3), value, dtype=np.uint8)
            outputs.append(extractor.update(7, frame, (5, 5, 30, 30), float(timestamp)))
        self.assertGreater(outputs[-1].flicker_score, 0.0)

    def test_invalid_box_is_rejected(self) -> None:
        extractor = TemporalFeatureExtractor()
        with self.assertRaises(ValueError):
            extractor.update(1, np.zeros((10, 10, 3), dtype=np.uint8), (5, 5, 5, 8), 0.0)


if __name__ == "__main__":
    unittest.main()
