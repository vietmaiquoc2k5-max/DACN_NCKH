"""Unit tests cho hàm tính khoảng tin cậy Poisson chính xác 95% (tests/test_poisson_ci.py).

Kiểm tra:
1. Trường hợp 0 false alarm (k=0): lower bound phải là 0.0, upper bound > 0.0.
2. Trường hợp 1 false alarm (k=1): lower bound > 0.0, upper bound > point estimate.
3. Trường hợp số false alarms lớn (k=67, k=354, k=421, k=547, k=625, k=988): không bị underflow/tràn số.
4. Nhiều mức exposure khác nhau (T=1.0h, T=5.0h, T=7.2026h, T=24.0h, T=100.0h).
5. Point estimate (k / T) phải luôn nằm chặt trong khoảng tin cậy [lower, upper].
6. Tính đơn điệu: khi k tăng thì lower và upper đều tăng đơn điệu; khi exposure tăng thì bounds thu hẹp.
7. Xử lý ngoại lệ đầu vào không hợp lệ (k < 0, exposure <= 0).
"""
import unittest
from src.event_eval import exact_poisson_ci


class TestExactPoissonCI(unittest.TestCase):
    def test_zero_false_alarms(self):
        """1. Kiểm tra 0 false alarm: lower=0.0, upper > 0.0 và point estimate nằm trong khoảng."""
        exposure = 7.2026
        low, high = exact_poisson_ci(0, exposure)
        point = 0.0 / exposure
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertTrue(low <= point <= high)
        self.assertAlmostEqual(high, 0.5122, places=3)

    def test_one_false_alarm(self):
        """2. Kiểm tra 1 false alarm: lower > 0.0, point estimate nằm trong khoảng."""
        exposure = 7.2026
        low, high = exact_poisson_ci(1, exposure)
        point = 1.0 / exposure
        self.assertGreater(low, 0.0)
        self.assertGreater(high, point)
        self.assertTrue(low <= point <= high)

    def test_large_false_alarms(self):
        """3. Kiểm tra số false alarms lớn (k=67, 354, 421, 547, 625, 988): không bị lỗi underflow float."""
        exposure = 7.2026
        large_k_values = [67, 140, 354, 407, 421, 547, 625, 988]
        for k in large_k_values:
            low, high = exact_poisson_ci(k, exposure)
            point = k / exposure
            self.assertGreater(low, 0.0)
            self.assertGreater(high, low)
            self.assertTrue(
                low <= point <= high,
                f"Lỗi: Point estimate {point} không nằm trong [{low}, {high}] cho k={k}",
            )

    def test_various_exposure_hours(self):
        """4. Kiểm tra nhiều mức exposure khác nhau từ 0.5h đến 100h."""
        exposures = [0.5, 1.0, 5.0, 7.2026, 12.0, 24.0, 100.0]
        for exp in exposures:
            for k in [0, 1, 10, 50, 100, 421]:
                low, high = exact_poisson_ci(k, exp)
                point = k / exp
                self.assertTrue(
                    low <= point <= high,
                    f"Point estimate {point} không nằm trong [{low}, {high}] tại exposure {exp}h",
                )

    def test_monotonicity_wrt_k(self):
        """5. Kiểm tra tính đơn điệu tăng theo k khi cố định exposure."""
        exposure = 7.2026
        prev_low = -1.0
        prev_high = -1.0
        for k in range(0, 50, 5):
            low, high = exact_poisson_ci(k, exposure)
            self.assertGreater(low, prev_low)
            self.assertGreater(high, prev_high)
            prev_low = low
            prev_high = high

    def test_exposure_narrowing(self):
        """6. Kiểm tra khi exposure tăng thì độ rộng khoảng tin cậy (width) phải thu hẹp."""
        k = 50
        w1 = exact_poisson_ci(k, 5.0)[1] - exact_poisson_ci(k, 5.0)[0]
        w2 = exact_poisson_ci(k, 10.0)[1] - exact_poisson_ci(k, 10.0)[0]
        w3 = exact_poisson_ci(k, 50.0)[1] - exact_poisson_ci(k, 50.0)[0]
        self.assertGreater(w1, w2)
        self.assertGreater(w2, w3)

    def test_invalid_inputs(self):
        """7. Kiểm tra xử lý ngoại lệ đầu vào không hợp lệ."""
        # exposure <= 0 trả về (0.0, 0.0)
        self.assertEqual(exact_poisson_ci(10, 0.0), (0.0, 0.0))
        self.assertEqual(exact_poisson_ci(10, -5.0), (0.0, 0.0))
        # k < 0 ném ValueError
        with self.assertRaises(ValueError):
            exact_poisson_ci(-1, 7.2026)


if __name__ == "__main__":
    unittest.main()

