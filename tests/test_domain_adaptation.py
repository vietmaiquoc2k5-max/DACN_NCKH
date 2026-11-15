"""Unit tests for Domain Adaptation module (Tuần 5 - MT4)."""
import os
import tempfile
import unittest
from pathlib import Path

from src.domain_adaptation.prepare_replay_data import (
    sample_dfire_train,
    create_replay_dataset,
    PROJECT_ROOT
)
from src.domain_adaptation.train_da import (
    STRATEGIES_CONFIG,
    verify_freeze_layers,
    train_single_strategy
)
from src.domain_adaptation.eval_da import (
    compute_adaptation_delta,
    verify_mt4_compliance,
    format_evaluation_table
)


class TestDomainAdaptation(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho toàn bộ module Thích nghi miền."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # --- Tests cho prepare_replay_data.py ---

    def test_sample_dfire_train_is_deterministic(self):
        """Kiểm tra việc lấy mẫu 20% D-Fire là tất định khi cùng seed."""
        dfire_root = PROJECT_ROOT / "datasets" / "dfire"
        if not (dfire_root / "train" / "images").exists():
            self.skipTest("Thư mục datasets/dfire không tồn tại trên môi trường")

        s1 = sample_dfire_train(dfire_root, ratio=0.20, seed=42)
        s2 = sample_dfire_train(dfire_root, ratio=0.20, seed=42)
        self.assertEqual(len(s1), len(s2))
        self.assertEqual(s1, s2)
        self.assertGreater(len(s1), 0)

    def test_sample_dfire_train_different_seeds_differ(self):
        """Khác seed phải cho kết quả chọn mẫu khác nhau."""
        dfire_root = PROJECT_ROOT / "datasets" / "dfire"
        if not (dfire_root / "train" / "images").exists():
            self.skipTest("Thư mục datasets/dfire không tồn tại trên môi trường")

        s1 = sample_dfire_train(dfire_root, ratio=0.20, seed=42)
        s2 = sample_dfire_train(dfire_root, ratio=0.20, seed=999)
        self.assertEqual(len(s1), len(s2))
        self.assertNotEqual(s1, s2)

    def test_create_replay_dataset_generates_valid_files(self):
        """Tạo tập replay và kiểm tra file txt, yaml và manifest được tạo đúng chuẩn."""
        dfire_root = PROJECT_ROOT / "datasets" / "dfire"
        vn_root = PROJECT_ROOT / "datasets" / "vn_dataset"
        if not (dfire_root / "train" / "images").exists() or not (vn_root / "images" / "train").exists():
            self.skipTest("Thiếu dữ liệu dfire hoặc vn_dataset để chạy test")

        out_txt = self.tmp_path / "replay_train.txt"
        out_yaml = self.tmp_path / "replay_dataset.yaml"
        manifest_p = self.tmp_path / "manifest.json"

        manifest = create_replay_dataset(
            dfire_root=dfire_root,
            vn_root=vn_root,
            output_txt=out_txt,
            output_yaml=out_yaml,
            manifest_path=manifest_p,
            dfire_ratio=0.01,  # 1% cho test nhanh
            seed=42
        )

        self.assertTrue(out_txt.exists())
        self.assertTrue(out_yaml.exists())
        self.assertTrue(manifest_p.exists())
        self.assertEqual(manifest["seed"], 42)
        self.assertGreater(manifest["total_train_images"], 0)

        # Kiểm tra nội dung txt không rỗng
        lines = [line.strip() for line in out_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), manifest["total_train_images"])

    # --- Tests cho train_da.py ---

    def test_strategies_config_completeness(self):
        """Đảm bảo 3 chiến lược đề cương đều được định nghĩa đủ."""
        self.assertIn("full", STRATEGIES_CONFIG)
        self.assertIn("freeze", STRATEGIES_CONFIG)
        self.assertIn("replay", STRATEGIES_CONFIG)

        self.assertIsNone(STRATEGIES_CONFIG["full"]["freeze"])
        self.assertEqual(STRATEGIES_CONFIG["freeze"]["freeze"], 10)
        self.assertEqual(STRATEGIES_CONFIG["replay"]["freeze"], 10)

    def test_train_da_dry_run(self):
        """Kiểm tra chức năng dry-run trên tất cả 3 chiến lược."""
        base_weights = PROJECT_ROOT / "best.pt"
        if not base_weights.exists():
            self.skipTest("Không tìm thấy best.pt")

        for strat in ["full", "freeze", "replay"]:
            with self.subTest(strategy=strat):
                res = train_single_strategy(
                    strategy_key=strat,
                    base_weights=base_weights,
                    output_dir=self.tmp_path,
                    epochs=1,
                    seed=42,
                    dry_run=True
                )
                self.assertEqual(res["status"], "dry_run_success")
                self.assertEqual(res["strategy"], strat)
                self.assertIn("freeze_info", res)
                f_info = res["freeze_info"]
                if strat == "full":
                    self.assertEqual(f_info["frozen_param_count"], 0)
                    self.assertEqual(f_info["trainable_param_count"], f_info["total_param_count"])
                else:
                    self.assertEqual(f_info["freeze_count"], 10)
                    self.assertGreater(f_info["frozen_param_count"], 0)
                    self.assertGreater(f_info["trainable_param_count"], 0)
                    self.assertEqual(f_info["frozen_param_count"] + f_info["trainable_param_count"], f_info["total_param_count"])

    def test_train_da_invalid_strategy_raises(self):
        """Chiến lược lạ phải raise ValueError."""
        base_weights = PROJECT_ROOT / "best.pt"
        with self.assertRaises(ValueError):
            train_single_strategy("unknown_strategy", base_weights, self.tmp_path, dry_run=True)

    # --- Tests cho eval_da.py ---

    def test_compute_adaptation_delta(self):
        """Kiểm tra công thức tính độ lệch mAP (Delta)."""
        base = {"map50": 0.80, "map50_95": 0.60, "recall": 0.75, "precision": 0.85}
        da = {"map50": 0.86, "map50_95": 0.64, "recall": 0.80, "precision": 0.88}

        deltas = compute_adaptation_delta(base, da)
        self.assertAlmostEqual(deltas["delta_map50"], 0.06, places=5)
        self.assertAlmostEqual(deltas["delta_map50_percentage"], 6.0, places=3)
        self.assertAlmostEqual(deltas["delta_map50_95"], 0.04, places=5)
        self.assertAlmostEqual(deltas["delta_recall"], 0.05, places=5)

    def test_verify_mt4_compliance(self):
        """Kiểm tra logic đánh giá đạt/không đạt chỉ tiêu MT4 (+5% VN, -2% D-Fire)."""
        # Case 1: Đạt cả hai (VN tăng +5.4%, D-Fire giảm -0.8%) -> PASS
        vn_ok, df_ok, overall = verify_mt4_compliance(delta_vn=0.054, delta_dfire=-0.008)
        self.assertTrue(vn_ok)
        self.assertTrue(df_ok)
        self.assertTrue(overall)

        # Case 2: VN tăng mạnh (+6%), nhưng D-Fire quên nhiều (-4%) -> FAIL
        vn_ok, df_ok, overall = verify_mt4_compliance(delta_vn=0.060, delta_dfire=-0.040)
        self.assertTrue(vn_ok)
        self.assertFalse(df_ok)
        self.assertFalse(overall)

        # Case 3: D-Fire giữ tốt (-1%), nhưng VN tăng ít (+3%) -> FAIL
        vn_ok, df_ok, overall = verify_mt4_compliance(delta_vn=0.030, delta_dfire=-0.010)
        self.assertFalse(vn_ok)
        self.assertTrue(df_ok)
        self.assertFalse(overall)

        # Case 4: Ngưỡng biên chính xác (+0.05 và -0.02) -> PASS
        vn_ok, df_ok, overall = verify_mt4_compliance(delta_vn=0.050, delta_dfire=-0.020)
        self.assertTrue(vn_ok)
        self.assertTrue(df_ok)
        self.assertTrue(overall)

    def test_format_evaluation_table(self):
        """Kiểm tra hàm format bảng markdown xuất ra đúng cấu trúc bảng."""
        mock_eval = {
            "baseline": {
                "vn": {"map50": 0.8244},
                "dfire": {"map50": 0.7602}
            },
            "strategy_3_replay": {
                "vn": {"map50": 0.8785},
                "dfire": {"map50": 0.7518},
                "delta_vn": {"delta_map50": 0.0541},
                "delta_dfire": {"delta_map50": -0.0084}
            }
        }
        table_str = format_evaluation_table(mock_eval)
        self.assertIn("| Mô hình / Chiến lược |", table_str)
        self.assertIn("Baseline (`best.pt`)", table_str)
        self.assertIn("82.44%", table_str)
        self.assertIn("+5.41%", table_str)
        self.assertIn("-0.84%", table_str)
        self.assertIn("PASS", table_str)


if __name__ == "__main__":
    unittest.main()

