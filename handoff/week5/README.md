# GÓI BÀN GIAO THỰC NGHIỆM DOMAIN ADAPTATION TUẦN 5 (PORTABLE HANDOFF PACKAGE)

Tài liệu này hướng dẫn người nhận (Recipient) thiết lập môi trường, xác thực tính toàn vẹn dữ liệu và thực thi chuỗi thực nghiệm Tuần 5 (Baseline, Chiến lược A, B, C) trên máy tính trang bị GPU NVIDIA.

---

## 1. Cấu trúc thư mục gói bàn giao `handoff/week5/`

```text
handoff/week5/
├── README.md                      # Tài liệu hướng dẫn này
├── checksums.sha256               # Bảng mã băm SHA256 của toàn bộ 2.366 tệp trong gói
├── dataset_manifest.json          # Manifest huấn luyện di động (portable relative paths)
├── protocol.json                  # Cấu hình protocol thích ứng GPU (seeds: [42], epochs: 30)
├── budget_receipt_gpu.json        # Biên bản dự toán tài nguyên cho GPU
├── budget_receipt_cpu.json        # Biên bản dự toán tài nguyên cho CPU (fallback kiểm thử)
├── manifests/
│   ├── clean_vn_manifest.json     # 786 ảnh VN (585 train, 201 val) đã duyệt người & lineage
│   ├── clean_dfire_manifest.json  # 14.956 ảnh D-Fire sạch (12.409 train, 2.547 val)
│   └── dfire_replay_candidates.json # 147 ảnh D-Fire replay đã chọn cho Chiến lược C
├── labels/                        # 786 nhãn .txt và 786 biên bản phê duyệt .approval.json
└── lineage/                       # 786 biên bản xác thực trích xuất khung hình gốc .json
```

---

## 2. Kiểm tra tính toàn vẹn gói bàn giao

Trước khi bắt đầu, hãy xác thực checksum của toàn bộ tệp trong `handoff/week5/`:

```powershell
python -c "import hashlib, os; [print('FAIL:', line) for line in open('handoff/week5/checksums.sha256') if hashlib.sha256(open(os.path.join('handoff/week5', line.strip().split('  ')[1]), 'rb').read()).hexdigest() != line.strip().split('  ')[0]]; print('All handoff checksums verified!')"
```

---

## 3. Hướng dẫn thiết lập tập dữ liệu (Datasets Setup)

### 3.1. Tập dữ liệu Việt Nam (`datasets/vn_dataset/`)
- Toàn bộ 3.121 ảnh gốc Việt Nam đã được đồng bộ sẵn trong repository tại `datasets/vn_dataset/images/` (1.868 train, 623 val, 630 test).
- Các nhãn chuẩn khoa học đã được duyệt tay độc lập (human-approved QA) và biên bản trích xuất lineage nằm ngay trong `handoff/week5/labels/` và `handoff/week5/lineage/`.
- Manifest `handoff/week5/dataset_manifest.json` đã trỏ trực tiếp đến các đường dẫn tương đối này.

### 3.2. Tập dữ liệu D-Fire (`datasets/dfire_author_linked_kaggle_v1/`)
Do dung lượng D-Fire lớn (~2.5 GB), tệp ảnh D-Fire không lưu trực tiếp trên git history thông thường. Người nhận có thể phục hồi theo một trong hai cách:

1. **Cách 1 (Khuyến nghị): Tải từ nguồn Kaggle chính thức**
   - Tải dataset D-Fire từ Kaggle: `https://www.kaggle.com/datasets/gpiosenka/dfire-dataset` hoặc repo gốc của tác giả Pedro Vinícius A. B. Venâncio (`https://github.com/pedro-va-b/d-fire-dataset`).
   - Đặt thư mục theo cấu trúc:
     ```text
     datasets/dfire_author_linked_kaggle_v1/data/
     ├── train/
     │   ├── images/
     │   └── labels/
     └── val/
         ├── images/
         └── labels/
     ```
2. **Cách 2: Sử dụng gói trích xuất D-Fire sạch**
   - Chạy lệnh kiểm tra tính hợp lệ D-Fire:
     ```powershell
     python scripts/verify_week5_handoff.py --check-dfire
     ```

---

## 4. Khởi chạy thực nghiệm nhanh bằng Entry Point

Script duy nhất để kiểm tra và huấn luyện trên máy mới: `scripts/run_week5_handoff.py`.

```powershell
# 1. Chạy GPU Smoke Test (xác thực CUDA, load checkpoint, kiểm tra 1 batch)
python scripts/run_week5_handoff.py --smoke-test

# 2. Huấn luyện Chiến lược A (Full Fine-tune)
python scripts/run_week5_handoff.py --strategy full_finetune --device 0

# 3. Huấn luyện Chiến lược B (Freeze Backbone)
python scripts/run_week5_handoff.py --strategy freeze_backbone --device 0

# 4. Huấn luyện Chiến lược C (Freeze + 20% D-Fire Replay)
python scripts/run_week5_handoff.py --strategy replay_20_percent --device 0

# 5. Đánh giá và tổng hợp so sánh mAP & Catastrophic Forgetting
python scripts/run_week5_handoff.py --compare
```

