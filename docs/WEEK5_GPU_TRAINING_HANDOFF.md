# HƯỚNG DẪN BÀN GIAO & THỰC THI HUẤN LUYỆN DOMAIN ADAPTATION TUẦN 5 TRÊN MÁY GPU (WEEK 5 GPU TRAINING HANDOFF)

---

## 1. TỔNG QUAN & BỐI CẢNH (EXECUTIVE SUMMARY)

- **Dự án:** Nghiên cứu ứng dụng Computer Vision để phát hiện và cảnh báo cháy từ camera giám sát (DACN_NCKH).
- **Hạng mục:** Tuần 5 — Thích nghi miền (Domain Adaptation: Baseline, Chiến lược A, B, C, đo mAP và Catastrophic Forgetting).
- **Repository:** `https://github.com/vietmaiquoc2k5-max/DACN_NCKH`
- **Branch mục tiêu:** `fix-4`
- **Commit gốc đối chiếu:** `62b926aab240616c63e7a10af8df26ec3d38ba92`
- **Lý do bàn giao sang máy GPU thứ hai:**
  Máy trạm ban đầu (Sender) dù trang bị phần cứng GPU NVIDIA RTX 3060 6GB nhưng môi trường Python hệ thống cài đặt bản PyTorch CPU-only (`torch 2.15.0+cpu`). Việc huấn luyện 30 epochs trên CPU tốn khoảng 45 phút cho mỗi chiến lược và hơn 30 giờ cho baseline toàn phần. Để tối ưu hóa tiến độ và tăng tốc 30–50 lần, toàn bộ pipeline thực nghiệm, dữ liệu, nhãn duyệt người, biên bản truy vết và entry point di động được đóng gói để người nhận (Recipient) huấn luyện trực tiếp trên GPU chuyên dụng có hỗ trợ CUDA.

---

## 2. BẢNG KIỂM KÊ PHỤ THUỘC & TRẠNG THÁI GIT (FILE INVENTORY)

| Đường dẫn tệp / thư mục | Trạng thái Git | Kích thước | SHA256 | Vai trò & Tính an toàn công bố |
|---|---|---|---|---|
| `requirements-training.txt` | Mới tạo | 0.3 KB | *(tạo theo cam kết)* | Bắt buộc (pinned dependencies tương thích CUDA) |
| `scripts/run_week5_handoff.py` | Mới tạo | 8.5 KB | *(tạo theo cam kết)* | Bắt buộc (Entry point điều phối duy nhất) |
| `scripts/verify_week5_handoff.py`| Mới tạo | 10.2 KB| *(tạo theo cam kết)* | Bắt buộc (Bộ kiểm tra 13 tiêu chí chất lượng) |
| `scripts/setup_cuda_env.ps1` | Mới tạo | 2.5 KB | *(tạo theo cam kết)* | Tiện ích (PowerShell script tự động cài CUDA) |
| `configs/research/week5_handoff_manifest.json` | Mới tạo | 3.5 KB | *(tạo theo cam kết)* | Bắt buộc (Bản kê khai kỹ thuật di động) |
| `handoff/week5/dataset_manifest.json` | Mới tạo | 10.5 MB | `b72e210b29724943...` | Bắt buộc (Manifest huấn luyện đường dẫn tương đối) |
| `handoff/week5/protocol.json` | Mới tạo | 3.3 KB | `4f386c91a030cb7c...` | Bắt buộc (Cấu hình protocol huấn luyện GPU) |
| `handoff/week5/budget_receipt_gpu.json` | Mới tạo | 0.5 KB | `cbba6f1b3c95977a...` | Bắt buộc (Biên bản ngân sách tài nguyên GPU) |
| `handoff/week5/labels/` (786 txt + 786 json) | Mới tạo | ~1.2 MB | 786 cặp hash khớp 100% | Bắt buộc (Nhãn duyệt tay người & approval receipt) |
| `handoff/week5/lineage/` (786 json) | Mới tạo | ~0.8 MB | 786 hash khớp 100% | Bắt buộc (Biên bản xác thực trích xuất khung hình gốc) |
| `handoff/week5/checksums.sha256` | Mới tạo | 165 KB | *(bảng hash 2.367 tệp)* | Bắt buộc (Kiểm tra toàn vẹn gói handoff) |
| `datasets/vn_dataset/images/` (3.121 ảnh)| Đã track trên Git | ~550 MB | Khớp git tree | Bắt buộc (Toàn bộ ảnh Việt Nam gốc) |
| `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` | Đã track trên Git | 6.1 MB | `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6` | Bắt buộc (Checkpoint Baseline MT2 chính thức) |
| `yolov8n.pt` | Đã track trên Git | 6.4 MB | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | Bắt buộc (Trọng số khởi tạo COCO) |

---

## 3. KIẾN TRÚC DỮ LIỆU & HƯỚNG DẪN THIẾT LẬP (DATASETS)

### 3.1. Tập dữ liệu Việt Nam (`datasets/vn_dataset/`)
- **Số lượng có sẵn:** 3.121 ảnh gốc đã được git track đầy đủ trong `datasets/vn_dataset/images/` (1.868 train, 623 val, 630 test).
- **Tập con khoa học được duyệt cho Tuần 5:** Đúng **786 khung hình** thỏa mãn cả 3 điều kiện:
  1. `human_approved == True`: được chuyên gia duyệt tay độc lập (không dùng nhãn tự sinh của model).
  2. `lineage_verified == True`: có biên bản trích xuất khớp native PTS/ticks/decode từ video nguồn.
  3. `split in {'train', 'val'}`: 585 ảnh train (từ 61 video) và 201 ảnh val (từ 23 video).
- **Tính độc lập:** Hai tập train và val hoàn toàn độc lập về nguồn video (overlap = 0 video), bảo toàn nghiêm ngặt phân hoạch 60/20/20 theo video của Đề cương NCKH.
- **Đại diện lớp:** Tập validation 201 ảnh bao gồm: 107 ảnh có SMOKE (140 instances), 34 ảnh có FIRE (101 instances), và 76 ảnh cảnh âm tính (negative), đảm bảo đầy đủ căn cứ tính mAP@0.5 cho cả 2 lớp.
- **Vị trí nhãn di động:** Đã đóng gói toàn bộ 786 tệp `.txt` và `.approval.json` vào `handoff/week5/labels/`.

### 3.2. Tập dữ liệu D-Fire (`datasets/dfire_author_linked_kaggle_v1/`)
- **Nguồn gốc:** D-Fire dataset của Pedro Vinícius A. B. Venâncio (Kaggle: `gpiosenka/dfire-dataset`).
- **Phân tách sạch (Clean Subset):**
  - Đã loại trừ 340 bbox lỗi tọa độ và 751 nhóm nghi vấn rò rỉ cross-split.
  - Còn lại: **12.409 ảnh clean train** và **2.547 ảnh clean val**.
  - Chiến lược C (Replay) sử dụng đúng $\lceil 585 / 4 \rceil = 147$ ảnh train duy nhất được chọn cố định theo seed 42 (danh sách lưu tại `handoff/week5/manifests/dfire_replay_candidates.json`).
- **Hướng dẫn phục hồi trên máy mới:**
  Tải D-Fire từ Kaggle hoặc copy thư mục `dfire_author_linked_kaggle_v1` đặt vào đường dẫn:
  ```text
  datasets/dfire_author_linked_kaggle_v1/data/
  ├── train/ (images, labels)
  └── val/   (images, labels)
  ```

---

## 4. QUY TRÌNH CÀI ĐẶT MÔI TRƯỜNG CUDA TRÊN MÁY MỚI (RECIPIENT SETUP)

Người nhận chỉ cần mở PowerShell trên máy GPU và chạy theo các bước chuẩn:

### Bước 1: Clone Repository & Checkout Branch
```powershell
git clone https://github.com/vietmaiquoc2k5-max/DACN_NCKH.git
cd DACN_NCKH
git checkout fix-4
git status
```

### Bước 2: Thiết lập môi trường ảo & PyTorch CUDA
Chạy script tự động:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_cuda_env.ps1
```
*Hoặc thực hiện thủ công:*
```powershell
# 1. Tạo virtual environment
python -m venv .venv_gpu
.\.venv_gpu\Scripts\Activate.ps1

# 2. Cài đặt PyTorch với CUDA 12.1 (hoặc cu118 / cu124 tùy driver NVIDIA)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài đặt dependencies dự án
pip install -r requirements-training.txt
```

### Bước 3: Chạy bộ kiểm định 13 tiêu chí chất lượng
```powershell
python scripts/verify_week5_handoff.py
```
> **Yêu cầu bắt buộc:** Toàn bộ 13/13 tiêu chí phải đạt `[PASS]`, kết thúc với `ALL 13 HANDOFF CRITERIA PASSED! HANDOFF PACKAGE IS READY.` (Exit code: 0).

---

## 5. HƯỚNG DẪN THỰC THI HUẤN LUYỆN (EXECUTION WORKFLOW)

Tất cả các lệnh được thực hiện thông qua entry point: `scripts/run_week5_handoff.py`.

### 5.1. Chạy thử nghiệm GPU Smoke Test
Kiểm tra nạp mô hình vào VRAM, thực hiện 1 forward pass trên CUDA:
```powershell
python scripts/run_week5_handoff.py --smoke-test --device 0
```
*Thời gian chạy: ~2 giây. Kết quả: GPU smoke test PASSED!*

### 5.2. Huấn luyện Chiến lược A — Full Fine-tuning
Huấn luyện toàn bộ tham số mạng trên tập Việt Nam (30 epochs, batch 8, seed 42):
```powershell
python scripts/run_week5_handoff.py --strategy full_finetune --device 0
```
*Ước tính thời gian trên GPU RTX 3060: ~2 - 3 phút.*

### 5.3. Huấn luyện Chiến lược B — Freeze Backbone
Đóng băng toàn bộ backbone (10 stages), chỉ cập nhật detection head (30 epochs, batch 8, seed 42):
```powershell
python scripts/run_week5_handoff.py --strategy freeze_backbone --device 0
```
*Ước tính thời gian trên GPU RTX 3060: ~1 - 2 phút.*

### 5.4. Huấn luyện Chiến lược C — Freeze Backbone + 20% D-Fire Replay
Đóng băng backbone, bổ sung 147 ảnh D-Fire train vào mỗi epoch (tổng 732 ảnh/epoch, 30 epochs, batch 8, seed 42):
```powershell
python scripts/run_week5_handoff.py --strategy replay_20_percent --device 0
```
*Ước tính thời gian trên GPU RTX 3060: ~2 - 3 phút.*

### 5.5. Chạy tuần tự cả 3 chiến lược & tự động tổng hợp so sánh
```powershell
python scripts/run_week5_handoff.py --strategy all --device 0
```

### 5.6. Chạy bộ so sánh độc lập (Comparison Collector)
Sau khi có đủ 3 receipts của A, B, C:
```powershell
python scripts/run_week5_handoff.py --compare
```
Lệnh này sẽ gọi `scripts/compare_week5_runs.py` để sinh ra:
- `comparison.csv`
- `comparison.json`
- Tính toán $\Delta \text{VN mAP50}$, $\Delta \text{D-Fire mAP50}$ và mức quên `Catastrophic Forgetting`.

---

## 6. LƯU GIỮ KẾT QUẢ ĐÃ CHẠY TỪ MÁY SENDER (PRESERVED RUNS)

Trên máy Sender, **Chiến lược B (Freeze Backbone)** đã hoàn thành trọn vẹn 30 epochs trên CPU:
- **Thư mục lưu trữ:** `runs/benchmarks/week5_domain_adaptation/execution_20261116/run_freeze_backbone_seed42/`
- **Thời gian chạy:** 0.716 hours (~43 phút).
- **Checkpoints:**
  + `best.pt`: 6.223.914 bytes, SHA256: `280511cd51b778121f9f1ec89d8411e0d2daa556d99c0845423c5873b934ef5f`
  + `last.pt`: 6.223.914 bytes, SHA256: `8a6f7f5bf0096626775fff6d4867a31ec0f952d0ae64893ef38d26015caf5236`
- **Bảo toàn trạng thái Backbone:** `Frozen backbone state match: True` (100% weights & buffers được giữ nguyên bất biến).
- **Kết quả đo thực tế:**
  + Baseline trên VN val: mAP50 = 0.477, mAP50-95 = 0.341
  + Strategy B trên VN val: mAP50 = 0.4728, mAP50-95 = 0.3326
  + Baseline trên D-Fire val: mAP50 = 0.8387, mAP50-95 = 0.5227
  + Strategy B trên D-Fire val: mAP50 = 0.8129, mAP50-95 = 0.5013
- Người nhận có thể tái sử dụng trực tiếp checkpoint này nếu không muốn chạy lại Chiến lược B trên GPU.

---

## 7. XỬ LÝ SỰ CỐ & TỐI ƯU HÓA GPU (TROUBLESHOOTING GUIDE)

### Lỗi CUDA Out Of Memory (OOM)
Nếu GPU của người nhận có VRAM nhỏ (< 6GB) và gặp lỗi `torch.cuda.OutOfMemoryError`:
1. Giảm kích thước batch: thêm cờ `--batch 4` khi chạy `run_week5_handoff.py`.
2. Kiểm tra tiến trình chiếm dụng VRAM qua `nvidia-smi` và tắt các ứng dụng đồ họa nặng.
3. Không tăng `imgsz` vượt quá 640.

### Lỗi DataLoader Worker Hanging trên Windows
Mặc định `workers=0` đã được cấu hình trong protocol để ngăn chặn lỗi multiprocessing deadlock thường gặp trên hệ điều hành Windows. Giữ nguyên cấu hình này.

---

## 8. KẾT LUẬN & TRẠNG THÁI NGHIỆM THU BÀN GIAO

- **Trạng thái Handoff:** **READY**
- Toàn bộ mã nguồn, cấu hình, nhãn duyệt tay, biên bản trích xuất lineage, entry point điều phối và tài liệu hướng dẫn đã sẵn sàng 100%. Người nhận chỉ cần clone branch `fix-4`, thiết lập môi trường theo hướng dẫn và tiến hành huấn luyện mà không phụ thuộc vào bất kỳ tệp cục bộ chưa khai báo nào từ máy Sender.

