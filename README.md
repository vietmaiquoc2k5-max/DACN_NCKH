# 🔥 Hệ Thống Giám Sát & Cảnh Báo Cháy Sớm Đa Luồng Từ Camera (9-Camera AI Dashboard)

Ứng dụng **Computer Vision & Deep Learning (YOLOv8 / YOLO11)** phát hiện sớm đám cháy (**Lửa - `FIRE`** và **Khói - `SMOKE`**) từ hệ thống camera giám sát thời gian thực, tích hợp bộ xác minh theo chuỗi khung hình (**Temporal Verifier $k/N$**) nhằm giảm thiểu báo động giả và lưu trữ bằng chứng vào **SQLite**.

Repository GitHub: [https://github.com/vietmaiquoc2k5-max/DACN_NCKH](https://github.com/vietmaiquoc2k5-max/DACN_NCKH)

---

## 🌟 Tính Năng Nổi Bật

1. **Trung Tâm Giám Sát 9 Camera Thời Gian Thực (Bố cục Lưới $3 \times 3$):**
   - Hỗ trợ đồng thời tối đa **9 nguồn video độc lập**: Webcam (`0`, `1`), File Video cục bộ (`.mp4`, `.avi`) và luồng IP Camera `RTSP`.
   - Hiển thị trực tiếp trên từng ô camera: **Tên camera**, **Badge trạng thái kết nối** (`🟢 ONLINE`, `🟡 PAUSED`, `🔴 DISCONNECTED`, `⚪ STANDBY`), **FPS thực tế**, **Thời gian cập nhật (`HH:MM:SS`)** và **Tỷ lệ xác minh $k/N$**.
   - **Cô lập lỗi hoàn toàn:** Mỗi camera chạy trên một luồng (worker thread) riêng biệt; một camera mất kết nối hoặc lỗi tuyệt đối không làm treo hay gián đoạn các camera còn lại.
   - Điều khiển linh hoạt từng camera hoặc toàn hệ thống: **Bắt đầu (`▶`)**, **Tạm dừng (`⏸`)**, **Kết nối lại (`🔄`)** và **Dừng (`■`)**.

2. **Tích Hợp Mô Hình YOLO Chuyên Dụng (`best.pt`):**
   - Sử dụng mô hình đã huấn luyện của dự án (`best.pt` hoặc các mô hình benchmark trên tập D-Fire tại `runs/detect/runs/benchmarks/`).
   - Nhận diện chính xác 2 lớp chuẩn của dự án: **`0: SMOKE` (Khói - màu cam vàng)** và **`1: FIRE` (Lửa - màu đỏ tươi)** kèm độ tin cậy (`Confidence %`) và `Track ID`.
   - Kiểm tra tính hợp lệ của mô hình khi nạp: **Không tự ý dùng mô hình pretrained COCO (`yolov8n.pt`) để giả lập kết quả cháy thật**; báo lỗi rõ ràng nếu thiếu file trọng số.

3. **Xác Minh Chuỗi Thời Gian & Chống Báo Động Giả (`src/temporal_verifier.py`):**
   - Theo dõi đối tượng qua các khung hình liên tiếp bằng thuật toán ghép cặp IoU (`Track ID`).
   - Hỗ trợ **Vùng loại trừ (Exclusion Zones / ROI)** theo từng camera.
   - Phân cấp 2 mức cảnh báo trên cửa sổ trượt $N$ khung hình:
     - **Mức 1 (`SUSPECT` / Nghi ngờ):** Xuất hiện $\ge k_{\text{suspect}}/N$ khung hình.
     - **Mức 2 (`FIRE` / Báo cháy):** Xuất hiện $\ge k_{\text{confirm}}/N$ khung hình.
   - Cơ chế **Cooldown theo từng camera** giúp tránh phát cảnh báo và lưu ảnh trùng lặp liên tục cho cùng một sự kiện.

4. **Quản Lý Cảnh Báo, Lịch Sử Sự Kiện SQLite & Thống Kê Plotly:**
   - Lưu tự động sự kiện kèm ảnh chụp bằng chứng (`alerts/`) vào cơ sở dữ liệu SQLite (`alerts.db`).
   - Hỗ trợ **Xác nhận cháy thật** hoặc **Bỏ qua (Báo động giả)** ngay trên bảng điều khiển trực tiếp.
   - Lọc lịch sử theo **Khoảng ngày**, **Camera (`CAM-01`..`CAM-09`)**, **Loại đối tượng (`FIRE`/`SMOKE`)**, **Trạng thái xác nhận** và **Xuất báo cáo CSV**.
   - Biểu đồ thống kê trực quan (**Plotly**) tách biệt minh bạch giữa **Dữ liệu vận hành thực tế** và **Dữ liệu Demo/Kiểm thử**.

---

## 📂 Cấu Trúc Mã Nguồn

```text
├── web_app.py                     # Giao diện chính Web Dashboard 9 Camera (Streamlit Dark Mode)
├── app.py                         # Giao diện Desktop truyền thống (CustomTkinter 1 luồng)
├── detector.py                    # Bộ phát hiện 1 luồng (tương thích ngược cho app.py)
├── alert_manager.py               # Quản lý âm thanh còi hú, lưu ảnh bằng chứng & SQLite lõi
├── best.pt                        # Trọng số mô hình YOLO phát hiện Khói (0) & Lửa (1)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Cấu hình trung tâm (9 camera mặc định, đường dẫn model, tham số)
│   ├── detector.py                # SharedFireSmokeDetector: nạp model an toàn đa luồng & vẽ bbox
│   ├── temporal_verifier.py       # Bộ xác minh chuỗi khung hình k/N, Track ID, ROI & Cooldown
│   ├── camera_manager.py          # MultiCameraManager & CameraStreamWorker (quản lý 9 luồng độc lập)
│   ├── database.py                # EventDatabase: truy vấn, lọc đa chiều, thống kê & xuất CSV
│   └── alert_manager.py           # Cầu nối quản lý cảnh báo cho gói src/
├── tests/
│   └── test_dashboard_system.py   # Bộ kiểm thử tự động cho Detector, Verifier, Database & 9 Camera
├── test_integration.py            # Kiểm thử tích hợp tương thích ngược toàn dự án
├── scripts/                       # Các script chuẩn hóa D-Fire, huấn luyện & đánh giá benchmark MT2
├── baselines/                     # Mã nguồn Baseline B1 (HSV/YCbCr + MOG2) & B2 (MobileNetV3)
├── docs/                          # Báo cáo phân tích dữ liệu, kiểm định MD5 và kết quả thực nghiệm
└── requirements.txt               # Danh sách thư viện Python cần thiết
```

---

## 🛠️ Hướng Dẫn Cài Đặt

### 1. Yêu cầu hệ thống
- **Hệ điều hành:** Windows 10/11, Linux hoặc macOS.
- **Python:** 3.9 – 3.12 (khuyến nghị sử dụng môi trường ảo `.venv` có sẵn trong dự án).

### 2. Cài đặt thư viện
Mở Terminal / PowerShell tại thư mục gốc dự án:

```powershell
# Kích hoạt môi trường ảo (trên Windows)
.venv\Scripts\activate

# Cài đặt đầy đủ các gói thư viện
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Khởi Động Ứng Dụng

### Cách 1: Chạy Web Dashboard 9 Camera (Khuyên dùng)
```powershell
.venv\Scripts\python.exe -m streamlit run web_app.py
```
Hoặc nhấp đúp vào file **`chay_giao_dien_web.bat`**. Sau khi khởi động, trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`.

### Cách 2: Chạy Giao Diện Desktop (1 Camera)
```powershell
.venv\Scripts\python.exe app.py
```
Hoặc nhấp đúp vào file **`chay_giao_dien_desktop.bat`**.

---

## ⚙️ Hướng Dẫn Cấu Hình Nguồn Camera & Mô Hình

1. **Cấu hình 9 nguồn Camera:**
   - Trên giao diện Web Dashboard, chọn tab **`⚙️ Cấu Hình Camera & Model`** ở thanh menu bên trái.
   - Tại mỗi ô từ `CAM-01` đến `CAM-09`, bạn có thể:
     - Chọn nhanh các video thực tế có sẵn trong dự án (`videovietnam/VN_POS_01.mp4` .. `VN_POS_10.mp4`, `FireSmokeLabeler/videos/...`).
     - Nhập `0` hoặc `1` để kết nối Webcam USB.
     - Nhập đường dẫn RTSP (ví dụ: `rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101`).
   - Nhấn **`💾 Lưu`** để áp dụng ngay cho từng camera mà không làm gián đoạn các camera khác.

2. **Tối ưu hiệu năng khi chạy nhiều luồng:**
   - Ở thanh công cụ bên trái (**Sidebar**), bạn có thể điều chỉnh:
     - **Giới hạn FPS tối đa / Camera** (mặc định `12 FPS`).
     - **Chu kỳ suy luận AI (`frame_skip`)**: Đặt bằng `2` hoặc `3` nếu chạy trên CPU yếu để giảm tải tính toán.

---

## 🧪 Chạy Kiểm Thử Tự Động (Testing)

Để kiểm tra toàn bộ chức năng của Detector, Temporal Verifier, SQLite Database và cơ chế cô lập lỗi 9 Camera:

```powershell
# Kiểm thử hệ thống Dashboard 9 Camera
.venv\Scripts\python.exe tests/test_dashboard_system.py

# Kiểm thử tương thích ngược toàn dự án
.venv\Scripts\python.exe test_integration.py
```

