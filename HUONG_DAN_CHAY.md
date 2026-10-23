# 🚨 HƯỚNG DẪN SỬ DỤNG VÀ VẬN HÀNH DỰ ÁN CẢNH BÁO CHÁY THÔNG MINH (CÓ PHÂN QUYỀN ĐĂNG NHẬP)

Hệ thống nhận diện và cảnh báo đám cháy (Lửa & Khói) thời gian thực ứng dụng công nghệ thị giác máy tính **AI YOLOv8 (Ultralytics)**. Phần mềm được tích hợp cơ chế **Phân quyền Đăng nhập (Role-Based Authentication)** rõ ràng giữa **Giao diện Người Dùng (User View)** và **Giao diện Admin Quản Lý (Admin View)** cùng **Hệ thống Cảnh báo Nội bộ (In-App Notification & SQLite Database Center)**.

---

## 🔑 1. Tài Khoản & Phân Quyền Đăng Nhập

Hệ thống chia làm 2 cấp quyền sử dụng:

| Vai Trò (Role) | Tài Khoản Đăng Nhập | Mật Khẩu (Password) | Quyền Hạn Hợp Lệ |
|---|---|---|---|
| **👤 NGƯỜI DÙNG (User / Bảo vệ)** | Trực tiếp mở (Không cần Pass) | *Không yêu cầu* | - Xem Livestream Camera trực tiếp.<br>- Nhận Pop-up Cảnh báo nổi tức thì.<br>- Tắt còi báo động khẩn cấp (`🔕 TẮT CÒI KHẨN CẤP`).<br>- Xem đồ thị nhận diện AI thời gian thực. |
| **🔑 ADMIN (Quản Trị Viên)** | **`admin`** | **`admin123`** | - Tất cả quyền của Người Dùng.<br>- Cấu hình nguồn Camera (Webcam, RTSP IP Camera, Video).<br>- Tinh chỉnh tham số AI (Conf threshold, Sensitivity, Cooldown).<br>- Truy cập CSDL Lịch sử `alerts.db` (Xem ảnh snapshot, Xóa bản ghi, Xuất báo cáo CSV). |

---

## 🌟 2. Các Tính Năng Đã Được Tích Hợp

1. **Hệ thống Cảnh báo Nội bộ (In-App Notification System)**:
   - Tự động nẩy **Hộp thoại Pop-up Cảnh báo Nổi** trên màn hình ứng dụng kèm thông tin vị trí, thời gian, độ tin cậy và ảnh chụp snapshot.
   - Phát âm thanh còi hú báo động qua loa máy tính (có nút **Tắt Còi Khẩn Cấp**).
   - Tự động lưu ảnh snapshot và nhật ký vào CSDL SQLite nội bộ (`alerts.db`).

2. **Bảo mật & Phân quyền Truy cập**:
   - Khi chọn tab **📋 Lịch sử Cảnh báo** hoặc **⚙️ Admin Quản lý**, hệ thống sẽ yêu cầu nhập Mật khẩu Admin (`admin123`).
   - Có nút **[🔒 Đăng Xuất Admin]** ở menu bên trái để khóa lại giao diện về quyền Người Dùng.

---

## 📋 3. Môi Trường & Yêu Cầu Cài Đặt

### Yêu cầu tiên quyết:
- **Python**: Phiên bản khuyên dùng **3.9 - 3.11**.
- **Môi trường ảo (.venv)** đã được khởi tạo trong dự án.

### Các bước cài đặt thư viện phụ thuộc:
Mở **Terminal / Command Prompt** tại thư mục dự án và chạy các lệnh sau:

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

*(Hoặc chạy trực tiếp bằng pip của môi trường ảo: `.venv\Scripts\pip.exe install -r requirements.txt`)*

---

## 🚀 4. Hướng Dẫn Vận Hành 2 Giao Diện (Desktop & Web)

Dự án cung cấp 2 hình thức giao diện chuyên nghiệp:

### 🔹 Cách 1: Giao diện Desktop (CustomTkinter GUI)
* **Khởi động nhanh:** Nhấp đúp vào file **`chay_giao_dien_desktop.bat`**.
* **Chạy bằng dòng lệnh:**
  ```bash
  .venv\Scripts\python.exe app.py
  ```

### 🔹 Cách 2: Giao diện Web Dashboard (Streamlit)
* **Khởi động nhanh:** Nhấp đúp vào file **`chay_giao_dien_web.bat`**.
* **Chạy bằng dòng lệnh:**
  ```bash
  .venv\Scripts\python.exe -m streamlit run web_app.py
  ```

---

---

## ⚙️ 5. Chế Độ Hoạt Động & Quản Lý Model AI

### 🎯 Mô Hình Hiện Hành:
- Hệ thống sử dụng mô hình YOLOv8 tùy biến: **`best.pt`** (kích thước ~6.2MB, hash: `52c041957a0a81069a20a139b1bd8409`).
- Mô hình hỗ trợ 2 lớp đối tượng chuẩn:
  - **Class 0: `smoke` (Khói)**
  - **Class 1: `fire` (Lửa)**
- Cả 2 giao diện (Desktop & Web) đều hỗ trợ xem chi tiết và **chuyển đổi model linh hoạt** ngay trong tab Admin.

### 🧪 Chế độ Dự Phòng (Fallback Simulation):
- Nếu không tìm thấy model chuyên dụng ở các đường dẫn (`best.pt`, `FireSmokeLabeler/model/best.pt`, `models/fire_smoke_v2.pt`), hệ thống tự động nạp **`yolov8n.pt`** và hoạt động ở chế độ mô phỏng để kiểm thử luồng còi hú & thông báo.

---

## 🏷️ 6. Bộ Công Cụ Tự Động Gắn Nhãn & Trực Quan Hóa (FireSmokeLabeler)

Dự án đã tích hợp trọn vẹn module **`FireSmokeLabeler`**:

### 1. Trực quan hóa kiểm tra nhãn dự đoán (Khuyên dùng trước khi train):
```bash
.venv\Scripts\python.exe FireSmokeLabeler/visualize_labels.py
```
- Kết quả ảnh đã vẽ hộp bounding box và nhãn sẽ được lưu tại: `FireSmokeLabeler/visualized/`.

### 2. Tự động gắn nhãn thêm video mới (nếu có):
- Chép video (.avi / .mp4) vào thư mục `FireSmokeLabeler/videos/`.
- Chạy lệnh:
```bash
.venv\Scripts\python.exe FireSmokeLabeler/create_fire_smoke_dataset.py
```

---

## 🏋️ 7. Huấn Luyện (Train) & Đánh Giá Model Mới An Toàn

### Bước 1: Kiểm tra tính toàn vẹn dataset (Không huấn luyện)
```bash
.venv\Scripts\python.exe train.py --validate-only
```
Lệnh này sẽ kiểm tra thư mục ảnh train/val, số lượng bounding box, phân bố class và xác nhận không có thư mục rỗng.

### Bước 2: Chạy huấn luyện (Lưu trọng số mới riêng biệt)
```bash
.venv\Scripts\python.exe train.py --epochs 50 --batch 16
```
- Trọng số mới tốt nhất sẽ tự động được lưu an toàn tại: **`models/fire_smoke_v2.pt`**.
- Model đang hoạt động `best.pt` được bảo vệ nguyên vẹn, **không bị ghi đè**.

### Bước 3: Đánh giá & So sánh mô hình trên tập Validation:
```bash
.venv\Scripts\python.exe evaluate_model.py --model models/fire_smoke_v2.pt --compare best.pt
```
Lệnh sẽ hiển thị bảng so sánh trực quan các chỉ số: **mAP@50**, **mAP@50-95**, **Precision**, **Recall** giữa mô hình mới và mô hình cũ.

### Bước 4: Chuyển sang mô hình mới (Khi đạt kết quả tốt hơn):
- Trong giao diện **Desktop** (`app.py`): Vào tab **Admin** -> Mục 4 -> Nhấn "📁 Chọn Model" chọn `models/fire_smoke_v2.pt` -> Nhấn "🔄 Nạp Model".
- Trong giao diện **Web** (`web_app.py`): Vào tab **Admin Quản Lý** -> Mục 5 -> Điền `models/fire_smoke_v2.pt` -> Nhấn "🔄 Nạp Lại Model".

