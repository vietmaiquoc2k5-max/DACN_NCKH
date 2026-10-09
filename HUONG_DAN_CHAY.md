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

## ⚙️ 5. Chế Độ Hoạt Động & Hướng Dẫn Sử Dụng

### 🧪 Chế độ Mô Phỏng (Thử nghiệm)
- Khi chưa có file `best.pt` ở thư mục gốc, phần mềm tự động tải mô hình **`yolov8n.pt`** và hoạt động ở **Chế độ Mô phỏng**.
- Trong chế độ này, hệ thống sẽ nhận diện vật thể thử nghiệm (như camera, điện thoại, người...) để bạn thử nghiệm Pop-up cảnh báo nội bộ và còi hú báo động.

### 🎯 Chế độ Thực Tế (Phát hiện Lửa & Khói)
- Sau khi bạn huấn luyện mô hình và copy file **`best.pt`** vào thư mục gốc dự án, hệ thống sẽ tự động chuyển sang chế độ thực tế và chỉ cảnh báo khi thấy **Lửa** hoặc **Khói**.

---

## 🏋️ 6. (Tùy Chọn) Huấn Luyện (Train) Mô Hình Dữ Liệu Riêng

Nếu muốn huấn luyện lại mô hình YOLOv8 với tập dữ liệu tùy chỉnh:

1. **Chuẩn bị Dataset:** Tải bộ dữ liệu định dạng **YOLOv8** từ Roboflow, giải nén và đổi tên thư mục thành **`dataset`** nằm cùng cấp với file `train.py`.
2. **Chạy huấn luyện:**
   ```bash
   .venv\Scripts\python.exe train.py
   ```
3. **Sử dụng trọng số mới:** Copy file **`best.pt`** từ `runs/detect/train_fire/weights/best.pt` ra thư mục gốc dự án.
