# 🚨 Hệ Thống Cảnh Báo Cháy Thông Minh Sử Dụng Python & YOLOv8

Hệ thống nhận diện và cảnh báo đám cháy (Lửa và Khói) chạy trên Desktop, được phát triển bằng ngôn ngữ lập trình Python, mô hình thị giác máy tính YOLOv8 (Ultralytics) và giao diện CustomTkinter hiện đại.

---

## 🌟 Tính Năng Chính
- **Giám sát thời gian thực (Live View)**: Hỗ trợ Webcam, video file cục bộ, và IP Camera (RTSP).
- **Nhận diện đám cháy bằng AI**: Sử dụng mô hình YOLOv8 tự huấn luyện để phát hiện chính xác lửa và khói.
- **Cơ chế chống báo động giả**: Lọc nhiễu dựa trên tần suất xuất hiện liên tục của đám cháy trên khung hình.
- **Báo động âm thanh**: Tự động phát còi hú cảnh báo qua loa máy tính (có nút bật/tắt trên giao diện).
- **Cảnh báo qua Telegram**: Tự động gửi tin nhắn cảnh báo văn bản kèm ảnh chụp thực tế từ camera đến nhóm chat Telegram của bạn.
- **Bảng Log lịch sử**: Lưu nhật ký các sự kiện và thời gian phát hiện đám cháy thời gian thực.

---

## 🛠️ Hướng Dẫn Cài Đặt (Môi trường máy tính của bạn)

### Bước 1: Cài đặt Python
Tải và cài đặt Python (phiên bản khuyên dùng **3.9 - 3.11**) tại [python.org](https://www.python.org/downloads/).
*(Lưu ý: Tích chọn **"Add Python to PATH"** trong quá trình cài đặt).*

### Bước 2: Cài đặt các thư viện phụ thuộc
Mở terminal (Command Prompt trên Windows hoặc terminal trên VS Code) tại thư mục dự án, kích hoạt môi trường ảo (Virtual Environment) và chạy lệnh sau:

* **Kích hoạt môi trường ảo và cài đặt:**
  ```bash
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```
* **Hoặc chạy trực tiếp qua pip của môi trường ảo:**
  ```bash
  .venv\Scripts\pip.exe install -r requirements.txt
  ```

---

## 🤖 Hướng Dẫn Cấu Hình Telegram Bot (Nhận Cảnh Báo)

Để hệ thống gửi tin nhắn và ảnh chụp đám cháy về điện thoại của bạn, hãy làm theo các bước sau:

### 1. Tạo Bot Telegram mới
1. Mở ứng dụng Telegram, tìm kiếm tài khoản **`@BotFather`** (có tích xanh).
2. Gửi tin nhắn `/newbot`.
3. Nhập tên hiển thị cho Bot (ví dụ: `FireAlarmBot`).
4. Nhập username cho Bot (phải kết thúc bằng chữ `_bot` hoặc `Bot`, ví dụ: `my_fire_alarm_123_bot`).
5. `@BotFather` sẽ gửi lại cho bạn một chuỗi **HTTP API Token** (Ví dụ: `6987451230:AAElkJu...`). Hãy lưu token này lại.

### 2. Lấy Chat ID của bạn (Hoặc Group)
* **Nếu nhận tin nhắn cá nhân:**
  1. Click vào link bot của bạn (ví dụ: `t.me/my_fire_alarm_123_bot`) và nhấn **Start**.
  2. Tìm kiếm bot **`@userinfobot`** trên Telegram và nhấn **Start**. Bot này sẽ lập tức trả về một dãy số **ID** (Ví dụ: `987654321`). Đây chính là **Chat ID** của bạn.
* **Nếu nhận tin nhắn trong Group (Khuyên dùng để cả nhà cùng biết):**
  1. Tạo một Group Telegram mới và thêm Bot của bạn vào Group đó.
  2. Thêm bot **`@RawDataBot`** vào Group đó. Nó sẽ hiển thị dữ liệu thô chứa thông tin Group. Hãy tìm dòng `"chat": {"id": -100xxxxxxxxx, ...}`. Số âm bắt đầu bằng `-100` chính là **Chat ID** của Group. (Sau khi lấy xong, bạn có thể kích `@RawDataBot` ra khỏi group).

---

## 📂 Hướng Dẫn Tải Dataset & Huấn Luyện (Train) Mô Hình

Vì dự án yêu cầu tự huấn luyện dữ liệu riêng để nhận diện Lửa & Khói tối ưu, bạn làm theo các bước sau:

### Bước 1: Tải tập dữ liệu đã gán nhãn sẵn
1. Truy cập trang web gán nhãn cộng đồng **[Roboflow](https://roboflow.com/)** (Đăng ký tài khoản miễn phí).
2. Tìm kiếm tập dữ liệu với từ khóa: `fire smoke yolov8`.
3. Chọn một bộ dữ liệu tốt (ví dụ: bộ [Fire-Smoke-Detection-YOLOv8](https://universe.roboflow.com/sherry-u0mha/fire-detection-tby69) hoặc bộ tương đương).
4. Nhấp vào nút **Download Dataset** -> Chọn Format là **YOLOv8** -> Chọn **download zip to computer**.
5. Giải nén file tải về và đổi tên thư mục thành **`dataset`**. Sau đó, đưa thư mục `dataset` này vào cùng cấp với file `train.py` của dự án.
   *Cấu trúc thư mục chuẩn phải như sau:*
   ```text
   phan mem canh bao chay/
   ├── dataset/
   │   ├── images/
   │   │   ├── train/ (chứa ảnh .jpg)
   │   │   └── val/   (chứa ảnh .jpg)
   │   └── labels/
   │       ├── train/ (chứa file nhãn .txt)
   │       └── val/   (chứa file nhãn .txt)
   ```

### Bước 2: Chạy Huấn Luyện
Chạy file `train.py` để hệ thống tự động nhận diện thiết bị và tiến hành train 50 epochs:
```bash
.venv\Scripts\python.exe train.py
```
- Nếu máy bạn có card đồ họa NVIDIA (và đã cài CUDA), chương trình sẽ dùng GPU để train rất nhanh (khoảng 15-30 phút).
- Nếu không có GPU, chương trình sẽ tự chuyển sang CPU (quá trình train sẽ mất vài tiếng).
- Sau khi kết thúc, file trọng số tốt nhất sẽ nằm tại: `runs/detect/train_fire/weights/best.pt`.
- **Hãy copy file `best.pt` này đặt ra ngoài thư mục gốc dự án** (cùng cấp với `app.py`).

---

## 🚀 Hướng Dẫn Vận Hành Hệ Thống

Hệ thống hỗ trợ 2 giao diện: **Giao diện Desktop (CustomTkinter)** và **Giao diện Web (Streamlit)**.

Để vận hành nhanh nhất, bạn chỉ cần nhấp đúp chuột vào file batch tương ứng:
* **Giao diện Desktop (CustomTkinter):** Chạy file `chay_giao_dien_desktop.bat`
* **Giao diện Web (Streamlit):** Chạy file `chay_giao_dien_web.bat`

Hoặc chạy thủ công qua dòng lệnh:
* **Chạy giao diện Desktop:**
  ```bash
  .venv\Scripts\python.exe app.py
  ```
* **Chạy giao diện Web (Streamlit):**
  ```bash
  .venv\Scripts\python.exe -m streamlit run web_app.py
  ```

### Các bước vận hành trên giao diện:

1. **Chạy thử nghiệm (Không cần train trước)**:
   - Khi chưa có file `best.pt`, hệ thống sẽ tự động dùng mô hình mặc định `yolov8n.pt`.
   - Hệ thống sẽ chạy ở **Chế độ Giả lập (Simulation)**: Nhận diện bất kỳ vật thể nào (Ví dụ: Đưa điện thoại hoặc đưa khuôn mặt trước camera) để kích hoạt thử còi báo động và tính năng gửi Telegram. Điều này giúp bạn kiểm tra tính năng báo động hoạt động tốt.
2. **Chạy thực tế (Sau khi đã train xong)**:
   - Sau khi bạn đã copy file `best.pt` vào thư mục gốc, khởi động lại ứng dụng. Hệ thống sẽ tắt chế độ giả lập và chỉ báo động khi nhìn thấy **Lửa** hoặc **Khói**.
3. **Cấu hình trên giao diện**:
   - Chọn nguồn Camera đầu vào: Webcam mặc định, hoặc nhập link RTSP camera nhà bạn (Ví dụ: `rtsp://admin:password@192.168.1.100:554/stream`).
   - Nhập **Telegram Token** và **Chat ID** -> Nhấn **Gửi Thử Tin Nhắn Tele** để kiểm tra bot (Hệ thống sẽ kiểm thử kết nối đồng bộ và báo lỗi cụ thể nếu cấu hình sai).
   - Điều chỉnh thanh trượt **Ngưỡng Nhận Diện (Conf)** (Khuyên dùng 40%-50%) và **Ngưỡng Cảnh Báo** (số khung hình xuất hiện liên tục, khuyên dùng 8-12 frames để tránh nhiễu ánh sáng đỏ).
   - Nhấn **▶ BẮT ĐẦU** (hoặc **▶ BẮT ĐẦU GIÁM SÁT**) để khởi động luồng AI. Khi có cháy, còi sẽ hú và tin nhắn kèm ảnh thực tế sẽ gửi về Telegram của bạn!
