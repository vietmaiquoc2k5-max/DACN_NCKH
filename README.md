# 🔥 Hệ Thống Giám Sát & Cảnh Báo Cháy Sớm Đa Luồng Từ Camera (9-Camera AI Dashboard)

Ứng dụng **Computer Vision & Deep Learning (YOLOv8 / YOLO11)** phát hiện sớm đám cháy (**Lửa - `FIRE`** và **Khói - `SMOKE`**) từ hệ thống camera giám sát thời gian thực (**Camera IP/RTSP, Webcam USB và Video kiểm thử**), tích hợp bộ xác minh theo chuỗi khung hình (**Temporal Verifier $k/N$**) nhằm giảm thiểu báo động giả và lưu trữ bằng chứng vào **SQLite**.

Repository GitHub: [https://github.com/vietmaiquoc2k5-max/DACN_NCKH](https://github.com/vietmaiquoc2k5-max/DACN_NCKH)

---

## 🌟 Tính Năng Nổi Bật

1. **4 Chế Độ Nguồn Đầu Vào Linh Hoạt (`VIDEO_TEST`, `WEBCAM`, `IP_CAMERA`, `MULTI_CAMERA`):**
   - **`MULTI_CAMERA` (Giám sát đa luồng 9 Camera $3 \times 3$):** Cấu hình và giám sát đồng thời tối đa 9 camera độc lập (kết hợp cả RTSP, Webcam và Video kiểm thử).
   - **`VIDEO_TEST` (Kiểm thử bằng Video):** Chọn nhanh các video kiểm thử có sẵn trong dự án hoặc nhập đường dẫn file `.mp4`/`.avi` từ máy tính.
   - **`WEBCAM` (Camera máy tính / USB):** Kết nối trực tiếp với Webcam theo chỉ số thiết bị (`0`, `1`, `2`...) qua OpenCV DirectShow/V4L2.
   - **`IP_CAMERA` (Camera giám sát thực tế qua RTSP):** Kết nối luồng RTSP thực tế với cơ chế tự động kết nối lại (`RECONNECTING`), timeout chống treo và bảo mật ẩn mật khẩu.

2. **Trung Tâm Giám Sát 9 Camera Thời Gian Thực (Bố cục Lưới $3 \times 3$):**
   - Hiển thị trực tiếp trên từng ô camera: **Tên camera**, **Loại nguồn**, **Trạng thái kết nối chuẩn** (`🟢 CONNECTED`, `🔄 RECONNECTING`, `🟡 PAUSED`, `🔴 ERROR`, `⚪ DISCONNECTED`), **FPS đọc khung hình (`FPS đọc`)**, **FPS suy luận mô hình (`FPS YOLO`)**, **Thời gian cập nhật (`HH:MM:SS`)** và **Tỷ lệ xác minh $k/N$**.
   - **Cô lập lỗi hoàn toàn & Tự động kết nối lại:** Mỗi camera chạy trên một luồng (`CameraStreamWorker`) riêng biệt; một camera mất tín hiệu sẽ tự động chuyển sang `RECONNECTING` định kỳ mà tuyệt đối không làm treo các camera còn lại và không tạo cảnh báo cháy giả.
   - **Phân biệt rõ trạng thái mất tín hiệu và an toàn:** Khi có camera lỗi kết nối hoặc chưa kết nối, hệ thống hiển thị cảnh báo lỗi kết nối riêng biệt (`CONN_ERROR` / `STANDBY`), không báo trạng thái `SAFE` giả tạo.

3. **Tích Hợp Mô Hình YOLO Chuyên Dụng (`best.pt`):**
   - Tất cả nguồn đầu vào (`VIDEO_TEST`, `WEBCAM`, `IP_CAMERA`) đều đi qua cùng một pipeline thống nhất:
     `Camera/Video → OpenCV → YOLO → Bounding Box → Xác minh theo thời gian (k/N) → Cảnh báo → Lưu sự kiện SQLite`.
   - Nhận diện chính xác 2 lớp chuẩn của dự án: **`0: SMOKE` (Khói - màu cam vàng)** và **`1: FIRE` (Lửa - màu đỏ tươi)** kèm tên camera, `Track ID` và `Confidence %`.
   - Kiểm tra tính hợp lệ của mô hình khi nạp: **Không tự ý dùng mô hình pretrained COCO (`yolov8n.pt`) để giả lập kết quả cháy thật**.

4. **Xác Minh Chuỗi Thời Gian & Quản Lý Sự Kiện SQLite:**
   - Theo dõi đối tượng bằng thuật toán ghép cặp IoU (`Track ID`), lọc **Vùng loại trừ (Exclusion Zones / ROI)** và phân cấp 2 mức cảnh báo **`SUSPECT` (Nghi ngờ)** / **`FIRE` (Báo cháy)** trên cửa sổ trượt $k/N$.
   - Cơ chế **Cooldown theo từng camera** chống phát cảnh báo trùng lặp liên tục cho cùng một đám cháy.
   - Lưu sự kiện kèm ảnh chụp bằng chứng (`alerts/`) vào SQLite (`alerts.db`), hỗ trợ lọc đa chiều, xác nhận/bỏ qua sự kiện, thống kê Plotly và xuất CSV.

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
│   ├── config.py                  # Cấu hình 4 chế độ hoạt động, 9 camera & hàm ẩn mật khẩu RTSP
│   ├── detector.py                # SharedFireSmokeDetector: nạp model an toàn đa luồng & vẽ bbox
│   ├── temporal_verifier.py       # Bộ xác minh chuỗi khung hình k/N, Track ID, ROI & Cooldown
│   ├── camera_manager.py          # MultiCameraManager & CameraStreamWorker (OpenCV Video/Webcam/RTSP)
│   ├── database.py                # EventDatabase: truy vấn, lọc đa chiều, thống kê & xuất CSV
│   └── alert_manager.py           # Cầu nối quản lý cảnh báo cho gói src/
├── tests/
│   └── test_dashboard_system.py   # Bộ kiểm thử tự động 5 hạng mục (Detector, Verifier, DB, 9 Cam, RTSP)
├── test_integration.py            # Kiểm thử tích hợp tương thích ngược toàn dự án
├── scripts/                       # Các script chuẩn hóa D-Fire, huấn luyện & đánh giá benchmark MT2
├── baselines/                     # Mã nguồn Baseline B1 (HSV/YCbCr + MOG2) & B2 (MobileNetV3)
├── docs/                          # Báo cáo phân tích dữ liệu, kiểm định MD5 và kết quả thực nghiệm
└── requirements.txt               # Danh sách thư viện Python cần thiết
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Khởi Động

### 1. Cài đặt môi trường
```powershell
# Kích hoạt môi trường ảo (trên Windows)
.venv\Scripts\activate

# Cài đặt đầy đủ các gói thư viện
pip install -r requirements.txt
```

### 2. Khởi động Web Dashboard 9 Camera
```powershell
.venv\Scripts\python.exe -m streamlit run web_app.py
```
Hoặc nhấp đúp vào file **`chay_giao_dien_web.bat`**. Trình duyệt sẽ mở tại địa chỉ `http://localhost:8501`.

---

## 🎥 Hướng Dẫn Cấu Hình Nguồn Camera Thực Tế (Webcam & RTSP)

### 1. Chọn chế độ hoạt động (`Operating Mode`)
Ở thanh công cụ bên trái (**Sidebar**) hoặc thanh **Chế độ nhanh** phía trên màn hình giám sát, bạn có thể chọn 1 trong 4 chế độ:
- **`MULTI_CAMERA`:** Bật và giám sát đồng thời nhiều camera (tối đa 9 luồng) với nguồn đầu vào độc lập cho từng ô.
- **`VIDEO_TEST`:** Chuyển nhanh sang kiểm thử bằng các file video có sẵn trong dự án hoặc đường dẫn file trên máy.
- **`WEBCAM`:** Kết nối trực tiếp Webcam máy tính (mặc định chỉ số `0`) vào ô `CAM-01`.
- **`IP_CAMERA`:** Kết nối trực tiếp luồng Camera IP/RTSP vào ô `CAM-01`.

### 2. Cấu hình chi tiết từng Camera & Kiểm tra kết nối
Truy cập tab **`⚙️ Cấu Hình Nguồn Camera & RTSP`** trên thanh điều hướng:
1. Mở thẻ của camera cần cấu hình (`CAM-01` .. `CAM-09`).
2. **Đặt tên camera** (ví dụ: `CAM 01 - Kho Hàng A`) và chọn **Loại nguồn đầu vào**:
   - **`VIDEO_TEST`:** Chọn video từ danh sách hoặc nhập đường dẫn file `.mp4` / `.avi`.
   - **`WEBCAM`:** Nhập **Chỉ số thiết bị Webcam** (`0` cho webcam tích hợp, `1`, `2` cho webcam rời qua cổng USB).
   - **`IP_CAMERA`:** Nhập địa chỉ RTSP vào ô mật khẩu (mật khẩu sẽ tự động được che thành `rtsp://user:******@ip:port/...` sau khi lưu).
3. Sử dụng 4 nút điều khiển trực tiếp trên từng camera:
   - **`💾 Lưu cấu hình`:** Lưu thiết lập nguồn, trạng thái Bật/Tắt (`enabled`) và chế độ tự kết nối lại (`auto_reconnect`).
   - **`🩺 Kiểm tra kết nối`:** Thử mở luồng bằng OpenCV, đọc 1 khung hình thật, chạy thử qua YOLO và hiển thị ngay **Ảnh xem trước (Preview)** bên phải để xác nhận camera hoạt động trước khi đưa vào giám sát liên tục.
   - **`🔌 Kết nối`:** Khởi chạy luồng đọc và phát hiện cháy thời gian thực cho camera đó.
   - **`⏹ Ngắt kết nối`:** Dừng luồng và giải phóng tài nguyên `cv2.VideoCapture` ngay lập tức.

### 3. Định dạng URL RTSP phổ biến cho Camera IP
Lưu ý: Chỉ kết nối tới các camera mà bạn được cấp quyền quản trị hợp pháp. Không đưa mật khẩu thật lên GitHub.
- **Hikvision:** `rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101` (Luồng chính) hoặc `.../Channels/102` (Luồng phụ nhẹ hơn, khuyến nghị khi chạy 9 camera).
- **Dahua / Kbvision / Imou:** `rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0` (`subtype=1` cho luồng phụ).
- **Ezviz / Tapo / Camera ONVIF:** `rtsp://username:password@IP_ADDRESS:554/stream1` (hoặc `/h264Preview_01_main`).
- **Giả lập RTSP bằng MediaMTX / FFmpeg:** `rtsp://localhost:8554/live`

### 4. Xử lý lỗi kết nối thường gặp (Troubleshooting)
| Trạng thái / Hiện tượng | Nguyên nhân phổ biến | Cách khắc phục |
|---|---|---|
| **`🔴 ERROR` / Không thể mở nguồn** | Sai địa chỉ IP, sai cổng RTSP (mặc định `554`), sai tài khoản/mật khẩu hoặc Webcam đang bị ứng dụng khác (Zoom/Teams) chiếm dụng. | Nhấn **`🩺 Kiểm tra kết nối`** để xem thông báo lỗi chi tiết; kiểm tra ping tới IP camera hoặc tắt ứng dụng đang dùng Webcam. |
| **`🔄 RECONNECTING` liên tục** | Tín hiệu mạng Wi-Fi/LAN chập chờn hoặc băng thông RTSP quá cao gây timeout quá 5 giây. | Hệ thống đã tự động ép giao thức `rtsp_transport;tcp`. Khuyến nghị chuyển sang **luồng phụ (Sub-stream)** của camera hoặc giảm **Độ phân giải xử lý** xuống `480px`/`640px` ở Sidebar. |
| **`FPS đọc` cao nhưng `FPS YOLO` thấp** | Phần cứng CPU/GPU quá tải khi chạy đồng thời nhiều luồng độ phân giải cao. | Tăng **Chu kỳ suy luận (`frame_skip`)** lên `2` hoặc `3` và giảm **Giới hạn FPS đọc** về `10–12 FPS` trên Sidebar. |

> **⚠️ Khuyến cáo vận hành PCCC:** Ứng dụng sử dụng thị giác máy tính để hỗ trợ phát hiện và cảnh báo sớm nguy cơ cháy/khói từ hình ảnh camera. Hệ thống **không thay thế** các thiết bị báo cháy chuyên dụng theo tiêu chuẩn PCCC và không cam kết phát hiện 100% mọi đám cháy trong mọi điều kiện che khuất hoặc ánh sáng cực đoan. Cần kiểm thử thực tế tại hiện trường trước khi đưa vào vận hành.

---

## 🧪 Chạy Kiểm Thử Tự Động (Testing)

```powershell
# Kiểm thử toàn diện 5 hạng mục Dashboard 9 Camera & Chế độ Camera thực tế
.venv\Scripts\python.exe tests/test_dashboard_system.py

# Kiểm thử tương thích ngược toàn dự án
.venv\Scripts\python.exe test_integration.py
```
