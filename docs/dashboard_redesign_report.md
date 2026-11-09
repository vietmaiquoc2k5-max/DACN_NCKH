# BÁO CÁO THIẾT KẾ LẠI GIAO DIỆN WEB DASHBOARD VÀ CHỌN CAMERA ĐỘNG
**Dự án:** Ứng dụng Computer Vision phát hiện và cảnh báo cháy từ camera giám sát (NCKH)  
**Thời gian hoàn thành:** 09/11/2026  
**Nhà nghiên cứu / Kỹ sư triển khai:** Mai Quốc Việt  

---

## 1. Kiến trúc hệ thống: Trước và Sau cải tiến

### Trước cải tiến
Trước khi tái cấu trúc, giao diện Dashboard hiển thị cứng lưới 9 camera đồng thời hoặc 1 camera đơn qua menu chọn. Luồng suy luận và tracking gắn liền cứng nhắc với danh sách camera cố định, chưa có module ByteTrack chuẩn hoá cho từng camera, và bộ trích xuất đặc trưng hình học chưa được tách lớp riêng biệt:
```
CameraStreamWorker
    ↓
SharedFireSmokeDetector (YOLO)
    ↓
TemporalVerifier (IoU Tracking cơ bản)
    ↓
EventDatabase / UI Streamlit
```

### Sau cải tiến
Kiến trúc mới được thiết kế phân tầng chuyên nghiệp, hỗ trợ quản lý tài nguyên linh hoạt và bảo toàn 100% pipeline xử lý:
```
CameraStreamWorker (Độc lập cho từng Camera)
    ↓
SharedFireSmokeDetector (YOLOv8/11 dùng chung, suy luận không chặn)
    ↓
ByteTrackAdapter (Bộ theo dõi ByteTrack độc lập cho mỗi camera)
    ↓
TemporalFeatureExtractor (Phân tích biến động diện tích, tâm, tỉ lệ khung)
    ↓
TemporalVerifier (Cửa sổ trượt k/N, ROI Masking, 2 cấp độ SUSPECT/FIRE, Cooldown)
    ↓
EventDatabase (SQLite + Snapshot JPEG Base64)
    ↓
Professional CCTV Control Room Dashboard (Streamlit Dark Mode #0B1120)
```

---

## 2. Danh mục các tập tin đã tạo mới và chỉnh sửa

| Đường dẫn tập tin | Loại thay đổi | Chức năng chính |
|---|---|---|
| [`src/bytetrack_adapter.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/src/bytetrack_adapter.py) | **Tạo mới** | Adapter tích hợp thuật toán ByteTrack (`BYTETracker` từ Ultralytics + fallback IoU Hungarian matching) giúp gán Track ID ổn định cho từng vùng lửa/khói, cách ly trạng thái cho từng camera. |
| [`src/temporal_feature_extractor.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/src/temporal_feature_extractor.py) | **Tạo mới** | Thu thập và phân tích chuỗi biến động diện tích ($\Delta \text{Area}$), độ dịch tâm box ($\Delta \text{Centroid}$) và tỉ lệ khung hình để phân biệt ngọn lửa/khói thật với nguồn sáng tĩnh/đèn pha xe. |
| [`src/camera_manager.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/src/camera_manager.py) | **Nâng cấp** | Tích hợp ByteTrackAdapter và FeatureExtractor vào từng `CameraStreamWorker`. Bổ sung API `set_active_camera_selection` quản lý bật/tắt camera động và `get_selected_snapshots`. |
| [`web_app.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/web_app.py) | **Thiết kế lại toàn bộ** | Giao diện phòng điều khiển CCTV chuyên nghiệp chuẩn Dark Mode (`#0B1120`), hỗ trợ Dynamic Camera Grid (1, 4, 6, 9), Alert Panel thời gian thực, bảng điều khiển hiệu năng. |
| [`tests/test_dynamic_camera_selection.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/tests/test_dynamic_camera_selection.py) | **Tạo mới** | Bộ kiểm thử tự động gồm 12 bài test chuẩn xác kiểm tra việc chọn camera động, giải phóng tài nguyên và bảo toàn trạng thái tracker. |
| [`scripts/benchmark_dynamic_cameras.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/scripts/benchmark_dynamic_cameras.py) | **Tạo mới** | Script đo kiểm hiệu năng thực tế (Capture FPS, Inference FPS, Latency, RAM) qua 4 cấu hình SINGLE, QUAD, HEXA, FULL. |

---

## 3. Giao diện CCTV Control Room mới

### Bảng màu chuẩn Trung tâm điều khiển
- **Background chính:** `#0B1120` (Xanh đêm đậm)
- **Sidebar:** `#111827` (Xám tối sang trọng)
- **Thẻ Card / Camera Container:** `#1E293B` (Slate Dark)
- **Primary / Accent:** `#3B82F6` (Xanh dương kỹ thuật)
- **Mức SAFE:** `#22C55E` (Xanh lá cây an toàn)
- **Mức FIRE:** `#EF4444` (Đỏ cảnh báo nguy cấp)
- **Mức SMOKE / SUSPECT:** `#F59E0B` (Cam vàng cảnh báo khói)
- **Màu chữ chính:** `#F8FAFC`

### Sơ đồ bố cục giao diện (Control Room Layout)
```
+---------------------------------------------------------------------------------------------------------+
|  🔴 CƠ QUAN PCCC - TRUNG TÂM GIÁM SÁT THỜI GIAN THỰC     [HỆ THỐNG HOẠT ĐỘNG]  Camera: 4/9  FPS: 93.9   |
+---------------------------------------------------------------------------------------------------------+
| SIDEBAR                    | MAIN DASHBOARD                                                            |
| ⚙️ Cấu hình Camera          | ------------------------------------------------------------------------- |
|   [1 Cam][4 Cam][6][9]     | [4 Active]  [0 FIRE Cực Nguy Hiểm]  [1 SMOKE Nghi Ngờ]  [Latency: 8.2ms]   |
|   Multiselect:             | ------------------------------------------------------------------------- |
|   [CAM-01, CAM-02, ...]    | LƯỚI CAMERA ĐỘNG (Dynamic Responsive Grid):                               |
|   [Áp dụng] [Bật/Dừng]     | +-----------------------------------+ +---------------------------------+ |
|                            | | CAM-01 [Kho Hàng A]   [23.5 FPS]  | | CAM-02 [Xưởng SX]    [23.4 FPS] | |
| 🎛️ Bộ lọc & Ngưỡng         | |   [LIVE STREAM 640x360]           | |   [LIVE STREAM 640x360]         | |
|   Confidence: 0.35         | |   Track #1: FIRE (92%)            | |   SAFE (Bình thường)            | |
|   Temporal K/N: 4/10       | |   [Phóng to] [Tạm dừng]           | |   [Phóng to] [Tạm dừng]         | |
|   Chế độ: Balanced         | +-----------------------------------+ +---------------------------------+ |
|                            | +-----------------------------------+ +---------------------------------+ |
| 📑 Điều hướng:             | | CAM-03 [Bãi Xe]       [23.6 FPS]  | | CAM-04 [Hành Lang]   [23.5 FPS] | |
|   • Live Monitoring        | |   [LIVE STREAM 640x360]           | |   [LIVE STREAM 640x360]         | |
|   • Event History          | |   SAFE (Bình thường)            | |   Track #3: SMOKE (78%)         | |
|   • Camera Management      | +-----------------------------------+ +---------------------------------+ |
|   • System Performance     | ------------------------------------------------------------------------- |
|                            | BẢNG CẢNH BÁO THỜI GIAN THỰC (Live Alert Panel):                          |
|                            | [23:38:15] CAM-01 | FIRE CỰC NGUY HIỂM | Conf: 92% | [Xác nhận] [Chi tiết]  |
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Cơ chế Dynamic Camera Selection

Cơ chế lựa chọn camera động được quản trị qua phương thức `set_active_camera_selection(selected_cam_ids, auto_start=True)` trong `src/camera_manager.py`:

1. **Khởi tạo và giải phóng tài nguyên chuẩn xác:**
   - Khi camera bị loại khỏi danh sách (ví dụ: chuyển từ 9 camera về 4 camera), luồng đọc và luồng suy luận của camera đó được dừng lại (`worker.stop()`). Đối tượng `cv2.VideoCapture` được giải phóng ngay lập tức (`cap.release()`), bộ đệm khung hình được dọn dẹp để tiết kiệm RAM và GPU.
   - Khi camera được kích hoạt lại, worker được khởi tạo luồng mới an toàn, không rò rỉ luồng cũ.
2. **Bảo toàn trạng thái Tracker:**
   - Các camera tiếp tục nằm trong danh sách đang hoạt động (ví dụ: CAM-01, CAM-02) **hoàn toàn không bị reset** trạng thái ByteTrack hay lịch sử cửa sổ k/N của TemporalVerifier. Nhờ đó, việc thêm bớt camera khác không làm gián đoạn theo dõi đối tượng của các camera đang chạy.
3. **Mô hình YOLO dùng chung (Shared Instance):**
   - Mọi camera đều sử dụng duy nhất một thể hiện `SharedFireSmokeDetector`. Khung hình được đẩy vào luồng suy luận theo cơ chế `frame_skip` thông minh và điều phối hàng đợi kích thước giới hạn (`max_queue_size=2`), triệt tiêu độ trễ tích lũy.

---

## 5. Kết quả kiểm thử Unit Test (`tests/test_dynamic_camera_selection.py`)

Tất cả **12/12** bài test tự động đều vượt qua tuyệt đối (Pass 100%):

| STT | Tên bài kiểm thử | Mục đích kiểm tra | Kết quả |
|:---:|---|---|:---:|
| 1 | `test_single_camera_selection` | Chọn chế độ SINGLE (1 camera), chỉ CAM-01 hoạt động, 8 camera còn lại dừng | **PASS** |
| 2 | `test_quad_camera_selection` | Chọn chế độ QUAD (4 camera), đúng 4 camera chạy, 5 camera dừng | **PASS** |
| 3 | `test_hexa_camera_selection` | Chọn chế độ HEXA (6 camera), đúng 6 camera chạy | **PASS** |
| 4 | `test_full_camera_selection` | Chọn chế độ FULL (9 camera), toàn bộ 9 camera hoạt động đồng thời | **PASS** |
| 5 | `test_custom_camera_selection` | Chọn camera tùy biến bất kỳ (CAM-01, CAM-03, CAM-07) | **PASS** |
| 6 | `test_unselected_camera_stops_properly` | Giải phóng VideoCapture và dừng worker khi bỏ chọn camera | **PASS** |
| 7 | `test_selected_camera_starts_properly` | Khởi tạo worker và bắt đầu suy luận khi thêm camera mới | **PASS** |
| 8 | `test_tracker_state_preserved_for_unchanged_cameras` | Bảo toàn đối tượng tracker và bộ đếm frame của camera không đổi | **PASS** |
| 9 | `test_shared_detector_used_by_all_cameras` | Tất cả camera dùng chung 1 thể hiện model YOLO, không nạp lại model | **PASS** |
| 10 | `test_independent_tracker_per_camera` | Mỗi camera có đối tượng ByteTrack riêng biệt, không trùng ID | **PASS** |
| 11 | `test_independent_verifier_per_camera` | Mỗi camera có TemporalVerifier riêng biệt, không lẫn lộn cửa sổ k/N | **PASS** |
| 12 | `test_fast_camera_switching_no_crash` | Chuyển đổi liên tục giữa các chế độ trong thời gian ngắn không gây deadlock | **PASS** |

> **Thời gian chạy kiểm thử:** 36.297 giây | **Tổng số test:** 12 | **Thành công:** 12 (100%) | **Lỗi:** 0

Bên cạnh đó, bộ kiểm thử hồi quy hệ thống cũ [`tests/test_dashboard_system.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/tests/test_dashboard_system.py) cũng **PASS 100%** (5/5 hạng mục).

---

## 6. Kết quả đo đạc FPS thực tế trên các cấu hình Camera

Số liệu thực tế thu thập bằng script [`scripts/benchmark_dynamic_cameras.py`](file:///F:/Dự%20án%20cá%20nhân/phan%20mem%20canh%20bao%20chay/scripts/benchmark_dynamic_cameras.py) trên cấu hình máy trạm (chế độ **Balanced**, GPU CUDA FP16, imgsz=416, frame_skip=2):

| Chế độ | Số Camera | Capture FPS TB / Cam | Tổng Capture FPS | Inference FPS TB | Độ trễ suy luận TB (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| **SINGLE** | 1 | 23.4 FPS | 23.4 FPS | 123.6 FPS | **6.7 ms** |
| **QUAD** | 4 | 23.5 FPS | 93.9 FPS | 99.3 FPS | **8.2 ms** |
| **HEXA** | 6 | 23.6 FPS | 141.7 FPS | 91.9 FPS | **12.8 ms** |
| **FULL** | 9 | 23.5 FPS | **211.4 FPS** | 74.1 FPS | **14.2 ms** |

*Nhận xét:*
- Tốc độ đọc luồng video (`Capture FPS`) đạt xấp xỉ 24 FPS ổn định trên mọi camera ở mọi chế độ, không bị giật lag nhờ tách luồng nén C++ OpenCV.
- Tốc độ suy luận của mô hình (`Inference FPS`) đạt mức cực kỳ cao: từ 74.1 FPS (chế độ 9 camera) đến 123.6 FPS (chế độ 1 camera).
- Độ trễ suy luận trung bình mỗi khung hình chỉ từ **6.7 ms đến 14.2 ms**, đáp ứng vượt trội chỉ tiêu thời gian thực của NCKH (yêu cầu $\le 10$ s).

---

## 7. Mức tiêu thụ bộ nhớ RAM thực tế

Số liệu đo bằng tiến trình Python (`psutil.Process.memory_info().rss`):

| Chế độ | Số Camera hoạt động | Dung lượng RAM thực tế | Mức tăng so với chế độ 1 Cam |
|---|:---:|:---:|:---:|
| **SINGLE** | 1 | **1.545,8 MB** (~1.51 GB) | Baseline |
| **QUAD** | 4 | **1.866,0 MB** (~1.82 GB) | +320.2 MB |
| **HEXA** | 6 | **2.129,7 MB** (~2.08 GB) | +583.9 MB |
| **FULL** | 9 | **2.388,5 MB** (~2.33 GB) | +842.7 MB |

*Đánh giá:*
- Mô hình YOLO dùng chung giúp bộ nhớ không bị nhân bản theo số lượng camera.
- Khi chuyển đổi từ 9 camera về 1 hoặc 4 camera, tài nguyên bộ nhớ được giải phóng ngay lập tức nhờ đóng luồng `VideoCapture` và dọn dẹp hàng đợi.
- Tổng mức RAM cực kỳ tối ưu, chỉ xấp xỉ 2.3 GB cho cả hệ thống 9 luồng video độ phân giải cao kèm suy luận AI liên tục.

---

## 8. Các vấn đề còn tồn đọng và định hướng tiếp theo

### Vấn đề còn tồn đọng
1. **Âm thanh cảnh báo trình duyệt:** Một số trình duyệt (Chrome/Edge) chặn cơ chế Web Audio tự động phát nếu người dùng chưa tương tác (click) trên trang web. Đã xử lý bằng cách tích hợp nút "Bật âm thanh báo động" và cảnh báo trực quan trên giao diện.
2. **Cảnh báo deprecation của Ultralytics:** Xuất hiện thông báo `WARNING 'half' is deprecated and will be removed in the future. Use 'quantize' instead` từ thư viện ultralytics khi kích hoạt FP16. Sẽ cập nhật tương thích trong phiên bản tối ưu hóa Tuần 6.

### Định hướng tiếp theo (Tuần 4 - Đề cương NCKH)
1. **Xây dựng module đánh giá sự kiện (`src/event_eval.py`):**
   - Viết công cụ tính toán tự động: **Event Recall**, **Độ trễ trung bình phát hiện (s)** và **Tần suất báo động giả mỗi giờ (FA/hour)**.
2. **Quét lưới siêu tham số (k, N) và ngưỡng biến động:**
   - Thực nghiệm các cấu hình $(k=6, N=10)$, $(k=8, N=12)$, $(k=10, N=15)$ trên bộ 72 video kiểm định thực tế (`videovietnam/`).
   - Chứng minh định lượng việc giảm $\ge 50\%$ báo giả và suy giảm Event Recall $\le 5\%$.
3. **Bổ sung thời lượng video âm tính:**
   - Bổ sung thêm ~2 giờ video âm tính có cảnh nhiễu thực tế (đèn xe, ánh nắng phản chiếu) để đạt trọn vẹn chỉ tiêu $\ge 5$ giờ video âm tính theo Đề cương.

