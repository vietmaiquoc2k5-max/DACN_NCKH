# Quản lý vị trí camera FireWatch

## Cách dùng

Mở trang **Web Push**, chọn camera, rồi chọn một cách xác định vị trí:

- Bấm **Lấy vị trí hiện tại** để trình duyệt hỏi quyền. Đây là vị trí của thiết bị đang mở Dashboard, không phải vị trí tự động suy ra của camera RTSP. Chỉ lưu nếu đã đối chiếu rằng camera ở đúng nơi đó.
- Bấm vào bản đồ hoặc kéo marker để chọn vị trí camera.
- Nhập tọa độ hoặc địa chỉ thủ công. Có thể lưu địa chỉ khi chưa có tọa độ; khi đó FireWatch không tạo bản đồ tọa độ giả.

Vị trí được lưu bằng API quản trị đã xác thực. Việc chọn bản đồ hoặc lấy GPS không xác minh camera. Muốn đánh dấu đã xác minh, người vận hành phải chọn ô xác nhận sau khi kiểm tra thực địa. Tọa độ `(0, 0)` yêu cầu xác nhận riêng.

## Dữ liệu và bảo mật

`camera_location_records` lưu vị trí hiện tại, độ chính xác GPS, nguồn, trạng thái xác minh và thời điểm xác minh. Bảng vị trí cũ vẫn được giữ; hàng cũ được sao chép bổ sung khi khởi tạo. `event_location_records` lưu snapshot theo event ID và không cập nhật lại khi vị trí camera đổi. Migration chỉ tạo/bổ sung dữ liệu, không xóa bảng lịch sử.

Không có reverse geocoding tự động. Địa chỉ người dùng nhập được giữ nguyên; Google Maps dùng tọa độ đã lưu hoặc địa chỉ đã cấu hình. Bản đồ tải thư viện Leaflet từ unpkg và tile từ OpenStreetMap; các dịch vụ này nhận yêu cầu mạng để hiển thị bản đồ. Không gửi tọa độ đến dịch vụ geocoding.

## Kiểm tra trình duyệt thủ công

Khởi động bằng `python -m streamlit run web_app.py`, mở `http://localhost:8501` và vào **Web Push**:

1. Chọn camera không có vị trí và xác nhận bản đồ không đặt marker tại `(0, 0)`.
2. Bấm **Lấy vị trí hiện tại** và cấp quyền. Kiểm tra tọa độ, sai số GPS và marker.
3. Thử từ chối quyền hoặc tắt dịch vụ định vị; giao diện phải báo lỗi phù hợp và cho phép thử lại.
4. Chọn điểm khác trên bản đồ, rồi lưu. Xác nhận nguồn là chọn bản đồ và trạng thái còn chưa xác minh.
5. Nhập địa chỉ thủ công, lưu và dùng liên kết Google Maps.
6. Xác nhận vị trí thực tế bằng ô xác nhận riêng nếu phù hợp.
7. Chọn camera khác, quay lại camera đầu, rồi tải lại ứng dụng; vị trí phải được giữ.
8. Tạo một event đã xác nhận, đổi vị trí camera, rồi kiểm tra snapshot event cũ không đổi.
9. Lặp lại với camera RTSP ở xa thiết bị; không chấp nhận GPS của thiết bị như vị trí RTSP nếu hai nơi khác nhau.

Test tự động dùng dữ liệu giả cho tọa độ và kiểm tra mã component. Chúng không thay thế kiểm thử permission/GPS/map trong trình duyệt thật.

## Tích hợp cảnh báo Telegram kèm GPS & Google Maps

Khi sự kiện cháy/khói được xác nhận bởi `TemporalVerifier`, FireWatch tự động trích xuất snapshot vị trí của camera tại thời điểm xảy ra sự kiện (`event_location_records`) và gửi thông báo qua Telegram:

1. **Định dạng tin nhắn Telegram**:
   - Tiêu đề: `🚨 FIREWATCH — CẢNH BÁO ĐÃ XÁC NHẬN #<ID>`
   - Loại sự kiện: `🔥 CHÁY (FIRE)` hoặc `💨 KHÓI (SMOKE)`
   - Thông tin camera: `📷 Camera: <CAM_ID> / <Tên hiển thị>`
   - Vị trí camera:
     * Tên khu vực: `🏢 Khu vực: <Tên khu vực>` (nếu có)
     * Địa chỉ: `🏠 Địa chỉ: <Địa chỉ vật lý>` (nếu có)
     * Tọa độ GPS: `🌐 GPS: <Vĩ độ>, <Kinh độ> (<Đã xác minh / Chưa xác minh>)` (khi tọa độ hợp lệ)
     * Liên kết Google Maps: `🗺️ Google Maps:\nhttps://www.google.com/maps?q=<lat>,<lon>` (mở trực tiếp vị trí trên Google Maps)
     * Trường hợp camera chưa cấu hình vị trí: `📍 Vị trí camera: Chưa cấu hình` (không tạo GPS hay liên kết Maps giả mạo)
     * Trường hợp chỉ có địa chỉ: hiển thị địa chỉ mà không tạo GPS giả.
   - Snapshot đính kèm: Ảnh frame phát hiện có vẽ bounding box và độ tin cậy được gửi kèm qua API `sendPhoto`. Nếu ảnh lỗi/thiếu, hệ thống tự động fallback gửi văn bản qua `sendMessage` mà không làm gián đoạn cảnh báo.
   - An toàn độ dài caption: Tự động kiểm soát độ dài không vượt quá giới hạn 1.024 ký tự của Telegram `sendPhoto`, bảo toàn 100% ID sự kiện, loại sự kiện, camera và đường link Google Maps.

