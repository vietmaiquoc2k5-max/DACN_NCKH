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
