# Kết nối lại Telegram

Mở trang **Telegram** trong FireWatch, nhập Bot Token và Chat ID rồi bấm **Gửi tin kiểm tra và bật cảnh báo**. API gửi tin kiểm tra trước; chỉ sau khi Telegram xác nhận, FireWatch mới bật dispatcher. Cảnh báo sự kiện chỉ được xếp hàng cho camera đang bật tùy chọn Telegram.

Thông tin nhập từ Dashboard chỉ được giữ trong bộ nhớ backend và mất khi tiến trình khởi động lại. Để tự kết nối sau restart, đặt `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` trong service environment (hoặc `.env` được bảo vệ), đặt `ALERTS_ENABLED=false`, khởi động lại rồi kiểm tra/kết nối từ Dashboard. Không commit credential.

Lịch sử test riêng với lịch sử alert event. Trạng thái `DELIVERED` trong lịch sử là Telegram API đã xác nhận request; trạng thái `AMBIGUOUS` nghĩa là kết quả mạng chưa rõ.
