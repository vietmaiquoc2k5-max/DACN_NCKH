# Nguồn video / khung hình Việt Nam (MT1, MT4)

> Mục tiêu: ≥ 1.000 khung hình Việt Nam gán nhãn tay, chia 60/20/20 **theo video**.
> Hạn chót đề xuất: 08/11/2026 (trước tuần 5 – thích nghi miền).

## 1. Video có cháy/khói (dương tính)

| # | Nguồn | Cách tìm | Ghi chú |
|---|---|---|---|
| 1 | Báo điện tử: VnExpress Video, Tuổi Trẻ, Thanh Niên, Dân trí, VietNamNet | Tìm trên site: "camera ghi lại cảnh cháy", "camera an ninh ghi lại vụ hỏa hoạn" | Nhiều clip từ camera an ninh thật, góc cao, độ phân giải thấp → đúng miền CCTV |
| 2 | YouTube kênh truyền hình: VTV24, ANTV, Truyền hình Công an nhân dân, đài tỉnh (HTV, Hà Nội…) | Từ khóa ở mục 4 | Ưu tiên đoạn quay từ camera cố định, bỏ đoạn quay điện thoại cầm tay |
| 3 | Diễn tập PCCC có thẩm quyền | Fanpage/YouTube Công an tỉnh, Phòng Cảnh sát PCCC&CNCH; hoặc xin phép quay trực tiếp buổi diễn tập tại trường/khu dân cư | Nguồn lửa thật hợp lệ duy nhất được phép quay |
| 4 | Mạng xã hội (Facebook, TikTok) – nhóm camera an ninh, nhóm "Camera giám sát" | Từ khóa ở mục 4 | Chất lượng không đều; chỉ lấy clip camera cố định, kiểm tra trùng lặp với nguồn 1–2 |

## 2. Video không cháy có cảnh nhiễu (âm tính)

| # | Nguồn | Nhiễu điển hình | Ghi chú |
|---|---|---|---|
| 1 | Cổng camera giao thông TP.HCM: giaothong.hochiminhcity.gov.vn | Đèn xe ban đêm, đèn đỏ, biển quảng cáo, mưa | Là ảnh chụp làm mới định kỳ, không phải video liên tục → dùng làm khung âm tính, không dùng cho đánh giá sự kiện |
| 2 | Ứng dụng iHanoi (mục Giao thông → Camera) | Như trên | Ghi màn hình nếu điều khoản cho phép |
| 3 | YouTube: livestream/walking tour phố Việt Nam (chợ đêm, phố ăn uống) | Quán nướng ven đường, khói bếp, đèn LED đỏ/cam, hoàng hôn | Nguồn tốt để đạt ≥ 5 giờ âm tính |
| 4 | Camera an ninh của nhà/phòng trọ/cửa hàng người quen (xin phép) | Ánh nắng, đèn, người mặc áo đỏ/cam | Đúng miền nhất (Hikvision/Dahua/Imou phổ biến ở VN) |

## 3. Bộ ảnh Việt Nam có sẵn (chỉ tham khảo)

- Roboflow Universe: có các dataset fire/smoke do nhóm/cá nhân Việt Nam đăng (ví dụ workspace của ĐHQG – `vietnam-national-university-vtwut`).
- **Lưu ý:** nhãn do người khác gán, phần lớn ảnh không phải cảnh Việt Nam và có thể trùng D-Fire → **không tính vào 1.000 khung gán nhãn tay**, chỉ dùng tham khảo hoặc làm dữ liệu phụ (phải ghi rõ trong báo cáo).

## 4. Từ khóa tìm kiếm

- "camera ghi lại khoảnh khắc cháy nhà", "camera an ninh ghi cảnh hỏa hoạn"
- "cháy kho xưởng camera", "cháy chung cư mini camera", "cháy quán karaoke camera"
- "khói bốc lên camera giám sát", "cháy xe máy camera"
- "diễn tập phòng cháy chữa cháy [tên tỉnh]"
- Âm tính: "phố đi bộ ban đêm livestream", "chợ đêm Việt Nam 4K walking tour", "quán nướng vỉa hè"

## 5. Tiêu chí chọn & quy trình

1. Ưu tiên camera **cố định**, góc cao, giống CCTV; loại clip quay tay rung lắc, đã dựng/ghép hiệu ứng.
2. Ghi vào `data/video_dataset_list.csv`: `video_id` dạng `VN_POS_xx` / `VN_NEG_xx`, URL gốc, ngày tải, `fire_start_sec`, `smoke_start_sec`.
3. Cắt khung bằng `scripts/tag_video_timestamps.py` (hoặc 1–2 khung/giây), tránh khung gần trùng nhau.
4. **Làm mờ khuôn mặt và biển số** trước khi chia sẻ/công bố.
5. Chia 60/20/20 **theo video**, không để khung của cùng video ở hai tập.
6. Gán nhãn tay trên CVAT/Roboflow; nếu dùng `best.pt` gợi ý khung thì phải duyệt/sửa từng khung và ghi rõ trong báo cáo.

## 6. Bản quyền & đạo đức

- Video báo chí/mạng xã hội có bản quyền: chỉ dùng cho mục đích nghiên cứu, **không phát hành lại video gốc**; khi công bố chỉ phát hành khung đã làm mờ + nhãn, kèm danh sách URL nguồn.
- Không tự đốt lửa để quay dữ liệu; lửa thật chỉ quay trong diễn tập PCCC có thẩm quyền.
- Camera tư nhân: phải được chủ sở hữu đồng ý.
