# Khoảng thiếu negative independent test

Ngày (Asia/Saigon): 2026-11-14T01:58:33.051196+07:00; HEAD `6942fa567683ed9fdc3ef455188db1a74cef2c7f`. **NEGATIVE_INDEPENDENT_TEST=BLOCKED.**

| Phạm vi | Video / exposure | Diễn giải |
|---|---|---|
| VN_NEG đã review | 63 submitted records theo export thật | Không phải63 nguồn âm tính đủ điều kiện |
| Nguồn NEG được chấp nhận gần nhất | 47 video /5.7496550970h | GT joint consensus và kỹ thuật hợp lệ trong scope lịch sử; đã có inference/evaluation |
| Validation NEG lịch sử | 24 video /0.5243872730h | Đã sweep, không dùng làm test mới |
| Exploratory holdout NEG lịch sử | 23 video /5.2252678241h | Prior exposure, không chứng minh independent test |
| Independent NEG test đã chứng minh | **0h** | Là thiếu bằng chứng đủ điều kiện, không phải FAR=0 hoặc không có negative media |
| Cần bổ sung | **>=5h test NEG mới độc lập** | Không lấy lại5.7497h cũ để PASS |

80 VN_POS là positive candidates chưa có GT, không được đổi thành negative để bù mẫu số FAR. Không dùng model để quyết định video mới nào là negative. Người thật cần xem toàn bộ nguồn, phân biệt FIRE/SMOKE/uncertain, Submit với source SHA, onset/interval nếu có; nguồn lỗi decode không tính full exposure.

Holdout NEG mới phải có source/camera/scene provenance, chống trùng/cắt/encode lại với train/val/tuning/benchmark cũ và cả VN_POS; audit full decode/source PTS, duration và hash trước/sau. Ghi dữ liệu đã dùng ở đâu, lock video/group list trước inference; OP phải khóa bằng validation hợp lệ trước independent test. Một file mới hoặc tên mới không chứng minh footage chưa được dùng. Nếu chưa chứng minh lịch sử train thì giữ TEST_INDEPENDENCE_NOT_VERIFIED.

Protocol mục tiêu >=5h là exposure test sau mọi exclusion/dedup, không phải tổng download/container duration hoặc ảnh tĩnh. Cần reserve validation/train NEG riêng nếu bổ sung, không lấy từ test sau khi xem kết quả. Khi test đủ điều kiện, tính FAR/giờ và exact Poisson CI95% trên cùng scope/pipeline; không đo mới hoặc tái chọn OP trong tác vụ này.

Đã giữ nguyên GT/model/video/benchmarks; không tạo negative signoff, không inference mới, không giả số FAR. [Audit positive](VN_POS_PREANNOTATION_AUDIT_REPORT.md), [Week5 readiness](WEEK5_VN_POS_DATA_READINESS.md).
