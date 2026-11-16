# BBox resolution — xác minh xác nhận lại và sửa log lịch sử

Ngày16/11/2026. Branch fix-4; HEAD `85d8f6b3e8266af226e66a6cec0dc25c2744e0b2`; worktree ban đầu sạch. Không commit/push/xóa. Các original versions, resolution receipts, backup, SQLite và frozen GT không sửa.

## Kết quả từng ảnh

| Ảnh | Trạng thái | Annotation version mới | Decision | Thời gian reviewer (UTC) |
|---|---|---|---|---|
| VN_NEG_03_f000591.jpg | RESOLVED | `861fc9ee-dbb4-4437-a8b3-5af95733fcd6` | APPROVE | 2026-11-16T07:10:38.895215+00:00 |
| VN_NEG_03_f000609.jpg | RESOLVED | `4a57995f-321a-4b12-b956-740af4477946` | APPROVE | 2026-11-16T07:11:01.631776+00:00 |
| VN_NEG_03_f000618.jpg | RESOLVED | `cce0a296-3f86-430c-8f2a-efead069a03a` | APPROVE | 2026-11-16T07:11:15.805222+00:00 |
| VN_NEG_03_f000627.jpg | RESOLVED | `bf81eee7-62a1-45c8-80a3-c63669c7d248` | APPROVE | 2026-11-16T07:11:29.199463+00:00 |
| VN_NEG_03_f000635.jpg | RESOLVED | `71dcf778-9f78-4bf2-a362-96acd0178340` | APPROVE | 2026-11-16T07:11:40.615181+00:00 |

Cả5 receipt checksum hợp lệ; reviewer/time/image/evidence khớp annotation mới; metadataSHA/labelSHA và fingerprint review.json/review.sha256/labels.txt của từng affected version khớp. SHA ảnh nguồn đối chiếu catalog đạt. Không có receipt dở dang hoặc INVALID_RESOLUTION, không cần phê duyệt lại. Chi tiết đầy đủ reviewer/evidence/checksums/affected version IDs trong private local `.tools/bbox_resolution_logfix_20261116/audit_export.json`; không đưa metadata riêng tư lên Git.

Unresolved=0, resolved=5, invalid_resolution=0. Strict export integrity **PASS** trước và sau export. Chốt snapshot kiểm toán tại 2026-11-16T14:18:55.318785+07:00: valid versions=1129, unique valid images=1108, decisions `{'APPROVE': 1108}`. Số live có thể tăng nếu ngài tiếp tục review.

## Vì sao ERROR lặp dù đã giải quyết

bbox_snapshot.scan_reviews cũ ghi ERROR ngay trong vòng đọc raw version khi marker checksum không khớp, trước apply_resolutions. Mỗi Streamlit rerun/history/export scan do đó in lại5 lỗi lịch sử, trong khi kết quả cuối đã là RESOLVED_BY_HUMAN_RECONFIRMATION. Original checksum hỏng vẫn giữ đúng trạng thái forensic; resolution xác nhận quyết định mới, không sửa lại tính xác thực của checksum cũ.

Đã chuyển logging sang sau apply_resolutions. Resolved lịch sử ghi DEBUG với version/resolution ID và thông báo original checksum remains invalid. Unresolved và receipt invalid vẫn ERROR, vẫn làm strict export fail. Không tắt hoặc bỏ checksum/provenance validation.

## Export thực

Đã tạo version mới riêng: `D:\workspace\DACN_NCKH\.tools\bbox_resolution_logfix_20261116\approved_export`. Không ghi đè bundle cũ. Số human-approved annotation=1108; training eligible=0 vì adapter vẫn giữ các gate lineage/split riêng. Bundle YOLO labels/JSONL/provenance/checksums đã được kiểm lại; checksum manifest SHA256 `ef801114ea028a65fc775714c6dc94dd65b4150f3705d1c7097fdda13bf92c93`.

Các file private đã tồn tại trước export được hash trước/sau, không có thay đổi. Backup3265file vẫn khớp manifest và forensic bytes production giữ nguyên. Không tạo approval hay receipt thay reviewer. SQLite không truy cập/ghi trong lượt này; frozen eventGT SHA khớp HEAD.

## Files changed và kiểm thử

Chỉ sửa production `ground_truth_studio/bbox_snapshot.py` để đổi thời điểm/cấp log. Thêm2 regression tests trong `tests/test_bbox_resolution.py`: resolved không sinh ERROR; receipt lỗi vẫn ERROR và chặn export. Thêm báo cáo này.

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_bbox_resolution.py",
      "tests/test_bbox_integrity_recovery.py",
      "tests/test_week5_readiness.py",
      "tests/test_bbox_qa.py",
      "tests/test_gt_studio_ui.py"
    ],
    "exit_code": 0,
    "duration_seconds": 26.97699999999986,
    "log": ".tools/bbox_resolution_logfix_20261116/targeted.log",
    "tail": [
      "............................................                       [100%]",
      "44 passed, 6 subtests passed in 25.47s"
    ],
    "log_sha256": "d42a1e129e002ca112dbb17a92b1035075d1a4b6661b91c73e187931500b1c6e"
  },
  {
    "name": "full",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests"
    ],
    "exit_code": 0,
    "duration_seconds": 183.84758799999963,
    "log": ".tools/bbox_resolution_logfix_20261116/full.log",
    "tail": [
      "................................................         [100%]",
      "416 passed, 80 subtests passed in 181.83s (0:03:01)"
    ],
    "log_sha256": "40bdf37c24624474721483b0d7bc12bcfa0616ba54a58a29ec02ea1965ac7608"
  }
]

Studio production health8503=`ok`. Không giả lập login/click production; việc xác nhận thật của ngài đã được chứng minh bằng5 receipt. Regression tests dùng synthetic storage. Lượt audit phụ đầu đọc JSON bằng Windows default encoding làm evidence tiếng Việt không khớp giả; đã sửa công cụ audit đọc UTF-8 rồi kiểm lại đạt. Đó là lỗi decoding công cụ audit, không sửa bất cứ dữ liệu gốc nào và không yêu cầu người dùng làm lại.

## Ngài cần làm gì

Không cần phê duyệt lại5ảnh. Refresh BBoxQA để process đang chạy áp dụng code logging mới. Có thể tiếp tục review/xuất annotation; nếu vẫn có ERROR mới, kiểm version/resolution ID của thông báo vì validation vẫn hoạt động. Giữ forensic originals và backup. Không restore hoặc sửa marker cũ.
