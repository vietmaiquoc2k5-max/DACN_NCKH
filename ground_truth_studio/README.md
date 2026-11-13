# Ground Truth Review Studio

Ứng dụng Streamlit chạy local, độc lập với `web_app.py` và inference. Chỉ phát nguồn gốc và ghi review do người dùng thực hiện. Không import YOLO, không tạo nhãn, không tự xác nhận các claim `cross_verified` lịch sử.

## Cài đặt và chạy

Python 3.10+; môi trường đã kiểm tra là Python 3.11.9 / Streamlit 1.66.1 / OpenCV 5.0.0 trên Windows. `sqlite3`, HTTP Range server, hashing và account provisioning dùng standard library. FFmpeg/ffprobe cần cho proxy AVI và source PTS; Studio tìm trong PATH rồi `.tools/ffmpeg/`.

```powershell
python -m pip install -r ground_truth_studio/requirements.txt
python -m streamlit run ground_truth_studio/app.py --server.address 127.0.0.1
```

Entry point cũng hỗ trợ lệnh `python -m streamlit run ground_truth_studio/app.py`. Để chạy cạnh dashboard camera đang dùng port mặc định, thêm `--server.port 8514`. Studio không sửa cấu hình Streamlit chung của dashboard.

Mặc định private directory là thư mục cùng cấp repo: `D:\workspace\week4_private_human_review`. Đổi bằng `GT_STUDIO_PRIVATE_DIR` hoặc cấu hình:

```powershell
$env:GT_STUDIO_CONFIG = 'D:\workspace\DACN_NCKH\ground_truth_studio\settings.example.json'
```

Các file `positive_annotator_02.csv`, `positive_adjudication.csv`, `negative_annotator_01.csv`, `negative_annotator_02.csv`, `negative_adjudication.csv` hiện có được đọc và snapshot vào SQLite. Không ghi đè chúng. Tệp protocol/status chuẩn bị vẫn nguyên vẹn, không được hiểu là signoff.

## Người quản lý cấp tài khoản trước khi review

Không có tài khoản/mật khẩu mặc định hoặc đăng ký tự do. Chạy từng lệnh bằng người quản lý trên máy local; thay placeholder bằng ID thực đã được kiểm tra. Mật khẩu được nhập kín bằng getpass, không đưa lên command line.

```powershell
python -m ground_truth_studio.admin create-account --reviewer-id REVIEWER01 --person-id REAL_PERSON_01 --role annotator_01 --identity-evidence HUMAN_IDENTITY_CHECK_REFERENCE
python -m ground_truth_studio.admin create-account --reviewer-id REVIEWER02 --person-id REAL_PERSON_02 --role annotator_02 --identity-evidence HUMAN_IDENTITY_CHECK_REFERENCE
python -m ground_truth_studio.admin create-account --reviewer-id ADJUDICATOR --person-id REAL_PERSON_03 --role adjudicator --identity-evidence HUMAN_IDENTITY_CHECK_REFERENCE
```

Mỗi người thật có một `person_id` ổn định, một tài khoản/vai trò. SQLite từ chối dùng cùng person ID cho hai tên đăng nhập. Không cấp alias để một người tự hoàn thành hai lượt độc lập. Adjudicator cần là người khác với hai annotator. Role `admin` chỉ xem overview/export, không tự trở thành annotator hay adjudicator.

Tài khoản dùng salted PBKDF2; opaque token hết hạn sau 12 giờ và được thu hồi khi logout. Server kiểm tra quyền, không dựa vào menu ẩn. Annotator không có API để xem người kia hoặc Ground Truth lịch sử, kể cả sau khi mình submit. Adjudicator chỉ xem được cả hai và lịch sử sau khi hai phiên đã submit.

Đây là kiểm soát trong ứng dụng, không phải chứng minh sinh trắc học hoặc sandbox của OS. Người quản lý phải xác nhận người thật, bố trí review độc lập và kiểm soát quyền đọc file bên ngoài trình duyệt. Không coi hai chuỗi ID hay một checkbox là bằng chứng người thật đã độc lập.

## Review độc lập

1. Đăng nhập đúng tài khoản, chọn Positive Review hoặc Negative Review.
2. Chọn video và nhấn **Mở / tiếp tục phiên review**. Positive ID ánh xạ qua filename trong catalog; không dùng mốc GT để điền annotation. Ví dụ POS_01 ánh xạ tới `posVideo1.868.avi`, không tự giả định `POS_01.avi` tồn tại.
3. Bật trình phát nếu muốn. AVI được transcode thành proxy MP4 riêng trong private cache; bản gốc không thay đổi. HTTP Range trả dữ liệu từng khối 64 KiB, không đọc cả video vào RAM.
4. Dùng Play/Pause, tốc độ, seek hoặc frame index / Previous / Next Frame. Trình xem frame decode source theo ordinal, không seek VFR bằng phép frame/FPS giả định. Việc lùi xa trong video dài có thể chậm vì phải decode lại tuần tự.
5. Đánh dấu Start và End, chọn FIRE/SMOKE rồi Add Event. Một video có thể có cả hai lớp và sự kiện mới ngoài GT cũ. Negative có thêm interval UNCERTAIN. Start/end chưa hoàn chỉnh không được submit.
6. Edit giữ event ID; Delete chỉ tác động draft của chính reviewer. **Save Draft** để lưu bền vững trước khi chuyển video/đóng trình duyệt. Các thay đổi chưa Save Draft ở bộ nhớ phiên có thể mất khi đóng tab; chưa có autosave theo timer.
7. Negative phải chọn explicit decision. `VERIFIED_NEGATIVE` ở đây là quyết định cá nhân, chưa phải video đủ điều kiện FAR. FIRE/SMOKE cần interval tương ứng; không được đánh negative nếu còn interval positive/uncertain.
8. Ghi evidence reference, xem đầy đủ nguồn và tự đánh dấu lời xác nhận trước **Submit Review**. Studio lưu thời điểm submit; không tự ký thay. Submit khóa phiên, không chuyển nó thành GT VERIFIED.

Timestamp có source PTS khi ffprobe đọc được các frame records hợp lệ. Thiếu PTS hoặc timeout sẽ ghi rõ `ESTIMATED_FRAME_INDEX_OVER_FPS`; không che giới hạn này. Thời lượng cuối dùng delta frame cuối, được ghi riêng trong `duration_provenance`; chưa phải phép đo duration độc lập để vượt qua mọi gate kỹ thuật.

Nếu CSV bên ngoài đã được điền bởi chính reviewer, nút nạp CSV kiểm tra reviewer ID/hash nguồn và nạp thành **draft**, giữ event ID nếu có. Các frame nhập từ CSV phải được xác nhận lại trên source trước khi submit. Signoff YES trong file không tự submit và không tự tái xác minh.

## Đối chiếu và phân xử

Annotator 01 cũng tạo review mới; không dùng 11 claim lịch sử hoặc 35 events cũ làm lượt độc lập mới. Chỉ adjudicator được xem đồng thời hai phiên đã khóa và lịch sử.

- Matching one-to-one cùng class, sai lệch onset và offset ≤0.5s; vượt ngưỡng, thiếu event hay khác class là DISPUTED.
- Không lấy trung bình timestamp. Chọn interval của một người hoặc nhập kết quả phân xử có source/frame provenance.
- Event lịch sử không tìm thấy phải có explicit disagreement keyed theo `GT11-index` và lý do.
- ACCEPT, DISPUTED, REQUIRE_REVIEW, REJECT đều cần lý do và evidence. Quyết định được append vào audit trail, không sửa nguồn frozen.
- REQUIRE_REVIEW tạo các draft mới khi hai reviewer mở lại phiên. Các submission cũ giữ nguyên; adjudication tiếp theo yêu cầu cả hai lượt mới đã submit.
- VERIFIED trong filter Studio chỉ có nghĩa human adjudicated **candidate**, không phải công nhận v1.1/v1.2 OFFICIAL.

## Lưu trữ, conflict và backup

Private directory chứa `ground_truth_studio.sqlite3`, WAL, `backups/`, `playback_cache/`, `exports/`. Không đưa chúng lên Git. WAL + transaction + optimistic revision chống hai process ghi đè nhau; stale editor phải reload. Backup dùng SQLite backup API và đóng connection ngay sau thao tác, kể cả Windows.

CSV root không được đồng bộ ghi ngược. Nếu người dùng sửa CSV bên ngoài, export chặn và giữ cả DB lẫn CSV; người quản lý đối chiếu, sau đó chủ động snapshot phiên bản mới:

```powershell
python -m ground_truth_studio.admin snapshot-csv --accept-external-changes
```

Lệnh trên chỉ ghi một snapshot mới, giữ snapshot cũ; không tự merge signoff hoặc tạo review. Khi export gặp row thuộc người khác, cần giải quyết identity conflict trước.

## Export

Adjudicator/admin nhấn **Xuất bundle mới**. Mỗi lần tạo UUID directory mới trong private `exports/`, kèm backup nhất quán. Xem [EXPORT_SCHEMA.md](EXPORT_SCHEMA.md).

- Giữ năm tên CSV của biểu mẫu hiện có; thêm `positive_annotator_01.csv` cho lượt độc lập mới.
- `ground_truth_review_summary.json`, `verified_gt_candidate.jsonl`, `ground_truth_provenance.json` và bản snapshot CSV gốc.
- Candidate luôn có `verification_status=pending_verification`, dù đã có human adjudication; chỉ người phụ trách được phê duyệt version transition sau đó. Không thay v1.1, không tự ghi v1.2, không đổi Operating Point.
- Bộ recovery hiện tại đọc được schema positive/negative CSV; event JSONL được test bằng loader của `src.event_eval`. Event mới/mismatch với GT lịch sử không bị ép vào một row cũ để làm PASS; legacy approval để trống khi không có mapping phù hợp.

Chỉ cộng negative exposure khi có hai negative submissions độc lập, ACCEPT có evidence, không positive/uncertain intervals, hash nguồn khớp và strict technical allowlist (`NORMAL_EOF`, full decode, no decode errors, inference alignment). Decode thành công hoặc giờ video vật lý không thay thế các gate này.

## Kiểm chứng

```powershell
$env:PATH = (Join-Path $PWD '.tools\ffmpeg') + ';' + $env:PATH
python -m unittest discover -s tests -p "test_gt_studio_*.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests dùng video và review fixture tổng hợp. UI test chạy login, điều hướng frame, Mark Start/End, Add Event và Save Draft; VFR integration dùng FFmpeg thực khi có. Không dùng test synthetic để chứng minh con người đã hoàn thành annotation. Các giới hạn, screenshot và kết quả chạy nằm trong `docs/GROUND_TRUTH_STUDIO_IMPLEMENTATION_REPORT.md`.
