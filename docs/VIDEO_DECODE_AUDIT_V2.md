# Independent decode EOF v2

`independent_decode_eof_v2` bổ sung kiểm chứng kỹ thuật bên cạnh v1; không sửa
`src/video_audit.py`, manifest, Studio catalog, video, checkpoint hay Ground Truth.
Không chạy inference và không mở khóa FAR hoặc Operating Point.

## Điều kiện chấp nhận

SHA trước/sau phải khớp catalog. FFmpeg giải mã toàn stream với `-xerror`, output
passthrough; exit0, stderr sạch, progress=end. FFprobe chạy full count_frames và
count_packets, sau đó một lượt full show_frames/show_packets. OpenCV grab tuần tự
được đếm độc lập; bốn số frame/packet phải khớp chính xác. Chưa có ngoại lệ backend.

Kiểm tra timeline bằng số nguyên trong stream time_base: PTS frame bằng
best_effort_timestamp, duration dương, end của frame bằng PTS frame tiếp theo,
không gap/overlap. Packet PTS/duration sau sắp xếp phải khớp frame display order;
packet DTS không đi lùi. B-frame PTS reorder trong packet order được phép. Packet
corrupt/discard, missing timestamp/duration, sai stream start hoặc stream duration
đều cần review. VFR dùng duration từng frame, không chia frame count cho FPS.

Exposure kỹ thuật mới là `(last_pts + last_explicit_duration - first_pts) * time_base`,
lưu cả rational chính xác và giây. Không cộng phần audio/container kéo dài ngoài
timeline video. Metadata nb_frames nếu tồn tại mà mâu thuẫn full decode vẫn bị loại.
CAP_PROP_FRAME_COUNT không phải tiêu chuẩn bắt buộc khi mọi bằng chứng trên xác
minh luồng; không có quy tắc chênh lệch ≤N frame hay sửa/bỏ frame để đạt chuẩn.

Trong v1, nhánh count mismatch trả `abs(reported-actual)` vào vị trí sau đó được
lưu là `full_decode_errors`. Vì vậy giá trị này ở các NEEDS_REVIEW không đồng nghĩa
đã quan sát decoder corruption. V2 giữ nguyên v1, ghi độc lập exit/stderr/count/
timestamp reasons; không lấy tên trường cũ làm kết luận nguồn hỏng.

Các lượt tool/OpenCV là các lần thực thi độc lập; FFmpeg và FFprobe vẫn dùng cùng
họ codec FFmpeg, OpenCV cũng có thể dùng backend FFmpeg. Không diễn giải chúng
thành ba thuật toán codec độc lập hoặc bằng chứng nhãn âm tính do con người xác nhận.
Clean EOF chứng minh tính đầy đủ của luồng được lưu, không chứng minh camera gốc
chưa từng bỏ frame trước lúc ghi. Không có nguồn upstream để xác thực giả thuyết đó.

## Chạy trên Windows

Python3.10+, OpenCV hiện có, FFmpeg/ffprobe trên PATH hoặc `.tools/ffmpeg/`.
CPU là đủ. Một worker/video, tool decode threads2, không cần GPU.

```powershell
python scripts/recover_negative_decode_v2.py --mode preflight
python scripts/recover_negative_decode_v2.py --mode audit --scope needs_review --policy independent_decode_eof_v2
python scripts/recover_negative_decode_v2.py --mode summarize
```

Mặc định output `runs/benchmarks/week4_decode_audit_v2/run_20261113/`. Preflight từ
chối thư mục đã tồn tại; dùng `--output-dir` khác cho run mới. Muốn tiếp tục run
đã preflight/audit, thêm `--resume` vào mode audit. Resume kiểm tra fingerprint code,
policy, SHA source, record checksum, raw artifacts và kết quả phân loại. Attempt
chưa hoàn tất được giữ trong `incomplete_attempts/` trước khi chạy lại video đó;
không di chuyển/xóa nguồn. Không chạy đồng thời hai coordinator vào cùng run.

Timeout tool mặc định1800s, điều chỉnh `--timeout`; coordinator giới hạn cả worker
bao gồm OpenCV bằng 4×timeout+120s. Tool timeout luôn là NEEDS_REVIEW, không giả làm
corruption. Worker crash/timeout dừng coordinator, giữ log/partial và không cấp
eligibility; dùng resume để thực hiện lại attempt. Không nuốt lỗi worker.

Scope predeclared gồm toàn bộ58 NEEDS_REVIEW v1. Thứ tự A/B/C/D theo v1 evidence,
trong nhóm theo thời lượng v1 giảm dần. Không đọc số alert/FAR/model output để chọn
video, không early-stop ở 5 giờ. Preflight lưu source scope, code và tool versions,
SHA của v1/raw benchmark/manifest/frozen/checkpoint. Summarize kiểm tra lại tất cả.

## Artifacts và cách đọc

Mỗi `records/<video_id>/` có raw counts/timeline/FFmpeg stdout+stderr, command,
`record.json`, `record.sha256`; worker OpenCV stderr nằm ở `<video_id>.worker.log`.
`summary.json`, `comparison.csv` có phân loại, counts, lý do, exposure và scope hash.
`v2_accepted` chỉ là các nguồn thực sự audit v2. Ba NORMAL_EOF v1 được lưu riêng trong
`v1_retained_normal_eof`, tái kiểm SHA, không báo chúng là đã chạy v2. Pool kỹ thuật
phục hồi là ba nguồn v1 này cộng các nguồn v2 chấp nhận; trường
`recovered_hours_v2_only` thể hiện riêng phần mới. Hai TRUNCATED v1 không được tự
đưa vào pool hay tự khôi phục. Video chưa chạy ở `not_audited`, không cấp eligibility.

Scope SHA là SHA256 của JSON canonical (sorted keys, separators comma/colon) danh
sách eligible ID/source SHA/duration/policy, sorted video ID. Không có GT/inference
timeline/validation lock: scientific Week4 PARTIAL, FAR và CI null, OP PROVISIONAL.

## Kiểm thử

```powershell
python -m unittest discover -s tests -p test_video_audit_v2.py -v
python -m pytest -q tests/test_video_audit_v2.py tests/test_week4_critical_acceptance.py
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests có fixture tổng hợp để kiểm tra gate, actual full decode, corrupt input,
timeout, resume, giữ bytes nguồn/v1. Fixture không được cộng vào giờ video thật.
Full suite có test legacy ghi báo cáo RTSP: bộ chạy bằng chứng sao lưu byte hiện tại
và phục hồi sau test; không phục hồi từ HEAD làm mất thay đổi của người dùng.
