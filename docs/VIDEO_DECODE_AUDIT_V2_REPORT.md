# Báo cáo independent_decode_eof_v2 — 13/11/2026

BRANCH: `fix-4`

HEAD: `6ec06e407ed8a6043d003dbbdca6958080b5fce6`

WORKTREE: chỉ thêm module/CLI/tests/docs/artifacts v2; không commit/push. Baseline sạch.
Không sửa video, audit v1, manifest, frozen GT, checkpoint hoặc raw inference.

AUDIT V1:

- 63 video: 3 NORMAL_EOF, 58 NEEDS_REVIEW, 2 TRUNCATED_STREAM.
- Technical hours: 2.6180916667.

AUDIT V2:

- Audited: 58/58; scope predeclared toàn bộ58, không early stop.
- NORMAL_EOF: 58; NEEDS_REVIEW: 0;
  TRUNCATED_STREAM: 0; DECODE_ERROR: 0.
- Thứ tự groupA/B/C/D, longest v1 duration first; tất cả58 thuộc groupA theo v1.
  Không đọc kết quả detector để chọn nguồn.

TECHNICAL DURATION:

- v1 giữ lại: 2.6180916667 giờ (3 nguồn strict-v1, tái kiểm SHA, không gọi là đã audit v2).
- v2 mới phục hồi: 4.2042507361 giờ.
- Pool kỹ thuật sau phục hồi: 6.8223424028 giờ.
- Target ≥5h: PASS. Đây là candidate theo kỹ thuật, chưa phải FAR exposure.
- Duration v2 dùng PTS tick + duration frame cuối, đối chiếu stream start/duration
  và tổng explicit frame durations. Không lấy container/audio duration hoặc frame/FPS.
- Hai nguồn v1 TRUNCATED VN_NEG_10/48 vẫn bị loại, không audit lại trong scope58.

ROOT CAUSES:

- Metadata reported-count mismatch nhưng decode/timeline được kiểm chứng: 58.
- Số reported count khớp phép làm tròn container duration×OpenCV FPS: 58.
  Đây là quan sát số học, không phải chứng minh mã bên trong backend hoặc acceptance gate.
- Decoder corruption mới trong58: 0.
- Unresolved NEEDS_REVIEW: 0; lý do từng file trong root_cause_analysis/summary.
- Ở v1, CAP_PROP_FRAME_COUNT buộc bằng actual count; các mẫu06/07 có reported
  3143/1571 nhưng independent actual 3138/1569. V2 xác nhận count, từng packet/frame,
  integer PTS/duration và EOF; không dùng chênh lệch ≤N để ép PASS. Không khái quát
  kết quả hai ví dụ cho các video khác: cả58 đã được đo lại riêng.
- Ở06/07, FFprobe xác nhận audio AAC dài hơn video và quyết định container duration;
  raw metadata nằm ở root_cause_examples/. Trường full_decode_errors của v1 có thể
  chứa frame-count delta (5/2), không phải bằng chứng đã quan sát decoder corruption.

TESTS:

| Command (python -m) | Kết quả | Exit | Log |
|---|---|---:|---|
| unittest discover -s tests -p test_video_audit_v2.py -v | 23/23 PASS | 0 | unittest_v2.log |
| pytest -q tests/test_video_audit_v2.py tests/test_week4_critical_acceptance.py | 33 PASS, 16 subtests PASS | 0 | pytest_v2.log |
| unittest discover -s tests -p test_*.py -v | 272/272 PASS | 0 | unittest_full.log |

Actual test results có command/count/exit/time trong test_results.json. Các test
fixture không được cộng vào exposure thật. Full suite dùng portable FFmpeg trên
PATH; báo cáo RTSP được giữ nguyên byte sau test. Chỉ kiểm chứng Windows hiện tại,
không tuyên bố đã chạy Linux/macOS. Resume fixture kiểm tra policy/source/raw hashes;
source hoặc artifact thay đổi sẽ bị chặn.

CLI resume trên run thật đã chạy exit0, tái dùng58 record; 464 file record/raw giữ
nguyên byte hash trước/sau (`resume_receipt.json`). Đối chiếu cuối cả63 video vật
lý đều khớp SHA v1 (`source_integrity_receipt.json`). Không phát hiện output lỗi
trong các worker log OpenCV của58 video.

ARTIFACTS:

- Output: `runs/benchmarks/week4_decode_audit_v2/run_20261113/`.
- CLI: `scripts/recover_negative_decode_v2.py`; module: `src/video_audit_v2.py`.
- preflight.json: scope58/source SHA, HEAD, code fingerprints, tools, ordering,
  SHA171 protected files. Bằng chứng bảo toàn được kiểm lại khi summarize.
- records/VIDEO_ID/: record.json + checksum, counts/timeline/ffmpeg raw stdout,
  stderr và command; VIDEO_ID.worker.log lưu OpenCV stderr.
- summary.json, comparison.csv, root_cause_analysis.json: accepted/excluded,
  lý do, per-video counts, duration, diff v1→v2, scope identities/hash.
- Technical eligible scope SHA256: `cc77669f0d5b4695fb7f1f043a6a446ba5ca949ecf46a7cc3f455e667cd89a78`.
- v1-retained và v2-accepted được liệt kê riêng, không tính trùng.
- artifact_sha256.json: checksum bundle sau cùng; source SHA từng video có trong
  preflight, record và eligible_scope_identity.

SCIENTIFIC STATUS:

- Human review hiện tại (read-only SQLite): {'DRAFT': 6}; table counts: {'users': 1, 'review_sessions': 6, 'adjudications': 0, 'audit_log': 10}.
- Không tạo submission/signoff hay đổi human labels; scientific exposure chưa cấp.
- Inference: NOT RUN trong tác vụ này; chưa xác thực full inference timeline cùng scope.
- FAR/hour: null; Poisson CI95%: null; Recall/F1/Delay: chưa tính.
- Operating Point: PROVISIONAL; Official acceptance: false; Scientific Week4 PARTIAL.
- Engineering: PASS cho phạm vi audit v2 trên Windows; technical dataset: PASS.

LIMITATIONS:

- FFmpeg/FFprobe/OpenCV là các lần thực thi độc lập nhưng có thể chia sẻ họ codec
  FFmpeg; không gọi đây là ba thuật toán giải mã độc lập.
- Clean stored-stream EOF không chứng minh camera chưa mất frame trước khi ghi;
  thiếu upstream source để xác thực điều đó. Tính hợp lệ ở đây thuộc luồng được lưu.
- Không nối v2 vào Studio/old scientific gates để tự promote nguồn; artifacts v1
  và trạng thái khoa học được giữ nguyên. Scope mới cần được dùng rõ ràng ở bước sau.

NEXT ACTIONS:

1. Human review các candidate trong pool, Submit và xuất provenance thật.
2. Chuẩn bị instrumented inference cùng source/hash/interval A/B, index/PTS,
   preprocessing/run/checkpoint binding; không dùng aggregate lịch sử làm timeline proof.
3. Khóa ngưỡng trên validation và hoàn thành yêu cầu independent GT trước nghiệm thu.
4. Giữ VN_NEG_10/48 ngoài pool; nếu có nguồn thay thế, đăng ký source identity mới.

Lệnh chạy và chi tiết policy/resume/timeout trong `docs/VIDEO_DECODE_AUDIT_V2.md`.
Run mới phải chọn output mới; preflight từ chối ghi đè thư mục có sẵn.
