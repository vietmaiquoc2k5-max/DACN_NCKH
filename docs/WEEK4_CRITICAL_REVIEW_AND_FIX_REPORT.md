# Week 4 — Báo cáo sửa CI, toàn vẹn dữ liệu và nghiệm thu khoa học

Ngày: 13/11/2026, Asia/Saigon. Tôi thực hiện trên workspace `D:\workspace\DACN_NCKH` của ngài.

**Kết quả: 200/200 unittest PASS cục bộ; nghiệm thu khoa học PARTIAL.** Không commit/push, không huấn luyện, không đổi checkpoint, không chọn lại ngưỡng. Chưa có CI mới cho bản sửa.

## 1. Branch và HEAD đã xác minh

- Branch: `main`.
- HEAD trước sửa và tại bàn giao: `70f6991deccf16ef0b5d9dae840adc58a3e2799e`.
- Commit triển khai trước đó: `8848bc3e6c3dc163e0fbe0f31eb0b9c80c4bf63b`.
- `git status --short` ban đầu rỗng; `git diff --check` ban đầu không có lỗi. Toàn bộ thay đổi trong phiên chưa commit.
- Python 3.11.9, Windows 10 build 26300, NumPy 2.4.6, OpenCV package 5.0.0.93/runtime 5.0.0, PyTorch 2.15.0, Ultralytics 8.4.198. SciPy không được cài; hàm Poisson hiện có dùng PyTorch.
- FFmpeg/FFprobe ban đầu không có trong PATH. Tôi tải bản portable 9.0.2 essentials vào `.tools/ffmpeg`, không cài hệ thống. Nguồn tải và SHA-256 gói nằm trong `.tools/ffmpeg/download.json`; `.tools/` được ignore.
- Chỉ tìm thấy **63/79** video theo đường dẫn manifest hiện tại; checkpoint có trên đĩa. Không đồng nhất môi trường hiện tại với môi trường audit lịch sử.
- Bằng chứng: `runs/benchmarks/week4_critical_fix/preflight.json`.

## 2. GitHub Actions và nguyên nhân hai lỗi

Tôi đọc trực tiếp [run 43289457717](https://github.com/vietmaiquoc2k5-max/DACN_NCKH/actions/runs/43289457717), job `131431501506`. Run ở đúng HEAD trên, kết luận `failure`. Bước Video Audit & Poisson thành công; bước discover toàn bộ unittest chạy 185 test, lỗi đúng hai test được nêu trong yêu cầu.

Log được lưu tại `runs/benchmarks/week4_critical_fix/github_job_131431501506.log`; trạng thái các bước ở `github_jobs.json` cùng thư mục.

| Vấn đề | Mức độ | Nguyên nhân | Ảnh hưởng | Cách sửa | Kiểm chứng | Bằng chứng | Trạng thái |
|---|---|---|---|---|---|---|---|
| Checksum v1.1 trên Linux | P0 | Hash frozen tính trên CRLF; Git đã chuẩn hóa blob thành LF ngay tại commit tạo snapshot | CI không khớp hash, byte snapshot không còn đúng bản frozen | Khôi phục byte CRLF đã được hash lịch sử xác thực, đặt `-text`; không đổi checksum | Test checksum, sửa một byte phải bị từ chối, so toàn bộ JSON với commit gốc | `gt_forensics.json`, `gt_final.log` | PASS cục bộ |
| Manual reconnect ở test 17 | P0 | Test dùng video ngoài fixture; nhánh fallback không có file, rồi chờ cố định 0.2s | Mở nguồn thất bại nên status ERROR; tăng sleep không tạo được file | MJPG tạm có kiểm tra mở/đọc được; chờ điều kiện tối đa 5s; cleanup mỗi test | Test 17, reconnect nguồn đang chạy, thay nguồn lỗi, release capture và shutdown timeout | `baseline_camera.log`, `camera_final.log` | PASS cục bộ |

CI Ubuntu dùng Python 3.10; môi trường kiểm chứng cục bộ là Windows/Python 3.11.9. PASS cục bộ chưa chứng minh CI mới PASS.

## 3. Phân tích pháp chứng Ground Truth snapshot

`events_v1.1_snapshot.jsonl` chỉ có một commit thay đổi trong lịch sử truy xuất được: `a41988cf31819472e64786fb9fc4a5d31472d439`, cũng là commit tạo v1.1 và checksum. Không tìm thấy một phiên bản nhãn mới sau commit đó.

| Dạng byte | SHA-256 |
|---|---|
| Blob Git LF | `0ff6dcae59645b70fb08665c79064eec9f3484aec8b1e3c8cfa9dd5679bff405` |
| Bản frozen CRLF, cũng là bản làm việc Windows ban đầu | `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3` |
| Chuyển riêng LF của blob Git về CRLF | `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3` |

Đối chiếu toàn bộ 35 object JSON và thứ tự bản ghi cho kết quả bằng nhau, bao gồm video, label, start/end, trạng thái, annotator và metadata kiểm định chéo. Snapshot gốc không có `event_id`; tôi không thêm ID vào dữ liệu frozen. Không mất/trùng thêm bản ghi, không sửa timestamp, không nâng pending. Các ID thứ tự trong review queue cũ được giữ nguyên và kiểm tra identity/video/class/onset/offset khi merge.

Tôi lưu hai bản pháp chứng riêng: `events_v1.1_git_lf_forensic.jsonl` và `events_v1.1_original_disk_forensic.jsonl`. Hash các tệp GT liên quan, gồm v1.0 và checksum manifests, ở `gt_forensics.json`. v1.0 không bị thay đổi; không tạo v1.2 hoặc chuyển consumer sang phiên bản mới.

`events.jsonl` có cùng byte/checksum lịch sử với v1.1 nên cũng được bảo vệ khỏi chuẩn hóa newline. Diff Git hiển thị thay đổi 35 dòng mỗi tệp do khôi phục CRLF; nội dung JSON không thay đổi. Thuộc tính `whitespace=cr-at-eol` mô tả chính xác byte CRLF cần giữ.

Phát hiện bổ sung: script standardize cũ suy ra hai annotator và timestamp đồng thuận từ allowlist video. Đây không phải bằng chứng kiểm định độc lập. Tôi thay entry point đó bằng cơ chế từ chối chạy, cả khi snapshot đã tồn tại và khi chưa tồn tại. Các claim `cross_verified` lịch sử được bảo toàn nhưng cần hồ sơ gán nhãn gốc để xác thực. **Hash hợp lệ không chứng minh nhãn đã được con người kiểm tra chéo.**

## 4. Phân tích reconnect và vòng đời worker

Manager cập nhật nguồn bằng cách stop worker và chỉ start lại theo thao tác được gọi. Không thấy bằng chứng nguyên nhân trực tiếp của test 17 là worker giữ ERROR sau khi mở được nguồn hợp lệ: nguồn fallback không tồn tại, lỗi đã tái hiện cục bộ trước sửa.

Tôi dùng video MJPG 64×48, 30 frame, 10 FPS trong TemporaryDirectory cho bộ test camera; OpenCV xác nhận fixture mở và đọc được trước khi chạy test. Test 17 chờ ERROR rồi thay nguồn và manual reconnect; test 9 kiểm tra hai thread cũ đã dừng trước khi reconnect hoàn tất. Test 23 kiểm tra VideoCapture thực đã release; test 24 mô phỏng thread bị kẹt và kiểm tra giữ handle, từ chối thread trùng.

Trong production, lifecycle được tuần tự hóa bằng RLock; kết nối ban đầu mang `CONNECTING`, lỗi/kết nối lại tự động vẫn dùng `RECONNECTING`. Khoảng chờ retry có thể được ngắt bằng stop event. `stop()` chỉ bỏ handle sau khi thread dừng; timeout báo RuntimeError và giữ handle cho xử lý/retry, thay vì âm thầm mở thêm thế hệ worker. Backend RTSP thật bị treo chưa được kiểm chứng trên phần cứng.

Dashboard vẫn chọn camera với `auto_start=False`; test manual start/no auto video vẫn chạy trong bộ camera. Tôi không thay điều khiển UI, inference, checkpoint hoặc Telegram.

## 5. Các tệp đã đọc và thay đổi

Tệp đã đọc/đối chiếu: `.github/workflows/ci.yml`; `src/video_audit.py`, `event_eval.py`, `temporal_verifier.py`, `camera_manager.py`; phần điều khiển kết nối trong `web_app.py`; `scripts/run_phase1_inventory.py`, `run_phase3_final_audit.py`, `run_phases4_to_8_recovery.py`, `standardize_ground_truth_v1_1.py`, `run_week4_complete_validation.py`, `export_week4_frame_jsonl.py`; các test GT/camera/video audit/Poisson/consistency; GT snapshots, checksum và pending lists; negative manifest; inference run manifest/config/summary/errors/event records; artifacts trong `week4_final_recovery`; `docs/WEEK4_FINAL_RECOVERY_AND_ACCEPTANCE_REPORT.md`; Git history và log Actions.

Tệp mã/cấu hình/dữ liệu thay đổi chính xác:

- `.gitattributes` (mới), `.gitignore`.
- `data/ground_truth/events.jsonl`, `data/ground_truth/events_v1.1_snapshot.jsonl` (chỉ khôi phục byte CRLF).
- `src/camera_manager.py`.
- `src/ground_truth_integrity.py`, `src/week4_acceptance.py` (mới).
- `scripts/standardize_ground_truth_v1_1.py`, `scripts/run_phases4_to_8_recovery.py`.
- `tests/test_ground_truth_cross_validation.py`, `tests/test_dynamic_camera_selection.py`, `tests/test_week4_critical_acceptance.py` (tệp cuối mới).
- Báo cáo này (mới).
- Bằng chứng, log, bản pháp chứng và artifacts mới trong `runs/benchmarks/week4_critical_fix/`; danh sách đệ quy/hash cuối cùng ở `handoff_inventory.json`.

Không sửa workflow, manifest nguồn, video, checkpoint hoặc artifacts lịch sử. Một test hiện có tự ghi báo cáo RTSP; tôi đã khôi phục tệp đó về nội dung ban đầu sau kiểm chứng để không sửa kết quả lịch sử ngoài phạm vi.

## 6. FAR và điều kiện scope

Tôi tính lại từ cùng bộ per-video audit và event records, lưu **riêng** tại `runs/benchmarks/week4_critical_fix/recalculated/`. Đây là phép đối soát artifacts lịch sử, không phải một lượt full decode mới.

| Scope | Video | Giờ | A episodes | B2 episodes | Diễn giải |
|---|---:|---:|---:|---:|---|
| A: chẩn đoán lịch sử | 79 | 7.1311539536 | 988 | 421 | Giảm 57.3887%, chẩn đoán |
| B: kỹ thuật nghiêm ngặt đầy đủ | 0 | 0 | 0 | 0 | Chưa có bằng chứng alignment inference đạt yêu cầu |
| B3: sơ bộ không truncated | 77 | 6.9846327379 | 971 | 417 | Giảm 57.0546%, chỉ chẩn đoán, còn NEEDS_REVIEW |
| C: âm tính độc lập đủ điều kiện | 0 | 0 | 0 | 0 | BLOCKED, rate và CI là null |

Giá trị chưa làm tròn chính xác nằm trong JSON. Con số B3 ở bảng được hiển thị làm tròn; máy đọc phải dùng `far_final.json`.

Strict scope yêu cầu đồng thời `NORMAL_EOF`, hash nguồn hiện tại khớp, full decode verified, số decode errors bằng 0, alignment inference verified và duration hữu hạn >0. Mọi `NEEDS_REVIEW`, lỗi decode và trạng thái lạ đều bị loại. Riêng nhóm đạt điều kiện decode lịch sử có **19 video/2.780386837h**, chưa đủ 5 giờ và chưa đủ toàn bộ bằng chứng strict scope.

Scope A: A = 138.5470 episode/h; B2 = 59.0367 episode/h, CI Poisson 95% `[53.5313, 64.9546]`. Đây là CI cho **alert-event rate chẩn đoán**, không chứng nhận FAR trên dữ liệu âm tính độc lập. Không ghi FAR=0 khi exposure=0.

## 7. So sánh baseline theo sự kiện

Tôi bỏ baseline cố định `137.1727` khỏi script recovery. A, B1, B2 đều lấy numerator từ event records thực tế, cùng video ID và cùng khoảng `[0, decoded_frames/native_fps)` cho từng scope. Không fallback sang số cảnh báo frame trong summary khi thiếu event records.

Tôi đối chiếu code lịch sử: cả ba pipeline đã gộp cảnh báo thành episode với gap=2s. Bản sửa kiểm tra lại cùng quy tắc video/camera/class, clip về khoảng đánh giá và merge gap≤2s. Kết quả A/B1/B2 vẫn lần lượt **988/547/421**; không có episode bị cắt hoặc loại vì nằm ngoài interval. Raw frame alarms và episode được phân biệt.

So sánh 79-video baseline với 77-video B2 trước đây không được dùng cho nghiệm thu. Cả so sánh cùng scope mới cũng chưa là bằng chứng cải thiện FAR khoa học vì thiếu negative GT và alignment timeline độc lập.

## 8. Quyết định tái dùng inference

**Verdict: `BLOCKED_MISSING_INPUT`; `rerun_required=null`.** Tôi không chạy lại inference trên dataset.

Checkpoint hiện tại khớp `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`. ID/path map, decoded frame counts, cấu hình lưu trữ, completion và error records được đối chiếu. Tất cả 79 số sampled-frame tổng hợp khớp phép tính stride lịch sử; đây vẫn chưa xác minh từng timestamp hay chứng minh không bỏ frame suy luận.

Thiếu 16 video `negsVideo*` thuộc `datasets/test_videos/neg/` tại đường dẫn manifest. 63 video hiện có được hash lại; không phát hiện mismatch với manifest/audit của chúng. Không có fingerprint preprocessing đầy đủ, sampling schedule/timestamp từng frame, skip records đầy đủ hoặc bản manifest lịch sử có hash `75626b6b…` được xác thực. Hash manifest hiện tại khác historical manifest không tự chứng minh video thay đổi; trạng thái đó được giữ UNVERIFIED.

Các trạng thái quyết định được kiểm thử: REUSE_VERIFIED, REUSE_WITH_LIMITATIONS, RERUN_REQUIRED, BLOCKED_MISSING_INPUT, UNVERIFIED. Không còn gán `rerun_required=false` vô điều kiện. Chi tiết từng kiểm tra và video thiếu ở `inference_rerun_decision.json`, từng hash ở `inference_reconciliation_final.json`.

## 9. Video audit

Artifacts lịch sử báo 79 video, 929,939 decoded frames, 19 NORMAL_EOF, 58 NEEDS_REVIEW, 2 TRUNCATED_STREAM. Tôi giữ nguyên báo cáo đó và ghi rõ chúng chưa được tái chạy full decode ở workspace hiện tại. Việc cài FFmpeg chỉ cho phép kiểm chứng các fixture nhỏ trong unit/integration tests; không biến thành full dataset audit PASS.

## 10. Trạng thái Ground Truth

Snapshot v1.1 có 35 bản ghi: **24 pending_verification**, **11 cross_verified được ghi nhận lịch sử**. Chưa xác thực lại chứng từ gán nhãn độc lập cho 11 bản ghi đó. Review queue âm tính hiện có **79 pending**, exposure độc lập đủ điều kiện bằng **0h**.

Regenerate các CSV giữ nguyên annotator, quyết định, adjudication, timestamps, evidence references và các cột bổ sung; reject header/row lỗi, ID trùng và thay đổi source interval. Chỉ append identity mới, không xóa identity cũ. Khi cần thay nội dung queue hiện có, lưu backup theo hash trong cùng thư mục output rồi atomic replace. Chạy lại trên ba queue hiện tại cho kết quả byte không đổi; regression còn mô phỏng review một phần và VERIFIED để kiểm tra không mất tiến độ.

Không tự động xác minh và không dùng nhãn AI làm independent GT. Snapshot chỉ đọc, có gate checksum trước khi tạo positive queue.

## 11. Operating Point

Giữ **PROVISIONAL**: eval confidence 0.35 (raw inference confidence 0.25), K=3, N=8, feature=0.02, cooldown=5s, gap=2s, sampling target=5 FPS. Không retune, không đổi checkpoint.

Tách riêng `alert_rate_reduction_target_met=true` cho chẩn đoán; `far_reduction_verified=null`, Recall constraint=null, Delay constraint=null, negative exposure requirement=false, official eligibility=false, Week 4=PARTIAL. Không giữ các PASS/Recall/Delay hardcode trong báo cáo mới. Khóa chọn ngưỡng trên validation cũng cần provenance; giữ nguyên tham số không đồng nghĩa đã chứng minh quy trình chọn ngưỡng.

## 12. Provenance và sai lệch exposure

- Inference source base: `256065e7e54c35c335e6953d0bc373829ac94997`; packaged commit: `62785d7b4cdd1b499c99abbd016ecedf6fa1ee35`.
- Inventory lịch sử tự khai `git_head=015811cad1b44073e51893b80c46ad04a5cc5c0c`; nhiều artifact được đóng gói tại `8848bc3e…`. Không thay các commit này bằng HEAD mới.
- Packaging commit không đủ xác lập executing commit/worktree. Các trường historical execution/environment/start/completion thiếu được giữ unknown/null; không bịa thông tin còn thiếu.
- Artifact mới lưu producer base HEAD, diff digest, source file digests, manifest/checkpoint/source-video hashes, audit policy, OpenCV/FFmpeg/Python/platform, thời điểm bắt đầu/kết thúc và các scope/interval.
- `7.1309h`: tổng các `duration_hours` đã làm tròn từng video. Tổng `duration_seconds` là `7.13115h`; dùng decoded frames/native FPS cho `7.131153953574723h`. Cùng 79 video: sai lệch nhỏ này do làm tròn trung gian.
- Historical inference `7.2026h` dùng tổng `vrec.duration_seconds` lịch sử; khác mẫu số physical-decode hiện tại. Không được giải thích toàn bộ khác biệt này là rounding hoặc tự xác nhận timeline coverage khi chưa có nguồn timestamp/bản manifest gốc.
- Bằng chứng máy đọc: `recalculated/artifact_lineage.json`, `consistency_report.json`.

## 13. Kết quả test thực tế

| Lệnh | PASS | FAIL | ERROR | SKIP | Exit | Log trong `week4_critical_fix/` |
|---|---:|---:|---:|---:|---:|---|
| `python -m unittest discover -s tests -p test_ground_truth_cross_validation.py -v` | 8 | 0 | 0 | 0 | 0 | `gt_final.log` |
| `python -m unittest discover -s tests -p test_dynamic_camera_selection.py -v` | 24 | 0 | 0 | 0 | 0 | `camera_final.log` |
| `python -m unittest discover -s tests -p test_week4_critical_acceptance.py -v` | 10 | 0 | 0 | 0 | 0 | `acceptance_final.log` |
| `python -m unittest discover -s tests -p test_*.py -v` | 200 | 0 | 0 | 0 | 0 | `full_final.log` |

Đây là unittest discovery; không tuyên bố các test dạng hàm chỉ dành cho pytest đã được unittest thực thi. Test results JSON lưu chính xác command/count/exit/log. Các test giải mã dùng video fixture thực; thông báo decoder trên video cố ý làm hỏng là một phần test lỗi.

Baseline trước sửa: GT 5/5 PASS trên Windows CRLF; camera 21/22 PASS, test 17 FAIL. Không có baseline full-suite trước mọi chỉnh sửa. Lần full-suite sau sửa nhưng chưa có FFmpeg chạy 187 test: 10 FAIL, 1 SKIP; sau bổ sung FFmpeg chạy 197 test PASS; sau thêm regression cuối chạy 200 test PASS. Tôi giữ log các lần chạy, không xóa kết quả thất bại.

Script recovery exit 0. Integration regenerate ba review CSV giữ nguyên byte. `git diff --check` sau sửa không có lỗi. Các kiểm tra đều là cục bộ; cần CI riêng sau khi ngài cho phép push.

## 14. Tổng hợp mức độ và giới hạn

| Vấn đề | Mức độ | Nguyên nhân | Ảnh hưởng | Sửa | Test | Bằng chứng | Trạng thái cuối |
|---|---|---|---|---|---|---|---|
| Scientific PASS thiếu GT | P1 | Gộp chẩn đoán và nghiệm thu, dùng giá trị cố định | Có thể chấp nhận MT3 sai | Gate riêng và null cho bằng chứng thiếu | Missing GT dù reduction 99% không PASS | `acceptance_final.log`, `far_final.json` | Logic PASS; khoa học PARTIAL |
| Scope B3 | P1 | Chỉ loại TRUNCATED_STREAM | NEEDS_REVIEW bị coi strict | Allowlist kèm toàn bộ evidence; B3 đổi thành diagnostic | Loại mọi status lỗi/lạ và từng trường thiếu | `test_week4_critical_acceptance.py` | PASS cục bộ |
| Baseline khác scope/đơn vị | P1 | Rate global cố định và fallback raw frame count | Reduction không công bằng | Cùng IDs/interval/episode protocol | Scope matching, clip/merge, empty exposure | `normalized_event_episodes.json`, `far_final.json` | Diagnostic đối soát; chưa FAR verified |
| Reuse inference vô điều kiện | P1 | Quyết định hardcode | Che thiếu timeline/provenance | State machine theo checks | Năm trạng thái quyết định | `inference_rerun_decision.json` | BLOCKED_MISSING_INPUT |
| Mất tiến độ review | P1 | Ghi đè template | Mất annotator/adjudication | Merge bảo toàn, reject malformed, backup/atomic replace | Partial/verified review và rerun ba queue | `integration_results.json` | PASS cục bộ |
| Provenance/duration | P2 | Lẫn source/producer, làm tròn trước cộng | Báo cáo khó tái lập | Lineage tách nguồn, tính từ frame/FPS | Reconciliation và precision test | `consistency_report.json`, `artifact_lineage.json` | PARTIAL do nguồn lịch sử thiếu |
| Live RTSP/Linux CI | P2 | Chưa chạy với bản sửa | Chưa chứng minh hành vi phần cứng/CI | Bàn giao để chạy đúng môi trường | Chưa thực hiện | Log cục bộ và job lịch sử | NOT_TESTED cho bản sửa |

## 15. Bước còn lại và lệnh tái lập

1. Khôi phục đúng 16 video thiếu theo manifest; hash lại và giữ nguồn video bất biến.
2. Thu hồi manifest/checkpoint/preprocessing/sampling provenance của run gốc; đối soát skip/error records. Nếu không thể xác minh timeline thì lập run inference mới vào thư mục mới, lưu timestamp của mọi sampled frame, kể cả không detection. Không chạy entry point legacy ghi đè manifest/Operating Point.
3. Hoàn thành human review âm tính độc lập để có ít nhất 5 giờ vừa đạt kỹ thuật vừa verified-negative; không cộng NEEDS_REVIEW vào exposure nghiêm ngặt.
4. Hoàn thành 24 pending positive và xác thực chứng từ của 11 cross_verified lịch sử; tính Event Recall/F1, Delay và ràng buộc giảm Recall bằng independent eligible GT.
5. Xác thực validation/test split theo video và provenance chọn/khóa Operating Point trước test. So sánh A/B2 cùng scope và episode protocol; báo FAR/h và CI95% khi đủ điều kiện.
6. Sau khi ngài cho phép commit/push, kiểm tra GitHub Actions của đúng SHA mới. Chạy kiểm thử RTSP thật theo kế hoạch triển khai riêng.

```powershell
$env:PATH = (Join-Path $PWD '.tools\ffmpeg') + ';' + $env:PATH
ffmpeg -version
ffprobe -version
python -m unittest discover -s tests -p "test_ground_truth_cross_validation.py" -v
python -m unittest discover -s tests -p "test_dynamic_camera_selection.py" -v
python -m unittest discover -s tests -p "test_week4_critical_acceptance.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_phases4_to_8_recovery.py --output-dir runs/benchmarks/week4_critical_fix/recalculated
git diff --check
```

FFmpeg portable không nằm trong Git; trên Linux cần có `ffmpeg` và `ffprobe` trong PATH như workflow hiện tại. Nguồn Windows được liên kết từ [trang tải chính thức FFmpeg](https://ffmpeg.org/download.html). Các lệnh test trong repo có thể tạo artifacts runtime; test RTSP hiện có tự ghi báo cáo NOT_TESTED, cần giữ hoặc khôi phục nội dung lịch sử như đã nêu.

Chưa có bằng chứng yêu cầu rerun inference dứt khoát nên tôi không cung cấp lệnh chạy lại như một bước đã được quyết định. Verdict hiện tại yêu cầu bổ sung input và lineage trước. Ngài có thể tái lập toàn bộ sửa phần mềm và đối soát hiện có bằng các lệnh trên mà không chạy lại mô hình trên dataset.
