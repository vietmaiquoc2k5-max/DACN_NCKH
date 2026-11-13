# FIX-4 — Kiểm chứng đa nền tảng và điều kiện bằng chứng

Ngày: 13/11/2026, Asia/Saigon. Workspace: `D:\workspace\DACN_NCKH`.

Tôi đã kiểm chứng bản sửa của ngài và bổ sung các điều kiện còn thiếu. **Windows: 211/211 unittest PASS. Engineering tổng thể: PARTIAL vì Linux/CI của SHA được review chưa được xác minh. Scientific Week 4: PARTIAL, với Ground Truth ở trạng thái BLOCKED_HUMAN_REVIEW.**

## HEAD, diff và CI

- Branch: `fix-4`; HEAD: `30239cc18f07a963b97461e7191b8d11e2e70cbf`, khớp yêu cầu. Working tree ban đầu sạch, branch theo `origin/fix-4`.
- Tôi đọc diff `main...fix-4`: 47 tệp từ bản sửa trước. Baseline trên HEAD chưa chỉnh sửa chạy 200 test, exit 0, không FAIL/ERROR/SKIP; log `baseline_windows_full.log`.
- Lần đầu `git diff main...fix-4 --check` exit 2: archived Actions log có trailing whitespace và dòng trống cuối, không phải lỗi nội dung Python/GT. Tôi giữ nguyên log, chỉ thêm `-diff` cho đúng một tệp log lưu nguyên văn trong `.gitattributes`. Hai lệnh diff-check hiện exit 0 **khi dùng thuộc tính mới trong working tree**. Thay đổi này chưa nằm trong reviewed SHA và chưa được push.
- Workflow giữ nguyên: push chỉ vào `main/master`; PR có base `main/master`. Push riêng `fix-4` không kích hoạt workflow hiện tại.
- Truy vấn GitHub tất cả trạng thái cho `head:fix-4 base:main` không tìm thấy PR. Connector truy vấn PR-triggered workflow của SHA trên trả về danh sách rỗng. Connector này không cung cấp toàn bộ push runs qua truy vấn đó; tôi không dùng danh sách rỗng để chứng minh mọi loại run đều không tồn tại.
- **CI run URL của reviewed SHA: không có run phù hợp được tìm thấy (`null`); conclusion: NOT_VERIFIED.** Không có PR CI để kiểm tra, không tạo PR hoặc rerun workflow. Run lỗi cũ [43289457717](https://github.com/vietmaiquoc2k5-max/DACN_NCKH/actions/runs/43289457717) thuộc HEAD `70f6991d…`, không phải bằng chứng CI cho `30239cc1…`.
- Bằng chứng: `runs/benchmarks/week4_fix4_verification/git_ci_verification.json`, `fix4_prs.json`, `fix4_runs.json`.

Mọi kết quả 211 test dưới đây thuộc **working tree có bổ sung FIX-4 chưa commit**, dựa trên HEAD trên; không gắn nhầm chúng cho một commit CI mới.

## Kiểm chứng snapshot và clean checkout

Tôi tạo hai local clone sạch, thực hiện Git checkout thực của reviewed SHA, chỉ lấy `.gitattributes`, snapshot và checksum. Một clone dùng `core.autocrlf=false`, clone còn lại dùng `true`.

| Kiểm tra | Kết quả | Bằng chứng |
|---|---|---|
| Working tree Windows và hai checkout sạch | PASS: cùng SHA-256 frozen | `clean_checkout_windows.json`, `targeted_gt.log`, `targeted_fix4.log` |
| Byte CRLF | PASS: 35 CRLF trong mỗi checkout; không normalize khi xác minh | `clean_checkout_windows.json` |
| LF thay cho CRLF | PASS cho khả năng phát hiện lỗi: checksum bị từ chối | `test_lf_conversion_is_detected_without_hash_fallback` |
| Sửa snapshot dù một byte | PASS cho khả năng phát hiện lỗi | `test_snapshot_mutation_rejected` |
| JSON/annotation status so với main và bản pháp chứng | PASS: 35 bản ghi không thay đổi | `json_matches_main=true`; GT regression |
| Checkout/test trên Linux thật | NOT VERIFIED | Không có Docker trong PATH; WSL liệt kê thành công nhưng không có distro; chưa có CI phù hợp |

Hash frozen giữ nguyên: `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3`. Dạng LF có hash `0ff6dcae59645b70fb08665c79064eec9f3484aec8b1e3c8cfa9dd5679bff405` và không được chấp nhận thay thế.

Không sửa snapshot, checksum, label, thời gian hoặc trạng thái annotation. Vẫn 24 `pending_verification` và 11 `cross_verified` được ghi nhận lịch sử; hồ sơ kiểm định độc lập của 11 bản ghi chưa được xác thực lại. Không tự nâng pending. Test clean checkout mới chạy được trên Windows và sẽ chạy trên Linux khi workflow được kích hoạt; **mô phỏng chính sách Git Linux trên Windows chưa phải Linux PASS**.

## Camera và tài nguyên

Bộ camera chạy 25 test PASS trên Windows. Fixture MJPG tạm được tạo và đọc kiểm tra bằng OpenCV, không cần dataset ngoài. Các kiểm tra đã chạy gồm nguồn sai → ERROR, thay nguồn rồi manual reconnect → CONNECTED, không mở thread trùng, startup không tự mở camera, giải phóng VideoCapture, giữ handle nếu shutdown timeout và cleanup.

Test mới `test_25_observable_manual_connection_transitions` chặn thao tác mở capture bằng Event có timeout 5s để quan sát trạng thái thật: DISCONNECTED → CONNECTING → ERROR, sau thay nguồn và manual reconnect quan sát RECONNECTING → CONNECTED. Gọi start lặp giữ nguyên thread. Sau cập nhật nguồn, manager có bước DISCONNECTED hợp lệ trước thao tác reconnect; test không bỏ qua bước này. Việc chờ trạng thái dùng deadline hữu hạn; không tăng sleep để che lỗi.

Tôi không sửa production CameraManager trong lượt này. Live RTSP và camera phần cứng chưa được chạy; Linux chưa được chạy. Bằng chứng: `targeted_camera.log`, `windows_full.log`.

## Raw frame evidence và quyết định từng video

Tôi đọc toàn bộ bốn JSONL hiện có, tính hash, đếm từng video và kiểm tra trường dữ liệu:

| Tệp trong `runs/benchmarks/week4_final/` | Frame records | Video IDs | Timestamp đã lưu | Frame index / input config trong record |
|---|---:|---|---|---|
| `frame_level_detections.jsonl` | 4,289 | 24 `POS_*` | Có | Không / Không |
| `tracked_frame_detections.jsonl` | 4,289 | 24 `POS_*` | Có | Không / Không |
| `negative_raw_detections.jsonl` | 6,306 | 5 `REAL_NEG_*` | Có | Không / Không |
| `negative_tracked_detections.jsonl` | 6,306 | 5 `REAL_NEG_*` | Có | Không / Không |

Các cặp raw/tracked mô tả cùng các frame, **không cộng chúng thành exposure hai lần**. Các tệp có schema `video_id/camera_id/timestamp_sec/detections`; số sampled records tính bằng đọc từng dòng, không phải một trường count lưu sẵn. Không tìm thấy JSON lỗi trong bốn tệp. Chúng không có ID trùng với 79 video của target run, không có run/source/checkpoint binding trong từng record. Không suy diễn alias `REAL_NEG_* → VN_NEG_*`.

Code exporter hiện có tính timestamp bằng frame index/native FPS rồi chỉ ghi timestamp, bỏ frame index; đây không phải bằng chứng timestamp PTS độc lập của target run. Default `imgsz` trong source exporter cũng không xác lập được input config thực tế của từng artifact lịch sử. Tôi không dùng chúng để xác nhận coverage của run `run_20261111_233035_neg_full`.

Tôi bổ sung inspector streaming và quyết định reuse **cho từng video**, dùng cùng checks với quyết định tổng. Chỉ xét frame evidence sau khi khớp run ID, source SHA, checkpoint SHA; thiếu metadata giữ UNVERIFIED. Full timeline còn yêu cầu frame indices, timestamps hợp lệ, count/config phù hợp và reference timestamp độc lập. Count tổng hợp hoặc tự khai `timestamp_source=source_pts` chưa đủ chứng minh coverage. Những reference trong unit tests là fixture tổng hợp, không phải chứng từ dataset.

| Phân loại reuse thực tế | Số video | Lý do |
|---|---:|---|
| UNVERIFIED | 63 | Source hiện có/hash khớp, nhưng thiếu frame evidence gắn đúng run, preprocessing/sampling/manifest lineage |
| BLOCKED_MISSING_INPUT | 16 | Thiếu video nguồn `negsVideo*` theo manifest và vẫn thiếu bằng chứng timeline |
| REUSE_VERIFIED | 0 | Chưa đủ toàn bộ checks |
| REUSE_WITH_LIMITATIONS | 0 | Các checks khác cũng chưa đầy đủ để áp dụng trạng thái này |
| RERUN_REQUIRED | 0 | Chưa có kiểm tra mâu thuẫn đủ để ép rerun; thiếu bằng chứng không tự biến thành false |

Verdict tổng: **BLOCKED_MISSING_INPUT**, `rerun_required=null`. Không chạy inference trên dataset. Việc không tìm được commit cũ trong checkout nông được sửa thành UNVERIFIED thay vì coi là mâu thuẫn bắt buộc rerun.

**Phạm vi rerun nhỏ nhất có cơ sở nếu bắt buộc tạo đủ bằng chứng timeline cho cả run 79 video:** 79 ID còn thiếu frame evidence, gồm 63 source đang có và 16 source bị chặn. Đây là danh sách có điều kiện, không phải quyết định tự chạy 79 video. Trước tiên thu hồi raw logs/manifest/provenance đúng run; loại từng video khỏi danh sách khi có bằng chứng được đối soát. Chưa thể chỉ ra một subset nhỏ hơn đáp ứng toàn scope, hoặc đủ 5h strict/semantic-negative, bằng dữ liệu hiện tại. Danh sách chính xác nằm trong `rerun_scope_if_independent_timeline_required`; không rerun video khác hoặc model khác.

Bằng chứng trong `week4_fix4_verification/recalculated/`: `raw_frame_jsonl_audit.json`, `inference_reuse_per_video.json`, `inference_reconciliation_final.json`, `inference_rerun_decision.json`.

## FAR, Ground Truth và Operating Point

| Scope | Video | Giờ | Kết luận |
|---|---:|---:|---|
| A: nominal-negative chẩn đoán | 79 | 7.1311539536 | A/B2 = 988/421 episodes; giảm 57.3887%, chưa phải verified FAR |
| B: strict kỹ thuật đầy đủ | 0 | 0 | Thiếu alignment inference; NEEDS_REVIEW không được nhận vào strict |
| B3: sơ bộ không truncated | 77 | 6.9846327379 | A/B2 = 971/417 episodes; chỉ chẩn đoán |
| C: negative GT độc lập đủ điều kiện | 0 | 0 | BLOCKED_HUMAN_REVIEW; FAR/CI không có exposure là null |

Nhóm chỉ đạt decode lịch sử có 19 video, khoảng 2.780386837h; điều này không tự cấp điều kiện strict hoặc semantic-negative. Tôi không chạy lại full physical decode, không xác nhận 79 source hiện có: workspace chỉ có **63/79**.

Baseline/temporal đều dùng cùng video IDs, interval `[0, decoded_frames/native_fps)` và cùng protocol gộp episode video/camera/class, gap≤2s. Không dùng raw frame alarms thay episode; không dùng baseline khác scope. `far_final.json` lưu numerator, denominator, intervals và CI chẩn đoán. Không tái xác nhận Event Recall/Delay từ các giá trị frame-level hoặc số cố định.

Tôi thêm gate `independent_positive_gt_complete`: chỉ True khi caller có đủ bằng chứng, mặc định None; pending False chặn official eligibility kể cả khi mọi gate khác True. Trên dataset hiện tại giá trị là False từ 24 pending. Negative exposure vẫn 0h; Recall constraint/Delay/FAR reduction verified đều null. **Operating Point PROVISIONAL, official eligibility=false; Week 4 PARTIAL.** Không thay confidence 0.35/raw 0.25, K=3, N=8, feature=0.02, cooldown=5s, gap=2s, checkpoint hoặc chọn lại ngưỡng.

Sinh lại ba review CSV trong output FIX-4 cho kết quả byte không đổi, exit 0. Regression fixture kiểm tra cả quyết định human một phần và VERIFIED được bảo toàn, không đưa fixture đó vào Ground Truth thật. Các CSV nguồn và artifacts cũ giữ nguyên.

## Nguyên nhân, sửa và kiểm tra lại

| Vấn đề | Ảnh hưởng | Sửa | Kiểm tra và trạng thái |
|---|---|---|---|
| Verbatim Actions log chứa whitespace | Diff-check baseline thất bại | Một quy tắc `-diff` đúng tệp lưu bằng chứng; không sửa log | Hai diff-check exit 0 với worktree attributes; nội dung log không đổi |
| Bản trước chưa đối soát raw JSONL và thiếu verdict từng video | Không chỉ ra video nào có thể reuse | Inspector streaming, bind provenance, verdict per-video và danh sách rerun có điều kiện | Tests missing timestamp/source, wrong run/video, mismatch reference/count PASS; thực tế 63 UNVERIFIED/16 BLOCKED |
| Gate positive GT không có tham số riêng | Helper có thể cho OFFICIAL nếu caller truyền mọi gate cũ True dù positive GT chưa hoàn tất | Thêm gate bảo thủ, đọc pending count từ snapshot đã xác minh hash | Test pending/unknown GT chặn OFFICIAL PASS |
| Source commit không có trong local clone bị coi là false | Checkout nông có thể yêu cầu rerun sai | Giữ None khi chưa có commit để xác thực | Không biến thiếu bằng chứng thành mâu thuẫn; trạng thái thực ghi trong checks |
| CLI có thể ghi vào thư mục artifact đã tracked | Có thể ghi đè bằng chứng cũ nếu chọn output cũ | Default output FIX-4 mới; từ chối output chứa tracked files | Test gọi CLI với output lịch sử exit 2 và nội dung không đổi |

## Test đã chạy và môi trường

Windows 10 build 26300, Python 3.11.9, NumPy 2.4.6, OpenCV package 5.0.0.93, Torch 2.15.0, Ultralytics 8.4.198. SciPy chưa cài; test Poisson dùng implementation PyTorch hiện có. FFmpeg/FFprobe portable 9.0.2 trong `.tools/ffmpeg` được đưa vào PATH cho subprocess test, không đổi dependencies.

| Lệnh unittest discovery, `-s tests -p PATTERN -v` | PASS | FAIL | ERROR | SKIP | Exit | Log |
|---|---:|---:|---:|---:|---:|---|
| `test_*.py` trên reviewed HEAD trước chỉnh sửa | 200 | 0 | 0 | 0 | 0 | `baseline_windows_full.log` |
| `test_fix4_evidence_gates.py` | 10 | 0 | 0 | 0 | 0 | `targeted_fix4.log` |
| `test_ground_truth_cross_validation.py` | 8 | 0 | 0 | 0 | 0 | `targeted_gt.log` |
| `test_dynamic_camera_selection.py` | 25 | 0 | 0 | 0 | 0 | `targeted_camera.log` |
| `test_week4_critical_acceptance.py` | 10 | 0 | 0 | 0 | 0 | `targeted_acceptance.log` |
| `test_*.py` sau bổ sung FIX-4 | **211** | **0** | **0** | **0** | **0** | `windows_full.log` |

Lệnh đầy đủ, thời điểm bắt đầu/kết thúc, Python, dependencies và exit codes ở `windows_test_results.json`. Script reconciliation exit 0 (`reconciliation_execution.json`); regenerate review exit 0 (`review_regeneration_result.json`). Windows không có skip không có nghĩa Linux đã chạy; Linux/CI là NOT VERIFIED. Unittest không tự thực thi các function-only pytest tests; tôi không cộng chúng vào 211 test.

Mười regression FIX-4 cùng test camera mới bao phủ: clean checkout hai chính sách Git; CRLF/LF; timestamp/frame index thiếu; reference timestamp; sai run/video/checkpoint/source binding; count mismatch; source thiếu; pending GT; chống ghi đè tracked output; quan sát state transition. Bộ cũ bổ sung mutation snapshot, strict zero eligible scope và giữ human decisions qua lần sinh queue tiếp theo.

## Tệp đã đọc và thay đổi

Tôi đọc toàn bộ các tệp ngài chỉ định, diff với main, historical inference manifest/config/summary/events, artifacts reconciliation cũ, bốn raw/tracked JSONL và code exporter có liên quan. Tôi đọc production CameraManager/start/stop/reconnect và các test; không sửa implementation camera trong lượt này.

Thay đổi trong lượt FIX-4 này:

- `.gitattributes`: quy tắc hẹp cho archived Actions log.
- `src/week4_acceptance.py`: gate positive GT.
- `src/inference_frame_evidence.py` (mới): đọc và đối soát evidence từng frame.
- `scripts/run_phases4_to_8_recovery.py`: đối soát raw evidence, per-video verdict, rerun scope có điều kiện, xử lý commit thiếu, pending GT, default output mới và chống ghi đè tracked output.
- `tests/test_fix4_evidence_gates.py` (mới), `tests/test_dynamic_camera_selection.py`, `tests/test_week4_critical_acceptance.py`.
- Báo cáo này và bằng chứng mới trong `runs/benchmarks/week4_fix4_verification/`; manifest/hash tệp bàn giao ở `verification_inventory.json`.

`src/ground_truth_integrity.py`, `src/camera_manager.py`, workflow, snapshot/checksum, source videos/checkpoint, manifest và các artifacts lịch sử đều không đổi. Test RTSP hiện có ghi vào báo cáo cũ khi chạy; tôi khôi phục đúng byte trước test sau mỗi lượt. `historical_preservation.json` ghi diff exit 0 cho các phạm vi được bảo vệ.

## Bước còn thiếu và tái lập

Engineering chỉ được PASS khi CI trên SHA ngài review chạy thành công và các kiểm tra còn lại giữ nguyên. Hiện chưa có PR phù hợp, chưa có commit/push mới, không có Linux runtime khả dụng; tôi để PARTIAL và không tự tạo PR.

Scientific acceptance còn bị chặn bởi human review âm tính ≥5h đủ điều kiện, 24 pending positive và provenance độc lập của GT lịch sử, timeline/source/config/manifest binding của inference, Event Recall/F1/Delay hợp lệ, cùng provenance validation/test split và Operating Point đã khóa trước test. Những việc gán nhãn được ghi **BLOCKED_HUMAN_REVIEW**; không thay bằng kiểm định AI.

Tái lập Windows từ workspace, ghi artifacts vào thư mục mới chưa tracked:

```powershell
git status
git rev-parse HEAD
git diff main...fix-4 --check
$env:PATH = (Join-Path $PWD '.tools\ffmpeg') + ';' + $env:PATH
python -m unittest discover -s tests -p "test_fix4_evidence_gates.py" -v
python -m unittest discover -s tests -p "test_ground_truth_cross_validation.py" -v
python -m unittest discover -s tests -p "test_dynamic_camera_selection.py" -v
python -m unittest discover -s tests -p "test_week4_critical_acceptance.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_phases4_to_8_recovery.py --output-dir runs/benchmarks/week4_fix4_verification/recalculated
git diff --check
```

Nếu output đã được commit, chọn một đường dẫn mới; CLI chủ động từ chối ghi đè tracked artifacts. FFmpeg portable không nằm trong Git. Trên Linux/CI cần Git, Python và dependencies/FFmpeg như workflow, sau đó chạy cùng lệnh unittest. Test clean checkout tự clone/checkout từ HEAD và kiểm tra checksum; không cần commit lịch sử ngoài shallow HEAD.

Tái lập riêng checkout sạch bằng một thư mục đích chưa tồn tại:

```powershell
git clone --shared --no-checkout . .tools/fix4-clean-repro
git -C .tools/fix4-clean-repro -c core.autocrlf=false checkout 30239cc18f07a963b97461e7191b8d11e2e70cbf -- .gitattributes data/ground_truth/events_v1.1_snapshot.jsonl data/ground_truth/checksums_v1.1.sha256
python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('.tools/fix4-clean-repro/data/ground_truth/events_v1.1_snapshot.jsonl').read_bytes()).hexdigest())"
```

Tôi không commit, push, merge, tạo PR, xóa dữ liệu, huấn luyện, chọn lại ngưỡng hoặc xác minh Ground Truth thay con người.
