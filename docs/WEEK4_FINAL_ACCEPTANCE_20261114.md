# Nghiệm thu sản phẩm Tuần 4 — 14/11/2026

BRANCH: `fix-4`. HEAD thực tế: `730342623b5946f0eb9d25241eb4b8fc20488e79`. Worktree sạch trước tác vụ; hiện có code, tests, metadata và báo cáo mới, chưa commit/push. Không sửa video, checkpoint, export submitted, private SQLite hoặc frozen GT. Đây là phụ lục mới; [báo cáo thực nghiệm lịch sử](WEEK4_FINAL_EXPERIMENTAL_REPORT.md) và run cũ giữ nguyên.

## Kết luận theo từng phạm vi

| Phạm vi | Trạng thái | Căn cứ và giới hạn |
|---|---|---|
| WEEK4_DELIVERABLES_STATUS | **ACCEPTED** | Các thành phần phần mềm, bảng đo và artifact chọn cấu hình theo đề cương đã chạy kiểm chứng; chấp nhận OP ở mức bàn giao provisional |
| WEEK4_EXPERIMENTAL_STATUS | **VERIFIED_EXPLORATORY** | 71 nguồn, 107.805 mẫu 5 FPS, SHA/PTS/coverage, 36 cấu hình và A/B được xác thực lại |
| WEEK4_INDEPENDENT_GENERALIZATION_STATUS | **PARTIAL** | Thiếu bằng chứng test/scene/train-corpus độc lập; không có test-not-yet-used declaration hợp lệ |
| WEEK4_SCIENTIFIC_STATUS | **PARTIAL** | Không nâng nghiệm thu khoa học tổng thể thành PASS |
| Operating Point | **PROVISIONAL** | Không tạo lock chính thức hoặc chọn lại theo holdout |

## Human GT mới: kiểm chứng, không ghi đè

Bundle nguồn local: `D:\workspace\week4_private_human_review\exports\single_reviewer_9a3ee4c140464fdab2821aa77c468f4d`. SHA256 manifest checksum: `6a1137feec288b356b1a9b68eb1882a9dda091bd3ca527187601d6137ad401f5`. Sáu payload (bảy file kể cả manifest) khớp SHA; **166/166 submitted records** khớp DB snapshot chỉ đọc và checksum nguồn thực: 24 POS lịch sử + 63 VN_NEG + 79 VN_POS. Bundle không chứa draft; DB có 6 draft, riêng VN_POS có 3, đều không đưa vào scope.

Protocol: **JOINT_HUMAN_CONSENSUS**, theo xác nhận của ngài trong phiên làm việc. Export lưu một account; không suy ra hai lượt độc lập từ account hay đổi submitted bytes. `cross_verified=false`; không cần người B hoặc adjudicator thứ ba để bàn giao sản phẩm này.

| Chỉ số VN_POS | Số thực tế |
|---|---:|
| Submitted / kỹ thuật NORMAL_EOF | 79 / 79 |
| Nguồn SHA duy nhất trong 79 | 78 |
| Video có FIRE / SMOKE / cả hai | 67 / 72 / 64 |
| Không có event / có uncertainty | 4 / 0 |
| GT events gốc | 286 = FIRE 138 + SMOKE 148 |
| Thời lượng 79 alias | 17436.715980556s = 4.843532217h |
| Thời lượng tính một lần/source SHA | 17347.752105556s = 4.818820029h |
| Scope positive-event bảo thủ | **70 video / 70 SHA / 242 events** = FIRE 118 + SMOKE 124 |

Chín submission không đưa vào scope positive-event: VN_POS_24/60/61/71 không có event (vẫn là submission hợp lệ, không tự đổi thành NEG/FAR); VN_POS_50/51 cùng source SHA nhưng khác onset/offset (quarantine cả hai, không chọn/chép/trung bình nhãn); VN_POS_06/23/78 có ba cặp interval cùng lớp chồng lấn cần xác nhận quy ước sự kiện. Đây là vấn đề định nghĩa/boundary cục bộ, không phải source PTS lỗi. VN_POS_28 không có submission, giữ loại do TRUNCATED_STREAM. Tổng cộng 10 mục ngoài scope trong catalog 80, gồm 9 đã submit và một thiếu/truncated.

Chỉ cần xử lý các boundary/alias được chỉ ra trong [localized_gt_review_issues.json](../runs/benchmarks/week4_acceptance_closure/run_20261114_73034262/localized_gt_review_issues.json); không yêu cầu xem/gán lại toàn bộ 79. Bản gốc không bị chỉnh sửa.

Artifacts nghiên cứu mới ở [run closure](../runs/benchmarks/week4_acceptance_closure/run_20261114_73034262/): `week4_gt_bundle_audit.json`, `vn_pos_gt_events.jsonl`, `vn_pos_gt_scope.csv`, `vn_pos_gt_exclusions.csv`, `vn_pos_gt_provenance.json`, `week4_gt_lineage.json`. Không chứa danh tính, person/review/event UUID riêng hoặc private DB.

Đã re-audit bằng code cuối có guard conflict uncertainty/decision giữa các alias: **năm artifacts GT dẫn xuất khớp bytes hoàn toàn**, scope vẫn 70/242; [gt_current_code_recheck.json](../runs/benchmarks/week4_acceptance_closure/run_20261114_73034262/gt_current_code_recheck.json) giữ hash/code binding. Bản re-audit đọc DB chỉ đọc, không tạo submission hoặc sửa nhãn.

## Split/scene sau annotation

Giữ nguyên reservation **27 TRAIN_CANDIDATE / 11 VALIDATION_CANDIDATE / 13 INDEPENDENT_TEST_CANDIDATE / 29 BLOCKED_UNASSIGNED**. Manifest canonical SHA `24e3064daefd0d108da179caeeb35223966daec586321a6edd401f97c6513d08`; raw JSON SHA `8acd4311db47fe7e100a54e4503be12746fc82b7742f8e7ab9775baddc544e37`. Hai SHA có ý nghĩa khác nhau, không thay lẫn nhau. Không dùng nhãn hoặc model performance để phân lại split.

Trong 70 video positive-event được nhận: **26 train candidate / 9 validation candidate / 11 test candidate / 24 blocked**. Đây là event GT và reservation, chưa là dataset bbox sẵn sàng. [Manifest sau annotation](../runs/benchmarks/week4_acceptance_closure/run_20261114_73034262/vn_pos_postannotation_reservations.json) dẫn về prior SHA, cập nhật completion dưới phiên bản mới, không sửa prior manifest. Scene/camera chưa có metadata xác nhận; giữ danh sách [scene cần review](../data/vn_pos_scene_review_required.csv). Các test candidate tiếp tục tách khỏi training; không gọi là independent test hợp lệ.

## Kiểm chứng thực nghiệm đã có

Run lịch sử: `runs/benchmarks/week4_final_scientific/run_20261113_consensus_v1`. **292/292 artifacts** xác thực trước/sau; ba file code sealed khớp runtime bytes; checkpoint SHA `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`, class 0=SMOKE, 1=FIRE. Nguồn/hash/frame JSONL/completion/config binding/source PTS integer và rational đều được kiểm lại. **71/71 video, 107.805/107.805 mẫu 5 FPS**, không chạy lại NN. Mẫu 5 FPS không được mô tả là inference mọi native frame.

Đã replay trực tiếp evaluator/verifier trên raw inference đã xác thực: **36/36 cấu hình validation**, so sánh toàn bộ counts/metrics với kết quả ghi lại, rồi replay holdout bằng cấu hình provisional đã chọn. Receipt: [existing_week4_evidence_verification.json](../runs/benchmarks/week4_acceptance_closure/run_20261114_73034262/existing_week4_evidence_verification.json). Mọi artifact run cũ giữ bytes. Log local `.tools/week45_closure_73034262/existing_evidence.log`.

Exposure âm tính hợp lệ toàn pool: **47 video / 5.749655097h**. Phân scope đo: validation 24 NEG / 0.524387273h; exploratory holdout 23 NEG / **5.225267824h**, cùng 12 POS / 14 GT events. Không lấy positive video hay bốn VN_POS không có event để tăng mẫu số FAR.

| Metric exploratory holdout | Baseline A | Temporal B |
|---|---:|---:|
| Event Recall (instance) | 13/14 = 92.86% | 9/14 = 64.29% |
| Event F1 | 0.065163 | 0.153846 |
| False alarms trên NEG | 362 | 79 |
| FAR/hour | 69.278746 | 15.118842 |
| Poisson CI95% FAR/hour | [62.3251, 76.7961] | [11.9697, 18.8426] |
| Mean matched delay (s) | 1.659197, 13 matches | 1.556491, 9 matches |
| Delay trên cùng 9 GT events (s) | 0.285370 | 1.556491 |

FAR giảm khoảng 78.18%, đồng thời event recall giảm 28.57 điểm phần trăm. Hai mean delay toàn pipeline có tập matched khác nhau; không dùng chúng để kết luận B nhanh hơn. Delay dùng source time, chưa gồm inference/network/notification latency. CI là exact Poisson conditional trên counts/exposure; không chứng minh scene independence hoặc bao quát sai số human GT.

Validation có **34/36 điểm đạt ràng buộc video-recall gốc**, nhưng **0/36** đạt screen instance-recall khắt khe hơn. Không đổi hai định nghĩa để tạo PASS. Cấu hình đã chọn và không retune: conf=0.50, K=4, N=10, variation=0.02, cooldown=12s, merge gap=2s; config SHA `c34a3ba639bd8f31fb40a819712d5afc4e3a04e2c04c1d3946d4c2091850e18c`. OP còn PROVISIONAL vì independence/history chưa đủ, không phải vì không có candidate thỏa ràng buộc gốc.

Không mở rộng inference sang VN_POS/test candidates trong tác vụ này. GT mới không thay bảng lịch sử; các metrics mới, nếu chạy sau, phải có run/scope riêng.

## Checklist nghiệm thu sản phẩm

| Sản phẩm | IMPLEMENTED | TESTED | VERIFIED | ACCEPTED |
|---|---|---|---|---|
| ByteTrack | Có | PASS | Có | ACCEPTED |
| Temporal k/N | Có | PASS | Có | ACCEPTED |
| Variation | Có | PASS | Có | ACCEPTED |
| Exclusion zones | Có | PASS | Có | ACCEPTED |
| temporal_verifier.py | Có | PASS | Có | ACCEPTED |
| event_eval.py | Có | PASS | Có | ACCEPTED |
| Parameter sweep | Có | PASS | Có | ACCEPTED |
| A/B comparison | Có | PASS | Có | ACCEPTED |
| FAR/hour + CI95% | Có | PASS | Có | ACCEPTED |
| Event Recall/F1/Delay | Có | PASS | Có | ACCEPTED |
| Operating Point | Có | PASS | Có | ACCEPTED¹ |

¹ Chỉ nghiệm thu việc chọn, lưu và bàn giao cấu hình **PROVISIONAL**; chưa nghiệm thu lock deployment hoặc independent scientific acceptance.

- **ByteTrack:** src/camera_manager.py:624; tests/test_bytetrack_adapter.py; real tracked frame JSONL. Phạm vi: Pipeline association.
- **Temporal k/N:** tests/test_temporal_verifier_bytetrack.py:68; 36 replayed validation candidates. Phạm vi: Per-track/class window.
- **Variation:** src/temporal_features.py; tests/test_temporal_features.py; threshold 0.02 in measured B. Phạm vi: Dynamic feature verification.
- **Exclusion zones:** src/camera_manager.py:1493; tests/test_temporal_verifier_bytetrack.py:156. Phạm vi: Normalized rectangles only; integration accepted, no measured ROI efficacy.
- **temporal_verifier.py:** Actual B alerts replayed against recorded results; targeted tests. Phạm vi: Temporal alerts, not frame-only detections.
- **event_eval.py:** 36 validation comparisons and exploratory holdout replay; tests/test_week4_event_eval.py. Phạm vi: One-to-one class/video/camera/onset event matching.
- **Parameter sweep:** 36/36 actual configurations reauthenticated. Phạm vi: 34 feasible under original video-recall constraint; 0 under stricter instance screen.
- **A/B comparison:** Historical measured A/B and current exact replay. Phạm vi: Exploratory scope, not independent test.
- **FAR/hour + CI95%:** Verified negative exposure/counts and tests/test_poisson_ci.py. Phạm vi: Exact Poisson CI conditional on count/exposure assumptions.
- **Event Recall/F1/Delay:** Verified GT matches and paired-delay caveat. Phạm vi: Loss of event recall explicitly reported.
- **Operating Point:** Historical provisional selection/config hash and current validation replay. Phạm vi: Selection artifact delivered as PROVISIONAL; scientific/deployment lock not accepted.

ROI là normalized rectangle, có pipeline/update/test thật. Benchmark measured có zones=[]; chưa đo ROI ablation và không có bằng chứng polygon. ByteTrack → temporal features → verifier được gọi tại `src/camera_manager.py:624/632/643`; raw frame detections và triggered event alerts được phân biệt trong A/B.

## Tests và tái lập

Targeted pytest: **119 passed, 58 subtests passed**, exit **0**, thời gian wall **22.441s**. Full unittest: **350 tests, OK**, exit **0**, thời gian framework **148.664s**, wall **152.649s**. Không failed/skipped. Các fixture stream hỏng có thể in cảnh báo codec; không phải lỗi nguồn thực nghiệm. Report RTSP cũ được khôi phục đúng bytes trong `finally`.

Lệnh thực tế, timestamps, count, SHA log: [test_results.json](../runs/benchmarks/week4_acceptance_closure/run_20261114_73034262/test_results.json). Raw logs local: `.tools/week45_closure_73034262/targeted_final.log`, `.tools/week45_closure_73034262/full_unittest_final.log`; không commit raw logs.

```powershell
New-Item -ItemType Directory -Force .tools/week45_closure_73034262/tmp | Out-Null
$env:TEMP = (Resolve-Path .tools/week45_closure_73034262/tmp).Path
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PATH = "$((Resolve-Path .tools/ffmpeg).Path);$env:PATH"
python -m pytest -q tests/test_week45_acceptance.py tests/test_domain_adaptation.py tests/test_preannotation.py tests/test_preannotation_studio.py tests/test_scientific_inference.py tests/test_bytetrack_adapter.py tests/test_temporal_verifier_bytetrack.py tests/test_temporal_features.py tests/test_week4_event_eval.py tests/test_poisson_ci.py tests/test_fix4_evidence_gates.py
python -m unittest discover -s tests -p test_*.py -v
# Tạo output mới, không chạy evaluator main vào benchmark cũ.
python scripts/audit_week45_human_gt.py --bundle D:\workspace\week4_private_human_review\exports\single_reviewer_9a3ee4c140464fdab2821aa77c468f4d --output-dir runs/benchmarks/week4_acceptance_closure/NEW_GT_RUN --gt-protocol JOINT_HUMAN_CONSENSUS
python scripts/verify_week4_closure_evidence.py --historical-run runs/benchmarks/week4_final_scientific/run_20261113_consensus_v1 --output-dir runs/benchmarks/week4_acceptance_closure/NEW_VERIFY_RUN
```

`unittest` hiện có test tạo report RTSP lịch sử; wrapper local giữ bytes trước và phục hồi trong `finally`, đã kiểm đúng bytes. Khi chạy lại độc lập cũng phải bảo toàn file này. Audit GT chỉ mở SQLite `mode=ro` + `query_only=ON`; sandbox có thể cần quyền đọc DB/WAL. Không mở repository DB bằng constructor có initialization ghi. Không suy ra đã chạy Linux/CI từ các test Windows local này.

## SCIENTIFIC_LIMITATIONS và công việc còn lại

- Historical exposure/test contamination chưa đủ lịch sử quyết định để gọi holdout/test candidates là test mới; không dùng holdout để chọn lại ngưỡng.
- GT consensus do hai người cùng xem theo xác nhận của ngài, chưa là independent cross verification; không bịa inter-annotator agreement.
- Thiếu metadata camera/scene và content manifest corpus train gốc; không khẳng định có hoặc không có training contamination.
- OP provisional; event recall suy giảm; ROI ablation chưa đo; camera IDs logic không chứng minh camera vật lý độc lập.
- Muốn nghiệm thu generalization: hoàn thiện scene/training provenance, giải quyết 5 video có vấn đề boundary/alias, thiết lập holdout mới chưa dùng và lock từ validation phù hợp trước test. Không sửa GT theo lỗi/miss của detector.

Bắt đầu Tuần 5 theo [báo cáo thực thi và blocker](WEEK5_DOMAIN_ADAPTATION_20261114.md); không coi ba file config hoặc các lần dừng ở data gate là thí nghiệm huấn luyện đã hoàn thành.
