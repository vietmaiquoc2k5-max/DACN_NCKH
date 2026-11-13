# Tuần 4 — báo cáo thực nghiệm cuối

BRANCH: `fix-4`. HEAD: `973970918e97cad6ac91d180fb455959601fae67`. WORKTREE: sạch trước tác vụ; thay đổi hiện tại là runner, evaluator, kiểm chứng, tests và báo cáo/artifacts mới, chưa commit/push. Báo cáo này bổ sung bằng chứng chạy mới, giữ nguyên các báo cáo lịch sử.

**FINAL: PARTIAL.** Inference, validation sweep và đo A/B đã thực sự hoàn tất; có cấu hình đạt ràng buộc validation gốc theo video recall nhưng chưa có independent test hợp lệ. Không tạo `operating_point_locked.json` hoặc khai báo test chưa từng được dùng.

## GT

- Protocol: `JOINT_HUMAN_CONSENSUS`; nguồn export `single_reviewer_1d11dd80c2c7427a8c00ea6a2803382b`. Ngài xác nhận hai người cùng xem/thống nhất; đây không phải hai annotation độc lập. Không cần tạo người B/adjudicator để chạy thí nghiệm này. Luồng independent review hiện có được giữ nguyên.
- Đã kiểm chứng thực tế 87 submitted records (24 POS historical + 63 VN_NEG), đối chiếu DB ở chế độ chỉ đọc; ba draft khác không tham gia. Export có sáu payload trong manifest checksum, bảy tệp kể cả manifest; tất cả payload và hash 87 nguồn khớp. Không công bố danh tính, review IDs hay DB riêng.
- 63 VN_NEG có 48 `VERIFIED_NEGATIVE`, 14 `CONTAINS_SMOKE`, một `UNCERTAIN` (VN_NEG_48). Loại 14 video có khói; VN_NEG_48 uncertain/truncated và VN_NEG_10 truncated. Tổng cộng 16 video bị loại; còn **47 âm tính, 5.7496550970 giờ**. Pool kỹ thuật 61 video/6.8223424028 giờ không phải mẫu số FAR sau lọc semantic.
- Chấp nhận 24 POS; 35 GT FIRE/SMOKE events có hash nguồn, frame và onset/end kiểm theo source PTS, nằm trong video, không có event đồng nhất trùng lặp. Không thay nhãn bằng frozen GT cũ.
- VN_NEG_01/04/20/31 có duration export từ frame/FPS. Chỉ nhận nhờ full-view submission trên đúng nguồn, không có events/uncertain; exposure giới hạn bằng min(source PTS duration, human-reviewed duration, technical duration). Inference mới dùng PTS nguyên/rational cho cả bốn nguồn, không suy timestamp từ FPS.
- GT này được dùng theo quyết định của ngài; mức đúng semantic vẫn phụ thuộc review của người thật. Không suy ra gán chéo 10% frame Việt Nam Tuần 2 đã hoàn tất: **chưa có bằng chứng**. Không khẳng định near-duplicate/scene độc lập khi thiếu metadata.
- Evidence: `gt_evidence_audit.json`, `negative_gt_scope.csv`, `positive_gt_events.jsonl`, `gt_exclusion_report.csv`, `ground_truth_provenance.json`, `submission_db_binding.json` trong run. Frozen GT giữ SHA `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3`; 547 audit artifacts giữ checksum.

## Checkpoint

`runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`; SHA256 `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`. YOLO DetectionModel, class 0=SMOKE, 1=FIRE; `model.eval()` và `predict` inference. Thiết bị thực tế CPU, Torch 2.15.0+cpu, Ultralytics 8.4.198. CUDA fallback được hỗ trợ; lần chạy này không sử dụng CUDA. Real-source smoke test POS_01 thành công (1 detection). Không retrain hoặc sửa checkpoint.

Input 416, square letterbox (`rect=False`), FP32, NMS IoU 0.7, max_det 300, lưu raw từ conf 0.10; A/B cùng raw detector. ByteTrack có ba stream thật theo conf 0.25/0.35/0.50, high/new bằng conf, low=0.10, buffer=15, per-class; feature history=12, state mới cho từng video. Low-score association hỗ trợ tracking B; chỉ boxes đạt conf ứng viên mới tạo hit temporal. Các per-class/association rules thuộc khác biệt phương pháp B, không phải checkpoint khác.

## Split và leakage

Split theo video/SHA, không theo frame. POS số lẻ giữ validation lịch sử; POS chẵn vào exploratory holdout. Negative sort tăng duration rồi SHA, lấy tới >=0.5h validation nhưng giữ >=5h holdout nếu pool cho phép; quy tắc metadata được khai báo trước model inference mới. Source SHA trùng chính xác: 0; không chứng minh được near-duplicate khác encode/cảnh hoặc camera vật lý khác nhau.

| Scope | POS videos | GT events | NEG videos | NEG hours |
|---|---:|---:|---:|---:|
| Validation | 12 | 21 | 24 | 0.5243872730 |
| Exploratory holdout | 12 | 14 | 23 | 5.2252678241 |

`INDEPENDENT_TEST_CONTAMINATION`: historical `validation_recall_ablation.json` và `operating_point_reconciliation.json` đã công bố kết quả trên POS; negative full-run cũ đã công bố FAR. Không xác thực được đầy đủ lịch sử mọi quyết định tuning; bằng chứng hiện tại không thiết lập test mới độc lập. `BENCH_*` là định danh logic, không chứng minh scene/camera khác nhau. Training args khai báo `data_dfire.yaml`, nhưng không có content-hash manifest corpus train gốc để xác minh train/holdout overlap. Không kết luận chắc chắn có/không training contamination.

Split hash `7737b7d5a47a2c3f9ed0b7aa1e05e5f478a168edeb0c2cbd7d8758d851a827ee`; validation scope hash `1e67907b755c0168413db26054e80c13892cc5d9fced9624580edcd4323380ef`. `split_audit.json` ghi hash và nguồn bằng chứng lịch sử. Independent test valid: **false**.

## Inference

Run `run_20261113_consensus_v1` trong `runs/benchmarks/week4_final_scientific/`. **71/71 video, 107,805/107,805 mẫu thật, 798,583 native frames giải mã; hoàn tất.** Protocol `FULL_DURATION_SAMPLED_5FPS`, không gọi là native-frame inference. VFR lấy frame đầu tiên tại/sau mốc thời gian nguồn 0, 0.2, 0.4,...; bỏ các mốc đã vượt khi nguồn thưa, không lặp/interpolate frame. PTS tick và time_base rational được kiểm ở từng mẫu.

Frame JSONL chia theo video, binding run/video/camera/source/checkpoint/config; lưu raw detections và ba stream tracker/features. **86,035 mẫu thực sự không có detection** được lưu sau predict, không bù record giả. Hash nguồn trước/sau, SHA JSONL, native-count, exact sample-index coverage, PTS tick và giây, mapping lớp, track IDs/features hữu hạn đều được chạy kiểm lại. Positive 24/24 có strict full-stream FFmpeg EOF sạch; negative dùng audit đã xác thực. Resume chỉ nhận video completed có checksum/coverage; partial output được giữ và từ chối ghi đè.

Inference config hash `701402b5eb2bb5c2b26f7402cc3892f7be6819740e08e716cca5b0395c237a18`. `experiment_protocol_lock.json` đóng băng grid/constraints và SHA ba file implementation trước inference; postvalidator xác nhận chúng không đổi. `run_manifest.json` giữ snapshot `PREFLIGHT_COMPLETE`; `integrity_receipt.json`, `frame_evidence_verification.json`, `run_status.json` phản ánh hoàn tất, không sửa lịch sử preflight. Log có cảnh báo deprecation tham số `half`, không có ERROR/Traceback; cảnh báo được giữ nguyên. Delay dưới đây dùng source time, không gồm thời gian CPU decode/inference hay gửi thông báo.

## Validation sweep và Operating Point

Đã chạy thật **36/36** cấu hình: conf=[0.25,0.35,0.50], (K,N)=[(2,8),(3,8),(4,10)], variation=[0,0.02], cooldown=[5,12] giây; merge gap=2s cố định. ROI tắt trong benchmark vì không có vùng loại trừ được khai báo trước; không giả thêm policy để đủ 72 cấu hình plan cũ. N là số mẫu đã xử lý ở protocol 5FPS, không phải native frames.

Ràng buộc gốc trong `event_eval.grid_search` là **Video Event Recall** B >= A-0.05, FAR B giảm >=50% so với A, mean matched delay <=10s. **34/36 cấu hình đạt tiêu chí này.** Script mới đã khai báo trước một screen khắt khe hơn bằng **Event-instance Recall**; theo screen đó có **0/36** cấu hình đạt. Việc mô tả screen mới như hoàn toàn trùng tiêu chí legacy là lỗi diễn giải, được phát hiện và ghi `validation_recall_definition_audit.json/csv` trên dữ liệu validation trước khi đo holdout, không sửa sealed code/grid/GT/preseal. Cột `feasible` trong `parameter_sweep.csv` chỉ screen từng event; không dùng nó kết luận tiêu chí gốc thất bại.

Mục tiêu max Event F1, rồi FAR thấp, delay thấp và config SHA cho **cùng cấu hình** ở hai cách chọn (max-F1 trong 34 điểm đạt tiêu chí video và max-F1 chẩn đoán toàn grid): conf=0.50, K=4, N=10, variation=0.02, cooldown=12s, gap=2s; SHA `c34a3ba639bd8f31fb40a819712d5afc4e3a04e2c04c1d3946d4c2091850e18c`. Không chọn lại theo holdout.

| Metric | Baseline A | Temporal B |
|---|---:|---:|
| Event Recall (instance) | 0.6667 | 0.5714 |
| Video event recall | 0.9167 | 0.9167 |
| Event Precision | 0.1538 | 0.2609 |
| Event F1 | 0.2500 | 0.3582 |
| Mean matched delay (s) | 0.5645 | 2.1950 |
| Median/P95 delay (s; higher order statistic) | 0.0000 / 4.4000 | 1.3680 / 9.4000 |
| Negative exposure (h) | 0.5244 | 0.5244 |
| FAR/hour | 131.5821 | 19.0699 |
| Poisson CI95% FAR/hour | [102.3788, 166.5256] | [9.1447, 35.0702] |
| Matched / GT events | 14 / 21 | 12 / 21 |
| Negative false alarm events | 69 | 10 |
| Unmatched positive alerts | 8 | 24 |
| Missed GT events | 7 | 9 |

Tại điểm này FAR giảm 85.51%, video recall giữ 91.67%, mean delay <=10s: đạt các ràng buộc gốc. Tuy vậy instance recall giảm 9.52 điểm phần trăm, là đánh đổi phải công bố; không nới target hoặc đổi config để che mất event. Validation negative chỉ 0.5244h, CI rộng. `validation_selection_preseal.json` lúc 2026-11-13T16:53:06Z đã chọn cấu hình trước holdout metrics; replay cuối xác nhận đủ 36 kết quả và cấu hình trùng nguyên vẹn. Audit định nghĩa recall ghi lúc 2026-11-13T17:11:01Z, cũng trước holdout metrics.

**OPERATING_POINT=PROVISIONAL.** `operating_point_provisional.json` được ghi trước đo holdout, hash kiểm giữ nguyên sau đo. Trường selection của sealed script phản ánh screen instance khắt khe hơn; đọc cùng audit recall để thấy 34 candidate đạt tiêu chí video gốc. Không có official lock hoặc test-not-yet-used declaration đúng sự thật vì dữ liệu có prior exposure; không đổ thiếu lock cho việc không có cấu hình đạt tiêu chí gốc.

## A/B trên exploratory holdout

Phạm vi **không phải independent test**, kết quả dưới đây phục vụ chẩn đoán, không dùng chọn lại ngưỡng. A là raw frame alarms chuyển thành event episodes; B là các thời điểm `TemporalVerifier.triggered_alerts` thật, cũng merge cùng video/camera/class với gap<=2s. Cùng checkpoint, input, source frames, sampling, GT và negative exposure. Dùng trực tiếp `src/event_eval.evaluate`: one-to-one maximum cardinality/minimum cost theo video/camera/class, onset mặc định +/-10s; delay=max(0,pred onset-GT onset). Không dùng offset IoU để thay thế onset matching.

| Metric | Baseline A | Temporal B |
|---|---:|---:|
| Event Recall (instance) | 0.9286 | 0.6429 |
| Video event recall | 1.0000 | 0.7500 |
| Event Precision | 0.0338 | 0.0874 |
| Event F1 | 0.0652 | 0.1538 |
| Mean matched delay (s) | 1.6592 | 1.5565 |
| Median/P95 delay (s; higher order statistic) | 0.2000 / 9.0000 | 0.8008 / 5.7600 |
| Negative exposure (h) | 5.2253 | 5.2253 |
| FAR/hour | 69.2787 | 15.1188 |
| Poisson CI95% FAR/hour | [62.3251, 76.7961] | [11.9697, 18.8426] |
| Matched / GT events | 13 / 14 | 9 / 14 |
| Negative false alarm events | 362 | 79 |
| Unmatched positive alerts | 10 | 15 |
| Missed GT events | 1 | 5 |

Trên holdout thăm dò, B giảm FAR **78.18%** nhưng mất **28.57 điểm phần trăm event recall**, video recall giảm 100% xuống 75%; không đạt mức giảm recall <=5 điểm nếu áp tiêu chí gốc vào phạm vi này. B miss POS_02/SMOKE, POS_04/FIRE, POS_08/SMOKE, POS_14/SMOKE, POS_18/SMOKE; các onset/frame cụ thể nằm trong `missed_events.jsonl`. Không chọn lại ngưỡng từ kết quả này. Hai mean delay có quần thể match khác nhau (A=13, B=9), nên không kết luận B nhanh hơn: trên **9 GT events cả hai cùng match**, mean delay A=0.2854s, B=1.5565s. Delay B tăng khi đối chiếu cùng event.

Event Recall chính là matched GT event instances / toàn bộ GT events; video recall báo riêng. Precision tính mọi predicted event cả POS và NEG; FAR chỉ đếm false alarm episodes trên NEG chia đúng exposure NEG. Alert lặp sau cooldown trong event dài có thể trở thành positive unmatched episode; không gộp chúng vào negative FAR. Mean delay chỉ trên matched events, phải đọc cùng recall/missed count. Median là phần tử thứ n//2 và P95 phần tử floor(.95*n), không nội suy; số mẫu là matched count ở bảng, chưa là ước lượng tail ổn định. Poisson CI95% dùng exact count interval/exposure từ `event_eval.exact_poisson_ci`, giả định quá trình Poisson; các cảnh/alert tương quan có thể làm CI quá lạc quan.

CSV/JSON kết quả: `baseline_vs_temporal.csv`, `baseline_vs_temporal.json`; `comparison.csv` và `metrics.json` là bản tóm tắt nhỏ theo retention policy. `parameter_sweep.csv`, `validation_results.json` chứa 36 candidate thật. `per_video_results.csv`, `event_matching.jsonl`, `false_alarm_events.jsonl`, `missed_events.jsonl` lưu bằng chứng từng video/event. `false_alarm_events.jsonl` chỉ chứa NEG; mọi positive unmatched vẫn ở raw comparison JSON. `error_analysis.json` tóm tắt video nhiều FP và lớp GT bị miss; chưa gán nguyên nhân thị giác khi chưa xem clip lỗi. Biểu đồ thật: [FAR–Delay](../runs/benchmarks/week4_final_scientific/run_20261113_consensus_v1/far_delay_comparison.svg).

## Sản phẩm Tuần 4 và kiểm thử

| Sản phẩm | Bằng chứng và phạm vi |
|---|---|
| `src/temporal_verifier.py` | ByteTrack integration, k/N, variation, confirmation, cooldown/reset theo worker/source; benchmark replay dùng engine thật |
| `src/temporal_features.py` | Extractor đang dùng cho tracked boxes; khác legacy `temporal_feature_extractor.py` |
| Exclusion zones | Normalized rectangles, kiểm center-in-ROI; code/UI và tests có. Polygon chưa hỗ trợ; hiệu quả ROI trên benchmark này chưa đo vì zones=[] |
| `src/camera_manager.py` | Tracker.update → features.update → temporal.update; worker/source state riêng |
| `src/event_eval.py` | Event Recall/Precision/F1/Delay, matching và exact Poisson CI; được dùng trên inference thật |
| Bảng baseline A/B | Đã đo, đủ CSV/JSON/per-video/matching/FP/FN; kết quả exploratory, không scientific PASS |

Đã chạy `python -m pytest -q tests/test_scientific_inference.py tests/test_bytetrack_adapter.py tests/test_temporal_verifier_bytetrack.py tests/test_temporal_features.py tests/test_week4_event_eval.py tests/test_poisson_ci.py tests/test_fix4_evidence_gates.py`: **61 passed, 10 subtests passed**, exit 0. `python -m unittest discover -s tests -p test_*.py -v`: **292 tests, OK**, exit 0. Unit tests kiểm CFR/VFR, coverage có empty detections, thiếu/lặp, binding/checkpoint/PTS lỗi, guard independent test, zero-FP CI khác0 và exposure thiếu. Test fixture không là nguồn metrics thực nghiệm. Full-suite tạo lại một report RTSP cũ; bytes ban đầu đã được khôi phục, không để thay đổi ngoài phạm vi. Postvalidator mới chạy trên toàn bộ dữ liệu thật, exit 0; xem `verify.log`. `test_results.json` và raw logs giữ lệnh/count/exit thực.

## Bàn giao và tái lập

```powershell
# Run đã có: kiểm chứng mà không ghi đè frame outputs
python scripts/verify_week4_scientific_run.py --run-dir runs/benchmarks/week4_final_scientific/run_20261113_consensus_v1
# Resume chỉ kiểm hash/coverage, không infer lại video đã hoàn tất
python scripts/run_week4_final_scientific.py --phase infer --run-dir runs/benchmarks/week4_final_scientific/run_20261113_consensus_v1
# Thực nghiệm mới phải dùng thư mục mới; --joint-export là thư mục export riêng thật
python scripts/run_week4_final_scientific.py --phase prepare --run-dir runs/benchmarks/week4_final_scientific/NEW_RUN --joint-export D:\workspace\week4_private_human_review\exports\single_reviewer_1d11dd80c2c7427a8c00ea6a2803382b
# Tái lập cùng protocol/code: khai báo protocol trong thư mục mới trước inference
Copy-Item -LiteralPath runs/benchmarks/week4_final_scientific/run_20261113_consensus_v1/experiment_protocol_lock.json -Destination runs/benchmarks/week4_final_scientific/NEW_RUN/experiment_protocol_lock.json
python scripts/run_week4_final_scientific.py --phase infer --run-dir runs/benchmarks/week4_final_scientific/NEW_RUN
python scripts/verify_week4_scientific_run.py --run-dir runs/benchmarks/week4_final_scientific/NEW_RUN
python scripts/evaluate_week4_final_scientific.py --run-dir runs/benchmarks/week4_final_scientific/NEW_RUN
```

Evaluator từ chối overwrite OP đã đánh giá; không chạy lại main evaluator vào run cũ. Protocol/grid/code SHA của run này được lưu trước inference trong `experiment_protocol_lock.json` (không là official OP lock); khi tái lập nghiên cứu mới phải khai báo/niêm phong protocol trước inference/sweep. `artifact_sha256.json` kiểm kê run, `artifact_storage_manifest.json` ghi retention; raw JSONL/PTS/logs/plots giữ local theo `.gitignore`, không stage. Không commit/push/xóa/untrack, sửa original video/checkpoint/export/frozen GT hoặc rewrite history. 416 raw artifacts lịch sử còn chờ authorization cleanup được giữ nguyên.

## Phần còn thiếu để nghiệm thu

1. Phân tích miss bằng source clip và ablation tracker/kN/variation/cooldown trên validation, không thay GT vì detector hoặc tuning theo holdout. Cấu hình đã đạt ràng buộc gốc theo video recall; cần công bố và cân nhắc event-instance recall bị mất trước deployment lock, không thay định nghĩa recall âm thầm.
2. Tạo holdout mới chưa được dùng chọn model/ngưỡng, có >=5h NEG được người review và POS có onset, chứng minh disjoint video/content/scene, kèm training corpus provenance. Với pool hiện tại không thể tự tuyên bố test độc lập.
3. Khóa cấu hình từ validation khả thi cùng scope/GT/checkpoint/code/hash và test-not-yet-used declaration đúng sự thật, rồi mới đo independent test một lần. Không cần thêm adjudicator ngoài đề cương cho GT consensus hiện tại.

Trạng thái phần mềm và inference đã có bằng chứng chạy; nghiệm thu khoa học tổng thể **PARTIAL** vì các thiếu hụt trên. `targets.overall_pass` trong raw legacy evaluator chỉ kiểm exposure/delay, không thay thế scientific acceptance; kết luận chính thức đọc `summary.json`/`scientific_acceptance.json`.
