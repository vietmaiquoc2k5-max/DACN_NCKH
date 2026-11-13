# Báo cáo phục hồi dataset và Single-Reviewer Research Mode — 13/11/2026

BRANCH: `fix-4`

BASE SHA: `b6a5d08644a95c9554b09a207a33ea2cac0a96eb` (HEAD được kiểm tra trước và sau triển khai).

WORKTREE STATUS: có thay đổi chưa commit; không commit/push/merge/xóa/di chuyển nguồn.
Trước tác vụ đã có sửa `app.py`, báo cáo RTSP, `single_reviewer.py`, bản backup app và
`studio_catalog_fix/`. Đã tích hợp bản vá một người bằng sửa có chọn lọc. Không chạy
patch catalog cũ. Báo cáo RTSP được giữ nguyên byte so với backup ban đầu, kể cả
trailing whitespace có sẵn; `git diff --check` cho phạm vi sửa code/docs Studio PASS.

FILES INSPECTED:

- Các service Studio: app, integrity, repository, annotation, adjudication, export,
  single_reviewer, video, models, config, admin; README/EXPORT_SCHEMA và patch cục bộ.
- Manifest âm tính, video_dataset_list, events hiện tại, snapshot/checksum frozen;
  schema/metadata CSV và SQLite private; các test Studio và critical acceptance.
- `src/video_audit.py`, `src/event_eval.py`, `src/week4_acceptance.py`,
  `src/inference_frame_evidence.py`, các script recovery/complete validation;
  run manifest/config/CSV/events/errors/log của run inference tham chiếu và báo cáo trước.

FILES MODIFIED:

- `ground_truth_studio/app.py`: tách nhóm, progress, single export, blind re-review;
  inspect/FFmpeg proxy chạy worker nền, progress PREPARING_SOURCE.
- `integrity_service.py`: discovery riêng VN_POS, source hash pins, overlay audit mới.
- `review_repository.py`: migration v2 có backup, UPSERT catalog, giữ lịch sử lượt review.
- `single_reviewer.py`: mở rộng bản người dùng, provenance/hash/round và export một người.
- `annotation_service.py`, `export_service.py`: hỗ trợ candidate mà không gán âm tính.
- `video_service.py`: cache vẫn kiểm tra expected SHA; không bỏ qua hash pin mới.
- `README.md`, `EXPORT_SCHEMA.md`, `tests/test_gt_studio_ui.py`,
  `tests/test_gt_studio_video.py`.

FILES CREATED:

- `ground_truth_studio/background_jobs.py`, `catalog_recovery.py`,
  `inference_reconciliation.py`, `provisional_evaluation.py`.
- `scripts/recover_week4_dataset.py`; `tests/test_gt_studio_recovery.py`,
  `tests/test_gt_studio_evaluation.py`; `data/video_catalog_local_v2.json`.
- Báo cáo này và output mới tại
  `runs/benchmarks/week4_dataset_recovery/run_20261113_v2/` (gọi là OUTPUT bên dưới).

DATASET:

| Hạng mục | Kết quả đo |
|---|---:|
| Physical videos trong datasets/negative_videos | 143 |
| VN_NEG candidate | 63; metadata 7,0403636989 giờ |
| VN_POS candidate | 80; metadata 4,9344593192 giờ |
| POS lịch sử giữ riêng | 24 |
| Nguồn có file trong toàn catalog | 167 |
| References lịch sử thiếu, vẫn giữ | 16 |
| Source SHA mismatch khi đối chiếu cuối | 0/167 |
| Full strict decode NORMAL_EOF | 3 |
| NEEDS_REVIEW | 58 |
| TRUNCATED_STREAM | 2 |
| Technical eligible hours, chỉ decode | 2,6180916667 |
| Human-reviewed/scientific eligible negative hours | 0 |

`VN_POS_50` và `VN_POS_51` có cùng SHA
`fe0ff123e94b10e72f60c83b0e8313e0376cd468022c226afa583a4ee42cce6c`.
Giữ cả hai ID và mapping duplicate; chưa chia train/validation/test. Ba archive
trong thư mục không được tính là video và không giải nén/di chuyển.

Audit fresh hoàn tất 63/63, exit 0, khoảng 942 giây; dùng nguyên logic strict
`src/video_audit.py`, không cache lịch sử, không tolerance. NORMAL_EOF gồm
VN_NEG_01/03/04. 58 NEEDS_REVIEW có bất đồng frame count giữa OpenCV/FFmpeg/metadata;
không tự kết luận các file này dùng được hay tự bỏ frame cuối. VN_NEG_10/48 có lỗi
NAL/packet decode, phân loại TRUNCATED_STREAM. Per-video JSON giữ số frame, timestamp,
FFmpeg stderr/exit/progress và lý do; `technical_audit_summary.json` ghi delta cụ thể.
Metadata giờ video không được dùng làm mẫu số FAR.

Artifacts: `current_physical_catalog.json`, `missing_historical_sources.json`,
`catalog_reconciliation_report.json`, `video_catalog_local_v2.json` (capture lúc
discovery), `video_catalog_post_audit.json` (bản dẫn xuất sau audit),
`technical_audit_63.json`, `audit_records/`, `technical_audit_summary.json`.
Scope nguồn được khóa bằng danh sách ID/path/SHA tại `scope_identity`; evaluator
tạo scope riêng gồm hash, review ID và interval thực sự đủ bằng chứng.

STUDIO:

- Catalog/dropdown: 63 negative, 80 positive candidate, 24 positive lịch sử; 16 thiếu
  bị ẩn khỏi dropdown, không xóa khỏi DB/manifest. Overview tách kỹ thuật/người duyệt/khoa học.
- Single Reviewer: một annotator có thể Save Draft, Submit, resume sau khởi tạo lại,
  export và blind re-review với draft trắng. Bản SUBMITTED cũ bất biến. Test fixture
  có đúng một account; không tạo account/signoff thay người trong DB thật.
- Export: bundle mới chứa review records, GT candidate, negative decisions,
  source/timestamp/reviewer provenance, hash manifest và intra-annotator comparison.
  Trạng thái SINGLE_HUMAN_REVIEWED_PROVISIONAL; cross_verified=false.
- Production: 1 user, 6 DRAFT, 0 SUBMITTED, 0 adjudication, 10 audit records như trước.
  Vì HUMAN REVIEW PENDING, chưa có candidate production để xuất. Không lấy fixture
  hay draft làm nhãn thật; `ground_truth_candidate_status.json` ghi rõ giới hạn này.
- Migration: user_version 0→2; backup nhất quán trước migration, test idempotent.
  `migration_receipt.json` và `integrity_receipt.json` xác nhận user/review/head/
  adjudication/audit values không đổi. Năm CSV giữ nguyên SHA; snapshot frozen giữ SHA
  `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3`.
- 11 historical cross_verified claims vẫn UNVERIFIED trong báo cáo tái xác thực;
  thiếu hồ sơ độc lập gốc, không tự nâng lên REAUTHENTICATED.
- Screenshot Chrome thật: `studio_catalog_overview.png`, `studio_single_reviewer.png`,
  `browser_evidence.json`. Dùng DB fixture riêng `.tools/gt_studio_ui_smoke`, không phải
  bằng chứng human review. Server/browser do tác vụ mở đã được dừng.

INFERENCE:

- Run đối chiếu: `run_20261111_233035_neg_full`.
- Reuse verified: 0; Unverified missing evidence: 63; Blocked missing input: 16.
- Rerun required do mâu thuẫn đã chứng minh: 0; Rerun completed: 0; new inference NOT_RUN.
- Checkpoint local khớp SHA claim
  `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`.
  Catalog lịch sử có SHA khớp nguồn hiện tại, nhưng manifest digest run không xác
  thực được bằng manifest hiện tại. Đây chưa phải binding chứng minh từng input run.
- Thiếu runtime imgsz/preprocessing và sampled-frame index/source PTS, completion
  binding; CSV aggregate không thay thế timeline. Script hiện tại cho thấy imgsz416
  và stride `max(1, round(native_fps/5))`, thời gian frame_index/FPS; quan sát mã nguồn
  không chứng minh config lúc chạy. 5 FPS sampled không phải full native-frame inference.
- Thiếu metadata không chứng minh model chạy sai. Chưa chạy inference mới vì chưa
  có GT submit, preflight chỉ 3 file qua gate; cần giải quyết nguồn và chuẩn bị run
  instrumented riêng. Không train/fine-tune/chọn threshold từ test.
- Chi tiết từng video/config/checkpoint/count/error/completion và quyết định:
  `inference_reconciliation.json`.

TESTS:

| Lệnh/lượt chạy | Kết quả | Exit | Log trong OUTPUT |
|---|---|---:|---|
| unittest discover -s tests -p test_gt_studio_*.py -v | 38/38 PASS | 0 | studio_final.log |
| unittest discover -s tests -p test_*.py -v, lượt đầu | 249 chạy, 248 thành công, 1 error | 1 | unittest_full.log |
| unittest discover -s tests -p test_dynamic_camera_selection.py -v, retry | 25/25 PASS | 0 | camera_retry.log |
| unittest discover -s tests -p test_*.py -v, retry toàn bộ | 249/249 PASS | 0 | unittest_full_retry.log |
| pytest -q tests/test_gt_studio_recovery.py tests/test_gt_studio_evaluation.py tests/test_week4_critical_acceptance.py | 21 PASS, 16 subtests PASS | 0 | pytest_targeted.log |
| pytest -q tests/test_gt_studio_video.py tests/test_gt_studio_evaluation.py, guard cuối | 11 PASS, 1 skip | 0 | pytest_final_guards.log |

Các lệnh unittest/pytest có tiền tố `python -m`; portable FFmpeg được thêm PATH
trong bộ chạy chính. Guard bổ sung không thêm PATH nên skip VFR fixture ở lượt đó;
VFR integration đã chạy PASS trong unittest có FFmpeg. Không cộng các lượt thành
số test độc lập. Lưu command, count, exit và log ở `test_results.json`; môi trường
Windows/Python3.11.9, Streamlit1.66.1, OpenCV5.0.0.93, pytest9.2.0 trong environment.json.
Chưa chạy Linux/macOS, không tuyên bố cross-platform PASS cho bản sửa này.

Error đầu: test_13_dual_and_hexa_camera_selection, Camera shutdown timed out.
Ảnh hưởng: lượt full đầu chưa PASS. Stop giữ thread handle an toàn theo code hiện có.
Đã dừng audit/Chrome rồi chạy lại nhóm và full suite, đều PASS. Chưa đủ bằng chứng
kết luận tải đồng thời là nguyên nhân gốc; không sửa hoặc nới timeout camera để che lỗi.
Giữ log lỗi đầu, coi đây là rủi ro không ổn định cần theo dõi.

METRICS:

- Mode: PROVISIONAL_SINGLE_REVIEWER; status PARTIAL_INSUFFICIENT_VERIFIED_EXPOSURE.
- Event Recall/Precision/F1/Detection Delay: null cho cả A/B.
- Valid FAR/hour và Poisson CI95%: null cho cả A/B; eligible negative hours: 0.
- Thiếu evidence: human SUBMITTED GT, nguồn âm tính đủ điều kiện ≥5 giờ,
  inference full timeline cùng hash/interval và independent GT/validation lock.
- Evaluator API chỉ mở metric khi đủ evidence; dùng cùng scope A/B, gộp episode
  video/camera/class gap2s và Garwood Poisson95CI. CLI với run lịch sử giữ metric null.
  Không có frontend nhập metrics hay tự gán acceptance PASS.

STATUS:

- Engineering: PASS cho phạm vi triển khai và kiểm thử trên Windows; giữ lưu ý
  timeout camera ở lượt đầu và hạn chế evidence inference. Không bao gồm nghiệm thu model.
- Scientific Week4: PARTIAL. Operating Point PROVISIONAL. Không có official acceptance.

REMAINING BLOCKERS:

- HUMAN REVIEW PENDING; chưa thể xuất GT candidate production từ sáu draft.
- 58 nguồn cần hòa giải frame count, 2 truncated; 16 nguồn lịch sử thiếu.
- 63 inference records chưa có provenance đầy đủ để tái sử dụng VERIFIED.
- Thiếu independent GT, ≥5h âm tính hợp lệ và operating-point validation lock.

NEXT ACTIONS FOR USER:

1. Chạy `python -m streamlit run ground_truth_studio/app.py --server.address 127.0.0.1`;
   dùng tài khoản thật hiện có, resume draft, xem nguồn, đánh dấu và Submit.
2. Vào Single Reviewer Export để xuất bundle riêng. Không tạo tài khoản thứ hai
   cho cùng người. Blind re-review chỉ kiểm tra độ nhất quán cá nhân.
3. Hòa giải kỹ thuật bằng nguồn/provenance, phục hồi bản hợp lệ riêng khi có thể;
   giữ nguyên nguồn cũ và tham chiếu lịch sử. Không tự chuyển NEEDS_REVIEW thành NORMAL_EOF.
4. Sau preflight/GT, chạy instrumented inference có index/PTS/config/run binding
   trên scope hợp lệ, dùng cùng exposure A/B và ngưỡng khóa từ validation.
5. Chỉ xét nghiệm thu Week4 sau đủ independent GT, ≥5 giờ và metrics/CI hợp lệ.

README và EXPORT_SCHEMA có lệnh catalog/audit/reconcile/evaluate, schema export,
backup/rollback và giới hạn evaluator. Tất cả artifacts mới ở OUTPUT riêng, không
ghi đè raw benchmark lịch sử.
