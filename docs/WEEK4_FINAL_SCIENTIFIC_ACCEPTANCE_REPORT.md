# Tuần 4 — kiểm định khoa học có điều kiện

BRANCH: fix-4. HEAD:887052a01c16595f274a704cf8ec72674c349cf7.
WORKTREE trước sạch. Không commit/push/untrack/delete dữ liệu trong tác vụ này.

## Yêu cầu và bằng chứng

| Sản phẩm | Trạng thái và source |
|---|---|
| ByteTrack | IMPLEMENTED; src/bytetrack_adapter.py, camera_manager.py:624 gọi update; per-camera worker/tracker; tests adapter/isolation |
| k/N, cooldown/reset | IMPLEMENTED; temporal_verifier.py, camera_manager.py:647; test verifier/dashboard/source reset |
| Variation | IMPLEMENTED; module đang dùng src/temporal_features.py (khác legacy temporal_feature_extractor.py); camera_manager.py:632; ablation thực chưa VERIFIED |
| Exclusion zones | IMPLEMENTED; normalized rectangles, center-in-ROI inclusive boundary; camera_manager.update_camera_roi lưu config và cập nhật worker; polygon chưa hỗ trợ |
| temporal_verifier.py | API update(tracked detections, source time, features), reset; không tạo module trùng |
| event_eval.py | IMPLEMENTED; camera/class-aware one-to-one onset matching, tối đa cardinality/min cost, tolerance10s default; delay=max(0,pred onset−GT onset); không phải yêu cầu offset IoU |
| Sweep | scripts/ablation_validation_recall.py đã tồn tại; lịch sử chưa đủ độc lập để ACCEPTED. Plan mới72 candidates, chưa chạy |
| Operating Point | BLOCKED_VALIDATION_EVIDENCE, PROVISIONAL; không tạo lock hash giả |
| A/B comparison | Null table; frame detections phải gộp thành episodes cùng merge gap để so event công bằng |
| Event Recall/Precision/F1/Delay | null; thiếu independent GT + split + inference binding |
| FAR/CI95% | null; pool kỹ thuật không tự trở thành scientific negative exposure |

## HUMAN GROUND TRUTH

Ngài xác định export single_reviewer_1d11dd80c2c7427a8c00ea6a2803382b là bộ hai người
cùng xem/thống nhất. Checksum toàn bundle và source hashes của87 records đã được
đo lại:24 POS historical+63 negative. Được tham chiếu JOINT_CONSENSUS_PRIOR trong
configs/research/joint_consensus_prior_reference.json; original export không sửa.
Không coi bundle single-reviewer hay87 submission này là hai lượt độc lập.

Backup SQLite nhất quán ở .tools/week4_final_887052a0/human_before.sqlite3. Hiện có
1 annotator_01,87 SUBMITTED và3 DRAFT. Không tạo tài khoản B/adjudicator giả; không
rollback về sáu draft của các tác vụ cũ. Không migration/schema change. Production
review/head/audit không bị sửa: UI chỉ tạo lượt mới khi người thật bấm nút.

IndependentReview.start bảo toàn previous session và backup trước khi đổi head,
tạo draft trắng với audit marker INDEPENDENT_REVIEW_ROUND_A/B. UI không điền nhãn
cũ; source được kiểm SHA và người review phải attest independent/full view trước
Submit. Single-reviewer lịch sử không được export như dual independent. Two-role
export chỉ mở khi hai submission có marker, hai person_id khác nhau và cùng SHA.
Ngay cả agreement vẫn INDEPENDENT_DUAL_REVIEW_PENDING_ADJUDICATION; không thay role
third adjudicator. Disagreement/identity/source/review IDs/checksum nằm private
dual_review.json. Không gọi candidate là official verified negative hours.

HUMAN_ACTION_REQUIRED:

1. Operator xác minh người B thật, tạo annotator_02/person_id riêng qua
   `python -m ground_truth_studio.admin create-account --reviewer-id REAL_B --person-id REAL_PERSON_B --role annotator_02 --identity-evidence REAL_IDENTITY_REFERENCE`
   (thay placeholders bằng danh tính thật đã xác minh, password nhập kín theo CLI).
   Dùng annotator_01 hiện có; không tạo alias cùng người.
2. Mỗi người mở Positive Review hoặc Negative Review → Mở/tiếp tục phiên → expander
   “Đánh giá độc lập A/B” → đánh dấu prior là JOINT_CONSENSUS_PRIOR khi đúng →
   “Mở lượt độc lập trắng A/B”. Review nguồn thực, không xem old joint/other/model,
   Save Draft, attest full view và independent, Submit. Không cần80 VN_POS để xét FAR.
3. Sau cả hai submit: “Xuất hai lượt độc lập đã Submit”. Nếu thiếu người thứ ba,
   giữ pending adjudication; không bypass role. Nếu có adjudicator thật dùng workflow cũ.
4. Import CSV hiện có chỉ là draft có source validation, không tự tạo timestamp/
   signoff. Hồ sơ independent ngoài Studio cần provenance thật để bổ sung importer;
   export hiện được cung cấp là joint nên không đủ để nhập thành independent.

Phần mềm không chứng minh danh tính/người có cùng xem hay không; đó là trách nhiệm
operator và evidence thật, không chỉ hai strings person_id hoặc checkbox.

## INFERENCE / SCOPE / EVALUATION

Pool technical61 trong63,6.8223424028h; VN_NEG_10/48 vẫn loại. Bundle checksum547
members được kiểm lại; source của87 reviewed videos khớp. Scope SHA
cc77669f0d5b4695fb7f1f043a6a446ba5ca949ecf46a7cc3f455e667cd89a78 giữ nguyên.
Checkpoint seed42 đã đọc trên CPU, SHA
f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6, class0 smoke/1 fire.

Historical run run_20261111_233035_neg_full có aggregate, không đủ bound source-PTS/
runtime preprocessing/completion proof; không xác nhận timeline bằng aggregate.
Không chạy inference mới vì chưa chốt independent GT và validation/test source
scopes/OP. FULL_DURATION_SAMPLED_5FPS là protocol đề xuất, không native-frame inference.
Runner hiện có export_week4_frame_jsonl.py cần provenance contract đầy đủ trước
khi dùng làm accepted evidence; không gọi historical frame/FPS timestamps là source PTS.

Không tự chia61 negative videos vào validation/test sau khi biết model outcomes.
Cần explicit disjoint video scope, nguồn hash và training contamination audit;
quét72 validation candidates rồi lock config/hash trước independent test. Không
đánh đồng technical file-list hash với validation lock. Negative FIRE/SMOKE hoặc
uncertain không dùng toàn video làm mẫu số FAR.

| Metric | A Frame baseline | B Temporal |
|---|---|---|
| Recall/Precision/F1 | null | null |
| Mean Delay | null | null |
| False alarm events | null | null |
| Valid negative hours | null | null |
| FAR/hour, Poisson CI95% | null | null |

Không có dữ liệu không phải0 false alarm. CSV per-video/matching/FP/FN/sweep mới
chỉ có header và chart hiển thị BLOCKED; không được gọi là thực nghiệm đã chạy.
Source event evaluator và exact Poisson CI đã có tests; acceptance model còn PARTIAL.

## Artifacts / tái lập

Script mới scripts/prepare_week456_acceptance.py xác minh joint export/source hashes,
frozen GT, raw audit checksum, VN split/labels, checkpoint mapping/backbone và môi
trường rồi xuất plans/provenance; không sửa dữ liệu, train hoặc inference.
Chạy với --output-dir <fresh path beneath .tools> --joint-export <private bundle>.
Output thực tế: .tools/week4_final_887052a0/evidence/. File chứa human review IDs giữ
private; configs/research chỉ chứa reference không bao gồm private absolute path.
Readiness file_sha256.json có thể tái tạo; chart FAR–Delay mới được tạo riêng sau
manifest ban đầu và sẽ được đưa vào checksum final tại bước chốt.

Week5 BLOCKED: D-Fire thiếu;3 training plans và model freeze inspection đã có,
chưa training/ΔmAP. Week6 PLANNED: matrix/backend checks có, FPS/export/error-cases
chưa đo. Xem hai readiness reports đi kèm. 416 raw artifacts pending cleanup approval
được giữ nguyên; không dùng tác vụ này làm authorization untrack.

## TESTS / FINAL STATUS

Xem docs/WEEK4_FINAL_TEST_RESULTS.json và .tools/week4_final_887052a0/*_tests.log:
compileall, Studio (gồm3 tests mới independent round), targeted temporal/evaluation/
replay, full unittest. Chỉ ghi PASS theo kết quả chạy, không suy ra scientific PASS.

Kết quả thực chạy: compileall exit0; Studio41/41 PASS (trước guard cuối), targeted
pytest53 PASS +10 subtests PASS; guard independent4/4 PASS; full suite bản cuối
283/283 PASS, exit0, không failures/errors/skips. Full trước guard282/282 cũng PASS,
log giữ riêng, không cộng các lượt thành số test độc lập. Test dùng synthetic data
không phải bằng chứng người thật đã review. Báo cáo RTSP legacy đã giữ nguyên byte
so với trạng thái sạch đầu tác vụ; git diff không có thay đổi ở báo cáo đó.

WEEK4: PARTIAL — HUMAN_ACTION_REQUIRED, independent GT/adjudication, split/inference/
validation OP lock chưa đủ. WEEK5: BLOCKED dataset. WEEK6: PLANNED.
Chưa sản phẩm nào được nâng lên ACCEPTED về khoa học bằng cách thêm docs/tests.
Engineering/source hoạt động được báo TESTED chỉ sau log kết thúc thành công.
