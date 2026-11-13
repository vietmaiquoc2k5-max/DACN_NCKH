# Kiểm kê và chia tập trước gán nhãn 80 VN_POS

Thời điểm bàn giao (Asia/Saigon): 2026-11-14T01:58:33.051196+07:00. BRANCH: `fix-4`. HEAD: `6942fa567683ed9fdc3ef455188db1a74cef2c7f`.
Worktree sạch khi bắt đầu; hiện chỉ có code/metadata/tài liệu mới trong phạm vi và hai file Studio được sửa. Không commit/push, di chuyển/xóa/sửa video gốc, sửa checkpoint, GT frozen hoặc benchmark lịch sử.

**FINAL: PARTIAL. INVENTORY_PASS; PRE_ANNOTATION_SPLIT_PROVISIONAL; TEST_INDEPENDENCE_NOT_VERIFIED; HUMAN_ANNOTATION_PENDING.** Khóa checksum danh sách provisional không phải `LOCKED_INDEPENDENT_TEST` hay chứng minh đã hoàn tất test. Không train, không inference để chia tập, không chọn lại OP, không dùng nhãn AI làm GT.

## Repository và phạm vi đã xác minh

Đúng commit tham chiếu. Catalog `data/video_catalog_local_v2.json` schema 2 có 183 references: 80 VN_POS, 24 POS historical, 79 negative references; 16 references lịch sử thiếu không được giả thành nguồn hiện có. Đọc/hash thực tế **167 nguồn tồn tại** (80+24+63). Các trường paths trong CSV cũ không được dùng để đoán alias; catalog local pin đúng bytes là đầu vào. Không đổi việc 80 VN_POS đang ở thư mục có tên `datasets/negative_videos/` thành nhãn âm tính.

Đã kiểm checksum 292 artifacts run khoa học trước: 24 POS + 47 NEG hợp lệ, 5.7496550970 giờ NEG, 71 video/107.805 mẫu. Đây là lịch sử đã có exposure, OP vẫn PROVISIONAL; không tái đánh giá/tuning bằng chúng trong tác vụ này.

## VN_POS inventory và strict decode

| Chỉ số đo | Kết quả |
|---|---:|
| Expected / tìm thấy / hash khớp | 80 / 80 / 80 |
| Thiếu file | 0 |
| NORMAL_EOF theo strict v2 | 79 file |
| TRUNCATED_STREAM | 1 file: VN_POS_28 |
| SHA khác nhau trong 80 file | 79 |
| SHA khác nhau trong 79 file đạt | 78 |
| Tổng duration source PTS của 79 file, có lặp byte | 17436.715981s = 4.8435322168h |
| Duration tính một lần mỗi SHA trong nhóm đạt | 17347.752106s = 4.8188200293h |

Đây là duration của **positive candidates chưa có GT**, không phải negative exposure hoặc số giờ footage/camera độc lập. SHA khác nhau cũng chưa chứng minh footage độc lập.

Tái sử dụng nguyên `src/video_audit_v2.py`: FFprobe full frame/packet counts + integer PTS/durations, FFmpeg `-xerror`/progress=end, OpenCV sequential count, stream start/duration, source hash trước/sau; không dựa riêng vào `grab()==False`. Không nới tolerance số frame/timeline để nhận video. VN_POS_28 có decoder/probe errors, count/timeline mâu thuẫn, FFmpeg nonzero; giữ nguyên bytes và loại khỏi split đủ điều kiện. Không gọi lỗi metadata đơn thuần là file hỏng.

CSV/JSON inventory phân biệt nominal r_frame_rate, average FPS metadata, số frame thực decode, header frame count, container/stream duration và duration PTS đã được nhận. Trường không được đo/xác minh là null; duration full-source của VN_POS_28 là null. Codec metadata của cả 80 là H264; video/audio/container/resolution/time_base được probe thật.

## Exact, near-duplicate và nhóm cảnh

**1 cặp trùng toàn bộ byte: VN_POS_50 / VN_POS_51**, SHA `fe0ff123e94b10e72f60c83b0e8313e0376cd468022c226afa583a4ee42cce6c`. Không tính thành hai nguồn độc lập; không xóa hoặc tự sao chép submission/nhãn. Không tìm thấy exact-file pair giữa VN_POS và các 24 POS/63 VN_NEG được hash hiện tại.

So khớp pixel đo thật trên 167 videos và **14.413 ảnh hiện có**: 3.121 VN, 3.538 FireSmokeLabeler, 7.754 legacy. Có **83 candidate pairs video** (52 nội bộ VN_POS, 31 với references lịch sử) và 1.245 cặp quan sát query–ảnh corpus khác cần triage; không diễn giải các pixel hits là số footage hoặc người review độc lập.

Giao thức: 48 yêu cầu ordinal đồng đều theo count nguồn; DCT pHash/dHash ba crop, HSV histogram, bỏ ảnh gần phẳng (std<10), pHash<=6, dHash<=8, histogram intersection>=0.8; cần nhiều quan sát hoặc một match rất sát, kiểm chuỗi tăng một-một. Không tải embedding/model hoặc chạy detector. Extra ordinals từ manifest ảnh cũ chỉ dùng kiểm lineage; chúng có thể do pipeline AI-onset cũ lựa chọn. **Nhóm/split chỉ dùng query đồng đều**, không dùng extra ordinals AI-conditioned, onsets, scores hoặc nhãn để phân nhóm/chia tập. Xem `pixel_sampling_interpretation.json` để hiểu rõ flag collector ban đầu chỉ có nghĩa không đọc trực tiếp prediction/onset.

Fingerprint dùng random seek để lấy **pixel thật**, lưu requested/backend-reported position; không coi vị trí đó là ordinal/PTS có thẩm quyền cho GT. VFR GT/annotation dùng timeline strict riêng. Sampling hữu hạn có thể bỏ sót clip ngắn, crop/màu/watermark hoặc cảnh liên quan; không có match không đồng nghĩa PROVEN_UNSEEN.

**56 components ứng viên**, trong đó 14 có nhiều nguồn. Đây không phải 56 camera/cảnh vật lý đã xác nhận. Mọi liên hệ perceptual vẫn `SCENE_RELATION_UNCERTAIN`; singleton chỉ là source identity với camera unknown. Ngài xác nhận chưa có metadata nguồn/camera. Năm URL trong CSV chỉ là gợi ý suy từ tên cũ (một URL còn thiếu ID), không coi là provenance đã xác nhận hoặc đồng nhất các URL/kênh thành một camera.

Quan sát triage của trợ lý (không phải human signoff): nhóm `SCENE_38c9947cd3542ea2` có các match logo/outro ANTV giống nhau nhưng khung ngữ cảnh khác. **Chung logo không chứng minh chung camera/event.** Không tự xóa edge hoặc sửa manifest đã hash; giữ nhóm bảo thủ tới khi người thật review các đoạn không có branding. Đã tạo 14 sheet match và 16 sheet có ngữ cảnh cho 14 nhóm; [index ngữ cảnh](../runs/benchmarks/vn_pos_preannotation/run_20261114_6942fa56/analysis_v1/scene_contact_sheet_context_index.json). Các ảnh chỉ giữ local, không commit/chia sẻ ảnh gốc chưa xử lý. `assistant_visual_triage.json` không chứa xác nhận của người thật.

## Historical exposure

80/80 có record `AI_GOI_Y_MOC`/`AI_DO_MOC` trong CSV lịch sử, đối chiếu scripts thực sự gọi predict và các ảnh/labels đã tồn tại. Phân loại bảo thủ **KNOWN_EVALUATION_EXPOSURE theo video ID: 80**. Record cũ không có SHA runtime nên không khẳng định đã xác thực bytes chính xác của từng lần chạy; không coi suy đoán là bằng chứng mọi ngưỡng đã được tuning trên cả 80 video.

KNOWN_TRAIN / KNOWN_VALIDATION / KNOWN_TUNING_EXPOSURE có runtime/source binding đủ để khẳng định: chưa xác minh được, không suy từ tên split. NO_EVIDENCE_OF_PRIOR_USE không có video nào trong nhóm này; không có PROVEN_UNSEEN. Training/model-selection history vẫn UNKNOWN cho cả 80 ở khía cạnh corpus hash/scene gốc. Legacy image reservations 48 train /16 val /16 test không tự là bằng chứng đã train hoặc là test độc lập.

Đã load **6 checkpoints trên CPU, chỉ đọc** và kiểm SHA trước/sau: 5 benchmark checkpoints khai báo data_dfire.yaml; root best.pt tham chiếu archive config gốc không hiện có. Không có original training corpus hash manifest. D-Fire không có tại các route cấu hình/Ultralytics datasets root/aliases workspace đã khảo sát, nên overlap D-Fire là BLOCKED_UNAVAILABLE, không ghi “0 overlap” giả. Corpus legacy/FireSmoke được so pixel/hash như references hiện có; không giả định chúng chính là snapshot train gốc.

## Split trước annotation

Seed **42**, policy `vn_pos_group_preannotation_v1`. Giữ reservation cũ khi toàn bộ component tương thích; quarantine component có kỹ thuật không đạt, giao references/corpus lịch sử, hoặc conflict train/val/test cũ. Không di chuyển frame, không thay split_manifest.csv cũ, không chuyển old test vào train để đạt tỷ lệ. Seed/duration allocator chỉ dành nhóm chưa có reservation; 80 video hiện tại đều có reservation cũ nên không dùng nó để bịa tập test mới.

| Proposed split | Videos | Trạng thái |
|---|---:|---|
| TRAIN_CANDIDATE | 27 | Chờ scene review và nhãn người; chưa cho phép train |
| VALIDATION_CANDIDATE | 11 | Chờ GT trước sweep/chọn OP |
| INDEPENDENT_TEST_CANDIDATE | 13 | **BLOCKED, TEST_INDEPENDENCE_NOT_VERIFIED** |
| BLOCKED_UNASSIGNED | 29 | 9 conflict reservation, 19 overlap reference cần review, 1 truncated |

Không có SHA/source component/scene component đã nhận diện vượt split mới: **0**. Điều này không chứng minh mọi camera/cảnh chưa có metadata đều độc lập; split chưa được công nhận chính thức. Near candidates bao gồm false positives branding nên các nhóm blocked có thể được tách sau review người thật và policy version mới, không thay âm thầm.

Canonical manifest SHA256: `24e3064daefd0d108da179caeeb35223966daec586321a6edd401f97c6513d08`.
File manifest SHA256: `8acd4311db47fe7e100a54e4503be12746fc82b7742f8e7ab9775baddc544e37`.
Catalog SHA256: `1c8facb8d2cfebb24602bbf7093c8b43770aacb6618f966d49d731acff9b7943`.
`vn_pos_split_provenance.json` ghi seed/policy/HEAD/UTC/source hashes/payload hashes. Tất cả rows PENDING_HUMAN_REVIEW, independence_verified=false. Danh sách provisional đã hash trước bất kỳ inference mới nào lên test candidates. Không có `LOCKED_INDEPENDENT_TEST`, không có OP lock mới.

## Ground Truth Studio và sửa lỗi thực tế

Catalog real DB có đủ 80 VN_POS; audit read-only trước/sau vẫn 87 submitted cũ, 3 draft; **0 review records VN_POS**. Không khởi tạo/migrate DB thật trong QA, không tạo fake users/submissions.

UI thêm split/group/readiness, ưu tiên test-reserved → validation → train, giữ ẩn nhãn/prediction/onset cũ; helper chỉ expose allowlist routing fields. Manifest/catalog/record/timeline checksum thay đổi thì fail closed. Reuse explicit-duration audited PTS để mở video dài mà không probe lại hàng phút. Mark Start/End vẫn gắn source SHA/ordinal/PTS, Save Draft/Submit vẫn do người thật chủ động; không promote GT frozen. Luồng independent review cũ và single-reviewer export giữ nguyên.

Real QA **79/79 nguồn hợp lệ**: source frames 0/1/quay lại, PTS/count, authenticated HTTP Range, Chrome headless local phát thật/seek thật. VN_POS_28 không được đưa vào phép kiểm chứng full-video hợp lệ. API source-frame viewer và browser playback là hai phép kiểm riêng, không gọi random fingerprint seek là frame navigation GT.

Lỗi tìm và sửa: (1) AppTest/Windows spawn có thể import lại app.py, entry point trước đây chạy ngay và cố mở Studio mặc định; thêm `__main__` guard, test đảm bảo spawn import không mở Repository. (2) Raw MP4 có PTS đầu khác0: clock browser và elapsed-source time bị lệch; đã thêm đổi clock hai chiều, sửa seek/mark. Quan sát Chrome chứng minh mapping cũ chọn sai frame đầu ở **44/79 nguồn**; mapping mới đúng 79/79, có actual MP4 fixture PTS đầu0.1s. Không sửa hồi tố GT/benchmark cũ. (3) HTTP Range bị hủy khi seek trên Windows có thể WinError10053; xử lý ConnectionAbortedError cùng reset/broken pipe, không che lỗi decode.

Chrome sandbox từng bị chặn GPU process, không phải video hỏng; QA thành công dùng profile tạm riêng ngoài sandbox, loopback, không DB thật. Các log thử lỗi và chạy thành công đều giữ. Không mở cửa sổ Chrome cá nhân hoặc chạy model trong review.

## Negative gap và Week5

NEGATIVE_INDEPENDENT_TEST=**BLOCKED**. 47 NEG hợp lệ/5.7496550970h là pool lịch sử đã có inference/evaluation exposure; **0 giờ independent negative test đã chứng minh**, cần bổ sung **>=5h test NEG mới** với full decode, GT/provenance/disjoint source/scene và lock trước test. Không dùng 80 VN_POS hoặc đổi nhãn chúng để bù FAR. Xem [gap report](NEGATIVE_TEST_INDEPENDENCE_GAP.md).

VN corpus hiện có 3.121 ảnh/label pairs: SHA kiểm lại, 0 missing, 0 sai syntax, 0 exact-image SHA crossing trong legacy VN split. Các con số đó không xác nhận labels đúng semantic hoặc source/scene independence. 27 train candidates chỉ liên quan **691 legacy positive frame rows** trước review, không phải tập được phê duyệt train. Có **15 NEG video IDs /129 ảnh có label rỗng** cần review lại vì video hiện được human đánh dấu chứa khói/uncertain; không suy mọi frame đều sai và không tự thay labels. D-Fire/replay20% BLOCKED; chưa fine-tune/freeze/replay hoặc đo mAP/forgetting. [Readiness Week5](WEEK5_VN_POS_DATA_READINESS.md).

## Tests, artifacts và tái lập

Targeted: **75 passed**, exit 0; full unittest: **320 tests, OK**, exit 0. Cả hai là kết quả chạy sau bản sửa cuối, không phải số dự kiến. Thêm source SHA/missing/metadata separation, duplicate/source/scene leakage, deterministic group allocation, unknown exposure, manifest tamper, no prediction-driven split, no automatic GT/test promotion, audited PTS tampering, nonzero MP4 clock, annotation isolation/UI và spawn import. Fixture chỉ là QA, không làm GT thực nghiệm. Full suite có test tạo lại report RTSP cũ; runner khôi phục bytes cũ trong finally. Dữ liệu/GT/model gốc giữ checksum.

Lượt đầu có expectation một ảnh đơn quá mạnh và lỗi Windows spawn/DB readonly; đổi ca near-duplicate thành chuỗi tái encode (không nới thresholds), sửa entrypoint. Log initial failures vẫn giữ. Thiết lập TEMP/TMP QA trong .tools và FFmpeg trên PATH. Lượt delivery cuối 75/320 đều exit0 sau bản sửa clock, không lấy lượt cũ thay thế.

Deliverables dữ liệu trong `data/`: inventory.csv/json; exact_duplicates.csv; near_duplicate_candidates.csv; scene_groups.csv; scene_review_required.csv; historical_exposure_audit.csv; preannotation_split_manifest.csv/json; split_provenance.json, tất cả prefix `vn_pos_`. [Run](../runs/benchmarks/vn_pos_preannotation/run_20261114_6942fa56/summary.json) giữ raw v2 records, fingerprints, pair evidence, contact sheets, tests/logs/checksum. Raw video frames/hashes/logs/plots tuân thủ .gitignore, không stage. 416 raw artifacts lịch sử còn chờ cleanup authorization được giữ nguyên.

```powershell
python -X utf8 scripts/audit_vn_pos_decode.py --run-dir runs/benchmarks/vn_pos_preannotation/NEW_RUN --workers 2
python -X utf8 scripts/fingerprint_vn_pos_sources.py --run-dir runs/benchmarks/vn_pos_preannotation/NEW_RUN --workers 2
python -X utf8 scripts/prepare_vn_pos_preannotation.py --run-dir runs/benchmarks/vn_pos_preannotation/NEW_RUN --analysis-id v1 --seed 42
# --publish chỉ khi đích data/vn_pos_* chưa tồn tại; không ghi đè manifest đang frozen.
```

Allocation tái lập trên dữ liệu thật cả thứ tự input xuôi/đảo đều khớp. Timestamp/file hash sẽ khác khi tạo run mới; cùng inputs/policy/seed phải giữ source/group/split assignment. Sau annotation nếu coverage FIRE/SMOKE/events chưa phù hợp: review policy công bố trước, tạo version mới có prior-manifest SHA và lý do; không điều chỉnh theo test model performance.

`.gitattributes` bổ sung quy tắc `-text` chỉ cho các manifests/artifacts VN_POS mới để Git giữ nguyên byte CSV/JSON có checksum. Không normalize hoặc thay bytes catalog, legacy manifests hay GT cũ. Catalog vẫn được pin bằng SHA byte thực tế của lần chạy; nếu môi trường checkout làm đổi byte thì cần đối soát/khôi phục đúng input, không âm thầm cập nhật hash để bỏ qua guard. Kiểm thử thực tế của tác vụ là Windows, không tuyên bố đã chạy Linux CI.

[Bắt đầu gán nhãn](VN_POS_ANNOTATION_START_GUIDE.md). Cần người thật xử lý quan hệ cảnh/provenance, gán nhãn và cung cấp negative holdout mới; không cần agent chạy inference mới hoặc giả lập chứng nhận để lấp các thiếu hụt này.
