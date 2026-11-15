# Week 5 — data readiness, bbox QA và domain adaptation

Ngày 15/11/2026. **PARTIAL / BLOCKED training**. Không commit/push, không xóa dữ liệu gốc, không đổi checkpoint hoặc frozen GT.

## Repository và dữ liệu người dùng

BRANCH: `fix-4`.
HEAD: `ed406cca8fced883ec2a35de6a912987e9d69f2e` (trùng tham chiếu).
WORKTREE: sạch trước khi sửa; hiện có thay đổi chưa commit cho bbox QA, lineage, export, audit, tests và protocol v2.
Snapshot human review: **2026-11-15T14:42:01.531334+07:00**. Các số review dưới đây gắn snapshot, không khóa ngài tiếp tục làm việc.

HUMAN BBOX QA: 34 theo prompt; 38 ảnh snapshot đầu; **107 ảnh/108 versions snapshot cuối**, APPROVE mới nhất 107, REJECT 0, pending 1.815. Trong nhóm 129 còn 22 pending. 74 ảnh được ngài xác nhận âm tính, 33 ảnh có bbox. Tất cả 129 ảnh còn tồn tại, đúng hash và có thể tìm qua UI/backend; không có ảnh mất. [Báo cáo đối chiếu queue](WEEK5_BBOX_QA_QUEUE_RECONCILIATION.md) giải thích 3.121 đường dẫn → 3.101 SHA → 1.922 queue.

## Lineage thực nghiệm

Đã đọc extractor cũ, `src/preannotation.py`, `src/video_audit_v2.py`, Studio video/bbox services và source manifests. Scope chạy: **129 ảnh ưu tiên / 15 video**, bao gồm toàn bộ 107 ảnh đã duyệt ở snapshot cuối.

- **51 VERIFIED_DERIVED_IMAGE**: hash video candidate khớp, decode tuần tự từ frame 0, lấy đúng index khai báo, đọc integer PTS/time_base bằng ffprobe, tái tạo resize INTER_AREA tối đa 1920 + JPEG95 có SHA ảnh khớp byte. Mỗi receipt ghi hash nguồn/ảnh/code, transform, PTS, index và thuật toán.
- **78 UNVERIFIED_LEGACY**: ảnh không tái tạo khớp theo pipeline xác định trên. Không suy ra là ảnh giả/sai nguồn; anonymization/encoder hoặc transform lịch sử chưa có đủ chứng cứ. Không dùng similarity model để nâng cấp trạng thái.
- Ambiguous/source missing/source hash mismatch trong scope chạy: **0**. Đây là kiểm chứng candidate nguồn/index đã khai báo, không phải tìm kiếm toàn bộ kho video; receipt ghi rõ giới hạn đó. Không cấp camera/scene independence từ frame match.
- **2.992 đường dẫn ứng viên còn lại chưa chạy lineage trong lượt này**. Không công nhận toàn corpus VN verified.

Bbox đã duyệt không cần gán lại vì đổi format/storage. Với 78 ảnh chưa khớp, giữ nguyên approval và original image; cần tìm extraction/anonymization receipt cũ hoặc review nguồn để xác định transform. Nếu buộc trích xuất lại thì tạo version mới và kiểm chứng chuyển bbox trước khi dùng; lượt này chưa thay ảnh hay chuyển tọa độ tự động.

## D-Fire và split

Corpus giữ tại `datasets/dfire_author_linked_kaggle_v1`, provenance **author-linked Kaggle mirror v1**, không tự nhận byte-identical archive OneDrive. Không tải lại. Rehash **43.055 file** khớp manifest trước; **21.527 images + 21.527 labels**; 0 corrupt image. Mapping `0=SMOKE, 1=FIRE` giữ theo source YAML và kiểm tra semantic trước, không remap.

**21.187 ảnh có nhãn hợp lệ; 340 ảnh có lỗi**, gồm 397 dòng lỗi: 18 ZERO_OR_NEGATIVE_AREA, 379 OUT_OF_BOUNDS. Không clamp; [chi tiết từng dòng](../runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/dfire_bbox_issues.json). Không phát hiện class/parser/nonfinite lỗi ở lượt phân loại này.

0 nhóm trùng SHA. **751 nhóm nghi trùng theo pHash+dHash**, tất cả chưa được tự động phân xử bằng exact decoded pixels. Đã tạo **766 contact sheets** phủ mọi member và [chỉ mục local](../.tools/week5_readiness_ed406cca/dfire_contacts_v2/index.html). Kiểm tra trực quan kỹ thuật nhóm DFIRE_0000 thấy cùng đường/cây/skyline/ban công ở train/test; giữ quarantine. Giữ thêm ví dụ Camera0001 val/test từ audit trước. Không gọi toàn bộ 751 nhóm là leakage đã xác nhận; còn 750 nhóm chưa được phân xử trực quan trong lượt này. Quan sát của agent không phải human bbox approval.

Quarantine cũ **3.003 ảnh** giữ nguyên. D-Fire candidates **12.409 train / 2.547 val / 3.568 reserved test**, chưa phải clean READY. VN reservations: **1.376 train / 528 val / 572 test / 645 blocked**; chưa cấp quyền training/eval. VN có 20 nhóm exact trùng nằm cùng legacy train (VN_POS_50/51) và 75 near-duplicate candidate pairs qua split; chưa xác minh xong scene. Không chuyển test sang train, không dùng nhãn AI hay kết quả mô hình để quyết định split.

Đã tạo 7 manifest version mới (VN train/val/test/quarantine, D-Fire train/val/test), có parent SHA/reason và giữ reservation. Chưa tạo YAML chạy YOLO vì chưa có subset đáp ứng approval + provenance + độc lập scene. Baseline `args.yaml` trỏ `data_dfire.yaml` nhưng thiếu manifest SHA membership của lần train thực tế; không coi đây là bằng chứng eval chưa từng xuất hiện trong train.

## Export approval dùng được nhưng không vượt cổng dữ liệu

Adapter: Human Approved Annotation → kiểm image/label/version SHA → lineage receipt → locked split + source/scene disjoint → eligible rows. Có test tổng hợp đi hết nhánh eligible, test reserved test không được đưa vào train, negative cần attestation. Training validator kiểm cả approval receipt và lineage receipt; derived images chỉ được chấp nhận khi có chứng cứ tái tạo xác định.

Export thật tại `.tools/week5_readiness_ed406cca/approved_export_final/`: **107 approvals**, YOLO labels phiên bản mới, JSONL, approval provenance, checksum manifest; **0 eligible training rows**. 66 APPROVED_PENDING_LINEAGE, 41 APPROVED_LINEAGE_VERIFIED_PENDING_SPLIT; 10 ảnh khác LINEAGE_VERIFIED_PENDING_HUMAN_APPROVAL. Original labels/private submitted reviews không đổi. Phê duyệt không bị yêu cầu lặp lại chỉ vì bổ sung lineage. Không đưa raw SQLite/reviewer metadata lên Git.

```powershell
python scripts/export_week5_bbox_approvals.py --snapshot .tools/week5_readiness_ed406cca/private_snapshot_final --output .tools/week5_readiness_ed406cca/NEW_PRIVATE_EXPORT --lineage-dir runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/lineage
# Khi có locked split đã review: thêm --split-manifest VERIFIED_SPLIT.json
```

Đích export phải là thư mục mới trong `.tools`. Split đầu vào phải có parent path/SHA, source/scene grouping, provenance evidence và các khóa readiness; không chỉ điền Boolean để bỏ qua kiểm chứng. Export annotation không tự tạo WEEK5_TRAINING_READY.

## Cổng nghiệm thu và training

| Gate | Kết quả | Nguyên nhân |
|---|---|---|
| DFIRE_READY | BLOCKED | scene/split provenance và baseline train membership chưa đủ |
| VN_BBOX_READY | BLOCKED | approval + lineage + split chưa tạo tập train/val hợp lệ |
| SPLIT_READY | BLOCKED | 751 D-Fire groups, 75 VN pairs và unknown scenes còn cần phân xử |
| BASELINE_CHECKPOINT_READY | PASS | SHA, model load, mapping thực tế đạt |
| RESOURCE_PILOT_READY | BLOCKED | chưa có subset hợp lệ để chạy pilot |
| WEEK5_TRAINING_READY | BLOCKED | các cổng dữ liệu/pilot chưa đạt |

Checkpoint SHA `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6` giữ nguyên. CPU fallback, CUDA không sẵn có trong lần kiểm tra; chưa đo step/epoch/throughput/peak training memory. Không gọi kiểm tra hardware là resource pilot.

Đã sửa **Strategy C = freeze backbone + replay 20%**, dùng protocol v2 có parent SHA. Strategy B/C đều 1.738.566 tham số trainable; A 3.011.222. Kiểm chứng optimizer steps trên mạng tổng hợp giữ nguyên backbone parameters/BN buffers và đổi head; checkpoint YOLO thật kiểm topology/policy, chưa train. D-Fire replay chỉ train: ceil(V/4)/(V+ceil(V/4)), ghi rounding; cùng lựa chọn seed qua mỗi epoch. Legacy protocol v1 không được resume C với nghĩa freeze mới.

TRAINING: actual runs **0**, completed epochs **0**. Ba CLI gate checks trả **exit 2**, `training_executed=false`. Baseline VN/D-Fire mAP, A/B/C mAP, delta/forgetting và video metrics mới: **null / NOT_MEASURED**. Không chọn checkpoint hoặc threshold bằng test.

Lệnh sau chỉ dành cho khi manifest, checkpoint, pilot và protocol đều hợp lệ; các placeholder chưa tồn tại:

```powershell
python scripts/run_week5_domain_adaptation.py --protocol configs/research/week5_execution_protocol_v2.json --dataset-manifest READY_DATASET.json --budget-receipt MEASURED_PILOT.json --strategy full_finetune --seed 42 --output-dir runs/benchmarks/week5_domain_adaptation/NEW_FULL_RUN
# B: --strategy freeze_backbone; C: --strategy replay_20_percent; mỗi lần dùng output mới.
```

## Tests và lỗi đã sửa khi kiểm chứng

- `syntax`: py_compile thành công; exit `0`; log `.tools/week5_readiness_ed406cca/syntax_final_verified.log`.
- `targeted`: 77 passed, 54 subtests passed in 20.65s; exit `0`; log `.tools/week5_readiness_ed406cca/targeted_final_verified.log`.
- `full`: 380 passed, 80 subtests passed in 183.87s (0:03:03); exit `0`; log `.tools/week5_readiness_ed406cca/full_final_verified.log`.

Lệnh full: `python -m pytest -q tests`. Không có failed/skipped trong lượt thành công cuối. Targeted gồm test_week5_readiness, test_bbox_qa, test_domain_adaptation, test_gt_studio_ui, test_preannotation, test_week45_acceptance. Kiểm tra `git diff --check` và artifact SHA thực hiện khi chốt bàn giao.

Lượt phát triển đầu gặp: snapshot đọc path label tuyệt đối cũ cần bind original private root; test SQLite chưa đóng connection trên Windows gây WinError32; contact sheet relative_path chưa resolve. Đã sửa bằng xác minh origin explicit, `contextlib.closing` và resolve đường dẫn; giữ log thất bại riêng, chạy lại thành công. Reader hiện chỉ đọc version có checksum marker; writer publish marker bằng rename nguyên tử để reader không đọc bản đang viết.

## Bàn giao và phần còn thiếu

IMPLEMENTED: queue/filter/search/pagination, snapshot, append-only review handling, lineage verifier, D-Fire line/group audit, private export adapter, fail-closed manifests và freeze+replay.
TESTED: UI AppTest, backend/catalog thật, SQLite concurrency, approval/negative/hash/split/freeze/replay contracts, targeted/full suite.
VERIFIED: nguồn/nhãn checksum, review preservation, 51 derived frames, 340 bbox issues, actual checkpoint policy.
ACCEPTED: phần triển khai có bằng chứng test. Scientific training acceptance **PARTIAL/BLOCKED**, không chuyển thành PASS.

NEXT HUMAN ACTIONS: tiếp tục 22 ảnh ưu tiên còn chờ ở snapshot; phân xử bbox/same-scene trên contact sheets; cung cấp provenance extraction/anonymization và baseline training membership nếu còn. Không cần review lại 107 approval còn đúng SHA. Các số trực tiếp có thể tăng sau snapshot.
NEXT CODEX ACTIONS sau khi có chứng cứ: nối approval/lineage mới vào manifest, hoàn tất scene grouping có test reservation, xác minh subset; chỉ khi đạt data gates mới pilot, baseline validation, ba strategy/seed theo giới hạn đo thực và video evaluation validation.

Bằng chứng máy đọc tập trung ở [`runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/`](../runs/benchmarks/week5_domain_adaptation/readiness_20261115_ed406cca_v2/). Raw images, backups, review exports và logs ở local `.tools`; không commit/push trong lượt này.
