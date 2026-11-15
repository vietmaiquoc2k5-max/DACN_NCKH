# Week 5 — final data readiness and baseline recovery

Ngày 15/11/2026; branch `fix-4`, HEAD `e733a6c3238aa4962e7752b5753d7c8e8da4fa16`. Worktree sạch trước khi sửa; thay đổi hiện tại chưa commit/push. **PARTIAL; training BLOCKED**.

## Đồng bộ review thật

Snapshot mới: **933 phiên bản / 922 ảnh**. APPROVED=922, REJECTED=0, PENDING=1000. Negative attestation=569; positive bbox=353; tăng 533 approval so với389. Không mặc định con số922; đã đo từ storage. Mọi version cũ của snapshot389 vẫn tồn tại và đúng checksum. SQLite backup API integrity=ok; không restore/reset. Snapshot cố định, live count có thể tiếp tục tăng.

Export mới ở `.tools/week5_recovery_e733a6c3/approved_export/` gồm JSONL, YOLO labels version, approval metadata và checksums; không sửa original labels. Phân loại eligibility: `{"APPROVED_PENDING_LINEAGE": 508, "APPROVED_LINEAGE_VERIFIED_PENDING_SPLIT": 414, "LINEAGE_VERIFIED_PENDING_HUMAN_APPROVAL": 383}`. Eligible train/validation=**0/0**. Approval thật được giữ kể cả lineage/split chưa đủ.

## Decoder: đã tìm đúng nguồn lỗi

**VN_NEG_10.mp4** SHA `3ed5b015393ce74d1da992dece66ce4a4b7abcd4b77ce5d3d23264794c4eacbe` tái hiện chính xác `Invalid NAL unit size (7937 > 7149)` và `Error splitting the input into NAL units`. Đây là bằng chứng lỗi video, không chỉ lỗi wrapper. Chỉ decode lại video nghi vấn này; không chạy lại toàn corpus. File nguồn không đổi. Lệnh FFmpeg và log SHA ở decoder_integrity.json.

FFmpeg native exit=**0**, nhưng stderr có lỗi nên integrity=**ERROR**. Python diagnostic exit0 chỉ xác nhận đã ghi bằng chứng; tool/process cũ exit1 giữ nguyên FAILED. Không nâng thành PASS. Wrapper mới `scripts/diagnose_lineage_source.py` lưu riêng nguồn, hash, stderr, timeout, native exit và Python exit. Mọi nguồn tương lai cần dùng receipt này để phân biệt successful launcher với healthy media.

Có 10 frame ứng viên gắn nguồn này, lưu source_quarantine.json; không cấp source eligibility. Receipt cũ thiếu PTS tại frame29802 được giữ UNVERIFIED. Các receipt khớp frame trước đó không chứng minh toàn video healthy. Chỉ nguồn VN_NEG_10 được quy lỗi bằng lần tái hiện này; không suy diễn các nguồn khác healthy tuyệt đối.

Lineage corpus đã chốt trước vẫn **797 verified /1.125 unresolved** trong1.922 ảnh. Không tự chuyển UNVERIFIED thành VERIFIED, không tạo PTS hay chạy lại anonymizer để giả provenance. Run1793 trước có đủ receipt nhưng lệnh lỗi; bảo toàn kết luận này.

## Scene và clean subset

Reused75 VN suspicious pairs và751 D-Fire groups, parent checksums ở evidence_parents.json. Không có scene decision mới được người thật cung cấp ở lượt này. Exact-pixel audit trước không đủ phân xử những nhóm chỉ giống perceptual. Giữ prior technical same-camera observations, quarantine và tất cả test reservations. Human priority queue giữ member IDs/contact sheets/evidence yêu cầu; không blanket-confirm751 là duplicate.

Chưa có tập train/validation độc lập theo scene được chứng minh, vì vậy clean_vn_manifest.json và clean_dfire_manifest.json có rows=[]/BLOCKED rõ ràng. Không chọn subset dựa trên mAP/test hay đổi tên scene để vượt cổng. Approval+lineage riêng lẻ chưa đủ chứng cứ độc lập giữa các scene. Mẫu thuộc nguồn lỗi không được cấp eligibility. Chưa tạo trainable YAML giả.

## Baseline recovery

**BASELINE_MEMBERSHIP_NOT_VERIFIED**. Metadata checkpoint, data_dfire.yaml, training args/results đã được đối chiếu; tìm trong training folder/dataset paths không thu được membership image-SHA/cache gốc. YAML path và metrics không thay thế ảnh thực đã train. Không báo forgetting trên test potentially exposed.

Đã tạo protocol cụ thể `new_baseline_protocol.json`: initialization `yolov8n.pt` SHA `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`, tên lớp head phải xác minh theo locked smoke/fire dataset. Local initialization SHA đã kiểm, receipt nguồn download gốc chưa phục hồi nên protocol vẫn conditional. Dataset/split SHA để null cho đến khi có chứng cứ thật, không bịa manifest.

Baseline mới chỉ train D-Fire TRAIN từ split source/scene độc lập đã khóa, chọn checkpoint bằng validation. VN/test không vào baseline training. Ghi image-SHA membership/presentation logs. A/B/C cùng bắt đầu từ một baseline mới có SHA; cùng VN/D-Fire validation/evaluator. StrategyC=freeze backbone+20% D-Fire train-only, denominator V+ceil(V/4). Giữ protocolv2 limits và thêm measured resource budget trước chạy.

## Gates, kiểm thử và thực nghiệm

Gates: `{"DFIRE_READY": false, "VN_BBOX_READY": false, "SPLIT_READY": false, "BASELINE_CHECKPOINT_READY": true, "RESOURCE_PILOT_READY": false, "WEEK5_TRAINING_READY": false}`. Resource pilot NOT_RUN_DATASET_BLOCKED, timing null. Baseline validation/A/B/C/forgetting/video metrics chưa đo; training **0 run/0 epoch**. Không tự coi tạo protocol là đã huấn luyện.

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_decoder_diagnostic.py",
      "tests/test_lineage_run_audit.py",
      "tests/test_week5_readiness.py",
      "tests/test_bbox_qa.py",
      "tests/test_domain_adaptation.py",
      "tests/test_week5_unblock.py"
    ],
    "exit_code": 0,
    "duration_seconds": 14.308661999999458,
    "log": ".tools/week5_recovery_e733a6c3/targeted.log",
    "tail": [
      "...................................................                                                   [100%]",
      "51 passed, 43 subtests passed in 12.60s"
    ],
    "log_sha256": "15cf3ee1bfc3175110efb4ab77bba09cfa7c4dfaefa3ae8f9d0b6636a98454a3"
  },
  {
    "name": "full",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests"
    ],
    "exit_code": 0,
    "duration_seconds": 173.2184273999992,
    "log": ".tools/week5_recovery_e733a6c3/full.log",
    "tail": [
      "..................................................................................                                    [100%]",
      "389 passed, 80 subtests passed in 171.70s (0:02:51)"
    ],
    "log_sha256": "ca9058e4f2504de86ab122a8c962ebcb082320c19e5fe38b56775937635fdb8c"
  }
]

Tests cover annotation/snapshot preservation, decoder stderr with native exit0, lineage completeness/hash/PTS, scene leakage/reserved test, baseline evidence, export/negative attestation, freeze/replay và fail-closed gates. Full suite dùng pytest; logs UTF-8 được subprocess ghi trực tiếp để tránh PowerShell UTF-16. Source/GT/checkpoint không sửa; diff check thực hiện khi seal delivery.

## Tiếp tục

NEXT HUMAN ACTIONS: tiếp tụcBBoxQA; phân xử queue scene bằng nguồn/camera/footage và decision/evidence thực; tìm extraction/anonymization receipts và membership baseline nếu còn. Không yêu cầu review lại nhãn đúng SHA. Với VN_NEG_10, cần nguồn nguyên vẹn có provenance hoặc giữ loại khỏi subset; không sửa video để giả continuity.

NEXT CODEX ACTIONS khi có evidence: validate scene decisions, dựng subset mới có parent SHA, kiểm tra disjoint và test reservation, nối approval+lineage, xác minh initialization/head, chạy measured pilot, baseline mới và A/B/C khi gates đạt. Hiện không có lý do hợp lệ để mở training.

Reproduce diagnostic: `python scripts/diagnose_lineage_source.py --source datasets/negative_videos/VN_NEG_10.mp4 --output NEW_LOCAL_DIAGNOSTIC`. Đích phải mới. Targeted/full test commands trong tests.json. Private backups/exports/logs vẫn ở ignored .tools. Không commit/push/xóa hay sửa dữ liệu nguồn.
