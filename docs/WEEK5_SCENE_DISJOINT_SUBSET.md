# Week 5 — scene-disjoint subset and baseline unblock

Ngày15/11/2026; branch fix-4; HEAD72a6d1ebc297da649894472eb10f5d4f1aa971f3. Worktree ban đầu sạch. Chưa commit/push/xóa dữ liệu. Trạng thái scientific **PARTIAL/BLOCKED**.

## Snapshot và bảo toàn

{
  "versions": 1057,
  "approved": 1046,
  "pending": 876,
  "rejected": 0,
  "delta_from_922": 124,
  "negative": 693,
  "positive_bbox": 353,
  "exclusion_statuses": {
    "APPROVED_PENDING_LINEAGE": 623,
    "APPROVED_LINEAGE_VERIFIED_PENDING_SPLIT": 423,
    "LINEAGE_VERIFIED_PENDING_HUMAN_APPROVAL": 374
  },
  "prior_versions_unchanged": true,
  "sqlite_integrity": "ok",
  "eligible_train": 0,
  "eligible_val": 0
}

Snapshot SQLite nhất quán, integrity ok. Tất cả file version thuộc snapshot922 trước vẫn đúng checksum và còn nguyên. Không sửa submitted reviews/original labels. Export riêng ở `.tools/week5_scene_72a6d1eb/approved_export/`; human approval và train eligibility phân biệt rõ. Số snapshot cố định, UI live có thể tăng.

## Scene decisions và tiêu chí subset

{
  "VN_candidate_pairs": 75,
  "VN_resolved": 0,
  "VN_unresolved": 75,
  "DFIRE_candidate_groups": 751,
  "DFIRE_confirmed_overlap": 1,
  "DFIRE_confirmed_independent": 0,
  "DFIRE_unresolved": 750,
  "cross_source_independence_shortlist": 120,
  "no_test_to_train": true,
  "parent_scene_sha256": "3ff3acafd3c66c9a5f3a68104593fdd6b54c375d12f6823564130975670b5c01"
}

D-Fire DFIRE_0000 có prior technical visual evidence cùng đường/cây/skyline/ban công giữa train/test; contact sheet SHA kiểm lại khớp. Được phân loại CONFIRMED_OVERLAP_SAME_CAMERA_SCENE; không dùng pHash/dHash đơn lẻ, không tự tạo cameraID và không gọi đây là human bbox approval.750 nhóm D-Fire còn unresolved,0 confirmed independent.75 VN pairs vẫn unresolved. Các contact sheets cũ giữ nguyên, không render lại corpus không đổi; queue references chỉ đến ảnh/evidence có sẵn.

`src/scene_subset.py` mới kiểm candidate theo approval thật, exact lineage, reservation và nguồn quarantine. Mọi cặp source đối lập train/val cần evidence độc lập; source trùng luôn bị chặn. Không có quyết định độc lập mới đủ chứng cứ, nên clean train=0, validation=0. Source VN_NEG_10 giữ quarantine do corruption. Không thay scene ID, không chọn ảnh bằng model metrics, không chuyển reserved test.

Shortlist có **120 cặp source đối lập** cần evidence, sắp theo số approval có lineage chịu tác động; tie-break SHA, không dựa mAP. `human_source_shortlist.json` cung cấp sourceSHA và yêu cầu provenance/reviewer/time; `scene_group_manifest.json` giữ75/751 candidate groups và contact sheets. Đây là shortlist kiểm chứng source independence, không phải bắt duyệt lại bbox. Không mặc nhiên coi các source không xuất hiện trong75 pairs là độc lập.

`validate_scene_decision` yêu cầu memberSHA chính xác, classification, reviewer/time và nguồn evidence camera/footage/visual; từ chối perceptual-only. Helper này kiểm schema, việc hash file evidence phải do caller thực hiện; không tuyên bố mọi chuỗi evidence_ref là đã được xác minh.

## Baseline

Membership cũ vẫn BASELINE_MEMBERSHIP_NOT_VERIFIED. Reuse metadata/training artifact audit có parent SHA, không thu được original image-SHA train list. Không suy membership từ YAML/checkpoint. Protocol baseline mới tại new_baseline_protocol.json giữ initialization `yolov8n.pt` SHA f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36; split/data SHA chưa điền vì chưa hợp lệ. Cần xác minh provenance khởi tạo và head mapping trước chạy; không coi protocol là trained baseline.

Thêm helper `training_membership` để ghi chính xác image/label SHA, source và scene của planned baseline TRAIN; từ chối val/test, duplicate và missing hash. Đã kiểm bằng test tổng hợp. Helper chưa chạy với corpus READY và chưa được tích hợp vào callback presentation logger; không coi planned membership là bằng chứng batch thực sự đã train. A/B/C vẫn phải chung baseline mới có SHA và evaluator, C freeze backbone +20% D-Fire TRAIN.

## Gates và thực nghiệm

DFIRE_READY=false; VN_BBOX_READY=false; SPLIT_READY=false; RESOURCE_PILOT_READY=false; WEEK5_TRAINING_READY=false. Historical checkpoint SHA kiểm ở receipt trước; không dùng checkpoint đó để tự xác nhận baseline independence. Không có manifest READY hoặc trainable YAML. Resource pilot NOT_RUN_DATASET_BLOCKED; training0run/0epoch, mAP/forgetting/video metrics mới chưa đo.

## Tests và reproduction

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_scene_subset.py",
      "tests/test_decoder_diagnostic.py",
      "tests/test_lineage_run_audit.py",
      "tests/test_week5_readiness.py",
      "tests/test_bbox_qa.py",
      "tests/test_domain_adaptation.py",
      "tests/test_week5_unblock.py"
    ],
    "exit_code": 0,
    "duration_seconds": 23.72304969999999,
    "log": ".tools/week5_scene_72a6d1eb/targeted.log",
    "tail": [
      ".........................................................                                             [100%]",
      "57 passed, 43 subtests passed in 21.08s"
    ],
    "log_sha256": "502a2b53df11ae2746044e289f796d49e3ab47b2e166a5fc8045bb357d151bdc"
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
    "duration_seconds": 194.13955170000003,
    "log": ".tools/week5_scene_72a6d1eb/full.log",
    "tail": [
      "...............................                              [100%]",
      "395 passed, 80 subtests passed in 192.50s (0:03:12)"
    ],
    "log_sha256": "d4732502fb1e38325783517180078b8b7647c98cbb4f71d0002d3e94c1c6465b"
  }
]

Targeted kiểm scene evidence, same-source/cross-split overlap, reserved test, quarantine, baseline membership, approval/lineage/class mapping, replay/freeze và fail-closed. `python -m pytest -q tests/test_scene_subset.py tests/test_decoder_diagnostic.py tests/test_lineage_run_audit.py tests/test_week5_readiness.py tests/test_bbox_qa.py tests/test_domain_adaptation.py tests/test_week5_unblock.py`; full `python -m pytest -q tests`.

Build reproduction source snapshot được lưu analysis_implementation.json; output hiện có không ghi đè. Original raw images/video, event GT, checkpoint và private annotations không sửa. Local private export/backup/logs không đưa lên Git.

## Việc tiếp theo

Ngài tiếp tục bboxQA. Ưu tiên đối chiếu source/camera/footage cho các cặp đứng đầu human_source_shortlist (SHA ánh xạ sang source paths trong candidate manifests); ghi reviewer/time và contact-sheet/source evidence cho quyết định same-scene hoặc independent. Không cần duyệt lại bbox đúng SHA. Cần source provenance/extraction logs cho lineage unresolved và initialization provenance nếu còn.

Tôi tiếp tục khi có evidence mới: kiểm hash và phạm vi quyết định, dựng subset version mới với parentSHA/exclusion reasons, nối approval+lineage, xác minh train/val disjoint và source integrity; sau đó mới pilot, baseline và A/B/C. Blocker hiện tại là chứng cứ độc lập scene còn thiếu, không phải thiếu số approval. Không hạ tiêu chí để mở training.
