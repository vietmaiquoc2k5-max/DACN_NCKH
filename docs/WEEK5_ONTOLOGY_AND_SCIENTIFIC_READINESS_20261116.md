# Week 5 — ontology proposal and final scientific readiness audit

Ngày16/11/2026. Branch fix-4; HEAD `20b050e6a865888ba9f76cecc190f6f56dff25a8`. Worktree trước khi sửa sạch. Chưa commit/push/xóa dữ liệu. Week5:16–22/11/2026. Trạng thái scientific: **PARTIAL/BLOCKED**; chưa có thực nghiệm domain adaptation.

## IMPLEMENTED

Đề xuất [FIRE/SMOKE/welding ontology](WEEK5_FIRE_SMOKE_WELDING_POLICY_PROPOSAL.md) và module src/annotation_policy.py cho guidance không ràng buộc/guard policy approval; không relabel và không auto-GT. D-Fire giữ class0SMOKE/1FIRE, original labels nguyên trạng. Controlled flame có thể positive cấp frame nhưng không tự là incident; welding sparks/fumes không tự thành FIRE/fire-event SMOKE. Đây là lựa chọn ontology **chờ phê duyệt**, không thay convention cũ trong Studio.

Đã dựng snapshot impact/class/source/reservation, cập nhật shortlist source từ approvals mới, versioned clean candidate manifest và readiness gates có parent/hash. Reuse receipt/scene audits trước thay vì decode/rehash toàn corpus lần nữa; source YAML D-Fire và mọi ảnh/label approval trong scope mới kiểm hash trực tiếp. Không ghi dữ liệu private vào Git.

## VERIFIED — integrity và human GT

Snapshot 2026-11-16T17:56:47.030357+07:00. Resolved5, unresolved0; strict integrityPASS, không phát ERROR lịch sử trong hai lần scan/strict. Cả5resolution checksum giữ nguyên và được reader đối chiếu annotation mới/reviewer/time/evidence/forensic bytes. Bản export1108 trước được kiểm lại toàn bộ checksum, không tạo lại human decisions. Backup3265file và original forensic bytes vẫn khớp. Production không sửa checksum/labels/receipts/SQLite; frozen GT kiểm lại khi seal delivery.

Valid versions=1748; unique valid decisions=1727; `{'APPROVE': 1727}`. Đếm theo imageSHA/latest version, không cộng approval trùng của reconfirmation. Live count có thể tăng sau timestamp.

## Phân bố nhãn — chỉ scope đã APPROVE, chưa phải eligible training

| Loại ảnh | Số ảnh |
|---|---:|
| Âm tính có human approval | 1127 |
| SMOKE only | 237 |
| FIRE only | 95 |
| Cả FIRE và SMOKE | 268 |

Tổng ảnh approved=1727; tổng bboxSMOKE=613, bboxFIRE=765. Số bbox khác số ảnh. Bảng theo từng source/reservation trong annotation_impact.json; các reserved test giữ nguyên, không đưa vào model selection. Eligible train/validation/test hiện **0/0/0**, nên phân bố class/source trong eligible scope là0, không lấy approved pool làm training size.

Impact policy: actual welding images **UNKNOWN** vì thiếu metadata semantic được xác thực. Keyword candidates=0; keyword không xác nhận có/không hàn. Không tự áp chính sách mới lên các nhãn đã duyệt. Labels modified0, automatic relabels0; cần human approval policy với version/SHA/reviewer/time/evidence trước mọi migration, rồi annotation version mới cho đúng ảnh được xác định. Không buộc review lại toàn bộ pool theo giả định.

## Scene, lineage và tính phù hợp subset

VN75suspected pairs vẫn unresolved; D-Fire1confirmed overlap cùng camera/scene bằng evidence trước,750unresolved,0confirmed independent. Đã giữ parent scene-group manifest/contact sheets và test reservations. Shortlist mới **627 cặp source đối lập** trong snapshot hiện tại, xếp impact/SHA, có source paths/video IDs; file human_source_shortlist_current.json thay cho việc coi shortlist120 cặp cũ là số hiện tại. Các contact sheets cũ vẫn là evidence hợp lệ cho scope cũ; nguồn mới chưa có visual decision thì vẫn UNKNOWN.

Full authenticated receipt scope1922ảnh: verified797/unresolved1125. Lượt này không tự nâng UNVERIFIED hoặc fake sourcePTS. Source VN_NEG_10 giữ quarantine do decode corruption; frame match không xác nhận toàn video healthy. Không khẳng định camera/scene độc lập chỉ vì sourceSHA khác nhau hay perceptual khác nhau.

Clean subset được chọn trước đánh giá theo approval+lineage+reservation+source health+scene evidence; không sử dụng metric/model prediction. Không có cặp train/val độc lập đủ evidence nên rows=[] và SPLIT_READY=false. Small-subset helper có thể mở một scope nhỏ nếu có evidence, không bắt chờ toàn corpus, nhưng suitability/sample-size requirement vẫn phải được duyệt theo protocol. Hiện không có cơ sở coi subset rỗng hoặc một cặp source bất kỳ là đủ cỡ mẫu nghiên cứu.

## Baseline và experiments

Baseline SHA f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6 kiểm lại và load được. Mapping đúng; train_args.data=data_dfire.yaml, epoch=-1. Không khôi phục được imageSHA membership, giữ BASELINE_MEMBERSHIP_NOT_VERIFIED. Không báo mAP holdout cũ là clean test performance.

Protocol baseline mới giữ initialization yolov8n.pt SHA f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 đã hash lại, dataset/splitSHA chưa điền khi chưa hợp lệ. Baseline mới chỉ dùng locked D-FireTRAIN, validation selection, exact membership/presentation logging và class mapping rõ ràng; A/B/C cùng baseline và evaluator. Helper planned membership không được coi là actual batch presentation log. Không training trước khi protocol/data/resources đủ.

Gates: `{'BBOX_INTEGRITY_READY': True, 'SPLIT_READY': False, 'VN_BBOX_READY': False, 'DFIRE_READY': False, 'RESOURCE_PILOT_READY': False, 'WEEK5_TRAINING_READY': False}`. CUDAfalse tại preflight; CPU availability không phải pilot measurement. Resource pilot NOT_RUN_NO_VALID_TRAIN_VAL; timings/VRAM/batch feasibility/throughput null. Baseline validation/three strategies không chạy: **0run,0epoch,0seed completed**. VN/D-Fire mAP, per-class AP, delta/forgetting, EventRecall/F1, delay, FAR/hour, exposure và CI mới đều **NOT_MEASURED/null**. Không thay metricTuần4 vào kết quảTuần5. C vẫn freezebackbone+20%D-FireTRAIN, denominatorV+ceil(V/4).

## TESTED

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_annotation_policy.py",
      "tests/test_scene_subset.py",
      "tests/test_domain_adaptation.py",
      "tests/test_bbox_resolution.py",
      "tests/test_bbox_integrity_recovery.py",
      "tests/test_week5_readiness.py",
      "tests/test_bbox_qa.py",
      "tests/test_gt_studio_ui.py"
    ],
    "exit_code": 0,
    "duration_seconds": 31.11517140000069,
    "log": ".tools/week5_policy_20b050e6/targeted.log",
    "tail": [
      ".........................................                          [100%]",
      "76 passed, 43 subtests passed in 29.15s"
    ],
    "log_sha256": "330c1c3ad937adf0a9f593cbd0dfe91a77d809a97ea9c223f7fb8a70d55e5297"
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
    "duration_seconds": 192.94746529999975,
    "log": ".tools/week5_policy_20b050e6/full.log",
    "tail": [
      "......................................................   [100%]",
      "422 passed, 80 subtests passed in 190.83s (0:03:10)"
    ],
    "log_sha256": "2db6c712f54daae76fba0c52a0b253471d8e6990d5e420d974941b3827305226"
  }
]

Tests bao phủ resolution và chống duplicate approval, strict export/logging, welding guidance/HUMAN_REVIEW_REQUIRED, class mapping, scene/test isolation, baseline membership, freeze/replay train-only và readiness fail-closed. Full pytest chạy thực, logs UTF-8/exit thật. Production label/GT/checkpoint không sửa. src/annotation_policy.py chưa gắn vào flow auto-label hay training; proposal chưa có human approval nên không tự migration.

## ACCEPTED và bước tiếp

ACCEPTED: integrity closure, proposal/code regression và audit artifacts có evidence. Scientific training acceptance **BLOCKED**; không tuyên bố hoàn tất Tuần5. Files inspected: bốn báo cáo integrity/readiness/scene, bbox_snapshot/resolution/service/page/export_adapter, frame_lineage, domain_adaptation, scene_subset, scripts/manifests/protocol/test tương ứng. Files modified mới: policy module/test, policy proposal, báo cáo này, metadata version mới và gitignore whitelist.

NEXT HUMAN ACTIONS: review/phê duyệt hoặc chỉnh đề xuất ontology (đặc biệt controlled flame và welding fume), cung cấp/phân xử source/camera/scene evidence theo shortlist; không phê duyệt lại5ảnh đãresolved, không review lại nhãn hợp lệ chỉ vì đổi format. Nếu có original baseline membership/initialization provenance thì cung cấp để đối chiếu. Không xem keyword hoặc tênvideo là quyết định semantic.

NEXT CODEX ACTIONS: khi evidence được phê duyệt, kiểm hash/thẩm quyền, dựng subset source/scene-disjoint có mẫu âm/hardnegative đúngontology, đánh giá cỡ mẫu, ghi membership; sau gatesPASS mới pilot, baseline validation và A/B/C/seeds theo budget đo thật. Không có lý do hợp lệ để mở training hiện tại.

Reproduction: audit source snapshot ở analysis_implementation.json; các input/parent SHA trong parents.json. Tests command/exit/log trong tests.json. Read-only integrity check: `python -c "from ground_truth_studio.config import Settings; from ground_truth_studio.bbox_snapshot import committed_reviews; print(len(committed_reviews(Settings.load().private)))"`. Không dùng lệnh này làm readiness toàn dataset.
