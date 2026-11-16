# Week 5 — final BBox QA snapshot and dataset readiness

Ngày16/11/2026. Branch `fix-4`; HEAD `57d1b82665c7f826f09c0a4ae44ab6dfaa152c74`. Worktree trước sửa sạch; hiện có code/audit/test/report chưa commit. Không commit/push/xóa, không relabel, không thay frozenGT/checkpoint, không training.

## 1. Kết quả review thật

Queue **1922 unique imageSHA**. Snapshot valid history **1943 versions**, latest reviewed1922: **1919 APPROVE /3 REJECT /0 PENDING**.21 extra versions là lịch sử, không cộng trùng approval.5original corrupt versions giữ nguyên,5resolution receipts hợp lệ,0unresolved quarantine; strict integrityPASS. Raw snapshot giữ cả forensic originals và resolutions, SQLite backup integrityok. Không tự tạo annotation/approval mới.

Snapshot mới không ghi đè: `.tools/week5_final_bbox_20261116/snapshot/`; manifest SHA256 `aece9539d87ab97caa2acd4c12333cebb94434df3fad33d3975eb2809fb23fae`. Tất cả snapshot files kiểm checksum trước/sau; không restore production. Snapshot đã chốt là version mới có hashes, không phải khóa storage live. Hash/checksum là cơ chế phát hiện thay đổi, không phải WORM chống chủ máy sửa cả bytes lẫn hash.

## 2. Nhãn và phân bố

| Scope APPROVE | Ảnh |
|---|---:|
| Human-confirmed negative |1258|
| SMOKE only |256|
| FIRE only |102|
| Both |303|

BboxSMOKE675, bboxFIRE874. Không trộn số bbox với số ảnh. Metadata/label checksum, reviewer/time/evidence và negative attestation đã kiểm. Original labels trong mọi queue alias vẫn đúngSHA; không có silent relabel. Welding/fire/smoke policy vẫn là proposal, không áp lên labels đãduyệt. Hard-negative semantic tags chưa xác thực để trống/UNKNOWN, không tự coi1258negative là welding negatives.

| Reservation trong queue đãduyệt | APPROVE | Ghi chú |
|---|---:|---|
| Train |1376|Chưa eligible|
| Validation |525|3ảnhREJECT vẫn được giữ làREJECT|
| Reserved test |18|Không đưa vào train/selection|

Raw queue reservation total1376train/528val/18test; con số18 là test trongqueue này, không phải toàn bộ reservedtest của corpusVN.1199candidate paths ngoàiqueue1922 không được coi là đã review. Full prior video/source/test reservations vẫn giữ qua parent manifests.

## 3. Lineage toàn1922

Verified derived797; unresolved legacy1125; thiếu receipt0 trong scopequeue. SHA ảnh/nhãn và90source videos kiểm trực tiếp, frameindex binding/PTS/timebase/transform evidence kiểm từreceipt. Source identity không suy từ filename. Đây là xác minh receipts và bytes, không decode lại toàn corpus. Parent receipts có thuật toán exact decoded-frame→documented resize/JPEG95 và nativePTS; không tự invent timestamps hoặc nângUNVERIFIED.

VN_NEG_10 sourceSHA3ed5b015393ce74d1da992dece66ce4a4b7abcd4b77ce5d3d23264794c4eacbe tiếp tục bị loại vìcorruption. Frame khớp không xác nhận toànvideohealthy. Mọi ảnh nguồn này được ghi SOURCE_CORRUPTED trong exclusion manifest;1125unresolved không vào eligibleGT.

## 4. Scene và subset

Đã dựng90sourceSHA groups,0same-source cross-split trongqueue. Camera/sceneID chưa có chứng cứ thì giữnull/UNKNOWN. Không coi sourceSHA khác nhau là sceneindependent. VN75suspected visualpairs unresolved; D-Fire1confirmed same-camera overlap và750unresolved, không blanket-confirm751duplicates. Reuse scene manifests/contact sheets đãhash, không dựa pHash đơn lẻ.

Shortlist cuối có **1380 quan hệ source đối lập chưa được chứng minh**, xếp impact/SHA; đây là các quan hệ cần evidence nếu dùng chung train/val, không phải 1380 near-duplicate pairs được phát hiện. Các ảnh humanREJECT/sourcecorrupt/reservedtest/lineageunverified bị loại theo rule cố định; không dùng metrics/test để chọnsubset. Có thể dựng subset nhỏ khi evidence đủ, không bắt duyệt hết mọi quan hệ trước.

**VN eligible train/validation/test=0/0/0; D-Fire eligible=0/0/0. SPLIT_READY=false.** Unknown scene relations chưa giải quyết, không cấp sceneID giả để mởgate. Empty eligible scope không phải bằng chứng khoa học đạt mục tiêu; không tuyên bố cỡ mẫu đạt đềcương.

`candidate_provenance_manifest.json` có từngimageSHA, annotationversion/metadataSHA, humanlabelSHA/path, original label/aliasSHA, sourcevideoSHA/path/PTS/frameindex, sceneunknown, splitreservation, classmapping, negativeattestation/hard-negative status, parentSHA và exclusionreasons. `source_groups.json`, `human_source_shortlist_final.json`, `clean_vn_manifest.json` và preservedD-Fire manifest bổ sung traceability. Manifest rows dùng cho audit/candidate, không được đổi tên thànhREADY.

## 5. Baseline/pilot

Baseline checkpoint SHA f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6 và mapping0SMOKE/1FIRE giữ nguyên; không tìm thêm image-SHA trainmembership trong training artifacts/dataset paths đãtìm. BASELINE_MEMBERSHIP_NOT_VERIFIED giữ nguyên, không báoclean test performance/forgetting từholdoutcũ.

Protocol baseline mới lưu initializationSHA f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 kiểm lại, chỉ lockedD-FireTRAIN, exactmembership/presentation logging, validation-onlyselection và testseal. Split chưaREADY nên chưarun. A/B/C phải cùngbaseline; Cfreeze backbone+20%D-FireTRAIN replay.

Gates: `{'BBOX_REVIEW_COMPLETE': True, 'BBOX_INTEGRITY_READY': True, 'SPLIT_READY': False, 'VN_BBOX_READY': False, 'DFIRE_READY': False, 'RESOURCE_PILOT_READY': False, 'WEEK5_TRAINING_READY': False, 'eligible_train': 0, 'eligible_validation': 0, 'eligible_test': 0, 'reason': 'No authenticated opposing source/scene independence decisions; UNKNOWN remains excluded', 'resource_pilot': None, 'training_runs': 0, 'epochs': 0, 'metrics': None, 'parent_readiness_sha256': 'efa815b86737da9f6f563ecb414f0509ca916b3bd3f4b00825955442303c1ca4'}`. BBOX_REVIEW_COMPLETE và BBOX_INTEGRITY_READY đạt không đồng nghĩa VN_BBOX_READY/TRAINING_READY. ResourcepilotNOT_RUN; step/epoch timing,VRAM/batch feasibility null. Training0run/0epoch; không có mAP/event/FAR/delay/CI mới.

## 6. Export và kiểm thử

Export version mới: `D:\workspace\DACN_NCKH\.tools\week5_final_bbox_20261116\approved_export`;1919approved,3rejected,0pending; trainingeligible0. JSONL/YOLOlabels/provenance/checksums kiểm lại, không overwritebundlecũ. Exportchecksum manifestSHA256 `a033eb8144b71683466316ac01dbc6b17108f13b10f6474f697995acd600543b`. PrivateSQLite/reviewer metadata/labels ở ignored.tools, không đưaGit mặcđịnh.

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_bbox_final_audit.py",
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
    "duration_seconds": 28.56363169999895,
    "log": ".tools/week5_final_bbox_20261116/targeted.log",
    "tail": [
      "............................................                       [100%]",
      "79 passed, 43 subtests passed in 26.86s"
    ],
    "log_sha256": "0690d3cfaa88fd6c31b3b12a727b25ec255f6ed0aea4267682e9d22373a6b1f9"
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
    "duration_seconds": 182.07409709999774,
    "log": ".tools/week5_final_bbox_20261116/full.log",
    "tail": [
      ".                                                                        [100%]",
      "425 passed, 80 subtests passed in 180.49s (0:03:00)"
    ],
    "log_sha256": "c66368628b1bf4b329127988307ea7a576bfd4ded1c1af21c8c630fd68882df6"
  }
]

Tests mới kiểm1922-style accounting, duplicatehistory, latestREJECT, pending/unknownID/decisionfailclosed; targeted/full còn kiểm integrity/resolution/label/lineage/scene/test/replay/gates. Logs SHA ghi thật, không dùng test mới viết làmPASS trướcchạy. Reproduce: `python scripts/audit_final_bbox_qa.py --snapshot .tools/week5_final_bbox_20261116/snapshot --output NEW_VERSION_OUTPUT --export NEW_PRIVATE_EXPORT`; dùng output mới. Audit script thực hiện export annotation, khôngtraining. Targeted/fullcommands trongtests.json. Metadata augmentation source được lưu tronganalysis_implementation.json.

## 7. Việc tiếp theo

Ngài không cần làm lại1922BBoxQA hoặc5reconfirmations. Cần source/camera/scene evidence cho shortlist ưu tiên và extraction/anonymization provenance cho1125unresolved; phê duyệt ontology nếu muốn đổi quyước.3REJECT được giữ loại, chỉ xem lại khi người thật chủ động có evidence mới. Nếu còn originalbaseline imageSHA manifest thì đốichiếu, nếukhông dùng protocolbaseline mới.

Tôi tiếp tục: verify evidence và tạo source/scene-disjoint subset có đủ lớp/negative samples, kiểm cỡmẫu/provenance/testisolation; sau đó mới measuredpilot, baseline vàA/B/C theoBudget. Hiện **reviewcompletionACCEPTED**, scientific dataset/training **PARTIAL/BLOCKED**, không báo dựánhoànthành.
