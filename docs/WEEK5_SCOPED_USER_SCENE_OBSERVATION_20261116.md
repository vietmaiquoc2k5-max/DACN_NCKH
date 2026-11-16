# Ghi nhận nhận xét 12 contact sheets và kiểm tra kỹ thuật bổ sung

16/11/2026; branch fix-4, HEAD965d3e81be4e98022a0b5292a40c7f36e0291cd0. Worktree đã có thay đổi deadline từ lượt trước; được giữ nguyên. Không commit/push/xóa hoặc sửa GT/annotations/receipts gốc.

## Nhận xét của ngài đã được tiếp nhận

Đã lưu nguyên văn nhận xét trực quan trong `.tools/week5_user_observation_20261116/evidence/user_observation.json`, cùng12evidence paths/SHA. Attribution là USER_IN_CURRENT_CONVERSATION; Studio reviewerID và thời điểm ngài xem thực tế đểnull vì không được cung cấp. recorded_at_utc là thời điểm công cụ ghi nhận, không giả thành reviewed_at. Không tạo signed scene decision hoặc human approval thay ngài, không yêu cầu xác nhận lại.

Nhận xét được ghi như USER_REPORTED_VISUAL_NON_OVERLAP của các ảnh đãhiểnthị. Không mở rộng thành tất cả tổ hợp nguồn hay toàn bộ footage. Một source, đặc biệt VN_POS_08, lặp lại trong nhiều sheet; vì vậy không diễn giải câu “không trùng” thành mọi24ảnh/source đều khácnhau tuyệtđối.

## Ánh xạ chính xác

| Sheet | Source IDs | Split ảnh hiển thị | Whole-source independence |
|---|---|---|---|
| 1 | VN_NEG_01 ↔ VN_POS_08 | train / val | UNKNOWN |
| 2 | VN_POS_05 ↔ VN_POS_08 | train / val | UNKNOWN |
| 3 | VN_NEG_01 ↔ VN_POS_07 | train / val | UNKNOWN |
| 4 | VN_POS_05 ↔ VN_POS_07 | train / val | UNKNOWN |
| 5 | VN_POS_13 ↔ VN_POS_08 | train / val | UNKNOWN |
| 6 | VN_POS_02 ↔ VN_POS_08 | train / val | UNKNOWN |
| 7 | VN_POS_14 ↔ VN_POS_08 | train / val | UNKNOWN |
| 8 | VN_POS_17 ↔ VN_NEG_01 | val / train | UNKNOWN |
| 9 | VN_POS_11 ↔ VN_POS_08 | train / val | UNKNOWN |
| 10 | VN_NEG_01 ↔ VN_NEG_44 | train / val | UNKNOWN |
| 11 | VN_POS_12 ↔ VN_POS_08 | train / val | UNKNOWN |
| 12 | VN_NEG_01 ↔ VN_NEG_43 | train / val | UNKNOWN |

`contact_sheet_mapping.json` giữ sourceSHA đầyđủ, imageSHA, split/reservation, lineage status, evidence checksum và ID cho từngsheet. Đã kiểm sheet/image/source hashes. Không thay sourceID hay sceneID. Không dùng lại6sheets mới thay cho12sheets ngài thực sự đã xem.

## Kiểm tra metadata và nguồn khác

Đã probe header12nguồn trong cácsheet, rồi mở rộng **90source groups** từfinalmanifest.90/90probe exit0; đây là kiểm container/streams/tags, không phải chứng nhận full decode integrity. VN_NEG_10 tiếp tụcquarantine do lỗi đãchứngminh trước.

Chỉ VN_POS_08 có location/GPS và timestampAndroid trongcontainer; sourceSHA của file này kiểm lại đạt. Toạđộ cụthể giữ trong private all_candidate_source_headers.json. VN_POS_14 có timestamp2025/reencode marker; VN_POS_11 timestamp2022; VN_NEG_43/44 có tênkênh/tựađề truyền hình; một số khác chỉ có codec/container/encoder. CácCSV scene hiện hữu có cameraID trống hoặc status SOURCE_IDENTITY_ONLY_CAMERA_UNKNOWN/SCENE_RELATION_UNCERTAIN; IDSCENE được tạo trước không phải bằngchứng độc lập.

Không có cặp GPS/định danh camera/location đãxácthực ở hai phía train/val. Ngày tạo file khác nhau không chứng minh camera khác; title/kênh/encoder không xác nhận toànbộ clip không có footage chung. MetadataGPS là chứng cứ bổ sung đáng chúý cho VN_POS_08, nhưng một đầu cóGPS không định vị/định danh được đầu cònlại. Không biến thiếu metadata thành INDEPENDENT.

## Subset và giới hạn

Đã chạy lại build_subset trên finalcandidate1922 với1919APPROVE/3REJECT, chỉ exactverifiedlineage, sourcecorrupt loại, giữtestreservation. Không có relation whole-sourceindependent đủchứngcứ đểcấp; **train0/val0**, SPLIT_READY=false. Đây là kết quả tối đa của selector **trong tập quan hệ độc lập hiện được chứng minh (rỗng)**, không phải kết luận không tồn tại subset có thể xác minh trongtươnglai.

Nhận xét trựcquan của ngài không bị bác bỏ hoặc yêu cầu làm lại; nó chỉ không chứngminh các đoạn footage/camera ngoàiảnh đãhiểnthị. Các nguồn khác đãđược rà metadata, chưa thấy cặp chứngcứ đầyđủ để thaythế.3REJECT,1125unresolvedlineage vàtest vẫn giữ loại theoprotocol. Không tạo pilot trên dữ liệu chưa đủgate. Baselineprotocol cũ giữ conditional, không córun/epoch/checkpoint/metricmới.

**Không cần ngài xem lại/nhấn xác nhận hàngloạt.** Bằngchứng còn thiếu để mởsubset là định danh nguồn/camera/địa điểm hoặc phạm vi toànfootage đángtin cậy của ít nhất một nguồntrain đốichiếu nguồnval (ưu tiên VN_POS_08 đã cóGPS), hoặc chứngcứ tươngđương cho một cặp khác. Nếu không có, trạngthái UNKNOWN giữ nguyên; không yêu cầu một lượt reviewảnh lặp lại để thay thếmetadata cònthiếu.

## Kiểm thử

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_scoped_scene_observation.py",
      "tests/test_scene_decision_import.py",
      "tests/test_bbox_final_audit.py",
      "tests/test_scene_subset.py",
      "tests/test_domain_adaptation.py",
      "tests/test_bbox_resolution.py",
      "tests/test_bbox_qa.py",
      "tests/test_gt_studio_ui.py"
    ],
    "exit_code": 0,
    "duration_seconds": 20.869687899998098,
    "log": ".tools/week5_user_observation_20261116/targeted.log",
    "tail": [
      "............................................................                                          [100%]",
      "60 passed, 43 subtests passed in 19.48s"
    ],
    "log_sha256": "e6daf923afc874bce3222a9892e3a67e648a37ae703b3c2b2c689f15d222f842",
    "summary": [
      "60 passed, 43 subtests passed in 19.48s"
    ]
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
    "duration_seconds": 171.1472732000002,
    "log": ".tools/week5_user_observation_20261116/full.log",
    "tail": [
      "...........                                                              [100%]",
      "435 passed, 80 subtests passed in 169.64s (0:02:49)"
    ],
    "log_sha256": "a1c2712be4eb7f473ad60117534146bf1a32ee15f863a21a2a0bc7cb310fec1c",
    "summary": [
      "435 passed, 80 subtests passed in 169.64s (0:02:49)"
    ]
  }
]

Testmới chứngminh userstatement giữ nguyên, không tự gán reviewer/thờiđiểm xem, không mởwhole-sourceindependence/training; thiếustatement/evidencebị từchối. Targeted/full cònbao phủ importer, conflicts, reservations, lineage, reviewcounts vàgates. Audit script: `python scripts/audit_user_scene_observation.py` tạo outputmới, từchốioverwrite; chạy lại cần versionoutputmới. Metadataheader probe source được lưu local probe_candidates.py. Khôngcó productionhuman decision đượcghi.
