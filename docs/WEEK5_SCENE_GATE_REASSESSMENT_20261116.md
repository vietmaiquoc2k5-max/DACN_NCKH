# Week 5 — điều chỉnh cách đánh giá source/scene independence

Branch fix-4, HEAD6427859c002be9e4f08400154e3ca67dc385c810; worktree ban đầu sạch. Không sửa protocolv2, traininggates, nhãn, eventGT, testreservations hoặc dữ liệu gốc; không commit/push. Proposalv3 chỉ PROPOSED_NOT_ACTIVE.

## Yêu cầu gốc và điều kiện bổ sung

Đã đọc trực tiếp `DeCuong_NCKH_PhatHienChay_Camera.docx` và `.agents/rules/de_cuong_nckh.md`. Đề cương yêu cầu tổng≥1000frameVN gán nhãn, chia60/20/20 theo video, không frame cùngvideo ởhai tập;3chiếnlược vớiCfreezebackbone+replay20%D-Fire, validationselection vàđánhgiáVN/D-Fire. Không thấy yêu cầu bắtbuộc GPS/URL/cameraID gốc. ≥1000 là chỉtiêu datasetVN toàn đềtài, không phải yêu cầu1000ảnh cho từng tập train/val; dataset nhỏ đểpilot không tự đápứng kếtluận đềcương.

Protocolv2 bổ sung hashbinding annotation/image/source, sourcePTSlineage, lockedsplit vàresolvedleakage. `build_subset` thực tế **không kiểm GPS/URL/cameraID**. `validate_scene_decision` đã chấp nhận VISUAL_SCENE_REVIEW; các regression tests mới chứngminh một cặp có evidence độc lập có thể được chọn khi không có ba trường metadata đó. Bottleneck là tậpindependent_pairs rỗng và việc chưa ghi nhận decision đủphạmvi, không phải một câu lệnh bắt buộc GPS.

Các lượt trước đã diễn đạt thiếu metadata như một điều kiện quá tuyệt đối. Điều đó cần sửa trong cách đánh giá evidence: metadata chỉ là corroboration. Không cần yêu cầu người dùng tìmGPS/URL cho mọi nguồn. Tuy nhiên không suy UNKNOWN→INDEPENDENT chỉ để mởtraining, hoặc coi no-matchscreen là proof tuyệtđối.

## Kiểm tra footage đã chạy

Decode tuần tự toànfile:

| Source | Frames decoded | Samples cách2giây | Vai trò |
|---|---:|---:|---|
| VN_POS_05 |2594|44|Train reservation|
| VN_POS_08 |321|6|Validation reservation|
| VN_POS_13 |9072|152|Train reservation|

SHA nguồn trước/sau khớp; actualframes khớp metadata count đãprobe. OpenCVretFalse riêng không được gọi là independent clean-EOF receipt; technical receipts trước vẫnđược giữ. Khôngdecode lại toàncorpus. Source13 là camera hànhlang/cửa sổ ổn định trong overview đãxem; nguồn08 là cảnhvườn/đốngvậtliệu; nguồn05 tin tức cóintro/outro, thay góc/cảnh và cùngfootage lặp trong chínhvideo. Giữ mọi đoạn của05 cùng sourcegroup, không chia randomframe theo góc quay.

Source05↔08:264samplepairs. Source13↔08:912samplepairs. Trên1176cặp mẫu:0exactpixelSHA matches,0pHash+dHash hypotheses theo thresholds cốđịnh,0ORB-ratiotest+RANSAChomography hypotheses. SourcefileSHA cũngkhác. Đây là **no-overlap detected trong sampling scope**, không phải phủtừngframeđốichiếutừngframe hay chứngminh kháccamera/khôngcóshort reusedfootage giữa mốc2giây.

Đã xem overview của05trọn3sheet,08toàn1sheet,13đầu/giữa/cuối; không giả rằng đã kiểm thủ công từngframe của11987frames. Ghi nhận trước của ngài về12sheet vẫn giữ nguyên scope/checksum, không bắt xemlại. Scene/geometry/context cùng multiplemethods hiện **ủng hộ candidatepilot**, đủ cơ sở đềxuất reviewgating theo evidence, không phải từchối vìGPSthiếu.

## Grouping và test

75VN suspectedpairs tạo **4 uncertainty riskcomponents**, **1**component chạmtest. Đã ghi edges/sourceSHA/videoIDs/splits; chỉ gọi suspected/riskcomponent, không giảconfirmedidentity. Giữ chung hoặcquarantine toàncomponent nếuchọnscope đốilập; không chuyển test sangtrain. Không có metricstest được đọc để chọnsource.

Confirmed exact/reused/same-scene evidence phải được gomconnectedcomponent; unknown relations không được khai independent. Existing importer cótransitive conflictcheck để khôngchấpnhận A~B~C đồngthời AindependentC. D-Fire vẫn1overlap đãxácnhận/750unresolved; chưa chạy lại auditcorpus hoặc tựcho D-FireREADY.

## Đề xuất gate và số lượng

`configs/research/week5_scene_evidence_proposal_v3.json` xácđịnh rõ GPS/URL/cameraID optional. Bằngchứng thaythế gồm sourcehash, coverage decode/sampling, exact/near/reused-footage screening, geometry/context, visual observation cóphạmvi vàgroupingtestisolation. Khôngchỉ pHash/no-match, không bịa sourceID. Chưa kíchhoạt proposal vàkhông thay protocolđãkhóa, đúng yêu cầu ngài.

**Eligible thực tế theo gate hiện hành: train0/val0; pilot chưaREADY.** Các phương án kỹthuật có evidence mới, chưa cấp quyền training:

- VN_POS_13train30 ↔ VN_POS_08val55:85ảnh approved+verifiedlineage. Train:22FIRE-only,1SMOKE-only,7BOTH; val:42SMOKE-only,13negative. Validation không cóFIREGT nên không đo được FIREAP/đầyđủ so sánhper-class.
- VN_POS_05train55 ↔ VN_POS_08val55:110ảnh. Train36SMOKE-only/19negative; val42SMOKE-only/13negative. ThiếuFIRE ở cảhai phía, không đại diện taskhai lớp để kếtluậnformal.

Các số30/55 và55/55 là **candidate capacity**, không phải eligibleđãđược mở. Dataset1919approved đạt chỉtiêu sốảnh gánnhãn ≥1000 nhưng đó không tựchứngminh một subset85/110 đủ độc lập/đại diện. 60/20/20reservationgốc không đổi; không chuyểnheldouttest cho pilot.

Lýdo pilot/formaltraining chưa chạy: protocolscene evidence v2 chưa được cậpnhật/decision source chưađủphạmvi theo workflowđãkhóa; candidatevalidation chưa phủFIRE; D-Firecleantrain/val chưaREADY và membershipbaselinecũ NOT_VERIFIED. **Không phải thiếuGPS.** Có thể đềxuất PILOT_ONLY scope85/110 khi các điều kiện kỹthuật và mụcđíchpilot đượcchấpnhận; không trìnhbày như formalresearch.

## Tests và artifacts

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_scene_evidence_requirements.py",
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
    "duration_seconds": 20.934378199999628,
    "log": ".tools/week5_footage_evidence_20261116/targeted.log",
    "tail": [
      "......................                                       [100%]",
      "63 passed, 43 subtests passed in 19.58s"
    ],
    "summary": [
      "63 passed, 43 subtests passed in 19.58s"
    ],
    "log_sha256": "2c0cafb985da2d1c474bb4115020b24d9b0105534cfa1f7c1f6938cd803222c9"
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
    "duration_seconds": 174.00633740000194,
    "log": ".tools/week5_footage_evidence_20261116/full.log",
    "tail": [
      "..............                                                           [100%]",
      "438 passed, 80 subtests passed in 172.48s (0:02:52)"
    ],
    "summary": [
      "438 passed, 80 subtests passed in 172.48s (0:02:52)"
    ],
    "log_sha256": "3ca1e57f23725fd6dd860b5d25370903aab224f43aff83b85775419bfd3a7050"
  }
]

Tests mới: visual evidence không cầnGPS/URL/cameraID; selector khôngđòimetadata nhưng vẫn chặnunknownrelation; proposalv3 khôngactive. Footage script `python scripts/audit_footage_split_evidence.py` tạo `.tools/week5_footage_evidence_20261116` mới, từchốioverwrite; chạy lại phảichọnversionoutputmới trongscript. Rawframeoverviews/metrics/proposalquotes/tests/riskgroups lưu localdirectory này. Không viết decision nhân danh reviewer.

Kếtluận: có thể đánhgiá độc lập bằng footage/scene mà khôngGPS, nhưng cần protocol/evidence scope phùhợp. Đã đưa ra proposal cụthể và candidate thật, **không tự hạgate**. Ngài không cần xemlại12sheets; việc cần quyếtđịnh tiếp là cóchấpnhận proposal evidence và pilot-only scope cógiớihạn hay giữ protocolv2. Không yêu cầu bổsung GPS nhưđiềukiệnbắtbuộc.
