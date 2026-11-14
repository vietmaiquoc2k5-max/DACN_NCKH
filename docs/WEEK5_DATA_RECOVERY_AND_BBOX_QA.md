# Week 5 — Data recovery, split và human bbox QA

Bắt đầu 14/11/2026; kiểm chứng cuối 15/11/2026, Asia/Saigon. Branch **fix-4**, HEAD **5df62d8a824e9da294e30491d90a0b27b54628e2**, worktree sạch trước tác vụ, hiện có code/tests/metadata mới chưa commit. Phạm vi audit và UI đã chạy kiểm tra; dataset/training chưa đủ điều kiện. Áp dụng các skill filesystem-context và verification-before-completion trong `.agent/skills/`; không dùng số liệu ví dụ/memory cũ thay kết quả thực tế.

## Cổng nghiệm thu

| Gate | Kết quả | Nguyên nhân |
|---|---|---|
| DFIRE_READY | **PARTIAL** | Đã khôi phục author-linked mirror; còn 340 mẫu bbox invalid, grouped scope chưa nghiệm thu |
| SPLIT_READY | **BLOCKED** | 751 nhóm nghi gần trùng D-Fire, camera overlap và 75 cặp VN; scene/lineage chưa giải quyết |
| VN_BBOX_READY | **BLOCKED** | 0 human approvals, lineage chưa xác minh |
| Checkpoint SHA | **PASS** | baseline f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6 không đổi |
| Resource pilot | **BLOCKED** | Chưa có tập hợp lệ để đo pilot/budget đại diện |
| WEEK5_TRAINING_READY | **BLOCKED** | Cần cả ba gate và checkpoint/resource; không nâng BLOCKED thành PASS |

Các boolean gate hiện đều false: DFIRE_READY, SPLIT_READY, VN_BBOX_READY và WEEK5_TRAINING_READY. PARTIAL mô tả tiến độ recovery, không cấp quyền training.

## A. D-Fire/provenance

Nguồn được ngài xác nhận: https://github.com/gaia-solutions-on-demand/DFireDataset .
`D:\workspace\datasets\DFire_official_raw` có **14 file TXT**, **0 images**, **0 YOLO labels**. `dfire_train.txt` có **17.221** tên, `dfire_test.txt` có **4.306** tên, giao theo tên là **0**; không có pixels để chứng minh content/scene disjoint. Ngoài ra có 10 danh sách 5-fold, `video_test.txt` và `video_valid.txt`; đây không phải nhãn bbox. Không có archive ở raw root, thư mục cha datasets, project datasets hoặc Downloads đã kiểm. `data_dfire.yaml` vẫn trỏ `datasets/dfire` chưa có corpus.

Đã SHA256 toàn bộ 14 file trước/sau. Đây là tình trạng ban đầu tại đường dẫn ngài cung cấp; snapshot giữ trong `readiness_initial_raw_path.json`. Không đặt tên folder thành bằng chứng corpus đầy đủ.

### Archive và corpus đã khôi phục sau kiểm tra ban đầu

Metadata mạng mới truy cập thành công; OneDrive public API trả 401, bản Kaggle **được README tác giả liên kết** tải được HTTP 200 từ storage.googleapis.com. Nguồn: `https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo?datasetVersionNumber=1`, Kaggle version 1; repository tác giả commit **3890a756b224855f4f7ef8d237a70bc1834a4714**. Đây là mirror tác giả giới thiệu, không khẳng định byte-identical với archive OneDrive chưa truy cập được.

- Archive mới: `downloads\dfire_author_linked_kaggle_v1.zip`, **3,049,605,157 bytes**; SHA256 **65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83**.
- Giải nén an toàn sang `datasets\dfire_author_linked_kaggle_v1`; **43.055 files**, **21.527 ảnh + 21.527 labels + data.yaml**. Kiểm CRC khi đọc ZIP, chặn path traversal/symlink và đặt giới hạn dung lượng; không ghi đè nguồn. Tải/giải nén **231.750s**; audit nội dung **445.156s**.
- Mirror train **14.122**, validation **3.099**, test **4.306**; test reservation khớp danh sách tác giả, không có test chuyển vào train/val; toàn bộ tên trong author lists hiện diện. Không đánh đồng điều này với camera/scene independence.
- Đã SHA256 **43.055 files**, kiểm ảnh/label và xác thực hashes trước/sau; 0 missing labels, 0 orphan labels. Có **340 mẫu lỗi bbox/bounds** (train 224, val 54, test 62), không tự clip hoặc ghi đè.
- D-Fire có **0 nhóm duplicate SHA toàn corpus**, nhưng **751 nhóm cross-split suspected perceptual similarity** theo equal full-view pHash+dHash và texture threshold. Đây là screen hữu hạn, không chứng minh mọi nhóm cùng nguồn hoặc mọi near-duplicate đã được tìm hết.
- Ảnh val `AoF00008.jpg` và test `AoF06723.jpg` cùng có overlay **Camera0001**, cùng góc nhìn đồi/bãi xe: bằng chứng mẫu cần grouped review. Chưa có camera catalog đầy đủ để sửa split một cách khoa học.
- Source YAML và ba ảnh thực `AoF06978.jpg`, `AoF07718.jpg`, `AoF07738.jpg` nhất quán **0=smoke / 1=fire**. Đây là technical visual QA của AI, không phải human bbox approval mới. Mapping đã trùng baseline nên không remap hoặc sửa label nguồn. Xem ảnh local `source_preview.jpg`, `split_preview.jpg`.
- Legacy và D-Fire có **0 exact SHA overlap**, **3 perceptual candidate pairs**; xem `legacy_dfire_comparison.jpg` xác nhận ba cặp có cùng nội dung ảnh sau resize/reencode. Không suy ra toàn bộ legacy là D-Fire gốc. D-Fire–VN exact SHA overlap **0**, full cross-corpus near-duplicate screen chưa hoàn tất.

Artifacts: `archive_recovery.json`, `dfire_file_sha256.csv`, `dfire_image_manifest.csv`, `dfire_content_audit.json`, `dfire_curation.json`. Published label metadata không thay thế kiểm chứng chất lượng từng mẫu hoặc human approval VN.

Đối chiếu legacy bằng checksum: **15.162 file** giữ nguyên so với inventory đã có; legacy có **7.754 ảnh / 7.407 labels / 347 ảnh thiếu label**, **68 nhóm duplicate SHA**. Tên ảnh legacy trùng nguyên tên trong danh sách tác giả: **0**. Sau recovery đã đối chiếu bytes/perceptual và xem ba cặp cùng nội dung như ở trên; filename khác và SHA khác không đủ loại trừ shared content. Các tên `.rf.` và `data_legacy.yaml` gợi ý export Roboflow; không tự nhận toàn bộ corpus đó là D-Fire nguyên bản.

## B. Dataset version và leakage

Run mới: `runs/benchmarks/week5_domain_adaptation/recovery_20261114_5df62d8a/`.
Các CSV `candidate_train`, `candidate_val`, `reserved_test`, `quarantined` dẫn về reservation/source SHA candidate; **chưa là datasets được cấp quyền train/evaluate**. Tạo `quarantine_dispositions.csv` cho 3.121 ảnh, chỉ rõ lý do. Có **39 ảnh candidate** liên quan **75** cặp pixel similarity chưa giải quyết; toàn bộ bị loại khỏi scope evaluation được phép. Source/scene/camera chưa xác nhận không được tự chứng minh bằng tên file.

Đối với D-Fire đã phục hồi: loại toàn bộ mẫu bbox invalid, cross-split perceptual candidates, các cặp camera mẫu và shared legacy photos khỏi các candidate evaluation sets, đưa **3003 mẫu** vào `dfire_quarantined.csv`. Candidate còn **train 12409, val 2547, reserved test 3568**; đây vẫn chưa là scope đã nghiệm thu vì camera/scene chưa đủ, cần review nhóm. Không chuyển author test sang training. Chuẩn bị manifest candidate mới không có nghĩa đã có dataset READY.

Train/validation/test **được phép dùng hiện tại đều 0** do bbox/lineage/group gates chưa đạt. Không gọi các tập rỗng là SPLIT_READY. Không đổi reservation, không dùng model predictions để chia tập, không đánh giá test để chọn model. `data_legacy.yaml` có train=val và mapping 0 fire/1 smoke, ngược target 0 smoke/1 fire. Không dùng config legacy để train/select. D-Fire mapping khớp baseline nên không remap; legacy chưa được phép remap toàn corpus. Visualization trước/sau remap **không thực hiện vì không có remap được áp dụng**; đã có ảnh nguồn overlay để đối chiếu. Không tạo YAML training rỗng hoặc chưa hợp lệ.

`dataset_manifest.json` ghi BLOCKED_NOT_A_TRAINABLE_DATASET với rows=[], prior split SHA và blockers. Đây là bằng chứng chưa đủ điều kiện, không phải manifest READY.

## C. Human bbox workflow

Studio trước tác vụ có event review, chưa có bbox QA. Đã thêm **BBox QA** cho annotator/adjudicator, dùng token/role từ `ReviewRepository.actor`, không tạo account giả hay migration DB event. Module lưu dưới private storage `bbox_qa/versions/<UUID>` và `bbox_qa/exports/<UUID>`; không chạm labels gốc hoặc submitted event reviews.

Queue 1.904 ảnh và 129 ảnh ưu tiên được gộp theo image SHA: **1.922 ảnh duy nhất**, trong đó **129 ưu tiên**. Mỗi ảnh đọc đúng pixels/label với hash bindings. UI hiển thị boxes/class 0 SMOKE / 1 FIRE; canvas cho vẽ, chọn đổi class, xóa/clear; bảng cho sửa tọa độ normalized và thêm box. Label rỗng/thiếu/lỗi được hiển thị, không tự coi là negative.

Phê duyệt/chỉnh sửa/từ chối phải có full-image signoff và evidence. Empty label chỉ được APPROVE khi người thật xác nhận negative. Server kiểm class/bounds/finite và hash ảnh/label; danh tính lấy từ session, thời gian từ server. Mỗi thao tác tạo annotation version mới, label SHA và metadata SHA; không ghi đè nguồn. Bản bị từ chối mới nhất loại approval cũ khỏi export. Chỉ xem/export lịch sử của chính tài khoản.

Human approval tách khỏi lineage: không sinh PTS hoặc source SHA giả. Source SHA candidate, nếu có, được giữ riêng; source binding authoritative hiện null, `LINEAGE_NOT_VERIFIED`, `approved_for_training=false`. Export chứa actual approved reviews và labels version mới nhưng training rows vẫn rỗng khi thiếu lineage. Không dùng YOLO để approve.

**Thực tế hiện tại: 0 ảnh được người thật phê duyệt; 1.922 ảnh còn chờ; 0 nhãn thật do agent tạo; chưa có approved bbox export thật.** Không mượn synthetic test records làm approvals. Module đã kiểm bằng AppTest/Node synthetic, chưa có human/manual browser acceptance trên ảnh thật, nên **Bbox QA acceptance = PARTIAL**.

### Chạy và sử dụng

```powershell
python -m streamlit run ground_truth_studio/app.py --server.address 127.0.0.1 --server.port 8503
```

Dùng tài khoản Studio hiện có → **BBox QA** → xem ảnh/box → kéo vẽ hoặc sửa bảng → điền evidence, attest; nếu không có box thì xác nhận negative → phê duyệt hoặc từ chối. Cuối buổi nhấn **Xuất bbox đã phê duyệt của tôi**. Private storage/settings dùng cơ chế Studio hiện có; không đưa export chứa reviewer identity lên Git. Event review vẫn ở các trang cũ.

## D. Training readiness và baseline

Runner đã siết yêu cầu D-Fire cho cả ba strategy, cùng các gate `DFIRE_READY/SPLIT_READY/VN_BBOX_READY`, file/source/bbox hashes, không có split leakage, checkpoint SHA và resource pilot thực. Empty VN labels không thể lọt qua chỉ nhờ signoff chung nếu thiếu `negative_frame_approved=true`.

Chưa chạy baseline validation mAP hoặc training. mAP, delta, forgetting, best model đều **null**. Đã khôi phục corpus và kiểm source mapping, nhưng cần giải quyết quarantine/scene groups D-Fire, 75 cặp VN, human bbox approval và native-frame/PTS lineage; rồi đo pilot tài nguyên. Kaggle validation nằm trong author TRAIN; content-hash manifest training gốc của checkpoint chưa có, nên chưa xác thực được quan hệ với baseline training history. Không tự gọi split này độc lập. Chỉ sau đó mới tạo YAML và manifest READY đã khóa. Checkpoint/config nguồn giữ nguyên.

## Tests và giới hạn kiểm chứng

- Syntax check exit 0.
- Targeted: **64 passed, 54 subtests passed in 14.20s**, exit 0, wall **15.577s**.
- Full unittest: **Ran 362 tests in 148.354s; OK**, exit 0, wall **152.036s**; report RTSP lịch sử phục hồi đúng bytes.
- `node .tools/week5_recovery_5df62d8a/test_canvas.js`: exit 0, kiểm JS draw/edit/delete/clear/clamp trên DOM giả lập, không phải thao tác người thật trên browser.
- Ba CLI gate checks trên manifest hiện tại cho full/freeze/replay đều exit **2** theo thiết kế, **0 epoch** và không checkpoint mới; đây không phải ba training runs. Lệnh/elapsed/log thực trong `readiness_cli_checks.json`.
- Test mới kiểm auth/signoff, thiếu label/ảnh hỏng, bbox invalid, original preservation, approval/rejection/version/hash/export, negative approval và UI controls. Test domain/GT hiện có kiểm hash mismatch, class reversal, source/scene split leakage, replay train-only và không chọn model bằng test.

Đã gặp hai lỗi khi kiểm tra: Pillow bị Ultralytics patch, mở ảnh hỏng kích hoạt thử cài pi-heif và thất bại; đã đổi decode sang OpenCV, tắt auto-install trong test runner. AppTest từ chuỗi làm Windows spawn import nhầm script UI; đã thêm main guard và phục hồi `__main__.__file__` trong fixture. Giữ logs lần thất bại để đối chiếu, kết luận chỉ dùng lần verified sau sửa. Không cài dependency mới.

Log/lệnh/count/exit thực: `test_results.json`, `.tools/week5_recovery_5df62d8a/*_verified.log`. Ground Truth frozen, checkpoint, 14 split files và 33 metadata artifacts của run trước vẫn đúng SHA; xem `preservation_receipt.json`. Không khai PASS Linux/CI hoặc mAP/FPS mới từ các unit tests.

## Lệnh tiếp tục khi đủ dữ liệu

```powershell
# Chỉ dùng khi manifest thật READY, đủ ba gates và pilot có log/hash thực.
python scripts/run_week5_domain_adaptation.py --dataset-manifest READY_DATASET.json --strategy full_finetune --seed 42 --budget-receipt MEASURED_PILOT.json --output-dir runs/benchmarks/week5_domain_adaptation/NEW_FULL_RUN
# Thay strategy bằng freeze_backbone/replay_20_percent, mỗi run một thư mục mới.
```

Hiện lệnh trên chưa được phép chạy training; READY_DATASET/MEASURED_PILOT là đầu vào cần được tạo từ evidence thật, không có file giả được bàn giao. Không commit/push/xóa nguồn, thay checkpoint hoặc sửa private event DB. Không chạy script split legacy sửa/move nguồn tại chỗ.
