# Audit dataset được ngài chỉ định — 14/11/2026

## 1. Trạng thái dự án

Branch `fix-4`, HEAD `730342623b5946f0eb9d25241eb4b8fc20488e79`; worktree có thay đổi chưa commit. Áp dụng [filesystem-context](../.agent/skills/filesystem-context/SKILL.md) và [verification-before-completion](../.agent/skills/verification-before-completion/SKILL.md). Audit local **DONE**; danh tính corpus D-Fire **NOT VERIFIED**; training tuần 5 **BLOCKED**.

## 2. Mục tiêu

Kiểm tra `D:\workspace\DACN_NCKH\dataset` theo đường dẫn ngài cung cấp. Ngài xác nhận link nguồn là [repository D-Fire của tác giả](https://github.com/gaia-solutions-on-demand/DFireDataset). README tác giả mô tả 21.527 ảnh và cung cấp đường tải ảnh/nhãn cùng split. Link đúng không tự chứng minh các file trong một thư mục local là corpus nguyên bản hoặc một subset đã xác thực. Không suy danh tính dataset chỉ từ số ảnh/tên file.

## 3. Đã kiểm chứng

Đã hash trước/sau **15162 file**, đọc **7754 ảnh**, kiểm **7407 file YOLO**, thời gian audit **125.269s**, lệnh audit exit 0. Không thay bytes nguồn, không parse/execute binary label cache, không sửa nhãn.

## 4. Phần chưa hoàn tất

Chưa xác thực corpus local với manifest/hash nguồn tải của tác giả. Chưa tải archive gốc. Ngài xác nhận chưa có bbox export riêng cho ảnh VN; 79 video có event GT vẫn không tự trở thành bbox GT. Chưa baseline mAP mới, training, delta mAP hoặc đo quên; mọi metrics đó tiếp tục null.

## 5. Kết quả thực tế

| Hạng mục | Kết quả |
|---|---:|
| Ảnh `dataset/images/train/images` | 7.407 |
| Nhãn tương ứng `dataset/images/train/labels` | 7.407 |
| Ảnh tại `dataset/images` không có nhãn đi kèm | 347 |
| Ảnh không đọc được / label lỗi / label orphan | 0 / 0 / 0 |
| Label rỗng | 0 |
| Boxes class ID 0 / 1 | 10.957 / 2.442 |
| Nhóm ảnh trùng SHA | 68 |
| SHA của ảnh không nhãn cũng xuất hiện ở ảnh có nhãn | 20 |
| Ảnh có hậu tố `.rf.` | 7754 |
| Nhóm tên gốc lặp, chỉ là giả thuyết filename | 1892 |

`data_legacy.yaml` khai báo `0=fire,1=smoke`, ngược `data_dfire.yaml` và baseline `0=smoke,1=fire`. Đây là mapping được cấu hình khai báo; chưa kiểm chứng semantic từng box. Cả `train` và `val` của legacy đều là `images/train/images`: evaluation theo config đó trùng training scope. Không chuyển 347 ảnh thiếu nhãn thành negative, không chọn chúng làm test độc lập, không copy/remap label tự động. Hậu tố `.rf.` và comment Roboflow trong config gợi ý export legacy; chưa đủ kết luận mọi ảnh không có liên hệ với D-Fire.

Syntax/import checks exit 0. Targeted pytest: **33 passed, 48 subtests passed in 5.74s**, wall **7.348s**, exit 0. Full unittest: **Ran 353 tests in 159.683s; OK**, wall **164.225s**, exit 0. Report RTSP cũ được khôi phục đúng bytes. Không thay pipeline detector, nên inference/FPS/mAP mới không áp dụng cho audit này và không được tạo số giả để điền checklist. 33 metadata artifacts của hai run bàn giao trước vẫn đúng SHA.

## 6. Blockers, ảnh hưởng và cách sửa

- **NOT VERIFIED — provenance:** không có README/YAML/manifest nguồn trong thư mục này; chỉ có ảnh, nhãn, cache. Cần đối chiếu archive/split gốc từ repository tác giả và version/hash; không ghi đè bộ đang có.
- **BLOCKED — split/class mapping:** train=val và ID ngược baseline khiến mAP/replay không thể được nghiệm thu theo config hiện tại. Chỉ tạo dataset version/remapping mới sau khi xác minh semantic mapping và nhóm nguồn; giữ bản gốc.
- **BLOCKED — VN bbox:** event intervals khác bbox. Tiếp tục dùng bbox/129-empty-label review queues đã bàn giao; không yêu cầu làm lại toàn bộ 79 event reviews.
- **BLOCKED — tải mạng:** socket local bị sandbox chặn. Lượt xin quyền chỉ đọc metadata công khai bị dừng vì hệ thống automatic approval review gặp giới hạn sử dụng; thao tác không được thực thi, không phải kết luận nó không an toàn. Không né bước xét duyệt. Cần khôi phục xét duyệt hoặc archive được đặt sẵn trên máy để tiếp tục khôi phục corpus.

## 7. File và bằng chứng

- [candidate_audit.json](../runs/benchmarks/week5_domain_adaptation/dataset_candidate_20261114_73034262/candidate_audit.json): counts, duplicates, config SHA và blockers.
- [image_label_inventory.csv](../runs/benchmarks/week5_domain_adaptation/dataset_candidate_20261114_73034262/image_label_inventory.csv): binding ảnh/nhãn hiện có, không cấp quyền training.
- `source_files.csv`: snapshot SHA của toàn bộ file; không chứa raw ảnh.
- [test_results.json](../runs/benchmarks/week5_domain_adaptation/dataset_candidate_20261114_73034262/test_results.json): lệnh/log/exit/thời gian thực.
- Raw logs local `.tools/week5_candidate_73034262/candidate_audit.log`, `targeted.log`, `full.log`; không đưa raw data/cache lên Git.

## 8. Bước tiếp theo

Khi có archive gốc hoặc khả năng xét duyệt mạng được khôi phục: kiểm kích thước/dung lượng, tải vào vị trí mới, kiểm SHA, giải nén an toàn sang version mới, audit nguồn/split/classes/duplicates trước baseline validation. VN training vẫn cần bbox provenance hợp lệ. Không chạy `split_dfire_val_test.py` cũ trên nguồn vì script đó sửa/di chuyển label và file tại chỗ.

Lệnh đã chạy:

```powershell
python scripts/audit_week5_dataset_candidate.py --candidate-root dataset --output-dir runs/benchmarks/week5_domain_adaptation/dataset_candidate_20261114_73034262
python -m pytest -q tests/test_dataset_candidate_audit.py tests/test_domain_adaptation.py tests/test_week45_acceptance.py
python -m unittest discover -s tests -p test_*.py -v
```

Chưa commit/push, xóa, remap labels, sửa bundle/DB/GT hoặc huấn luyện.
