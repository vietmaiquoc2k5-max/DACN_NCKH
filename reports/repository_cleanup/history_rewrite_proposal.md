# ĐỀ XUẤT VIẾT LẠI LỊCH SỬ GIT (GIT HISTORY REWRITE PROPOSAL)
**Dự án:** DACN_NCKH — Hệ thống Phát hiện & Cảnh báo Cháy/Khói Thông minh (FireWatch)  
**Tài liệu tham chiếu:** `DeCuong_NCKH_PhatHienChay_Camera.docx`  
**Ngày lập đề xuất:** 17/11/2026  
**Trạng thái đề xuất:** `PROPOSED_NOT_ACTIVE` — CHỈ MANG TÍNH TƯ VẤN KỸ THUẬT, TUYỆT ĐỐI KHÔNG TỰ ĐỘNG THỰC THI.

---

## 1. NGUYÊN NHÂN DUNG LƯỢNG PACKFILE ĐẠT 1.91 GiB

Mặc dù việc thực thi hủy theo dõi tệp (`git rm --cached`) ở nhánh hiện tại sẽ giúp working tree giảm mạnh dung lượng từ 1.991 MB xuống ~1.076 MB, cơ sở dữ liệu đối tượng Git (`.git/objects/pack`) **vẫn giữ nguyên 1.91 GiB** (5 pack files, 33.214 đối tượng Git).
Điều này là do bản chất của hệ thống Git: các tệp nhị phân lớn đã từng được commit trong các commit cũ trong quá khứ vẫn tồn tại vĩnh viễn trong đồ thị commit (commit graph) để đảm bảo khả năng checkout lại bất kỳ commit nào.

---

## 2. TOP 20 ĐỐI TƯỢNG BLOB LỚN NHẤT TRONG LỊCH SỬ REPOSITORY

Dữ liệu trích xuất từ `git verify-pack` và `audit_summary.json`:

| STT | SHA Đối tượng Git | Dung lượng (MB) | Tên tệp / Đường dẫn ban đầu | Phân loại |
|---|---|---|---|---|
| 1 | `435d2bb954b9d4a2e9c87001b010604f30d5898b` | 96,44 MB | `uploads/VN_POS_20.mp4` | Video MP4 |
| 2 | `46f2f6955f7e37c19e8c30237112da87168d76b4` | 70,77 MB | `FireSmokeLabeler/videos/testpos03.819.avi` | Video AVI |
| 3 | `a6c24797c9872d5fff2eeffa2e61d94095d36bb8` | 37,59 MB | `uploads/VN_POS_15.mp4` | Video MP4 |
| 4 | `e74f8c8ea659168a32d0294aa8243278fb67f93c` | 36,71 MB | `uploads/VN_POS_11.mp4` | Video MP4 |
| 5 | `671cc792d02b876607483fb58902621754368741` | 33,75 MB | `uploads/VN_POS_06.mp4` | Video MP4 |
| 6 | `ce72c5b2e99ac8943cb7056986fd460391f3a01b` | 24,78 MB | `uploads/VN_POS_10.mp4` | Video MP4 |
| 7 | `ffaae4e513493793769c18e4eb8ae4ed86cb650c` | 24,04 MB | `FireSmokeLabeler/videos/testpos04.820.avi` | Video AVI |
| 8 | `67282e0e73ba9dcb74944010b1a327890d95bf17` | 23,36 MB | `runs/detect/runs/detect/train_fire-2/weights/last.pt` | Model Checkpoint |
| 9 | `9a57bece8dd48afabb7fbc87f9a80f905d5c6143` | 23,36 MB | `runs/detect/runs/detect/train_fire-2/weights/best.pt` | Model Checkpoint |
| 10 | `d47d035e948a581108e7e4d434872ed945429bc3` | 15,19 MB | `uploads/VN_POS_07.mp4` | Video MP4 |
| 11 | `d7fe21082e340d0883f1f935a573f82e91bdf58c` | 14,91 MB | `FireSmokeLabeler/videos/testpos10.826.avi` | Video AVI |
| 12 | `67438b82a303313eb7d04801cdab8e6209776288` | 13,18 MB | `FireSmokeLabeler/videos/posVideo9.878.avi` | Video AVI |
| 13 | `1a05f18b45c2f35790ea5c636e761d25828ebf30` | 12,64 MB | `uploads/VN_POS_04.mp4` | Video MP4 |
| 14 | `e469b227638ebf3af887a7eebfdf87250b7d711d` | 11,11 MB | `runs/benchmarks/.../VN_NEG_47/timeline.stdout` | Trace stdout |
| 15 | `81c68cff6d2a78e014cf4d0d33ec92aa53f73764` | 10,52 MB | `FireSmokeLabeler/videos/testpos02.818.avi` | Video AVI |
| 16 | `1b4d9375bdf960e0b0222be294645a8d6750cc66` | 10,31 MB | `FireSmokeLabeler/videos/posVideo8.877.avi` | Video AVI |
| 17 | `8a63bd280630a7139534b48c773c364ead536c5c` | 10,21 MB | `handoff/week5/dataset_manifest.json` | Manifest JSON |
| 18 | `4b0a7b4e2a72c0992335edb319833941aac7cf60` | 9,76 MB | `runs/benchmarks/.../dfire_train_manifest.json` | Manifest JSON |
| 19 | `74616841a0653db75d63a9bafc2bf22f449a0b4b` | 8,81 MB | `FireSmokeLabeler/videos/posVideo5.874.avi` | Video AVI |
| 20 | `7ff43a5741fcc08928c390d5332eed793d0cf926` | 8,26 MB | `FireSmokeLabeler/videos/testpos09.825.avi` | Video AVI |

---

## 3. PHÂN TÍCH RỦI RO KHI VIẾT LẠI LỊCH SỬ GIT

> [!WARNING]
> Viết lại lịch sử Git là một hành động phá vỡ tính tương thích ngược nghiêm trọng (Breaking Change) đối với toàn bộ các thành viên và bản sao lưu của repository:
> 1. **Thay đổi toàn bộ mã băm Commit SHA:** Mọi commit từ thời điểm tệp lớn đầu tiên xuất hiện sẽ bị thay đổi SHA. Tất cả các tài liệu, báo cáo, và tag trỏ tới commit cũ (ví dụ: commit nền Tuần 4 `5aaa25e`, `280cd63`, `e33b66b`, v.v.) sẽ không còn khớp.
> 2. **Xung đột phân nhánh (Divergent Branches):** Mọi bản sao repo trên máy các thành viên khác hoặc trên GitHub (`main`, `fix`, `fix-4`) sẽ bị lệch hoàn toàn, buộc phải dùng `git push --force`.
> 3. **Nguy cơ mất dữ liệu:** Nếu script viết lại lịch sử lọc nhầm tệp (ví dụ: nhầm các checkpoint 3-seed MT2 hoặc tài liệu NCKH), dữ liệu sẽ biến mất khỏi toàn bộ lịch sử Git.

Do các rủi ro trên, **Antigravity TUYỆT ĐỐI KHÔNG THỰC HIỆN việc viết lại lịch sử Git trong phiên làm việc này**.

---

## 4. QUY TRÌNH KỸ THUẬT TIÊU CHUẨN KHI ĐƯỢC PHÊ DUYỆT TRONG TƯƠNG LAI

Nếu nhóm nghiên cứu và Giảng viên hướng dẫn quyết định viết lại lịch sử để giảm dung lượng `.git` từ 1.91 GiB xuống < 100 MB, quy trình bắt buộc phải tuân theo 6 bước an toàn sau:

### Bước 1: Sao lưu toàn diện (Mirror Backup)
```powershell
# Tạo một bản sao lưu toàn bộ các nhánh, tags và reflog
git clone --mirror . ../DACN_NCKH_FULL_BACKUP_20261117.git
```

### Bước 2: Cài đặt và sử dụng `git-filter-repo` (Khuyến nghị chính thức của Git)
Không sử dụng `git filter-branch` (chậm và dễ lỗi) hoặc BFG (ít linh hoạt). Sử dụng công cụ chuẩn:
```powershell
pip install git-filter-repo
```

### Bước 3: Lọc bỏ các tệp nhị phân lớn và thư mục legacy khỏi lịch sử
```powershell
git filter-repo --path uploads/ --invert-paths
git filter-repo --path FireSmokeLabeler/videos/ --invert-paths
git filter-repo --path FireSmokeLabeler/dataset/ --invert-paths
git filter-repo --path dataset/ --invert-paths
git filter-repo --strip-blobs-bigger-than 20M
```

### Bước 4: Dọn rác và nén chặt packfile
```powershell
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Bước 5: Kiểm tra toàn diện chất lượng và tính toàn vẹn
```powershell
pytest -q
git log -n 5
git count-objects -vH
```

### Bước 6: Thông báo cho toàn bộ thành viên và force-push đồng bộ
Chỉ thực hiện khi tất cả thành viên trong nhóm đã commit xong công việc hiện tại và sẵn sàng re-clone repository mới.

