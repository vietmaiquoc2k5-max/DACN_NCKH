# Export contract — candidate only

Mọi file được tạo trong private `exports/<uuid>/`; không ghi vào `data/ground_truth/` và không ghi đè CSV nguồn. Chưa có export production khi chưa có người dùng thực sự thực hiện review; các bundle trong tests là fixture tổng hợp.

## CSV tương thích hiện có

| File | Các trường bắt buộc |
|---|---|
| positive_annotator_02.csv và positive_annotator_01.csv | video_id, relative_path, source_sha256, reviewer_id, reviewed_at_utc, annotation_events_json, evidence_ref, signoff |
| positive_adjudication.csv | gt_index, gt_snapshot_sha256, video_id, label, start_sec, end_sec, reviewer_02_event_index, approval, adjudicator_id, adjudicated_at_utc, evidence_ref, notes |
| negative_annotator_01.csv và negative_annotator_02.csv | video_id, relative_path, source_sha256, reviewer_id, reviewed_at_utc, decision, fire_intervals_json, smoke_intervals_json, evidence_ref, signoff |
| negative_adjudication.csv | video_id, source_sha256, approval, adjudicator_id, adjudicated_at_utc, evidence_ref, notes |

Negative decision mapping: VERIFIED_NEGATIVE → NEGATIVE, CONTAINS_FIRE → FIRE, CONTAINS_SMOKE → SMOKE, UNCERTAIN → UNCERTAIN. `uncertain_intervals_json` được thêm như cột tùy chọn. Interval JSON dùng object start_sec/end_sec và giữ thêm event/frame/source provenance. Dữ liệu imported chưa submit không biến thành signoff mới.

Legacy positive approval chỉ APPROVE khi human ACCEPT, original GT event còn trong final events và ghép hợp lệ với reviewer 02. Event mới nằm trong candidate JSONL, không tự tạo `gt_index` giả cho snapshot cũ. Legacy negative approval chỉ APPROVE cho negative consensus không positive/uncertain interval; vẫn cần gate kỹ thuật từ recovery. Không gọi row approval là OFFICIAL GT.

Các cột mở rộng ở CSV nguồn được giữ lại. Cell có thể chạy công thức spreadsheet được escape bằng apostrophe trong CSV xuất; raw original vẫn được bảo toàn trong `external_csv_snapshots.json`. Không eval CSV/JSON hoặc cho phép người dùng chọn tên file xuất tùy ý.

## verified_gt_candidate.jsonl

Mỗi event có `video_id`, `camera_id`, stable `event_id`, FIRE/SMOKE, start_sec/end_sec, frame_start/frame_end, timestamp_provenance, source_sha256, annotator_01/02, annotator_id, reviewer_id của adjudicator, reviewed_at_utc, evidence_ref, adjudication_id, review_session_ids.

Luôn `annotation_status=HUMAN_ADJUDICATED_CANDIDATE`, `verification_status=pending_verification`; loader dự án đọc được khi `only_verified=False` và loại khi `only_verified=True`. Candidate filename không làm nó trở thành verified GT. Mọi bước chuyển version/approval phải thực hiện riêng, có người chịu trách nhiệm.

## Provenance và summary

`ground_truth_provenance.json`: source commit khi có Git, snapshot hash, danh tính/role/identity-evidence references do operator đăng ký, reviewer session IDs, adjudication IDs, timestamp export, backup path, SHA-256 các file, official_ground_truth=false và operating_point_promoted=false. `ground_truth_review_summary.json` tách claim lịch sử, tiến độ trong Studio, semantic consensus và eligible negative exposure.

SQLite giữ toàn bộ source hash/duration provenance, phiên locked và các phiên REQUIRE_REVIEW mới, audit log login/draft/submit/adjudication/export cùng snapshots CSV nhiều phiên bản. Các ID và identity references là private; không commit bundle hoặc DB lên repo.
