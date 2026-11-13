# Bắt đầu annotation VN_POS trong Studio

Danh sách hiện tại đã hash provisional: `24e3064daefd0d108da179caeeb35223966daec586321a6edd401f97c6513d08`. **Không phải test độc lập đã xác minh.** Có thể bắt đầu review nguồn hợp lệ trong khi tiếp tục xác minh history/scene; không chạy detector/test hoặc chọn OP từ chúng.

```powershell
python -m streamlit run ground_truth_studio/app.py --server.address 127.0.0.1 --server.port 8514
```

1. Đăng nhập tài khoản annotator thật hiện có; không tạo tài khoản B/adjudicator giả. Chọn **VN_POS Candidate Review**. UI có đủ80 catalog entries; VN_POS_28 không đạt full decode, giữ ngoài scope nghiệm thu và không giả lập review hoàn chỉnh.
2. Xem split/group/readiness chỉ là metadata, không có predictions/onset cũ. Ưu tiên 13 nguồn test-reserved sau đây, rồi11 validation, rồi27 train candidates. 29 blocked/unassigned cần giải quyết nguyên nhân scene/source/technical trước khi release split. Annotation không tự xóa cảnh báo independence.

`VN_POS_09, VN_POS_10, VN_POS_19, VN_POS_20, VN_POS_70, VN_POS_71, VN_POS_74, VN_POS_75, VN_POS_76, VN_POS_77, VN_POS_78, VN_POS_79, VN_POS_80`

3. Mở/tiếp tục phiên. Nhấn **Bật trình phát video** khi cần. Dùng Previous/Next/Frame index và seek; Studio dùng source PTS, đã đổi raw MP4 player clock với PTS đầu khác0. Mark Start/End trong player được quy về đúng source frame. Player chỉ mở do thao tác của ngài, không inference nền.
4. Xem toàn bộ **video gốc**, tự đánh dấu FIRE/SMOKE onset/end bằng frame; chọn UNCERTAIN cho đoạn chưa kết luận, không nhìn nhãn AI cũ để đoán. Không có cháy/khói theo quan sát thì ghi nhận rõ theo protocol/source review, không tự đổi category bằng tên file. Với nguồn ghép nhiều cảnh, ghi đủ evidence và cần scene grouping phù hợp.
5. **Save Draft** để tiếp tục; **Submit Review** chỉ khi người thật đã xem đủ, kiểm source/onset/offset, evidence reference và attest đúng thực tế. Không đánh dấu independent nếu đã xem prior labels/predictions. Nếu hai người cùng xem/thống nhất, ghi đúng JOINT_HUMAN_CONSENSUS, không mô tả thành hai lượt độc lập. Quy trình independent A/B cũ vẫn có nếu thực hiện thật, không là điều kiện bắt đầu task này.
6. Submit được khóa trong private DB, chưa phải GT OFFICIAL; không ghi đè frozen GT. Không sao chép submission tự động giữa duplicate aliases50/51, không đếm chúng thành hai nguồn độc lập. Có thể gán theo đợt.

Trước khi chốt groups: mở [index contact sheets có ngữ cảnh](../runs/benchmarks/vn_pos_preannotation/run_20261114_6942fa56/analysis_v1/scene_contact_sheet_context_index.json), so đoạn không có logo/outro. Một match logo ANTV có thể không liên quan footage; mọi edge hiện vẫn nghi vấn. Ghi kết luận scene của người thật vào **bản sao/version review mới**, kèm video IDs, source hashes, người review thật/thời điểm/evidence và accept/reject relationship. Không sửa thẳng data/vn_pos_* đã hash; nếu đổi nhóm/coverage, công bố manifest version mới, giữ prior SHA và lý do, không dựa model/test performance.

Metadata nguồn/camera hiện chưa có; 13 test-reserved cũng đều TEST_INDEPENDENCE_NOT_VERIFIED. Dữ liệu có dấu vết AI/evaluation lịch sử theoID, training original hashes chưa có. Ground Truth mới không tự “làm sạch” prior exposure. Cần test NEG mới >=5h để đo FAR độc lập và OP khóa từ validation trước test. [Báo cáo đầy đủ](VN_POS_PREANNOTATION_AUDIT_REPORT.md).
