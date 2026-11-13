# Studio catalog UI filter hotfix

Áp dụng trên `ground_truth_studio/app.py` của nhánh fix-4, commit đã đọc `b6a5d086`. Công cụ sao lưu trước khi thay và từ chối áp dụng nếu nội dung không khớp.

```powershell
python .\studio_catalog_fix\apply_fix.py --repo (Get-Location).Path
python -m py_compile .\ground_truth_studio\app.py
python -m unittest discover -s tests -p "test_gt_studio_*.py" -v
python -m streamlit run ground_truth_studio/app.py --server.address 127.0.0.1 --server.port 8514
```

Bộ lọc chỉ ẩn các mục catalog có `available=False` khỏi dropdown. Không xóa mục lịch sử, không sửa manifest, không cập nhật trạng thái nhãn. Nếu kho có **143 video thật nhưng catalog không đủ**, cần đối chiếu danh mục nguồn riêng trước khi thêm video mới; bản vá này không tự đưa những file chưa ghi manifest vào Studio.
