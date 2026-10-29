"""Module quản lý cảnh báo tập trung (`src/alert_manager.py`).

Tái sử dụng `AlertManager` hiện có ở thư mục gốc dự án để đảm bảo tính nhất quán
giữa ứng dụng Desktop (`app.py`) và Web Dashboard (`web_app.py`).
"""
from alert_manager import AlertManager

__all__ = ["AlertManager"]

