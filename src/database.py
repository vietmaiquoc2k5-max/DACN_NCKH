"""Module quản lý cơ sở dữ liệu SQLite cho lịch sử sự kiện cháy/khói (`src/database.py`).

Tái sử dụng và mở rộng `AlertManager` từ `alert_manager.py`, cung cấp các truy vấn thống kê,
bộ lọc đa chiều (ngày, camera, loại đối tượng, trạng thái xác nhận) và phân tách dữ liệu thật vs demo.
"""
import os
import sqlite3
from typing import List, Dict, Any, Optional
from alert_manager import AlertManager


class EventDatabase:
    """Lớp truy xuất và thống kê dữ liệu sự kiện SQLite cho Web Dashboard."""

    def __init__(self, alert_manager: Optional[AlertManager] = None, db_path: str = "alerts.db"):
        self.alert_manager = alert_manager or AlertManager(db_path=db_path)
        self.db_path = self.alert_manager.db_path

    def record_event(
        self,
        frame,
        camera_name: str,
        label: str,
        confidence: float,
        alert_level: str = "FIRE",
        is_demo: int = 0,
    ) -> Dict[str, Any]:
        """Ghi nhận một sự kiện cảnh báo mới kèm ảnh chụp bằng chứng vào SQLite."""
        return self.alert_manager.save_alert_event(
            frame=frame,
            label=label,
            confidence=confidence,
            camera_name=camera_name,
            alert_level=alert_level,
            is_demo=is_demo,
        )

    def query_events(
        self,
        limit: int = 500,
        search_query: str = "",
        camera_filter: str = "Tất cả",
        label_filter: str = "Tất cả",
        status_filter: str = "Tất cả",
        start_date: str = "",
        end_date: str = "",
        include_demo: bool = False,
    ) -> List[Dict[str, Any]]:
        """Truy vấn danh sách sự kiện với bộ lọc theo ngày, camera, loại phát hiện và trạng thái."""
        return self.alert_manager.get_all_alerts(
            limit=limit,
            search_query=search_query,
            camera_filter=camera_filter,
            label_filter=label_filter,
            status_filter=status_filter,
            start_date=start_date,
            end_date=end_date,
            include_demo=include_demo,
        )

    def confirm_event(self, event_id: int, action_taken: str = "Đã xác nhận cháy thật", note: str = "") -> bool:
        """Đánh dấu sự kiện là đã xác nhận và lưu ghi chú xử lý."""
        return self.alert_manager.update_alert_action(
            alert_id=event_id,
            action_taken=action_taken,
            operator_note=note,
            status="ACKNOWLEDGED",
        )

    def dismiss_event(self, event_id: int, note: str = "Báo động giả / Đã bỏ qua") -> bool:
        """Đánh dấu sự kiện là báo động giả / bỏ qua."""
        return self.alert_manager.update_alert_action(
            alert_id=event_id,
            action_taken="Báo động giả (Đã bỏ qua)",
            operator_note=note,
            status="ACKNOWLEDGED",
        )

    def delete_event(self, event_id: int) -> bool:
        """Xóa một sự kiện và ảnh bằng chứng tương ứng."""
        return self.alert_manager.delete_alert(event_id)

    def clear_all(self) -> bool:
        """Xóa toàn bộ lịch sử sự kiện."""
        return self.alert_manager.clear_all_alerts()

    def export_to_csv(self, filepath: str, events: Optional[List[Dict[str, Any]]] = None):
        """Xuất danh sách sự kiện ra file CSV."""
        return self.alert_manager.export_csv(filepath, alerts_list=events)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Thống kê tổng quan số lượng sự kiện, phân biệt rõ dữ liệu thực tế và dữ liệu demo/test."""
        all_records = self.alert_manager.get_all_alerts(limit=5000, include_demo=True)
        real_records = [r for r in all_records if not r.get("is_demo")]
        demo_records = [r for r in all_records if r.get("is_demo")]

        fire_count = sum(1 for r in real_records if "FIRE" in str(r.get("label", "")).upper() or "LỬA" in str(r.get("label", "")).upper())
        smoke_count = sum(1 for r in real_records if "SMOKE" in str(r.get("label", "")).upper() or "KHÓI" in str(r.get("label", "")).upper())
        confirmed_count = sum(1 for r in real_records if r.get("status") == "ACKNOWLEDGED")
        unconfirmed_count = len(real_records) - confirmed_count

        return {
            "total_all": len(all_records),
            "total_real": len(real_records),
            "total_demo": len(demo_records),
            "real_fire": fire_count,
            "real_smoke": smoke_count,
            "confirmed": confirmed_count,
            "unconfirmed": unconfirmed_count,
            "real_records": real_records,
            "all_records": all_records,
        }

