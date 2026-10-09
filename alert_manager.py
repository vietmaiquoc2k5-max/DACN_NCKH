import os
import wave
import math
import struct
import threading
import time
import cv2
import sys
import sqlite3
import csv

# Import sound modules conditionally
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import winsound

try:
    import pygame
except ImportError:
    pygame = None

class AlertManager:
    def __init__(self, sound_path="alarm.wav", db_path="alerts.db", alerts_dir="alerts"):
        self.sound_path = sound_path
        self.db_path = db_path
        self.alerts_dir = alerts_dir
        self.alarm_playing = False
        self.mute_alarm = False
        
        # Lock for DB operations across threads
        self.db_lock = threading.Lock()
        
        # In-app notifications memory queue for live popups
        self.active_notifications = []
        self.notification_callback = None  # func(notification_dict)
        
        # Initialize pygame mixer for audio playing
        if pygame:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"Error initializing pygame mixer: {e}")
            
        # Ensure default alarm sound exists and directories/DB are set up
        self.ensure_alarm_sound()
        self.ensure_alerts_dir()
        self.init_db()

    def ensure_alerts_dir(self):
        """Ensure local directory for alert snapshots exists."""
        if not os.path.exists(self.alerts_dir):
            os.makedirs(self.alerts_dir)

    def init_db(self):
        """Initializes the SQLite database for storing in-app alert history and operator notes."""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alert_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        image_path TEXT,
                        status TEXT DEFAULT 'UNREAD',
                        action_taken TEXT DEFAULT 'Chưa xử lý',
                        operator_note TEXT DEFAULT ''
                    )
                """)
                # Migrations for existing database files
                try:
                    cursor.execute("ALTER TABLE alert_logs ADD COLUMN action_taken TEXT DEFAULT 'Chưa xử lý'")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE alert_logs ADD COLUMN operator_note TEXT DEFAULT ''")
                except Exception:
                    pass
                    
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Lỗi khi khởi tạo CSDL SQLite: {e}")

    def ensure_alarm_sound(self):
        """Generates a default alarm sound file if it doesn't exist."""
        if not os.path.exists(self.sound_path):
            print(f"Creating default alarm sound: {self.sound_path}")
            try:
                sample_rate = 44100
                duration = 1.5  # seconds
                num_samples = int(sample_rate * duration)
                
                with wave.open(self.sound_path, 'w') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    
                    for i in range(num_samples):
                        t = i / sample_rate
                        freq = 950 if int(t * 4) % 2 == 0 else 650
                        val = int(16384 * math.sin(2 * math.pi * freq * t))
                        data = struct.pack('<h', val)
                        wav_file.writeframesraw(data)
            except Exception as e:
                print(f"Failed to generate default alarm sound: {e}")

    def update_config(self, mute_alarm):
        """Update settings from GUI."""
        self.mute_alarm = mute_alarm
        if self.mute_alarm:
            self.stop_alarm()

    def play_alarm(self):
        """Starts playing the alarm sound in a loop if not muted."""
        if self.mute_alarm:
            self.stop_alarm()
            return
            
        if not self.alarm_playing:
            self.alarm_playing = True
            threading.Thread(target=self._play_loop, daemon=True).start()

    def _play_loop(self):
        """Internal audio playback loop."""
        play_path = self.sound_path
        is_temp_file = False
        temp_path = None
        try:
            if not os.path.exists(self.sound_path):
                self.ensure_alarm_sound()
                
            try:
                if os.path.exists(self.sound_path):
                    with open(self.sound_path, "rb") as f:
                        header = f.read(4)
                    if len(header) >= 4 and not header.startswith(b"RIFF"):
                        import uuid
                        import shutil
                        temp_path = f"temp_alarm_{uuid.uuid4().hex[:8]}.mp3"
                        shutil.copyfile(self.sound_path, temp_path)
                        play_path = temp_path
                        is_temp_file = True
            except Exception as ex:
                print(f"Error detecting file headers: {ex}")

            if pygame and pygame.mixer and pygame.mixer.get_init():
                pygame.mixer.music.load(play_path)
                pygame.mixer.music.play(-1)
                while self.alarm_playing and not self.mute_alarm:
                    time.sleep(0.2)
            elif IS_WINDOWS:
                winsound.PlaySound(
                    play_path, 
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                )
                while self.alarm_playing and not self.mute_alarm:
                    time.sleep(0.2)
        except Exception as e:
            print(f"Error in alarm audio loop: {e}")
        finally:
            self.alarm_playing = False
            try:
                if pygame and pygame.mixer and pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                elif IS_WINDOWS:
                    winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
                
            if is_temp_file and temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def stop_alarm(self):
        """Stops the alarm audio."""
        self.alarm_playing = False
        try:
            if pygame and pygame.mixer and pygame.mixer.get_init():
                pygame.mixer.music.stop()
            elif IS_WINDOWS:
                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def save_alert_event(self, frame, label="LỬA", confidence=0.0):
        """
        Saves alert snapshot image to disk and inserts alert record into SQLite DB.
        Triggers in-app notification popups.
        """
        self.ensure_alerts_dir()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = f"alert_{file_timestamp}_{int(confidence*100)}.jpg"
        filepath = os.path.join(self.alerts_dir, image_filename)
        
        # Save snapshot
        if frame is not None:
            try:
                cv2.imwrite(filepath, frame)
            except Exception as e:
                print(f"Lỗi khi lưu ảnh cảnh báo: {e}")
                filepath = ""
        else:
            filepath = ""

        # Insert record into database
        alert_id = None
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alert_logs (timestamp, label, confidence, image_path, status, action_taken, operator_note)
                    VALUES (?, ?, ?, ?, 'UNREAD', 'Chưa xử lý', '')
                """, (timestamp, label, float(confidence), filepath))
                conn.commit()
                alert_id = cursor.lastrowid
                conn.close()
            except Exception as e:
                print(f"Lỗi khi lưu cảnh báo vào CSDL: {e}")

        # Build notification object
        noti_item = {
            "id": alert_id,
            "timestamp": timestamp,
            "label": label,
            "confidence": confidence,
            "image_path": filepath,
            "status": "UNREAD",
            "action_taken": "Chưa xử lý",
            "operator_note": ""
        }

        self.active_notifications.append(noti_item)
        if len(self.active_notifications) > 50:
            self.active_notifications.pop(0)

        # Trigger notification callback if registered
        if self.notification_callback:
            try:
                self.notification_callback(noti_item)
            except Exception as e:
                print(f"Error in notification callback: {e}")

        return noti_item

    def update_alert_action(self, alert_id, action_taken="Báo động giả", operator_note=""):
        """Updates the incident action outcome and operator response note for an alert."""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE alert_logs 
                    SET action_taken = ?, operator_note = ?, status = 'ACKNOWLEDGED'
                    WHERE id = ?
                """, (action_taken, operator_note, alert_id))
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Lỗi cập nhật nhật ký xử lý vụ việc #{alert_id}: {e}")
                return False

    def get_all_alerts(self, limit=100, search_query=""):
        """Retrieves alert history from SQLite database including operator notes."""
        results = []
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                if search_query:
                    cursor.execute("""
                        SELECT id, timestamp, label, confidence, image_path, status, action_taken, operator_note
                        FROM alert_logs
                        WHERE label LIKE ? OR timestamp LIKE ? OR action_taken LIKE ?
                        ORDER BY id DESC LIMIT ?
                    """, (f"%{search_query}%", f"%{search_query}%", f"%{search_query}%", limit))
                else:
                    cursor.execute("""
                        SELECT id, timestamp, label, confidence, image_path, status, action_taken, operator_note
                        FROM alert_logs
                        ORDER BY id DESC LIMIT ?
                    """, (limit,))
                rows = cursor.fetchall()
                conn.close()
                for row in rows:
                    results.append({
                        "id": row[0],
                        "timestamp": row[1],
                        "label": row[2],
                        "confidence": row[3],
                        "image_path": row[4],
                        "status": row[5],
                        "action_taken": row[6] if len(row) > 6 and row[6] else "Chưa xử lý",
                        "operator_note": row[7] if len(row) > 7 and row[7] else ""
                    })
            except Exception as e:
                print(f"Lỗi truy vấn CSDL lịch sử cảnh báo: {e}")
        return results

    def delete_alert(self, alert_id):
        """Deletes a specific alert record and its image file."""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT image_path FROM alert_logs WHERE id = ?", (alert_id,))
                row = cursor.fetchone()
                if row and row[0] and os.path.exists(row[0]):
                    try:
                        os.remove(row[0])
                    except Exception:
                        pass
                cursor.execute("DELETE FROM alert_logs WHERE id = ?", (alert_id,))
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Lỗi khi xóa bản ghi cảnh báo: {e}")
                return False

    def clear_all_alerts(self):
        """Clears all alert history and removes snapshot images."""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT image_path FROM alert_logs")
                rows = cursor.fetchall()
                for row in rows:
                    if row[0] and os.path.exists(row[0]):
                        try:
                            os.remove(row[0])
                        except Exception:
                            pass
                cursor.execute("DELETE FROM alert_logs")
                conn.commit()
                conn.close()
                self.active_notifications.clear()
                return True
            except Exception as e:
                print(f"Lỗi khi xóa toàn bộ lịch sử cảnh báo: {e}")
                return False

    def export_csv(self, filepath):
        """Exports all alert records including incident response notes to a CSV file."""
        alerts = self.get_all_alerts(limit=1000)
        try:
            with open(filepath, mode="w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["ID", "Thời Gian", "Loại Cảnh Báo", "Độ Tin Cậy (%)", "Kết Quả Xử Lý Ca Trực", "Ghi Chú Chi Tiết", "Đường Dẫn Ảnh"])
                for a in alerts:
                    writer.writerow([
                        a["id"],
                        a["timestamp"],
                        a["label"],
                        f"{int(a['confidence'] * 100)}%",
                        a.get("action_taken", "Chưa xử lý"),
                        a.get("operator_note", ""),
                        a["image_path"]
                    ])
            return True, f"Đã xuất {len(alerts)} bản ghi ra file: {filepath}"
        except Exception as e:
            return False, f"Lỗi khi xuất file CSV: {e}"
