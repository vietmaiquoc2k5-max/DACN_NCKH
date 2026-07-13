import os
import wave
import math
import struct
import threading
import time
import requests
import cv2
import sys

# Import sound modules conditionally
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import winsound

try:
    import pygame
except ImportError:
    pygame = None

class AlertManager:
    def __init__(self, sound_path="alarm.wav"):
        self.sound_path = sound_path
        self.alarm_playing = False
        self.telegram_token = ""
        self.telegram_chat_id = ""
        self.mute_alarm = False
        
        # Initialize pygame mixer for audio playing (supports MP3 and WAV on all platforms)
        if pygame:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"Error initializing pygame mixer: {e}")
            
        # Ensure default alarm sound exists
        self.ensure_alarm_sound()

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
                        # Alternating siren frequencies (800Hz <-> 1000Hz) every 0.25 seconds
                        freq = 950 if int(t * 4) % 2 == 0 else 650
                        # Generate sine wave sample
                        val = int(16384 * math.sin(2 * math.pi * freq * t))  # volume scaled to 50%
                        data = struct.pack('<h', val)
                        wav_file.writeframesraw(data)
            except Exception as e:
                print(f"Failed to generate default alarm sound: {e}")

    def update_config(self, token, chat_id, mute_alarm):
        """Update Telegram configuration and settings from GUI."""
        self.telegram_token = token.strip()
        self.telegram_chat_id = chat_id.strip()
        self.mute_alarm = mute_alarm

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
                
            # Check if the file is a fake WAV (e.g. an MP3 renamed to .wav)
            try:
                if os.path.exists(self.sound_path):
                    with open(self.sound_path, "rb") as f:
                        header = f.read(4)
                    if len(header) >= 4 and not header.startswith(b"RIFF"):
                        # It is not a standard WAV file. Assume it is MP3 data and copy to .mp3 extension
                        import uuid
                        import shutil
                        temp_path = f"temp_alarm_{uuid.uuid4().hex[:8]}.mp3"
                        shutil.copyfile(self.sound_path, temp_path)
                        play_path = temp_path
                        is_temp_file = True
            except Exception as ex:
                print(f"Error detecting file headers: {ex}")

            # Prefer Pygame if available since it supports playing MP3 data even if named .wav
            if pygame and pygame.mixer and pygame.mixer.get_init():
                pygame.mixer.music.load(play_path)
                # Play in loop (-1)
                pygame.mixer.music.play(-1)
                while self.alarm_playing and not self.mute_alarm:
                    time.sleep(0.2)
            elif IS_WINDOWS:
                # Play asynchronously and in loop
                winsound.PlaySound(
                    play_path, 
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                )
                # Keep thread alive to monitor status
                while self.alarm_playing and not self.mute_alarm:
                    time.sleep(0.2)
            else:
                print("No supported sound library available for this platform.")
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

    def save_alert_locally(self, frame):
        """Saves the alert frame locally in the 'alerts' directory."""
        try:
            alerts_dir = "alerts"
            if not os.path.exists(alerts_dir):
                os.makedirs(alerts_dir)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(alerts_dir, f"alert_{timestamp}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"Saved alert image locally to: {filepath}")
        except Exception as e:
            print(f"Failed to save alert image locally: {e}")

    def send_telegram_alert_sync(self, frame, message="🚨 CẢNH BÁO: Phát hiện đám cháy!"):
        """Sends an alert message and captured frame to Telegram synchronously. Returns (bool, str)."""
        # Save locally first, independent of Telegram config
        if frame is not None:
            self.save_alert_locally(frame)
            
        if not self.telegram_token or not self.telegram_chat_id:
            return False, "Cấu hình Telegram trống (Thiếu Token hoặc Chat ID)."
            
        token = self.telegram_token
        chat_id = self.telegram_chat_id
        
        # 1. Send Text Message
        try:
            url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message}
            response = requests.post(url_msg, data=payload, timeout=10)
            res_json = response.json()
            if not res_json.get("ok"):
                return False, f"Telegram API error (Message): {res_json.get('description', response.text)}"
        except Exception as e:
            return False, f"Lỗi gửi Telegram text: {e}"

        # 2. Send Photo (if frame is provided)
        if frame is not None:
            temp_filename = None
            try:
                # Save frame as a temporary JPEG with a unique name
                import uuid
                temp_filename = f"temp_fire_alert_{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(temp_filename, frame)
                
                url_photo = f"https://api.telegram.org/bot{token}/sendPhoto"
                with open(temp_filename, "rb") as photo_file:
                    files = {"photo": photo_file}
                    payload = {"chat_id": chat_id, "caption": "📸 Ảnh chụp từ hiện trường"}
                    response = requests.post(url_photo, data=payload, files=files, timeout=15)
                    res_json = response.json()
                    if not res_json.get("ok"):
                        return False, f"Telegram API error (Photo): {res_json.get('description', response.text)}"
            except Exception as e:
                return False, f"Lỗi gửi Telegram photo: {e}"
            finally:
                # Cleanup
                if temp_filename and os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except Exception:
                        pass
        return True, ""

    def send_telegram_alert(self, frame, message="🚨 CẢNH BÁO: Phát hiện đám cháy!"):
        """Sends an alert message and captured frame to Telegram in a background thread."""
        def run():
            success, err = self.send_telegram_alert_sync(
                frame.copy() if frame is not None else None,
                message
            )
            if not success:
                print(f"Failed to send background telegram alert: {err}")
                
        threading.Thread(target=run, daemon=True).start()
