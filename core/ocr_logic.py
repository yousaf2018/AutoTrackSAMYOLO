import cv2
import pytesseract
from datetime import datetime, timedelta
import subprocess
import os
import re
import numpy as np
import sys

# Configure Tesseract Path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRVideoSplitter:
    def __init__(self, log_signal=None, progress_signal=None, date_format="%Y-%m-%d %H:%M:%S", ocr_config=None):
        self.log_signal = log_signal
        self.progress_signal = progress_signal
        self.stop_flag = False
        self.time_map = [] 
        self.date_format = date_format
        
        # Default config if None
        self.ocr_config = ocr_config if ocr_config else {
            'scale': 4, 'method': 'Otsu', 'thresh_val': 127, 'invert': False
        }

    def log(self, msg):
        if self.log_signal: self.log_signal.emit(msg)
        else: print(msg)

    def preprocess_image(self, img):
        """Dynamic preprocessing based on UI settings."""
        if img is None or img.size == 0: return img
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Upscale (Dynamic)
        scale = self.ocr_config.get('scale', 4)
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 3. Denoise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 4. Threshold (Dynamic)
        method = self.ocr_config.get('method', 'Otsu')
        if method == 'Manual':
            val = self.ocr_config.get('thresh_val', 127)
            _, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        else:
            # Otsu
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Inversion (Smart or Forced)
        force_invert = self.ocr_config.get('invert', False)
        
        if force_invert:
            # User wants explicit inversion
            thresh = cv2.bitwise_not(thresh)
        else:
            # Smart Check: If text is white on black, we invert to get black on white
            white_pixels = np.count_nonzero(thresh)
            if white_pixels < (thresh.size * 0.5):
                thresh = cv2.bitwise_not(thresh)
            
        # 6. Padding
        thresh = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255])
        
        return thresh

    def clean_ocr_text(self, text):
        """Fixes common OCR typos."""
        text = re.sub(r'[^A-Za-z0-9:\-\/ \.]', '', text).strip().upper()
        replacements = {
            'O': '0', 'Q': '0', 'D': '0', 'U': '0',
            'Z': '2', 'S': '5', '$': '5', 'B': '8',
            'I': '1', 'L': '1', '|': '1', ']': '1', '[': '1', 'A': '4'
        }
        cleaned = [replacements.get(char, char) for char in text]
        return "".join(cleaned)

    def parse_time_string(self, text):
        text = self.clean_ocr_text(text)
        try: return datetime.strptime(text, self.date_format)
        except ValueError: pass

        formats = [
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d%H:%M:%S", 
            "%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%H:%M:%S %d-%m-%Y", "%Y-%m-%d %I:%M:%S%p"
        ]
        for fmt in formats:
            try: return datetime.strptime(text, fmt)
            except ValueError: continue
        return None

    def run_single_frame_ocr(self, frame, roi):
        """GUI Preview Helper."""
        x, y, w, h = roi
        if w <= 0 or h <= 0: return "", "Invalid ROI"
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0: return "", "Empty Crop"
        
        clean_img = self.preprocess_image(crop)
        
        try:
            whitelist = "0123456789:-/AMPMampm. "
            cfg = f'--psm 7 -c tessedit_char_whitelist="{whitelist}"'
            raw_text = pytesseract.image_to_string(clean_img, config=cfg).strip()
            cleaned_text = self.clean_ocr_text(raw_text)
            dt = self.parse_time_string(cleaned_text)
            
            status = f"MATCH: {dt.strftime('%Y-%m-%d %H:%M:%S')}" if dt else "NO MATCH"
            return cleaned_text, status
        except Exception as e:
            return "Error", str(e)

    def scan_video(self, video_path, crop_rect, sample_interval=30):
        self.log(f"Scanning {os.path.basename(video_path)}...")
        self.time_map = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 25.0
        
        step_frames = int(fps * sample_interval)
        x, y, w, h = crop_rect
        current_frame = 0
        
        whitelist = "0123456789:-/AMPMampm. "
        cfg = f'--psm 7 -c tessedit_char_whitelist="{whitelist}"'
        
        while current_frame < total_frames:
            if self.stop_flag: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret: break
            
            clean = self.preprocess_image(frame[y:y+h, x:x+w])
            try:
                raw = pytesseract.image_to_string(clean, config=cfg).strip()
                dt = self.parse_time_string(raw)
                if dt: self.time_map.append((current_frame / fps, dt))
            except: pass
            
            current_frame += step_frames
            if self.progress_signal:
                self.progress_signal.emit(int((current_frame / total_frames) * 50), f"Mapped {len(self.time_map)} timestamps")

        cap.release()
        if len(self.time_map) < 2: 
            if sample_interval > 5: return self.scan_video(video_path, crop_rect, 5) # Retry finer
            else: raise Exception("OCR Failed. Adjust Threshold/ROI.")
            
        self.log(f"Scan Complete. Points: {len(self.time_map)}")
        return self.time_map

    # ... (get_video_time and execute_split remain the same as previous) ...
    def get_video_time_from_real_time(self, target_dt):
        self.time_map.sort(key=lambda x: x[0])
        for i in range(len(self.time_map) - 1):
            t1_v, t1_r = self.time_map[i]; t2_v, t2_r = self.time_map[i+1]
            if t1_r <= target_dt <= t2_r:
                tot = (t2_r - t1_r).total_seconds()
                if tot == 0: return t1_v
                return t1_v + ((target_dt - t1_r).total_seconds() / tot) * (t2_v - t1_v)
        return None

    def execute_split(self, video_path, output_dir, target_start, target_end, stitch=False):
        if not self.time_map: return
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        s_real = self.time_map[0][1]; e_real = self.time_map[-1][1]
        segments = []
        cur_day = s_real.replace(hour=target_start.hour, minute=target_start.minute, second=0) - timedelta(days=1)
        
        while cur_day < e_real + timedelta(days=1):
            w_start = cur_day
            if target_end < target_start:
                w_end = w_start + timedelta(days=1); w_end = w_end.replace(hour=target_end.hour, minute=target_end.minute, second=0)
            else: w_end = w_start.replace(hour=target_end.hour, minute=target_end.minute, second=0)
            
            act_s = max(w_start, s_real); act_e = min(w_end, e_real)
            if act_s < act_e:
                v1 = self.get_video_time_from_real_time(act_s)
                v2 = self.get_video_time_from_real_time(act_e)
                if v1 is not None and v2 is not None: segments.append((v1, v2, act_s))
            cur_day += timedelta(days=1)

        self.log(f"Found {len(segments)} segments.")
        gen_files = []
        for i, (s, e, ref) in enumerate(segments):
            if self.stop_flag: break
            dur = e - s
            if dur < 5: continue
            name = f"{os.path.splitext(os.path.basename(video_path))[0]}_part{i+1}_{ref.strftime('%Y%m%d_%H%M')}.mp4"
            out = os.path.join(output_dir, name)
            subprocess.run(["ffmpeg", "-ss", str(s), "-i", video_path, "-t", str(dur), "-c", "copy", "-y", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gen_files.append(out); self.log(f"Exported: {name}")
            if self.progress_signal: self.progress_signal.emit(50+int((i/len(segments))*50), f"Exporting {i+1}...")

        if stitch and len(gen_files) > 1:
            lp = os.path.join(output_dir, "list.txt")
            with open(lp, "w") as f: 
                for g in gen_files: f.write(f"file '{g}'\n")
            fo = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_merged.mp4")
            subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", lp, "-c", "copy", "-y", fo], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log(f"Merged: {fo}"); os.remove(lp)