import sys
import os
import subprocess
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

class VideoSplitterWorker(QThread):
    # Signals
    overall_progress = pyqtSignal(int, int, str) # current_idx, total_count, filename
    file_progress = pyqtSignal(int, str)         # percent, status_text
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, video_files, output_dir, chunk_seconds, use_subfolders):
        super().__init__()
        self.video_files = video_files
        self.output_dir = output_dir
        self.chunk_seconds = chunk_seconds
        self.use_subfolders = use_subfolders
        self.is_running = True
        self.process = None

    def stop(self):
        self.is_running = False
        self.log_signal.emit("🛑 Stopping process...")
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def _check_ffmpeg(self):
        try:
            # Check if ffmpeg is in PATH
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except:
            return False

    def _needs_reencode(self, video_path):
        """Checks if video needs re-encoding (e.g. H.264 High Profile)."""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,profile",
                "-of", "default=nokey=1:noprint_wrappers=1", video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            info = result.stdout.lower()
            return "h264" in info and ("high" in info or "high 10" in info)
        except:
            return True # Fallback to re-encode if check fails

    def run(self):
        if not self._check_ffmpeg():
            self.error_signal.emit("FFmpeg not found. Please install FFmpeg and add to PATH.")
            return

        total_files = len(self.video_files)

        for i, video_path in enumerate(self.video_files):
            if not self.is_running: break

            filename = os.path.basename(video_path)
            self.overall_progress.emit(i + 1, total_files, filename)
            self.log_signal.emit(f"Processing: {filename}")

            try:
                # 1. Get Duration
                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", video_path]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                try:
                    duration = float(result.stdout.strip())
                except ValueError:
                    self.log_signal.emit(f"Could not read duration for {filename}")
                    continue
                
                num_chunks = int(duration // self.chunk_seconds)
                if duration % self.chunk_seconds > 1: num_chunks += 1
                
                self.log_signal.emit(f"  > Duration: {duration:.1f}s | Chunks: {num_chunks}")

                # 2. Prepare Output
                base_name = os.path.splitext(filename)[0]
                if self.use_subfolders:
                    current_output_dir = os.path.join(self.output_dir, base_name)
                else:
                    current_output_dir = self.output_dir
                    
                os.makedirs(current_output_dir, exist_ok=True)

                reencode = self._needs_reencode(video_path)
                if reencode:
                    self.log_signal.emit("  > High Profile detected. Re-encoding (slower but safe)...")

                # 3. Split Loop
                for chunk_idx in range(num_chunks):
                    if not self.is_running: break

                    start_time = chunk_idx * self.chunk_seconds
                    output_file = os.path.join(current_output_dir, f"{base_name}_p{chunk_idx + 1:02d}.mp4")
                    
                    if reencode:
                        # Slow but robust re-encode
                        cmd = [
                            "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(self.chunk_seconds),
                            "-map", "0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                            "-c:a", "aac", "-y", output_file
                        ]
                        # For re-encoding, we just wait for process (parsing progress line-by-line is complex)
                        self.file_progress.emit(int((chunk_idx/num_chunks)*100), f"Encoding Part {chunk_idx+1}...")
                        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.process.wait()
                    else:
                        # Fast Stream Copy
                        cmd = [
                            "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(self.chunk_seconds),
                            "-c", "copy", "-y", output_file
                        ]
                        self.file_progress.emit(int((chunk_idx/num_chunks)*100), f"Copying Part {chunk_idx+1}...")
                        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.process.wait()

                self.log_signal.emit(f"  > Completed {filename}")

            except Exception as e:
                self.log_signal.emit(f"Error processing {filename}: {e}")
                continue

        self.finished_signal.emit()