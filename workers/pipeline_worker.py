from PyQt6.QtCore import QThread, pyqtSignal
import time
import traceback
import sys
import os

# --- DUAL LOGGER CLASS ---
class DualLogger:
    """
    Writes output to BOTH the GUI (via Signal) and a Text File (disk).
    Uses .flush() to ensure logs are saved even if the app crashes.
    """
    def __init__(self, signal, file_path, is_terminal=False):
        self.signal = signal
        self.is_terminal = is_terminal
        
        # Open file in Append mode
        self.file = open(file_path, 'a', encoding='utf-8')

    def write(self, text):
        """Standard write for sys.stdout redirection."""
        if not text: return
        
        # 1. Send to GUI
        self.signal.emit(str(text))
        
        # 2. Write to File
        try:
            self.file.write(text)
            self.file.flush() # Force save immediately
        except: pass

    def emit(self, text):
        """Adapter for PyQt Signals (App Logs). Adds newline automatically."""
        self.write(text + "\n")

    def flush(self):
        """Required for stream compatibility."""
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()

class AnalysisWorker(QThread):
    # Signals
    progress_signal = pyqtSignal(int, str)
    log_app_signal = pyqtSignal(str)
    log_sys_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_videos, templates, output_dir, config):
        super().__init__()
        self.input_videos = input_videos
        self.templates = templates 
        self.output_dir = output_dir
        self.config = config
        self.is_running = True
        
        # Paths for log files
        self.log_dir = os.path.join(output_dir, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def run(self):
        # --- 1. SETUP LOGGERS ---
        app_log_path = os.path.join(self.log_dir, "app_log.txt")
        sys_log_path = os.path.join(self.log_dir, "system_log.txt")
        
        # Create Dual Loggers
        self.app_logger = DualLogger(self.log_app_signal, app_log_path)
        self.sys_logger = DualLogger(self.log_sys_signal, sys_log_path, is_terminal=True)

        # Redirect System Console (SAM3 Output)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = self.sys_logger
        sys.stderr = self.sys_logger

        try:
            # --- 2. LINK LOCAL SAM3 ---
            sam_path = self.config.get("sam_path", "")
            if sam_path and os.path.exists(sam_path):
                if sam_path not in sys.path:
                    sys.path.insert(0, sam_path)
                    self.app_logger.emit(f"Linked Local SAM3: {sam_path}")

            # --- 3. IMPORT ENGINE ---
            from core.sam_logic import SAM3Pipeline
            
            self.app_logger.emit("Initializing Pipeline...")
            
            # Pass the APP LOGGER (which has .emit) to the pipeline
            self.pipeline = SAM3Pipeline(self.output_dir, self.config, self.app_logger)
            
            # --- 4. LOAD MODEL ---
            self.pipeline.load_model()
            
            total_videos = len(self.input_videos)
            start_time = time.time()

            # --- 5. EXTRACT TEMPLATES ---
            if self.input_videos:
                self.pipeline.extract_templates_from_rects(self.input_videos[0], self.templates)
            
            # --- 6. PROCESSING LOOP ---
            for i, video_path in enumerate(self.input_videos):
                if not self.is_running: break
                
                video_name = os.path.basename(video_path)
                self.app_logger.emit(f"Processing Video [{i+1}/{total_videos}]: {video_name}")
                
                def update_progress(step, total, msg, curr_batch=0, tot_batch=0):
                    # Calculate progress
                    video_chunk = 100.0 / total_videos
                    base_progress = i * video_chunk
                    current_progress = (step / total) * video_chunk
                    overall_progress = int(base_progress + current_progress)
                    
                    self.progress_signal.emit(overall_progress, msg)
                    
                    # Calculate Stats
                    elapsed = time.time() - start_time
                    time_str = "--:--"
                    if overall_progress > 0:
                        total_est = elapsed / (overall_progress / 100.0)
                        rem = total_est - elapsed
                        if rem >= 0:
                            time_str = f"{int(rem // 60)}m {int(rem % 60)}s"
                    
                    # Chunk Info string
                    chunk_str = msg.split('(')[0].strip() if "Chunk" in msg else "Init"
                    batch_str = f"{curr_batch}/{tot_batch}" if tot_batch > 0 else "-"

                    self.stats_signal.emit({
                        "fps": f"{self.pipeline.current_fps:.1f}",
                        "time_left": time_str,
                        "objects": str(self.pipeline.active_objects),
                        "chunk": chunk_str,
                        "batch": batch_str
                    })

                self.pipeline.process_video(video_path, progress_callback=update_progress)
            
            self.app_logger.emit("Batch Analysis Complete.")
            self.finished_signal.emit()

        except Exception as e:
            # Log error to file as well
            traceback.print_exc() 
            self.error_signal.emit(str(e))
        
        finally:
            # --- RESTORE CONSOLE & CLOSE FILES ---
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if hasattr(self, 'app_logger'): self.app_logger.close()
            if hasattr(self, 'sys_logger'): self.sys_logger.close()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'pipeline'):
            self.pipeline.stop_flag = True