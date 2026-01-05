from PyQt6.QtCore import QThread, pyqtSignal
import time
import traceback
import sys
import os

class SignalStream:
    """Redirects console output to a Qt signal."""
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        if text:
            self.signal.emit(str(text))

    def flush(self):
        pass

class AnalysisWorker(QThread):
    # Signals
    progress_signal = pyqtSignal(int, str)
    log_app_signal = pyqtSignal(str)
    log_sys_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    # --- UPDATED CONSTRUCTOR: 3 ARGUMENTS ---
    def __init__(self, tasks, output_dir, config):
        """
        :param tasks: List of tuples [(video_path, annotation_dict), ...]
        :param output_dir: Directory to save results
        :param config: Configuration dictionary
        """
        super().__init__()
        self.tasks = tasks 
        self.output_dir = output_dir
        self.config = config
        self.is_running = True
        self.pipeline = None

    def run(self):
        # Redirect Sys Output
        orig_out = sys.stdout
        orig_err = sys.stderr
        stream = SignalStream(self.log_sys_signal)
        sys.stdout = stream
        sys.stderr = stream

        try:
            # 1. Link SAM3
            sam_path = self.config.get("sam_path", "")
            if sam_path and os.path.exists(sam_path):
                if sam_path not in sys.path:
                    sys.path.insert(0, sam_path)
                    self.log_app_signal.emit(f"Linked Local SAM3: {sam_path}")

            # 2. Import Engine
            from core.sam_logic import SAM3Pipeline
            
            self.log_app_signal.emit("Initializing Pipeline...")
            self.pipeline = SAM3Pipeline(self.output_dir, self.config, self.log_app_signal)
            
            # 3. Load Model
            self.pipeline.load_model()
            
            total_videos = len(self.tasks)
            start_time = time.time()
            
            # 4. Processing Loop
            for i, (video_path, ann_dict) in enumerate(self.tasks):
                if not self.is_running: break
                
                video_name = os.path.basename(video_path)
                self.log_app_signal.emit(f"Processing Video [{i+1}/{total_videos}]: {video_name}")
                
                # Pre-load templates for this specific video
                if ann_dict:
                    self.pipeline.extract_templates_from_rects(video_path, ann_dict)
                else:
                    self.log_app_signal.emit("No manual templates. Running auto-scan if enabled.")
                
                # Progress Callback
                def update_progress(step, total, msg, curr_batch=0, tot_batch=0):
                    # Calculate overall progress
                    video_chunk = 100.0 / total_videos
                    base_progress = i * video_chunk
                    current_progress = (step / total) * video_chunk
                    overall_progress = int(base_progress + current_progress)
                    
                    self.progress_signal.emit(overall_progress, msg)
                    
                    # Stats Calculation
                    elapsed = time.time() - start_time
                    time_str = "--:--"
                    if overall_progress > 0:
                        total_est = elapsed / (overall_progress / 100.0)
                        rem = total_est - elapsed
                        if rem >= 0:
                            time_str = f"{int(rem // 60)}m {int(rem % 60)}s"
                    
                    # Chunk Info string
                    chunk_str = msg.split('(')[0].strip() if "Chunk" in msg else "-"
                    batch_str = f"{curr_batch}/{tot_batch}" if tot_batch > 0 else "-"

                    # Emit Stats
                    self.stats_signal.emit({
                        "fps": f"{self.pipeline.current_fps:.1f}",
                        "time_left": time_str,
                        "objects": str(self.pipeline.active_objects),
                        "chunk": chunk_str,
                        "batch": batch_str 
                    })

                self.pipeline.process_video(video_path, progress_callback=update_progress)
            
            self.log_app_signal.emit("Batch Analysis Complete.")
            self.finished_signal.emit()

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
        
        finally:
            # Clean up
            if self.pipeline:
                self.pipeline.cleanup()
            # Restore console
            sys.stdout = orig_out
            sys.stderr = orig_err

    def stop(self):
        self.is_running = False
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop_flag = True