from PyQt6.QtCore import QThread, pyqtSignal
import time
import traceback
import sys
import os

class SignalStream:
    def __init__(self, signal): self.signal = signal
    def write(self, text): 
        if text: self.signal.emit(str(text))
    def flush(self): pass

class ImageAnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    log_app_signal = pyqtSignal(str)
    log_sys_signal = pyqtSignal(str) # Connects to System Log
    stats_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_videos, templates, output_dir, config):
        super().__init__()
        self.input_videos = input_videos
        self.templates = templates # Dict of QRects
        self.output_dir = output_dir
        self.config = config
        self.is_running = True
        self.pipeline = None

    def run(self):
        # Redirect Console
        orig_out, orig_err = sys.stdout, sys.stderr
        stream = SignalStream(self.log_sys_signal)
        sys.stdout = stream; sys.stderr = stream

        try:
            sam_path = self.config.get("sam_path", "")
            if sam_path and os.path.exists(sam_path) and sam_path not in sys.path:
                sys.path.insert(0, sam_path)
                self.log_app_signal.emit(f"Linked SAM3: {sam_path}")

            from core.sam_image_logic import SAM3ImagePipeline
            
            self.log_app_signal.emit("Initializing Image Pipeline...")
            self.pipeline = SAM3ImagePipeline(self.output_dir, self.config, self.log_app_signal)
            self.pipeline.load_model()
            
            # Prepare manual boxes
            self.pipeline.prepare_manual_prompts(self.templates)

            total = len(self.input_videos)
            start = time.time()

            for i, vid in enumerate(self.input_videos):
                if not self.is_running: break
                
                name = os.path.basename(vid)
                self.log_app_signal.emit(f"Processing [{i+1}/{total}]: {name}")
                
                def prog_cb(curr, tot, msg):
                    chunk_p = 100/total
                    base = i*chunk_p
                    # Safety divide
                    if tot == 0: tot = 1
                    curr_p = (curr/tot)*chunk_p
                    
                    self.progress_signal.emit(int(base+curr_p), msg)
                    
                    t_str = "--:--"
                    if (base+curr_p) > 0:
                        elap = time.time() - start
                        rem = (elap / ((base+curr_p)/100)) - elap
                        if rem >= 0: t_str = f"{int(rem//60)}m {int(rem%60)}s"

                    self.stats_signal.emit({
                        "fps": f"{self.pipeline.current_fps:.1f}",
                        "time_left": t_str,
                        "objects": str(self.pipeline.active_objects),
                        "chunk": "ImageMode"
                    })

                self.pipeline.process_video(vid, prog_cb)
            
            self.log_app_signal.emit("Segmentation Complete.")
            self.finished_signal.emit()

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
        
        finally:
            if self.pipeline: self.pipeline.cleanup()
            sys.stdout = orig_out; sys.stderr = orig_err

    def stop(self):
        self.is_running = False
        if self.pipeline: self.pipeline.stop_flag = True