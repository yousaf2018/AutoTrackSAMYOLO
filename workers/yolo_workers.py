from PyQt6.QtCore import QThread, pyqtSignal
import traceback
import sys
import os
from core.dataset_logic import YoloDatasetGenerator

# Capture Training output
class SignalStream:
    def __init__(self, signal): self.signal = signal
    def write(self, text): 
        if text.strip(): self.signal.emit(text.strip())
    def flush(self): pass

class DatasetWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generator = None

    def run(self):
        try:
            self.log_signal.emit("Starting Dataset Generation...")
            self.generator = YoloDatasetGenerator(self.config, self.log_signal, self.progress_signal)
            self.generator.run()
            self.finished_signal.emit()
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

    def stop(self):
        if self.generator: self.generator.stop_flag = True

class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def run(self):
        orig_out = sys.stdout
        sys.stdout = SignalStream(self.log_signal)
        
        try:
            self.log_signal.emit("Initializing YOLO...")
            from ultralytics import YOLO
            
            # Load Model
            model_path = self.config.get("model_weights", "yolov8n.pt")
            model = YOLO(model_path)
            
            # Train
            self.log_signal.emit(f"Starting Training on {self.config['data_yaml']}...")
            
            results = model.train(
                data=self.config['data_yaml'],
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                project=self.config['project_dir'],
                name=self.config['run_name'],
                device=self.config['device']
            )
            
            self.log_signal.emit(f"Training Finished. Results in {results.save_dir}")
            self.finished_signal.emit()

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
        finally:
            sys.stdout = orig_out