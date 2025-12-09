from PyQt6.QtCore import QThread, pyqtSignal
import traceback
import sys
import os

# Import the logic engine
from core.dataset_logic import YoloDatasetGenerator

class SignalStream:
    """
    Redirects sys.stdout/stderr to a PyQt Signal.
    Used to show YOLO training progress bars in the GUI log.
    """
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        # Filter out empty strings to prevent blank lines
        if text.strip(): 
            self.signal.emit(str(text))

    def flush(self):
        pass

class DatasetWorker(QThread):
    """
    Background thread for generating the YOLO dataset
    from SAM 3 CSV results and Raw Videos.
    """
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
            self.log_signal.emit("Initializing Dataset Generator...")
            
            # Instantiate the logic class
            self.generator = YoloDatasetGenerator(
                self.config, 
                log_signal=self.log_signal, 
                progress_signal=self.progress_signal
            )
            
            # Run the heavy processing
            self.generator.run()
            
            self.finished_signal.emit()
            
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

    def stop(self):
        """Sets the internal stop flag in the logic engine."""
        if self.generator:
            self.generator.stop_flag = True

class TrainingWorker(QThread):
    """
    Background thread for training the YOLO model using Ultralytics.
    Captures console output to show training epochs/loss in GUI.
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def run(self):
        # 1. Redirect Standard Output to GUI Log
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stream = SignalStream(self.log_signal)
        sys.stdout = stream
        sys.stderr = stream
        
        try:
            self.log_signal.emit("Importing Ultralytics YOLO...")
            from ultralytics import YOLO
            
            # 2. Load Model
            model_path = self.config.get("model_weights", "yolov8n.pt")
            self.log_signal.emit(f"Loading Base Model: {model_path}")
            model = YOLO(model_path)
            
            # 3. Determine Task (Detect vs Segment)
            # Default to 'detect' if not specified
            task_type = self.config.get('task', 'detect')
            self.log_signal.emit(f"Starting {task_type.upper()} Training...")
            self.log_signal.emit(f"Data: {self.config['data_yaml']}")
            
            # 4. Start Training
            # The output of this function goes to stdout, which we have redirected
            results = model.train(
                data=self.config['data_yaml'],
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                project=self.config['project_dir'],
                name=self.config['run_name'],
                device=self.config['device'],
                task=task_type,
                verbose=True  # Ensure YOLO prints to stdout
            )
            
            self.log_signal.emit("-" * 30)
            self.log_signal.emit(f"Training Finished Successfully.")
            self.log_signal.emit(f"Best Model Saved at: {results.save_dir}")
            self.finished_signal.emit()

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
        
        finally:
            # 5. Restore Standard Output (Critical cleanup)
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def stop(self):
        """
        YOLO training is blocking C++ code mostly, so stopping gracefully 
        is difficult. We flag is_running, but usually, we just wait 
        or let the user kill the app if it hangs.
        """
        self.is_running = False