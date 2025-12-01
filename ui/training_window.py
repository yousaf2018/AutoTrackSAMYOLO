import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QSpinBox, QComboBox, 
    QGroupBox, QTextEdit, QMessageBox, QLabel, QFileDialog  # <--- Added QFileDialog here
)
from PyQt6.QtCore import Qt, pyqtSignal
from workers.yolo_workers import TrainingWorker

class TrainingWindow(QMainWindow):
    # Signal to return to Launcher
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Model Trainer")
        self.resize(900, 700)
        self.setup_ui()
        self.worker = None

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # --- 0. HEADER (Home Button) ---
        header_layout = QHBoxLayout()
        
        self.btn_home = QPushButton("🏠 Home")
        self.btn_home.setFixedWidth(100)
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton { background-color: #444; color: white; border: 1px solid #555; border-radius: 5px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
        """)
        self.btn_home.clicked.connect(self.go_home_signal.emit)
        
        title = QLabel("YOLO Training Dashboard")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(self.btn_home)
        header_layout.addWidget(title, 1)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # --- 1. Configuration Group ---
        grp_cfg = QGroupBox("1. Training Configuration")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.path_data = QLineEdit()
        self.path_data.setPlaceholderText("Path to data.yaml (inside generated dataset folder)")
        self.path_data.setStyleSheet("background-color: #1e1e1e; color: #ddd; padding: 8px; border: 1px solid #444;")
        
        self.btn_data = QPushButton("Browse data.yaml")
        self.btn_data.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_data.setStyleSheet("background-color: #0d6efd; color: white; padding: 5px;")
        self.btn_data.clicked.connect(self.browse_yaml)
        
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.path_data)
        data_layout.addWidget(self.btn_data)
        form.addRow("Dataset YAML:", data_layout)
        
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt"])
        self.combo_model.setStyleSheet("background-color: #1e1e1e; color: #fff; padding: 5px;")
        self.combo_model.setCursor(Qt.CursorShape.PointingHandCursor)
        form.addRow("Base Model:", self.combo_model)
        
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 5000)
        self.spin_epochs.setValue(50)
        self.spin_epochs.setStyleSheet("background-color: #1e1e1e; color: #fff; padding: 5px;")
        form.addRow("Epochs:", self.spin_epochs)
        
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(16)
        self.spin_batch.setStyleSheet("background-color: #1e1e1e; color: #fff; padding: 5px;")
        form.addRow("Batch Size:", self.spin_batch)
        
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 1280)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setStyleSheet("background-color: #1e1e1e; color: #fff; padding: 5px;")
        form.addRow("Image Size:", self.spin_imgsz)
        
        grp_cfg.setLayout(form)
        layout.addWidget(grp_cfg)
        
        # --- 2. Output Logs ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Training logs will appear here...")
        self.log_text.setStyleSheet("background-color: #000; color: #0f0; font-family: Consolas; border: 1px solid #444;")
        layout.addWidget(self.log_text)
        
        # --- 3. Action Button ---
        self.btn_train = QPushButton("START TRAINING")
        self.btn_train.setMinimumHeight(60)
        self.btn_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_train.setStyleSheet("""
            QPushButton { background-color: #e0a800; color: black; font-weight: bold; font-size: 16px; border-radius: 5px; }
            QPushButton:hover { background-color: #d39e00; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """)
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

    def browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML (*.yaml)")
        if f: self.path_data.setText(f)

    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def start_training(self):
        yaml_path = self.path_data.text()
        if not os.path.exists(yaml_path):
            QMessageBox.warning(self, "Input Error", "Please select a valid data.yaml file first.")
            return
            
        # Configure Training
        cfg = {
            'data_yaml': yaml_path,
            'epochs': self.spin_epochs.value(),
            'imgsz': self.spin_imgsz.value(),
            'batch': self.spin_batch.value(),
            'model_weights': self.combo_model.currentText(),
            'project_dir': os.path.join(os.getcwd(), "YOLO_Training_Runs"),
            'run_name': 'train_run',
            'device': 0 # Default to GPU 0
        }
        
        self.btn_train.setEnabled(False)
        self.btn_train.setText("Training in Progress...")
        self.log("Initializing YOLO worker...")
        
        self.worker = TrainingWorker(cfg)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("START TRAINING")
        QMessageBox.information(self, "Success", "Training Complete!\nCheck 'YOLO_Training_Runs' folder.")

    def on_error(self, err):
        self.log(f"CRITICAL ERROR: {err}")
        self.btn_train.setEnabled(True)
        self.btn_train.setText("START TRAINING")
        QMessageBox.critical(self, "Training Failed", f"Error:\n{err}")