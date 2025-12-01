import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from workers.yolo_workers import DatasetWorker

class DatasetWindow(QMainWindow):
    # Signal to switch back to Launcher
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dataset Creator")
        self.resize(1000, 850)
        self.setup_ui()
        self.worker = None

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # ==========================================
        # 0. HEADER (Home Button)
        # ==========================================
        header_layout = QHBoxLayout()
        
        btn_home = QPushButton("🏠 Home")
        btn_home.setFixedWidth(100)
        btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_home.setStyleSheet("""
            QPushButton { background-color: #444; color: white; border: 1px solid #555; border-radius: 5px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
        """)
        btn_home.clicked.connect(self.go_home_signal.emit)
        
        title = QLabel("YOLO Dataset Generator")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(btn_home)
        header_layout.addWidget(title, 1)
        header_layout.addStretch() 
        
        main_layout.addLayout(header_layout)
        
        # ==========================================
        # 1. INPUT CONFIGURATION
        # ==========================================
        grp_in = QGroupBox("1. Input & Output Paths")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setSpacing(10)

        # Raw Videos
        self.path_raw = QLineEdit()
        self.path_raw.setPlaceholderText("Folder containing original .mp4 files")
        self.path_raw.setStyleSheet("padding: 8px; color: #eee; background: #222; border: 1px solid #444;")
        btn_raw = self.create_button("Browse...", "#0d6efd")
        btn_raw.setFixedWidth(80)
        btn_raw.clicked.connect(lambda: self.browse(self.path_raw))
        row_raw = QHBoxLayout(); row_raw.addWidget(self.path_raw); row_raw.addWidget(btn_raw)
        form.addRow("Raw Videos Dir:", row_raw)

        # SAM Results
        self.path_sam = QLineEdit(os.path.join(os.getcwd(), "SAM3_Results"))
        self.path_sam.setPlaceholderText("Folder containing SAM3 output folders")
        self.path_sam.setStyleSheet("padding: 8px; color: #eee; background: #222; border: 1px solid #444;")
        btn_sam = self.create_button("Browse...", "#0d6efd")
        btn_sam.setFixedWidth(80)
        btn_sam.clicked.connect(lambda: self.browse(self.path_sam))
        row_sam = QHBoxLayout(); row_sam.addWidget(self.path_sam); row_sam.addWidget(btn_sam)
        form.addRow("SAM 3 Results Dir:", row_sam)

        # Output
        self.path_out = QLineEdit(os.path.join(os.getcwd(), "YOLO_Dataset"))
        self.path_out.setStyleSheet("padding: 8px; color: #eee; background: #222; border: 1px solid #444;")
        btn_out = self.create_button("Browse...", "#0d6efd")
        btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self.browse(self.path_out))
        row_out = QHBoxLayout(); row_out.addWidget(self.path_out); row_out.addWidget(btn_out)
        form.addRow("Output Dataset Dir:", row_out)

        grp_in.setLayout(form)
        main_layout.addWidget(grp_in)

        # ==========================================
        # 2. PARAMETERS
        # ==========================================
        grp_par = QGroupBox("2. Dataset Parameters")
        par_layout = QHBoxLayout()
        
        # --- Left Side: General Params ---
        form_par = QFormLayout()
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(100, 500000)
        self.spin_frames.setValue(1000)
        self.spin_frames.setSuffix(" frames")
        self.spin_frames.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_frames.setStyleSheet("padding: 5px; background: #222; color: #eee; border: 1px solid #444;")
        
        self.spin_box = QSpinBox()
        self.spin_box.setValue(30)
        self.spin_box.setSuffix(" px")
        self.spin_box.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_box.setStyleSheet("padding: 5px; background: #222; color: #eee; border: 1px solid #444;")
        
        form_par.addRow("Max Frames:", self.spin_frames)
        form_par.addRow("Crop Size:", self.spin_box)
        
        par_widget = QWidget()
        par_widget.setLayout(form_par)
        par_layout.addWidget(par_widget)

        # --- Right Side: Splits (FIXED VISIBILITY) ---
        split_group = QGroupBox("Splits (%)")
        split_group.setStyleSheet("border: 1px solid #555; font-weight: bold;")
        split_box_layout = QHBoxLayout()
        
        # Train
        self.spin_train = QSpinBox()
        self.spin_train.setRange(0, 100)
        self.spin_train.setValue(70)
        self.spin_train.setSuffix("% Train")
        self.spin_train.setMinimumWidth(110) # Ensure text fits
        self.spin_train.setStyleSheet("padding: 5px; background: #222; color: #eee; font-weight: bold;")
        
        # Val
        self.spin_val = QSpinBox()
        self.spin_val.setRange(0, 100)
        self.spin_val.setValue(20)
        self.spin_val.setSuffix("% Val")
        self.spin_val.setMinimumWidth(110) # Ensure text fits
        self.spin_val.setStyleSheet("padding: 5px; background: #222; color: #eee; font-weight: bold;")

        # Test (ReadOnly)
        self.spin_test = QSpinBox()
        self.spin_test.setRange(0, 100)
        self.spin_test.setValue(10)
        self.spin_test.setSuffix("% Test")
        self.spin_test.setReadOnly(True)
        self.spin_test.setMinimumWidth(110) # Ensure text fits
        self.spin_test.setStyleSheet("padding: 5px; background: #333; color: #aaa; border: 1px dashed #555;") # Distinct look
        
        # Auto-calc test logic
        self.spin_train.valueChanged.connect(self.update_split)
        self.spin_val.valueChanged.connect(self.update_split)
        
        split_box_layout.addWidget(self.spin_train)
        split_box_layout.addWidget(self.spin_val)
        split_box_layout.addWidget(self.spin_test)
        split_group.setLayout(split_box_layout)
        
        par_layout.addWidget(split_group)
        grp_par.setLayout(par_layout)
        main_layout.addWidget(grp_par)

        # ==========================================
        # 3. ACTION & LOGS
        # ==========================================
        self.btn_run = self.create_button("GENERATE DATASET", "#198754") # Green
        self.btn_run.setMinimumHeight(55)
        self.btn_run.setStyleSheet("""
            QPushButton { background-color: #198754; color: white; font-size: 16px; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background-color: #157347; }
            QPushButton:disabled { background-color: #444; }
        """)
        self.btn_run.clicked.connect(self.start_process)
        
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setTextVisible(True)
        self.pbar.setStyleSheet("QProgressBar { border: 1px solid #444; background: #222; height: 15px; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #0d6efd; }")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setCursor(Qt.CursorShape.IBeamCursor)
        self.log_text.setStyleSheet("background-color: #111; color: #0f0; font-family: Consolas; border: 1px solid #444;")

        main_layout.addWidget(self.btn_run)
        main_layout.addWidget(self.pbar)
        main_layout.addWidget(self.log_text)

    # --- Helpers ---
    def create_button(self, text, color):
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; border-radius: 4px; padding: 6px; }} QPushButton:hover {{ background-color: {color}dd; }}")
        return btn

    def update_split(self):
        """Auto-calculate Test split to ensure total is 100%"""
        train = self.spin_train.value()
        val = self.spin_val.value()
        rem = 100 - (train + val)
        
        if rem < 0:
            # If over 100, reduce Val
            self.spin_val.blockSignals(True)
            self.spin_val.setValue(100 - train)
            self.spin_val.blockSignals(False)
            self.spin_test.setValue(0)
        else:
            self.spin_test.setValue(rem)

    def browse(self, field):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d: field.setText(d)
        
    def log(self, msg):
        self.log_text.append(f">> {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def start_process(self):
        # Validate Paths
        if not os.path.exists(self.path_raw.text()):
            QMessageBox.critical(self, "Error", "Raw Videos directory does not exist.")
            return
        
        cfg = {
            'raw_video_dir': self.path_raw.text(),
            'sam_results_dir': self.path_sam.text(),
            'output_dataset_dir': self.path_out.text(),
            'max_frames': self.spin_frames.value(),
            'box_size': self.spin_box.value(),
            'train_ratio': self.spin_train.value() / 100.0,
            'val_ratio': self.spin_val.value() / 100.0,
            'seed': 42
        }
        
        self.btn_run.setEnabled(False)
        self.log("Starting generation...")
        
        self.worker = DatasetWorker(cfg)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(lambda v, m: (self.pbar.setValue(v), self.pbar.setFormat(f"{m} ({v}%)")))
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.pbar.setValue(100)
        self.pbar.setFormat("Completed")
        QMessageBox.information(self, "Success", "Dataset Created Successfully!")

    def on_error(self, err):
        self.log(f"ERROR: {err}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Failed", f"Error: {err}")