import os
import cv2
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QLabel, QFrame, QSplitter, QListWidget, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QAction

from ui.video_selector import VideoSelectorWidget
from workers.yolo_workers import DatasetWorker

class DatasetWindow(QMainWindow):
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dataset Creator & Visualizer")
        self.resize(1200, 900)
        self.matched_files = [] 
        self.preview_cap = None
        self.preview_csv_data = None
        self.total_frames = 0
        
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- HEADER ---
        header_layout = QHBoxLayout()
        btn_home = QPushButton("🏠 Home")
        btn_home.setFixedWidth(100)
        btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_home.setStyleSheet("background-color: #444; color: white; border: 1px solid #555; border-radius: 5px; padding: 6px; font-weight: bold;")
        btn_home.clicked.connect(self.go_home_signal.emit)
        
        title = QLabel("YOLO Dataset Generator")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(btn_home)
        header_layout.addWidget(title, 1)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # --- MAIN SPLIT ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL (Settings)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        grp_in = QGroupBox("1. Paths & Source")
        form = QFormLayout()
        
        self.path_raw = QLineEdit()
        self.path_raw.setPlaceholderText("Raw Videos Folder")
        btn_raw = self.create_button("Browse...", "#0d6efd")
        btn_raw.clicked.connect(lambda: self.browse(self.path_raw))
        
        self.path_sam = QLineEdit(os.path.join(os.getcwd(), "SAM3_Results"))
        btn_sam = self.create_button("Browse...", "#0d6efd")
        btn_sam.clicked.connect(lambda: self.browse(self.path_sam))
        
        self.path_out = QLineEdit(os.path.join(os.getcwd(), "YOLO_Dataset"))
        btn_out = self.create_button("Browse...", "#0d6efd")
        btn_out.clicked.connect(lambda: self.browse(self.path_out))
        
        self.btn_scan = self.create_button("Scan Folders for Matches", "#6f42c1")
        self.btn_scan.clicked.connect(self.scan_folders)

        form.addRow("Raw Videos:", self.path_raw)
        form.addRow("", btn_raw)
        form.addRow("SAM Results:", self.path_sam)
        form.addRow("", btn_sam)
        form.addRow("Output Dir:", self.path_out)
        form.addRow("", btn_out)
        form.addRow("", self.btn_scan)
        grp_in.setLayout(form)
        left_layout.addWidget(grp_in)
        
        grp_list = QGroupBox("Matched Files")
        list_layout = QVBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("background: #111; color: #eee; border: 1px solid #444;")
        self.file_list.currentRowChanged.connect(self.load_preview_data)
        list_layout.addWidget(self.file_list)
        grp_list.setLayout(list_layout)
        left_layout.addWidget(grp_list)

        grp_par = QGroupBox("2. Dataset Parameters")
        form_par = QFormLayout()
        
        self.spin_box = QSpinBox()
        self.spin_box.setRange(10, 500); self.spin_box.setValue(50); self.spin_box.setSuffix(" px")
        self.spin_box.valueChanged.connect(self.update_preview_overlay)
        
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(100, 999999); self.spin_frames.setValue(1000)
        
        self.spin_train = QDoubleSpinBox(); self.spin_train.setRange(0.0, 1.0); self.spin_train.setValue(0.8); self.spin_train.setSingleStep(0.1)
        self.spin_val = QDoubleSpinBox(); self.spin_val.setRange(0.0, 1.0); self.spin_val.setValue(0.1); self.spin_val.setSingleStep(0.1)
        
        form_par.addRow("Crop Size:", self.spin_box)
        form_par.addRow("Max Samples:", self.spin_frames)
        form_par.addRow("Train Ratio:", self.spin_train)
        form_par.addRow("Val Ratio:", self.spin_val)
        grp_par.setLayout(form_par)
        left_layout.addWidget(grp_par)
        
        self.btn_run = self.create_button("GENERATE DATASET", "#198754")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.clicked.connect(self.start_process)
        left_layout.addWidget(self.btn_run)
        
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        left_layout.addWidget(self.pbar)

        # RIGHT PANEL (Visualizer)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        vis_group = QGroupBox("3. Visual Preview (Verify Crops)")
        vis_layout = QVBoxLayout()
        
        self.viewer = VideoSelectorWidget()
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.seek_video)
        
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        vis_layout.addWidget(self.viewer, 1)
        vis_layout.addWidget(self.lbl_frame_info)
        vis_layout.addWidget(self.slider)
        vis_group.setLayout(vis_layout)
        
        right_layout.addWidget(vis_group)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)

    def create_button(self, text, color):
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; border-radius: 4px; padding: 6px; }} QPushButton:hover {{ background-color: {color}dd; }}")
        return btn

    def browse(self, field):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d: field.setText(d)

    def scan_folders(self):
        raw_dir = self.path_raw.text()
        sam_dir = self.path_sam.text()
        if not os.path.exists(raw_dir) or not os.path.exists(sam_dir):
            QMessageBox.warning(self, "Error", "Directories do not exist.")
            return
        self.matched_files = []
        self.file_list.clear()
        
        csv_map = {}
        for root, _, files in os.walk(sam_dir):
            for f in files:
                if f.endswith("_data.csv"):
                    stem = f.replace("_data.csv", "")
                    csv_map[stem] = os.path.join(root, f)
        
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    stem = os.path.splitext(f)[0]
                    if stem in csv_map:
                        self.matched_files.append((os.path.join(root, f), csv_map[stem]))
                        self.file_list.addItem(f"{f}")
        
        if not self.matched_files: QMessageBox.information(self, "Scan", "No matches found.")
        else: QMessageBox.information(self, "Scan", f"Found {len(self.matched_files)} pairs.")

    def load_preview_data(self, row):
        if row < 0: return
        vid_path, csv_path = self.matched_files[row]
        
        if self.preview_cap: self.preview_cap.release()
        self.preview_cap = cv2.VideoCapture(vid_path)
        self.total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        
        try:
            df = pd.read_csv(csv_path)
            self.preview_csv_data = {}
            for _, r in df.iterrows():
                fid = int(r['Global_Frame_ID'])
                if fid not in self.preview_csv_data: self.preview_csv_data[fid] = []
                self.preview_csv_data[fid].append((r['Centroid_X'], r['Centroid_Y']))
        except: self.preview_csv_data = {}
        self.seek_video(0)

    def seek_video(self, frame_idx):
        if not self.preview_cap: return
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.preview_cap.read()
        if ret:
            self.viewer.load_frame(frame) # FIX: load_frame now exists
            self.update_preview_overlay()
            self.lbl_frame_info.setText(f"Frame: {frame_idx} / {self.total_frames}")

    def update_preview_overlay(self):
        if not self.preview_csv_data: return
        frame_idx = self.slider.value()
        crop_size = self.spin_box.value()
        half = crop_size // 2
        points = self.preview_csv_data.get(frame_idx, [])
        rects = [(int(cx - half), int(cy - half), crop_size, crop_size) for cx, cy in points]
        self.viewer.set_detected_objects(rects)

    def start_process(self):
        if not self.matched_files: QMessageBox.warning(self, "Error", "No matches."); return
        cfg = {
            'raw_video_dir': self.path_raw.text(),
            'sam_results_dir': self.path_sam.text(),
            'output_dataset_dir': self.path_out.text(),
            'max_frames': self.spin_frames.value(),
            'box_size': self.spin_box.value(),
            'train_ratio': self.spin_train.value(),
            'val_ratio': self.spin_val.value(),
            'seed': 42
        }
        self.btn_run.setEnabled(False)
        self.worker = DatasetWorker(cfg)
        self.worker.progress_signal.connect(lambda v, m: (self.pbar.setValue(v), self.pbar.setFormat(m)))
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.pbar.setValue(100)
        QMessageBox.information(self, "Done", "Dataset Created!")

    def on_error(self, err):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed: {err}")