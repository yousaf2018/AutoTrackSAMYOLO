import os
import cv2
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QLabel, QFrame, QSplitter, QListWidget, QComboBox, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRect
from PyQt6.QtGui import QIcon, QAction, QPolygonF

from ui.video_selector import VideoSelectorWidget
from workers.yolo_workers import DatasetWorker

class DatasetWindow(QMainWindow):
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dataset Creator")
        self.resize(1300, 900)
        self.matched_files = []
        self.preview_cap = None
        self.preview_csv_data = {} 
        self.total_frames = 0
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # --- HEADER ---
        header_layout = QHBoxLayout()
        btn_home = self.create_button("🏠 Home", "#444444")
        btn_home.setFixedWidth(100)
        btn_home.clicked.connect(self.go_home_signal.emit)
        
        title = QLabel("YOLO Dataset Factory")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(btn_home)
        header_layout.addWidget(title, 1)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === LEFT PANEL (Config) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)

        # 1. Inputs
        grp_in = QGroupBox("1. Input Data")
        form_in = QFormLayout()
        
        self.path_raw = QLineEdit()
        self.path_raw.setPlaceholderText("Raw Videos Folder")
        self.path_raw.setStyleSheet("padding: 5px; background: #222; color: #eee; border: 1px solid #444;")
        btn_raw = self.create_button("...", "#0d6efd"); btn_raw.setFixedWidth(40)
        btn_raw.clicked.connect(lambda: self.browse(self.path_raw))
        
        self.path_sam = QLineEdit(os.path.join(os.getcwd(), "SAM3_Results"))
        self.path_sam.setStyleSheet("padding: 5px; background: #222; color: #eee; border: 1px solid #444;")
        btn_sam = self.create_button("...", "#0d6efd"); btn_sam.setFixedWidth(40)
        btn_sam.clicked.connect(lambda: self.browse(self.path_sam))
        
        self.path_out = QLineEdit(os.path.join(os.getcwd(), "YOLO_Dataset"))
        self.path_out.setStyleSheet("padding: 5px; background: #222; color: #eee; border: 1px solid #444;")
        btn_out = self.create_button("...", "#0d6efd"); btn_out.setFixedWidth(40)
        btn_out.clicked.connect(lambda: self.browse(self.path_out))
        
        form_in.addRow("Raw Videos:", self.horizontal(self.path_raw, btn_raw))
        form_in.addRow("SAM Results:", self.horizontal(self.path_sam, btn_sam))
        form_in.addRow("Output Dir:", self.horizontal(self.path_out, btn_out))
        
        self.btn_scan = self.create_button("Scan & Match Files", "#6f42c1")
        self.btn_scan.clicked.connect(self.scan_folders)
        form_in.addRow("", self.btn_scan)
        
        grp_in.setLayout(form_in)
        left_layout.addWidget(grp_in)
        
        # List of files
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("background: #1e1e1e; color: #eee; border: 1px solid #444;")
        self.file_list.currentRowChanged.connect(self.load_preview_data)
        left_layout.addWidget(QLabel("Matched Video/CSV Pairs:"))
        left_layout.addWidget(self.file_list)

        # 2. Config
        grp_cfg = QGroupBox("2. Dataset Configuration")
        form_cfg = QFormLayout()
        
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Detection (Bounding Boxes)", "Segmentation (Polygons)"])
        self.combo_type.currentIndexChanged.connect(self.update_preview_overlay)
        self.combo_type.setStyleSheet("padding: 5px; background: #222; color: white;")
        form_cfg.addRow("Task Type:", self.combo_type)
        
        self.spin_box = QSpinBox()
        self.spin_box.setRange(10, 500); self.spin_box.setValue(50); self.spin_box.setSuffix(" px")
        self.spin_box.valueChanged.connect(self.update_preview_overlay)
        self.spin_box.setStyleSheet("padding: 5px; background: #222; color: #fff; border: 1px solid #444;")
        form_cfg.addRow("Fixed Box Size:", self.spin_box)
        
        self.line_classes = QLineEdit("particle")
        self.line_classes.setStyleSheet("padding: 5px; background: #222; color: #fff; border: 1px solid #444;")
        form_cfg.addRow("Class Names:", self.line_classes)
        
        self.spin_frames = QSpinBox(); self.spin_frames.setRange(100, 1000000); self.spin_frames.setValue(1000)
        self.spin_frames.setStyleSheet("padding: 5px; background: #222; color: #fff; border: 1px solid #444;")
        form_cfg.addRow("Total Samples:", self.spin_frames)
        
        # --- SMART SPLIT CONFIGURATION (PERCENT) ---
        split_group = QGroupBox("Dataset Splits (Must sum to 100%)")
        split_group.setStyleSheet("border: 1px solid #555; padding: 5px;")
        split_layout = QHBoxLayout()
        
        self.spin_train = QSpinBox(); self.spin_train.setRange(0, 100); self.spin_train.setValue(70); self.spin_train.setSuffix("% Train")
        self.spin_train.setStyleSheet("color: #0d6efd; font-weight: bold; background: #222;")
        
        self.spin_val = QSpinBox(); self.spin_val.setRange(0, 100); self.spin_val.setValue(20); self.spin_val.setSuffix("% Val")
        self.spin_val.setStyleSheet("color: #ffc107; font-weight: bold; background: #222;")
        
        self.spin_test = QSpinBox(); self.spin_test.setRange(0, 100); self.spin_test.setValue(10); self.spin_test.setSuffix("% Test")
        self.spin_test.setReadOnly(True) # Read only, calculated
        self.spin_test.setStyleSheet("color: #20c997; font-weight: bold; background: #333; border: 1px dashed #555;")
        
        # Connect logic
        self.spin_train.valueChanged.connect(self.balance_splits)
        self.spin_val.valueChanged.connect(self.balance_splits)
        
        split_layout.addWidget(self.spin_train)
        split_layout.addWidget(self.spin_val)
        split_layout.addWidget(self.spin_test)
        split_group.setLayout(split_layout)
        
        left_layout.addWidget(grp_cfg)
        left_layout.addWidget(split_group) # Add splits below main config

        grp_cfg.setLayout(form_cfg)
        # left_layout.addWidget(grp_cfg) # Already added above logic

        # 3. Action
        self.btn_run = self.create_button("GENERATE DATASET", "#198754")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.clicked.connect(self.start_gen)
        left_layout.addWidget(self.btn_run)
        
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setStyleSheet("QProgressBar { border: 1px solid #444; background: #222; height: 15px; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #0d6efd; }")
        left_layout.addWidget(self.pbar)

        # ==========================
        # RIGHT PANEL (Visualizer)
        # ==========================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
        vis_group = QGroupBox("3. Visual Preview (Verify Crops)")
        vis_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; } QGroupBox::title { color: #e0a800; }")
        vis_layout = QVBoxLayout()
        
        self.lbl_vis_info = QLabel("Select a file to verify crops/polygons before generation.")
        self.lbl_vis_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_vis_info.setStyleSheet("color: #aaa; font-style: italic;")
        
        # Video Widget
        self.viewer = VideoSelectorWidget()
        
        # Timeline
        time_layout = QHBoxLayout()
        self.lbl_frame = QLabel("Frame: 0")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.seek_video)
        time_layout.addWidget(self.lbl_frame)
        time_layout.addWidget(self.slider)
        
        vis_layout.addWidget(self.lbl_vis_info)
        vis_layout.addWidget(self.viewer, 1)
        vis_layout.addLayout(time_layout)
        vis_group.setLayout(vis_layout)
        
        right_layout.addWidget(vis_group)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 800])
        main_layout.addWidget(splitter)

    # --- Helpers ---
    def create_button(self, text, color):
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; border-radius: 4px; padding: 6px; font-weight: bold; }} QPushButton:hover {{ background-color: {color}dd; }}")
        return btn
        
    def horizontal(self, w1, w2):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0,0,0,0); l.addWidget(w1); l.addWidget(w2); return w

    def browse(self, field):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d: field.setText(d)
        
    def balance_splits(self):
        """Ensures Train + Val + Test = 100"""
        train = self.spin_train.value()
        val = self.spin_val.value()
        
        # Logic: If Train + Val > 100, reduce the one NOT being edited? 
        # Simpler logic: Limit secondary based on primary
        if train + val > 100:
            if self.sender() == self.spin_train:
                val = 100 - train
                self.spin_val.blockSignals(True)
                self.spin_val.setValue(val)
                self.spin_val.blockSignals(False)
            else:
                train = 100 - val
                self.spin_train.blockSignals(True)
                self.spin_train.setValue(train)
                self.spin_train.blockSignals(False)

        test = 100 - (train + val)
        self.spin_test.setValue(test)

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
        if row < 0 or row >= len(self.matched_files): return
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
                self.preview_csv_data[fid].append(r)
            self.lbl_vis_info.setText(f"Loaded: {len(df)} tracking points.")
        except: 
            self.preview_csv_data = {}
        self.seek_video(0)

    def seek_video(self, frame_idx):
        if not self.preview_cap: return
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.preview_cap.read()
        if ret:
            self.viewer.set_current_frame(frame_idx, frame)
            self.update_preview_overlay()
            self.lbl_frame.setText(f"Frame: {frame_idx} / {self.total_frames}")

    def update_preview_overlay(self):
        if not self.preview_csv_data: return
        frame_idx = self.slider.value()
        mode = "polygon" if "Segmentation" in self.combo_type.currentText() else "box"
        rows = self.preview_csv_data.get(frame_idx, [])
        items_to_draw = []
        
        if mode == "box":
            crop_size = self.spin_box.value()
            half = crop_size // 2
            for r in rows:
                cx, cy = float(r['Centroid_X']), float(r['Centroid_Y'])
                items_to_draw.append(QRect(int(cx - half), int(cy - half), crop_size, crop_size))
        else:
            for r in rows:
                poly_str = str(r.get('Polygon_Coords', ''))
                if poly_str and poly_str != 'nan':
                    points = []
                    for pair in poly_str.split(';'):
                        if ',' in pair:
                            px, py = map(float, pair.split(','))
                            points.append(QPointF(px, py))
                    if points:
                        items_to_draw.append(QPolygonF(points))
        
        self.viewer.set_preview_data(items_to_draw, mode=mode)

    def start_gen(self):
        if not self.matched_files: QMessageBox.warning(self,"Error","Scan first"); return
        
        classes = [c.strip() for c in self.line_classes.text().split(',')]
        d_type = "Segmentation" if "Segmentation" in self.combo_type.currentText() else "Detection"
        
        cfg = {
            'raw_video_dir': self.path_raw.text(),
            'sam_results_dir': self.path_sam.text(),
            'output_dataset_dir': self.path_out.text(),
            'max_frames': self.spin_frames.value(),
            # Convert Percent (0-100) to Ratio (0.0-1.0)
            'train_ratio': self.spin_train.value() / 100.0,
            'val_ratio': self.spin_val.value() / 100.0,
            'seed': 42,
            'dataset_type': d_type,
            'class_names': classes,
            'box_size': self.spin_box.value()
        }
        
        self.btn_run.setEnabled(False); self.btn_run.setText("Generating...")
        self.worker = DatasetWorker(cfg)
        self.worker.progress_signal.connect(lambda v, m: (self.pbar.setValue(v), self.pbar.setFormat(m)))
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self):
        self.btn_run.setEnabled(True); self.btn_run.setText("GENERATE DATASET")
        self.pbar.setValue(100); self.pbar.setFormat("Completed")
        QMessageBox.information(self, "Success", "Dataset Created Successfully!")

    def on_error(self, err):
        self.btn_run.setEnabled(True); self.btn_run.setText("GENERATE DATASET")
        QMessageBox.critical(self, "Failed", f"Error: {err}")