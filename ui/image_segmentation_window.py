import os
import cv2
import json
import datetime
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QListWidget, QGroupBox, 
    QTextEdit, QLabel, QProgressBar, QMessageBox, 
    QSizePolicy, QSplitter, QFrame, QFormLayout,
    QDoubleSpinBox, QSpinBox, QTabWidget, QLineEdit,
    QListWidgetItem, QAbstractItemView, QCheckBox,
    QScrollArea, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QAction, QIcon, QColor, QTextCursor

# --- Import Custom Modules ---
from ui.video_selector import VideoSelectorWidget
from workers.image_worker import ImageAnalysisWorker

class ImageSegmentationWindow(QMainWindow):
    # Signal to return to Launcher
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 3 Image/Video Segmentation (Text + Box)")
        self.resize(1400, 950)
        
        # --- Internal State ---
        self.added_paths = set()
        self.worker = None
        self.current_preview_frame = None
        self.current_video_cap = None
        self.total_frames = 0
        
        # Default Paths
        self.output_dir = os.path.join(os.getcwd(), "SAM3_Image_Results")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.default_sam_path = "/mnt/Zebrafish_24TB/SAM3-Development/sam3"
        
        # --- Initialize UI ---
        self.setup_ui()

    def setup_ui(self):
        """Builds the entire GUI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        
        # =================================================
        # 0. HEADER BAR
        # =================================================
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #252526; border-bottom: 1px solid #333;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(15, 10, 15, 10)
        
        self.btn_home = QPushButton("🏠  Home")
        self.btn_home.setFixedSize(100, 35)
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton { background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #444; }
        """)
        self.btn_home.clicked.connect(self.go_home_signal.emit)
        
        lbl_title = QLabel("SAM 3 Zero-Shot Segmentation")
        lbl_title.setStyleSheet("font-size: 18px; color: #eee; font-weight: bold; border: none;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(self.btn_home)
        header_layout.addWidget(lbl_title, 1)
        header_layout.addWidget(QWidget(), 0) # Spacer
        
        root_layout.addWidget(header_widget)
        
        # =================================================
        # MAIN CONTENT AREA
        # =================================================
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        
        # -------------------------------------------------
        # 1. LEFT PANEL (Scrollable Controls)
        # -------------------------------------------------
        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(380)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(15)
        
        # --- Group 1: Video Input ---
        grp_vid = QGroupBox("1. Video Input")
        vid_layout = QVBoxLayout()
        
        row_btns = QHBoxLayout()
        self.btn_add_file = self.create_button("Add File", "#0d6efd")
        self.btn_add_file.clicked.connect(self.select_files)
        
        self.btn_add_folder = self.create_button("Add Folder", "#0d6efd")
        self.btn_add_folder.clicked.connect(self.select_folder)
        
        self.btn_clear = self.create_button("Clear", "#6c757d")
        self.btn_clear.clicked.connect(self.clear_videos)
        
        row_btns.addWidget(self.btn_add_file)
        row_btns.addWidget(self.btn_add_folder)
        row_btns.addWidget(self.btn_clear)
        
        self.list_videos = QListWidget()
        self.list_videos.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_videos.currentRowChanged.connect(self.load_preview)
        self.list_videos.setMinimumHeight(150)
        self.list_videos.setCursor(Qt.CursorShape.PointingHandCursor)
        self.list_videos.setStyleSheet("""
            QListWidget { background-color: #1e1e1e; color: #fff; border: 1px solid #444; border-radius: 4px; }
            QListWidget::item:selected { background-color: #0d6efd; }
        """)
        
        vid_layout.addLayout(row_btns)
        vid_layout.addWidget(self.list_videos)
        grp_vid.setLayout(vid_layout)
        
        # --- Group 2: Configuration ---
        grp_cfg = QGroupBox("2. Prompts & Settings")
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Local SAM
        sam_row = QHBoxLayout()
        self.line_sam = QLineEdit(self.default_sam_path)
        self.line_sam.setPlaceholderText("Select Local SAM3 Repo...")
        self.line_sam.setCursor(Qt.CursorShape.IBeamCursor)
        self.line_sam.setStyleSheet("background-color: #1e1e1e; color: #ddd; border: 1px solid #444; padding: 4px;")
        
        self.btn_sam_browse = self.create_button("...", "#495057")
        self.btn_sam_browse.setFixedWidth(30)
        self.btn_sam_browse.clicked.connect(self.sel_sam)
        
        sam_row.addWidget(self.line_sam)
        sam_row.addWidget(self.btn_sam_browse)
        form_layout.addRow("Local SAM3:", sam_row)
        
        # Text Prompt
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText("e.g., 'nanoparticle', 'cell', 'car'")
        self.txt_prompt.setCursor(Qt.CursorShape.IBeamCursor)
        self.txt_prompt.setStyleSheet("padding: 5px; font-size: 14px; border: 1px solid #0d6efd; background-color: #1e1e1e; color: #fff;")
        form_layout.addRow("Text Prompt:", self.txt_prompt)
        
        # Confidence
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 0.99)
        self.spin_conf.setValue(0.50)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_conf.wheelEvent = lambda event: event.ignore()
        self.spin_conf.setStyleSheet("background-color: #1e1e1e; color: #fff; padding: 4px;")
        form_layout.addRow("Confidence Thresh:", self.spin_conf)
        
        # Scale
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.0001, 100.0)
        self.spin_scale.setValue(0.3240)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setSuffix(" µm")
        self.spin_scale.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_scale.wheelEvent = lambda event: event.ignore()
        self.spin_scale.setStyleSheet("background-color: #1e1e1e; color: #fff; padding: 4px;")
        form_layout.addRow("Pixel Scale:", self.spin_scale)
        
        grp_cfg.setLayout(form_layout)
        
        # --- Group 3: Output ---
        grp_out = QGroupBox("3. Output Location")
        out_layout = QVBoxLayout()
        
        self.line_out = QLineEdit(self.output_dir)
        self.line_out.setReadOnly(True)
        self.line_out.setCursor(Qt.CursorShape.IBeamCursor)
        self.line_out.setStyleSheet("background-color: #1e1e1e; color: #aaa; border: 1px solid #444; padding: 5px;")
        
        self.btn_out_browse = self.create_button("Browse Output...", "#495057")
        self.btn_out_browse.clicked.connect(self.sel_out)
        
        out_layout.addWidget(self.line_out)
        out_layout.addWidget(self.btn_out_browse)
        grp_out.setLayout(out_layout)
        
        # --- Group 4: Run ---
        grp_run = QGroupBox("4. Execute")
        run_layout = QVBoxLayout()
        
        self.btn_start = self.create_button("START SEGMENTATION", "#198754")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.clicked.connect(self.start_analysis)
        
        self.btn_stop = self.create_button("STOP", "#dc3545")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        
        run_layout.addWidget(self.btn_start)
        run_layout.addWidget(self.btn_stop)
        grp_run.setLayout(run_layout)
        
        # Add to Left Layout
        left_layout.addWidget(grp_vid)
        left_layout.addWidget(grp_cfg)
        left_layout.addWidget(grp_out)
        left_layout.addWidget(grp_run)
        left_layout.addStretch()
        
        left_scroll.setWidget(left_widget)
        
        # -------------------------------------------------
        # 2. CENTER PANEL
        # -------------------------------------------------
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Visualizer ---
        grp_vis = QGroupBox("Visualizer: Optional Box Prompt")
        grp_vis.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 6px; margin-top: 12px; } QGroupBox::title { color: #0d6efd; background-color: #1e1e1e; padding: 0 5px; }")
        vis_layout = QVBoxLayout()
        vis_layout.setContentsMargins(5, 15, 5, 5)
        
        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_frame.setStyleSheet("color: #fff; font-weight: bold; font-family: Consolas;")
        
        self.video_selector = VideoSelectorWidget()
        self.video_selector.selection_changed.connect(self.on_box_drawn)
        
        # Timeline
        time_row = QHBoxLayout()
        
        self.btn_prev = self.create_button("<", "#555555")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.clicked.connect(lambda: self.change_frame_relative(-1))
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.slider_moved)
        
        self.btn_next = self.create_button(">", "#555555")
        self.btn_next.setFixedWidth(30)
        self.btn_next.clicked.connect(lambda: self.change_frame_relative(1))
        
        time_row.addWidget(self.lbl_frame)
        time_row.addWidget(self.btn_prev)
        time_row.addWidget(self.slider)
        time_row.addWidget(self.btn_next)
        
        # Tools
        tool_row = QHBoxLayout()
        self.btn_del_box = self.create_button("Delete Box", "#d9534f")
        self.btn_del_box.clicked.connect(self.video_selector.delete_selected)
        
        self.btn_clear_frame = self.create_button("Clear Frame", "#6c757d")
        self.btn_clear_frame.clicked.connect(self.video_selector.clear_current_frame)
        
        self.btn_clear_all = self.create_button("Clear All", "#555555")
        self.btn_clear_all.clicked.connect(self.video_selector.clear_all)
        
        self.btn_save_json = self.create_button("Save JSON", "#17a2b8")
        self.btn_save_json.clicked.connect(self.save_annotations)
        
        self.btn_load_json = self.create_button("Load JSON", "#17a2b8")
        self.btn_load_json.clicked.connect(self.load_annotations)
        
        tool_row.addWidget(self.btn_del_box)
        tool_row.addWidget(self.btn_clear_frame)
        tool_row.addWidget(self.btn_clear_all)
        tool_row.addStretch()
        tool_row.addWidget(self.btn_save_json)
        tool_row.addWidget(self.btn_load_json)
        
        vis_layout.addWidget(QLabel("Draw boxes to guide detection alongside text (Optional)."))
        vis_layout.addWidget(self.video_selector, 1)
        vis_layout.addLayout(time_row)
        vis_layout.addLayout(tool_row)
        grp_vis.setLayout(vis_layout)
        
        # --- Logs ---
        grp_log = QGroupBox("Logs & Statistics")
        log_layout = QVBoxLayout()
        
        stats_row = QHBoxLayout()
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_fps.setStyleSheet("color:#0dcaf0; font-weight: bold;")
        self.lbl_time = QLabel("Left: -")
        self.lbl_time.setStyleSheet("color:#ffc107; font-weight: bold;")
        self.lbl_obj = QLabel("Obj: 0")
        self.lbl_obj.setStyleSheet("color:#20c997; font-weight: bold;")
        
        stats_row.addWidget(self.lbl_fps)
        stats_row.addStretch()
        stats_row.addWidget(self.lbl_time)
        stats_row.addStretch()
        stats_row.addWidget(self.lbl_obj)
        
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setCursor(Qt.CursorShape.IBeamCursor)
        self.text_log.setStyleSheet("background-color: #111; color: #0f0; font-family: Consolas; border: 1px solid #444;")
        
        # DEFINED HERE to prevent AttributeError
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #0d6efd; border-radius: 4px; }")
        
        log_layout.addLayout(stats_row)
        log_layout.addWidget(self.text_log)
        log_layout.addWidget(self.progress_bar)
        grp_log.setLayout(log_layout)
        
        center_layout.addWidget(grp_vis, 2)
        center_layout.addWidget(grp_log, 1)
        
        content_layout.addWidget(left_scroll)
        content_layout.addWidget(center_panel, 1)
        
        root_layout.addLayout(content_layout)

    # =================================================
    # HELPER METHODS
    # =================================================
    def create_button(self, text, color):
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 4px;
                padding: 6px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {color}dd; 
            }}
            QPushButton:disabled {{
                background-color: #444;
                color: #888;
            }}
        """)
        return btn

    def log(self, m):
        self.text_log.append(f">> {m}")
        self.text_log.verticalScrollBar().setValue(self.text_log.verticalScrollBar().maximum())

    def sel_sam(self): 
        f = QFileDialog.getExistingDirectory(self, "Select Local SAM3 Repo")
        if f: self.line_sam.setText(f)

    def sel_out(self): 
        f = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if f: self.line_out.setText(f)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Videos (*.mp4 *.avi *.mov)")
        if files:
            for f in files:
                self.add_file(f)
            self.log(f"Added {len(files)} videos.")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov')):
                        self.add_file(os.path.join(root, f))
                break 

    def add_file(self, f):
        if f not in self.added_paths:
            self.added_paths.add(f)
            item = QListWidgetItem(os.path.basename(f))
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_videos.addItem(item)

    def clear_videos(self):
        self.added_paths.clear()
        self.list_videos.clear()
        self.video_selector.clear_all()
        self.log("Video list cleared.")

    def load_preview(self, row):
        if row < 0: return
        path = self.list_videos.item(row).data(Qt.ItemDataRole.UserRole)
        try:
            if self.current_video_cap: self.current_video_cap.release()
            self.current_video_cap = cv2.VideoCapture(path)
            self.total_frames = int(self.current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setValue(0)
            self.show_frame(0)
            self.log(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self.log(f"Error loading: {e}")

    def change_frame_relative(self, delta):
        if self.total_frames == 0: return
        new_pos = max(0, min(self.slider.value() + delta, self.total_frames - 1))
        self.slider.setValue(new_pos)
        self.show_frame(new_pos)

    def slider_released(self):
        self.show_frame(self.slider.value())
        
    def slider_moved(self, val):
        self.lbl_frame.setText(f"Frame: {val}/{self.total_frames}")

    def show_frame(self, idx):
        if self.current_video_cap:
            self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.current_video_cap.read()
            if ret:
                self.current_preview_frame = frame
                self.video_selector.set_current_frame(idx, frame)
                self.lbl_frame.setText(f"Frame: {idx}/{self.total_frames}")

    def on_box_drawn(self):
        pass

    def save_annotations(self):
        if not self.video_selector.annotations:
            QMessageBox.warning(self, "Empty", "No annotations to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON (*.json)")
        if path:
            data = {str(k): [[r.x(), r.y(), r.width(), r.height()] for r in v] for k,v in self.video_selector.annotations.items()}
            with open(path, 'w') as f: json.dump(data, f)
            self.log(f"Saved to {path}")

    def load_annotations(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load JSON", "", "JSON (*.json)")
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f: data = json.load(f)
                loaded = {int(k): [QRect(x,y,w,h) for x,y,w,h in v] for k,v in data.items()}
                self.video_selector.annotations = loaded
                self.video_selector.update()
                if self.current_preview_frame is not None:
                    self.video_selector.set_current_frame(self.slider.value(), self.current_preview_frame)
                self.log("Loaded boxes.")
            except Exception as e:
                self.log(f"Error loading: {e}")

    def start_analysis(self):
        active_videos = []
        for i in range(self.list_videos.count()):
            if self.list_videos.item(i).checkState() == Qt.CheckState.Checked:
                active_videos.append(self.list_videos.item(i).data(Qt.ItemDataRole.UserRole))
        
        if not active_videos:
            QMessageBox.warning(self, "Error", "No videos selected.")
            return
        
        text_prompt = self.txt_prompt.text().strip()
        box_prompts = self.video_selector.annotations
        
        if not text_prompt and not box_prompts:
            QMessageBox.warning(self, "Input Error", "Please provide a Text Prompt OR Draw Boxes.")
            return
        
        sam_path = self.line_sam.text()
        if not os.path.exists(sam_path):
            QMessageBox.critical(self, "Error", "Local SAM3 path does not exist.")
            return
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_out = os.path.join(self.output_dir, f"ImgRun_{ts}")
        os.makedirs(final_out, exist_ok=True)

        self.set_ui_processing(True)
        self.progress_bar.setValue(0)
        self.text_log.clear()
        
        config = {
            "sam_path": sam_path,
            "text_prompt": text_prompt,
            "match_threshold": self.spin_conf.value(),
            "pixel_scale_um": self.spin_scale.value(),
            "save_video": True
        }
        
        self.worker = ImageAnalysisWorker(active_videos, box_prompts, final_out, config)
        self.worker.log_app_signal.connect(self.log)
        self.worker.progress_signal.connect(lambda v, m: (self.progress_bar.setValue(v), self.progress_bar.setFormat(m)))
        self.worker.stats_signal.connect(self.update_stats)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.start()

    def stop_analysis(self):
        if self.worker:
            self.log("Stopping...")
            self.worker.stop()
            self.btn_stop.setText("Stopping...")
            self.btn_stop.setEnabled(False)

    def update_stats(self, s):
        self.lbl_fps.setText(f"FPS: {s.get('fps', '-')}")
        self.lbl_time.setText(f"Time: {s.get('time_left', '-')}")
        self.lbl_obj.setText(f"Obj: {s.get('objects', '0')}")

    def handle_error(self, msg):
        self.log(f"ERROR: {msg}")
        self.set_ui_processing(False)

    def analysis_finished(self):
        self.log("Analysis Completed.")
        self.set_ui_processing(False)
        QMessageBox.information(self, "Done", "Segmentation Complete")

    def set_ui_processing(self, active):
        self.btn_start.setEnabled(not active)
        self.btn_stop.setEnabled(active)
        self.btn_stop.setText("STOP")
        
        # Disable inputs
        self.list_videos.setEnabled(not active)
        self.line_sam.setEnabled(not active)
        self.txt_prompt.setEnabled(not active)
        self.video_selector.setEnabled(not active)
        self.spin_conf.setEnabled(not active)
        self.spin_scale.setEnabled(not active)
        self.btn_add_file.setEnabled(not active)
        self.btn_add_folder.setEnabled(not active)
        self.btn_clear.setEnabled(not active)
        self.btn_del_box.setEnabled(not active)
        self.btn_clear_frame.setEnabled(not active)
        self.btn_clear_all.setEnabled(not active)
        self.btn_save_json.setEnabled(not active)
        self.btn_load_json.setEnabled(not active)