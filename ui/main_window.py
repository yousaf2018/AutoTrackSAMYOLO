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
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QRect
from PyQt6.QtGui import QAction, QIcon, QColor, QTextCursor

# --- Import Custom Modules ---
from ui.video_selector import VideoSelectorWidget
from workers.pipeline_worker import AnalysisWorker

class MainWindow(QMainWindow):
    # Signal to switch back to Launcher
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        
        # --- Window Setup ---
        self.setWindowTitle("SAM 3 Particle Tracker Pro")
        self.resize(1400, 950)
        self.setMinimumSize(1100, 800)
        
        # --- Internal State ---
        self.added_paths = set() 
        self.worker = None 
        self.current_video_path = None
        self.current_video_cap = None
        self.total_frames = 0
        self.current_preview_frame = None
        
        # MASTER STORAGE: { 'full/path/video.mp4': { frame_idx: [(QRect, class_id), ...] } }
        self.project_annotations = {}
        
        # Default Paths
        self.output_dir = os.path.join(os.getcwd(), "SAM3_Results")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Default SAM Path
        self.default_sam_path = "/mnt/Zebrafish_24TB/SAM3-Development/sam3"
        
        # --- Initialize UI ---
        self.setup_ui()
        
    def setup_ui(self):
        """Builds the entire GUI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Root Layout
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
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setFixedWidth(100)
        self.btn_home.setStyleSheet("""
            QPushButton { background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #444; border-color: #777; }
        """)
        self.btn_home.clicked.connect(self.go_home_signal.emit)
        
        lbl_title = QLabel("SAM 3 Tracking Pipeline")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #eee; border: none;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        dummy_spacer = QWidget()
        dummy_spacer.setFixedWidth(100)
        
        header_layout.addWidget(self.btn_home)
        header_layout.addWidget(lbl_title, 1) 
        header_layout.addWidget(dummy_spacer)
        
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
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Group 1: Video Input
        file_group = QGroupBox("1. Video Input")
        file_layout = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.btn_add_file = self.create_button("Add File", "#0d6efd"); self.btn_add_file.clicked.connect(self.select_files)
        self.btn_add_folder = self.create_button("Add Folder", "#0d6efd"); self.btn_add_folder.clicked.connect(self.select_folder)
        self.btn_clear = self.create_button("Clear List", "#6c757d"); self.btn_clear.clicked.connect(self.clear_videos)
        btn_row.addWidget(self.btn_add_file); btn_row.addWidget(self.btn_add_folder); btn_row.addWidget(self.btn_clear)
        self.list_videos = QListWidget()
        self.list_videos.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_videos.currentRowChanged.connect(self.load_preview_video)
        self.list_videos.setMinimumHeight(150)
        self.list_videos.setCursor(Qt.CursorShape.PointingHandCursor)
        self.list_videos.setStyleSheet("""
            QListWidget { background-color: #1e1e1e; color: #fff; border: 1px solid #444; border-radius: 4px; }
            QListWidget::item { padding: 5px; }
            QListWidget::item:hover { background-color: #333; }
            QListWidget::item:selected { background-color: #0d6efd; }
        """)
        file_layout.addLayout(btn_row); file_layout.addWidget(self.list_videos)
        file_group.setLayout(file_layout)

        # Group 2: Configuration
        settings_group = QGroupBox("2. Configuration & Tuning")
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        
        sam_path_layout = QHBoxLayout()
        self.line_sam_path = QLineEdit(self.default_sam_path)
        self.line_sam_path.setPlaceholderText("Select Local SAM3 Repo...")
        self.line_sam_path.setCursor(Qt.CursorShape.IBeamCursor)
        self.line_sam_path.setStyleSheet("background-color: #1e1e1e; color: #ddd; border: 1px solid #444; padding: 4px;")
        self.btn_sam_browse = self.create_button("...", "#495057"); self.btn_sam_browse.setFixedWidth(30); self.btn_sam_browse.clicked.connect(self.select_sam_folder)
        sam_path_layout.addWidget(self.line_sam_path); sam_path_layout.addWidget(self.btn_sam_browse)
        form_layout.addRow("Local SAM 3:", sam_path_layout)

        self.chk_manual_mode = QCheckBox("Manual Mode (Strict Tracking)")
        self.chk_manual_mode.setToolTip("If checked, ONLY drawn boxes are tracked.")
        self.chk_manual_mode.setStyleSheet("color: #0dcaf0; font-weight: bold;")
        self.chk_manual_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_manual_mode.toggled.connect(self.toggle_manual_mode)
        form_layout.addRow("", self.chk_manual_mode)

        thresh_layout = QHBoxLayout()
        self.spin_threshold = QDoubleSpinBox(); self.spin_threshold.setRange(0.1, 0.99); self.spin_threshold.setValue(0.60); self.spin_threshold.setSingleStep(0.05); self.spin_threshold.wheelEvent=lambda e:e.ignore()
        self.spin_threshold.setCursor(Qt.CursorShape.IBeamCursor)
        self.btn_preview = self.create_button("Preview", "#e0a800"); self.btn_preview.setStyleSheet("background-color: #e0a800; color: black; font-weight: bold; border-radius: 4px;")
        self.btn_preview.clicked.connect(self.run_preview_detection)
        thresh_layout.addWidget(self.spin_threshold); thresh_layout.addWidget(self.btn_preview)
        form_layout.addRow("Sensitivity:", thresh_layout)

        self.spin_pixel_scale = QDoubleSpinBox(); self.spin_pixel_scale.setRange(0.0001, 100.0); self.spin_pixel_scale.setDecimals(4); self.spin_pixel_scale.setValue(0.3240); self.spin_pixel_scale.setSuffix(" µm"); self.spin_pixel_scale.wheelEvent=lambda e:e.ignore()
        self.spin_pixel_scale.setCursor(Qt.CursorShape.IBeamCursor)
        form_layout.addRow("Pixel Scale:", self.spin_pixel_scale)

        self.spin_batch = QSpinBox(); self.spin_batch.setRange(1, 64); self.spin_batch.setValue(16); self.spin_batch.wheelEvent=lambda e:e.ignore()
        self.spin_batch.setCursor(Qt.CursorShape.IBeamCursor)
        form_layout.addRow("GPU Batch:", self.spin_batch)
        
        self.spin_chunk = QSpinBox(); self.spin_chunk.setRange(2, 60); self.spin_chunk.setValue(5); self.spin_chunk.setSuffix(" s"); self.spin_chunk.wheelEvent=lambda e:e.ignore()
        self.spin_chunk.setCursor(Qt.CursorShape.IBeamCursor)
        form_layout.addRow("Split Time:", self.spin_chunk)

        settings_group.setLayout(form_layout)

        # Group 3: Output
        out_type_group = QGroupBox("3. Output Files")
        out_layout = QVBoxLayout(); out_layout.setSpacing(5)
        self.chk_video = QCheckBox("Annotated Video (.mp4)"); self.chk_video.setChecked(True); self.chk_video.setStyleSheet("color: #ffc107; font-weight: bold;") 
        self.chk_video.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_csv = QCheckBox("Data Spreadsheet (.csv)"); self.chk_csv.setChecked(True); self.chk_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_heatmap = QCheckBox("Heatmap (.png)"); self.chk_heatmap.setChecked(True); self.chk_heatmap.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_traj = QCheckBox("Trajectories (.png)"); self.chk_traj.setChecked(True); self.chk_traj.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_hist = QCheckBox("Size Histogram (.png)"); self.chk_hist.setChecked(True); self.chk_hist.setCursor(Qt.CursorShape.PointingHandCursor)
        for c in [self.chk_video, self.chk_csv, self.chk_heatmap, self.chk_traj, self.chk_hist]: out_layout.addWidget(c)
        out_type_group.setLayout(out_layout)

        # Group 4: Location
        out_loc_group = QGroupBox("4. Output Location")
        out_layout = QVBoxLayout()
        self.line_output = QLineEdit(self.output_dir); self.line_output.setReadOnly(True); self.line_output.setCursor(Qt.CursorShape.IBeamCursor)
        self.line_output.setStyleSheet("background-color: #1e1e1e; color: #aaa; border: 1px solid #444; padding: 5px;")
        self.btn_browse = self.create_button("Browse Output...", "#495057"); self.btn_browse.clicked.connect(self.select_output_folder)
        out_layout.addWidget(self.line_output); out_layout.addWidget(self.btn_browse)
        out_loc_group.setLayout(out_layout)

        # Group 5: Run
        run_group = QGroupBox("5. Execute")
        run_layout = QVBoxLayout()
        self.btn_start = self.create_button("START BATCH ANALYSIS", "#198754"); self.btn_start.setMinimumHeight(50); self.btn_start.clicked.connect(self.start_analysis)
        self.btn_stop = self.create_button("STOP", "#dc3545"); self.btn_stop.setMinimumHeight(40); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self.stop_analysis)
        run_layout.addWidget(self.btn_start); run_layout.addWidget(self.btn_stop)
        run_group.setLayout(run_layout)
        
        left_layout.addWidget(file_group); left_layout.addWidget(settings_group); left_layout.addWidget(out_type_group); left_layout.addWidget(out_loc_group); left_layout.addWidget(run_group); left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # -------------------------------------------------
        # 2. CENTER PANEL
        # -------------------------------------------------
        center_panel = QWidget(); center_layout = QVBoxLayout(center_panel); center_layout.setContentsMargins(0,0,0,0)
        
        vis_group = QGroupBox("Visualizer & Multi-Frame Annotation")
        vis_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #0d6efd; background-color: #1e1e1e; }")
        vis_layout = QVBoxLayout(); vis_layout.setContentsMargins(5, 15, 5, 5)
        
        self.lbl_frame = QLabel("Frame: 0 / 0"); self.lbl_frame.setStyleSheet("color: #fff; font-weight: bold;")
        self.lbl_info = QLabel("Select Video -> Draw Boxes. Move slider to annotate other frames."); self.lbl_info.setStyleSheet("color: #ccc; font-style: italic;")
        
        self.video_selector = VideoSelectorWidget()
        self.video_selector.selection_changed.connect(self.on_selection_changed)
        
        timeline_layout = QHBoxLayout()
        self.btn_prev = self.create_button("<", "#555555"); self.btn_prev.setFixedWidth(40); self.btn_prev.clicked.connect(lambda: self.change_frame_relative(-10))
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.sliderReleased.connect(self.slider_released); self.slider.valueChanged.connect(self.slider_moved)
        self.btn_next = self.create_button(">", "#555555"); self.btn_next.setFixedWidth(40); self.btn_next.clicked.connect(lambda: self.change_frame_relative(10))
        timeline_layout.addWidget(self.lbl_frame); timeline_layout.addWidget(self.btn_prev); timeline_layout.addWidget(self.slider); timeline_layout.addWidget(self.btn_next)

        tools_layout = QHBoxLayout()
        self.lbl_class = QLabel("Class ID:")
        self.spin_class = QSpinBox(); self.spin_class.setRange(0, 99); self.spin_class.setValue(0); self.spin_class.setFixedWidth(50)
        self.spin_class.valueChanged.connect(self.update_class_id)
        
        self.btn_del_box = self.create_button("Delete Box", "#d9534f"); self.btn_del_box.clicked.connect(self.video_selector.delete_selected)
        self.btn_clear_frame = self.create_button("Clear Frame", "#6c757d"); self.btn_clear_frame.clicked.connect(self.video_selector.clear_current_frame)
        self.btn_clear_all = self.create_button("Clear All Frames", "#343a40"); self.btn_clear_all.clicked.connect(self.clear_all_annotations)
        self.btn_save_ann = self.create_button("Save JSON", "#17a2b8"); self.btn_save_ann.clicked.connect(self.save_annotations)
        self.btn_load_ann = self.create_button("Load JSON", "#17a2b8"); self.btn_load_ann.clicked.connect(self.load_annotations)

        tools_layout.addWidget(self.lbl_class); tools_layout.addWidget(self.spin_class)
        tools_layout.addWidget(self.btn_del_box); tools_layout.addWidget(self.btn_clear_frame); tools_layout.addWidget(self.btn_clear_all); tools_layout.addStretch(); tools_layout.addWidget(self.btn_save_ann); tools_layout.addWidget(self.btn_load_ann)
        
        vis_layout.addWidget(self.lbl_info); vis_layout.addWidget(self.video_selector, 1); vis_layout.addLayout(timeline_layout); vis_layout.addLayout(tools_layout)
        vis_group.setLayout(vis_layout)
        
        # Logs
        log_container = QGroupBox("Logs & Statistics"); log_layout = QVBoxLayout()
        stats_w = QWidget(); stats_box = QHBoxLayout(stats_w); stats_box.setContentsMargins(0,0,0,0)
        self.lbl_fps = QLabel("FPS: --"); self.lbl_chunk = QLabel("Batch: -"); self.lbl_time = QLabel("Left: --"); self.lbl_obj = QLabel("Obj: 0")
        for l in [self.lbl_fps, self.lbl_chunk, self.lbl_time, self.lbl_obj]: l.setStyleSheet("color: #0dcaf0; font-weight: bold;"); stats_box.addWidget(l); stats_box.addStretch()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.text_app = QTextEdit(); self.text_app.setReadOnly(True); self.text_app.setStyleSheet("background:#111; color:#0f0;")
        self.text_sys = QTextEdit(); self.text_sys.setReadOnly(True); self.text_sys.setStyleSheet("background:#000; color:#aaa;")
        splitter.addWidget(self.text_app); splitter.addWidget(self.text_sys); splitter.setSizes([350, 450])
        self.pbar = QProgressBar(); self.pbar.setValue(0); self.pbar.setStyleSheet("text-align:center; chunk{background:#0d6efd;}")
        log_layout.addWidget(stats_w); log_layout.addWidget(splitter, 1); log_layout.addWidget(self.pbar); log_container.setLayout(log_layout)
        
        center_layout.addWidget(vis_group, 2); center_layout.addWidget(log_container, 1)
        
        content_layout.addWidget(left_scroll)
        content_layout.addWidget(center_panel, 1)

        root_layout.addLayout(content_layout)

    # --- Helpers ---
    def create_button(self, t, c):
        b = QPushButton(t); b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet(f"QPushButton{{background-color:{c};color:white;border-radius:4px;padding:6px;border:none;}}QPushButton:hover{{background-color:{c}dd;}}QPushButton:disabled{{background-color:#444;}}")
        return b
        
    def update_class_id(self, val): self.video_selector.set_current_class(val)

    # --- Logic ---
    def select_sam_folder(self): f=QFileDialog.getExistingDirectory(self); self.line_sam_path.setText(f) if f else None
    def select_output_folder(self): f=QFileDialog.getExistingDirectory(self); self.line_output.setText(f) if f else None
    
    def select_files(self): 
        fs,_=QFileDialog.getOpenFileNames(self,"Sel","","Video (*.mp4 *.avi)"); 
        if fs: [self.add_file(f) for f in fs]
    def select_folder(self): 
        d=QFileDialog.getExistingDirectory(self); 
        if d: [self.add_file(os.path.join(r,f)) for r,_,fs in os.walk(d) for f in fs if f.endswith('.mp4')]
    def add_file(self, f): 
        if f not in self.added_paths: self.added_paths.add(f); i=QListWidgetItem(os.path.basename(f)); i.setData(Qt.ItemDataRole.UserRole, f); i.setCheckState(Qt.CheckState.Checked); self.list_videos.addItem(i); self.project_annotations[f]={}
    def clear_videos(self): self.added_paths.clear(); self.list_videos.clear(); self.project_annotations.clear(); self.video_selector.clear_all()

    def load_preview_video(self, row):
        if row < 0: return
        # Save previous state
        if self.current_video_path and self.current_video_path in self.project_annotations:
            self.project_annotations[self.current_video_path] = self.video_selector.annotations

        path = self.list_videos.item(row).data(Qt.ItemDataRole.UserRole)
        self.current_video_path = path
        try:
            if self.current_video_cap: self.current_video_cap.release()
            self.current_video_cap = cv2.VideoCapture(path)
            self.total_frames = int(self.current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames - 1)
            
            # Load stored annotations
            saved_anns = self.project_annotations.get(path, {})
            self.video_selector.set_annotations(saved_anns)
            
            self.slider.setValue(0); self.show_frame(0)
            self.update_app_log(f"Loaded: {os.path.basename(path)}")
        except Exception as e: self.update_app_log(f"Error: {e}")

    def change_frame_relative(self, d):
        if self.total_frames: self.slider.setValue(max(0, min(self.slider.value()+d, self.total_frames-1))); self.show_frame(self.slider.value())
    def slider_released(self): self.show_frame(self.slider.value())
    def slider_moved(self, v): self.lbl_frame.setText(f"Frame: {v}/{self.total_frames}")
    def show_frame(self, idx):
        if self.current_video_cap:
            self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret,f=self.current_video_cap.read()
            if ret: self.current_preview_frame=f; self.video_selector.set_current_frame(idx, f); self.lbl_frame.setText(f"Frame: {idx}/{self.total_frames}")

    def on_selection_changed(self):
        if self.current_video_path: self.project_annotations[self.current_video_path] = self.video_selector.annotations
        t = sum(len(v) for v in self.video_selector.annotations.values())
        self.lbl_info.setText(f"{t} Boxes / {len(self.video_selector.annotations)} Frames.")

    def clear_all_annotations(self): self.video_selector.clear_all(); self.project_annotations[self.current_video_path]={} if self.current_video_path else None
    
    def save_annotations(self):
        if not self.project_annotations: return
        p,_=QFileDialog.getSaveFileName(self,"Save","","JSON (*.json)")
        if p:
            d = {}
            for v_path, ann in self.project_annotations.items():
                d[v_path] = {str(k): [[r.x(),r.y(),r.width(),r.height(), cid] for r, cid in v] for k,v in ann.items()}
            with open(p,'w') as f: json.dump(d,f)
            self.update_app_log(f"Saved: {p}")
    def load_annotations(self):
        p,_=QFileDialog.getOpenFileName(self,"Load","","JSON (*.json)")
        if p and os.path.exists(p):
            with open(p,'r') as f: d=json.load(f)
            self.project_annotations = {}
            for path, frames in d.items():
                parsed = {}
                for k, v in frames.items():
                    objs = []
                    for item in v:
                        if len(item) == 5: objs.append((QRect(*item[:4]), item[4]))
                        else: objs.append((QRect(*item), 0))
                    parsed[int(k)] = objs
                self.project_annotations[path] = parsed
            if self.current_video_path in self.project_annotations:
                self.video_selector.set_annotations(self.project_annotations[self.current_video_path])
                self.show_frame(self.slider.value())
            self.update_app_log("Loaded.")

    def toggle_manual_mode(self, c): self.spin_threshold.setEnabled(not c); self.btn_preview.setEnabled(not c)
    def update_app_log(self, m): self.text_app.append(f"> {m}"); self.text_app.verticalScrollBar().setValue(self.text_app.verticalScrollBar().maximum())
    def update_sys_log(self, m): 
        if not m: return
        cl = m.replace('\r', ''); c = self.text_sys.textCursor(); c.movePosition(QTextCursor.MoveOperation.End)
        if '\r' in m: c.insertText(cl+"\n")
        else: c.insertText(cl.strip()+"\n") if cl.strip() else None
        self.text_sys.verticalScrollBar().setValue(self.text_sys.verticalScrollBar().maximum())

    def run_preview_detection(self):
        if self.chk_manual_mode.isChecked(): return
        if not self.current_preview_frame is None:
            tmpls = self.video_selector.get_current_boxes()
            if not tmpls: return
            gray = cv2.cvtColor(self.current_preview_frame, cv2.COLOR_BGR2GRAY); det = []
            th = self.spin_threshold.value()
            for r in tmpls:
                x,y,w,h=r.x(),r.y(),r.width(),r.height()
                if w<2: continue
                res = cv2.matchTemplate(gray, gray[y:y+h, x:x+w], cv2.TM_CCOEFF_NORMED)
                loc = np.where(res>=th)
                for pt in zip(*loc[::-1]):
                    # Simplified NMS check
                    det.append((pt[0], pt[1], w, h))
            self.video_selector.set_detected_objects(det); self.update_app_log(f"Preview: {len(det)} found.")

    def start_analysis(self):
        active = []
        for i in range(self.list_videos.count()):
            if self.list_videos.item(i).checkState() == Qt.CheckState.Checked:
                path = self.list_videos.item(i).data(Qt.ItemDataRole.UserRole)
                ann = self.project_annotations.get(path, {})
                if path == self.current_video_path and not ann and self.video_selector.annotations: ann = self.video_selector.annotations
                active.append((path, ann))
        
        if not active: QMessageBox.warning(self, "Error", "No videos."); return
        
        if self.chk_manual_mode.isChecked():
            bad = [os.path.basename(p) for p,a in active if not a]
            if bad: QMessageBox.warning(self, "Manual Error", f"Missing boxes:\n{', '.join(bad)}"); return

        sam_path = self.line_sam_path.text()
        if not os.path.exists(sam_path): QMessageBox.critical(self,"Err","Bad SAM Path"); return

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); final = os.path.join(self.line_output.text(), f"Run_{ts}"); os.makedirs(final, exist_ok=True)
        self.set_ui_processing(True); self.pbar.setValue(0); self.text_app.clear(); self.text_sys.clear()
        
        cfg = {
            "match_threshold": self.spin_threshold.value(), "pixel_scale_um": self.spin_pixel_scale.value(),
            "batch_size": self.spin_batch.value(), "chunk_duration": self.spin_chunk.value(),
            "sam_path": sam_path, "save_video": self.chk_video.isChecked(),
            "save_csv": self.chk_csv.isChecked(), "save_heatmap": self.chk_heatmap.isChecked(),
            "save_traj": self.chk_traj.isChecked(), "save_hist": self.chk_hist.isChecked(),
            "fast_mode": not self.chk_video.isChecked(), "manual_mode": self.chk_manual_mode.isChecked()
        }
        
        # --- FIX: Passing exactly 3 arguments as expected by Worker ---
        self.worker = AnalysisWorker(active, final, cfg)
        
        self.worker.log_app_signal.connect(self.update_app_log); self.worker.log_sys_signal.connect(self.update_sys_log)
        self.worker.progress_signal.connect(lambda v, m: (self.pbar.setValue(v), self.pbar.setFormat(f"{v}% - {m}")))
        self.worker.stats_signal.connect(self.update_stats); self.worker.error_signal.connect(self.handle_error); self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()

    def stop_analysis(self): 
        if self.worker: self.update_app_log("Stopping..."); self.worker.stop(); self.btn_stop.setText("Stopping..."); self.btn_stop.setEnabled(False)
    def update_stats(self, s): self.lbl_fps.setText(f"FPS: {s.get('fps',0)}"); self.lbl_time.setText(f"T: {s.get('time_left','-')}"); self.lbl_obj.setText(f"Obj: {s.get('objects',0)}"); self.lbl_chunk.setText(f"B: {s.get('batch','-')}")
    def handle_error(self, m): self.update_app_log(f"ERR: {m}"); self.set_ui_processing(False)
    def analysis_finished(self): self.update_app_log("Done."); self.set_ui_processing(False); QMessageBox.information(self, "Done", "Complete")
    
    def set_ui_processing(self, a):
        self.btn_start.setEnabled(not a); self.btn_stop.setEnabled(a)
        # Disable all
        for w in [self.spin_threshold, self.spin_pixel_scale, self.spin_batch, self.spin_chunk, self.line_sam_path, self.btn_sam_browse, self.line_output, self.btn_browse, self.btn_add_file, self.btn_add_folder, self.btn_clear, self.btn_del_box, self.btn_clear_frame, self.btn_clear_all, self.btn_save_ann, self.btn_load_ann, self.list_videos, self.video_selector, self.btn_preview, self.chk_video, self.chk_csv, self.chk_heatmap, self.chk_traj, self.chk_hist, self.chk_manual_mode, self.slider, self.btn_prev, self.btn_next, self.spin_class, self.lbl_class]: w.setEnabled(not a)