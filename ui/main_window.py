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
        self.setWindowTitle("AutoTrackSAMYOLO")
        self.resize(1400, 950)
        self.setMinimumSize(1100, 800)
        
        # --- Internal State ---
        self.added_paths = set() 
        self.worker = None 
        self.current_video_path = None
        self.current_video_cap = None
        self.total_frames = 0
        self.current_preview_frame = None
        
        # Default Paths
        self.output_dir = os.path.join(os.getcwd(), "SAM3_Results")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Default SAM Path (Adjust if needed)
        self.default_sam_path = "/mnt/Zebrafish_24TB/SAM3-Development/sam3"
        
        # --- Initialize UI ---
        self.setup_ui()
        
    def setup_ui(self):
        """Builds the entire GUI layout with Header and Scrollable Settings."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Root Layout (Vertical): Header + Content
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
        
        # Home Button
        self.btn_home = QPushButton("🏠  Home")
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setFixedWidth(100)
        self.btn_home.setStyleSheet("""
            QPushButton { background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #444; border-color: #777; }
        """)
        self.btn_home.clicked.connect(self.go_home_signal.emit)
        
        # Title
        lbl_title = QLabel("SAM 3 Tracking Pipeline")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #eee; border: none;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Spacer
        dummy_spacer = QWidget()
        dummy_spacer.setFixedWidth(100)
        
        header_layout.addWidget(self.btn_home)
        header_layout.addWidget(lbl_title, 1) 
        header_layout.addWidget(dummy_spacer)
        
        root_layout.addWidget(header_widget)

        # =================================================
        # MAIN CONTENT AREA (Split Left/Right)
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
        
        # --- Group 1: Video Input ---
        file_group = QGroupBox("1. Video Input")
        file_layout = QVBoxLayout()
        
        btn_row = QHBoxLayout()
        self.btn_add_file = self.create_button("Add File", "#0d6efd")
        self.btn_add_file.clicked.connect(self.select_files)
        
        self.btn_add_folder = self.create_button("Add Folder", "#0d6efd")
        self.btn_add_folder.clicked.connect(self.select_folder)
        
        self.btn_clear = self.create_button("Clear", "#6c757d")
        self.btn_clear.clicked.connect(self.clear_videos)
        
        btn_row.addWidget(self.btn_add_file)
        btn_row.addWidget(self.btn_add_folder)
        btn_row.addWidget(self.btn_clear)

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
        
        file_layout.addLayout(btn_row)
        file_layout.addWidget(self.list_videos)
        file_group.setLayout(file_layout)

        # --- Group 2: Configuration ---
        settings_group = QGroupBox("2. Configuration & Tuning")
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # A. Local SAM 3 Path
        sam_path_layout = QHBoxLayout()
        self.line_sam_path = QLineEdit(self.default_sam_path)
        self.line_sam_path.setPlaceholderText("Select Local SAM3 Repo...")
        self.line_sam_path.setCursor(Qt.CursorShape.IBeamCursor)
        self.line_sam_path.setStyleSheet("background-color: #1e1e1e; color: #ddd; border: 1px solid #444; padding: 4px;")
        
        self.btn_sam_browse = self.create_button("...", "#495057")
        self.btn_sam_browse.setFixedWidth(30)
        self.btn_sam_browse.clicked.connect(self.select_sam_folder)
        
        sam_path_layout.addWidget(self.line_sam_path)
        sam_path_layout.addWidget(self.btn_sam_browse)
        form_layout.addRow("Local SAM 3:", sam_path_layout)

        # B. Manual vs Auto Mode
        self.chk_manual_mode = QCheckBox("Manual Mode (Strict Tracking)")
        self.chk_manual_mode.setToolTip("If checked, SAM tracks ONLY the boxes you draw on Frame 0.\nIf unchecked, it uses them as templates to find similar particles.")
        self.chk_manual_mode.setStyleSheet("color: #0dcaf0; font-weight: bold;")
        self.chk_manual_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_manual_mode.toggled.connect(self.toggle_manual_mode)
        form_layout.addRow("", self.chk_manual_mode)

        # C. Sensitivity
        thresh_layout = QHBoxLayout()
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.1, 0.99)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.60)
        self.spin_threshold.setToolTip("Lower = Finds more particles (risk of noise).\nHigher = Stricter matching.")
        self.spin_threshold.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_threshold.wheelEvent = lambda event: event.ignore()
        
        self.btn_preview = self.create_button("Preview", "#e0a800")
        self.btn_preview.setStyleSheet("background-color: #e0a800; color: black; font-weight: bold; border-radius: 4px;")
        self.btn_preview.clicked.connect(self.run_preview_detection)
        
        thresh_layout.addWidget(self.spin_threshold)
        thresh_layout.addWidget(self.btn_preview)
        form_layout.addRow("Sensitivity:", thresh_layout)

        # D. Calibration
        self.spin_pixel_scale = QDoubleSpinBox()
        self.spin_pixel_scale.setRange(0.0001, 100.0)
        self.spin_pixel_scale.setDecimals(4)
        self.spin_pixel_scale.setValue(0.3240)
        self.spin_pixel_scale.setSuffix(" µm")
        self.spin_pixel_scale.setToolTip("Physical size of 1 pixel in micrometers.\nDefault: 324nm = 0.324µm")
        self.spin_pixel_scale.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_pixel_scale.wheelEvent = lambda event: event.ignore()
        form_layout.addRow("Pixel Scale:", self.spin_pixel_scale)

        # E. Batch Size
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(16)
        self.spin_batch.setToolTip("Objects to track simultaneously.\nDecrease for 12GB GPUs.")
        self.spin_batch.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_batch.wheelEvent = lambda event: event.ignore()
        form_layout.addRow("GPU Batch:", self.spin_batch)
        
        # F. Chunk Duration
        self.spin_chunk = QSpinBox()
        self.spin_chunk.setRange(2, 60)
        self.spin_chunk.setValue(5)
        self.spin_chunk.setSuffix(" s")
        self.spin_chunk.setToolTip("Video split duration.\nShorter = Safer for RAM.")
        self.spin_chunk.setCursor(Qt.CursorShape.IBeamCursor)
        self.spin_chunk.wheelEvent = lambda event: event.ignore()
        form_layout.addRow("Split Time:", self.spin_chunk)

        settings_group.setLayout(form_layout)

        # --- Group 3: Output File Types ---
        out_type_group = QGroupBox("3. Output Files")
        out_type_layout = QVBoxLayout()
        out_type_layout.setSpacing(5)
        
        self.chk_video = QCheckBox("Annotated Video (.mp4)")
        self.chk_video.setChecked(True)
        self.chk_video.setToolTip("Uncheck to enable FAST MODE (Skips rendering).")
        self.chk_video.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_video.setStyleSheet("color: #ffc107; font-weight: bold;") 
        
        self.chk_csv = QCheckBox("Data Spreadsheet (.csv)")
        self.chk_csv.setChecked(True)
        self.chk_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.chk_heatmap = QCheckBox("Heatmap (.png)")
        self.chk_heatmap.setChecked(True)
        self.chk_heatmap.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.chk_traj = QCheckBox("Trajectories (.png)")
        self.chk_traj.setChecked(True)
        self.chk_traj.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.chk_hist = QCheckBox("Size Histogram (.png)")
        self.chk_hist.setChecked(True)
        self.chk_hist.setCursor(Qt.CursorShape.PointingHandCursor)
        
        out_type_layout.addWidget(self.chk_video)
        out_type_layout.addWidget(self.chk_csv)
        out_type_layout.addWidget(self.chk_heatmap)
        out_type_layout.addWidget(self.chk_traj)
        out_type_layout.addWidget(self.chk_hist)
        out_type_group.setLayout(out_type_layout)

        # --- Group 4: Output Location ---
        out_loc_group = QGroupBox("4. Output Location")
        out_layout = QVBoxLayout()
        
        self.line_output = QLineEdit(self.output_dir)
        self.line_output.setReadOnly(True)
        self.line_output.setCursor(Qt.CursorShape.IBeamCursor)
        self.line_output.setStyleSheet("background-color: #1e1e1e; color: #aaa; border: 1px solid #444; padding: 5px;")
        
        self.btn_browse = self.create_button("Browse Output...", "#495057")
        self.btn_browse.clicked.connect(self.select_output_folder)
        
        out_layout.addWidget(self.line_output)
        out_layout.addWidget(self.btn_browse)
        out_loc_group.setLayout(out_layout)

        # --- Group 5: Run Control ---
        run_group = QGroupBox("5. Execute")
        run_layout = QVBoxLayout()
        
        self.btn_start = self.create_button("START BATCH ANALYSIS", "#198754")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("background-color: #198754; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_analysis)
        
        self.btn_stop = self.create_button("STOP", "#dc3545")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        
        run_layout.addWidget(self.btn_start)
        run_layout.addWidget(self.btn_stop)
        run_group.setLayout(run_layout)
        
        # Add groups to Left Layout
        left_layout.addWidget(file_group) 
        left_layout.addWidget(settings_group)
        left_layout.addWidget(out_type_group)
        left_layout.addWidget(out_loc_group)
        left_layout.addWidget(run_group)
        left_layout.addStretch() 
        
        left_scroll.setWidget(left_widget)

        # -------------------------------------------------
        # 2. CENTER PANEL (Video & Logs)
        # -------------------------------------------------
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Visualizer ---
        vis_group = QGroupBox("Visualizer: Draw Templates & Preview")
        vis_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #0d6efd; background-color: #1e1e1e; }")
        
        vis_layout = QVBoxLayout()
        vis_layout.setContentsMargins(5, 15, 5, 5)
        
        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_frame.setStyleSheet("color: #fff; font-weight: bold;")
        
        self.lbl_info = QLabel("Select Video -> Draw Boxes. Move slider to annotate other frames.")
        self.lbl_info.setStyleSheet("color: #ccc; font-style: italic;")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.video_selector = VideoSelectorWidget()
        self.video_selector.selection_changed.connect(self.on_selection_changed)
        
        # Timeline Controls
        timeline_layout = QHBoxLayout()
        self.btn_prev = self.create_button("<", "#555555")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(lambda: self.change_frame_relative(-10))
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.valueChanged.connect(self.slider_moved)
        
        self.btn_next = self.create_button(">", "#555555")
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(lambda: self.change_frame_relative(10))
        
        timeline_layout.addWidget(self.btn_prev)
        timeline_layout.addWidget(self.slider)
        timeline_layout.addWidget(self.btn_next)
        
        # Tool Buttons
        tools_layout = QHBoxLayout()
        self.btn_del_box = self.create_button("Delete Selected Box", "#d9534f")
        self.btn_del_box.clicked.connect(self.video_selector.delete_selected)
        
        self.btn_clear_frame = self.create_button("Clear Frame", "#6c757d")
        self.btn_clear_frame.clicked.connect(self.video_selector.clear_current_frame)
        
        self.btn_clear_all = self.create_button("Clear All Frames", "#343a40")
        self.btn_clear_all.clicked.connect(self.video_selector.clear_all)
        
        self.btn_save_ann = self.create_button("Save JSON", "#17a2b8")
        self.btn_save_ann.clicked.connect(self.save_annotations)
        
        self.btn_load_ann = self.create_button("Load JSON", "#17a2b8")
        self.btn_load_ann.clicked.connect(self.load_annotations)

        tools_layout.addWidget(self.btn_del_box)
        tools_layout.addWidget(self.btn_clear_frame)
        tools_layout.addWidget(self.btn_clear_all)
        tools_layout.addStretch()
        tools_layout.addWidget(self.btn_save_ann)
        tools_layout.addWidget(self.btn_load_ann)
        
        vis_layout.addWidget(self.lbl_info)
        vis_layout.addWidget(self.lbl_frame)
        vis_layout.addWidget(self.video_selector, 1) 
        vis_layout.addLayout(timeline_layout)
        vis_layout.addLayout(tools_layout)
        vis_group.setLayout(vis_layout)
        
        # --- SPLIT LOGS ---
        log_container = QGroupBox("Logs & Statistics")
        log_layout = QVBoxLayout()
        
        # Stats Bar
        stats_widget = QWidget()
        stats_box = QHBoxLayout(stats_widget)
        stats_box.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet("color: #0dcaf0; font-weight: bold; font-size: 14px;")
        self.lbl_chunk = QLabel("Batch: -/-")
        self.lbl_chunk.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px;")
        self.lbl_time = QLabel("Left: --:--")
        self.lbl_time.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 14px;")
        self.lbl_obj = QLabel("Objects: 0")
        self.lbl_obj.setStyleSheet("color: #20c997; font-weight: bold; font-size: 14px;")
        
        stats_box.addWidget(self.lbl_fps)
        stats_box.addStretch()
        stats_box.addWidget(self.lbl_chunk)
        stats_box.addStretch()
        stats_box.addWidget(self.lbl_time)
        stats_box.addStretch()
        stats_box.addWidget(self.lbl_obj)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: App Log
        app_log_widget = QWidget(); app_log_vbox = QVBoxLayout(app_log_widget); app_log_vbox.setContentsMargins(0,0,0,0)
        lbl_app = QLabel("Application Status"); lbl_app.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.text_log_app = QTextEdit(); self.text_log_app.setReadOnly(True); self.text_log_app.setCursor(Qt.CursorShape.IBeamCursor)
        self.text_log_app.setStyleSheet("background-color: #111; color: #00ff00; font-family: Consolas; border: 1px solid #444;")
        app_log_vbox.addWidget(lbl_app); app_log_vbox.addWidget(self.text_log_app)
        
        # Right: System Log
        sys_log_widget = QWidget(); sys_log_vbox = QVBoxLayout(sys_log_widget); sys_log_vbox.setContentsMargins(0,0,0,0)
        lbl_sys = QLabel("System / SAM3 Output"); lbl_sys.setStyleSheet("color: #aaa; font-weight: bold;")
        self.text_log_sys = QTextEdit(); self.text_log_sys.setReadOnly(True); self.text_log_sys.setCursor(Qt.CursorShape.IBeamCursor)
        self.text_log_sys.setStyleSheet("background-color: #000; color: #aaa; font-family: Consolas; font-size: 11px; border: 1px solid #444;")
        sys_log_vbox.addWidget(lbl_sys); sys_log_vbox.addWidget(self.text_log_sys)
        
        splitter.addWidget(app_log_widget)
        splitter.addWidget(sys_log_widget)
        splitter.setSizes([350, 450]) 
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #0d6efd; border-radius: 4px; }")
        
        log_layout.addWidget(stats_widget)
        log_layout.addWidget(splitter, 1) 
        log_layout.addWidget(self.progress_bar)
        log_container.setLayout(log_layout)
        
        center_layout.addWidget(vis_group, 2)
        center_layout.addWidget(log_container, 1) 
        
        content_layout.addWidget(left_scroll)
        content_layout.addWidget(center_panel, 1)

        root_layout.addLayout(content_layout)

    # =================================================
    # HELPER: BUTTON FACTORY
    # =================================================
    def create_button(self, text, bg_color):
        """Creates a styled button with a pointing hand cursor."""
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border-radius: 4px;
                padding: 6px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {bg_color}dd; 
            }}
            QPushButton:disabled {{
                background-color: #444;
                color: #888;
            }}
        """)
        return btn

    # =================================================
    # LOGIC METHODS
    # =================================================
    def toggle_manual_mode(self, checked):
        """Enables/Disables detection controls based on Manual Mode."""
        self.spin_threshold.setEnabled(not checked)
        self.btn_preview.setEnabled(not checked)
        if checked:
            self.lbl_info.setText("MANUAL MODE: Only drawn boxes will be tracked. No auto-detection.")
            self.update_app_log("Manual Mode Enabled.")
        else:
            self.lbl_info.setText("AUTO MODE: Drawn boxes act as templates for search.")
            self.update_app_log("Auto Mode Enabled.")

    def update_app_log(self, msg):
        """Updates the left log (App Status)."""
        self.text_log_app.append(f">> {msg}")
        self.text_log_app.verticalScrollBar().setValue(self.text_log_app.verticalScrollBar().maximum())

    def update_sys_log(self, msg):
        """Updates the right log (System Output), handling \r for progress bars."""
        if not msg: return
        cl = msg.replace('\r', '') 
        cursor = self.text_log_sys.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if '\r' in msg and "it/s" in msg:
            cursor.insertText(cl + "\n")
        else:
            if cl.strip():
                cursor.insertText(cl + "\n")
        self.text_log_sys.verticalScrollBar().setValue(self.text_log_sys.verticalScrollBar().maximum())

    def run_preview_detection(self):
        """Runs a quick template match on the current frame."""
        if self.chk_manual_mode.isChecked():
            QMessageBox.information(self, "Manual Mode", "Preview disabled in Manual Mode.")
            return

        if self.current_preview_frame is None:
            QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        
        # FIX: Use get_current_boxes() instead of direct .templates
        templates = self.video_selector.get_current_boxes()
        if not templates:
            QMessageBox.warning(self, "No Templates", "Draw at least one green box on THIS frame.")
            return
            
        threshold = self.spin_threshold.value()
        self.update_app_log(f"Running Preview... (Threshold: {threshold})")
        
        gray_frame = cv2.cvtColor(self.current_preview_frame, cv2.COLOR_BGR2GRAY)
        detected_rects = []
        
        for r in templates:
            x, y, w, h = r.x(), r.y(), r.width(), r.height()
            if w < 2 or h < 2: continue
            
            template_img = gray_frame[y:y+h, x:x+w]
            res = cv2.matchTemplate(gray_frame, template_img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            min_dist = min(w, h) / 1.5
            
            for pt in zip(*loc[::-1]):
                cx = pt[0] + w // 2
                cy = pt[1] + h // 2
                is_new = True
                for (dx, dy, dw, dh) in detected_rects:
                    dcx = dx + dw // 2
                    dcy = dy + dh // 2
                    if np.sqrt((cx - dcx)**2 + (cy - dcy)**2) < min_dist:
                        is_new = False
                        break
                if is_new:
                    detected_rects.append((pt[0], pt[1], w, h))
        
        count = len(detected_rects)
        self.update_app_log(f"Preview Results: {count} objects found.")
        self.lbl_info.setText(f"Preview: {count} objects found. Adjust Sensitivity if needed.")
        self.video_selector.set_detected_objects(detected_rects)

    def select_sam_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Local SAM3 Repo")
        if folder:
            self.line_sam_path.setText(folder)
            self.update_app_log(f"Selected SAM3 Path: {folder}")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir = folder
            self.line_output.setText(self.output_dir)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Videos (*.mp4 *.avi *.mov)")
        if files:
            for f in files:
                if f not in self.added_paths:
                    self.added_paths.add(f)
                    item = QListWidgetItem(os.path.basename(f))
                    item.setData(Qt.ItemDataRole.UserRole, f)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    item.setCheckState(Qt.CheckState.Checked)
                    self.list_videos.addItem(item)
            self.update_app_log(f"Added {len(files)} videos.")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov')):
                        full = os.path.join(root, f)
                        if full not in self.added_paths:
                            self.added_paths.add(full)
                            item = QListWidgetItem(f)
                            item.setData(Qt.ItemDataRole.UserRole, full)
                            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                            item.setCheckState(Qt.CheckState.Checked)
                            self.list_videos.addItem(item)
                break 

    def clear_videos(self):
        self.added_paths.clear()
        self.list_videos.clear()
        self.video_selector.clear_selection()
        self.video_selector.update()
        self.update_app_log("Video list cleared.")

    def load_preview_video(self, row):
        if row < 0: return
        item = self.list_videos.item(row)
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            if self.current_video_cap: self.current_video_cap.release()
            self.current_video_cap = cv2.VideoCapture(path)
            self.total_frames = int(self.current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setValue(0)
            self.show_frame(0)
            self.update_app_log(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self.update_app_log(f"Exception: {e}")

    def change_frame_relative(self, delta):
        if self.total_frames == 0: return
        new_pos = max(0, min(self.slider.value() + delta, self.total_frames - 1))
        self.slider.setValue(new_pos); self.show_frame(new_pos)

    def slider_released(self):
        self.show_frame(self.slider.value())

    def slider_moved(self, val):
        self.lbl_frame.setText(f"Frame: {val} / {self.total_frames}")

    def show_frame(self, idx):
        if not self.current_video_cap: return
        self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.current_video_cap.read()
        if ret:
            self.current_preview_frame = frame
            self.video_selector.set_current_frame(idx, frame)
            # Box count update happens in video_selector, but we can update frame label
            current_boxes = len(self.video_selector.get_current_boxes())
            self.lbl_frame.setText(f"Frm: {idx}/{self.total_frames} | Box: {current_boxes}")

    def on_selection_changed(self):
        # Update Label in Timeline
        idx = self.video_selector.current_frame_idx
        current_boxes = len(self.video_selector.get_current_boxes())
        self.lbl_frame.setText(f"Frm: {idx}/{self.total_frames} | Box: {current_boxes}")
        
        total = sum(len(v) for v in self.video_selector.annotations.values())
        frames = len(self.video_selector.annotations)
        self.lbl_info.setText(f"{total} Boxes across {frames} Frames.")

    def save_annotations(self):
        if not self.video_selector.annotations:
            QMessageBox.warning(self, "Empty", "No annotations to save.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", os.path.join(self.output_dir, "settings.json"), "JSON (*.json)")
        if path:
            data = {}
            for f_idx, rects in self.video_selector.annotations.items():
                data[str(f_idx)] = [[r.x(), r.y(), r.width(), r.height()] for r in rects]
            
            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=4)
                self.update_app_log(f"Saved annotations to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_annotations(self):
        default_path = os.path.join(self.output_dir, "settings.json")
        path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", default_path, "JSON (*.json)")
        
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                loaded = {}
                count = 0
                for k, v in data.items():
                    frame_idx = int(k)
                    rects = []
                    for box in v:
                        rects.append(QRect(box[0], box[1], box[2], box[3]))
                    
                    loaded[frame_idx] = rects
                    count += len(rects)
                
                self.video_selector.annotations = loaded
                
                # Refresh current frame view to show loaded boxes
                self.video_selector.set_current_frame(self.slider.value(), self.current_preview_frame)
                
                self.update_app_log(f"Loaded {count} boxes for {len(loaded)} frames.")
                self.on_selection_changed() 
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def start_analysis(self):
            # 1. Collect Selected Videos
            active = []
            for i in range(self.list_videos.count()):
                if self.list_videos.item(i).checkState() == Qt.CheckState.Checked:
                    active.append(self.list_videos.item(i).data(Qt.ItemDataRole.UserRole))
            
            if not active: 
                QMessageBox.warning(self, "Error", "No videos selected.")
                return
            
            # 2. Collect Annotations (Dictionary: {frame_idx: [boxes]})
            templates_dict = self.video_selector.annotations
            if not templates_dict:
                if QMessageBox.question(self, "No Templates", "Proceed with FULL SCAN?", 
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No: 
                    return
                templates_dict = {} # Handle as empty if user says yes

            # 3. Validate SAM Path
            sam_path_val = self.line_sam_path.text()
            if not os.path.exists(sam_path_val):
                QMessageBox.critical(self, "Error", f"Local SAM3 path missing:\n{sam_path_val}")
                return

            # 4. CRITICAL: Cleanup Previous Worker (Prevents Re-run Crash)
            if self.worker is not None:
                if self.worker.isRunning():
                    QMessageBox.warning(self, "Busy", "Analysis is currently running. Please Stop it first.")
                    return
                self.worker.deleteLater()
                self.worker = None

            # 5. Prepare Output Directory
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            final_out = os.path.join(self.output_dir, f"Run_{timestamp}")
            os.makedirs(final_out, exist_ok=True)

            # 6. Reset UI State
            self.set_ui_processing(True)
            self.progress_bar.setValue(0)
            self.text_log_app.clear()
            self.text_log_sys.clear()
            
            # 7. Build Configuration
            config = {
                "match_threshold": self.spin_threshold.value(),
                "pixel_scale_um": self.spin_pixel_scale.value(),
                "batch_size": self.spin_batch.value(),
                "chunk_duration": self.spin_chunk.value(),
                "sam_path": sam_path_val,
                "save_video": self.chk_video.isChecked(),
                "save_csv": self.chk_csv.isChecked(),
                "save_heatmap": self.chk_heatmap.isChecked(),
                "save_traj": self.chk_traj.isChecked(),
                "save_hist": self.chk_hist.isChecked(),
                "fast_mode": not self.chk_video.isChecked(),
                "manual_mode": self.chk_manual_mode.isChecked()
            }
            
            # 8. Start New Worker
            self.worker = AnalysisWorker(active, templates_dict, final_out, config)
            self.worker.log_app_signal.connect(self.update_app_log)
            self.worker.log_sys_signal.connect(self.update_sys_log)
            self.worker.progress_signal.connect(lambda v, m: (self.progress_bar.setValue(v), self.progress_bar.setFormat(f"{v}% - {m}")))
            self.worker.stats_signal.connect(self.update_stats)
            self.worker.error_signal.connect(self.handle_error)
            self.worker.finished_signal.connect(self.analysis_finished)
            self.worker.start()
    def stop_analysis(self):
        if self.worker: 
            self.update_app_log("Stopping...")
            self.worker.stop()
            self.btn_stop.setText("Stopping...")
            self.btn_stop.setEnabled(False)

    def update_stats(self, stats):
        if 'fps' in stats: self.lbl_fps.setText(f"FPS: {stats.get('fps', 0)}")
        if 'time_left' in stats: self.lbl_time.setText(f"Left: {stats.get('time_left', '--:--')}")
        if 'objects' in stats: self.lbl_obj.setText(f"Objects: {stats.get('objects', 0)}")
        if 'chunk' in stats: self.lbl_chunk.setText(f"Batch: {stats.get('chunk', '-')}")

    def handle_error(self, msg):
        self.update_app_log(f"ERROR: {msg}")
        self.set_ui_processing(False)

    def analysis_finished(self):
        self.update_app_log("Done.")
        self.set_ui_processing(False)
        QMessageBox.information(self, "Done", "Analysis Complete")

    def set_ui_processing(self, active):
        self.btn_start.setEnabled(not active)
        self.btn_stop.setEnabled(active)
        self.btn_stop.setText("STOP")
        
        # Disable all inputs - Corrected Variable Names
        for w in [self.spin_threshold, self.spin_pixel_scale, self.spin_batch, self.spin_chunk, 
                  self.line_sam_path, self.btn_sam_browse, self.line_output, self.btn_browse,
                  self.btn_add_file, self.btn_add_folder, self.btn_clear, 
                  self.btn_clear_all, self.btn_clear_frame, # Fixed Names
                  self.btn_del_box, self.list_videos, self.video_selector, self.btn_preview, 
                  self.chk_video, self.chk_csv, self.chk_heatmap, self.chk_traj, self.chk_hist, 
                  self.chk_manual_mode, self.btn_save_ann, self.btn_load_ann,
                  self.slider, self.btn_prev, self.btn_next]:
            w.setEnabled(not active)