import os
import cv2
import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QTimeEdit, QCheckBox, 
    QGroupBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QLabel, QSplitter, QListWidget, QAbstractItemView, QComboBox,
    QSpinBox, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from ui.roi_selector import ROISelector
from core.ocr_logic import OCRVideoSplitter

class BatchOCRWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, videos, output_dir, crop_rect, start_time, end_time, stitch, date_format, ocr_config):
        super().__init__()
        self.videos = videos
        self.output_dir = output_dir
        self.crop_rect = crop_rect
        self.start_time = start_time
        self.end_time = end_time
        self.stitch = stitch
        self.date_format = date_format
        self.ocr_config = ocr_config # New Config Dict
        self.is_running = True
        self.current_logic = None

    def run(self):
        total_videos = len(self.videos)
        try:
            for i, video_path in enumerate(self.videos):
                if not self.is_running: break
                
                name = os.path.basename(video_path)
                self.log_signal.emit(f"=== Processing [{i+1}/{total_videos}]: {name} ===")
                
                # Pass config to logic
                self.current_logic = OCRVideoSplitter(
                    self.log_signal, None, self.date_format, self.ocr_config
                )
                
                self.progress_signal.emit(0, f"[{i+1}/{total_videos}] Scanning {name}...")
                self.current_logic.scan_video(video_path, self.crop_rect)
                
                if not self.is_running: break
                
                self.progress_signal.emit(50, f"[{i+1}/{total_videos}] Splitting {name}...")
                self.current_logic.execute_split(
                    video_path, self.output_dir, 
                    self.start_time, self.end_time, self.stitch
                )
                
                self.progress_signal.emit(100, f"Finished {name}")

            self.log_signal.emit("Batch processing finished.")
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self.is_running = False
        if self.current_logic: self.current_logic.stop_flag = True

class OCRSplitterWindow(QMainWindow):
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Video Splitter (Circadian Filter)")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 750)
        
        self.video_list = []
        self.current_frame = None
        self.worker = None
        
        self.setup_ui()

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root_layout = QVBoxLayout(central); root_layout.setContentsMargins(0,0,0,0); root_layout.setSpacing(0)

        # Header
        h = QHBoxLayout(); header_widget = QWidget(); header_widget.setStyleSheet("background-color: #252526; border-bottom: 1px solid #333;")
        h_layout = QHBoxLayout(header_widget); h_layout.setContentsMargins(15, 10, 15, 10)
        btn_home = QPushButton("🏠 Home"); btn_home.clicked.connect(self.go_home_signal.emit)
        btn_home.setCursor(Qt.CursorShape.PointingHandCursor); btn_home.setFixedWidth(100)
        btn_home.setStyleSheet("background:#444; color:white; border:1px solid #555; padding:6px; font-weight:bold;")
        h_layout.addWidget(btn_home); h_layout.addWidget(QLabel("Time-Based Splitter (OCR)", alignment=Qt.AlignmentFlag.AlignCenter), 1); h_layout.addStretch()
        root_layout.addWidget(header_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT PANEL ---
        left_w = QWidget(); left_l = QVBoxLayout(left_w); left_l.setContentsMargins(10, 10, 5, 10)
        
        # 1. Video Source
        grp_src = QGroupBox("1. Video Source"); f_src = QVBoxLayout()
        btn_row = QHBoxLayout()
        b_add = self.mk_btn("Add Files", "#0d6efd", self.select_files)
        b_clr = self.mk_btn("Clear", "#6c757d", self.clear_list)
        btn_row.addWidget(b_add); btn_row.addWidget(b_clr)
        self.list_widget = QListWidget(); self.list_widget.currentRowChanged.connect(self.load_preview_video)
        self.list_widget.setStyleSheet("background:#1e1e1e; color:#eee; border:1px solid #444;")
        f_src.addLayout(btn_row); f_src.addWidget(self.list_widget); grp_src.setLayout(f_src)
        left_l.addWidget(grp_src, 1)

        # 2. Time Settings
        grp_set = QGroupBox("2. Time Settings"); f_set = QFormLayout()
        self.ln_out = QLineEdit(os.path.join(os.getcwd(), "Split_Output"))
        self.ln_out.setStyleSheet("background:#222; color:#eee; padding:5px;")
        b_out = QPushButton("..."); b_out.clicked.connect(self.select_out); b_out.setFixedWidth(30)
        f_set.addRow("Output:", self.h_layout(self.ln_out, b_out))
        
        self.combo_format = QComboBox(); self.combo_format.setEditable(True)
        self.combo_format.addItems(["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %I:%M:%S%p", "%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"])
        self.combo_format.setStyleSheet("padding:5px; background:#222; color:white;")
        f_set.addRow("Format:", self.combo_format)
        
        self.time_start = QTimeEdit(); self.time_start.setDisplayFormat("HH:mm"); self.time_start.setTime(datetime.time(18,0))
        self.time_start.setStyleSheet("background:#222; color:#0d6efd; font-weight:bold; padding:5px;")
        self.time_end = QTimeEdit(); self.time_end.setDisplayFormat("HH:mm"); self.time_end.setTime(datetime.time(6,0))
        self.time_end.setStyleSheet("background:#222; color:#ffc107; font-weight:bold; padding:5px;")
        f_set.addRow("Start:", self.time_start); f_set.addRow("End:", self.time_end)
        
        self.chk_stitch = QCheckBox("Stitch segments"); self.chk_stitch.setChecked(True); self.chk_stitch.setStyleSheet("color:white;")
        f_set.addRow("", self.chk_stitch); grp_set.setLayout(f_set); left_l.addWidget(grp_set)

        # 3. Execution
        grp_run = QGroupBox("3. Execute"); run_l = QVBoxLayout()
        self.btn_run = self.mk_btn("START BATCH", "#198754", self.start); self.btn_run.setMinimumHeight(50)
        self.btn_stop = self.mk_btn("STOP", "#dc3545", self.stop_proc); self.btn_stop.setEnabled(False)
        self.pbar = QProgressBar(); self.pbar.setValue(0); self.pbar.setStyleSheet("text-align:center; chunk{background:#0d6efd;}")
        run_l.addWidget(self.btn_run); run_l.addWidget(self.btn_stop); run_l.addWidget(self.pbar)
        grp_run.setLayout(run_l); left_l.addWidget(grp_run)
        
        # Log
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(100)
        self.log_box.setStyleSheet("background:black; color:#0f0;"); left_l.addWidget(self.log_box)

        # --- RIGHT PANEL ---
        right_w = QWidget(); right_l = QVBoxLayout(right_w); right_l.setContentsMargins(5, 10, 10, 10)
        
        # 4. Advanced Tuning (New)
        grp_tune = QGroupBox("4. Advanced OCR Tuning (Live Preview)")
        grp_tune.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; } QGroupBox::title { color: #0d6efd; }")
        tune_layout = QHBoxLayout()
        
        # Threshold Method
        self.combo_thresh = QComboBox()
        self.combo_thresh.addItems(["Auto (Otsu)", "Manual"])
        self.combo_thresh.setStyleSheet("background:#222; color:white; padding:4px;")
        self.combo_thresh.currentIndexChanged.connect(self.toggle_manual_thresh)
        
        # Manual Threshold Value
        self.spin_thresh_val = QSpinBox()
        self.spin_thresh_val.setRange(0, 255); self.spin_thresh_val.setValue(127); self.spin_thresh_val.setEnabled(False)
        self.spin_thresh_val.setStyleSheet("background:#222; color:white;")
        self.spin_thresh_val.valueChanged.connect(self.roi_selected) # Live Update
        
        # Scale
        self.spin_scale = QSpinBox(); self.spin_scale.setRange(1, 6); self.spin_scale.setValue(4)
        self.spin_scale.setSuffix("x Zoom"); self.spin_scale.setStyleSheet("background:#222; color:white;")
        self.spin_scale.valueChanged.connect(self.roi_selected)
        
        # Invert
        self.chk_invert = QCheckBox("Force Invert")
        self.chk_invert.setToolTip("Toggle if text is Black on White")
        self.chk_invert.stateChanged.connect(self.roi_selected)
        
        tune_layout.addWidget(QLabel("Method:"))
        tune_layout.addWidget(self.combo_thresh)
        tune_layout.addWidget(QLabel("Val:"))
        tune_layout.addWidget(self.spin_thresh_val)
        tune_layout.addWidget(self.spin_scale)
        tune_layout.addWidget(self.chk_invert)
        
        grp_tune.setLayout(tune_layout)
        right_l.addWidget(grp_tune)

        # Visualizer
        vis_grp = QGroupBox("5. ROI Selector"); vis_l = QVBoxLayout()
        self.lbl_info = QLabel("Draw box around timestamp. Adjust tuning above to fix OCR errors."); self.lbl_info.setStyleSheet("color:#ccc; font-style:italic;")
        self.viewer = ROISelector()
        self.viewer.selection_changed.connect(self.roi_selected)
        
        vis_l.addWidget(self.lbl_info); vis_l.addWidget(self.viewer, 1); vis_grp.setLayout(vis_l)
        right_l.addWidget(vis_grp)
        
        splitter.addWidget(left_w); splitter.addWidget(right_w); splitter.setSizes([450, 800])
        root_layout.addWidget(splitter)

    # --- Helpers ---
    def mk_btn(self, t, c, s):
        b = QPushButton(t); b.setCursor(Qt.CursorShape.PointingHandCursor); b.clicked.connect(s)
        b.setStyleSheet(f"background:{c}; color:white; font-weight:bold; padding:6px;")
        return b
    def h_layout(self, w1, w2): w=QWidget(); l=QHBoxLayout(w); l.setContentsMargins(0,0,0,0); l.addWidget(w1); l.addWidget(w2); return w
    def log(self, m): self.log_box.append(m)

    # --- Logic ---
    def toggle_manual_thresh(self):
        is_manual = self.combo_thresh.currentText() == "Manual"
        self.spin_thresh_val.setEnabled(is_manual)
        self.roi_selected() # Trigger update

    def select_files(self):
        fs,_=QFileDialog.getOpenFileNames(self,"Select","","Video (*.mp4 *.avi)"); 
        if fs: 
            for f in fs:
                if f not in self.video_list: self.video_list.append(f); self.list_widget.addItem(os.path.basename(f))
            if self.video_list: self.list_widget.setCurrentRow(0)

    def clear_list(self): self.video_list=[]; self.list_widget.clear(); self.viewer.set_frame(None)
    def select_out(self): d=QFileDialog.getExistingDirectory(self); self.ln_out.setText(d) if d else None
    
    def load_preview_video(self, r):
        if r<0: return
        path = self.video_list[r]
        cap = cv2.VideoCapture(path); ret, f = cap.read(); cap.release()
        if ret:
            self.current_frame = f; self.viewer.set_frame(f)
            self.log(f"Loaded: {os.path.basename(path)}")

    def roi_selected(self):
        r = self.viewer.get_roi()
        if r and self.current_frame is not None:
            # Gather Config
            ocr_config = {
                'scale': self.spin_scale.value(),
                'method': 'Manual' if self.combo_thresh.currentText() == 'Manual' else 'Otsu',
                'thresh_val': self.spin_thresh_val.value(),
                'invert': self.chk_invert.isChecked()
            }

            # LIVE OCR PREVIEW
            logic = OCRVideoSplitter(None, None, self.combo_format.currentText(), ocr_config)
            raw, parsed = logic.run_single_frame_ocr(self.current_frame, (r.x(), r.y(), r.width(), r.height()))
            
            self.viewer.set_result_text(f"{raw} => {parsed}")
            # self.log(f"Test: {raw}") # Optional: Don't spam log

    def start(self):
        if not self.video_list: QMessageBox.warning(self,"Err","No videos"); return
        r = self.viewer.get_roi()
        if not r: QMessageBox.warning(self,"Err","Select ROI"); return
        
        rect = (r.x(), r.y(), r.width(), r.height())
        
        # Config for Worker
        ocr_config = {
            'scale': self.spin_scale.value(),
            'method': 'Manual' if self.combo_thresh.currentText() == 'Manual' else 'Otsu',
            'thresh_val': self.spin_thresh_val.value(),
            'invert': self.chk_invert.isChecked()
        }

        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True)
        
        self.worker = BatchOCRWorker(
            self.video_list, self.ln_out.text(), rect,
            self.time_start.time().toPyTime(), self.time_end.time().toPyTime(),
            self.chk_stitch.isChecked(), self.combo_format.currentText(),
            ocr_config # PASS CONFIG
        )
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(lambda v,m: (self.pbar.setValue(v), self.pbar.setFormat(m)))
        self.worker.finished_signal.connect(lambda: (self.btn_run.setEnabled(True), self.btn_stop.setEnabled(False), QMessageBox.information(self,"Done","Complete")))
        self.worker.error_signal.connect(lambda e: (self.log(f"Err: {e}"), self.btn_run.setEnabled(True)))
        self.worker.start()

    def stop_proc(self):
        if self.worker: self.worker.stop(); self.btn_stop.setEnabled(False)