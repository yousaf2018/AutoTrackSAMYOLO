import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QPushButton, QSpinBox, QCheckBox, 
    QGroupBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QLabel, QListWidget, QSplitter, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from workers.splitter_worker import VideoSplitterWorker

class SplitterWindow(QMainWindow):
    go_home_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Pre-Processing Tool")
        self.resize(1200, 800)
        self.video_files = []
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main vertical layout for the whole window
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10) 
        main_layout.setSpacing(10)

        # ==========================
        # 0. HEADER (Home + Title)
        # ==========================
        header_layout = QHBoxLayout()
        
        btn_home = QPushButton("🏠 Home")
        btn_home.setFixedWidth(100)
        btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_home.setStyleSheet("""
            QPushButton { background-color: #444; color: white; border: 1px solid #555; border-radius: 5px; padding: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
        """)
        btn_home.clicked.connect(self.go_home_signal.emit)
        
        title = QLabel("Video Splitter & Formatting")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(btn_home)
        header_layout.addWidget(title, 1)
        header_layout.addStretch() 
        
        main_layout.addLayout(header_layout)

        # ==========================
        # 1. CONTENT SPLITTER
        # ==========================
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # --- LEFT PANEL (Settings) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        
        # Group 1: Configuration
        grp_cfg = QGroupBox("1. Configuration")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(1, 3600)
        self.spin_duration.setValue(5)
        self.spin_duration.setSuffix(" sec")
        self.spin_duration.setStyleSheet("padding: 5px; background: #222; color: white; border: 1px solid #444;")
        
        self.chk_subfolder = QCheckBox("Create Subfolder per Video")
        self.chk_subfolder.setChecked(True)
        self.chk_subfolder.setStyleSheet("color: white; font-weight: bold;")

        self.path_out = QLineEdit(os.path.join(os.getcwd(), "Input-videos"))
        self.path_out.setPlaceholderText("Select Output Folder...")
        self.path_out.setStyleSheet("padding: 6px; background: #222; color: #eee; border: 1px solid #444;")
        
        btn_browse = QPushButton("Browse")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.setStyleSheet("background: #495057; color: white; padding: 6px; border: none; border-radius: 4px;")
        btn_browse.clicked.connect(self.browse_out)

        # Output Row
        out_row = QHBoxLayout()
        out_row.addWidget(self.path_out)
        out_row.addWidget(btn_browse)

        form.addRow("Chunk Duration:", self.spin_duration)
        form.addRow("", self.chk_subfolder)
        form.addRow("Output Folder:", out_row)
        
        grp_cfg.setLayout(form)
        left_layout.addWidget(grp_cfg)

        # Group 2: Actions
        grp_act = QGroupBox("2. Actions")
        act_layout = QVBoxLayout()
        
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Files")
        self.btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add.setStyleSheet("background: #0d6efd; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
        self.btn_add.clicked.connect(self.add_files)
        
        self.btn_clear = QPushButton("Clear List")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setStyleSheet("background: #6c757d; color: white; padding: 8px; border-radius: 4px;")
        self.btn_clear.clicked.connect(self.clear_list)
        
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_clear)

        self.btn_run = QPushButton("START SPLITTING")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet("background: #198754; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        self.btn_run.clicked.connect(self.start_process)

        # --- CANCEL BUTTON ---
        self.btn_cancel = QPushButton("CANCEL PROCESS")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setStyleSheet("background: #dc3545; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.stop_processing)

        act_layout.addLayout(btn_row)
        act_layout.addSpacing(10)
        act_layout.addWidget(self.btn_run)
        act_layout.addWidget(self.btn_cancel)
        grp_act.setLayout(act_layout)
        
        left_layout.addWidget(grp_act)
        left_layout.addStretch() 

        # --- RIGHT PANEL (File List) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
        lbl_list = QLabel("Selected Videos (Drag & Drop here):")
        lbl_list.setStyleSheet("font-weight: bold; color: #ccc;")
        
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget { background: #1e1e1e; color: #fff; border: 1px solid #444; border-radius: 4px; }
            QListWidget::item { padding: 5px; }
        """)
        
        right_layout.addWidget(lbl_list)
        right_layout.addWidget(self.file_list)

        # Add to Splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([350, 850]) 
        splitter.setCollapsible(0, False)
        
        main_layout.addWidget(splitter, 1)

        # ==========================
        # 3. BOTTOM PANEL (Logs)
        # ==========================
        bottom_grp = QGroupBox("Progress & Logs")
        bottom_layout = QVBoxLayout()
        
        prog_row = QHBoxLayout()
        self.lbl_prog = QLabel("Progress: 0/0")
        self.lbl_prog.setStyleSheet("color: white; font-weight: bold;")
        
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setTextVisible(True)
        self.pbar.setStyleSheet("QProgressBar { border: 1px solid #444; background: #222; height: 15px; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #0d6efd; }")
        
        prog_row.addWidget(self.lbl_prog)
        prog_row.addWidget(self.pbar, 1)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("background: #000; color: #0f0; font-family: Consolas; border: 1px solid #444;")

        bottom_layout.addLayout(prog_row)
        bottom_layout.addWidget(self.log_text)
        
        bottom_grp.setLayout(bottom_layout)
        main_layout.addWidget(bottom_grp)

    # ==========================
    # LOGIC
    # ==========================
    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output")
        if d: self.path_out.setText(d)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Video (*.mp4 *.avi *.mkv *.mov *.flv)")
        if files: self.add_files_to_list(files)

    def add_files_to_list(self, files):
        for f in files:
            if f not in self.video_files and os.path.isfile(f):
                self.video_files.append(f)
                self.file_list.addItem(os.path.basename(f))
                self.log(f"Added: {os.path.basename(f)}")

    def clear_list(self):
        self.video_files = []
        self.file_list.clear()
        self.log("List cleared.")

    def log(self, m):
        self.log_text.append(m)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def start_process(self):
        if not self.video_files:
            QMessageBox.warning(self, "Error", "No video files selected.")
            return

        out_dir = self.path_out.text()
        if not os.path.exists(out_dir):
            try: os.makedirs(out_dir)
            except: 
                QMessageBox.critical(self, "Error", "Invalid output path.")
                return

        self.set_ui_processing(True)
        self.pbar.setValue(0)
        self.log_text.clear()
        self.log("--- Starting Split Process ---")
        
        self.worker = VideoSplitterWorker(
            self.video_files, 
            out_dir, 
            self.spin_duration.value(), 
            self.chk_subfolder.isChecked()
        )
        
        self.worker.log_signal.connect(self.log)
        self.worker.overall_progress.connect(self.update_overall)
        self.worker.file_progress.connect(self.update_file)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def stop_processing(self):
        """Cancels the running worker."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "Confirm Cancel", 
                                       "Are you sure you want to cancel the splitting process?", 
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                self.log(">>> CANCEL REQUESTED. STOPPING...")
                self.worker.stop()
                self.btn_cancel.setText("Cancelling...")
                self.btn_cancel.setEnabled(False)

    def update_overall(self, idx, total, name):
        self.lbl_prog.setText(f"Processing {idx}/{total}: {name}")

    def update_file(self, pct, status):
        self.pbar.setValue(pct)
        self.pbar.setFormat(f"{status} ({pct}%)")

    def on_finished(self):
        self.set_ui_processing(False)
        self.pbar.setValue(100)
        self.pbar.setFormat("Complete")
        QMessageBox.information(self, "Done", "Splitting Complete!")

    def on_error(self, err):
        self.set_ui_processing(False)
        self.log(f"ERROR: {err}")
        QMessageBox.critical(self, "Error", err)

    def set_ui_processing(self, active):
        self.btn_run.setEnabled(not active)
        
        self.btn_cancel.setEnabled(active)
        self.btn_cancel.setText("CANCEL PROCESS" if active else "CANCEL PROCESS")
        
        self.btn_add.setEnabled(not active)
        self.btn_clear.setEnabled(not active)
        self.spin_duration.setEnabled(not active)
        self.chk_subfolder.setEnabled(not active)
        self.file_list.setEnabled(not active)