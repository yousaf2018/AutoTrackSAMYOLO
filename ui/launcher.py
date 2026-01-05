from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QGridLayout, QSizePolicy, QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QCursor, QColor, QIcon

class LauncherDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoTrackSamYOLO - Suite Launcher")
        self.setStyleSheet("background-color: #121212;") # Deep dark background
        
        # Main Layout (Vertical)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- 1. HEADER SECTION ---
        header_container = QWidget()
        header_container.setFixedHeight(150)
        header_layout = QVBoxLayout(header_container)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("AutoTrackSamYOLO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px; 
            font-weight: 900; 
            color: #ffffff; 
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 5px;
        """)
        
        subtitle = QLabel("Universal Object Tracking & Dataset Pipeline")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 20px; 
            color: #888888; 
            font-weight: 400;
            font-family: 'Segoe UI', sans-serif;
        """)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header_container)
        
        # --- 2. GRID SECTION (The Tiles) ---
        # We put the grid inside a centered widget to prevent stretching on ultrawide screens
        content_wrapper = QWidget()
        content_layout = QHBoxLayout(content_wrapper)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        grid_container = QWidget()
        grid_container.setFixedWidth(1000) # Constrain width for better look
        # grid_container.setStyleSheet("background: red;") # Debug
        
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(30)
        grid_layout.setContentsMargins(20, 20, 20, 20)
        
        # Button 1: Splitter
        self.btn_split = self.create_tile(
            "✂️", "Video Splitter", 
            "Pre-process large videos into small chunks.\nEssential for memory safety.", 
            "#a55eea" # Purple
        )
        
        # Button 2: Tracker
        self.btn_track = self.create_tile(
            "🚀", "SAM 3 Tracker", 
            "Zero-shot tracking using AI.\nGenerates CSVs, Heatmaps & Trajectories.", 
            "#4b7bec" # Blue
        )
        
        # Button 3: Dataset
        self.btn_dataset = self.create_tile(
            "📂", "Dataset Factory", 
            "Convert SAM tracks into YOLO format.\nAuto-split Train/Val/Test.", 
            "#26de81" # Green
        )
        
        # Button 4: Trainer
        self.btn_train = self.create_tile(
            "🧠", "Model Trainer", 
            "Train custom YOLO models locally.\nMonitor loss & mAP metrics.", 
            "#fd9644" # Orange
        )

        # Button 5: OCR Splitter
        self.btn_ocr = self.create_tile(
            "🕒", "OCR Time Splitter",
            "Split 7-day videos by Day/Night cycles.\nUses on-screen timestamp.",
            "#ff4757" # Red/Pink
        )

        # Add to Grid (Row, Col)
        grid_layout.addWidget(self.btn_split, 0, 0)
        grid_layout.addWidget(self.btn_track, 0, 1)
        grid_layout.addWidget(self.btn_dataset, 1, 0)
        grid_layout.addWidget(self.btn_train, 1, 1)
        grid_layout.addWidget(self.btn_ocr, 2, 0, 1, 2) # Span 2 columns at bottom
        content_layout.addWidget(grid_container)
        main_layout.addWidget(content_wrapper, 1) # 1 = Expand to fill vertical space

        # --- 3. FOOTER SECTION ---
        footer_container = QWidget()
        footer_container.setFixedHeight(80)
        footer_layout = QHBoxLayout(footer_container)
        footer_layout.setContentsMargins(40, 0, 40, 20)
        
        lbl_version = QLabel("v2.1  |  Powered by Meta SAM 3 & Ultralytics")
        lbl_version.setStyleSheet("color: #555; font-size: 14px; font-weight: bold;")
        
        # Quit Button
        self.btn_quit = QPushButton("Exit Application")
        self.btn_quit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_quit.setFixedWidth(160)
        self.btn_quit.setFixedHeight(45)
        self.btn_quit.setStyleSheet("""
            QPushButton {
                background-color: #1f1f1f;
                border: 1px solid #333;
                color: #aaa;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2a0a0a;
                border: 1px solid #ff4d4d;
                color: #ff4d4d;
            }
        """)
        
        footer_layout.addWidget(lbl_version)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_quit)
        
        main_layout.addWidget(footer_container)
        
    def create_tile(self, icon, title, desc, accent_color):
        btn = QPushButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        btn.setMinimumHeight(180) # Force a card shape
        
        # Advanced CSS for Card Look
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #1e1e1e;
                border: 2px solid #2b2b2b;
                border-radius: 16px;
                text-align: left;
                padding: 20px;
            }}
            QPushButton:hover {{
                background-color: #252525;
                border: 2px solid {accent_color};
                margin-top: -3px; /* Subtle lift effect */
            }}
            QPushButton:pressed {{
                background-color: #151515;
                margin-top: 0px;
                border-color: #333;
            }}
        """)
        
        # Internal Layout
        layout = QVBoxLayout(btn)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Header Row (Icon + Title)
        h_row = QHBoxLayout()
        h_row.setSpacing(15)
        
        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet("font-size: 40px; border: none; background: transparent;")
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"""
            font-size: 26px; 
            font-weight: 800; 
            color: white; 
            background: transparent;
            font-family: 'Segoe UI', sans-serif;
        """)
        
        h_row.addWidget(lbl_icon)
        h_row.addWidget(lbl_title, 1)
        
        # Description
        lbl_desc = QLabel(desc)
        lbl_desc.setStyleSheet("""
            font-size: 15px; 
            color: #bbbbbb; 
            background: transparent;
            line-height: 140%;
            border: none;
        """)
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        layout.addLayout(h_row)
        layout.addWidget(lbl_desc, 1) # Expand desc to push layout
        
        # Add a subtle drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 80))
        btn.setGraphicsEffect(shadow)
        
        return btn