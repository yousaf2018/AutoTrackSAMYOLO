from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont

class LauncherDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tool Selector - SAM 3 Tracker")
        self.resize(900, 600) # Slightly larger default size
        
        # Main Layout
        layout = QVBoxLayout(self)
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Title
        title = QLabel("Select Workflow")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: white; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Grid for Tile Buttons
        grid = QGridLayout()
        grid.setSpacing(20)
        
        # Create Buttons
        # We define them as class attributes so we can connect signals in main.py
        self.btn_track = self.create_tile("🚀 Tracking", "Detect & Track particles.\nGenerate CSVs & Videos.", "#0d6efd")
        self.btn_dataset = self.create_tile("📂 Dataset", "Convert SAM 3 results\ninto YOLO format.", "#198754")
        self.btn_train = self.create_tile("🧠 Training", "Train custom YOLO models\non your data.", "#e0a800")
        
        # Add to Grid
        grid.addWidget(self.btn_track, 0, 0)
        grid.addWidget(self.btn_dataset, 0, 1)
        grid.addWidget(self.btn_train, 0, 2)
        
        layout.addLayout(grid)
        
        # Footer info
        footer = QLabel("AutoTrackSAMYOLO v2.0")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #666; font-size: 12px; margin-top: 20px;")
        layout.addWidget(footer)
        
    def create_tile(self, title, desc, color):
        btn = QPushButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # This line caused the error before; QSizePolicy is now imported
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border-radius: 15px;
                text-align: center;
                padding: 20px;
                border: 2px solid {color};
            }}
            QPushButton:hover {{ 
                background-color: {color}dd; 
                border: 2px solid white;
                margin-top: -5px; 
            }}
        """)
        
        # Internal Layout for Title + Description
        l = QVBoxLayout(btn)
        
        t = QLabel(title)
        t.setStyleSheet("font-size: 22px; font-weight: bold; color: white; background: transparent;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        d = QLabel(desc)
        d.setStyleSheet("font-size: 14px; color: #eee; background: transparent;")
        d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        d.setWordWrap(True)
        
        l.addWidget(t)
        l.addWidget(d)
        
        return btn