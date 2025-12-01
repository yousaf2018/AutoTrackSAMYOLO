import os
from PyQt6.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap

class AnimatedSplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__(QPixmap()) # Initialize with empty pixmap
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # --- Main Container (Rounded & Dark) ---
        container = QWidget(self)
        container.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 15px;
                border: 2px solid #0d6efd;
            }
        """)
        container.setFixedSize(450, 350) # Increased height for logo
        
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # --- 1. LOGO IMAGE ---
        # Ensure 'logo.png' is in the main project folder
        logo_path = "logo.png" 
        
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("border: none;") # Remove border from image container
        
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale logo to fit nicely (e.g., 150x150 max)
            scaled_pixmap = pixmap.scaled(
                150, 150, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if image missing
            logo_label.setText("[Logo.png not found]")
            logo_label.setStyleSheet("color: #777; font-style: italic; border: none;")

        layout.addWidget(logo_label)
        
        # --- 2. TITLE TEXT ---
        title = QLabel("AutoTrack\nSAMYOLO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; font-weight: bold; font-size: 22px; border: none;")
        layout.addWidget(title)
        
        # --- 3. PROGRESS BAR ---
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar { 
                border: none; 
                background: #444; 
                height: 8px; 
                border-radius: 4px; 
                text-align: center;
                color: transparent;
            } 
            QProgressBar::chunk { 
                background-color: #0d6efd; 
                border-radius: 4px; 
            }
        """)
        self.progress.setFixedWidth(350)
        layout.addWidget(self.progress, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Set Splash Size
        self.setFixedSize(450, 350)
        self.center()
        
    def center(self):
        screen = self.screen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def simulate_loading(self, callback):
        self.callback = callback
        self.counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(30) # Speed of animation (lower is faster)

    def update_progress(self):
        self.counter += 2
        self.progress.setValue(self.counter)
        if self.counter >= 100:
            self.timer.stop()
            self.callback() # Close splash, open main window