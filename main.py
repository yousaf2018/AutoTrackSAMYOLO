import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt  # <--- Added this import
from ui.splash_screen import AnimatedSplashScreen
from ui.launcher import LauncherDialog
from ui.main_window import MainWindow
from ui.dataset_window import DatasetWindow
from ui.training_window import TrainingWindow
from ui.splitter_window import SplitterWindow
from ui.styles import DARK_THEME
from ui.ocr_tool import OCRSplitterWindow
class AppController:
    def __init__(self):
        # High DPI Scaling setup
        # Note: PyQt6 handles most of this automatically, but we check safely.
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        self.app = QApplication(sys.argv)
        self.app.setStyleSheet(DARK_THEME)
        
        # Windows
        self.launcher = None
        self.track_win = None
        self.data_win = None
        self.train_win = None
        self.split_win = None
        
        # Start
        self.splash = AnimatedSplashScreen()
        self.splash.show()
        self.splash.simulate_loading(self.show_launcher)

    def show_launcher(self):
        self.splash.finish(self.splash)
        
        if not self.launcher:
            self.launcher = LauncherDialog()
            
            # Connect Buttons
            self.launcher.btn_track.clicked.connect(lambda: self.switch_view("TRACK"))
            self.launcher.btn_dataset.clicked.connect(lambda: self.switch_view("DATASET"))
            self.launcher.btn_train.clicked.connect(lambda: self.switch_view("TRAIN"))
            self.launcher.btn_split.clicked.connect(lambda: self.switch_view("SPLIT"))
            
            # Connect Quit
            self.launcher.btn_quit.clicked.connect(self.app.quit)
            self.launcher.btn_ocr.clicked.connect(lambda: self.switch_view("OCR"))
        # Open in Full Screen
        self.launcher.showMaximized()

    def switch_view(self, view_name):
        self.launcher.hide()
        
        if view_name == "TRACK":
            if not self.track_win:
                self.track_win = MainWindow()
                self.track_win.go_home_signal.connect(self.go_home)
            self.track_win.showMaximized()
            
        elif view_name == "DATASET":
            if not self.data_win:
                self.data_win = DatasetWindow()
                self.data_win.go_home_signal.connect(self.go_home)
            self.data_win.showMaximized()
            
        elif view_name == "TRAIN":
            if not self.train_win:
                self.train_win = TrainingWindow()
                self.train_win.go_home_signal.connect(self.go_home)
            self.train_win.showMaximized()

        elif view_name == "SPLIT":
            if not self.split_win:
                self.split_win = SplitterWindow()
                self.split_win.go_home_signal.connect(self.go_home)
            self.split_win.showMaximized()
        elif view_name == "OCR":
            if not getattr(self, 'ocr_win', None):
                self.ocr_win = OCRSplitterWindow()
                self.ocr_win.go_home_signal.connect(self.go_home)
            self.ocr_win.show()

    def go_home(self):
        # Hide all children
        if self.track_win: self.track_win.hide()
        if self.data_win: self.data_win.hide()
        if self.train_win: self.train_win.hide()
        if self.split_win: self.split_win.hide()
        if getattr(self, 'ocr_win', None): self.ocr_win.hide()
        # Show launcher full screen
        self.launcher.showMaximized()

    def run(self):
        sys.exit(self.app.exec())

if __name__ == "__main__":
    controller = AppController()
    controller.run()