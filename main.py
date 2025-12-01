import sys
from PyQt6.QtWidgets import QApplication, QStackedWidget
from ui.splash_screen import AnimatedSplashScreen
from ui.launcher import LauncherDialog
from ui.main_window import MainWindow
from ui.dataset_window import DatasetWindow
from ui.training_window import TrainingWindow
from ui.styles import DARK_THEME

class AppController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setStyleSheet(DARK_THEME)
        
        # Windows (Lazy loaded or init all)
        self.launcher = None
        self.track_win = None
        self.data_win = None
        self.train_win = None
        
        # Start with Splash
        self.splash = AnimatedSplashScreen()
        self.splash.show()
        self.splash.simulate_loading(self.show_launcher)

    def show_launcher(self):
        self.splash.finish(self.splash) # Close splash
        
        if not self.launcher:
            self.launcher = LauncherDialog()
            # Connect custom signals if Launcher was a QMainWindow, 
            # but since it's a Dialog with buttons:
            self.launcher.btn_track.clicked.connect(lambda: self.switch_view("TRACK"))
            self.launcher.btn_dataset.clicked.connect(lambda: self.switch_view("DATASET"))
            self.launcher.btn_train.clicked.connect(lambda: self.switch_view("TRAIN"))

        self.launcher.show()

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
            self.data_win.show()
            
        elif view_name == "TRAIN":
            if not self.train_win:
                self.train_win = TrainingWindow()
                self.train_win.go_home_signal.connect(self.go_home)
            self.train_win.show()

    def go_home(self):
        # Hide all sub-windows
        if self.track_win: self.track_win.hide()
        if self.data_win: self.data_win.hide()
        if self.train_win: self.train_win.hide()
        
        # Show launcher
        self.launcher.show()

    def run(self):
        sys.exit(self.app.exec())

if __name__ == "__main__":
    controller = AppController()
    controller.run()