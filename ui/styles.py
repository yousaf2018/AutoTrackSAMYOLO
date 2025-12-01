DARK_THEME = """
QMainWindow, QDialog {
    background-color: #1e1e1e;
    color: #ffffff;
}
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 20px;
    font-weight: bold;
    background-color: #252526;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #61afef;
}
QLineEdit {
    background-color: #181818;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 8px;
    color: #eee;
    font-size: 13px;
}
QLineEdit:focus {
    border: 1px solid #0d6efd;
}
QSpinBox, QDoubleSpinBox {
    background-color: #181818;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 8px;
    color: #eee;
}
QPushButton {
    border-radius: 5px;
    padding: 8px 16px;
    font-weight: bold;
}
QProgressBar {
    border: 1px solid #333;
    background-color: #181818;
    height: 12px;
    border-radius: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #0d6efd;
    border-radius: 6px;
}
"""