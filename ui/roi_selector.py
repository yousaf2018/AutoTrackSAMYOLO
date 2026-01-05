from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

class ROISelector(QWidget):
    # Signal emitted when user finishes drawing a box
    selection_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #000000; border: 2px solid #333;")
        self.setMouseTracking(True)
        
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.img_draw_rect = QRect()
        
        self.start_point = QPoint()
        self.current_point = QPoint()
        self.is_drawing = False
        self.roi_rect = None
        
        # Stores the text result from OCR to display
        self.result_text = None

    def set_frame(self, frame_bgr):
        if frame_bgr is None: return
        import cv2
        h, w, ch = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(img)
        self.result_text = None 
        self.update()

    def set_result_text(self, text):
        self.result_text = text
        self.update()

    def get_roi(self):
        return self.roi_rect

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.original_pixmap:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load Video to Select ROI")
            return

        # Geometry
        ww, wh = self.width(), self.height()
        iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
        self.scale_factor = min(ww/iw, wh/ih)
        nw, nh = int(iw*self.scale_factor), int(ih*self.scale_factor)
        ox, oy = (ww-nw)//2, (wh-nh)//2
        self.img_draw_rect = QRect(ox, oy, nw, nh)

        painter.drawPixmap(self.img_draw_rect, self.original_pixmap)
        
        # Draw Box
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self.roi_rect:
            r_screen = self._to_screen(self.roi_rect)
            painter.drawRect(r_screen)

        if self.is_drawing:
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            r = QRect(self.start_point, self.current_point).normalized()
            painter.drawRect(r)

        # Draw OCR Result Overlay
        if self.result_text:
            painter.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            
            display_str = f"OCR: {self.result_text}"
            
            txt_w = fm.horizontalAdvance(display_str) + 20
            txt_h = fm.height() + 10
            
            # Draw at Top Left
            tx, ty = 10, 10
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 200))
            painter.drawRoundedRect(tx, ty, txt_w, txt_h, 5, 5)
            
            painter.setPen(QColor(255, 255, 0)) # Yellow
            painter.drawText(tx + 10, ty + fm.ascent() + 5, display_str)

    def _to_screen(self, r):
        ox = self.img_draw_rect.x(); oy = self.img_draw_rect.y(); sf = self.scale_factor
        return QRectF(ox + r.x()*sf, oy + r.y()*sf, r.width()*sf, r.height()*sf)

    def mousePressEvent(self, e):
        if not self.original_pixmap: return
        if self.img_draw_rect.contains(e.pos()) and e.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True; self.start_point = e.pos(); self.current_point = e.pos()
            self.roi_rect = None; self.result_text = None; self.update()

    def mouseMoveEvent(self, e):
        if self.is_drawing:
            x = max(self.img_draw_rect.left(), min(e.pos().x(), self.img_draw_rect.right()))
            y = max(self.img_draw_rect.top(), min(e.pos().y(), self.img_draw_rect.bottom()))
            self.current_point = QPoint(x, y); self.update()

    def mouseReleaseEvent(self, e):
        if self.is_drawing and e.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False
            screen_r = QRect(self.start_point, self.current_point).normalized()
            if screen_r.width() > 5 and screen_r.height() > 5:
                ox = (screen_r.x() - self.img_draw_rect.x()) / self.scale_factor
                oy = (screen_r.y() - self.img_draw_rect.y()) / self.scale_factor
                ow = screen_r.width() / self.scale_factor
                oh = screen_r.height() / self.scale_factor
                self.roi_rect = QRect(int(ox), int(oy), int(ow), int(oh))
                self.selection_changed.emit()
            self.update()