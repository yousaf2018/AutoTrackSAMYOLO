from PyQt6.QtWidgets import QWidget, QSizePolicy, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction
import cv2

class VideoSelectorWidget(QWidget):
    # Signal emitted whenever boxes are added/removed/moved
    selection_changed = pyqtSignal() 

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #000000; border: 2px solid #333;")
        self.setMouseTracking(True)
        
        # Image Data
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.img_draw_rect = QRect()
        
        # Logic
        self.start_point = QPoint()
        self.current_point = QPoint()
        self.is_drawing = False
        self.mode = "IDLE"
        
        # Data Storage: { frame_idx: [QRect, QRect...] }
        self.annotations = {} 
        self.current_frame_idx = 0
        
        # Selection State
        self.selected_index = -1
        self.drag_offset = QPoint()

        # Detected objects for preview (Cyan)
        self.detected_objects = []

    def load_frame(self, frame_bgr):
        """
        Legacy/Simple wrapper for loading a frame without specific index context.
        Used by DatasetWindow and Preview.
        """
        self.set_current_frame(0, frame_bgr)

    def set_current_frame(self, frame_idx, frame_bgr):
        """Updates the displayed frame and current index."""
        self.current_frame_idx = frame_idx
        
        if frame_bgr is None: return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qt_img)
        
        # Clear preview detections when frame changes
        self.detected_objects = []
        self.selected_index = -1
        self.update()

    def get_current_boxes(self):
        """Returns boxes for the current frame."""
        return self.annotations.get(self.current_frame_idx, [])

    def set_detected_objects(self, rects_list):
        self.detected_objects = rects_list
        self.update()

    # --- Coordinate Helpers ---
    def screen_to_image(self, pos):
        if not self.scale_factor: return QPoint(0,0)
        ix = (pos.x() - self.img_draw_rect.x()) / self.scale_factor
        iy = (pos.y() - self.img_draw_rect.y()) / self.scale_factor
        return QPoint(int(ix), int(iy))

    def image_to_screen(self, rect):
        sx = self.img_draw_rect.x() + (rect.x() * self.scale_factor)
        sy = self.img_draw_rect.y() + (rect.y() * self.scale_factor)
        sw = rect.width() * self.scale_factor
        sh = rect.height() * self.scale_factor
        return QRectF(sx, sy, sw, sh)

    # --- Paint ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self.original_pixmap:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Video Loaded")
            return

        # Geometry
        ww, wh = self.width(), self.height()
        iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
        
        # Calculate scale to fit
        scale_w = ww / iw
        scale_h = wh / ih
        self.scale_factor = min(scale_w, scale_h)
        
        nw, nh = int(iw*self.scale_factor), int(ih*self.scale_factor)
        ox, oy = (ww-nw)//2, (wh-nh)//2
        self.img_draw_rect = QRect(ox, oy, nw, nh)

        painter.drawPixmap(self.img_draw_rect, self.original_pixmap)
        painter.setClipRect(self.img_draw_rect)

        # Draw DETECTED (Cyan)
        pen_det = QPen(QColor(0, 255, 255), 2)
        painter.setPen(pen_det)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for (x, y, w, h) in self.detected_objects:
            painter.drawRect(self.image_to_screen(QRect(x, y, w, h)))

        # Draw MANUAL (Green) for Current Frame
        current_boxes = self.get_current_boxes()
        pen_norm = QPen(QColor(57, 255, 20), 2)
        pen_sel = QPen(QColor(255, 0, 0), 3, Qt.PenStyle.DashLine)
        
        for i, r in enumerate(current_boxes):
            painter.setPen(pen_sel if i == self.selected_index else pen_norm)
            painter.drawRect(self.image_to_screen(r))

        # Draw Dragging (Red)
        if self.is_drawing:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DotLine))
            rect = QRect(self.start_point, self.current_point).normalized()
            painter.drawRect(rect)

    # --- Interaction ---
    def mousePressEvent(self, event):
        if not self.original_pixmap: return
        if not self.img_draw_rect.contains(event.pos()):
            self.selected_index = -1; self.update(); return

        img_pt = self.screen_to_image(event.pos())
        current_boxes = self.get_current_boxes()

        # Hit Test (Backwards)
        clicked_idx = -1
        for i in range(len(current_boxes)-1, -1, -1):
            if current_boxes[i].contains(img_pt):
                clicked_idx = i
                break
        
        if event.button() == Qt.MouseButton.LeftButton:
            if clicked_idx != -1:
                self.mode = "MOVING"
                self.selected_index = clicked_idx
                self.drag_offset = img_pt - current_boxes[clicked_idx].topLeft()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self.mode = "DRAWING"
                self.selected_index = -1
                self.start_point = event.pos()
                self.current_point = event.pos()
                self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton and clicked_idx != -1:
            self.selected_index = clicked_idx
            self.update()
            self.show_context_menu(event.globalPosition().toPoint())

    def mouseMoveEvent(self, event):
        if self.mode == "MOVING":
            img_pt = self.screen_to_image(event.pos())
            boxes = self.get_current_boxes()
            if self.selected_index < len(boxes):
                r = boxes[self.selected_index]
                w, h = r.width(), r.height()
                new_tl = img_pt - self.drag_offset
                iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
                x = max(0, min(new_tl.x(), iw - w))
                y = max(0, min(new_tl.y(), ih - h))
                boxes[self.selected_index].moveTo(x, y)
                self.update()
        elif self.mode == "DRAWING":
            x = max(self.img_draw_rect.left(), min(event.pos().x(), self.img_draw_rect.right()))
            y = max(self.img_draw_rect.top(), min(event.pos().y(), self.img_draw_rect.bottom()))
            self.current_point = QPoint(x, y)
            self.update()
        else:
            # Hover cursor
            img_pt = self.screen_to_image(event.pos())
            hover = any(r.contains(img_pt) for r in self.get_current_boxes())
            self.setCursor(Qt.CursorShape.OpenHandCursor if hover else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "DRAWING":
                scr_r = QRect(self.start_point, self.current_point).normalized()
                if scr_r.width() > 5 and scr_r.height() > 5:
                    ox = (scr_r.x() - self.img_draw_rect.x()) / self.scale_factor
                    oy = (scr_r.y() - self.img_draw_rect.y()) / self.scale_factor
                    ow = scr_r.width() / self.scale_factor
                    oh = scr_r.height() / self.scale_factor
                    
                    new_box = QRect(int(ox), int(oy), int(ow), int(oh))
                    
                    if self.current_frame_idx not in self.annotations:
                        self.annotations[self.current_frame_idx] = []
                        
                    self.annotations[self.current_frame_idx].append(new_box)
                    self.selected_index = len(self.annotations[self.current_frame_idx]) - 1
                    self.selection_changed.emit()
            
            elif self.mode == "MOVING":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.selection_changed.emit()

            self.mode = "IDLE"
            self.update()

    def delete_selected(self):
        boxes = self.get_current_boxes()
        if self.selected_index != -1 and self.selected_index < len(boxes):
            del boxes[self.selected_index]
            self.selected_index = -1
            if not boxes:
                del self.annotations[self.current_frame_idx]
            self.selection_changed.emit()
            self.update()

    def clear_current_frame(self):
        if self.current_frame_idx in self.annotations:
            del self.annotations[self.current_frame_idx]
            self.selected_index = -1
            self.selection_changed.emit()
            self.update()

    def clear_all(self):
        self.annotations = {}
        self.selected_index = -1
        self.selection_changed.emit()
        self.update()

    def show_context_menu(self, pos):
        menu = QMenu(self)
        delete_action = QAction("Delete Box", self)
        delete_action.triggered.connect(self.delete_selected)
        menu.addAction(delete_action)
        menu.exec(pos)