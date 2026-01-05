from PyQt6.QtWidgets import QWidget, QSizePolicy, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction, QPolygonF
import cv2

class VideoSelectorWidget(QWidget):
    # Signal emitted whenever boxes are added/removed/moved
    selection_changed = pyqtSignal() 

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #000000; border: 2px solid #333;")
        self.setMouseTracking(True)
        
        # --- Image Data ---
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.img_draw_rect = QRect()
        
        # --- Interaction Logic ---
        self.start_point = QPoint()
        self.current_point = QPoint()
        self.is_drawing = False
        self.mode = "IDLE"
        
        # --- Data Storage ---
        # Manual Annotations: { frame_idx : [(QRect, class_id), ...] }
        self.annotations = {} 
        self.current_frame_idx = 0
        
        # --- Selection State ---
        self.selected_index = -1
        self.drag_offset = QPoint()
        self.current_class_id = 0

        # --- Preview Data (Auto-Detect / Dataset View) ---
        # Format: [(shape, class_id), ...]
        self.preview_items = []    
        self.preview_mode = "box" 

        # Class Colors (0-9)
        self.class_colors = [
            QColor(57, 255, 20),   # 0: Neon Green
            QColor(255, 50, 50),   # 1: Red
            QColor(50, 100, 255),  # 2: Blue
            QColor(255, 255, 0),   # 3: Yellow
            QColor(255, 0, 255),   # 4: Magenta
            QColor(0, 255, 255),   # 5: Cyan
            QColor(255, 128, 0),   # 6: Orange
            QColor(128, 0, 128),   # 7: Purple
            QColor(255, 255, 255), # 8: White
            QColor(128, 128, 128)  # 9: Gray
        ]

    def set_current_class(self, class_id):
        self.current_class_id = class_id

    def set_annotations(self, new_annotations):
        self.annotations = new_annotations if new_annotations is not None else {}
        self.selected_index = -1
        self.update()

    def set_current_frame(self, frame_idx, frame_bgr):
        self.current_frame_idx = frame_idx
        if frame_bgr is None: return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qt_img)
        
        # Note: We do not clear preview_items here to allow scrubbing
        self.selected_index = -1
        self.update()

    def get_current_boxes_with_classes(self):
        if self.current_frame_idx not in self.annotations:
            self.annotations[self.current_frame_idx] = []
        return self.annotations[self.current_frame_idx]

    def get_current_boxes(self):
        """Compatibility helper: returns just QRects."""
        return [b[0] for b in self.get_current_boxes_with_classes()]

    def set_detected_objects(self, rects_list):
        """Helper for standard bounding box previews. Wraps them with Class 0 (Default)."""
        clean_list = []
        for item in rects_list:
            if isinstance(item, (tuple, list)) and len(item) == 4:
                rect = QRect(int(item[0]), int(item[1]), int(item[2]), int(item[3]))
                clean_list.append((rect, 0)) # Default to class 0
            elif isinstance(item, QRect):
                clean_list.append((item, 0))
        self.set_preview_data(clean_list, mode="box")

    def set_preview_data(self, items_with_classes, mode="box"):
        """
        items_with_classes: List of tuples [(shape, class_id), ...]
        mode: 'box' or 'polygon'
        """
        self.preview_items = items_with_classes
        self.preview_mode = mode
        self.update()

    def screen_to_image(self, pos):
        if not self.scale_factor: return QPoint(0,0)
        ix = (pos.x() - self.img_draw_rect.x()) / self.scale_factor
        iy = (pos.y() - self.img_draw_rect.y()) / self.scale_factor
        return QPoint(int(ix), int(iy))

    def image_to_screen_rect(self, rect):
        sx = self.img_draw_rect.x() + (rect.x() * self.scale_factor)
        sy = self.img_draw_rect.y() + (rect.y() * self.scale_factor)
        sw = rect.width() * self.scale_factor
        sh = rect.height() * self.scale_factor
        return QRectF(sx, sy, sw, sh)

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
        self.scale_factor = min(ww/iw, wh/ih)
        nw, nh = int(iw*self.scale_factor), int(ih*self.scale_factor)
        ox, oy = (ww-nw)//2, (wh-nh)//2
        self.img_draw_rect = QRect(ox, oy, nw, nh)

        painter.drawPixmap(self.img_draw_rect, self.original_pixmap)
        painter.setClipRect(self.img_draw_rect)

        # 1. Draw PREVIEW Items (Dataset Visualization) - COLOR CODED
        if self.preview_items:
            for item, cls_id in self.preview_items:
                # Pick color based on class ID
                c_idx = cls_id % len(self.class_colors)
                base_color = self.class_colors[c_idx]

                if self.preview_mode == "box":
                    pen = QPen(base_color, 2)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    if isinstance(item, (QRect, QRectF)):
                        painter.drawRect(self.image_to_screen_rect(item))
                        # Draw ID
                        painter.drawText(self.image_to_screen_rect(item).topLeft(), str(cls_id))
                
                elif self.preview_mode == "polygon":
                    pen = QPen(base_color, 2)
                    # Semi-transparent fill matching class color
                    brush_color = QColor(base_color)
                    brush_color.setAlpha(60) 
                    
                    painter.setPen(pen)
                    painter.setBrush(brush_color)
                    
                    if isinstance(item, QPolygonF):
                        scaled_poly = QPolygonF()
                        for pt in item:
                            sx = ox + (pt.x() * self.scale_factor)
                            sy = oy + (pt.y() * self.scale_factor)
                            scaled_poly.append(QPointF(sx, sy))
                        painter.drawPolygon(scaled_poly)

        # 2. Draw MANUAL Annotations
        current_data = self.get_current_boxes_with_classes()
        
        for i, (rect, cls_id) in enumerate(current_data):
            c_idx = cls_id % len(self.class_colors)
            color = self.class_colors[c_idx]
            
            pen_style = Qt.PenStyle.DashLine if i == self.selected_index else Qt.PenStyle.SolidLine
            pen_width = 3 if i == self.selected_index else 2
            
            painter.setPen(QPen(color, pen_width, pen_style))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            scr_rect = self.image_to_screen_rect(rect)
            painter.drawRect(scr_rect)
            painter.drawText(scr_rect.topLeft() - QPointF(0, 5), f"ID:{cls_id}")

        # 3. Drawing Action
        if self.is_drawing:
            cur_color = self.class_colors[self.current_class_id % len(self.class_colors)]
            painter.setPen(QPen(cur_color, 2, Qt.PenStyle.DotLine))
            rect = QRect(self.start_point, self.current_point).normalized()
            painter.drawRect(rect)

    # --- Mouse Interaction ---
    def mousePressEvent(self, event):
        if not self.original_pixmap: return
        img_pt = self.screen_to_image(event.pos())
        current_data = self.get_current_boxes_with_classes()

        clicked_idx = -1
        for i in range(len(current_data)-1, -1, -1):
            if current_data[i][0].contains(img_pt):
                clicked_idx = i; break
        
        if event.button() == Qt.MouseButton.LeftButton:
            if clicked_idx != -1:
                self.mode = "MOVING"; self.selected_index = clicked_idx
                self.drag_offset = img_pt - current_data[clicked_idx][0].topLeft()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self.mode = "DRAWING"; self.selected_index = -1
                self.start_point = event.pos(); self.current_point = event.pos()
                self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
        elif event.button() == Qt.MouseButton.RightButton and clicked_idx != -1:
             self.selected_index = clicked_idx; self.update()
             self.show_context_menu(event.globalPosition().toPoint())

    def mouseMoveEvent(self, event):
        if self.mode == "MOVING":
            img_pt = self.screen_to_image(event.pos())
            data = self.get_current_boxes_with_classes()
            if self.selected_index < len(data):
                rect, cls = data[self.selected_index]
                w, h = rect.width(), rect.height()
                new_tl = img_pt - self.drag_offset
                iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
                x = max(0, min(new_tl.x(), iw - w))
                y = max(0, min(new_tl.y(), ih - h))
                rect.moveTo(x, y)
                self.update()
        elif self.mode == "DRAWING":
            x = max(self.img_draw_rect.left(), min(event.pos().x(), self.img_draw_rect.right()))
            y = max(self.img_draw_rect.top(), min(event.pos().y(), self.img_draw_rect.bottom()))
            self.current_point = QPoint(x, y)
            self.update()
        else:
            img_pt = self.screen_to_image(event.pos())
            hover = any(r[0].contains(img_pt) for r in self.get_current_boxes_with_classes())
            self.setCursor(Qt.CursorShape.OpenHandCursor if hover else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "DRAWING":
                scr_r = QRect(self.start_point, self.current_point).normalized()
                if scr_r.width() > 5:
                    ox = (scr_r.x() - self.img_draw_rect.x()) / self.scale_factor
                    oy = (scr_r.y() - self.img_draw_rect.y()) / self.scale_factor
                    ow = scr_r.width() / self.scale_factor
                    oh = scr_r.height() / self.scale_factor
                    new_box = QRect(int(ox), int(oy), int(ow), int(oh))
                    
                    if self.current_frame_idx not in self.annotations:
                        self.annotations[self.current_frame_idx] = []
                    self.annotations[self.current_frame_idx].append((new_box, self.current_class_id))
                    self.selection_changed.emit()
            elif self.mode == "MOVING":
                self.selection_changed.emit()
            self.mode = "IDLE"; self.update()

    def delete_selected(self):
        data = self.get_current_boxes_with_classes()
        if self.selected_index != -1 and self.selected_index < len(data):
            del data[self.selected_index]
            self.selected_index = -1
            if not data: del self.annotations[self.current_frame_idx]
            self.selection_changed.emit(); self.update()

    def clear_current_frame(self):
        if self.current_frame_idx in self.annotations: del self.annotations[self.current_frame_idx]
        self.selected_index = -1; self.selection_changed.emit(); self.update()

    def clear_all(self):
        self.annotations = {}; self.selected_index = -1; self.selection_changed.emit(); self.update()
    
    def show_context_menu(self, pos):
        menu = QMenu(self); act = QAction("Delete", self); act.triggered.connect(self.delete_selected); menu.addAction(act); menu.exec(pos)