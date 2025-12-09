from PyQt6.QtWidgets import QWidget, QSizePolicy, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction, QPolygonF
import cv2

class VideoSelectorWidget(QWidget):
    # Signal emitted whenever boxes are added/removed/moved
    selection_changed = pyqtSignal() 

    def __init__(self):
        super().__init__()
        # Allow widget to expand to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Black background for better contrast
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
        self.mode = "IDLE"  # IDLE, DRAWING, MOVING
        
        # --- Data Storage ---
        # { frame_idx (int) : [QRect, QRect...] }
        self.annotations = {} 
        self.current_frame_idx = 0
        
        # --- Selection State ---
        self.selected_index = -1
        self.drag_offset = QPoint()

        # --- Preview Data (Auto-Detect or Dataset Visuals) ---
        self.preview_items = []    # List of QRect or QPolygonF
        self.preview_mode = "box"  # "box" or "polygon"

    def load_frame(self, frame_bgr):
        """Legacy wrapper for loading a single frame (defaults to index 0)."""
        self.set_current_frame(0, frame_bgr)

    def set_current_frame(self, frame_idx, frame_bgr):
        """Updates the displayed frame and internal frame index."""
        self.current_frame_idx = frame_idx
        
        if frame_bgr is None: return

        # Convert CV2 BGR to Qt RGB
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qt_img)
        
        # Note: We do NOT clear preview_items here automatically, 
        # allowing the preview to persist while scrubbing if desired.
        self.selected_index = -1
        self.update()

    def get_current_boxes(self):
        """Returns the list of manual boxes for the current frame."""
        return self.annotations.get(self.current_frame_idx, [])

    def set_detected_objects(self, rects_list):
        """Helper for standard bounding box previews."""
        # Convert tuples to QRects if necessary
        clean_list = []
        for item in rects_list:
            if isinstance(item, (tuple, list)) and len(item) == 4:
                clean_list.append(QRect(int(item[0]), int(item[1]), int(item[2]), int(item[3])))
            elif isinstance(item, QRect):
                clean_list.append(item)
        self.set_preview_data(clean_list, mode="box")

    def set_preview_data(self, items, mode="box"):
        """
        Sets data to be drawn in Cyan/Yellow (Auto-detected).
        mode: 'box' (expects QRects) or 'polygon' (expects QPolygonF)
        """
        self.preview_items = items
        self.preview_mode = mode
        self.update()

    # --- Coordinate Transformation Helpers ---
    def screen_to_image(self, pos):
        """Maps a mouse click on screen to the original image pixel coordinates."""
        if not self.scale_factor: return QPoint(0,0)
        ix = (pos.x() - self.img_draw_rect.x()) / self.scale_factor
        iy = (pos.y() - self.img_draw_rect.y()) / self.scale_factor
        return QPoint(int(ix), int(iy))

    def image_to_screen_rect(self, rect):
        """Maps an image rectangle to screen coordinates for drawing."""
        sx = self.img_draw_rect.x() + (rect.x() * self.scale_factor)
        sy = self.img_draw_rect.y() + (rect.y() * self.scale_factor)
        sw = rect.width() * self.scale_factor
        sh = rect.height() * self.scale_factor
        return QRectF(sx, sy, sw, sh)

    # --- Painting ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self.original_pixmap:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Video Loaded")
            return

        # 1. Calculate Geometry (Letterboxing)
        widget_w = self.width()
        widget_h = self.height()
        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()

        scale_w = widget_w / img_w
        scale_h = widget_h / img_h
        self.scale_factor = min(scale_w, scale_h)

        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)
        
        # Center the image
        off_x = (widget_w - new_w) // 2
        off_y = (widget_h - new_h) // 2

        self.img_draw_rect = QRect(off_x, off_y, new_w, new_h)

        # 2. Draw The Video Frame
        painter.drawPixmap(self.img_draw_rect, self.original_pixmap)
        
        # Set clipping so we don't draw boxes outside the video area
        painter.setClipRect(self.img_draw_rect)

        # 3. Draw Preview Items (Auto-Detected / Dataset Preview)
        if self.preview_items:
            if self.preview_mode == "box":
                # Draw Cyan Boxes
                pen = QPen(QColor(0, 255, 255), 2) 
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for item in self.preview_items:
                    # Handle both QRect and tuple input safely
                    if isinstance(item, (QRect, QRectF)):
                        painter.drawRect(self.image_to_screen_rect(item))
            
            elif self.preview_mode == "polygon":
                # Draw Filled Cyan Polygons
                pen = QPen(QColor(0, 255, 255), 2)
                brush = QColor(0, 255, 255, 50) # Transparent fill
                painter.setPen(pen)
                painter.setBrush(brush)
                
                for poly in self.preview_items:
                    # Scale polygon points
                    scaled_poly = QPolygonF()
                    for pt in poly:
                        sx = off_x + (pt.x() * self.scale_factor)
                        sy = off_y + (pt.y() * self.scale_factor)
                        # FIX: Use QPointF for QPolygonF
                        scaled_poly.append(QPointF(sx, sy))
                    painter.drawPolygon(scaled_poly)

        # 4. Draw Manual Annotations (Green)
        current_boxes = self.get_current_boxes()
        pen_norm = QPen(QColor(57, 255, 20), 2) # Neon Green
        pen_sel = QPen(QColor(255, 0, 0), 3, Qt.PenStyle.DashLine) # Red Dashed for selected
        
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        for i, r in enumerate(current_boxes):
            painter.setPen(pen_sel if i == self.selected_index else pen_norm)
            painter.drawRect(self.image_to_screen_rect(r))

        # 5. Draw New Box being dragged (Red Dotted)
        if self.is_drawing:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DotLine))
            rect = QRect(self.start_point, self.current_point).normalized()
            painter.drawRect(rect)

        # 6. Draw Overlay Stats (Outside Clipping Area)
        # Note: Text drawing logic is handled in the Widget paint or external labels now to avoid obscuring.
        
    # --- Mouse Interaction ---
    def mousePressEvent(self, event):
        if not self.original_pixmap: return
        
        # Ignore clicks outside the image
        if not self.img_draw_rect.contains(event.pos()):
            self.selected_index = -1
            self.update()
            return

        img_pt = self.screen_to_image(event.pos())
        current_boxes = self.get_current_boxes()

        # Hit Test (Check if clicking existing box)
        clicked_idx = -1
        # Loop backwards to select "top" box
        for i in range(len(current_boxes)-1, -1, -1):
            if current_boxes[i].contains(img_pt):
                clicked_idx = i
                break
        
        if event.button() == Qt.MouseButton.LeftButton:
            if clicked_idx != -1:
                # MODE: Move Existing Box
                self.mode = "MOVING"
                self.selected_index = clicked_idx
                self.drag_offset = img_pt - current_boxes[clicked_idx].topLeft()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                # MODE: Draw New Box
                self.mode = "DRAWING"
                self.selected_index = -1
                self.start_point = event.pos()
                self.current_point = event.pos()
                self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton and clicked_idx != -1:
            # Context Menu for Deletion
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
                
                # Boundary Checks
                iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
                x = max(0, min(new_tl.x(), iw - w))
                y = max(0, min(new_tl.y(), ih - h))
                
                boxes[self.selected_index].moveTo(x, y)
                self.update()
                
        elif self.mode == "DRAWING":
            # Clamp to image bounds
            x = max(self.img_draw_rect.left(), min(event.pos().x(), self.img_draw_rect.right()))
            y = max(self.img_draw_rect.top(), min(event.pos().y(), self.img_draw_rect.bottom()))
            self.current_point = QPoint(x, y)
            self.update()
            
        else:
            # Hover cursor effect
            img_pt = self.screen_to_image(event.pos())
            hover = any(r.contains(img_pt) for r in self.get_current_boxes())
            self.setCursor(Qt.CursorShape.OpenHandCursor if hover else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "DRAWING":
                # Create rect from start/end points
                screen_rect = QRect(self.start_point, self.current_point).normalized()
                
                # Minimum size check (5x5 pixels)
                if screen_rect.width() > 5 and screen_rect.height() > 5:
                    # Convert to image coordinates
                    ox = (screen_rect.x() - self.img_draw_rect.x()) / self.scale_factor
                    oy = (screen_rect.y() - self.img_draw_rect.y()) / self.scale_factor
                    ow = screen_rect.width() / self.scale_factor
                    oh = screen_rect.height() / self.scale_factor
                    
                    new_box = QRect(int(ox), int(oy), int(ow), int(oh))
                    
                    # Ensure list exists for this frame
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

    # --- Actions ---
    def delete_selected(self):
        """Deletes the currently selected red box."""
        boxes = self.get_current_boxes()
        if self.selected_index != -1 and self.selected_index < len(boxes):
            del boxes[self.selected_index]
            self.selected_index = -1
            
            if not boxes:
                del self.annotations[self.current_frame_idx]
                
            self.selection_changed.emit()
            self.update()

    def clear_current_frame(self):
        """Removes all manual boxes from the current frame."""
        if self.current_frame_idx in self.annotations:
            del self.annotations[self.current_frame_idx]
            self.selected_index = -1
            self.selection_changed.emit()
            self.update()

    def clear_all(self):
        """Removes ALL manual annotations from ALL frames."""
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