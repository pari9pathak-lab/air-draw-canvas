import cv2
import numpy as np
from theme import BRUSH_COLORS, glass_rect, outline_rect, ui_scale, EDGE, AA


class DrawingCanvas:
    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.alpha = np.zeros((height, width), dtype=np.uint8)   # stroke coverage
        self.prev_point = None
        self.color = BRUSH_COLORS[4]
        self.thickness = 5
        self._build_palette(width, height)

    # ---------- palette ----------
    def _build_palette(self, width, height):
        s = ui_scale(height)
        r = int(14 * s)
        gap = int(10 * s)
        n = len(BRUSH_COLORS)
        pill_w = n * (2 * r + gap) + gap
        pill_h = 2 * r + 2 * gap
        x2 = width - 14
        x1 = x2 - pill_w
        y1 = 14
        self.zone = (x1, y1, x2, y1 + pill_h)
        self.swatches = []
        for i, c in enumerate(BRUSH_COLORS):
            cx = x1 + gap + r + i * (2 * r + gap)
            self.swatches.append((c, cx, y1 + pill_h // 2, r))

    def draw_palette(self, frame):
        x1, y1, x2, y2 = self.zone
        rad = (y2 - y1) // 2
        glass_rect(frame, self.zone, rad)
        outline_rect(frame, self.zone, rad, EDGE, 0.18, 1)
        for c, cx, cy, r in self.swatches:
            cv2.circle(frame, (cx, cy), r, c, -1, AA)
            if c == self.color:
                cv2.circle(frame, (cx, cy), r + 4, (255, 255, 255), 2, AA)

    def pick_color(self, point):
        for c, cx, cy, r in self.swatches:
            if (point[0] - cx) ** 2 + (point[1] - cy) ** 2 <= (r + 8) ** 2:
                self.color = c
                return True
        return False

    # ---------- drawing ----------
    def draw(self, point):
        t = max(1, int(self.thickness))
        if self.prev_point is not None:
            cv2.line(self.canvas, self.prev_point, point, self.color, t, AA)
            cv2.line(self.alpha, self.prev_point, point, 255, t, AA)
        cv2.circle(self.canvas, point, t // 2, self.color, -1, AA)
        cv2.circle(self.alpha, point, t // 2, 255, -1, AA)
        self.prev_point = point

    def erase(self, point, radius=30):
        cv2.circle(self.canvas, point, radius, (0, 0, 0), -1, AA)
        cv2.circle(self.alpha, point, radius, 0, -1, AA)

    def reset_stroke(self):
        self.prev_point = None

    def clear(self):
        self.canvas[:] = 0
        self.alpha[:] = 0
        self.prev_point = None

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_color(self, color):
        self.color = color

    # ---------- output ----------
    def _composite(self, base):
        inv = cv2.bitwise_not(self.alpha)
        inv3 = cv2.merge([inv, inv, inv])
        return cv2.add(cv2.multiply(base, inv3, scale=1 / 255.0), self.canvas)

    def merge_with_frame(self, frame):
        return self._composite(frame)

    def save(self, folder, filename=None):
        import os
        from datetime import datetime

        os.makedirs(folder, exist_ok=True)
        if filename is None:
            filename = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(folder, filename)
        white = np.full_like(self.canvas, 255)
        cv2.imwrite(path, self._composite(white))
        return path