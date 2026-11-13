import cv2
import time
from theme import (glass_rect, outline_rect, put_text, put_text_center, text_size,
                   fit_scale, ui_scale, PANEL, TEXT, MUTED, ACCENT, ACCENT_LIGHT,
                   MINT, EDGE, AA)


class DwellButton:
    def __init__(self, label, rect, dwell_time=0.8):
        self.label = label
        self.rect = rect                      # (x1, y1, x2, y2)
        self.dwell_time = dwell_time
        self.hover_start = None
        self.locked = False
        self.hovered = False
        self.selected = False                 # used by the S / M / L size buttons

    def contains(self, point):
        if point is None:
            return False
        x1, y1, x2, y2 = self.rect
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

    def reset(self):
        self.hover_start = None
        self.locked = False
        self.hovered = False

    def update(self, point):
        """Returns True once, when the fingertip has dwelled long enough."""
        self.hovered = self.contains(point)
        if not self.hovered:
            self.hover_start = None
            self.locked = False
            return False
        if self.locked:
            return False
        now = time.time()
        if self.hover_start is None:
            self.hover_start = now
            return False
        if now - self.hover_start >= self.dwell_time:
            self.hover_start = None
            self.locked = True               # must leave the button before it can fire again
            return True
        return False

    def progress(self):
        if self.hover_start is None:
            return 0.0
        return min((time.time() - self.hover_start) / self.dwell_time, 1.0)

    def draw(self, frame, font_scale=0.55):
        x1, y1, x2, y2 = self.rect
        rad = int((y2 - y1) * 0.4)
        glass_rect(frame, self.rect, rad)
        if self.selected:
            glass_rect(frame, self.rect, rad, ACCENT, 0.55)
        p = self.progress()
        if p > 0:
            glass_rect(frame, (x1, y1, x1 + int((x2 - x1) * p), y2), rad, ACCENT, 0.85)
        if self.hovered:
            outline_rect(frame, self.rect, rad, ACCENT_LIGHT, 0.95, 2)
        elif self.selected:
            outline_rect(frame, self.rect, rad, MINT, 0.9, 2)
        else:
            outline_rect(frame, self.rect, rad, EDGE, 0.22, 1)
        fs = fit_scale(self.label, (x2 - x1) - 12, font_scale)
        put_text_center(frame, self.label, self.rect, fs, TEXT, 1)


# ---------- HUD pieces ----------
def draw_status(frame, title, detail, dot_color, s, reserve=None):
    """Pill in the top-left. `reserve` is the widest detail text, so the pill doesn't jitter."""
    fs = 0.58 * s
    dfs = fs * 0.9
    pad = int(14 * s)
    hh = int(36 * s)
    dr = int(6 * s)
    sep = int(14 * s)
    x1, y1 = 14, 14

    tw1, th1 = text_size(title, fs, 2)
    measure = reserve if reserve else detail
    tw2 = text_size(measure, dfs, 1)[0] if measure else 0
    width = pad + 2 * dr + int(10 * s) + tw1 + (sep + tw2 if measure else 0) + pad

    rect = (x1, y1, x1 + width, y1 + hh)
    glass_rect(frame, rect, hh // 2)
    outline_rect(frame, rect, hh // 2, EDGE, 0.18, 1)
    cv2.circle(frame, (x1 + pad + dr, y1 + hh // 2), dr, dot_color, -1, AA)

    tx = x1 + pad + 2 * dr + int(10 * s)
    put_text(frame, title, (tx, y1 + (hh + th1) // 2), fs, TEXT, 2, shadow=False)
    if detail:
        dh = text_size(detail, dfs, 1)[1]
        put_text(frame, detail, (tx + tw1 + sep, y1 + (hh + dh) // 2),
                 dfs, MUTED, 1, shadow=False)
    return y1 + hh


def draw_toast(frame, text, s):
    fs = 0.6 * s
    tw, th = text_size(text, fs, 1)
    pad = int(18 * s)
    hh = int(38 * s)
    w = tw + 2 * pad + int(20 * s)
    x1 = (frame.shape[1] - w) // 2
    y1 = 14
    rect = (x1, y1, x1 + w, y1 + hh)
    glass_rect(frame, rect, hh // 2)
    outline_rect(frame, rect, hh // 2, MINT, 0.9, 2)
    cv2.circle(frame, (x1 + pad + int(5 * s), y1 + hh // 2), int(6 * s), MINT, -1, AA)
    put_text(frame, text, (x1 + pad + int(20 * s), y1 + (hh + th) // 2), fs, TEXT, 1)


# ---------- help menu ----------
class HelpMenu:
    LEFT = [
        ('h', "DRAW MODE", ""),
        ('r', "Index finger", "draw"),
        ('r', "Fist", "erase"),
        ('r', "Open palm", "thumb out: clear 2 s"),
        ('r', "2 fingers", "hold: 8 px"),
        ('r', "3 fingers", "hold: 12 px"),
        ('r', "4 fingers", "thumb in: 16 px"),
        ('r', "On color", "pick it"),
        ('r', "On button", "hold 0.8 s"),
    ]
    RIGHT = [
        ('h', "KEYBOARD MODE", ""),
        ('r', "On a key", "hold 0.8 s"),
        ('r', "CLEAR key", "hold 1.6 s"),
        ('h', "SHORTCUTS", ""),
        ('r', "K", "switch mode"),
        ('r', "M or Esc", "menu"),
        ('r', "S", "save all"),
        ('r', "1 2 3", "key size"),
        ('r', "[  ]", "brush size"),
        ('r', "H", "hide keys"),
        ('r', "F", "fullscreen"),
        ('r', "Q", "quit"),
    ]

    def __init__(self, w, h, dwell_time=0.8):
        self.w, self.h = w, h
        self.s = ui_scale(h)
        self.margin = int(w * 0.03)
        bh = int(h * 0.09)
        y2 = h - self.margin - 10
        y1 = y2 - bh
        sw, cw, gap = int(w * 0.45), int(w * 0.2), 20
        x = (w - (sw + cw + gap)) // 2
        self.switch_btn = DwellButton("", (x, y1, x + sw, y2), dwell_time)
        self.close_btn = DwellButton("CLOSE", (x + sw + gap, y1, x + sw + gap + cw, y2),
                                     dwell_time)

    @property
    def buttons(self):
        return [self.switch_btn, self.close_btn]

    def _rows(self, frame, x, y0, items, col_w, line_h):
        s = self.s
        base = 0.62 * s
        key_w = int(col_w * 0.5)
        ch = int(line_h * 0.78)
        for i, (kind, a, b) in enumerate(items):
            yc = y0 + i * line_h
            if kind == 'h':
                fs = 0.7 * s
                th = text_size(a, fs, 2)[1]
                put_text(frame, a, (x, yc + th // 2), fs, ACCENT_LIGHT, 2, shadow=False)
            elif kind == 'r':
                chip = (x, yc - ch // 2, x + key_w - 8, yc + ch // 2)
                rad = int(ch * 0.3)
                glass_rect(frame, chip, rad, ACCENT, 0.22)
                outline_rect(frame, chip, rad, ACCENT, 0.55, 1)
                fa = fit_scale(a, (chip[2] - chip[0]) - 12, base)
                put_text_center(frame, a, chip, fa, MINT, 1)
                fb = fit_scale(b, col_w - key_w - 6, base)
                th = text_size(b, fb, 1)[1]
                put_text(frame, b, (x + key_w + 4, yc + th // 2), fb, TEXT, 1)

    def draw(self, frame, mode):
        s, m, w, h = self.s, self.margin, self.w, self.h
        panel = (m, m, w - m, h - m)
        rad = int(26 * s)
        glass_rect(frame, panel, rad, PANEL, 0.84)
        outline_rect(frame, panel, rad, ACCENT, 0.45, 2)

        pad = int(26 * s)
        put_text(frame, "Air Draw Canvas", (m + pad, m + int(46 * s)), 1.15 * s, TEXT, 2)
        sub = "Thumb position is ignored   |   Shortcuts work when the video window is focused"
        sfs = fit_scale(sub, w - 2 * (m + pad), 0.5 * s)
        put_text(frame, sub, (m + pad, m + int(72 * s)), sfs, MUTED, 1, shadow=False)

        line_h = int(h * 0.052)
        y0 = int(h * 0.27)
        gap = int(30 * s)
        col_w = (w - 2 * (m + pad) - gap) // 2
        self._rows(frame, m + pad, y0, self.LEFT, col_w, line_h)
        self._rows(frame, m + pad + col_w + gap, y0, self.RIGHT, col_w, line_h)

        self.switch_btn.label = ("SWITCH TO KEYBOARD MODE" if mode == 'draw'
                                 else "SWITCH TO DRAW MODE")
        self.switch_btn.draw(frame, 0.62 * s)
        self.close_btn.draw(frame, 0.62 * s)