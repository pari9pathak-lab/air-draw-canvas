import cv2
import os
import re
import time
from datetime import datetime
from theme import (glass_rect, outline_rect, put_text, put_text_center, text_size,
                   fit_scale, ui_scale, TEXT, MUTED, ACCENT_LIGHT, MINT, EDGE,
                   PANEL, ACCENT, AA)

_TOKEN = re.compile(r'\S+\s*|\s+')


class VirtualKeyboard:
    ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM,.?"]
    # (key width as a fraction of frame width, key height as a fraction of frame height)
    SIZES = {'small': (0.060, 0.085), 'medium': (0.075, 0.105), 'large': (0.090, 0.130)}

    def __init__(self, frame_w, frame_h, dwell_time=0.8, size='medium'):
        self.w, self.h = frame_w, frame_h
        self.dwell_time = dwell_time
        self.s = ui_scale(frame_h)
        self.size = size
        self.text = ""
        self.keys = []              # (label, x1, y1, x2, y2)
        self.hover_key = None
        self.hover_start = None
        self._wrap_key = None
        self._wrap_lines = [""]
        self._build_layout()

    # ---------- layout ----------
    def _build_layout(self):
        w, h, s = self.w, self.h, self.s
        kw_frac, kh_frac = self.SIZES[self.size]
        gap = max(4, int(8 * s))
        key_w = min(int(w * kw_frac), (w - 24 - 9 * gap) // 10)   # never wider than the screen
        key_h = int(h * kh_frac)

        self.keys = []
        self.top = h - (len(self.ROWS) + 1) * (key_h + gap) - 14

        for r, row in enumerate(self.ROWS):
            row_width = len(row) * (key_w + gap) - gap
            x = (w - row_width) // 2
            y = self.top + r * (key_h + gap)
            for ch in row:
                self.keys.append((ch, x, y, x + key_w, y + key_h))
                x += key_w + gap

        y = self.top + len(self.ROWS) * (key_h + gap)
        special = [("BACK", 2), ("SPACE", 3), ("ENTER", 2), ("CLEAR", 2)]
        widths = [key_w * n + gap * (n - 1) for _, n in special]
        total = sum(widths) + gap * (len(special) - 1)
        x = (w - total) // 2
        for (label, _), kw in zip(special, widths):
            self.keys.append((label, x, y, x + kw, y + key_h))
            x += kw + gap

    def set_size(self, name):
        self.size = name
        self.hover_key = None
        self.hover_start = None
        self._build_layout()

    # ---------- typing ----------
    def _key_at(self, point):
        px, py = point
        for label, x1, y1, x2, y2 in self.keys:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return label
        return None

    def _auto_cap(self):
        t = self.text
        return t == "" or t.endswith("\n") or re.search(r'[.!?] +$', t) is not None

    def _press(self, key):
        if key == "BACK":
            self.text = self.text[:-1]
        elif key == "SPACE":
            self.text += " "
        elif key == "ENTER":
            self.text += "\n"
        elif key == "CLEAR":
            self.text = ""
        elif key.isalpha():
            self.text += key.upper() if self._auto_cap() else key.lower()
        else:
            self.text += key

    def _dwell(self, key):
        # CLEAR wipes the whole document, so it needs a longer hold
        return self.dwell_time * 2 if key == "CLEAR" else self.dwell_time

    def update(self, point):
        key = self._key_at(point) if point else None
        now = time.time()
        if key is None or key != self.hover_key:
            self.hover_key = key
            self.hover_start = now if key else None
            return None
        if now - self.hover_start >= self._dwell(key):
            self._press(key)
            self.hover_start = now
            return key
        return None

    def progress(self):
        if self.hover_key is None or self.hover_start is None:
            return 0.0
        return min((time.time() - self.hover_start) / self._dwell(self.hover_key), 1.0)

    # ---------- saving ----------
    def save(self, folder):
        if not self.text.strip():
            return None
        os.makedirs(folder, exist_ok=True)
        name = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text)
        return path

    # ---------- drawing: keys ----------
    def draw_keys(self, frame):
        s = self.s
        for label, x1, y1, x2, y2 in self.keys:
            rect = (x1, y1, x2, y2)
            rad = int((y2 - y1) * 0.25)
            active = (label == self.hover_key)

            glass_rect(frame, rect, rad, PANEL, 0.78)
            if active:
                fw = int((x2 - x1) * self.progress())
                glass_rect(frame, (x1, y1, x1 + fw, y2), rad, ACCENT, 0.85)
            outline_rect(frame, rect, rad, MINT if active else EDGE,
                         0.95 if active else 0.22, 2 if active else 1)

            base = 0.85 * s if len(label) == 1 else 0.5 * s
            fs = fit_scale(label, (x2 - x1) - 10, base)
            put_text_center(frame, label, rect, fs, TEXT, 1)

    # ---------- drawing: document page ----------
    def _wrap(self, text, max_w, fs):
        key = (text, max_w, round(fs, 3))
        if key == self._wrap_key:
            return self._wrap_lines

        lines = []
        for para in text.split("\n"):
            cur = ""
            for tok in _TOKEN.findall(para):
                # a single word wider than the page gets broken up
                while text_size(tok.rstrip(), fs)[0] > max_w:
                    n = len(tok.rstrip())
                    while n > 1 and text_size(tok[:n], fs)[0] > max_w:
                        n -= 1
                    if cur:
                        lines.append(cur)
                        cur = ""
                    lines.append(tok[:n])
                    tok = tok[n:]
                    if not tok:
                        break
                if not tok:
                    continue
                if cur and text_size((cur + tok).rstrip(), fs)[0] > max_w:
                    lines.append(cur)
                    cur = tok
                else:
                    cur += tok
            lines.append(cur)

        self._wrap_key = key
        self._wrap_lines = lines
        return lines

    def draw_document(self, frame, rect):
        s = self.s
        x1, y1, x2, y2 = rect
        rad = int(18 * s)
        glass_rect(frame, rect, rad, PANEL, 0.80)
        outline_rect(frame, rect, rad, ACCENT, 0.5, 2)

        pad = int(18 * s)
        hfs = 0.5 * s
        hth = text_size("Ag", hfs)[1]
        hy = y1 + pad + hth
        words = len(self.text.split())
        meta = f"{words} words  |  {len(self.text)} characters"
        put_text(frame, "Untitled document", (x1 + pad, hy), hfs, ACCENT_LIGHT, 1, shadow=False)
        mw = text_size(meta, hfs)[0]
        put_text(frame, meta, (x2 - pad - mw, hy), hfs, MUTED, 1, shadow=False)
        div_y = hy + int(10 * s)
        cv2.line(frame, (x1 + pad, div_y), (x2 - pad, div_y), (95, 95, 115), 1, AA)

        fs = 0.8 * s
        line_h = int(36 * s)
        asc = text_size("Ag", fs)[1]
        top_text = div_y + int(12 * s)
        max_w = (x2 - x1) - 2 * pad
        n_vis = max(1, (y2 - pad - top_text) // line_h)

        lines = self._wrap(self.text, max_w, fs)
        visible = lines[-n_vis:]                      # scrolls so the cursor stays on screen

        for i, ln in enumerate(visible):
            if ln:
                put_text(frame, ln, (x1 + pad, top_text + i * line_h + asc),
                         fs, TEXT, 1, shadow=False)

        if not self.text:
            put_text(frame, "Point at a key and hold to start typing...",
                     (x1 + pad, top_text + asc), fs * 0.85, MUTED, 1, shadow=False)

        last = visible[-1]
        cx = x1 + pad + (text_size(last, fs)[0] if last else 0) + 2
        cy = top_text + (len(visible) - 1) * line_h
        if int(time.time() * 2) % 2 == 0:
            cv2.line(frame, (cx, cy), (cx, cy + line_h - int(8 * s)), MINT, 2, AA)