"""Air-writing: collect fingertip strokes, recognise one character with the EMNIST CNN,
append it to a text buffer. Strokes are kept as points and rasterised only when a
character is committed, so the result does not depend on camera resolution."""
import os
import sys
import time
import cv2
import numpy as np

from theme import glass_rect, progress_bar, put_text, MINT, ACCENT, MUTED, TEXT, AA

def _load_model(model_path, classes_path):
    """Rebuild AirCNN from train_airwrite.py and load the saved weights."""
    import json
    import torch
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (root, os.path.dirname(os.path.abspath(__file__))):
        if p not in sys.path:
            sys.path.append(p)
    from train_airwrite import AirCNN, MEAN, STD       # train_airwrite.py must be importable
    with open(classes_path) as f:
        classes = json.load(f)
    model = AirCNN(len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, classes, MEAN, STD


def strokes_to_tensor(strokes, mean, std, canvas=280, content=200, thickness=24):
    """Same preprocessing as stroke_to_tensor() in train_airwrite.py, but strokes are drawn
    separately so multi-stroke letters (E, T, A) are not joined by stray lines."""
    import torch
    pts = np.array([p for st in strokes for p in st], dtype=np.float32)
    if len(pts) < 5:
        return None
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    side = float((mx - mn).max())
    if side < 10:
        return None
    centre = (mn + mx) / 2
    img = np.zeros((canvas, canvas), dtype=np.uint8)
    for st in strokes:
        q = (np.array(st, dtype=np.float32) - centre) * (content / side) + canvas / 2
        q = q.astype(np.int32).reshape(-1, 1, 2)
        if len(q) == 1:
            cv2.circle(img, tuple(q[0, 0]), thickness // 2, 255, -1)
        else:
            cv2.polylines(img, [q], False, 255, thickness, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().div(255.0)
    return ((x - mean) / std).view(1, 1, 28, 28)


class AirWriter:
    def __init__(self, width, height, model_path, classes_path, doc, commit_delay=1.0,
                 action_hold=0.8, min_conf=0.30):
        self.w, self.h = width, height
        self.model, self.classes, self.mean, self.std = _load_model(model_path, classes_path)
        import torch
        torch.set_num_threads(1)               # tiny model: extra threads only compete with MediaPipe
        with torch.no_grad():                  # warm-up so the first character doesn't stall
            self.model(torch.zeros(1, 1, 28, 28))
        self.commit_delay = commit_delay       # pen-up time before a character is final
        self.action_hold = action_hold         # hold palm = space, hold fist = backspace
        self.min_conf = min_conf
        self.strokes = []
        self.cur = None
        self.last_draw = 0.0
        self.doc = doc                         # VirtualKeyboard: its .text is the shared document
        self.top3 = []
        self._hold = {'clear': None, 'erase': None}
        self._fired = {'clear': False, 'erase': False}
        self._seen = {'clear': 0.0, 'erase': 0.0}
        self.grace = 0.3
        self.flash = ('', 0.0)                 # last space/backspace action, shown briefly

    # ---------- state ----------
    def reset_pen(self):
        self.cur = None

    def clear_pending(self):
        self.strokes, self.cur = [], None

    def _commit(self):
        if not self.strokes:
            return
        x = strokes_to_tensor(self.strokes, self.mean, self.std)
        if x is not None:                      # None = flick / dot, ignored
            import torch
            with torch.no_grad():
                p = torch.softmax(self.model(x), dim=1)[0].numpy()
            idx = np.argsort(p)[::-1][:3]
            self.top3 = [(self.classes[i], float(p[i])) for i in idx]
            if self.top3[0][1] >= self.min_conf:
                ch = self.top3[0][0]
                if ch.isalpha():               # same auto-capital rule as the keyboard
                    ch = ch.upper() if self.doc._auto_cap() else ch.lower()
                self.doc.text += ch
        self.clear_pending()

    # ---------- per frame ----------
    def update(self, gesture, tip, in_ui):
        now = time.time()

        if gesture == 'draw' and tip is not None and not in_ui:
            if self.cur is None:
                self.cur = []
                self.strokes.append(self.cur)
            self.cur.append(tip)
            self.last_draw = now
        else:
            self.cur = None

        if self.strokes and self.cur is None and now - self.last_draw >= self.commit_delay:
            self._commit()

        # palm hold = space, fist hold = backspace (fist keeps deleting while held).
        # A short grace period lets the classifier flicker without restarting the hold.
        for g in ('clear', 'erase'):
            if gesture == g:
                self._seen[g] = now
                if self._hold[g] is None:
                    self._hold[g] = now
                if now - self._hold[g] >= self.action_hold:
                    self.clear_pending()           # drop stray points picked up while the hand changed shape
                    if g == 'clear':
                        if not self._fired[g]:
                            self.doc.text += " "
                            self._fired[g] = True
                            self.flash = ("SPACE added", now)
                    else:
                        self.doc.text = self.doc.text[:-1]
                        self.flash = ("BACKSPACE", now)
                        self._hold[g] = now        # repeat every action_hold seconds
            elif self._hold[g] is not None and now - self._seen[g] > self.grace:
                self._hold[g] = None
                self._fired[g] = False

    def pending_progress(self):
        if not self.strokes or self.cur is not None:
            return None
        return min(1.0, (time.time() - self.last_draw) / self.commit_delay)

    # ---------- drawing ----------
    def draw(self, frame, scale=1.0):
        for st in self.strokes:
            pts = np.array(st, dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, MINT, 6, AA)
            elif len(pts) == 1:
                cv2.circle(frame, tuple(pts[0]), 4, MINT, -1, AA)

        # text strip at the bottom
        x1, x2 = 14, self.w - 14
        y2 = self.h - 14
        y1 = y2 - int(86 * scale)
        roi = frame[y1:y2, x1:x2]                       # cheap dark overlay instead of a blurred glass panel
        cv2.addWeighted(roi, 0.35, np.full_like(roi, 28), 0.65, 0, dst=roi)
        shown = self.doc.text.replace("\n", " ")[-38:] + "_"
        put_text(frame, shown, (x1 + 16, y1 + int(40 * scale)), 0.9 * scale, TEXT, 2, shadow=False)
        hint = "pen up 1s = accept  |  palm hold = space  |  fist hold = backspace"
        put_text(frame, hint, (x1 + 16, y2 - 10), 0.42 * scale, MUTED, 1, shadow=False)

        if self.top3:
            guess = "   ".join(f"{c} {p * 100:.0f}%" for c, p in self.top3)
            put_text(frame, guess, (x1 + 16, y1 - 10), 0.55 * scale, ACCENT, 1, shadow=False)

        now = time.time()
        for g, label in (('clear', 'SPACE'), ('erase', 'BACKSPACE')):
            if self._hold[g] is not None:
                frac = min(1.0, (now - self._hold[g]) / self.action_hold)
                bar = (x1 + int(190 * scale), y1 - 30, x1 + int(360 * scale), y1 - 22)
                progress_bar(frame, bar, frac, MINT)
                put_text(frame, label, (bar[2] + 10, y1 - 20), 0.45 * scale, MUTED, 1, shadow=False)
        if self.flash[0] and now - self.flash[1] < 1.0:
            put_text(frame, self.flash[0], (x2 - int(190 * scale), y1 + int(40 * scale)),
                     0.7 * scale, MINT, 2, shadow=False)

        prog = self.pending_progress()
        if prog is not None:
            progress_bar(frame, (x1, y1 - 30, x1 + int(170 * scale), y1 - 22), prog, ACCENT)
