import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


def rgb(r, g, b):
    return (b, g, r)  # OpenCV uses BGR


# ---------- colours ----------
PANEL = rgb(16, 18, 30)
PANEL_ALPHA = 0.62
TEXT = rgb(245, 246, 250)
MUTED = rgb(160, 166, 185)
ACCENT = rgb(124, 92, 255)         # violet
ACCENT_LIGHT = rgb(176, 156, 255)
MINT = rgb(46, 230, 166)
EDGE = rgb(255, 255, 255)

BRUSH_COLORS = [
    rgb(255, 99, 132),    # rose
    rgb(255, 176, 59),    # amber
    rgb(46, 230, 166),    # mint
    rgb(64, 170, 255),    # sky
    rgb(160, 108, 255),   # violet
    rgb(245, 245, 250),   # white
]


def ui_scale(h):
    """Scales every UI element with the frame height."""
    return max(0.8, h / 720)


# ---------- rounded, translucent shapes ----------
_mask_cache = {}
_outline_cache = {}


def _rounded_mask(w, h, r):
    key = (w, h, r)
    if key not in _mask_cache:
        m = np.zeros((h, w), np.uint8)
        r = max(0, min(r, w // 2, h // 2))
        cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(m, (cx, cy), r, 255, -1, AA)
        _mask_cache[key] = m.astype(np.float32) / 255.0
    return _mask_cache[key]


def _outline_mask(w, h, r, t):
    key = (w, h, r, t)
    if key not in _outline_cache:
        outer = _rounded_mask(w, h, r)
        inner = np.zeros_like(outer)
        iw, ih = w - 2 * t, h - 2 * t
        if iw > 2 and ih > 2:
            inner[t:t + ih, t:t + iw] = _rounded_mask(iw, ih, max(r - t, 0))
        _outline_cache[key] = np.clip(outer - inner, 0, 1)
    return _outline_cache[key]


def _clip(frame, rect):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = rect
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


_blend_cache = {}


def _blend_cached(frame, x1, y1, x2, y2, key, make_mask, color, alpha):
    """Same look as a float alpha-blend, but done with cached uint8 arrays and OpenCV
    (roughly 5-10x faster per panel)."""
    ent = _blend_cache.get(key)
    if ent is None:
        mask = make_mask()[..., None]
        inv = np.ascontiguousarray(((1.0 - mask * alpha) * 255).astype(np.uint8).repeat(3, axis=2))
        pm = np.ascontiguousarray((mask * alpha * np.array(color, np.float32)).astype(np.uint8))
        if len(_blend_cache) > 600:
            _blend_cache.clear()
        ent = _blend_cache[key] = (inv, pm)
    inv, pm = ent
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.add(cv2.multiply(roi, inv, scale=1 / 255.0), pm)


def glass_rect(frame, rect, radius, color=PANEL, alpha=PANEL_ALPHA):
    x1, y1, x2, y2 = _clip(frame, rect)
    w, h = x2 - x1, y2 - y1
    if w < 3 or h < 3:
        return
    _blend_cached(frame, x1, y1, x2, y2, ('g', w, h, radius, color, round(alpha, 3)),
                  lambda: _rounded_mask(w, h, radius), color, alpha)


def outline_rect(frame, rect, radius, color=EDGE, alpha=0.25, thickness=1):
    x1, y1, x2, y2 = _clip(frame, rect)
    w, h = x2 - x1, y2 - y1
    if w < 3 * thickness or h < 3 * thickness:
        return
    _blend_cached(frame, x1, y1, x2, y2, ('o', w, h, radius, thickness, color, round(alpha, 3)),
                  lambda: _outline_mask(w, h, radius, thickness), color, alpha)


# ---------- text ----------
def text_size(txt, scale, thick=1):
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    return tw, th


def put_text(frame, txt, org, scale=0.6, color=TEXT, thick=1, shadow=True):
    x, y = org
    if shadow:
        cv2.putText(frame, txt, (x + 1, y + 1), FONT, scale, (0, 0, 0), thick + 1, AA)
    cv2.putText(frame, txt, (x, y), FONT, scale, color, thick, AA)


def put_text_center(frame, txt, rect, scale=0.6, color=TEXT, thick=1, shadow=True):
    x1, y1, x2, y2 = rect
    tw, th = text_size(txt, scale, thick)
    put_text(frame, txt, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
             scale, color, thick, shadow)


def fit_scale(txt, max_w, base, thick=1, min_scale=0.3):
    s = base
    while s > min_scale and text_size(txt, s, thick)[0] > max_w:
        s -= 0.03
    return s


# ---------- small widgets ----------
def progress_bar(frame, rect, p, color=MINT):
    x1, y1, x2, y2 = rect
    r = (y2 - y1) // 2
    glass_rect(frame, rect, r, PANEL, 0.6)
    fw = int((x2 - x1) * p)
    if fw > 3:
        glass_rect(frame, (x1, y1, x1 + fw, y2), r, color, 0.95)


def draw_cursor(frame, pt, color=TEXT, radius=12):
    cv2.circle(frame, pt, radius + 2, (0, 0, 0), 2, AA)
    cv2.circle(frame, pt, radius, color, 2, AA)
    cv2.circle(frame, pt, 3, color, -1, AA)