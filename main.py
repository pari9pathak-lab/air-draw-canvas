import cv2
import sys
import os
import time
import collections

sys.path.append(os.path.dirname(__file__))
from hand_tracker import HandTracker
from gesture_detector import GestureDetector
from canvas import DrawingCanvas
from virtual_keyboard import VirtualKeyboard
from ui import DwellButton, HelpMenu, draw_status, draw_toast
from theme import (MINT, ACCENT, MUTED, TEXT, AA, ui_scale,
                   progress_bar, put_text, put_text_center, draw_cursor)
from utils import OneEuroSmoother, HoldTimer, StableValue
from camera import ThreadedCamera
from gesture_classifier import GestureClassifier
from airwrite import AirWriter
from suggest import WordModel, SuggestionBar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_FOLDER = os.path.join(ROOT, 'assets', 'saved_drawings')
SAVE_TEXT_FOLDER = os.path.join(ROOT, 'assets', 'saved_texts')
WINDOW_NAME = "Air Draw Canvas"
SIZE_ORDER = ['small', 'medium', 'large']
THUMB_OUT = 0.6     # thumb-out ratio above this = open palm (clear); below = thumb tucked (4 fingers = 16 px).
                    # The live value is shown in the status pill as "T 0.xx": tune this number to your hand.


def thumb_ratio(landmarks):
    """How far the thumb tip is from the base of the index finger, relative to palm length."""
    p = {i: (x, y) for i, x, y in landmarks}
    palm = max(1.0, ((p[0][0] - p[9][0]) ** 2 + (p[0][1] - p[9][1]) ** 2) ** 0.5)
    return (((p[4][0] - p[5][0]) ** 2 + (p[4][1] - p[5][1]) ** 2) ** 0.5) / palm


def inside(p, r):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def grow(r, n):
    return (r[0] - n, r[1] - n, r[2] + n, r[3] + n)


def main():
    cap = ThreadedCamera(0, 640, 480)           # background capture thread, 640x480, MJPG
    T = collections.defaultdict(float)          # per-stage timing, printed when you press q
    frame_count = 0
    # Recommended for the document view (more room to type). Remove if your camera struggles:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker()
    detector = GestureDetector()
    clf = GestureClassifier(every = 2, smooth = 3)      # trained classifier
    use_classifier = True          # press 'c' while running to switch between classifier and old rules
    smoother = OneEuroSmoother()                # replaces the 5-point average: steadier AND less laggy
    clear_timer = HoldTimer(hold_duration=2.0)
    thickness_hold = StableValue(hold_duration=0.8)

    ret, frame = cap.read()
    h, w, _ = frame.shape
    s = ui_scale(h)
    drawing_canvas = DrawingCanvas(w, h)

    kb_size = 'medium'
    keys_visible = True
    keyboard = VirtualKeyboard(w, h, dwell_time=0.8, size=kb_size)
    menu = HelpMenu(w, h, dwell_time=0.8)
    word_model = WordModel(os.path.join(ROOT, 'data', 'ngram.pkl'))   # first run builds + caches bigrams
    sugg = SuggestionBar(word_model, dwell_time=0.8)
    writer = AirWriter(w, h,
                       os.path.join(ROOT, 'airwrite_cnn.pt'),
                       os.path.join(ROOT, 'airwrite_classes.json'),
                       keyboard)             # air-written letters go into the keyboard's document

    # Buttons (their positions are set by layout(), which depends on the mode)
    menu_btn = DwellButton("MENU", (0, 0, 0, 0), 0.8)
    mode_btn = DwellButton("TO KEYBOARD", (0, 0, 0, 0), 0.8)
    save_btn = DwellButton("SAVE", (0, 0, 0, 0), 0.8)
    write_btn = DwellButton("WRITE", (0, 0, 0, 0), 0.8)
    keys_btn = DwellButton("HIDE KEYS", (0, 0, 0, 0), 0.8)
    size_btns = {name: DwellButton(lbl, (0, 0, 0, 0), 0.8)
                 for name, lbl in zip(SIZE_ORDER, ("S", "M", "L"))}
    L = {}                       # layout results: page rect + "SIZE" caption rect
    btn_font = 0.55 * s

    mode = 'draw'
    menu_open = False
    last_save_time = 0
    toast_text = ""
    fullscreen = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    # ---------- layout ----------
    def layout():
        gap = int(8 * s)
        if mode in ('draw', 'write'):
            # vertical stack under the colour palette
            x2 = w - 14
            x1 = x2 - int(160 * s)
            bh = int(40 * s)
            y = drawing_canvas.zone[3] + int(10 * s)
            for b in (menu_btn, mode_btn, save_btn, write_btn):
                b.rect = (x1, y, x2, y + bh)
                y += bh + gap
            if mode == 'write':
                # suggestion row sits just above the air-writing text strip
                sw = min(w - 28, int(620 * s))
                sb = h - 14 - int(86 * s) - int(44 * s)
                sx = (w - sw) // 2
                sugg.set_rect((sx, sb - int(38 * s), sx + sw, sb))
        else:
            # toolbar row, sitting just above the keys (or at the bottom if keys are hidden)
            bh = int(38 * s)
            y2 = (keyboard.top - gap) if keys_visible else (h - 14)
            y1 = y2 - bh
            items = [(menu_btn, 84), (mode_btn, 118), (save_btn, 84), (keys_btn, 112),
                     (None, 46)] + [(size_btns[n], 38) for n in SIZE_ORDER]
            widths = [int(wd * s) for _, wd in items]
            total = sum(widths) + gap * (len(items) - 1)
            x = (w - total) // 2
            for (b, _), wd in zip(items, widths):
                if b is None:
                    L['caption'] = (x, y1, x + wd, y2)
                else:
                    b.rect = (x, y1, x + wd, y2)
                x += wd + gap
            page_top = 14 + int(36 * s) + 10
            sh = int(38 * s)
            sw = min(w - 28, int(620 * s))
            sx = (w - sw) // 2
            sugg.set_rect((sx, y1 - 8 - sh, sx + sw, y1 - 8))
            L['page'] = (14, page_top, w - 14, y1 - 8 - sh - 8)

    def go(new_mode):
        nonlocal mode
        mode = new_mode
        drawing_canvas.reset_stroke()
        writer.clear_pending()
        smoother.reset()
        keyboard.update(None)
        thickness_hold.update(None)
        clear_timer.update(False)
        layout()

    def toggle_mode():
        go('draw' if mode == 'keyboard' else 'keyboard')

    def toggle_write():
        go('draw' if mode == 'write' else 'write')

    def toggle_menu():
        nonlocal menu_open
        menu_open = not menu_open
        for b in (menu_btn, mode_btn, save_btn, write_btn, keys_btn, *size_btns.values(), *menu.buttons, *sugg.buttons):
            b.reset()
        drawing_canvas.reset_stroke()
        writer.reset_pen()
        smoother.reset()
        thickness_hold.update(None)
        clear_timer.update(False)

    def set_size(name):
        nonlocal kb_size
        kb_size = name
        keyboard.set_size(name)
        layout()

    def toggle_keys():
        nonlocal keys_visible
        keys_visible = not keys_visible
        keyboard.update(None)
        layout()

    def do_save():
        nonlocal toast_text, last_save_time
        saved = []
        if drawing_canvas.alpha.any():
            print(f"Saved: {drawing_canvas.save(SAVE_FOLDER)}")
            saved.append("drawing")
        text_path = keyboard.save(SAVE_TEXT_FOLDER)
        if text_path:
            print(f"Saved: {text_path}")
            saved.append("text")
        toast_text = ("Saved " + " + ".join(saved)) if saved else "Nothing to save yet"
        last_save_time = time.time()

    layout()

    prev_t = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        t0 = time.perf_counter()
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=False)
        landmarks = tracker.get_landmark_positions(frame)
        T['1 tracker'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        conf = 1.0
        if landmarks:
            if use_classifier:
                gesture, conf = clf.predict(landmarks, w, h)
            else:
                gesture = detector.get_gesture(landmarks)
                if gesture == 'hover':
                    gesture = 'clear'      # the old rules call the open palm 'hover'
        else:
            gesture = 'none'
            clf.history.clear()
        finger_count = detector.count_fingers(landmarks) if landmarks else 0
        t_ratio = thumb_ratio(landmarks) if landmarks else 0.0
        if landmarks and finger_count in (2, 3):
            gesture = 'unknown'      # 2 / 3 fingers are brush-size poses: decided by finger count, not the classifier
        elif landmarks and finger_count == 4:
            # four fingers up: thumb tucked = brush 16 px, thumb out = open palm = clear
            gesture, conf = ('clear', 1.0) if t_ratio > THUMB_OUT else ('unknown', 1.0)
        T['2 classifier'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        raw_tip = (landmarks[8][1], landmarks[8][2]) if landmarks else None
        mode_btn.label = "TO DRAW" if mode == 'keyboard' else "TO KEYBOARD"
        write_btn.label = "EXIT WRITE" if mode == 'write' else "WRITE"
        keys_btn.label = "HIDE KEYS" if keys_visible else "SHOW KEYS"
        for n, b in size_btns.items():
            b.selected = (n == kb_size)

        if menu_open:
            # ---------- HELP MENU ----------
            frame = drawing_canvas.merge_with_frame(frame)
            menu.draw(frame, 'keyboard' if mode == 'keyboard' else 'draw')
            if menu.switch_btn.update(raw_tip):
                toggle_mode()
            if menu.close_btn.update(raw_tip):
                toggle_menu()
            if raw_tip:
                draw_cursor(frame, raw_tip, TEXT)

        elif mode == 'draw':
            # ---------- DRAW MODE ----------
            in_ui = raw_tip is not None and (
                inside(raw_tip, grow(drawing_canvas.zone, 8))
                or inside(raw_tip, grow(menu_btn.rect, 6))
                or inside(raw_tip, grow(mode_btn.rect, 6))
                or inside(raw_tip, grow(save_btn.rect, 6))
                or inside(raw_tip, grow(write_btn.rect, 6)))

            candidate = None
            if (landmarks and not in_ui
                    and gesture not in ('draw', 'erase', 'clear')
                    and finger_count in (2, 3, 4)):
                candidate = finger_count

            stable = thickness_hold.update(candidate)
            if stable is not None:
                drawing_canvas.set_thickness(stable * 4)

            if gesture == 'draw' and landmarks and not in_ui:
                drawing_canvas.draw(smoother.smooth(raw_tip))
            else:
                drawing_canvas.reset_stroke()
                smoother.reset()

            erase_pt = None
            if gesture == 'erase' and landmarks:
                erase_pt = (landmarks[9][1], landmarks[9][2])
                drawing_canvas.erase(erase_pt)

            clearing = gesture == 'clear' and conf >= 0.8
            if clear_timer.update(clearing):
                drawing_canvas.clear()

            frame = drawing_canvas.merge_with_frame(frame)

            drawing_canvas.draw_palette(frame)
            if landmarks and gesture != 'erase' and inside(raw_tip, drawing_canvas.zone):
                drawing_canvas.pick_color(raw_tip)

            bottom = draw_status(frame, "DRAW",
                                 f"{gesture}  |  {drawing_canvas.thickness} px  |  T {t_ratio:.2f}",
                                 drawing_canvas.color, s, reserve="unknown  |  20 px  |  T 0.00")
            bar_y = bottom + int(10 * s)
            bar_w, bar_h = int(170 * s), int(8 * s)
            if clearing:
                progress_bar(frame, (14, bar_y, 14 + bar_w, bar_y + bar_h),
                             clear_timer.progress(), MINT)
                put_text(frame, "CLEARING", (14 + bar_w + 10, bar_y + bar_h),
                         0.42 * s, MUTED, 1, shadow=False)
            elif candidate is not None:
                progress_bar(frame, (14, bar_y, 14 + bar_w, bar_y + bar_h),
                             thickness_hold.progress(), ACCENT)
                put_text(frame, "BRUSH SIZE", (14 + bar_w + 10, bar_y + bar_h),
                         0.42 * s, MUTED, 1, shadow=False)

            pt = raw_tip if gesture != 'erase' else None
            if menu_btn.update(pt):
                toggle_menu()
            elif mode_btn.update(pt):
                toggle_mode()
            elif save_btn.update(pt):
                do_save()
            elif write_btn.update(pt):
                toggle_write()
            for b in (menu_btn, mode_btn, save_btn, write_btn):
                b.draw(frame, btn_font)

            if erase_pt:
                cv2.circle(frame, erase_pt, 30, TEXT, 2, AA)
            elif raw_tip:
                if in_ui:
                    draw_cursor(frame, raw_tip, TEXT)
                else:
                    cv2.circle(frame, raw_tip, max(4, drawing_canvas.thickness // 2 + 4),
                               drawing_canvas.color, 2, AA)

        elif mode == 'write':
            # ---------- AIR-WRITE MODE ----------
            in_ui = raw_tip is not None and any(
                inside(raw_tip, grow(b.rect, 6))
                for b in (menu_btn, mode_btn, save_btn, write_btn, *sugg.buttons))
            pen = None
            if landmarks and gesture == 'draw' and not in_ui:
                pen = smoother.smooth(raw_tip)
            else:
                smoother.reset()
            g = gesture if landmarks else 'none'
            if g == 'clear' and conf < 0.6:
                g = 'unknown'
            writer.update(g, pen, in_ui)

            frame = drawing_canvas.merge_with_frame(frame)
            writer.draw(frame, s)
            draw_status(frame, "WRITE", gesture, MINT, s, reserve="unknown")

            pt = raw_tip if gesture != 'erase' else None
            if menu_btn.update(pt):
                toggle_menu()
            elif mode_btn.update(pt):
                toggle_mode()
            elif save_btn.update(pt):
                do_save()
            elif write_btn.update(pt):
                toggle_write()
            else:
                sugg.update(pt, keyboard)
            for b in (menu_btn, mode_btn, save_btn, write_btn):
                b.draw(frame, btn_font)
            sugg.draw(frame, btn_font)
            if raw_tip:
                draw_cursor(frame, raw_tip, MINT)

        else:
            # ---------- KEYBOARD / DOCUMENT MODE ----------
            frame = drawing_canvas.merge_with_frame(frame)

            tip = None
            if landmarks:
                tip = smoother.smooth(raw_tip)
            else:
                smoother.reset()
            keyboard.update(tip if keys_visible else None)

            keyboard.draw_document(frame, L['page'])
            if keys_visible:
                keyboard.draw_keys(frame)

            draw_status(frame, "KEYBOARD",
                        f"{kb_size} keys" if keys_visible else "keys hidden",
                        MINT, s, reserve="medium keys")

            if menu_btn.update(raw_tip):
                toggle_menu()
            elif mode_btn.update(raw_tip):
                toggle_mode()
            elif save_btn.update(raw_tip):
                do_save()
            elif keys_btn.update(raw_tip):
                toggle_keys()
            elif sugg.update(raw_tip, keyboard):
                pass
            else:
                for n, b in size_btns.items():
                    if b.update(raw_tip):
                        set_size(n)
                        break
            sugg.draw(frame, btn_font)

            for b in (menu_btn, mode_btn, save_btn, keys_btn, *size_btns.values()):
                b.draw(frame, btn_font)
            put_text_center(frame, "SIZE", L['caption'], 0.5 * s, MUTED, 1, shadow=False)

            cursor_pt = tip if (keys_visible and tip) else raw_tip
            if cursor_pt:
                draw_cursor(frame, cursor_pt, MINT)

        T['3 ui + drawing'] += time.perf_counter() - t0

        if time.time() - last_save_time < 2:
            draw_toast(frame, toast_text, s)

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(frame, f"{fps:.0f} fps", (w - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        t0 = time.perf_counter()
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        T['4 show + waitKey'] += time.perf_counter() - t0
        if key == ord('q'):
            break
        elif key == ord('s'):
            do_save()
        elif key == ord('k'):
            toggle_mode()
        elif key in (ord('['), ord(']')):
            step = -2 if key == ord('[') else 2
            drawing_canvas.set_thickness(min(40, max(2, drawing_canvas.thickness + step)))
        elif key == ord('m'):
            toggle_menu()
        elif key == ord('w'):
            toggle_write()
        elif key == 27 and menu_open:
            toggle_menu()
        elif key == ord('c'):
            use_classifier = not use_classifier
            clf.history.clear()
            toast_text = "Gestures: classifier" if use_classifier else "Gestures: old rules"
            last_save_time = time.time()
        elif key in (ord('1'), ord('2'), ord('3')):
            set_size(SIZE_ORDER[key - ord('1')])
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    n = max(1, frame_count)
    print("\nAverage time per frame:")
    for k in sorted(T):
        print(f"  {k:<18} {T[k] / n * 1000:6.1f} ms")


if __name__ == "__main__":
    main()