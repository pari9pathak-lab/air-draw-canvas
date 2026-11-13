"""
Step 8c: use the trained gesture classifier in the app (and test it live).

In your app (replacing GestureDetector.get_gesture):

    from gesture_classifier import GestureClassifier
    clf = GestureClassifier()                                   # once, at startup (loads gesture_clf.joblib)
    ...
    gesture, conf = clf.predict(landmarks, frame_w, frame_h)    # every frame; landmarks = [(id, x, y), ...]

Live test (webcam, shows the classifier and your old rules side by side):
    python src/gesture_classifier.py

IMPORTANT, mirroring: the model was trained on landmarks from a MIRRORED frame (collect_gestures.py does
cv2.flip(frame, 1) before detection). Your app must feed it landmarks from the same kind of frame.
If your app does NOT flip the frame before running MediaPipe, create the classifier with mirror_x=True.
"""
import os
import sys
import time
from collections import Counter, deque

import joblib
import numpy as np


MODEL_NAMES = ("hand_landmarker.task", "hand_landmark.task")   # accepts either spelling


def find_model():
    """Find the MediaPipe hand landmarker .task file (current folder, project root, or next to this file)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (os.getcwd(), os.path.join(here, os.pardir), here):
        for name in MODEL_NAMES:
            path = os.path.abspath(os.path.join(folder, name))
            if os.path.exists(path):
                return path
    raise FileNotFoundError("Could not find the hand landmarker .task file in the project root.")


class _FastForest:
    """
    Random Forest prediction for ONE sample using plain numpy (all trees walked at once).
    sklearn's predict_proba has a large per-call overhead that is wasted on a single webcam frame.
    Gives the same probabilities as sklearn (checked in bench_gesture.py).
    """

    def __init__(self, rf):
        trees = [e.tree_ for e in rf.estimators_]
        T = len(trees)
        N = max(t.node_count for t in trees)
        C = len(rf.classes_)
        self.left = np.zeros((T, N), dtype=np.int32)
        self.right = np.zeros((T, N), dtype=np.int32)
        self.feat = np.zeros((T, N), dtype=np.int32)
        self.thr = np.full((T, N), np.inf, dtype=np.float64)
        self.val = np.zeros((T, N, C), dtype=np.float64)
        for k, t in enumerate(trees):
            n = t.node_count
            ids = np.arange(n)
            leaf = t.children_left == -1
            self.left[k, :n] = np.where(leaf, ids, t.children_left)     # a leaf points to itself
            self.right[k, :n] = np.where(leaf, ids, t.children_right)
            self.feat[k, :n] = np.where(leaf, 0, t.feature)
            self.thr[k, :n] = np.where(leaf, np.inf, t.threshold)         # so "go left" = stay on the leaf
            v = t.value[:, 0, :]
            self.val[k, :n] = v / v.sum(axis=1, keepdims=True)
        self.depth = int(max(t.max_depth for t in trees))
        self.rows = np.arange(T)

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        node = np.zeros(len(self.rows), dtype=np.int64)
        for _ in range(self.depth):
            go_left = x[self.feat[self.rows, node]] <= self.thr[self.rows, node]
            node = np.where(go_left, self.left[self.rows, node], self.right[self.rows, node])
        return self.val[self.rows, node].mean(axis=0)


class GestureClassifier:
    def __init__(self, model_path=None, smooth=5, min_conf=0.55, mirror_x=False, every=1, fast=True):
        """
        smooth   : majority vote over the last N frames (stops flicker between gestures)
        min_conf : below this probability the answer becomes 'unknown'
        mirror_x : set True if your frames are NOT flipped before hand detection
        every    : run the model on every Nth frame only (2 = half the cost); other frames reuse the last answer
        fast     : use the numpy forest walker instead of sklearn for Random Forest models (same answers, quicker)
        """
        if model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            for folder in (os.getcwd(), os.path.join(here, os.pardir), here):
                cand = os.path.abspath(os.path.join(folder, "gesture_clf.joblib"))
                if os.path.exists(cand):
                    model_path = cand
                    break
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError("gesture_clf.joblib not found. Run python src/train_gesture.py first.")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = [str(c) for c in bundle["classes"]]
        self.min_conf = min_conf
        self.mirror_x = mirror_x
        self.history = deque(maxlen=smooth)
        self.every = max(1, int(every))
        self._fast = None
        try:
            from sklearn.ensemble import RandomForestClassifier
            if fast and isinstance(self.model, RandomForestClassifier):
                self._fast = _FastForest(self.model)
        except Exception:
            self._fast = None                      # anything unexpected: quietly use sklearn
        self._n = 0
        self._last = ("unknown", 0.0)

    @staticmethod
    def _features(pts):
        """Same recipe as training: wrist at the origin, scaled by the largest distance, x and y only."""
        pts = pts - pts[0]
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts = pts / scale
        return pts.reshape(1, -1)

    def predict(self, landmarks, frame_w, frame_h):
        """landmarks: [(id, x_pixels, y_pixels), ...]. Returns (gesture, confidence)."""
        if not landmarks or len(landmarks) < 21:
            self.history.clear()
            return "unknown", 0.0
        self._n += 1
        if self.every > 1 and self.history and self._n % self.every != 0:
            return self._last                      # skip the model on this frame
        pts = np.zeros((21, 2), dtype=np.float32)
        for i, x, y in landmarks:
            pts[i] = (x / frame_w, y / frame_h)     # back to 0..1 so it matches the training data
        if self.mirror_x:
            pts[:, 0] = 1.0 - pts[:, 0]
        feat = self._features(pts)
        proba = self._fast.predict_proba(feat) if self._fast is not None else self.model.predict_proba(feat)[0]
        k = int(np.argmax(proba))
        label, conf = self.classes[k], float(proba[k])
        if conf < self.min_conf:
            label = "unknown"
        self.history.append(label)
        self._last = (Counter(self.history).most_common(1)[0][0], conf)
        return self._last


def _demo():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    try:
        from gesture_detector import GestureDetector
        rules = GestureDetector()
    except Exception:
        rules = None

    clf = GestureClassifier()
    with open(find_model(), "rb") as f:
        model_bytes = f.read()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                           # same mirroring as collect_gestures.py
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                        int((time.time() - t0) * 1000))
        landmarks = []
        if res.hand_landmarks:
            landmarks = [(i, int(p.x * w), int(p.y * h)) for i, p in enumerate(res.hand_landmarks[0])]
            for _, x, y in landmarks:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        label, conf = clf.predict(landmarks, w, h)
        cv2.putText(frame, f"classifier: {label} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if rules is not None:
            cv2.putText(frame, f"old rules: {rules.get_gesture(landmarks) if landmarks else '-'}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 120), 2)
        cv2.imshow("gesture test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
