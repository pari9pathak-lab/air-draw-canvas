"""
Step 8a: collect a labelled hand-landmark dataset for the gesture classifier.

Controls
  1-5   select the gesture class to record
  SPACE start / stop recording samples for the selected class
  q     quit

Output: data/gestures.csv  ->  label, f0 ... f62  (21 landmarks x (x, y, z), normalised)
"""
import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_NAMES = ("hand_landmarker.task", "hand_landmark.task")   # accepts either spelling


def find_model():
    """Look for the model in the current folder, the project root (parent of src/) and next to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (os.getcwd(), os.path.join(here, os.pardir), here):
        for name in MODEL_NAMES:
            path = os.path.abspath(os.path.join(folder, name))
            if os.path.exists(path):
                return path
    raise FileNotFoundError("Could not find the hand landmarker .task file. Put it in the project root "
                            "(next to src/) or set the name in MODEL_NAMES.")
OUT_FILE = "data/gestures.csv"
SAMPLES_PER_SECOND = 15               # cap so consecutive frames are not near-duplicates

# Edit this to match the gestures your app uses.
CLASSES = {
    ord("1"): "draw",      # index finger only (thumb anywhere)
    ord("2"): "clear",     # full open palm, all five fingers up and spread (hold to clear the canvas)
    ord("3"): "erase",     # fist (thumb anywhere, thumbs-up counts as erase)
    ord("4"): "unknown",   # every other pose: 2/3/4 fingers, pinky only, pinch, hand turning, relaxed hand
}


def normalise(landmarks):
    """Make features independent of where the hand is and how big it appears."""
    pts = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    pts -= pts[0]                                       # wrist becomes the origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))  # hand size in the image plane
    if scale > 0:
        pts /= scale
    return pts.flatten()


# labels used in earlier recordings -> current names (so the counters include your old data)
LEGACY = {"hover": "clear", "none": "unknown"}


def load_counts():
    counts = {name: 0 for name in CLASSES.values()}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, newline="") as f:
            for row in csv.reader(f):
                if row:
                    name = LEGACY.get(row[0], row[0])
                    if name in counts:
                        counts[name] += 1
    return counts


def main():
    os.makedirs("data", exist_ok=True)
    counts = load_counts()

    model_path = find_model()
    print("using model:", model_path)
    with open(model_path, "rb") as f:          # loading from bytes avoids Windows path problems
        model_bytes = f.read()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    current = list(CLASSES.values())[0]
    recording = False
    last_saved = 0.0
    t0 = time.time()

    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, int((time.time() - t0) * 1000))

            h, w = frame.shape[:2]
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                for p in lm:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
                now = time.time()
                if recording and now - last_saved > 1.0 / SAMPLES_PER_SECOND:
                    writer.writerow([current] + normalise(lm).tolist())
                    counts[current] += 1
                    last_saved = now

            status = "REC" if recording else "paused"
            cv2.putText(frame, f"class: {current} [{status}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (255, 255, 255), 2)
            for i, (key, name) in enumerate(CLASSES.items()):
                cv2.putText(frame, f"{chr(key)}: {name} = {counts[name]}", (10, 60 + 25 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("collect gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
            if key in CLASSES:
                current = CLASSES[key]
                recording = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()