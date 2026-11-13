import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Standard 21-point hand skeleton connections (since mp.solutions no longer exists)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                 # palm base
]

import os

class HandTracker:
    def __init__(self, model_path=None, max_hands=1, min_detection_confidence=0.7, detect_width=320):
        if model_path is None:
            # Go up one level from src/ to the project root, then find the model file
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, 'hand_landmarker.task')

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.latest_result = None
        self.frame_timestamp_ms = 0
        self.detect_width = detect_width   # detection runs on a small copy; landmarks are normalised

    def find_hands(self, frame, draw=True):
        h0, w0 = frame.shape[:2]
        small = frame
        if w0 > self.detect_width:
            small = cv2.resize(frame, (self.detect_width, int(h0 * self.detect_width / w0)),
                               interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts = int(time.monotonic() * 1000)             # real time, so tracking knows the true frame gap
        self.frame_timestamp_ms = max(ts, self.frame_timestamp_ms + 1)
        self.latest_result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        if self.latest_result.hand_landmarks and draw:
            h, w, _ = frame.shape
            for hand_landmarks in self.latest_result.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for connection in HAND_CONNECTIONS:
                    start, end = connection
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for point in points:
                    cv2.circle(frame, point, 4, (0, 0, 255), -1)

        return frame

    def get_landmark_positions(self, frame, hand_index=0):
        landmark_list = []
        if self.latest_result and self.latest_result.hand_landmarks:
            if hand_index < len(self.latest_result.hand_landmarks):
                h, w, _ = frame.shape
                for id, lm in enumerate(self.latest_result.hand_landmarks[hand_index]):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((id, cx, cy))
        return landmark_list