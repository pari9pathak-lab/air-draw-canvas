"""
Quick speed test for the gesture classifier on YOUR laptop (no webcam needed).

Run from the project root:   python src/bench_gesture.py
It prints how many milliseconds one prediction takes with sklearn and with the faster numpy path,
and checks that both give the same answers. A webcam frame at 30 FPS has about 33 ms in total.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gesture_classifier import GestureClassifier

N = 400
rng = np.random.default_rng(0)

clf = GestureClassifier(every=1, smooth=1)
model = clf.model
print("model:", type(model).__name__, "| trees:", getattr(model, "n_estimators", "n/a"),
      "| fast path:", "yes" if clf._fast is not None else "no")

feats = rng.normal(size=(N, 42)).astype(np.float32)

model.predict_proba(feats[:1])                      # warm-up
t = time.perf_counter()
ref = np.array([model.predict_proba(f[None])[0] for f in feats])
sk = (time.perf_counter() - t) / N * 1000
print(f"sklearn predict_proba:  {sk:6.2f} ms per frame")

if clf._fast is not None:
    t = time.perf_counter()
    mine = np.array([clf._fast.predict_proba(f) for f in feats])
    fa = (time.perf_counter() - t) / N * 1000
    print(f"numpy fast path:        {fa:6.2f} ms per frame  (x{sk / fa:.1f} faster)")
    print("same answers as sklearn:", bool(np.allclose(ref, mine, atol=1e-9)))

# the whole predict() call as main.py uses it, with fake pixel landmarks
landmarks = [[(i, int(x * 640), int(y * 480)) for i, (x, y) in enumerate(rng.random((21, 2)))]
             for _ in range(N)]
t = time.perf_counter()
for lm in landmarks:
    clf.predict(lm, 640, 480)
full = (time.perf_counter() - t) / N * 1000
print(f"full clf.predict():     {full:6.2f} ms per frame")
