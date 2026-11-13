from collections import deque
import math
import time

class PointSmoother:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.x_points = deque(maxlen=buffer_size)
        self.y_points = deque(maxlen=buffer_size)

    def smooth(self, point):
        x, y = point
        self.x_points.append(x)
        self.y_points.append(y)
        avg_x = int(sum(self.x_points) / len(self.x_points))
        avg_y = int(sum(self.y_points) / len(self.y_points))
        return (avg_x, avg_y)

    def reset(self):
        self.x_points.clear()
        self.y_points.clear()

class HoldTimer:
    def __init__(self, hold_duration=2.0):
        self.hold_duration = hold_duration
        self.start_time = None

    def update(self, condition_met):
        """Call every frame with True/False. Returns True once held long enough."""
        if condition_met:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            return elapsed >= self.hold_duration
        else:
            self.start_time = None
            return False

    def progress(self):
        """Returns 0.0–1.0 progress toward the hold, for an on-screen indicator."""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(elapsed / self.hold_duration, 1.0)

class StableValue:
    """Returns a value only after it has stayed the same for hold_duration seconds."""
    def __init__(self, hold_duration=0.8):
        self.hold_duration = hold_duration
        self.candidate = None
        self.start_time = None

    def update(self, value):
        now = time.time()
        if value != self.candidate:
            self.candidate = value
            self.start_time = now if value is not None else None
            return None
        if value is not None and now - self.start_time >= self.hold_duration:
            return value
        return None

    def progress(self):
        if self.candidate is None or self.start_time is None:
            return 0.0
        return min((time.time() - self.start_time) / self.hold_duration, 1.0)


class OneEuroSmoother:
    """Drop-in replacement for PointSmoother (same smooth()/reset() calls).
    Steady and jitter-free when the finger is slow, almost no lag when it moves fast.
      min_cutoff: lower = smoother when still   beta: higher = less lag when moving fast"""

    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.t = None
        self.x = self.y = None
        self.dx = self.dy = 0.0

    def smooth(self, point):
        now = time.time()
        x, y = point
        if self.t is None:
            self.t, self.x, self.y = now, float(x), float(y)
            return (int(x), int(y))
        dt = max(now - self.t, 1e-3)
        self.t = now
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx += a_d * ((x - self.x) / dt - self.dx)
        self.dy += a_d * ((y - self.y) / dt - self.dy)
        speed = math.hypot(self.dx, self.dy)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        return (int(round(self.x)), int(round(self.y)))
