class GestureDetector:
    # Landmark indices for fingertips and their corresponding lower joints
    TIP_IDS = [4, 8, 12, 16, 20]       # thumb, index, middle, ring, pinky tips
    PIP_IDS = [3, 6, 10, 14, 18]       # the joint just below each tip

    def fingers_up(self, landmarks):
        """Returns a list of 5 booleans: [thumb, index, middle, ring, pinky] — True if extended."""
        if not landmarks or len(landmarks) < 21:
            return [False] * 5

        fingers = []
        lm = {id: (x, y) for id, x, y in landmarks}

        # Thumb: compare x-coordinates (thumb moves sideways, not up/down)
        fingers.append(lm[self.TIP_IDS[0]][0] > lm[self.PIP_IDS[0]][0])

        # Other four fingers: tip above (smaller y) than the joint below it = extended
        for i in range(1, 5):
            tip_y = lm[self.TIP_IDS[i]][1]
            pip_y = lm[self.PIP_IDS[i]][1]
            fingers.append(tip_y < pip_y)

        return fingers

    def count_fingers(self, landmarks):
        """Counts extended fingers EXCLUDING the thumb (index, middle, ring, pinky)."""
        return sum(self.fingers_up(landmarks)[1:])

    def get_gesture(self, landmarks):
        fingers = self.fingers_up(landmarks)
        thumb, index, middle, ring, pinky = fingers

        if index and not (middle or ring or pinky):
            return 'draw'      # index up, thumb can be anywhere
        elif all(fingers):
            return 'hover'     # open palm, all 5 up (still used for clear)
        elif not (index or middle or ring or pinky):
            return 'erase'     # fist, thumb can be anywhere
        else:
            return 'unknown'