"""
Step 10b: NLP word suggestion engine + suggestion bar helpers.

Three things in one:
  1. Prefix completion      "hel"  -> help, hello, held        (ranked by word + context probability)
  2. Next-word prediction   "i am" + <nothing typed> -> going, not, the   (bigram model)
  3. Air-writing autocorrect "he11" / "hrllo" -> hello         (fixes typos AND the characters the
                                                                 CNN tends to confuse: O/0, I/1/l, S/5 ...)

Quick test:  python suggester.py        (needs wordmodel.json from build_wordmodel.py)
"""
import json
import math
import re
from bisect import bisect_left

ALPHA = "abcdefghijklmnopqrstuvwxyz"

# characters the air-writing CNN is likely to mix up -> cheap to correct
CONF = {
    "0": "oq", "1": "il", "2": "z", "5": "s", "6": "gb", "8": "b", "9": "gq",
    "o": "0q", "i": "1lj", "l": "1i", "s": "5", "z": "2", "b": "68", "g": "96q",
    "q": "9go", "u": "v", "v": "u",
}
DIGIT_PRIMARY = {"0": "o", "1": "l", "2": "z", "5": "s", "6": "g", "8": "b", "9": "g"}


class Suggester:
    def __init__(self, path="wordmodel.json"):
        with open(path) as f:
            d = json.load(f)
        self.uni = d["uni"]
        self.bi = d["bi"]
        self.total = float(sum(self.uni.values()))
        self.words = sorted(self.uni)
        self.common = sorted(self.uni, key=self.uni.get, reverse=True)[:50]

    # ---------------------------------------------------------- helpers
    def _matches(self, prefix):
        lo = bisect_left(self.words, prefix)
        hi = bisect_left(self.words, prefix + "\uffff")
        return self.words[lo:hi]

    def _score(self, w, prev):
        """log of an interpolated unigram/bigram probability."""
        p_uni = self.uni[w] / self.total
        p = p_uni
        if prev and prev in self.bi and prev in self.uni:
            p_bi = self.bi[prev].get(w, 0) / self.uni[prev]
            p = 0.6 * p_bi + 0.4 * p_uni
        return math.log(p)

    def _variants(self, s):
        """Edit-distance-1 variants of s with a penalty. cheap=True means a confusable-character swap."""
        norm = "".join(DIGIT_PRIMARY.get(c, c) for c in s)
        if norm != s:
            yield norm, 0.3, True            # e.g. he11 -> hell (all digits mapped at once)
        n = len(s)
        for i in range(n):
            old = s[i]
            for c in ALPHA:
                if c != old:
                    cheap = c in CONF.get(old, "")
                    yield s[:i] + c + s[i + 1:], (0.7 if cheap else 2.5), cheap
            if n > 1:
                yield s[:i] + s[i + 1:], 2.5, False                      # deletion
            if i < n - 1:
                yield s[:i] + s[i + 1] + s[i] + s[i + 2:], 2.0, False    # transposition
        for i in range(n + 1):
            for c in ALPHA:
                yield s[:i] + c + s[i:], 2.5, False                      # insertion

    # ---------------------------------------------------------- main API
    def suggest(self, prefix, prev=None, k=3):
        cap = prefix[:1].isupper()
        prefix = prefix.lower()
        prev = prev.lower() if prev else None

        # nothing typed yet -> predict the next word
        if not prefix:
            out = sorted(self.bi.get(prev, {}), key=self.bi.get(prev, {}).get, reverse=True)[:k]
            for w in self.common:
                if len(out) >= k:
                    break
                if w not in out:
                    out.append(w)
            return out

        if not re.search(r"[a-z]", prefix):      # only digits: not a word
            return []

        scores = {}

        def add(w, s):
            if w not in scores or s > scores[w]:
                scores[w] = s

        for w in self._matches(prefix):
            add(w, self._score(w, prev))

        # fall back to correction when completion alone is thin, or when digits slipped in
        if len(scores) < k or re.search(r"\d", prefix):
            for v, pen, cheap in self._variants(prefix):
                if cheap:
                    for w in self._matches(v):
                        add(w, self._score(w, prev) - pen)
                elif v in self.uni:
                    add(v, self._score(v, prev) - pen)

        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        return [w.capitalize() if cap else w for w in ranked]


# ---------------------------------------------------------- text helpers (work with your text area string)
def split_context(text):
    """Return (previous_word, word_being_typed) from the text typed so far."""
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    if not tokens:
        return None, ""
    if text[-1].isalnum() or text[-1] == "'":
        return (tokens[-2] if len(tokens) > 1 else None), tokens[-1]
    if text.rstrip()[-1:] in ".!?":
        return None, ""
    return tokens[-1], ""


def apply_suggestion(text, word):
    """Replace the word being typed with the chosen suggestion and add a space."""
    m = re.search(r"[A-Za-z0-9']+$", text)
    base = text[:m.start()] if m else text
    return base + word + " "


# ---------------------------------------------------------- bar UI helpers (OpenCV)
def bar_rects(frame_w, y, h=60, k=3, margin=20, gap=12):
    bw = (frame_w - 2 * margin - gap * (k - 1)) // k
    return [(margin + i * (bw + gap), y, bw, h) for i in range(k)]


def bar_hit_test(rects, x, y):
    """Index of the suggestion button under the fingertip, or -1."""
    for i, (rx, ry, rw, rh) in enumerate(rects):
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return i
    return -1


def draw_bar(frame, rects, words, hover_idx=-1, progress=0.0):
    """Dark-glass suggestion bar. progress (0..1) is the dwell fill for the hovered button."""
    import cv2
    overlay = frame.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (40, 34, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (x, y, w, h) in enumerate(rects):
        hot = i == hover_idx
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 120) if hot else (110, 100, 90), 2)
        if hot and progress > 0:
            cv2.rectangle(frame, (x, y + h - 6), (x + int(w * min(progress, 1.0)), y + h),
                          (255, 200, 120), -1)
        if i < len(words):
            (tw, th), _ = cv2.getTextSize(words[i], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, words[i], (x + (w - tw) // 2, y + (h + th) // 2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)


if __name__ == "__main__":
    s = Suggester()
    tests = [("hel", None), ("hell0", None), ("he11", None), ("hrllo", None), ("wor", "hello"),
             ("", "i"), ("", "thank"), ("Pro", None), ("th", "in")]
    for prefix, prev in tests:
        print(f"prefix={prefix!r:8} prev={prev!r:8} -> {s.suggest(prefix, prev)}")
    print(split_context("Hello wor"), split_context("Hello world "), split_context("Done. "))
    print(repr(apply_suggestion("Hello wor", "world")))
