"""NLP word suggestion bar for the air-draw canvas.

WordModel
  - unigram frequencies from the `wordfreq` package (top 30k English words)
  - bigram counts from the Brown corpus (nltk), built once and cached in data/ngram.pkl
  - learns from what the user types (personal unigrams + bigrams, weighted higher)
  - three behaviours: complete a partly typed word, predict the next word, and fuzzy-fix
    a misspelled/misrecognised word (difflib) when no completion exists

SuggestionBar
  - three dwell buttons (hold 0.8 s) that replace the current word with the chosen one
"""
import math
import os
import pickle
import re
import difflib
from collections import Counter, defaultdict

from ui import DwellButton

PARTIAL = re.compile(r"[A-Za-z']+$")
WORD = re.compile(r"[A-Za-z']+")
SENTENCE_END = re.compile(r"[.!?]\s*$")
FALLBACK = ["the", "to", "and", "a", "of", "in", "is", "it"]


class WordModel:
    def __init__(self, cache_path=None, vocab_size=30000):
        from wordfreq import top_n_list, word_frequency
        self.words = [w for w in top_n_list('en', vocab_size)
                      if re.fullmatch(r"[a-z']+", w) and (len(w) > 1 or w in ('a', 'i'))]
        self.freq = {w: word_frequency(w, 'en') for w in self.words}
        self.big = defaultdict(Counter)           # corpus bigrams
        self.user_uni = Counter()                 # learned from the user's typing
        self.user_big = defaultdict(Counter)
        self._prefix_cache = {}
        self._load_bigrams(cache_path)

    # ---------- corpus bigrams ----------
    def _load_bigrams(self, cache_path):
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.big = pickle.load(f)
            return
        try:
            import nltk
            try:
                from nltk.corpus import brown
                tokens = brown.words()
            except LookupError:
                nltk.download('brown', quiet=True)
                from nltk.corpus import brown
                tokens = brown.words()
        except Exception as e:                     # no nltk / no internet: still works, just weaker
            print(f"[suggest] bigram corpus unavailable ({e}); next-word prediction will be basic")
            return
        toks = [t.lower() for t in tokens if re.fullmatch(r"[A-Za-z']+", t)]
        pairs = Counter(zip(toks, toks[1:]))
        for (a, b), c in pairs.items():
            if c >= 2 and b in self.freq:          # drop one-off pairs: smaller file, less noise
                self.big[a][b] = c
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(dict(self.big), f)

    # ---------- learning ----------
    def learn(self, word, prev=None):
        word = word.lower()
        if not word:
            return
        self.user_uni[word] += 1
        if prev:
            self.user_big[prev][word] += 1

    # ---------- suggestions ----------
    @staticmethod
    def context(text):
        """-> (partial word being typed, previous word or None)"""
        m = PARTIAL.search(text)
        partial = m.group(0).lower() if m else ""
        before = text[:m.start()] if m else text
        prevs = WORD.findall(before.lower())
        prev = prevs[-1] if prevs and not SENTENCE_END.search(before) else None
        return partial, prev

    def _pair(self, prev, w):
        if not prev:
            return 0
        return self.big.get(prev, {}).get(w, 0) + 5 * self.user_big.get(prev, {}).get(w, 0)

    def _score(self, w, prev):
        return (math.log(self.freq.get(w, 1e-9))
                + 2.0 * math.log1p(self._pair(prev, w))
                + 1.5 * math.log1p(self.user_uni.get(w, 0)))

    def suggest(self, text, n=3):
        partial, prev = self.context(text)
        out = []

        if partial:
            cands = self._prefix_cache.get(partial)
            if cands is None:
                cands = [w for w in self.words if w.startswith(partial)]
                if len(self._prefix_cache) > 500:
                    self._prefix_cache.clear()
                self._prefix_cache[partial] = cands
            ranked = sorted((w for w in cands if w != partial),
                            key=lambda w: -self._score(w, prev))
            out = ranked[:n]
            if len(out) < n and len(partial) >= 3:     # nothing to complete: try a spelling fix
                for w in difflib.get_close_matches(partial, self.words[:15000], n=n, cutoff=0.7):
                    if w != partial and w not in out:
                        out.append(w)
        else:                                           # start of a word: predict the next one
            nxt = Counter()
            if prev:
                for w, c in self.big.get(prev, {}).items():
                    nxt[w] += c
                for w, c in self.user_big.get(prev, {}).items():
                    nxt[w] += 5 * c
            out = [w for w, _ in nxt.most_common(n)]
            for w in FALLBACK:
                if len(out) >= n:
                    break
                if w not in out:
                    out.append(w)
        return out[:n]


class SuggestionBar:
    def __init__(self, model, dwell_time=0.8, n=3):
        self.model = model
        self.buttons = [DwellButton("", (0, 0, 0, 0), dwell_time) for _ in range(n)]
        self.words = []
        self._last_text = None

    def set_rect(self, rect):
        x1, y1, x2, y2 = rect
        n, gap = len(self.buttons), 8
        bw = (x2 - x1 - gap * (n - 1)) // n
        for i, b in enumerate(self.buttons):
            b.rect = (x1 + i * (bw + gap), y1, x1 + i * (bw + gap) + bw, y2)

    # ---------- casing ----------
    @staticmethod
    def _cased(word, partial_raw, doc):
        if word == 'i' or word.startswith("i'"):
            return 'I' + word[1:]
        if partial_raw:
            return word.capitalize() if partial_raw[0].isupper() else word
        return word.capitalize() if doc._auto_cap() else word

    # ---------- per frame ----------
    def _refresh(self, doc):
        text = doc.text
        if text == self._last_text:
            return
        old = self._last_text or ""
        # learn a word the moment it is completed by typing/writing a space or punctuation
        if len(text) == len(old) + 1 and text.startswith(old) and text[-1] in " \n.,!?":
            partial, prev = self.model.context(old)
            if partial:
                self.model.learn(partial, prev)
        self._last_text = text
        self.words = self.model.suggest(text, len(self.buttons))

    def _apply(self, word, doc):
        text = doc.text
        m = PARTIAL.search(text)
        partial_raw = m.group(0) if m else ""
        base = text[:len(text) - len(partial_raw)]
        if base and not base[-1].isspace():
            base += " "
        _, prev = self.model.context(text)
        self.model.learn(word, prev)
        doc.text = base + self._cased(word, partial_raw, doc) + " "
        self._last_text = None                     # force a refresh of the suggestions

    def update(self, point, doc):
        """Returns True when a suggestion was applied."""
        self._refresh(doc)
        partial_raw = (PARTIAL.search(doc.text) or [""])[0] if doc.text else ""
        applied = False
        for i, b in enumerate(self.buttons):
            if i < len(self.words):
                b.label = self._cased(self.words[i], partial_raw, doc)
                if not applied and b.update(point):
                    self._apply(self.words[i], doc)
                    applied = True
            else:
                b.label = ""
                b.reset()
        return applied

    def draw(self, frame, font_scale=0.55):
        for b in self.buttons:
            if b.label:
                b.draw(frame, font_scale)
