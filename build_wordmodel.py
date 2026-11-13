"""
Step 10a: build the word model (unigram + bigram counts) used by the suggestion bar.

Run:   python build_wordmodel.py
       python build_wordmodel.py --extra classroom.txt hospital.txt --extra-weight 5

Needs: pip install nltk      (the Brown corpus is downloaded automatically on first run)
Output: wordmodel.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict

MIN_COUNT = 3        # drop very rare words (also drops most corpus noise)
MAX_VOCAB = 30000
TOP_NEXT = 15        # keep this many likely next words per word


def brown_sentences():
    import nltk
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    for sent in brown.sents():
        toks = [w.lower() for w in sent if re.fullmatch(r"[A-Za-z]+", w)]
        if toks:
            yield toks


def file_sentences(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()
    for chunk in re.split(r"[.!?\n]+", text):
        toks = re.findall(r"[a-z]+", chunk)
        if toks:
            yield toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra plain-text files (domain vocabulary)")
    ap.add_argument("--extra-weight", type=int, default=5, help="how many times to count the extra files")
    args = ap.parse_args()

    uni = Counter()
    sents = list(brown_sentences())
    for path in args.extra:
        extra = list(file_sentences(path))
        sents += extra * args.extra_weight

    for toks in sents:
        uni.update(toks)

    vocab = {w for w, c in uni.most_common(MAX_VOCAB) if c >= MIN_COUNT}
    uni = {w: c for w, c in uni.items() if w in vocab}

    bi = defaultdict(Counter)
    for toks in sents:
        for a, b in zip(toks, toks[1:]):
            if a in vocab and b in vocab:
                bi[a][b] += 1
    bi = {a: dict(c.most_common(TOP_NEXT)) for a, c in bi.items()}

    with open("wordmodel.json", "w") as f:
        json.dump({"uni": uni, "bi": bi}, f)
    print(f"saved wordmodel.json: {len(uni)} words, {len(bi)} words with next-word stats")


if __name__ == "__main__":
    main()
