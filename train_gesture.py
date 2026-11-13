"""
Step 8b: train the gesture classifier from data/gestures.csv and compare it with your rule-based detector.
 
Run from the project root:   python src/train_gesture.py
Needs: pip install scikit-learn joblib matplotlib numpy
 
What it does
  1. Loads the landmark rows recorded by collect_gestures.py (uses x and y only; z is dropped).
  2. Renames old labels to the names your app uses: none -> unknown, and hover -> clear
     (the open palm is now the dedicated "clear" gesture).
  3. Splits BLOCK-WISE: consecutive frames are near-duplicates, so a random split would give fake-high accuracy.
  4. Trains a Random Forest and a small neural network (MLP), and scores your existing GestureDetector
     on the same test rows.
  5. Saves the better model to gesture_clf.joblib and a confusion matrix image for the black book.
"""
import argparse
import csv
import os
import sys
from collections import Counter
 
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
CSV_PATH = "data/gestures.csv"
OUT_MODEL = "gesture_clf.joblib"
BLOCK = 50          # rows per block (about 3 seconds of recording)
 
 
def load(path):
    rename = {"none": "unknown", "hover": "clear"}
    labels, feats = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 64:
                continue
            label = rename.get(row[0], row[0])
            labels.append(label)
            feats.append([float(v) for v in row[1:64]])
    X = np.array(feats, dtype=np.float32).reshape(-1, 21, 3)[:, :, :2].reshape(-1, 42)
    return np.array(labels), X
 
 
def block_split(y):
    """Every 4th block of each class goes to the test set (the last block if a class has fewer than 4)."""
    test = np.zeros(len(y), dtype=bool)
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        blocks = np.arange(len(idx)) // BLOCK
        nb = int(blocks.max()) + 1
        test_blocks = [b for b in range(nb) if b % 4 == 3] or [nb - 1]
        test[idx[np.isin(blocks, test_blocks)]] = True
    return ~test, test
 
 
def rule_predictions(X):
    """Run YOUR existing GestureDetector.get_gesture on the same landmarks (None if it can't be imported)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gesture_detector import GestureDetector
        det = GestureDetector()
        out = []
        for row in X:
            pts = row.reshape(21, 2)
            g = det.get_gesture([(i, float(x), float(y)) for i, (x, y) in enumerate(pts)])
            out.append("clear" if g == "hover" else g)      # the old rules call the open palm "hover"
        return np.array(out)
    except Exception as e:
        print("could not run the rule-based detector for comparison:", e)
        return None
 
 
def report(name, y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name}: test accuracy {acc:.4f}")
    for c in classes:
        m = y_true == c
        if m.any():
            print(f"   {c:8s} recall {np.mean(y_pred[m] == c):.3f}  ({m.sum()} test rows)")
    return acc
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--trees", type=int, default=200,
                    help="Random Forest size; fewer trees = faster in the live app (try 60)")
    ap.add_argument("--save", choices=["auto", "rf", "mlp"], default="auto",
                    help="which model to save (auto = best balanced accuracy)")
    args = ap.parse_args()
 
    y, X = load(args.csv)
    counts = Counter(y.tolist())
    print("samples per class:", dict(counts))
    for c, n in counts.items():
        if n < 150:
            print(f"  warning: only {n} samples for '{c}', record more for a reliable result")
    if len(counts) < 3:
        sys.exit("need at least 3 classes in the CSV")
 
    train, test = block_split(y)
    classes = sorted(counts)
    print(f"train rows {train.sum()}, test rows {test.sum()}")
 
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=args.trees, class_weight="balanced_subsample", random_state=0),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=0)),
    }
    results = {}
    for name, model in models.items():
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        acc = report(name, y[test], pred, classes)
        bal = balanced_accuracy_score(y[test], pred)
        print(f"   balanced accuracy {bal:.4f}  (every gesture counts equally, so a big class cannot hide a weak one)")
        if "clear" in classes:
            others = y[test] != "clear"
            print(f"   false-clear rate {np.mean(pred[others] == 'clear'):.3f}  "
                  f"(other poses read as clear, the mistake that can wipe the canvas)")
        print("   confusion matrix (rows = true, columns = predicted):", classes)
        print("  ", str(confusion_matrix(y[test], pred, labels=classes)).replace("\n", "\n   "))
        results[name] = (bal, model, pred, acc)
 
    rules = rule_predictions(X[test])
    if rules is not None:
        report("Existing rule-based detector", y[test], rules, classes)
 
    if args.save == "auto":
        best_name = max(results, key=lambda k: results[k][0])
    else:
        best_name = {"rf": "Random Forest", "mlp": "MLP"}[args.save]
    best_bal, best_model, best_pred, best_acc = results[best_name]
    joblib.dump({"model": best_model, "classes": list(best_model.classes_), "name": best_name}, OUT_MODEL)
    print(f"\nsaved {best_name} (accuracy {best_acc:.4f}, balanced accuracy {best_bal:.4f}) to {OUT_MODEL}")
 
    cm = confusion_matrix(y[test], best_pred, labels=classes)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Gesture classifier ({best_name})")
        fig.tight_layout()
        fig.savefig("gesture_confusion.png", dpi=150)
        print("saved gesture_confusion.png")
    except ImportError:
        pass
 
 
if __name__ == "__main__":
    main()
 