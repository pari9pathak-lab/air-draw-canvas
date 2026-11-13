"""
Step 9a: train a small CNN on EMNIST (balanced split: digits + letters) for air-writing.

Run:   python train_airwrite.py --epochs 8
Needs: pip install torch torchvision matplotlib opencv-python numpy

Outputs
  airwrite_cnn.pt          best model weights (chosen on the validation set)
  airwrite_classes.json    class index -> character
  airwrite_confusion.png   confusion matrix on the test set (for the black book)

Also contains stroke_to_tensor(): turns a list of fingertip points into the 28x28
input the model expects. Import it later when integrating with the canvas app.
"""
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MEAN, STD = 0.1307, 0.3081


# ---------------------------------------------------------------- model
class AirCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------- preprocessing for the live app
def stroke_to_tensor(points, canvas=280, content=200, thickness=24):
    """
    points: list of (x, y) fingertip positions for ONE character (pixel coordinates).
    Returns a tensor of shape (1, 1, 28, 28) ready for AirCNN, or None if the stroke is too small.
    Mimics EMNIST style: white stroke on black, centred, with a margin around it.
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 5:
        return None
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    side = float((mx - mn).max())
    if side < 10:                      # a dot or tiny jitter, not a character
        return None
    pts = (pts - (mn + mx) / 2) * (content / side) + canvas / 2   # centre + scale, keep aspect ratio
    img = np.zeros((canvas, canvas), dtype=np.uint8)
    cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, 255,
                  thickness, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().div(255.0)
    x = (x - MEAN) / STD
    return x.view(1, 1, 28, 28)


# ---------------------------------------------------------------- data
def fix_orientation(x):
    # EMNIST images are stored transposed; swapping H and W makes them upright.
    return x.transpose(1, 2)


def build_loaders(batch_size):
    train_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(fix_orientation),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.1)),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(fix_orientation),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    full_train = datasets.EMNIST("data_emnist", split="balanced", train=True,
                                 download=True, transform=train_tf)
    full_train_eval = datasets.EMNIST("data_emnist", split="balanced", train=True,
                                      download=True, transform=eval_tf)
    test = datasets.EMNIST("data_emnist", split="balanced", train=False,
                           download=True, transform=eval_tf)

    n_val = len(full_train) // 10
    g = torch.Generator().manual_seed(42)
    idx_train, idx_val = random_split(range(len(full_train)), [len(full_train) - n_val, n_val], generator=g)
    train_set = torch.utils.data.Subset(full_train, idx_train.indices)
    val_set = torch.utils.data.Subset(full_train_eval, idx_val.indices)   # no augmentation on validation

    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=0)
    return mk(train_set, True), mk(val_set, False), mk(test, False), list(full_train.classes)


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    train_loader, val_loader, test_loader, classes = build_loaders(args.batch)
    print("classes:", len(classes), classes)
    with open("airwrite_classes.json", "w") as f:
        json.dump(classes, f)

    model = AirCNN(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        p, l = predict_all(model, val_loader, device)
        val_acc = float((p == l).mean())
        print(f"epoch {epoch}/{args.epochs}  train loss {total / n:.4f}  val acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "airwrite_cnn.pt")

    # final test with the best-on-validation weights
    model.load_state_dict(torch.load("airwrite_cnn.pt", map_location=device))
    p, l = predict_all(model, test_loader, device)
    print(f"\nTEST accuracy: {(p == l).mean():.4f}")

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, q in zip(l, p):
        cm[t, q] += 1
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print("weakest classes:")
    for i in np.argsort(per_class)[:8]:
        print(f"  {classes[i]}: {per_class[i]:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(classes, fontsize=7); ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("EMNIST balanced, test set")
        fig.tight_layout()
        fig.savefig("airwrite_confusion.png", dpi=150)
        print("saved airwrite_confusion.png")
    except ImportError:
        print("matplotlib not installed, skipped confusion matrix image")


if __name__ == "__main__":
    main()
