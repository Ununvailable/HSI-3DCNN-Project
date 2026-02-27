
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Utilities for MAT datacube loading
# ============================================================

def guess_datacube_key(mat_dict, prefer=("DataCube", "datacube", "cube"), return_value=False):
    """Automatically find a 3D array in .mat file."""
    def unwrap(arr):
        v = arr
        for _ in range(3):
            if isinstance(v, np.ndarray) and v.dtype == object:
                if v.size == 1:
                    v = v.item()
                else:
                    break
            else:
                break
        return v

    # 1. name-based search
    lower = {k.lower(): k for k in mat_dict.keys()}
    for name in prefer:
        k = lower.get(name.lower())
        if k:
            v = unwrap(mat_dict[k])
            if isinstance(v, np.ndarray) and v.ndim == 3:
                return v if return_value else k

    # 2. find all 3D arrays
    candidates = []
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        v2 = unwrap(v)
        if isinstance(v2, np.ndarray) and v2.ndim == 3:
            size = int(np.prod(v2.shape))
            candidates.append((size, k, v2))
    if not candidates:
        raise KeyError("No 3D datacube found.")

    # pick largest
    candidates.sort(reverse=True)
    size, k, arr = candidates[0]
    return arr if return_value else k


def load_datacube(path, datacube_key=None):
    M = loadmat(path)
    key = datacube_key or guess_datacube_key(M)
    cube = np.array(M[key])

    if cube.ndim != 3:
        raise ValueError(f"{key} is not 3D.")

    # ensure shape = (B,H,W) = (Y,X,B)
    # MATLAB 與 NumPy 的陣列記憶體順序（row-major vs column-major）
    # 不同，# 因此有時會出現資料顯示順序怪異的情況。若希望在 MATLAB 
    # 中正確顯示影像，須事先轉成 (Y, X, B) = (13, 1632, 1232)
    raw = cube.shape
    band_axis = int(np.argmin(raw))

    if band_axis == 0:
        cube_bhw = cube
    elif band_axis == 1:
        cube_bhw = np.transpose(cube, (1, 0, 2))
    else:
        cube_bhw = np.transpose(cube, (2, 0, 1))

    return cube_bhw.astype(np.float32), key, raw, band_axis


def extract_patches(cube_bhw, patch_size=9, stride=9):
    B, H, W = cube_bhw.shape
    r = patch_size // 2
    patches = []

    for y in range(r, H - r, stride):
        for x in range(r, W - r, stride):
            patches.append(cube_bhw[:, y-r:y+r+1, x-r:x+r+1])

    if len(patches) == 0:
        raise ValueError("No patches extracted.")

    return np.array(patches, dtype=np.float32)


def build_dataset(files, label_map, patch_size, stride, datacube_key=None, normalize="minmax"):
    X_list, y_list = [], []

    for cls, path in files.items():
        cube_bhw, key, raw, band_axis = load_datacube(path, datacube_key)
        patches = extract_patches(cube_bhw, patch_size, stride)

        print(f"[{cls}] key={key}, raw={raw}, cube={cube_bhw.shape}, patches={patches.shape}")

        X_list.append(patches)
        y_list.append(np.full((len(patches),), label_map[cls], dtype=np.int32))

    X = np.concatenate(X_list, axis=0)   # (N, B, P, P)
    y = np.concatenate(y_list, axis=0)

    X = X[..., np.newaxis]   # (N, B, P, P, 1)

    if normalize == "minmax":
        mn, mx = X.min(), X.max()
        if mx > mn:
            X = (X - mn) / (mx - mn)
    elif normalize == "max":
        mx = X.max()
        if mx > 0:
            X = X / mx

    print("Final dataset:", X.shape, y.shape)
    return X, y


# ============================================================
# 2. PyTorch Dataset
# ============================================================

class HSIDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, B, P, P, 1)
        self.X = torch.from_numpy(X).permute(0, 4, 1, 2, 3).float()  # (N,1,B,P,P)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 3. Models (Auto flatten dimension)
# ============================================================

class Simple3DCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d((1, 2, 2)),
        )

        # auto flatten dimension
        dummy = torch.zeros(1, *input_shape)  # shape = (1, C, B, P, P)
        with torch.no_grad():
            flat = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class Li2017(nn.Module):
    def __init__(self, input_shape, num_classes, fc_units=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=(3,3,7), padding=(1,1,3)),
            nn.ReLU(),
            nn.Conv3d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            flat = self.conv(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


# ============================================================
# 4. EarlyStopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None
        self.stop = False

    def step(self, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "li2017"])
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datacube_key", type=str, default=None)
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "max", "none"])
    parser.add_argument("--save", type=str, default="hsi_model.pth")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    files = {
        "Red": "Red.mat",
        "Green": "Green.mat",
        "Blue": "Blue.mat",
        "Paper": "Paper.mat"
    }
    label_map = {"Red": 0, "Green": 1, "Blue": 2, "Paper": 3}
    num_classes = 4

    # Load dataset
    normalize = None if args.normalize == "none" else args.normalize
    X, y = build_dataset(files, label_map, args.patch_size, args.stride, args.datacube_key, normalize or "minmax")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    train_ds = HSIDataset(X_train, y_train)
    val_ds   = HSIDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    # obtain real input shape (1,B,P,P)
    sample_X, _ = train_ds[0]
    input_shape = sample_X.shape

    # build model
    if args.model == "simple":
        model = Simple3DCNN(input_shape, num_classes)
    else:
        model = Li2017(input_shape, num_classes, fc_units=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    early = EarlyStopping(patience=5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # training
    for epoch in range(args.epochs):
        model.train()
        tot, correct, train_loss = 0, 0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * Xb.size(0)
            correct += logits.argmax(1).eq(yb).sum().item()
            tot += yb.size(0)

        train_loss /= tot
        train_acc = correct / tot

        # validation
        model.eval()
        tot, correct, val_loss = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item() * Xb.size(0)
                correct += logits.argmax(1).eq(yb).sum().item()
                tot += yb.size(0)

        val_loss /= tot
        val_acc = correct / tot

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        early.step(val_loss, model)
        if early.stop:
            print("Early stopping triggered.")
            break

    # restore best model
    if early.best_state:
        model.load_state_dict(early.best_state)

    # plot curves
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    plt.plot(train_accs, label="train acc")
    plt.plot(val_accs, label="val acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.show()

    torch.save(model.state_dict(), args.save)
    print(f"Model saved → {args.save}")


if __name__ == "__main__":
    main()
