# -*- coding: utf-8 -*-
"""
Conv1D + LSTM — PyTorch port of 20250502_Conv1D_LSTM.py
Maintains original script structure and behaviour.
"""

# === 匯入必要的函式庫 ===
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import time

# === 設定資料路徑 ===
# base_path = r'C:\Users\stust\Desktop\test'
dataset_path = r'hsi_datasets/v303'

# === 載入四種顏色的高光譜影像 ===
cube_r = scipy.io.loadmat(os.path.join(dataset_path, "Red.mat"))["DataCube"]
cube_g = scipy.io.loadmat(os.path.join(dataset_path, "Green.mat"))["DataCube"]
cube_b = scipy.io.loadmat(os.path.join(dataset_path, "Blue.mat"))["DataCube"]
cube_p = scipy.io.loadmat(os.path.join(dataset_path, "Paper.mat"))["DataCube"]

# === 展平成 2D 特徵 ===
H, W, B = cube_r.shape
X_r = cube_r.reshape(-1, B)
X_g = cube_g.reshape(-1, B)
X_b = cube_b.reshape(-1, B)
X_p = cube_p.reshape(-1, B)

# === 建立整數類別標籤（Red=0, Green=1, Blue=2, Paper=3） ===
y_r = np.zeros(X_r.shape[0], dtype=np.int64)
y_g = np.ones(X_g.shape[0],  dtype=np.int64)
y_b = np.full(X_b.shape[0],  2, dtype=np.int64)
y_p = np.full(X_p.shape[0],  3, dtype=np.int64)

# === 合併資料與標籤 ===
X_train = np.vstack([X_r, X_g, X_b, X_p])
y_train = np.concatenate([y_r, y_g, y_b, y_p])

# === 特徵標準化 ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# === 將資料轉換為 (N, 1, B) — 保持與原始相同的 shape ===
X_train_3d = X_train.reshape(-1, 1, B).astype(np.float32)

# === 建立 PyTorch Dataset，切分 90/10 訓練/驗證 ===
dataset = TensorDataset(
    torch.from_numpy(X_train_3d),
    torch.from_numpy(y_train)
)
val_size   = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=2048, shuffle=False)

# === 建立 Conv1D + LSTM 模型 ===
class Conv1DLSTM(nn.Module):
    def __init__(self, input_bands, num_classes=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=64,
                              kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Conv1d output: (N, 64, B) — LSTM expects (N, seq, features)
        # permute to (N, B, 64) before LSTM
        self.lstm  = nn.LSTM(input_size=64, hidden_size=128,
                             batch_first=True)
        self.fc1   = nn.Linear(128, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (N, 1, B)
        x = self.relu(self.conv(x))      # (N, 64, B)
        x = x.permute(0, 2, 1)          # (N, B, 64)
        _, (h_n, _) = self.lstm(x)      # h_n: (1, N, 128)
        x = h_n.squeeze(0)              # (N, 128)
        x = self.relu(self.fc1(x))      # (N, 128)
        x = self.fc2(x)                 # (N, 4)  — raw logits
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Conv1DLSTM(input_bands=B, num_classes=4).to(device)

# === 編譯設定 ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 訓練模型（計時） ===
epochs = 10
start_train = time.time()

for epoch in range(epochs):
    # ── 訓練 ──
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * Xb.size(0)
        train_correct += logits.argmax(dim=1).eq(yb).sum().item()
        train_total   += Xb.size(0)

    # ── 驗證 ──
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            val_loss    += loss.item() * Xb.size(0)
            val_correct += logits.argmax(dim=1).eq(yb).sum().item()
            val_total   += Xb.size(0)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss={train_loss/train_total:.4f}, Acc={train_correct/train_total:.4f} | "
        f"Val Loss={val_loss/val_total:.4f}, Acc={val_correct/val_total:.4f}"
    )

end_train = time.time()
print(f"訓練時間：{end_train - start_train:.2f} 秒")

# === 載入欲轉換為 RGB 的高光譜影像 ===
input_cube = scipy.io.loadmat(os.path.join(dataset_path, "Spectrum-Simplified.mat"))["DataCube"]
H_test, W_test, B = input_cube.shape
X_input = input_cube.reshape(-1, B).astype(np.float32)

# === 標準化並轉換為 (N, 1, B) ===
X_input    = scaler.transform(X_input)
X_input_3d = torch.from_numpy(X_input.reshape(-1, 1, B)).float()

# === 預測分類結果（計時） ===
start_pred = time.time()
model.eval()
all_probs = []
with torch.no_grad():
    loader_pred = DataLoader(TensorDataset(X_input_3d), batch_size=1024)
    for (Xb,) in loader_pred:
        Xb    = Xb.to(device)
        probs = torch.softmax(model(Xb), dim=1)
        all_probs.append(probs.cpu().numpy())

y_pred    = np.concatenate(all_probs, axis=0)   # (N, 4)
max_proba = np.max(y_pred,  axis=1)
y_class   = np.argmax(y_pred, axis=1)

# === 將最大機率 < 0.88 的分類轉為白色類別（索引 3） ===
y_class[max_proba < 0.88] = 3
end_pred = time.time()
print(f"上色時間：{end_pred - start_pred:.2f} 秒")

# === 類別對應 RGB 顏色 ===
# class_colors = {
#     0: [1.0, 0.0, 0.0],  # Red
#     1: [0.0, 1.0, 0.0],  # Green
#     2: [0.0, 0.0, 1.0],  # Blue
#     3: [1.0, 1.0, 1.0]   # Paper (White)
# }
class_colors = {
    0: [0.84, 0.15, 0.16],  # Red
    1: [0.17, 0.63, 0.17],  # Green
    2: [0.12, 0.47, 0.7],  # Blue
    3: [0.8, 0.8, 0.8]   # Paper (White)
}
rgb_array = np.array([class_colors[c] for c in y_class])
rgb_img   = rgb_array.reshape(H_test, W_test, 3)

# === 顯示結果影像 ===
plt.imshow(rgb_img)
plt.title("RGB_Conv1D+LSTM_PyTorch")
plt.axis("off")
plt.show()