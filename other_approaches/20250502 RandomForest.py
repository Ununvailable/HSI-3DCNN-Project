# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:53:00 2025

@author: stust
"""

# === 匯入必要的函式庫 ===
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import time  # 用來計算時間

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

# === 建立 one-hot 類別標籤（Red=0, Green=1, Blue=2, Paper=3） ===
y_r = np.zeros((X_r.shape[0], 4)); y_r[:, 0] = 1
y_g = np.zeros((X_g.shape[0], 4)); y_g[:, 1] = 1
y_b = np.zeros((X_b.shape[0], 4)); y_b[:, 2] = 1
y_p = np.zeros((X_p.shape[0], 4)); y_p[:, 3] = 1

# === 合併資料與標籤 ===
X_train = np.vstack([X_r, X_g, X_b, X_p])
y_train = np.vstack([y_r, y_g, y_b, y_p])

# === 特徵標準化 ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# === 建立並訓練模型（計時） ===
start_train = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, np.argmax(y_train, axis=1))
end_train = time.time()
print(f"✅ 訓練時間：{end_train - start_train:.2f} 秒")

# === 載入欲轉換為 RGB 的高光譜影像 ===
input_cube = scipy.io.loadmat(os.path.join(dataset_path, "Spectrum-Simplified.mat"))["DataCube"]
H_test, W_test, B = input_cube.shape
X_input = input_cube.reshape(-1, B).astype(np.float32)

# === 使用訓練時的標準化方式 ===
X_input = scaler.transform(X_input)

# === 預測分類與上色（計時） ===
start_pred = time.time()
y_proba = rf_model.predict_proba(X_input)  # 每個樣本的每類別機率
max_proba = np.max(y_proba, axis=1)
y_pred = np.argmax(y_proba, axis=1)

# === 機率門檻設定：低於 85% 的點歸類為 Paper 類別（index=3） ===
y_pred[max_proba < 0.61] = 3
end_pred = time.time()
print(f"🎨 上色時間：{end_pred - start_pred:.2f} 秒")

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
rgb_array = np.array([class_colors[c] for c in y_pred])
rgb_img = rgb_array.reshape(H_test, W_test, 3)

# === 顯示結果影像 ===
plt.imshow(rgb_img)
plt.title("RGB_RandomForest")
plt.axis("off")
plt.show()
