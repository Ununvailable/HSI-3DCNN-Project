import os
import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -------------------------
# 1) Utils: find cube key
# -------------------------
def guess_datacube_key(mat_dict):
    """
    Try to find the most likely 3D datacube key inside a .mat dict.
    Preference order:
      1) "DataCube" if exists
      2) any variable that is a 3D ndarray and not meta keys (__header__/__version__/__globals__)
    """
    if "DataCube" in mat_dict:
        return "DataCube"

    candidates = []
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            candidates.append((k, v.shape))

    if not candidates:
        raise KeyError(
            f"找不到 3D DataCube。可用 keys={list(mat_dict.keys())}"
        )

    # heuristic: choose the one with largest total size
    candidates.sort(key=lambda x: np.prod(x[1]), reverse=True)
    return candidates[0][0]


def load_datacube(mat_path, datacube_key=None):
    """
    Load .mat and return cube in (B, H, W) float32.
    Band axis is guessed as the smallest dimension.
    """
    M = loadmat(mat_path)
    key = datacube_key or guess_datacube_key(M)

    cube = np.array(M[key])
    if cube.ndim != 3:
        raise ValueError(f"{mat_path} 的 {key} 不是 3D，shape={cube.shape}")

    raw_shape = cube.shape
    band_axis = int(np.argmin(raw_shape))

    # to (B,H,W)
    if band_axis == 0:
        cube_bhw = cube
    elif band_axis == 1:
        cube_bhw = np.transpose(cube, (1, 0, 2))
    else:
        cube_bhw = np.transpose(cube, (2, 0, 1))

    return cube_bhw.astype(np.float32), key, raw_shape, band_axis


def extract_patches(cube_bhw, patch_size=9, stride=9):
    """
    cube_bhw: (B,H,W)
    return: (N,B,P,P)
    """
    B, H, W = cube_bhw.shape
    r = patch_size // 2
    patches = []

    for y in range(r, H - r, stride):
        for x in range(r, W - r, stride):
            patch = cube_bhw[:, y - r:y + r + 1, x - r:x + r + 1]
            patches.append(patch)

    if len(patches) == 0:
        raise ValueError(
            f"patch=0：可能 H/W 太小或 patch_size/stride 太大。cube={cube_bhw.shape}, patch={patch_size}, stride={stride}"
        )

    return np.array(patches, dtype=np.float32)


def build_dataset(files_dict, label_map, patch_size, stride, datacube_key=None, normalize="minmax"):
    """
    Build dataset from multiple .mat files.
    Output X: (N, B, P, P, 1), y: (N,)
    """
    X_list, y_list = [], []

    for cls_name, path in files_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到檔案：{path}")

        cube_bhw, key_used, raw_shape, band_axis = load_datacube(path, datacube_key=datacube_key)
        patches = extract_patches(cube_bhw, patch_size=patch_size, stride=stride)

        print(f"[{cls_name}] key='{key_used}' raw={raw_shape}, band_axis={band_axis} -> cube(B,H,W)={cube_bhw.shape}, patches={patches.shape}")

        X_list.append(patches)
        y_list.append(np.full((len(patches),), label_map[cls_name], dtype=np.int32))

    X = np.concatenate(X_list, axis=0)          # (N,B,P,P)
    y = np.concatenate(y_list, axis=0)          # (N,)
    X = X[..., np.newaxis]                      # (N,B,P,P,1)

    if normalize == "minmax":
        X_min, X_max = X.min(), X.max()
        if X_max > X_min:
            X = (X - X_min) / (X_max - X_min)
    elif normalize == "max":
        mx = X.max()
        if mx != 0:
            X = X / mx

    print("Final X:", X.shape, "y:", y.shape, "classes:", np.unique(y))
    return X, y


# -------------------------
# 2) Models
# -------------------------
def build_model_simple(input_shape, num_classes):
    """
    A practical 3D CNN baseline (similar to your existing train_hsi_3dcnn.py).
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((1, 2, 2)),  # pool only spatial

        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((1, 2, 2)),

        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((1, 2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.6),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_model_li2017(input_shape, num_classes, fc_units=128):
    """
    Li (2017) style (as you pasted):
      C1: 2 filters, 3×3×7
      C2: 4 filters, 3×3×3
      Flatten -> FC -> Softmax
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv3D(filters=2, kernel_size=(3, 3, 7), activation="relu", padding="same")(inp)
    x = layers.Conv3D(filters=4, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(fc_units, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inp, outputs=out)


# -------------------------
# 3) Train
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "li2017"])
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datacube_key", type=str, default=None, help="If you know the variable names of the cubes within the MATLAB, you can specify them; otherwise, they will be automatically detected.")  # "如果你知道 mat 內 cube 的變數名，可指定；不指定則自動偵測"
    parser.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "max", "none"])
    parser.add_argument("--save", type=str, default="hsi_model.keras")
    args = parser.parse_args()

    print("TensorFlow:", tf.__version__)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---- Your 4-class files ----
    files = {
        "Red":   "hsi_datasets/v303/Red.mat",
        "Green": "hsi_datasets/v303/Green.mat",
        "Blue":  "hsi_datasets/v303/Blue.mat",
        "Paper": "hsi_datasets/v303/Paper.mat"
    }
    label_map = {"Red": 0, "Green": 1, "Blue": 2, "Paper": 3}
    num_classes = 4

    # ---- Build dataset ----
    normalize = None if args.normalize == "none" else args.normalize
    X, y = build_dataset(
        files_dict=files,
        label_map=label_map,
        patch_size=args.patch_size,
        stride=args.stride,
        datacube_key=args.datacube_key,
        normalize=normalize or "minmax"
    )

    # ---- Split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed
    )

    # ---- Model ----
    input_shape = X_train.shape[1:]  # (B,P,P,1)
    if args.model == "simple":
        model = build_model_simple(input_shape, num_classes)
    else:
        model = build_model_li2017(input_shape, num_classes, fc_units=128)

    model.summary()

    # ---- Compile ----
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # ---- Train ----
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # ---- Evaluate + Save ----
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {acc:.4f}, loss: {loss:.4f}")

    model.save(f"training_results/{args.save}")
    print(f"Saved model -> {args.save}")


if __name__ == "__main__":
    main()
