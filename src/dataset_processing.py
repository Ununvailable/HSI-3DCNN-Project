import os
import argparse
import numpy as np
from scipy.io import loadmat

import tensorflow as tf
from tensorflow import keras

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

def main():
    files = {
        "Red":   "hsi_datasets/v303/Red.mat",
        "Green": "hsi_datasets/v303/Green.mat",
        "Blue":  "hsi_datasets/v303/Blue.mat",
        "Paper": "hsi_datasets/v303/Paper.mat",
        "Spectrum_1": "hsi_datasets/v303/Spectrum-1.mat",
        "Spectrum_3": "hsi_datasets/v303/Spectrum-3.mat",
        "Spectrum_Simplified": "hsi_datasets/v303/Spectrum-Simplified.mat",
        "Indian_pines_corrected": "hsi_datasets/hsi_researches/Indian_pines_corrected.mat",
    }

    # ✅ 這三行要在這裡（4格縮排）
    model = keras.models.load_model("training_results/hsi_model_str2_batch64_epoch50_w-callback.keras")
    print("model.input_shape =", model.input_shape)
    print("model.output_shape =", model.output_shape)

    for name, file in files.items():
        if not os.path.exists(file):
            print(f"[{name}] MISSING: {file}")
            continue

        cube_bhw, key, raw_shape, band_axis = load_datacube(file)

        print(f"[{name}] OK")
        print(f"  path      : {file}")
        print(f"  key       : {key}")
        print(f"  raw_shape : {raw_shape}")
        print(f"  band_axis : {band_axis}")
        print(f"  BHW shape : {cube_bhw.shape}\n")


if __name__ == "__main__":
    main()