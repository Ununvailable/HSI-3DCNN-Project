# -*- coding: utf-8 -*-
"""
HSI-Inference_Testing.py
=========================
Supporting script for running classification inference on HSI .mat datacubes
using a trained 3D-CNN model.
Can be imported as classes or invoked directly from the CLI.

Usage (CLI)
-----------
    # Single file
    python HSI-Inference_Testing.py \\
        --model training_results/hsi_model.keras \\
        --input_files hsi_datasets/v303/Spectrum-1.mat \\
        --class_names Red Green Blue Paper

    # Batch (entire directory)
    python HSI-Inference_Testing.py \\
        --model training_results/hsi_model.keras \\
        --input_dir hsi_datasets/v303 \\
        --class_names Red Green Blue Paper \\
        --stride 4

Usage (import)
--------------
    from HSI-Inference_Testing import InferenceEngine, ResultVisualizer

    engine = InferenceEngine(
        model_path="training_results/hsi_model.keras",
        patch_size=9, stride=4, normalize="minmax", batch_size=64,
    )
    engine.load_model()

    class_map, confidence_map, predictions = engine.predict(
        mat_path="hsi_datasets/v303/Spectrum-1.mat"
    )

    viz = ResultVisualizer(class_names=["Red", "Green", "Blue", "Paper"])
    viz.save_all(class_map, confidence_map, predictions,
                 dataset_name="Spectrum-1", output_dir="inference_result/Spectrum-1")
"""

import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

# TensorFlow imported lazily inside InferenceEngine.load_model()
# so the module can be imported for ResultVisualizer alone without
# requiring a GPU / TF environment.


# ─────────────────────────────────────────────
# 1. InferenceEngine
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Loads a trained Keras model and runs patch-based classification
    over entire HSI datacubes.

    Parameters
    ----------
    model_path : str   — Path to a saved .keras model file
    patch_size : int   — Spatial patch size (must match training)
    stride     : int   — Extraction stride (smaller = denser map)
    normalize  : str   — 'minmax' | 'max' | 'none'
    batch_size : int   — Prediction batch size

    Usage
    -----
        engine = InferenceEngine("model.keras")
        engine.load_model()
        class_map, conf_map, preds = engine.predict("Spectrum-1.mat")
    """

    def __init__(
        self,
        model_path: str,
        patch_size: int = 9,
        stride: int = 4,
        normalize: str = "minmax",
        batch_size: int = 64,
    ):
        self.model_path = model_path
        self.patch_size = patch_size
        self.stride     = stride
        self.normalize  = normalize
        self.batch_size = batch_size
        self._model     = None
        self._expected_bands: int | None = None

    # ── public ──────────────────────────────

    def load_model(self) -> "InferenceEngine":
        """Load the Keras model from disk. Returns self for chaining."""
        from tensorflow import keras  # lazy import
        self._model = keras.models.load_model(self.model_path)
        self._expected_bands = self._model.input_shape[1]
        print(f"Model loaded: {self.model_path}")
        print(f"  Expected bands: {self._expected_bands}")
        return self

    def predict(self, mat_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full-image classification on a single .mat datacube.

        Parameters
        ----------
        mat_path : Path to the .mat file

        Returns
        -------
        class_map      : (H, W) int32  — class index per pixel (-1 = unclassified)
        confidence_map : (H, W) float32 — max softmax probability per pixel
        predictions    : (N, C) float32 — raw softmax output for all patches
        """
        if self._model is None:
            raise RuntimeError("Call load_model() before predict().")

        cube_bhw = self._load_datacube(mat_path)
        B, H, W  = cube_bhw.shape

        # Band compatibility check
        if B != self._expected_bands:
            if B > self._expected_bands:
                print(
                    f"  ⚠  Band mismatch: truncating {B} → {self._expected_bands} bands"
                )
                cube_bhw = cube_bhw[: self._expected_bands]
            else:
                raise ValueError(
                    f"Data has {B} bands but model expects {self._expected_bands}."
                )

        print(f"  Extracting patches (size={self.patch_size}, stride={self.stride})...")
        patches, coords = self._extract_patches(cube_bhw)
        print(f"    → {len(patches)} patches")

        patches = self._normalize_patches(patches)
        patches = patches[..., np.newaxis]  # add channel dim → (N, B, P, P, 1)

        print(f"  Running predictions (batch_size={self.batch_size})...")
        predictions = self._model.predict(patches, batch_size=self.batch_size, verbose=0)
        class_ids   = np.argmax(predictions, axis=1)
        confidences = np.max(predictions,   axis=1)

        class_map      = np.full((H, W), -1,  dtype=np.int32)
        confidence_map = np.zeros((H, W),     dtype=np.float32)

        for (y, x), cls, conf in zip(coords, class_ids, confidences):
            class_map[y, x]      = cls
            confidence_map[y, x] = conf

        return class_map, confidence_map, predictions

    # ── private helpers ──────────────────────

    def _load_datacube(self, mat_path: str) -> np.ndarray:
        """Return cube in (B, H, W) float32."""
        M   = loadmat(mat_path)
        key = self._guess_key(M)
        cube = np.array(M[key])

        if cube.ndim != 3:
            raise ValueError(f"'{key}' in {mat_path} is not 3D — shape={cube.shape}")

        # band_axis = int(np.argmin(cube.shape))
        # if band_axis == 0:
        #     return cube.astype(np.float32)
        # elif band_axis == 1:
        #     return np.transpose(cube, (1, 0, 2)).astype(np.float32)
        # else:
        #     return np.transpose(cube, (2, 0, 1)).astype(np.float32)

        # Saved convention is (Y, X, B) → reorder to (B, Y, X)
        return np.transpose(cube, (2, 0, 1)).astype(np.float32)

    def _extract_patches(self, cube_bhw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        B, H, W = cube_bhw.shape
        r = self.patch_size // 2
        patches, coords = [], []

        for y in range(r, H - r, self.stride):
            for x in range(r, W - r, self.stride):
                patches.append(cube_bhw[:, y - r:y + r + 1, x - r:x + r + 1])
                coords.append([y, x])

        if not patches:
            raise ValueError(
                f"No patches extracted. Cube={cube_bhw.shape}, "
                f"patch_size={self.patch_size}, stride={self.stride}."
            )

        return np.array(patches, dtype=np.float32), np.array(coords)

    def _normalize_patches(self, patches: np.ndarray) -> np.ndarray:
        if self.normalize == "minmax":
            lo, hi = patches.min(), patches.max()
            if hi > lo:
                return (patches - lo) / (hi - lo)
        elif self.normalize == "max":
            mx = patches.max()
            if mx != 0:
                return patches / mx
        return patches

    @staticmethod
    def _guess_key(mat_dict: dict) -> str:
        if "DataCube" in mat_dict:
            return "DataCube"
        candidates = [
            (k, v.shape)
            for k, v in mat_dict.items()
            if not k.startswith("__")
            and isinstance(v, np.ndarray)
            and v.ndim == 3
        ]
        if not candidates:
            raise KeyError(f"No 3D datacube found. Keys: {list(mat_dict.keys())}")
        candidates.sort(key=lambda x: np.prod(x[1]), reverse=True)
        return candidates[0][0]


# ─────────────────────────────────────────────
# 2. ResultVisualizer
# ─────────────────────────────────────────────

class ResultVisualizer:
    """
    Handles all output for a completed inference run:
    figures, classification maps (.npy), and statistics text files.

    Parameters
    ----------
    class_names : list[str] — Ordered list matching model output indices

    Usage
    -----
        viz = ResultVisualizer(["Red", "Green", "Blue", "Paper"])
        viz.save_all(class_map, confidence_map, predictions,
                     dataset_name="Spectrum-1",
                     output_dir="inference_result/Spectrum-1")
    """

    def __init__(self, class_names: list[str]):
        self.class_names = class_names

    # ── public ──────────────────────────────

    def save_all(
        self,
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        predictions: np.ndarray,
        dataset_name: str,
        output_dir: str,
    ) -> None:
        """
        Save all outputs for one inference run.

        Writes:
          - <dataset_name>_class_map.npy
          - <dataset_name>_confidence_map.npy
          - <dataset_name>_classification.png      (3-panel figure)
          - <dataset_name>_classification_only.png (high-res map)
          - <dataset_name>_statistics.txt
        """
        os.makedirs(output_dir, exist_ok=True)
        self.save_arrays(class_map, confidence_map, output_dir, dataset_name)
        self.save_figures(class_map, confidence_map, output_dir, dataset_name)
        self.save_statistics(class_map, confidence_map, predictions, output_dir, dataset_name)

    def save_arrays(
        self,
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        output_dir: str,
        dataset_name: str,
    ) -> None:
        """Save classification and confidence maps as .npy arrays."""
        for arr, tag in [(class_map, "class_map"), (confidence_map, "confidence_map")]:
            path = os.path.join(output_dir, f"{dataset_name}_{tag}.npy")
            np.save(path, arr)
            print(f"  ✓ Saved: {path}")

    def save_figures(
        self,
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        output_dir: str,
        dataset_name: str,
    ) -> None:
        """Save 3-panel combined figure and a standalone high-res classification map."""
        n_classes = len(self.class_names)
        colors    = plt.cm.tab10(np.linspace(0, 1, n_classes))
        cmap_cls  = plt.matplotlib.colors.ListedColormap(colors)

        # ── 3-panel figure ──────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im1 = axes[0].imshow(
            class_map, cmap=cmap_cls, vmin=0, vmax=n_classes - 1, interpolation="nearest"
        )
        axes[0].set_title("Classification Map", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        cbar1 = plt.colorbar(im1, ax=axes[0], ticks=range(n_classes))
        cbar1.ax.set_yticklabels(self.class_names)

        im2 = axes[1].imshow(
            confidence_map, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
        )
        axes[1].set_title("Confidence Map", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("X (pixels)")
        plt.colorbar(im2, ax=axes[1], label="Confidence")

        alpha_map = np.clip(confidence_map, 0.3, 1.0)
        axes[2].imshow(
            class_map, cmap=cmap_cls, vmin=0, vmax=n_classes - 1,
            alpha=alpha_map, interpolation="nearest"
        )
        axes[2].set_title("Classification + Confidence", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("X (pixels)")

        fig.suptitle(f"Classification Results: {dataset_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        path_combined = os.path.join(output_dir, f"{dataset_name}_classification.png")
        plt.savefig(path_combined, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {path_combined}")
        plt.close(fig)

        # ── high-res standalone map ──────────
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        im3 = ax2.imshow(
            class_map, cmap=cmap_cls, vmin=0, vmax=n_classes - 1, interpolation="nearest"
        )
        ax2.set_title(f"{dataset_name} — Classification", fontsize=16, fontweight="bold")
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        cbar3 = plt.colorbar(im3, ax=ax2, ticks=range(n_classes), label="Class")
        cbar3.ax.set_yticklabels(self.class_names)
        plt.tight_layout()

        path_only = os.path.join(output_dir, f"{dataset_name}_classification_only.png")
        plt.savefig(path_only, dpi=200, bbox_inches="tight")
        print(f"  ✓ Saved: {path_only}")
        plt.close(fig2)

    def save_statistics(
        self,
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        predictions: np.ndarray,
        output_dir: str,
        dataset_name: str,
    ) -> None:
        """Write a detailed statistics report to a text file."""
        path = os.path.join(output_dir, f"{dataset_name}_statistics.txt")

        unique_classes, counts = np.unique(
            class_map[class_map >= 0], return_counts=True
        )
        total_classified = int((class_map >= 0).sum())
        valid_conf       = confidence_map[class_map >= 0]

        with open(path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CLASSIFICATION STATISTICS: {dataset_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Image Information:\n")
            f.write(f"  Size:                {class_map.shape[0]} × {class_map.shape[1]} px\n")
            f.write(f"  Classified pixels:   {total_classified}\n")
            f.write(f"  Unclassified pixels: {int((class_map < 0).sum())}\n\n")

            f.write("Class Distribution:\n")
            f.write(f"  {'Class':<15s} {'Pixels':>10s} {'%':>10s} {'Avg Conf':>12s}\n")
            f.write("  " + "-" * 50 + "\n")
            for cls, count in zip(unique_classes, counts):
                name     = self.class_names[cls] if cls < len(self.class_names) else f"Class {cls}"
                pct      = 100 * count / total_classified
                avg_conf = confidence_map[class_map == cls].mean()
                f.write(f"  {name:<15s} {count:>10d} {pct:>9.2f}% {avg_conf:>11.4f}\n")

            f.write("\nConfidence Statistics:\n")
            f.write(f"  Mean:   {valid_conf.mean():.4f}\n")
            f.write(f"  Median: {np.median(valid_conf):.4f}\n")
            f.write(f"  Std:    {valid_conf.std():.4f}\n")
            f.write(f"  Min:    {valid_conf.min():.4f}\n")
            f.write(f"  Max:    {valid_conf.max():.4f}\n\n")

            f.write("Per-Class Prediction Probabilities:\n")
            for i, name in enumerate(self.class_names):
                probs = predictions[:, i]
                f.write(
                    f"  {name:<15s} mean={probs.mean():.4f}  std={probs.std():.4f}\n"
                )

        print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HSI Inference Testing — classify HSI datacubes with a trained model"
    )
    parser.add_argument("--model", type=str, required=False, default=None,
                        help=(
                            "Full path to a .keras file, e.g.:\n"
                            "  training_results/hsi_simple_p9-s4-e20-b16_minmax"
                            "/20260302-1430/hsi_simple_p9-s4-e20-b16_minmax.keras"
                        ))
    parser.add_argument("--list_models", type=str, default=None,
                        metavar="TRAINING_RESULTS_DIR",
                        help="Scan a training_results dir and list all available .keras files, then exit.")
    parser.add_argument("--input_dir",   type=str, default="hsi_datasets/v303",
                        help="Directory of .mat files (used when --input_files not given)")
    parser.add_argument("--input_files", type=str, nargs="+", default=None,
                        help="Explicit .mat file paths (overrides --input_dir)")
    parser.add_argument("--output_dir",  type=str, default="inference_results",
                        help="Base output directory")
    parser.add_argument("--patch_size",  type=int, default=3)
    parser.add_argument("--stride",      type=int, default=1,
                        help="Patch stride (smaller = denser predictions)")
    parser.add_argument("--normalize",   type=str, default="minmax",
                        choices=["minmax", "max", "none"])
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--class_names", type=str, nargs="+",
                        default=["Red", "Green", "Blue", "Paper"])
    parser.add_argument("--skip_indian_pines", action="store_true",
                        help="Skip files containing 'indian_pines' in the name")
    return parser


def main():
    # if args.list_models:
    #     matches = []
    #     for root, _, files in os.walk(args.list_models):
    #         for f in files:
    #             if f.endswith(".keras"):
    #                 matches.append(os.path.join(root, f))
    #     matches.sort()
    #     print(f"\nAvailable models in '{args.list_models}':\n")
    #     for m in matches:
    #         # Print accompanying metadata summary if present
    #         meta = os.path.join(os.path.dirname(m), "metadata.txt")
    #         print(f"  {m}")
    #         if os.path.exists(meta):
    #             with open(meta) as mf:
    #                 for line in mf:
    #                     if any(k in line for k in ("val_accuracy", "val_loss", "Timestamp")):
    #                         print(f"    {line.rstrip()}")
    #         print()
    #     return

    # if not args.model:
    #     print("ERROR: --model is required unless --list_models is used.")
    #     return
    parser = _build_parser()
    args   = parser.parse_args()

    print("=" * 60)
    print("HSI BATCH CLASSIFICATION INFERENCE")
    print("=" * 60)
    print(f"Model      : {args.model}")
    print(f"Output dir : {args.output_dir}")
    print(f"Patch size : {args.patch_size}  Stride: {args.stride}")
    print(f"Classes    : {args.class_names}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve file list
    if args.input_files:
        files = args.input_files
    else:
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".mat")
        ]

    if args.skip_indian_pines:
        files = [f for f in files if "indian_pines" not in f.lower()]

    print(f"\nFiles to process: {len(files)}")

    # Instantiate engine and visualizer once for the whole batch
    engine = InferenceEngine(
        model_path=args.model,
        patch_size=args.patch_size,
        stride=args.stride,
        normalize=args.normalize,
        batch_size=args.batch_size,
    )
    engine.load_model()

    visualizer = ResultVisualizer(class_names=args.class_names)

    successful, failed = 0, 0

    for i, mat_path in enumerate(files, 1):
        dataset_name = os.path.splitext(os.path.basename(mat_path))[0]
        output_dir   = os.path.join(args.output_dir, dataset_name)

        print(f"\n[{i}/{len(files)}] {dataset_name}")
        print("=" * 60)

        try:
            class_map, confidence_map, predictions = engine.predict(mat_path)

            unique, counts = np.unique(class_map[class_map >= 0], return_counts=True)
            print(f"  Classified pixels : {int((class_map >= 0).sum())}")
            print(f"  Avg confidence    : {confidence_map[class_map >= 0].mean():.4f}")
            print(f"  Classes found     : {len(unique)}")

            print(f"  Saving to: {output_dir}/")
            visualizer.save_all(
                class_map, confidence_map, predictions,
                dataset_name=dataset_name,
                output_dir=output_dir,
            )
            successful += 1

        except Exception as exc:
            print(f"  ❌ ERROR: {exc}")
            failed += 1

    print(f"\n{'=' * 60}")
    print("BATCH COMPLETE")
    print(f"  Successful : {successful}/{len(files)}")
    print(f"  Failed     : {failed}/{len(files)}")
    print(f"  Results in : {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()