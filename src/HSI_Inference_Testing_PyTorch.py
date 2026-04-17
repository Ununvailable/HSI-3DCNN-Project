# -*- coding: utf-8 -*-
"""
HSI_Inference_Testing_PyTorch.py
==================================
Supporting script for running classification inference on HSI .mat datacubes
using a trained 3D-CNN model (PyTorch).
Migrated from HSI_Inference_Testing.py (TensorFlow/Keras).
Can be imported as classes or invoked directly from the CLI.

Usage (CLI)
-----------
    # List available models
    python HSI_Inference_Testing_PyTorch.py --list_models models

    # Single file
    python HSI_Inference_Testing_PyTorch.py \\
        --model models/hsi_simple_p3-s1-e50-b8_minmax/20260319-1430/hsi_simple_p3-s1-e50-b8_minmax.pth \\
        --input_files hsi_datasets/v303/Spectrum-1.mat \\
        --class_names Red Green Blue Paper

    # Batch (entire directory)
    python HSI_Inference_Testing_PyTorch.py \\
        --model models/hsi_simple_p3-s1-e50-b8_minmax/20260319-1430/hsi_simple_p3-s1-e50-b8_minmax.pth \\
        --input_dir hsi_datasets/v303 \\
        --class_names Red Green Blue Paper \\
        --stride 1

Usage (import)
--------------
    from HSI_Inference_Testing_PyTorch import InferenceEngine, ResultVisualizer

    engine = InferenceEngine(
        model_path="models/.../hsi_simple_p3-s1-e50-b8_minmax.pth",
        patch_size=3, stride=1, normalize="minmax", batch_size=8,
    )
    engine.load_model()

    class_map, confidence_map, predictions = engine.predict(
        mat_path="hsi_datasets/v303/Spectrum-1.mat"
    )

    viz = ResultVisualizer(class_names=["Red", "Green", "Blue", "Paper"])
    viz.save_all(class_map, confidence_map, predictions,
                 dataset_name="Spectrum-1", output_dir="inference_results/Spectrum-1")
"""

import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
# 0. Shared utility
# ─────────────────────────────────────────────

def guess_datacube_key(mat_dict: dict) -> str:
    """
    Infer the datacube variable name from a loaded .mat dictionary.

    Search order
    ------------
    1. Exact key 'DataCube'
    2. Largest 3D ndarray, with object-array unwrapping for nested MATLAB structs.

    Raises
    ------
    KeyError if no 3D array is found.
    """
    if "DataCube" in mat_dict:
        return "DataCube"

    def _unwrap(arr):
        v = arr
        for _ in range(3):
            if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
                v = v.item()
            else:
                break
        return v

    candidates = []
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        v2 = _unwrap(v)
        if isinstance(v2, np.ndarray) and v2.ndim == 3:
            candidates.append((int(np.prod(v2.shape)), k))

    if not candidates:
        raise KeyError(
            f"No 3D DataCube found. Available keys: {list(mat_dict.keys())}"
        )

    candidates.sort(reverse=True)
    return candidates[0][1]


# ─────────────────────────────────────────────
# 1. InferenceEngine
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Loads a trained PyTorch checkpoint and runs patch-based classification
    over entire HSI datacubes.

    The checkpoint must have been saved by ModelTrainer.save() in
    HSI_Train_All_In_One_PyTorch.py, which stores:
        model_state_dict, config (dict including 'bands'), train/val histories.

    Parameters
    ----------
    model_path    : str  — Path to a saved .pth checkpoint file
    architecture  : str  — 'simple' | 'li2017' (overridden by checkpoint config)
    patch_size    : int  — Spatial patch size (overridden by checkpoint config)
    stride        : int  — Extraction stride (smaller = denser map)
    normalize     : str  — 'minmax' | 'max' | 'none'
    batch_size    : int  — Prediction batch size

    Usage
    -----
        engine = InferenceEngine("model.pth")
        engine.load_model()
        class_map, conf_map, preds = engine.predict("Spectrum-1.mat")
    """

    def __init__(
        self,
        model_path: str,
        architecture: str = "simple",
        patch_size: int = 3,
        stride: int = 1,
        normalize: str = "minmax",
        batch_size: int = 8,
    ):
        self.model_path   = model_path
        self.architecture = architecture
        self.patch_size   = patch_size
        self.stride       = stride
        self.normalize    = normalize
        self.batch_size   = batch_size

        self._model: nn.Module | None = None
        self._expected_bands: int | None = None
        self._num_classes: int | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── public ──────────────────────────────

    def load_model(self) -> "InferenceEngine":
        """
        Load checkpoint, reconstruct model from metadata, restore weights.
        Architecture, band count, and patch size are read from checkpoint config.

        Returns self for method chaining.
        """
        from HSI_Train_All_In_One_PyTorch import HSIModelFactory

        checkpoint = torch.load(
            self.model_path, map_location=self._device, weights_only=True
        )

        config_meta = checkpoint.get("config", {})

        arch        = config_meta.get("model",       self.architecture)
        bands       = config_meta.get("bands",       None)
        saved_patch = config_meta.get("patch_size",  self.patch_size)
        num_classes = config_meta.get("num_classes", 4)

        if bands is None:
            raise RuntimeError(
                "Checkpoint does not contain 'bands' in config metadata.\n"
                "Re-train with the updated HSI_Train_All_In_One_PyTorch.py."
            )

        self._expected_bands = bands
        self._num_classes    = num_classes
        self.patch_size      = saved_patch  # override with trained value
        input_shape          = (1, bands, saved_patch, saved_patch)

        self._model = HSIModelFactory.build(arch, input_shape, num_classes)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        print(f"Model loaded   : {self.model_path}")
        print(f"  Architecture : {arch}")
        print(f"  Device       : {self._device}")
        print(f"  Bands        : {self._expected_bands}")
        print(f"  Patch size   : {saved_patch}")
        print(f"  Classes      : {num_classes}")
        return self

    def predict(self, mat_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full-image classification on a single .mat datacube.

        Parameters
        ----------
        mat_path : Path to the .mat file

        Returns
        -------
        class_map      : (H, W) int32   — class index per pixel (-1 = unclassified)
        confidence_map : (H, W) float32 — max softmax probability per pixel
        predictions    : (N, C) float32 — softmax probabilities for all patches
        """
        if self._model is None:
            raise RuntimeError("Call load_model() before predict().")

        cube_bhw = self._load_datacube(mat_path)
        B, H, W  = cube_bhw.shape

        # ── band compatibility ───────────────
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

        # (N, B, P, P) → (N, 1, B, P, P)  channels-first for Conv3d
        patches_t = torch.from_numpy(patches[:, np.newaxis]).float()

        print(f"  Running predictions (batch_size={self.batch_size})...")
        predictions = self._run_inference(patches_t)   # (N, C) softmax probs

        class_ids   = np.argmax(predictions,  axis=1)
        confidences = np.max(predictions,     axis=1)

        class_map      = np.full((H, W), -1,  dtype=np.int32)
        confidence_map = np.zeros((H, W),     dtype=np.float32)

        for (y, x), cls, conf in zip(coords, class_ids, confidences):
            class_map[y, x]      = cls
            confidence_map[y, x] = conf

        return class_map, confidence_map, predictions

    # ── private helpers ──────────────────────

    def _run_inference(self, patches_t: torch.Tensor) -> np.ndarray:
        """
        Batch inference with no_grad. Applies softmax to raw logits.

        Returns
        -------
        np.ndarray of shape (N, C), float32 softmax probabilities.
        """
        loader    = DataLoader(TensorDataset(patches_t), batch_size=self.batch_size)
        all_probs = []

        with torch.no_grad():
            for (Xb,) in loader:
                Xb    = Xb.to(self._device)
                probs = torch.softmax(self._model(Xb), dim=1) # type: ignore
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0).astype(np.float32)

    def _load_datacube(self, mat_path: str) -> np.ndarray:
        """
        Return cube in (B, H, W) float32.
        Assumes saved convention is (Y, X, B) — transposes to (B, Y, X).
        """
        M    = loadmat(mat_path)
        key  = guess_datacube_key(M)
        cube = np.array(M[key])

        if cube.ndim != 3:
            raise ValueError(
                f"'{key}' in {mat_path} is not 3D — shape={cube.shape}"
            )

        return np.transpose(cube, (2, 0, 1)).astype(np.float32)

    def _extract_patches(
        self, cube_bhw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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


# ─────────────────────────────────────────────
# 2. ResultVisualizer
# ─────────────────────────────────────────────

class ResultVisualizer:
    """
    Handles all output for a completed inference run:
    figures, classification maps (.npy), and statistics text files.

    matplotlib backend is set to 'Agg' only when no interactive display is
    detected, avoiding the global side-effect of the original script.

    Parameters
    ----------
    class_names : list[str] — Ordered list matching model output indices
    """

    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self._ensure_backend()

    @staticmethod
    def _ensure_backend() -> None:
        """
        Switch to non-interactive 'Agg' only when running headless.
        Leaves an already-active interactive backend (Jupyter, GUI) unchanged.
        """
        import matplotlib
        current = matplotlib.get_backend()
        if current.lower() in ("agg", "cairo", "pdf", "ps", "svg", "template"):
            return
        try:
            import tkinter  # noqa: F401
        except ImportError:
            matplotlib.use("Agg")

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

        Writes
        ------
          - <dataset_name>_class_map.npy
          - <dataset_name>_confidence_map.npy
          - <dataset_name>_classification.png      (3-panel figure)
          - <dataset_name>_classification_only.png (high-res map)
          - <dataset_name>_statistics.txt
        """
        os.makedirs(output_dir, exist_ok=True)
        self.save_arrays(class_map, confidence_map, output_dir, dataset_name)
        self.save_figures(class_map, confidence_map, output_dir, dataset_name)
        self.save_statistics(
            class_map, confidence_map, predictions, output_dir, dataset_name
        )

    def save_arrays(
        self,
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        output_dir: str,
        dataset_name: str,
    ) -> None:
        """Save classification and confidence maps as .npy arrays."""
        for arr, tag in [
            (class_map,      "class_map"),
            (confidence_map, "confidence_map"),
        ]:
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
        colors    = plt.cm.tab10(np.linspace(0, 1, n_classes)) # type: ignore
        cmap_cls  = plt.matplotlib.colors.ListedColormap(colors) # type: ignore

        # ── 3-panel figure ──────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im1 = axes[0].imshow(
            class_map, cmap=cmap_cls, vmin=0, vmax=n_classes - 1,
            interpolation="nearest"
        )
        axes[0].set_title("Classification Map", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        cbar1 = plt.colorbar(im1, ax=axes[0], ticks=range(n_classes), orientation="horizontal")
        cbar1.set_ticks(range(n_classes))
        cbar1.ax.set_xticklabels(self.class_names)

        im2 = axes[1].imshow(
            confidence_map, cmap="viridis", vmin=0, vmax=1,
            interpolation="nearest"
        )
        axes[1].set_title("Confidence Map", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("X (pixels)")
        plt.colorbar(im2, ax=axes[1], label="Confidence", orientation="horizontal")

        alpha_map = np.clip(confidence_map, 0.3, 1.0)
        axes[2].imshow(
            class_map, cmap=cmap_cls, vmin=0, vmax=n_classes - 1,
            alpha=alpha_map, interpolation="nearest"
        )
        axes[2].set_title(
            "Classification + Confidence", fontsize=14, fontweight="bold"
        )
        axes[2].set_xlabel("X (pixels)")

        fig.suptitle(
            f"Classification Results: {dataset_name}", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        path_combined = os.path.join(
            output_dir, f"{dataset_name}_classification.png"
        )
        plt.savefig(path_combined, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {path_combined}")
        plt.close(fig)

        # ── high-res standalone map ──────────
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        im3 = ax2.imshow(
            class_map, cmap=cmap_cls, vmin=0, vmax=n_classes - 1,
            interpolation="nearest"
        )
        ax2.set_title(
            f"{dataset_name} — Classification", fontsize=16, fontweight="bold"
        )
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        cbar3 = plt.colorbar(im3, ax=ax2, ticks=range(n_classes), label="Class",orientation="horizontal")
        cbar1.set_ticks(range(n_classes))
        cbar3.ax.set_xticklabels(self.class_names)
        plt.tight_layout()

        path_only = os.path.join(
            output_dir, f"{dataset_name}_classification_only.png"
        )
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
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("Image Information:\n")
            f.write(
                f"  Size:                "
                f"{class_map.shape[0]} × {class_map.shape[1]} px\n"
            )
            f.write(f"  Classified pixels:   {total_classified}\n")
            f.write(f"  Unclassified pixels: {int((class_map < 0).sum())}\n\n")

            f.write("Class Distribution:\n")
            f.write(
                f"  {'Class':<15s} {'Pixels':>10s} {'%':>10s} {'Avg Conf':>12s}\n"
            )
            f.write("  " + "-" * 50 + "\n")
            for cls, count in zip(unique_classes, counts):
                name = (
                    self.class_names[cls]
                    if cls < len(self.class_names)
                    else f"Class {cls}"
                )
                pct      = 100 * count / total_classified
                avg_conf = confidence_map[class_map == cls].mean()
                f.write(
                    f"  {name:<15s} {count:>10d} {pct:>9.2f}% {avg_conf:>11.4f}\n"
                )

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
                    f"  {name:<15s} mean={probs.mean():.4f}  "
                    f"std={probs.std():.4f}\n"
                )

        print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────
# 3. CLI entrypoint
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HSI Inference Testing (PyTorch) — "
            "classify HSI datacubes with a trained model"
        )
    )
    parser.add_argument(
        "--model", type=str, required=False, default=None,
        help=(
            "Full path to a .pth checkpoint file, e.g.:\n"
            "  models/hsi_simple_p3-s1-e50-b8_minmax"
            "/20260319-1430/hsi_simple_p3-s1-e50-b8_minmax.pth"
        ),
    )
    parser.add_argument(
        "--list_models", type=str, default=None,
        metavar="MODELS_DIR",
        help="Scan a models dir and list all available .pth files, then exit.",
    )
    parser.add_argument(
        "--architecture", type=str, default="simple",
        choices=["simple", "li2017"],
        help="Fallback architecture if not found in checkpoint config.",
    )
    parser.add_argument(
        "--input_dir", type=str, default="hsi_datasets/v303",
        help="Directory of .mat files (used when --input_files not given)",
    )
    parser.add_argument(
        "--input_files", type=str, nargs="+", default=None,
        help="Explicit .mat file paths (overrides --input_dir)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="inference_results",
        help="Base output directory",
    )
    parser.add_argument("--patch_size",  type=int,   default=3,
                        help="Fallback patch size if not found in checkpoint config.")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Patch stride (smaller = denser predictions)",
    )
    parser.add_argument(
        "--normalize", type=str, default="minmax",
        choices=["minmax", "max", "none"],
    )
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument(
        "--class_names", type=str, nargs="+",
        default=["Red", "Green", "Blue", "Paper"],
    )
    parser.add_argument(
        "--skip_indian_pines", action="store_true",
        help="Skip files containing 'indian_pines' in the name",
    )
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # ── --list_models mode ───────────────────
    if args.list_models:
        matches = []
        for root, _, files in os.walk(args.list_models):
            for f in files:
                if f.endswith(".pth"):
                    matches.append(os.path.join(root, f))
        matches.sort()
        print(f"\nAvailable models in '{args.list_models}':\n")
        for m in matches:
            meta = os.path.join(os.path.dirname(m), "metadata.txt")
            print(f"  {m}")
            if os.path.exists(meta):
                with open(meta) as mf:
                    for line in mf:
                        if any(
                            k in line
                            for k in ("val_accuracy", "val_loss", "Timestamp")
                        ):
                            print(f"    {line.rstrip()}")
            print()
        return

    if not args.model:
        print(
            "ERROR: --model is required unless --list_models is used.\n"
            "Use --list_models <models_dir> to browse available checkpoints."
        )
        return

    print("=" * 60)
    print("HSI BATCH CLASSIFICATION INFERENCE  (PyTorch)")
    print("=" * 60)
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA       : {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    print(f"Model      : {args.model}")
    print(f"Output dir : {args.output_dir}")
    print(f"Stride     : {args.stride}")
    print(f"Classes    : {args.class_names}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── resolve file list ────────────────────
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

    # ── instantiate engine and visualizer once ─
    engine = InferenceEngine(
        model_path=args.model,
        architecture=args.architecture,
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

            unique, _ = np.unique(class_map[class_map >= 0], return_counts=True)
            print(f"  Classified pixels : {int((class_map >= 0).sum())}")
            print(
                f"  Avg confidence    : "
                f"{confidence_map[class_map >= 0].mean():.4f}"
            )
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