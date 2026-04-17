# -*- coding: utf-8 -*-
"""
HSI_Result_Viewer.py
=====================
Standalone viewer for reloading and re-displaying saved HSI inference results
(.npy class_map and confidence_map arrays) with proper aspect ratio handling
for pushbroom datasets with very short scan axes (13–110 lines).

Can be used standalone via CLI or imported and called from inference scripts.

Classes
-------
    ResultLoader   — Loads .npy arrays from an inference output directory
    MapUpsampler   — Optional nearest-neighbour upsampling along the short axis
    ResultViewer   — Renders interactive figure with configurable panel display

Usage (CLI)
-----------
    # Basic — both panels (default)
    python HSI_Result_Viewer.py --input_dir inference_results/Paper

    # Class map only
    python HSI_Result_Viewer.py --input_dir inference_results/Paper --show class

    # Confidence map only
    python HSI_Result_Viewer.py --input_dir inference_results/Paper --show confidence

    # With upsampling — short axis scaled up to at least 300px
    python HSI_Result_Viewer.py --input_dir inference_results/Paper --upsample --min_height 300

    # Batch — open all datasets in a results directory sequentially
    python HSI_Result_Viewer.py --results_root inference_results --upsample

Usage (import)
--------------
    from HSI_Result_Viewer import ResultLoader, MapUpsampler, ResultViewer

    loader  = ResultLoader("inference_results/Paper")
    class_map, confidence_map = loader.load()

    # optional upsampling
    class_map, confidence_map = MapUpsampler.upsample(class_map, confidence_map, min_height=300)

    viewer = ResultViewer(class_names=["Red", "Green", "Blue", "Paper"])
    viewer.show(class_map, confidence_map, title="Paper")              # both panels
    viewer.show(class_map, confidence_map, title="Paper", show="class")       # class only
    viewer.show(class_map, confidence_map, title="Paper", show="confidence")  # confidence only
"""

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─────────────────────────────────────────────
# 1. ResultLoader
# ─────────────────────────────────────────────

class ResultLoader:
    """
    Loads saved class_map and confidence_map .npy arrays from an inference
    output directory produced by HSI_Inference_Testing_PyTorch.py.

    Expected filenames
    ------------------
    <dataset_name>_class_map.npy
    <dataset_name>_confidence_map.npy

    Parameters
    ----------
    input_dir    : str        — Path to the inference output directory
    dataset_name : str | None — Explicit dataset name; auto-detected if None

    Usage
    -----
        loader = ResultLoader("inference_results/Paper")
        class_map, confidence_map = loader.load()
    """

    def __init__(self, input_dir: str, dataset_name: str | None = None):
        self.input_dir    = input_dir
        self.dataset_name = dataset_name or self._detect_name(input_dir)

    # ── public ──────────────────────────────

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and return (class_map, confidence_map).

        Returns
        -------
        class_map      : (H, W) int32
        confidence_map : (H, W) float32

        Raises
        ------
        FileNotFoundError if either .npy file is missing.
        """
        class_path = os.path.join(
            self.input_dir, f"{self.dataset_name}_class_map.npy"
        )
        conf_path = os.path.join(
            self.input_dir, f"{self.dataset_name}_confidence_map.npy"
        )

        for p in (class_path, conf_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Result file not found: {p}")

        class_map      = np.load(class_path)
        confidence_map = np.load(conf_path)

        print(f"Loaded '{self.dataset_name}'")
        print(f"  class_map      : {class_map.shape}, dtype={class_map.dtype}")
        print(f"  confidence_map : {confidence_map.shape}, dtype={confidence_map.dtype}")
        print(f"  Classified px  : {int((class_map >= 0).sum())}")
        print(f"  Unique classes : {np.unique(class_map[class_map >= 0]).tolist()}")

        return class_map, confidence_map

    # ── private ──────────────────────────────

    @staticmethod
    def _detect_name(input_dir: str) -> str:
        """
        Infer dataset name from the first *_class_map.npy file in input_dir.
        Falls back to the directory basename if none found.
        """
        suffix = "_class_map.npy"
        for f in os.listdir(input_dir):
            if f.endswith(suffix):
                return f[: -len(suffix)]
        return os.path.basename(input_dir.rstrip("/\\"))


# ─────────────────────────────────────────────
# 2. MapUpsampler
# ─────────────────────────────────────────────

class MapUpsampler:
    """
    Nearest-neighbour upsampling along the short (scan) axis.

    Operates purely in memory — no stored arrays are modified.
    Intended for display only; the returned arrays should NOT be saved
    back to disk as they are not original inference outputs.

    Usage
    -----
        class_map, confidence_map = MapUpsampler.upsample(
            class_map, confidence_map, min_height=300
        )
    """

    @staticmethod
    def upsample(
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        min_height: int = 300,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Upsample both maps along the short axis using nearest-neighbour
        repetition so the short dimension reaches at least min_height pixels.

        Parameters
        ----------
        class_map      : (H, W) int32
        confidence_map : (H, W) float32
        min_height     : int — minimum pixels for the short axis after upsampling

        Returns
        -------
        (class_map_up, confidence_map_up) — upsampled copies, original unchanged
        """
        H, W = class_map.shape
        short_ax = 0 if H <= W else 1
        short_dim = H if short_ax == 0 else W

        if short_dim >= min_height:
            print(
                f"  Short axis ({short_dim}px) already ≥ min_height ({min_height}px) "
                f"— no upsampling applied."
            )
            return class_map, confidence_map

        scale = int(np.ceil(min_height / short_dim))

        if short_ax == 0:
            class_up = np.repeat(class_map,      scale, axis=0)
            conf_up  = np.repeat(confidence_map, scale, axis=0)
        else:
            class_up = np.repeat(class_map,      scale, axis=1)
            conf_up  = np.repeat(confidence_map, scale, axis=1)

        print(
            f"  Upsampled: {class_map.shape} → {class_up.shape} "
            f"(scale ×{scale} along axis {short_ax})"
        )
        print("  ⚠  Display only — upsampled arrays are NOT saved to disk.")

        return class_up, conf_up


# ─────────────────────────────────────────────
# 3. ResultViewer
# ─────────────────────────────────────────────

class ResultViewer:
    """
    Renders an interactive figure (class map and/or confidence map)
    with dynamic figsize computed from the actual map aspect ratio.

    The class colormap is built from the class_names list so colours
    match their semantic meaning:
        - Names containing 'red'    → red   (#d62728)
        - Names containing 'green'  → green (#2ca02c)
        - Names containing 'blue'   → blue  (#1f77b4)
        - Names containing 'paper'  → light grey (#cccccc)
        - All others                → tab10 fallback

    Parameters
    ----------
    class_names  : list[str] — Ordered class label list
    max_fig_dim  : float     — Longest figure dimension in inches (default 16)

    Usage
    -----
        viewer = ResultViewer(["Red", "Green", "Blue", "Paper"])
        viewer.show(class_map, confidence_map, title="Paper")
        viewer.show(class_map, confidence_map, title="Paper", show="class")
        viewer.show(class_map, confidence_map, title="Paper", show="confidence")
    """

    # Semantic colour map — matched case-insensitively against class names
    _SEMANTIC_COLORS: dict[str, str] = {
        "red":   "#d62728",
        "green": "#2ca02c",
        "blue":  "#1f77b4",
        "paper": "#cccccc",
    }

    def __init__(
        self,
        class_names: list[str],
        max_fig_dim: float = 16.0,
    ):
        self.class_names = class_names
        self.max_fig_dim = max_fig_dim
        self._cmap_cls   = self._build_class_cmap(class_names)
        self._ensure_interactive_backend()

    # ── public ──────────────────────────────

    def show(
        self,
        class_map: np.ndarray,
        confidence_map: np.ndarray,
        title: str = "",
        show: str = "both",
    ) -> None:
        """
        Display the figure in an interactive matplotlib window.

        Parameters
        ----------
        class_map      : (H, W) int32
        confidence_map : (H, W) float32
        title          : str — suptitle label (usually the dataset name)
        show           : 'both' | 'class' | 'confidence'
                         Controls which panels are rendered. Default 'both'.
        """
        if show not in ("both", "class", "confidence"):
            raise ValueError(
                f"show must be 'both', 'class', or 'confidence' — got '{show}'"
            )

        show_cls  = show in ("both", "class")
        show_conf = show in ("both", "confidence")
        n_panels  = 2 if show == "both" else 1

        H, W    = class_map.shape
        figsize = self._compute_figsize(H, W, n_panels=n_panels)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)

        # Normalise axes to always be a list for uniform handling
        if n_panels == 1:
            axes = [axes]

        ax_iter = iter(axes)

        # ── class map ───────────────────────
        if show_cls:
            ax = next(ax_iter)
            n_classes = len(self.class_names)
            im1 = ax.imshow(
                class_map,
                cmap=self._cmap_cls,
                vmin=0,
                vmax=n_classes - 1,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title("Classification Map", fontsize=12, fontweight="bold")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (scan lines)")
            cbar1 = plt.colorbar(
                im1, ax=ax, ticks=range(n_classes), orientation="horizontal"
            )
            cbar1.set_ticks(range(n_classes))
            cbar1.ax.set_xticklabels(self.class_names, rotation=15, ha="right")

        # ── confidence map ───────────────────
        if show_conf:
            ax = next(ax_iter)
            im2 = ax.imshow(
                confidence_map,
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title("Confidence Map", fontsize=12, fontweight="bold")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (scan lines)")
            plt.colorbar(
                im2, ax=ax, label="Confidence", orientation="horizontal"
            )

        # ── stats annotation ─────────────────
        classified = int((class_map >= 0).sum())
        avg_conf   = (
            float(confidence_map[class_map >= 0].mean())
            if classified > 0 else 0.0
        )
        unique_cls = np.unique(class_map[class_map >= 0])
        stats_text = (
            f"Shape: {H}×{W}  |  Classified: {classified}  |  "
            f"Avg conf: {avg_conf:.4f}  |  Classes: {len(unique_cls)}"
        )

        sup = f"{title}  —  {stats_text}" if title else stats_text
        fig.suptitle(sup, fontsize=10)

        plt.tight_layout()
        plt.show()

    # ── private ──────────────────────────────

    @classmethod
    def _build_class_cmap(cls, class_names: list[str]) -> mcolors.ListedColormap:
        """
        Build a ListedColormap by matching each class name against
        _SEMANTIC_COLORS (case-insensitive substring match).
        Falls back to tab10 colours for unrecognised names.
        """
        tab10      = plt.cm.tab10.colors
        tab10_iter = iter(tab10)
        colours    = []

        for name in class_names:
            lower = name.lower()
            matched = None
            for keyword, hex_colour in cls._SEMANTIC_COLORS.items():
                if keyword in lower:
                    matched = hex_colour
                    break
            if matched:
                colours.append(matched)
            else:
                # consume next tab10 colour, skipping any already used semantically
                colours.append(next(tab10_iter, "#888888"))

        return mcolors.ListedColormap(colours)

    # ── private ──────────────────────────────

    def _compute_figsize(
        self, H: int, W: int, n_panels: int
    ) -> tuple[float, float]:
        """
        Compute a figsize that respects the map's H:W aspect ratio and
        scales within max_fig_dim, multiplied by n_panels horizontally.

        For very short H (pushbroom), this prevents the squashed appearance
        without modifying the underlying data.
        """
        aspect = H / W  # e.g. 13/1632 ≈ 0.008 for a very short scan

        # Single panel width constrained by max_fig_dim
        panel_w = self.max_fig_dim / n_panels
        panel_h = panel_w * aspect

        # Apply a minimum panel height so the figure is never invisible
        min_panel_h = 1.5
        if panel_h < min_panel_h:
            panel_h = min_panel_h

        fig_w = panel_w * n_panels
        fig_h = panel_h + 1.5   # extra space for suptitle + horizontal colorbars

        return fig_w, fig_h

    @staticmethod
    def _ensure_interactive_backend() -> None:
        """
        Switch to an interactive backend if currently set to a non-interactive one.
        Tries TkAgg first, falls back to Qt5Agg, then leaves unchanged.
        """
        current = matplotlib.get_backend().lower()
        non_interactive = {"agg", "cairo", "pdf", "ps", "svg", "template"}

        if current not in non_interactive:
            return  # already interactive

        for backend in ("TkAgg", "Qt5Agg", "WXAgg"):
            try:
                matplotlib.use(backend)
                return
            except Exception:
                continue


# ─────────────────────────────────────────────
# 4. CLI entrypoint
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HSI Result Viewer — reload and display saved inference maps"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir", type=str,
        help="Single inference output directory containing _class_map.npy and _confidence_map.npy"
    )
    group.add_argument(
        "--results_root", type=str,
        help="Root directory containing multiple per-dataset inference output folders"
    )

    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="Explicit dataset name (auto-detected from filenames if omitted)"
    )
    parser.add_argument(
        "--class_names", type=str, nargs="+",
        default=["Red", "Green", "Blue", "Paper"]
    )
    parser.add_argument(
        "--upsample", action="store_true",
        help="Apply nearest-neighbour upsampling along the short axis for display"
    )
    parser.add_argument(
        "--min_height", type=int, default=300,
        help="Minimum short-axis pixels after upsampling (default: 300, requires --upsample)"
    )
    parser.add_argument(
        "--max_fig_dim", type=float, default=16.0,
        help="Maximum figure dimension in inches (default: 16)"
    )
    parser.add_argument(
        "--show", type=str, default="both",
        choices=["both", "class", "confidence"],
        help="Which panels to display: 'both' (default), 'class', or 'confidence'"
    )
    return parser


def _view_single(
    input_dir: str,
    dataset_name: str | None,
    class_names: list[str],
    upsample: bool,
    min_height: int,
    max_fig_dim: float,
    show: str = "both",
) -> None:
    """Load and display a single inference result directory."""
    loader = ResultLoader(input_dir, dataset_name)
    class_map, confidence_map = loader.load()

    if upsample:
        class_map, confidence_map = MapUpsampler.upsample(
            class_map, confidence_map, min_height=min_height
        )

    viewer = ResultViewer(class_names=class_names, max_fig_dim=max_fig_dim)
    viewer.show(class_map, confidence_map, title=loader.dataset_name, show=show)


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.input_dir:
        _view_single(
            input_dir=args.input_dir,
            dataset_name=args.dataset_name,
            class_names=args.class_names,
            upsample=args.upsample,
            min_height=args.min_height,
            max_fig_dim=args.max_fig_dim,
            show=args.show,
        )

    elif args.results_root:
        # collect all subdirectories that contain at least one _class_map.npy
        subdirs = sorted([
            os.path.join(args.results_root, d)
            for d in os.listdir(args.results_root)
            if os.path.isdir(os.path.join(args.results_root, d))
            and any(
                f.endswith("_class_map.npy")
                for f in os.listdir(os.path.join(args.results_root, d))
            )
        ])

        if not subdirs:
            print(f"No inference result directories found in '{args.results_root}'.")
            return

        print(f"Found {len(subdirs)} result(s) in '{args.results_root}'.")
        for i, subdir in enumerate(subdirs, 1):
            print(f"\n[{i}/{len(subdirs)}] {os.path.basename(subdir)}")
            try:
                _view_single(
                    input_dir=subdir,
                    dataset_name=None,
                    class_names=args.class_names,
                    upsample=args.upsample,
                    min_height=args.min_height,
                    max_fig_dim=args.max_fig_dim,
                    show=args.show,
                )
            except Exception as exc:
                print(f"  ❌ ERROR: {exc}")


if __name__ == "__main__":
    main()