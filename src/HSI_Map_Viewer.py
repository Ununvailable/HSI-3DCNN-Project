# -*- coding: utf-8 -*-
"""
HSI-MapViewer.py
=================
Standalone viewer for classification and confidence .npy maps
produced by HSI-Inference_Testing.py.

Can be imported as a class or invoked from the CLI anywhere in the project.

Usage (CLI) - maps only
-----------------------
    python HSI-MapViewer.py --scan inference_results/Spectrum-Simplified --class_names Red Green Blue Paper
    python HSI-MapViewer.py --class_map path/to/class_map.npy --class_names Red Green Blue Paper
    python HSI-MapViewer.py --class_map ... --conf_map ... --threshold 0.8 --output_dir results/

Usage (CLI) - overlay on .mat band image
-----------------------------------------
    # Single band background
    python HSI-MapViewer.py --scan inference_results/Spectrum-Simplified \
        --mat hsi_datasets/v303/Spectrum-Simplified.mat \
        --band 10 --class_names Red Green Blue Paper

    # False-color RGB background (R=band6, G=band3, B=band0)
    python HSI-MapViewer.py --scan inference_results/Spectrum-Simplified \
        --mat hsi_datasets/v303/Spectrum-Simplified.mat \
        --rgb 6 3 0 --class_names Red Green Blue Paper

    # Both overlay modes at once
    python HSI-MapViewer.py --scan inference_results/Spectrum-Simplified \
        --mat hsi_datasets/v303/Spectrum-Simplified.mat \
        --band 10 --rgb 6 3 0 --alpha 0.45 --output_dir results/

Usage (import)
--------------
    from HSI_MapViewer import MapViewer

    viewer    = MapViewer(class_names=["Red", "Green", "Blue", "Paper"])
    class_map = MapViewer.load_class_map("path/to/class_map.npy")
    conf_map  = MapViewer.load_conf_map("path/to/conf_map.npy")

    viewer.plot_both(class_map, conf_map, title="Spectrum-Simplified")
    viewer.plot_class_map(class_map, title="Spectrum-Simplified")
    viewer.plot_confidence_map(conf_map, title="Spectrum-Simplified")
    viewer.plot_filtered(class_map, conf_map, threshold=0.8)
    viewer.print_stats(class_map, conf_map)

    # Overlay on single band
    bg = MapViewer.load_band_from_mat("Spectrum-Simplified.mat", band_idx=10)
    viewer.plot_overlay_single_band(class_map, bg, title="Spectrum-Simplified", alpha=0.45)

    # Overlay on false-color RGB
    bg_rgb = MapViewer.load_rgb_from_mat("Spectrum-Simplified.mat", r=6, g=3, b=0)
    viewer.plot_overlay_rgb(class_map, bg_rgb, title="Spectrum-Simplified", alpha=0.45)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ─────────────────────────────────────────────
# MapViewer
# ─────────────────────────────────────────────

class MapViewer:
    """
    Loads and visualizes .npy classification and confidence maps,
    with optional semi-transparent overlay on HSI band images.

    Parameters
    ----------
    class_names : list[str]
        Ordered class labels matching model output indices.
        Unclassified pixels (value -1) are always transparent in overlays,
        white in standalone maps.
    cmap : str
        Matplotlib colormap for the class map (default: 'tab10').
    """

    def __init__(self, class_names: list[str] | None = None, cmap: str = "tab10"):
        self.class_names = class_names or []
        self.cmap        = cmap

    # ── loaders ─────────────────────────────

    @staticmethod
    def load_class_map(path: str) -> np.ndarray:
        """Load a class map .npy file. Returns (H, W) int32 array."""
        arr = np.load(path)
        print(f"  class_map  loaded: shape={arr.shape}, "
              f"range=[{arr.min()}, {arr.max()}]")
        return arr.astype(np.int32)

    @staticmethod
    def load_conf_map(path: str) -> np.ndarray:
        """Load a confidence map .npy file. Returns (H, W) float32 array."""
        arr = np.load(path)
        print(f"  conf_map   loaded: shape={arr.shape}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")
        return arr.astype(np.float32)

    @staticmethod
    def load_band_from_mat(mat_path: str, band_idx: int = 0) -> np.ndarray:
        """
        Extract a single spectral band from a .mat datacube.

        Returns
        -------
        (H, W) float32 array normalized to [0, 1] for display.
        """
        cube = MapViewer._load_cube_bhw(mat_path)
        B    = cube.shape[0]
        if not (0 <= band_idx < B):
            raise ValueError(f"band_idx={band_idx} out of range [0, {B - 1}]")
        img = cube[band_idx].astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        print(f"  band {band_idx} loaded from {os.path.basename(mat_path)}: shape={img.shape}")
        return img

    @staticmethod
    def load_rgb_from_mat(
        mat_path: str,
        r: int = 0,
        g: int = 1,
        b: int = 2,
    ) -> np.ndarray:
        """
        Build a false-color RGB image from three spectral bands.

        Returns
        -------
        (H, W, 3) uint8 array.
        """
        cube = MapViewer._load_cube_bhw(mat_path)
        B    = cube.shape[0]
        for idx, name in zip([r, g, b], ["r", "g", "b"]):
            if not (0 <= idx < B):
                raise ValueError(f"Band index {name}={idx} out of range [0, {B - 1}]")

        def _norm(ch: np.ndarray) -> np.ndarray:
            ch = ch.astype(np.float32)
            lo, hi = ch.min(), ch.max()
            if hi > lo:
                ch = (ch - lo) / (hi - lo)
            return (ch * 255).astype(np.uint8)

        rgb = np.stack([_norm(cube[r]), _norm(cube[g]), _norm(cube[b])], axis=-1)
        print(f"  RGB (r={r}, g={g}, b={b}) loaded from "
              f"{os.path.basename(mat_path)}: shape={rgb.shape}")
        return rgb

    # ── overlay plots ────────────────────────

    def plot_overlay_single_band(
        self,
        class_map: np.ndarray,
        band_img: np.ndarray,
        title: str = "",
        alpha: float = 0.45,
        save_path: str | None = None,
    ) -> None:
        """
        Overlay the classification map semi-transparently on a grayscale band image.
        Unclassified pixels (-1) are fully transparent in the overlay.

        Parameters
        ----------
        class_map : (H, W) int32
        band_img  : (H, W) float32 normalized to [0, 1], from load_band_from_mat()
        alpha     : Opacity of the classification overlay (0=invisible, 1=opaque)
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))

        # Left — band image alone
        axes[0].imshow(band_img, cmap="gray", interpolation="nearest")
        axes[0].set_title("Band Image", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")

        # Right — band image + classification overlay
        axes[1].imshow(band_img, cmap="gray", interpolation="nearest")
        rgba = self._class_map_to_rgba(class_map, transparent_unclassified=True)
        rgba[..., 3] = np.where(class_map >= 0, alpha, 0.0)  # per-pixel alpha
        axes[1].imshow(rgba, interpolation="nearest")
        axes[1].set_title(
            f"Classification Overlay  (alpha={alpha})",
            fontsize=13, fontweight="bold",
        )
        axes[1].set_xlabel("X (pixels)")
        self._add_class_legend(axes[1])

        fig.suptitle(
            f"{title} — Single Band Overlay" if title else "Single Band Overlay",
            fontsize=15, fontweight="bold",
        )
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    def plot_overlay_rgb(
        self,
        class_map: np.ndarray,
        rgb_img: np.ndarray,
        title: str = "",
        alpha: float = 0.45,
        save_path: str | None = None,
    ) -> None:
        """
        Overlay the classification map semi-transparently on a false-color RGB image.
        Unclassified pixels (-1) are fully transparent in the overlay.

        Parameters
        ----------
        class_map : (H, W) int32
        rgb_img   : (H, W, 3) uint8, from load_rgb_from_mat()
        alpha     : Opacity of the classification overlay (0=invisible, 1=opaque)
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))

        # Left — RGB image alone
        axes[0].imshow(rgb_img, interpolation="nearest")
        axes[0].set_title("False-Color RGB", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")

        # Right — RGB image + classification overlay
        axes[1].imshow(rgb_img, interpolation="nearest")
        rgba = self._class_map_to_rgba(class_map, transparent_unclassified=True)
        rgba[..., 3] = np.where(class_map >= 0, alpha, 0.0)
        axes[1].imshow(rgba, interpolation="nearest")
        axes[1].set_title(
            f"Classification Overlay  (alpha={alpha})",
            fontsize=13, fontweight="bold",
        )
        axes[1].set_xlabel("X (pixels)")
        self._add_class_legend(axes[1])

        fig.suptitle(
            f"{title} — RGB Overlay" if title else "RGB Overlay",
            fontsize=15, fontweight="bold",
        )
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ── standard map plots ───────────────────

    def plot_class_map(
        self,
        class_map: np.ndarray,
        title: str = "",
        save_path: str | None = None,
    ) -> None:
        """Plot the classification map. Unclassified pixels (-1) shown in white."""
        rgba = self._class_map_to_rgba(class_map)
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(rgba, interpolation="nearest")
        ax.set_title(
            f"{title} — Classification Map" if title else "Classification Map",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        self._add_class_legend(ax)
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    def plot_confidence_map(
        self,
        conf_map: np.ndarray,
        title: str = "",
        save_path: str | None = None,
    ) -> None:
        """Plot the confidence map as a viridis heatmap."""
        fig, ax = plt.subplots(figsize=(10, 12))
        im = ax.imshow(conf_map, cmap="viridis", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(
            f"{title} — Confidence Map" if title else "Confidence Map",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, label="Confidence", fraction=0.03)
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    def plot_both(
        self,
        class_map: np.ndarray,
        conf_map: np.ndarray,
        title: str = "",
        save_path: str | None = None,
    ) -> None:
        """3-panel: classification | confidence | confidence-weighted overlay."""
        rgba      = self._class_map_to_rgba(class_map)
        alpha_map = np.clip(conf_map, 0.3, 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        axes[0].imshow(rgba, interpolation="nearest")
        axes[0].set_title("Classification Map", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        self._add_class_legend(axes[0])

        im = axes[1].imshow(conf_map, cmap="viridis", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[1].set_title("Confidence Map", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("X (pixels)")
        plt.colorbar(im, ax=axes[1], label="Confidence", fraction=0.03)

        axes[2].imshow(rgba, alpha=alpha_map, interpolation="nearest")
        axes[2].set_title("Classification + Confidence", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("X (pixels)")

        fig.suptitle(title, fontsize=15, fontweight="bold")
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    def plot_filtered(
        self,
        class_map: np.ndarray,
        conf_map: np.ndarray,
        threshold: float = 0.8,
        title: str = "",
        save_path: str | None = None,
    ) -> None:
        """Show only pixels at or above the confidence threshold."""
        masked   = class_map.copy()
        masked[conf_map < threshold] = -1
        kept_pct = 100 * (masked >= 0).sum() / max((class_map >= 0).sum(), 1)
        rgba     = self._class_map_to_rgba(masked)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(rgba, interpolation="nearest")
        axes[0].set_title(
            f"Filtered  (conf >= {threshold:.2f})  —  {kept_pct:.1f}% retained",
            fontsize=12, fontweight="bold",
        )
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        self._add_class_legend(axes[0])

        im = axes[1].imshow(conf_map, cmap="viridis", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[1].set_title("Confidence Map", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("X (pixels)")
        plt.colorbar(im, ax=axes[1], label="Confidence", fraction=0.03)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ── stats ────────────────────────────────

    def print_stats(
        self,
        class_map: np.ndarray,
        conf_map: np.ndarray | None = None,
    ) -> None:
        """Print a concise summary of the loaded maps to stdout."""
        total      = class_map.size
        classified = int((class_map >= 0).sum())

        print("=" * 50)
        print("  Map Statistics")
        print("=" * 50)
        print(f"  Shape              : {class_map.shape}")
        print(f"  Total pixels       : {total:,}")
        print(f"  Classified         : {classified:,}  ({100*classified/total:.1f}%)")
        print(f"  Unclassified (-1)  : {total - classified:,}")

        if classified > 0:
            print("\n  Class Distribution:")
            unique, counts = np.unique(class_map[class_map >= 0], return_counts=True)
            for cls, count in zip(unique, counts):
                name     = (self.class_names[cls]
                            if cls < len(self.class_names) else f"Class {cls}")
                pct      = 100 * count / classified
                conf_str = ""
                if conf_map is not None:
                    avg      = conf_map[class_map == cls].mean()
                    conf_str = f"  avg_conf={avg:.4f}"
                print(f"    {name:<12s}  {count:>8,}  ({pct:5.1f}%){conf_str}")

        if conf_map is not None:
            valid = conf_map[class_map >= 0]
            if valid.size > 0:
                print(f"\n  Confidence  mean={valid.mean():.4f}  "
                      f"median={np.median(valid):.4f}  "
                      f"std={valid.std():.4f}")
        print("=" * 50)

    # ── private helpers ──────────────────────

    def _class_map_to_rgba(
        self,
        class_map: np.ndarray,
        transparent_unclassified: bool = False,
    ) -> np.ndarray:
        """
        Convert integer class_map to (H, W, 4) RGBA array.

        Parameters
        ----------
        transparent_unclassified : If True, unclassified pixels (-1) get alpha=0
                                   (used for overlays). If False, they are white.
        """
        n      = max(len(self.class_names), 1)
        colors = plt.cm.tab10(np.linspace(0, 1, n))

        H, W = class_map.shape
        # Default: white opaque
        rgba = np.ones((H, W, 4), dtype=np.float32)

        if transparent_unclassified:
            rgba[class_map < 0] = [0.0, 0.0, 0.0, 0.0]  # fully transparent

        for cls_idx in range(n):
            mask = (class_map == cls_idx)
            if mask.any():
                rgba[mask] = colors[cls_idx]

        return rgba

    def _add_class_legend(self, ax: plt.Axes) -> None:
        """Add a class color legend to the given axes."""
        if not self.class_names:
            return
        n      = len(self.class_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[i], label=self.class_names[i])
            for i in range(n)
        ]
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color="white", ec="gray",
                          label="Unclassified")
        )
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

    @staticmethod
    def _load_cube_bhw(mat_path: str) -> np.ndarray:
        """Load .mat and return cube in (B, H, W) float32."""
        M = loadmat(mat_path)
        # Find key
        if "DataCube" in M:
            key = "DataCube"
        else:
            candidates = [
                (k, v.shape)
                for k, v in M.items()
                if not k.startswith("__")
                and isinstance(v, np.ndarray)
                and v.ndim == 3
            ]
            if not candidates:
                raise KeyError(f"No 3D datacube found in {mat_path}.")
            candidates.sort(key=lambda x: np.prod(x[1]), reverse=True)
            key = candidates[0][0]

        cube = np.array(M[key])
        
        # band_axis = int(np.argmin(cube.shape))
        # if band_axis == 0:
        #     return cube.astype(np.float32)
        # elif band_axis == 1:
        #     return np.transpose(cube, (1, 0, 2)).astype(np.float32)
        # else:
        #     return np.transpose(cube, (2, 0, 1)).astype(np.float32)
        
        return np.transpose(cube, (2, 0, 1)).astype(np.float32)

    @staticmethod
    def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close(fig)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _find_map_pair(directory: str) -> tuple[str | None, str | None]:
    """Scan a directory for a *_class_map.npy / *_confidence_map.npy pair."""
    class_path = conf_path = None
    for f in sorted(os.listdir(directory)):
        if f.endswith("_class_map.npy"):
            class_path = os.path.join(directory, f)
        elif f.endswith("_confidence_map.npy"):
            conf_path = os.path.join(directory, f)
    return class_path, conf_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HSI Map Viewer — visualize .npy maps with optional .mat overlay"
    )

    # Map sources
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--scan", type=str, default=None, metavar="DIR",
                     help="Auto-scan directory for a _class_map / _confidence_map pair")
    parser.add_argument("--class_map",   type=str, default=None,
                        help="Path to *_class_map.npy")
    parser.add_argument("--conf_map",    type=str, default=None,
                        help="Path to *_confidence_map.npy")

    # Overlay source
    parser.add_argument("--mat", type=str, default=None,
                        help="Path to .mat datacube for overlay background")
    parser.add_argument("--band", type=int, default=None,
                        help="Band index for single-band overlay background")
    parser.add_argument("--rgb", type=int, nargs=3, default=None,
                        metavar=("R", "G", "B"),
                        help="Three band indices for false-color RGB overlay e.g. --rgb 6 3 0")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Classification overlay opacity 0.0-1.0 (default: 0.45)")

    # Options
    parser.add_argument("--class_names", type=str, nargs="+",
                        default=["Red", "Green", "Blue", "Paper"])
    parser.add_argument("--threshold",   type=float, default=None,
                        help="Confidence threshold for filtered view")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Save figures here instead of displaying")
    parser.add_argument("--stats",       action="store_true",
                        help="Print map statistics to stdout")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # ── resolve map paths ────────────────────
    class_path = args.class_map
    conf_path  = args.conf_map

    if args.scan:
        class_path, conf_path = _find_map_pair(args.scan)
        if not class_path and not conf_path:
            print(f"No maps found in '{args.scan}'.")
            return
        print(f"Found in '{args.scan}':")
        if class_path: print(f"  class_map → {class_path}")
        if conf_path:  print(f"  conf_map  → {conf_path}")

    if not class_path and not conf_path:
        print("Provide --class_map, --conf_map, or --scan.")
        parser.print_help()
        return

    # ── load maps ────────────────────────────
    viewer    = MapViewer(class_names=args.class_names)
    class_map = MapViewer.load_class_map(class_path) if class_path else None
    conf_map  = MapViewer.load_conf_map(conf_path)   if conf_path  else None

    base = (os.path.basename(class_path).replace("_class_map.npy", "")
            if class_path
            else os.path.basename(conf_path).replace("_confidence_map.npy", ""))

    def _out(suffix: str) -> str | None:
        if args.output_dir:
            return os.path.join(args.output_dir, f"{base}_{suffix}.png")
        return None

    # ── stats ────────────────────────────────
    if args.stats and class_map is not None:
        viewer.print_stats(class_map, conf_map)

    # ── standard map plots ───────────────────
    if class_map is not None and conf_map is not None:
        viewer.plot_both(class_map, conf_map, title=base,
                         save_path=_out("view_combined"))
        if args.threshold is not None:
            viewer.plot_filtered(class_map, conf_map,
                                 threshold=args.threshold, title=base,
                                 save_path=_out(f"view_filtered_{args.threshold}"))
    elif class_map is not None:
        viewer.plot_class_map(class_map, title=base,
                              save_path=_out("view_class"))
    elif conf_map is not None:
        viewer.plot_confidence_map(conf_map, title=base,
                                   save_path=_out("view_conf"))

    # ── overlay plots ────────────────────────
    if args.mat and class_map is not None:

        if args.band is not None:
            print(f"\nLoading band {args.band} from {args.mat}...")
            bg_band = MapViewer.load_band_from_mat(args.mat, band_idx=args.band)
            viewer.plot_overlay_single_band(
                class_map, bg_band,
                title=base,
                alpha=args.alpha,
                save_path=_out(f"overlay_band{args.band}"),
            )

        if args.rgb is not None:
            r, g, b = args.rgb
            print(f"\nLoading RGB (r={r}, g={g}, b={b}) from {args.mat}...")
            bg_rgb = MapViewer.load_rgb_from_mat(args.mat, r=r, g=g, b=b)
            viewer.plot_overlay_rgb(
                class_map, bg_rgb,
                title=base,
                alpha=args.alpha,
                save_path=_out(f"overlay_rgb_{r}_{g}_{b}"),
            )

        if args.band is None and args.rgb is None:
            print("--mat provided but neither --band nor --rgb specified. "
                  "Add --band N or --rgb R G B to generate overlay.")

    print("\nComplete.")


if __name__ == "__main__":
    main()