# -*- coding: utf-8 -*-
"""
HSI-Dataset_Analysis.py
========================
Supporting script for inspecting and visualizing HSI .mat datacubes.
Can be imported as a class or invoked directly from the CLI.

Usage (CLI)
-----------
    # Single band
    python HSI-Dataset_Analysis.py --input hsi_datasets/v303/Red.mat --band 10

    # All bands grid
    python HSI-Dataset_Analysis.py --input hsi_datasets/v303/Red.mat --all_bands

    # False-color RGB composite
    python HSI-Dataset_Analysis.py --input hsi_datasets/v303/Red.mat --rgb 6 3 0

    # Full report (shape, range, dtype, spectral signature)
    python HSI-Dataset_Analysis.py --input hsi_datasets/v303/Red.mat --report

Usage (import)
--------------
    from HSI-Dataset_Analysis import DatasetAnalyzer
    analyzer = DatasetAnalyzer("hsi_datasets/v303/Red.mat")
    analyzer.load()
    analyzer.print_report()
    analyzer.visualize_band(band_idx=10, save_path="out/band10.png")
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive — safe for both CLI and import
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ─────────────────────────────────────────────
# DatasetAnalyzer
# ─────────────────────────────────────────────

class DatasetAnalyzer:
    """
    Loads an HSI .mat datacube and provides inspection and visualization tools.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file.
    datacube_key : str, optional
        Explicit variable name inside the .mat file.
        Auto-detected if omitted.

    Attributes (populated after load())
    ------------------------------------
    cube_bhw : np.ndarray — shape (B, H, W)
    key      : str        — variable name used
    B, H, W  : int        — cube dimensions
    name     : str        — dataset name derived from filename
    """

    def __init__(self, mat_path: str, datacube_key: str | None = None):
        self.mat_path     = mat_path
        self.datacube_key = datacube_key
        self.cube_bhw: np.ndarray | None = None
        self.key: str | None = None
        self.B = self.H = self.W = 0
        self.name = os.path.splitext(os.path.basename(mat_path))[0]

    # ── public ──────────────────────────────

    def load(self) -> "DatasetAnalyzer":
        """Load the datacube into memory. Returns self for chaining."""
        M = loadmat(self.mat_path)
        self.key = self.datacube_key or self._guess_key(M)

        cube = np.array(M[self.key])
        if cube.ndim != 3:
            raise ValueError(
                f"'{self.key}' in {self.mat_path} is not 3D — shape={cube.shape}"
            )

        # band_axis = int(np.argmin(cube.shape))
        # if band_axis == 0:
        #     self.cube_bhw = cube.astype(np.float32)
        # elif band_axis == 1:
        #     self.cube_bhw = np.transpose(cube, (1, 0, 2)).astype(np.float32)
        # else:
        #     self.cube_bhw = np.transpose(cube, (2, 0, 1)).astype(np.float32)

        # Saved convention is (Y, X, B) → reorder to (B, Y, X)
        self.cube_bhw = np.transpose(cube, (2, 0, 1)).astype(np.float32)

        self.B, self.H, self.W = self.cube_bhw.shape
        return self

    def print_report(self) -> None:
        """Print a summary of the datacube to stdout."""
        self._require_loaded()
        cube = self.cube_bhw
        print("=" * 55)
        print(f"  Dataset : {self.name}")
        print(f"  File    : {self.mat_path}")
        print(f"  Key     : {self.key}")
        print("-" * 55)
        print(f"  Shape   : (B={self.B}, H={self.H}, W={self.W})")
        print(f"  Dtype   : {cube.dtype}")
        print(f"  Range   : [{cube.min():.4f}, {cube.max():.4f}]")
        print(f"  Mean    : {cube.mean():.4f}  Std: {cube.std():.4f}")
        print("=" * 55)

    def visualize_band(
        self,
        band_idx: int = 0,
        cmap: str = "gray",
        save_path: str | None = None,
        show_histogram: bool = True,
    ) -> np.ndarray:
        """
        Visualize a single spectral band.

        Parameters
        ----------
        band_idx       : Band index (0-based)
        cmap           : Matplotlib colormap
        save_path      : If provided, saves the figure here
        show_histogram : Show intensity histogram alongside the image

        Returns
        -------
        2D numpy array of the selected band
        """
        self._require_loaded()

        if not (0 <= band_idx < self.B):
            raise ValueError(f"band_idx={band_idx} out of range [0, {self.B - 1}]")

        img = self.cube_bhw[band_idx]

        if show_histogram:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            im = axes[0].imshow(img, cmap=cmap, interpolation="nearest")
            axes[0].set_title(
                f"{self.name} — Band {band_idx}/{self.B}", fontsize=14, fontweight="bold"
            )
            axes[0].set_xlabel("X (pixels)")
            axes[0].set_ylabel("Y (pixels)")
            plt.colorbar(im, ax=axes[0], label="Intensity")

            axes[1].hist(img.flatten(), bins=100, color="steelblue", alpha=0.7, edgecolor="black")
            axes[1].set_xlabel("Pixel Intensity", fontsize=12)
            axes[1].set_ylabel("Frequency", fontsize=12)
            axes[1].set_title("Intensity Distribution", fontsize=14, fontweight="bold")
            axes[1].grid(True, alpha=0.3)

            stats = (
                f"Min:  {img.min():.2f}\nMax:  {img.max():.2f}\n"
                f"Mean: {img.mean():.2f}\nStd:  {img.std():.2f}"
            )
            axes[1].text(
                0.98, 0.98, stats,
                transform=axes[1].transAxes,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=10,
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.set_title(
                f"{self.name} — Band {band_idx}/{self.B}", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            plt.colorbar(im, ax=ax, label="Intensity")

        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return img

    def visualize_all_bands(
        self,
        cmap: str = "gray",
        save_dir: str | None = None,
        max_cols: int = 5,
    ) -> None:
        """
        Render a grid of all spectral bands, each paired with its intensity histogram.
        Each band occupies two adjacent columns: [image | histogram].

        Parameters
        ----------
        cmap     : Matplotlib colormap
        save_dir : Directory for per-band PNGs and the combined grid
        max_cols : Number of band pairs per row (actual subplot columns = max_cols * 2)
        """
        self._require_loaded()

        n_pairs_per_row = min(max_cols, self.B)   # band pairs per row
        n_rows          = int(np.ceil(self.B / n_pairs_per_row))
        n_subplot_cols  = n_pairs_per_row * 2     # image col + histogram col per pair

        fig, axes = plt.subplots(
            n_rows, n_subplot_cols,
            figsize=(4 * n_subplot_cols, 3.5 * n_rows),
        )
        # Normalise axes to always be 2-D (n_rows, n_subplot_cols)
        axes = np.array(axes).reshape(n_rows, n_subplot_cols)

        for i in range(self.B):
            row         = i // n_pairs_per_row
            pair_offset = (i %  n_pairs_per_row) * 2   # left column of this pair
            img         = self.cube_bhw[i]

            # ── image ───────────────────────────────────────────────────────
            ax_img = axes[row, pair_offset]
            ax_img.imshow(img, cmap=cmap, interpolation="nearest")
            ax_img.set_title(f"Band {i}", fontsize=9, fontweight="bold")
            ax_img.axis("off")

            # ── histogram ───────────────────────────────────────────────────
            ax_hist = axes[row, pair_offset + 1]
            ax_hist.hist(
                img.flatten(), bins=60,
                color="steelblue", alpha=0.75, edgecolor="none",
            )
            ax_hist.set_xlabel("Intensity", fontsize=7)
            ax_hist.set_ylabel("Freq",      fontsize=7)
            ax_hist.tick_params(labelsize=6)
            ax_hist.grid(True, alpha=0.25, linewidth=0.5)

            stats = (
                f"μ={img.mean():.1f}\n"
                f"σ={img.std():.1f}\n"
                f"[{img.min():.0f}, {img.max():.0f}]"
            )
            ax_hist.text(
                0.97, 0.97, stats,
                transform=ax_hist.transAxes,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=6,
            )

            # ── optional individual file save (image + histogram side-by-side) ──
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                band_path = os.path.join(save_dir, f"band_{i:03d}.png")
                fig_s, (ax_s_img, ax_s_hist) = plt.subplots(1, 2, figsize=(12, 5))

                ax_s_img.imshow(img, cmap=cmap, interpolation="nearest")
                ax_s_img.set_title(f"{self.name} — Band {i}", fontsize=12, fontweight="bold")
                ax_s_img.set_xlabel("X (pixels)")
                ax_s_img.set_ylabel("Y (pixels)")
                norm = plt.Normalize(vmin=img.min(), vmax=img.max())
                plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                             ax=ax_s_img, label="Intensity")

                ax_s_hist.hist(img.flatten(), bins=100,
                               color="steelblue", alpha=0.75, edgecolor="black")
                ax_s_hist.set_xlabel("Pixel Intensity", fontsize=11)
                ax_s_hist.set_ylabel("Frequency",       fontsize=11)
                ax_s_hist.set_title("Intensity Distribution", fontsize=12, fontweight="bold")
                ax_s_hist.grid(True, alpha=0.3)
                full_stats = (
                    f"Min:  {img.min():.2f}\nMax:  {img.max():.2f}\n"
                    f"Mean: {img.mean():.2f}\nStd:  {img.std():.2f}"
                )
                ax_s_hist.text(
                    0.98, 0.98, full_stats,
                    transform=ax_s_hist.transAxes,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    fontsize=10,
                )

                plt.tight_layout()
                plt.savefig(band_path, dpi=150, bbox_inches="tight")
                plt.close(fig_s)

        # Hide unused subplot pairs
        for i in range(self.B, n_rows * n_pairs_per_row):
            row         = i // n_pairs_per_row
            pair_offset = (i %  n_pairs_per_row) * 2
            axes[row, pair_offset    ].axis("off")
            axes[row, pair_offset + 1].axis("off")

        fig.suptitle(
            f"{self.name} — All {self.B} Bands  (image | histogram)",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        grid_path = os.path.join(save_dir, "all_bands_grid.png") if save_dir else None
        self._save_or_show(fig, grid_path)

    def visualize_false_color(
        self,
        r_band: int,
        g_band: int,
        b_band: int,
        save_path: str | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compose and display a false-color RGB image from three spectral bands.

        Parameters
        ----------
        r_band, g_band, b_band : Band indices for R, G, B channels
        save_path              : Save path for the figure
        normalize              : Normalize each channel independently to 0–255

        Returns
        -------
        (H, W, 3) uint8 RGB array
        """
        self._require_loaded()

        def _channel(idx: int) -> np.ndarray:
            ch = self.cube_bhw[idx].astype(np.float32)
            if normalize:
                lo, hi = ch.min(), ch.max()
                ch = ((ch - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else ch.astype(np.uint8)
            return ch

        rgb = np.stack([_channel(r_band), _channel(g_band), _channel(b_band)], axis=-1)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb)
        ax.set_title(
            f"{self.name} — False Color RGB  (R={r_band}, G={g_band}, B={b_band})",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return rgb

    def spectral_signature(
        self,
        y: int,
        x: int,
        save_path: str | None = None,
    ) -> np.ndarray:
        """
        Plot the spectral signature at a single pixel (y, x).

        Returns
        -------
        1D array of length B
        """
        self._require_loaded()
        sig = self.cube_bhw[:, y, x]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sig, color="steelblue", linewidth=1.5)
        ax.set_title(
            f"{self.name} — Spectral Signature at ({y}, {x})",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Band index")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return sig

    # ── private helpers ──────────────────────

    def _require_loaded(self) -> None:
        if self.cube_bhw is None:
            raise RuntimeError("Call load() before using this method.")

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
            raise KeyError(
                f"No 3D datacube found. Keys: {list(mat_dict.keys())}"
            )
        candidates.sort(key=lambda x: np.prod(x[1]), reverse=True)
        return candidates[0][0]

    @staticmethod
    def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved: {save_path}")
        plt.close(fig)


# ─────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HSI Dataset Analysis — inspect and visualize .mat datacubes"
    )
    parser.add_argument("--input",        type=str,  required=True,
                        help="Path to .mat file")
    parser.add_argument("--band",         type=int,  default=0,
                        help="Band index to visualize (0-based, default: 0)")
    parser.add_argument("--all_bands",    action="store_true",
                        help="Render grid of all spectral bands")
    parser.add_argument("--rgb",          type=int,  nargs=3, default=None,
                        metavar=("R", "G", "B"),
                        help="False-color RGB: supply 3 band indices e.g. --rgb 6 3 0")
    parser.add_argument("--signature",    type=int,  nargs=2, default=None,
                        metavar=("Y", "X"),
                        help="Spectral signature at pixel (Y, X)")
    parser.add_argument("--report",       action="store_true",
                        help="Print datacube summary (shape, range, dtype)")
    parser.add_argument("--cmap",         type=str,  default="gray",
                        help="Matplotlib colormap (default: gray)")
    parser.add_argument("--output_dir",   type=str,  default=None,
                        help="Directory for saved figures")
    parser.add_argument("--datacube_key", type=str,  default=None,
                        help="Explicit .mat variable name; auto-detected if omitted")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    print(f"Loading: {args.input}")
    analyzer = DatasetAnalyzer(args.input, datacube_key=args.datacube_key)
    analyzer.load()

    print(f"  Shape: ({analyzer.B}, {analyzer.H}, {analyzer.W})")

    # Always print report if --report or as default info
    if args.report or not any([args.all_bands, args.rgb, args.signature]):
        analyzer.print_report()

    def _out(filename: str) -> str | None:
        if args.output_dir:
            return os.path.join(args.output_dir, filename)
        return None

    if args.all_bands:
        save_dir = os.path.join(args.output_dir, f"{analyzer.name}_bands") if args.output_dir else None
        analyzer.visualize_all_bands(cmap=args.cmap, save_dir=save_dir)

    if args.rgb:
        r, g, b = args.rgb
        analyzer.visualize_false_color(
            r, g, b,
            save_path=_out(f"{analyzer.name}_RGB_{r}_{g}_{b}.png"),
        )

    if args.signature:
        y, x = args.signature
        analyzer.spectral_signature(
            y, x,
            save_path=_out(f"{analyzer.name}_signature_{y}_{x}.png"),
        )

    # Default: visualize the requested single band
    if not args.all_bands:
        analyzer.visualize_band(
            band_idx=args.band,
            cmap=args.cmap,
            save_path=_out(f"{analyzer.name}_band_{args.band}.png"),
        )

    print("✓ Complete.")


if __name__ == "__main__":
    main()