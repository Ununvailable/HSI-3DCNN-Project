"""
Extract and visualize single wavelength images from HSI datacubes
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import argparse


def load_datacube(mat_path):
    """Load datacube from .mat file."""
    M = loadmat(mat_path)
    
    # Find DataCube key
    if "DataCube" in M:
        key = "DataCube"
    else:
        # Find first 3D array
        for k, v in M.items():
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:
                key = k
                break
    
    cube = np.array(M[key])
    
    # Determine shape: (B, H, W)
    band_axis = int(np.argmin(cube.shape))
    
    if band_axis == 0:
        cube_bhw = cube
    elif band_axis == 1:
        cube_bhw = np.transpose(cube, (1, 0, 2))
    else:
        cube_bhw = np.transpose(cube, (2, 0, 1))
    
    return cube_bhw, key


def visualize_wavelength(cube_bhw, band_idx=0, title="", cmap='gray', 
                         save_path=None, show_histogram=True):
    """
    Visualize a single wavelength band from datacube.
    
    Args:
        cube_bhw: (B, H, W) datacube
        band_idx: Which band to visualize (0 to B-1)
        title: Plot title
        cmap: Colormap ('gray', 'viridis', 'hot', etc.)
        save_path: Path to save figure
        show_histogram: Show histogram of pixel intensities
    """
    B, H, W = cube_bhw.shape
    
    if band_idx < 0 or band_idx >= B:
        raise ValueError(f"band_idx={band_idx} out of range [0, {B-1}]")
    
    # Extract single band
    img = cube_bhw[band_idx, :, :]
    
    # Create figure
    if show_histogram:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Image
        im = axes[0].imshow(img, cmap=cmap, interpolation='nearest')
        axes[0].set_title(f'{title}\nBand {band_idx} / {B}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[0], label='Intensity')
        
        # Histogram
        axes[1].hist(img.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Pixel Intensity', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Intensity Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Stats
        stats_text = f'Min: {img.min():.2f}\nMax: {img.max():.2f}\nMean: {img.mean():.2f}\nStd: {img.std():.2f}'
        axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(f'{title}\nBand {band_idx} / {B}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Intensity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return img


def visualize_all_bands(cube_bhw, title="", cmap='gray', save_dir=None, max_cols=5):
    """
    Create grid visualization of all bands.
    
    Args:
        cube_bhw: (B, H, W) datacube
        title: Overall title
        cmap: Colormap
        save_dir: Directory to save individual band images
        max_cols: Maximum columns in grid
    """
    B, H, W = cube_bhw.shape
    
    # Calculate grid size
    n_cols = min(max_cols, B)
    n_rows = int(np.ceil(B / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each band
    for i in range(B):
        row = i // n_cols
        col = i % n_cols
        
        img = cube_bhw[i, :, :]
        im = axes[row, col].imshow(img, cmap=cmap, interpolation='nearest')
        axes[row, col].set_title(f'Band {i}', fontsize=10)
        axes[row, col].axis('off')
        
        # Save individual band if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            band_path = os.path.join(save_dir, f'band_{i:03d}.png')
            
            fig_single = plt.figure(figsize=(8, 6))
            plt.imshow(img, cmap=cmap, interpolation='nearest')
            plt.title(f'{title} - Band {i}')
            plt.colorbar(label='Intensity')
            plt.tight_layout()
            plt.savefig(band_path, dpi=150, bbox_inches='tight')
            plt.close(fig_single)
    
    # Hide unused subplots
    for i in range(B, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'{title}\nAll {B} Bands', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_dir:
        grid_path = os.path.join(save_dir, 'all_bands_grid.png')
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved grid: {grid_path}")
    
    plt.show()


def create_false_color_rgb(cube_bhw, r_band, g_band, b_band, title="", 
                           save_path=None, normalize=True):
    """
    Create false-color RGB image from 3 selected bands.
    
    Args:
        cube_bhw: (B, H, W) datacube
        r_band, g_band, b_band: Band indices for R, G, B channels
        title: Plot title
        save_path: Save path
        normalize: Normalize each channel to 0-255
    """
    B, H, W = cube_bhw.shape
    
    # Extract bands
    r = cube_bhw[r_band, :, :].astype(np.float32)
    g = cube_bhw[g_band, :, :].astype(np.float32)
    b = cube_bhw[b_band, :, :].astype(np.float32)
    
    if normalize:
        # Normalize each channel independently
        r = ((r - r.min()) / (r.max() - r.min()) * 255).astype(np.uint8)
        g = ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)
        b = ((b - b.min()) / (b.max() - b.min()) * 255).astype(np.uint8)
    
    # Stack into RGB
    rgb = np.stack([r, g, b], axis=-1)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(f'{title}\nFalse Color RGB (R={r_band}, G={g_band}, B={b_band})', 
             fontsize=14, fontweight='bold')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return rgb


def main():
    parser = argparse.ArgumentParser(description="Visualize HSI datacube wavelengths")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to .mat file")
    parser.add_argument("--band", type=int, default=0,
                       help="Band index to visualize (0-based)")
    parser.add_argument("--all_bands", action='store_true',
                       help="Visualize all bands in grid")
    parser.add_argument("--rgb", type=int, nargs=3, default=None,
                       help="Create false-color RGB (e.g., --rgb 6 3 0)")
    parser.add_argument("--cmap", type=str, default='gray',
                       help="Colormap (gray, viridis, hot, jet, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for saved images")
    args = parser.parse_args()
    
    # Load datacube
    print(f"Loading: {args.input}")
    cube_bhw, key = load_datacube(args.input)
    B, H, W = cube_bhw.shape
    print(f"  Shape: (Bands={B}, Height={H}, Width={W})")
    print(f"  Key: {key}")
    
    dataset_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize based on mode
    if args.all_bands:
        print(f"\nVisualizing all {B} bands...")
        save_dir = os.path.join(args.output_dir, f"{dataset_name}_bands") if args.output_dir else None
        visualize_all_bands(cube_bhw, title=dataset_name, cmap=args.cmap, 
                          save_dir=save_dir, max_cols=5)
    
    elif args.rgb:
        r, g, b = args.rgb
        print(f"\nCreating false-color RGB (R={r}, G={g}, B={b})...")
        save_path = os.path.join(args.output_dir, f"{dataset_name}_RGB_{r}_{g}_{b}.png") if args.output_dir else None
        create_false_color_rgb(cube_bhw, r, g, b, title=dataset_name, save_path=save_path)
    
    else:
        print(f"\nVisualizing band {args.band}...")
        save_path = os.path.join(args.output_dir, f"{dataset_name}_band_{args.band}.png") if args.output_dir else None
        visualize_wavelength(cube_bhw, band_idx=args.band, title=dataset_name, 
                           cmap=args.cmap, save_path=save_path, show_histogram=True)
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()