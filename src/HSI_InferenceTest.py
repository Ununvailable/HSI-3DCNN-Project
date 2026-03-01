import os
import argparse
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime


# -------------------------
# 1) Utils
# -------------------------
def guess_datacube_key(mat_dict):
    """Find the most likely 3D datacube key inside a .mat dict."""
    if "DataCube" in mat_dict:
        return "DataCube"
    
    candidates = []
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            candidates.append((k, v.shape))
    
    if not candidates:
        raise KeyError(f"Cannot find 3D DataCube. Available keys={list(mat_dict.keys())}")
    
    candidates.sort(key=lambda x: np.prod(x[1]), reverse=True)
    return candidates[0][0]


def load_datacube(mat_path, datacube_key=None):
    """Load .mat and return cube in (B, H, W) float32."""
    M = loadmat(mat_path)
    key = datacube_key or guess_datacube_key(M)
    
    cube = np.array(M[key])
    if cube.ndim != 3:
        raise ValueError(f"{mat_path} {key} is not 3D, shape={cube.shape}")
    
    raw_shape = cube.shape
    band_axis = int(np.argmin(raw_shape))
    
    # Convert to (B, H, W)
    if band_axis == 0:
        cube_bhw = cube
    elif band_axis == 1:
        cube_bhw = np.transpose(cube, (1, 0, 2))
    else:
        cube_bhw = np.transpose(cube, (2, 0, 1))
    
    return cube_bhw.astype(np.float32), key, raw_shape, band_axis


def extract_patches(cube_bhw, patch_size=9, stride=9):
    """
    Extract patches from cube.
    Returns: (N, B, P, P) patches and (N, 2) coordinates
    """
    B, H, W = cube_bhw.shape
    r = patch_size // 2
    patches = []
    coords = []
    
    for y in range(r, H - r, stride):
        for x in range(r, W - r, stride):
            patch = cube_bhw[:, y - r:y + r + 1, x - r:x + r + 1]
            patches.append(patch)
            coords.append([y, x])
    
    if len(patches) == 0:
        raise ValueError(f"No patches extracted. Image too small or patch_size/stride too large.")
    
    return np.array(patches, dtype=np.float32), np.array(coords)


def normalize_data(X, method="minmax"):
    """Normalize data using minmax or max method."""
    if method == "minmax":
        X_min, X_max = X.min(), X.max()
        if X_max > X_min:
            return (X - X_min) / (X_max - X_min)
    elif method == "max":
        mx = X.max()
        if mx != 0:
            return X / mx
    return X


def predict_full_image(model, cube_bhw, patch_size=9, stride=9, 
                       normalize="minmax", batch_size=64):
    """
    Run model predictions on entire image.
    Returns classification map (H, W) and prediction probabilities
    """
    B, H, W = cube_bhw.shape
    
    # Extract patches
    print(f"  Extracting patches (patch_size={patch_size}, stride={stride})...")
    patches, coords = extract_patches(cube_bhw, patch_size=patch_size, stride=stride)
    print(f"    → {len(patches)} patches extracted")
    
    # Normalize
    patches = normalize_data(patches, method=normalize)
    patches = patches[..., np.newaxis]  # Add channel dimension
    
    # Predict
    print(f"  Running predictions (batch_size={batch_size})...")
    predictions = model.predict(patches, batch_size=batch_size, verbose=0)
    class_ids = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Create classification map
    class_map = np.zeros((H, W), dtype=np.int32) - 1  # -1 = unclassified
    confidence_map = np.zeros((H, W), dtype=np.float32)
    
    for (y, x), class_id, conf in zip(coords, class_ids, confidences):
        class_map[y, x] = class_id
        confidence_map[y, x] = conf
    
    return class_map, confidence_map, predictions


def visualize_classification(class_map, confidence_map, title, class_names, output_dir, dataset_name):
    """
    Create and save classification visualizations.
    Saves: classification map, confidence map, and combined view
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Classification Map
    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    cmap_class = plt.matplotlib.colors.ListedColormap(colors)
    
    im1 = axes[0].imshow(class_map, cmap=cmap_class, vmin=0, vmax=n_classes-1, interpolation='nearest')
    axes[0].set_title('Classification Map', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    # Add colorbar with class names
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=range(n_classes))
    cbar1.ax.set_yticklabels(class_names)
    
    # 2. Confidence Map
    im2 = axes[1].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('Confidence Map', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Confidence')
    
    # 3. Overlay (classification with confidence as alpha)
    # Normalize confidence for alpha
    alpha_map = np.clip(confidence_map, 0.3, 1.0)  # Min alpha 0.3 for visibility
    
    axes[2].imshow(class_map, cmap=cmap_class, vmin=0, vmax=n_classes-1, 
                   alpha=alpha_map, interpolation='nearest')
    axes[2].set_title('Classification + Confidence', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('X (pixels)')
    axes[2].set_ylabel('Y (pixels)')
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f"{dataset_name}_classification.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    # Save individual high-res classification map
    fig_class = plt.figure(figsize=(12, 10))
    plt.imshow(class_map, cmap=cmap_class, vmin=0, vmax=n_classes-1, interpolation='nearest')
    plt.title(f'{dataset_name} - Classification', fontsize=16, fontweight='bold')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    cbar = plt.colorbar(ticks=range(n_classes), label='Class')
    cbar.ax.set_yticklabels(class_names)
    plt.tight_layout()
    
    output_path_class = os.path.join(output_dir, f"{dataset_name}_classification_only.png")
    plt.savefig(output_path_class, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path_class}")
    plt.close()


def save_statistics(class_map, confidence_map, predictions, class_names, output_dir, dataset_name):
    """Save detailed statistics to text file."""
    output_path = os.path.join(output_dir, f"{dataset_name}_statistics.txt")
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"CLASSIFICATION STATISTICS: {dataset_name}\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Image info
        f.write("Image Information:\n")
        f.write(f"  Size: {class_map.shape[0]} × {class_map.shape[1]} pixels\n")
        f.write(f"  Classified pixels: {(class_map >= 0).sum()}\n")
        f.write(f"  Unclassified pixels: {(class_map < 0).sum()}\n\n")
        
        # Class distribution
        unique_classes, counts = np.unique(class_map[class_map >= 0], return_counts=True)
        total_classified = (class_map >= 0).sum()
        
        f.write("Class Distribution:\n")
        f.write(f"  {'Class':<15s} {'Pixels':>10s} {'Percentage':>12s} {'Avg Confidence':>15s}\n")
        f.write("  " + "-"*55 + "\n")
        
        for cls, count in zip(unique_classes, counts):
            class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            percentage = 100 * count / total_classified
            
            # Calculate average confidence for this class
            mask = (class_map == cls)
            avg_conf = confidence_map[mask].mean() if mask.sum() > 0 else 0
            
            f.write(f"  {class_name:<15s} {count:>10d} {percentage:>11.2f}% {avg_conf:>14.4f}\n")
        
        f.write("\n")
        
        # Overall confidence statistics
        valid_conf = confidence_map[class_map >= 0]
        f.write("Confidence Statistics:\n")
        f.write(f"  Mean:   {valid_conf.mean():.4f}\n")
        f.write(f"  Median: {np.median(valid_conf):.4f}\n")
        f.write(f"  Std:    {valid_conf.std():.4f}\n")
        f.write(f"  Min:    {valid_conf.min():.4f}\n")
        f.write(f"  Max:    {valid_conf.max():.4f}\n\n")
        
        # Prediction distribution (across all classes)
        f.write("Prediction Statistics:\n")
        for i, class_name in enumerate(class_names):
            class_probs = predictions[:, i]
            f.write(f"  {class_name:<15s} - Mean prob: {class_probs.mean():.4f}, "
                   f"Std: {class_probs.std():.4f}\n")
    
    print(f"  ✓ Saved: {output_path}")


def save_classification_map(class_map, confidence_map, output_dir, dataset_name):
    """Save classification and confidence maps as numpy arrays."""
    # Save as .npy for later analysis
    class_map_path = os.path.join(output_dir, f"{dataset_name}_class_map.npy")
    np.save(class_map_path, class_map)
    print(f"  ✓ Saved: {class_map_path}")
    
    conf_map_path = os.path.join(output_dir, f"{dataset_name}_confidence_map.npy")
    np.save(conf_map_path, confidence_map)
    print(f"  ✓ Saved: {conf_map_path}")


# -------------------------
# 2) Main Inference Function
# -------------------------
def run_inference(model_path, input_file, output_base_dir, class_names, 
                  patch_size=9, stride=9, normalize="minmax", batch_size=64):
    """
    Run inference on a single file and save all results.
    """
    # Extract dataset name from filename
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create output directory for this dataset
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    expected_bands = model.input_shape[1]
    print(f"  Model expects {expected_bands} bands")
    
    # Load data
    print(f"Loading data: {input_file}")
    cube_bhw, key, raw_shape, band_axis = load_datacube(input_file)
    print(f"  Key: {key}, Shape: {cube_bhw.shape}")
    
    # Check band compatibility
    actual_bands = cube_bhw.shape[0]
    if actual_bands != expected_bands:
        if actual_bands > expected_bands:
            print(f"  ⚠️  Band mismatch: using first {expected_bands} of {actual_bands} bands")
            cube_bhw = cube_bhw[:expected_bands, :, :]
        else:
            print(f"  ❌ ERROR: Data has only {actual_bands} bands, model needs {expected_bands}")
            return False
    
    # Run prediction
    print(f"Running classification...")
    class_map, confidence_map, predictions = predict_full_image(
        model, cube_bhw,
        patch_size=patch_size,
        stride=stride,
        normalize=normalize,
        batch_size=batch_size
    )
    
    # Print quick stats
    unique_classes, counts = np.unique(class_map[class_map >= 0], return_counts=True)
    print(f"\n  Results:")
    print(f"    Classified pixels: {(class_map >= 0).sum()}")
    print(f"    Average confidence: {confidence_map[class_map >= 0].mean():.4f}")
    print(f"    Classes found: {len(unique_classes)}")
    
    # Save all outputs
    print(f"\n  Saving results to: {output_dir}/")
    save_classification_map(class_map, confidence_map, output_dir, dataset_name)
    save_statistics(class_map, confidence_map, predictions, class_names, output_dir, dataset_name)
    visualize_classification(class_map, confidence_map, 
                            f"Classification Results: {dataset_name}", 
                            class_names, output_dir, dataset_name)
    
    print(f"  ✓ All results saved for {dataset_name}")
    return True


# -------------------------
# 3) Batch Processing
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="HSI Classification Inference")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.keras)")
    parser.add_argument("--input_dir", type=str, default="hsi_datasets/v303",
                       help="Directory containing .mat files")
    parser.add_argument("--input_files", type=str, nargs='+', default=None,
                       help="Specific .mat files to process (overrides input_dir)")
    parser.add_argument("--output_dir", type=str, default="inference_result",
                       help="Base output directory")
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=4,
                       help="Stride for patch extraction (smaller = denser predictions)")
    parser.add_argument("--normalize", type=str, default="minmax",
                       choices=["minmax", "max", "none"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--class_names", type=str, nargs='+',
                       default=["Red", "Green", "Blue", "Paper"],
                       help="Class names for visualization")
    parser.add_argument("--skip_indian_pines", action='store_true',
                       help="Skip Indian Pines dataset (different band count)")
    args = parser.parse_args()
    
    print("="*60)
    print("HSI BATCH CLASSIFICATION INFERENCE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patch size: {args.patch_size}, Stride: {args.stride}")
    print(f"Classes: {args.class_names}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files to process
    if args.input_files:
        files_to_process = args.input_files
    else:
        # Find all .mat files in input directory
        files_to_process = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith('.mat')
        ]
    
    # Filter out Indian Pines if requested
    if args.skip_indian_pines:
        files_to_process = [
            f for f in files_to_process
            if 'indian_pines' not in f.lower()
        ]
    
    print(f"\nFound {len(files_to_process)} file(s) to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] {os.path.basename(input_file)}")
        
        try:
            success = run_inference(
                model_path=args.model,
                input_file=input_file,
                output_base_dir=args.output_dir,
                class_names=args.class_names,
                patch_size=args.patch_size,
                stride=args.stride,
                normalize=args.normalize,
                batch_size=args.batch_size
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(files_to_process)}")
    print(f"Failed: {failed}/{len(files_to_process)}")
    print(f"\nResults saved to: {args.output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()