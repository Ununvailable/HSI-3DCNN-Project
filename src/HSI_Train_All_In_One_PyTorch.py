# -*- coding: utf-8 -*-
"""
HSI_Train_All_In_One_PyTorch.py
================================
Main training pipeline for HSI 3D-CNN classification (PyTorch).
Migrated from HSI_Train_All_In_One.py (TensorFlow/Keras).

Classes:
    TrainingConfig        — Hyperparameters, paths, and class definitions
    Li2017Config          — TrainingConfig subclass for Li (2017) Indian Pines replication
    HSIDataset            — Datacube loading, patch extraction, and dataset assembly
    HSITorchDataset       — PyTorch Dataset wrapper (channels-first conversion)
    HSIModelFactory       — Static model builders (simple, li2017)
    EarlyStopping         — Patience-based early stopping with best-weight restore
    ModelTrainer          — Training loop, evaluation, checkpointing, and plotting

--model flag
------------
    simple   : Spectral-first 3D CNN on v303 RGBP dataset (default)
    li2017   : Li (2017) replication — pixel-based patches on Indian Pines,
               16 classes, patch=5, 50% val split, SGD optimizer, no normalization
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# 1. TrainingConfig
# ─────────────────────────────────────────────

class TrainingConfig:
    """
    Holds all hyperparameters, file paths, and class definitions.
    Constructed directly from parsed CLI arguments.

    Attributes
    ----------
    model         : str   — Architecture choice ('simple' | 'li2017')
    patch_size    : int   — Spatial patch size (pixels)
    stride        : int   — Patch extraction stride
    epochs        : int   — Maximum training epochs
    batch_size    : int   — Mini-batch size
    test_size     : float — Validation fraction
    seed          : int   — Random seed for reproducibility
    datacube_key  : str   — Optional explicit .mat variable name
    normalize     : str   — Normalization method ('minmax' | 'max' | 'none')
    save          : str   — Output checkpoint filename (.pth)
    output_dir    : str   — Root directory for saved runs
    device        : str   — 'cuda' if available, else 'cpu'
    files         : dict  — {class_name: path_to_mat}
    label_map     : dict  — {class_name: integer_label}
    num_classes   : int   — Total number of classes

    Properties
    ----------
    run_name      : str   — hsi_{arch}_p{patch}-s{stride}-e{epochs}-b{batch}_{norm}
    run_dir       : str   — output_dir / run_name / timestamp
    model_filename: str   — run_name + .pth
    """

    DEFAULT_FILES = {
        "Red":   "hsi_datasets/v303/Red.mat",
        "Green": "hsi_datasets/v303/Green.mat",
        "Blue":  "hsi_datasets/v303/Blue.mat",
        "Paper": "hsi_datasets/v303/Paper.mat",
    }
    DEFAULT_LABEL_MAP = {"Red": 0, "Green": 1, "Blue": 2, "Paper": 3}

    def __init__(self, args: argparse.Namespace):
        self.model        = args.model
        self.patch_size   = args.patch_size
        self.stride       = args.stride
        self.epochs       = args.epochs
        self.batch_size   = args.batch_size
        self.test_size    = args.test_size
        self.seed         = args.seed
        self.datacube_key = args.datacube_key
        self.normalize    = None if args.normalize == "none" else args.normalize
        self.save         = args.save
        self.output_dir   = args.output_dir
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

        self.files       = self.DEFAULT_FILES
        self.label_map   = self.DEFAULT_LABEL_MAP
        self.num_classes = len(self.label_map)

        self._timestamp  = datetime.now().strftime("%Y%m%d-%H%M")

    @property
    def run_name(self) -> str:
        """hsi_{arch}_p{patch}-s{stride}-e{epochs}-b{batch}_{normalize}"""
        norm = self.normalize or "none"
        return (
            f"hsi_{self.model}_p{self.patch_size}-s{self.stride}"
            f"-e{self.epochs}-b{self.batch_size}_{norm}"
        )

    @property
    def run_dir(self) -> str:
        """Full path: output_dir / run_name / timestamp"""
        return os.path.join(self.output_dir, self.run_name, self._timestamp)

    @property
    def model_filename(self) -> str:
        """run_name + .pth"""
        return f"{self.run_name}.pth"

    def __repr__(self) -> str:
        lines = ["TrainingConfig("]
        for k, v in self.__dict__.items():
            if k not in ("files", "label_map"):
                lines.append(f"  {k}={v!r}")
        lines.append(")")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 1b. Li2017Config  (Indian Pines subclass)
# ─────────────────────────────────────────────

class Li2017Config(TrainingConfig):
    """
    TrainingConfig subclass that faithfully replicates the Li (2017) experimental
    setup on the Indian Pines dataset (Remote Sensing, 9(1):67).

    Overrides
    ---------
    patch_size  : 5     — 5×5 spatial window (best per Section 4.2.3)
    stride      : 1     — pixel-based, every labelled pixel becomes a sample
    test_size   : 0.5   — 50/50 train/val split (Table 3 of paper)
    normalize   : None  — no normalization applied (paper uses raw bands)
    epochs      : 200   — CLI-overridable
    batch_size  : 20    — 20 samples per iteration per paper Section 3
    output_dir  : 'models/li2017'
    num_classes : 16    — Indian Pines classes 1–16; class 0 excluded
    files       : points to indian_pines_corrected.mat + indian_pines_gt.mat

    Optimizer   : SGD, momentum=0.9, weight_decay=0.0005 (Equations 5–6 of paper)
                  Fixed learning rate — no ReduceLROnPlateau

    All run_name / run_dir / model_filename properties are inherited unchanged.
    """

    INDIAN_PINES_CORRECTED = "hsi_datasets/hsi_researches/indian_pines_corrected.mat"
    INDIAN_PINES_GT        = "hsi_datasets/hsi_researches/indian_pines_gt.mat"

    # Class names for Indian Pines (indices 1–16 mapped to 0–15 internally)
    INDIAN_PINES_CLASSES = [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
        "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
        "Soybean-clean", "Wheat", "Woods",
        "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
    ]

    def __init__(self, args: argparse.Namespace):
        # Call parent to inherit device, timestamp, and repr
        super().__init__(args)

        # Override all Li(2017) specific settings
        self.model        = "li2017"   # architecture is fixed for this config
        self.patch_size   = 5          # 5×5 per paper Section 4.2.3
        self.stride       = 1          # pixel-based extraction
        self.test_size    = 0.5        # 50/50 split per Table 3
        self.normalize    = None       # no normalization per paper
        self.epochs       = args.epochs      # allow CLI override; default 200
        self.batch_size   = args.batch_size  # allow CLI override; default 20
        self.output_dir   = "models/li2017"
        self.num_classes  = 16

        self.files = {
            "corrected": self.INDIAN_PINES_CORRECTED,
            "gt":        self.INDIAN_PINES_GT,
        }
        self.label_map = {
            cls: i for i, cls in enumerate(self.INDIAN_PINES_CLASSES)
        }


# ─────────────────────────────────────────────
# 2. HSIDataset  (NumPy level — framework-agnostic)
# ─────────────────────────────────────────────

class HSIDataset:
    """
    Loads and assembles a labelled patch dataset from multiple .mat datacubes.

    Two build modes, dispatched automatically from config.model:
      - Default  : per-class .mat files, non-overlapping patches
      - 'li2017' : Indian Pines corrected + GT, pixel-based overlapping patches,
                   class 0 (background) excluded, reflect-padded borders

    Usage
    -----
        ds = HSIDataset(config)
        ds.build()
        X_train, X_val, y_train, y_val = ds.split()

    Attributes (populated after build())
    -------------------------------------
    X : np.ndarray — shape (N, B, P, P, 1), float32, normalized
    y : np.ndarray — shape (N,), int32
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    # ── public ──────────────────────────────

    def build(self) -> None:
        """Dispatch to the appropriate build method based on config.model."""
        if self.config.model == "li2017":
            self._build_indian_pines()
        else:
            self._build_per_class_files()

    def split(self) -> tuple:
        """
        Stratified train/val split.

        Returns
        -------
        (X_train, X_val, y_train, y_val)
        """
        if self.X is None:
            raise RuntimeError("Call build() before split().")

        return train_test_split(
            self.X, self.y,
            test_size=self.config.test_size,
            stratify=self.y,
            random_state=self.config.seed,
        )

    @property
    def input_shape(self) -> tuple:
        """
        Per-sample input shape for model construction.
        Returns (1, B, P, P) — channels-first, matching PyTorch convention.
        """
        if self.X is None:
            raise RuntimeError("Call build() before accessing input_shape.")
        _, B, P, _, _ = self.X.shape
        return (1, B, P, P)

    # ── private build methods ────────────────

    def _build_per_class_files(self) -> None:
        """
        Default build path — loads one .mat file per class, extracts
        non-overlapping patches, and assembles X/y.
        """
        cfg = self.config
        X_list, y_list = [], []

        for cls_name, path in cfg.files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset file not found: {path}")

            cube_bhw, key_used, raw_shape, band_axis = self._load_datacube(path)
            patches = self._extract_patches(cube_bhw, cfg.patch_size, cfg.stride)

            print(
                f"  [{cls_name}] key='{key_used}' raw={raw_shape}, "
                f"band_axis={band_axis} -> cube(B,H,W)={cube_bhw.shape}, "
                f"patches={patches.shape}"
            )

            X_list.append(patches)
            y_list.append(
                np.full((len(patches),), cfg.label_map[cls_name], dtype=np.int32)
            )

        X = np.concatenate(X_list, axis=0)   # (N, B, P, P)
        y = np.concatenate(y_list, axis=0)   # (N,)
        X = X[..., np.newaxis]               # (N, B, P, P, 1)
        X = self._normalize(X, method=cfg.normalize or "minmax")

        print(f"\n  Dataset assembled — X: {X.shape}, y: {y.shape}, "
              f"classes: {np.unique(y)}")
        self.X = X
        self.y = y

    def _build_indian_pines(self) -> None:
        """
        Li (2017) build path — pixel-based patch extraction from Indian Pines.

        Steps
        -----
        1. Load corrected datacube (145, 145, 200) → transpose to (200, 145, 145)
        2. Load GT map (145, 145), integer labels 0–16
        3. Reflect-pad the cube so every labelled pixel (including borders)
           can produce a full patch_size × patch_size neighbourhood
        4. For each pixel where GT > 0 (exclude background class 0):
             - Extract the patch centred on that pixel
             - Assign the GT label shifted by -1 (so classes become 0–15)
        5. Normalize and store X/y
        """
        cfg = self.config
        corrected_path = cfg.files["corrected"]
        gt_path        = cfg.files["gt"]

        for p in (corrected_path, gt_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Indian Pines file not found: {p}")

        # ── load corrected cube ──────────────
        M_cube = loadmat(corrected_path)
        cube_key = self._guess_datacube_key(M_cube)
        cube_raw = np.array(M_cube[cube_key])   # (145, 145, 200)

        if cube_raw.ndim != 3:
            raise ValueError(
                f"Expected 3D corrected cube, got shape {cube_raw.shape}"
            )

        # Indian Pines corrected is stored (H, W, B) → transpose to (B, H, W)
        cube_bhw = np.transpose(cube_raw, (2, 0, 1)).astype(np.float32)
        B, H, W  = cube_bhw.shape
        print(f"  Corrected cube : key='{cube_key}', raw={cube_raw.shape} → (B,H,W)={cube_bhw.shape}")

        # ── load GT map ──────────────────────
        M_gt   = loadmat(gt_path)
        gt_key = next(
            k for k in M_gt.keys()
            if not k.startswith("__") and isinstance(M_gt[k], np.ndarray)
        )
        gt_map = np.array(M_gt[gt_key]).astype(np.int32)   # (145, 145)
        if gt_map.ndim != 2:
            raise ValueError(f"Expected 2D GT map, got shape {gt_map.shape}")
        print(f"  GT map         : key='{gt_key}', shape={gt_map.shape}, "
              f"unique labels={np.unique(gt_map).tolist()}")

        # ── reflect-pad cube for border pixels ──
        r   = cfg.patch_size // 2
        pad = ((0, 0), (r, r), (r, r))   # pad H and W only, not B
        cube_padded = np.pad(cube_bhw, pad, mode="reflect")

        # ── pixel-based patch extraction ─────
        X_list, y_list = [], []
        labelled_pixels = np.argwhere(gt_map > 0)   # exclude background

        for (row, col) in labelled_pixels:
            # In padded cube, pixel (row, col) is at (row+r, col+r)
            pr, pc  = row + r, col + r
            patch   = cube_padded[:, pr - r:pr + r + 1, pc - r:pc + r + 1]
            label   = gt_map[row, col] - 1   # shift 1–16 → 0–15

            X_list.append(patch)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)   # (N, B, P, P)
        y = np.array(y_list, dtype=np.int32)      # (N,)
        X = X[..., np.newaxis]                    # (N, B, P, P, 1)
        X = self._normalize(X, method=cfg.normalize or "minmax")

        print(f"\n  Indian Pines dataset assembled — "
              f"X: {X.shape}, y: {y.shape}, classes: {np.unique(y)}")
        print(f"  Labelled pixels: {len(labelled_pixels)} "
              f"(background excluded: {int((gt_map == 0).sum())})")

        self.X = X
        self.y = y

    # ── private helpers ──────────────────────

    def _load_datacube(self, mat_path: str) -> tuple:
        """
        Load .mat and return cube in (B, H, W) float32.
        Assumes saved convention is (Y, X, B) — transposes to (B, Y, X).
        """
        M   = loadmat(mat_path)
        key = self.config.datacube_key or self._guess_datacube_key(M)

        cube = np.array(M[key])
        if cube.ndim != 3:
            raise ValueError(
                f"{mat_path} — '{key}' is not 3D, shape={cube.shape}"
            )

        raw_shape = cube.shape          # (Y, X, B) as saved by HSI_Conversion
        cube_bhw  = np.transpose(cube, (2, 0, 1))   # → (B, Y, X)
        band_axis = 2                   # documented, no longer guessed

        return cube_bhw.astype(np.float32), key, raw_shape, band_axis

    @staticmethod
    def _guess_datacube_key(mat_dict: dict) -> str:
        """Infer the datacube variable name from a loaded .mat dictionary."""
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
                f"No 3D DataCube found. Available keys: {list(mat_dict.keys())}"
            )

        candidates.sort(key=lambda x: np.prod(x[1]), reverse=True)
        return candidates[0][0]

    @staticmethod
    def _extract_patches(cube_bhw: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
        """Extract (N, B, P, P) patches from a (B, H, W) datacube."""
        B, H, W = cube_bhw.shape
        r = patch_size // 2

        # Guard: warn if spatial coverage is very limited
        valid_y = len(range(r, H - r, stride))
        if valid_y < 5:
            print(
                f"  ⚠  Very few valid Y positions ({valid_y}) — "
                f"H={H} is small relative to patch_size={patch_size}. "
                f"Consider reducing patch_size."
            )

        patches = []
        for y in range(r, H - r, stride):
            for x in range(r, W - r, stride):
                patches.append(cube_bhw[:, y - r:y + r + 1, x - r:x + r + 1])

        if not patches:
            raise ValueError(
                f"No patches extracted. Cube=(B={B}, H={H}, W={W}), "
                f"patch_size={patch_size}, stride={stride}. "
                f"Reduce patch_size to at most {H - 1}."
            )

        return np.array(patches, dtype=np.float32)

    @staticmethod
    def _normalize(X: np.ndarray, method: str) -> np.ndarray:
        if method == "minmax":
            X_min, X_max = X.min(), X.max()
            if X_max > X_min:
                return (X - X_min) / (X_max - X_min)
        elif method == "max":
            mx = X.max()
            if mx != 0:
                return X / mx
        return X


# ─────────────────────────────────────────────
# 3. HSITorchDataset  (PyTorch Dataset wrapper)
# ─────────────────────────────────────────────

class HSITorchDataset(Dataset):
    """
    Wraps NumPy arrays into a PyTorch Dataset.

    Converts (N, B, P, P, 1) channels-last arrays to
    (N, 1, B, P, P) channels-first tensors expected by nn.Conv3d.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # (N, B, P, P, 1) -> (N, 1, B, P, P)
        self.X = torch.from_numpy(X).permute(0, 4, 1, 2, 3).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 4. HSIModelFactory
# ─────────────────────────────────────────────

class HSIModelFactory:
    """
    Static factory for HSI 3D-CNN architectures.

    Layer order mirrors the TensorFlow script:
        Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d
    Output is raw logits (no softmax); use nn.CrossEntropyLoss.
    """

    @staticmethod
    def build(architecture: str, input_shape: tuple, num_classes: int) -> nn.Module:
        """
        Parameters
        ----------
        architecture : 'simple' | 'li2017'
        input_shape  : (1, B, P, P)  — channels-first
        num_classes  : int
        """
        builders = {
            "simple": HSIModelFactory._Simple3DCNN,
            "li2017": HSIModelFactory._Li2017,
        }
        if architecture not in builders:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Choose from: {list(builders.keys())}"
            )
        return builders[architecture](input_shape, num_classes)

    # ── inner model classes ──────────────────

    class _Simple3DCNN(nn.Module):
        """
        Spectral-first 3D CNN — mirrors updated TF 'simple' model.

        Stage 1: Conv(7,1,1) — spectral kernel only
        Stage 2: Conv(5,1,1) — spectral kernel only
        Stage 3: Conv(3,3,3) — joint spectral-spatial

        MaxPool is spectral-only (4,1,1) throughout to preserve spatial dims.
        """

        def __init__(self, input_shape: tuple, num_classes: int):
            super().__init__()

            # Conv -> BN -> ReLU -> Pool  (consistent with TF script)
            self.features = nn.Sequential(
                # Stage 1 — spectral kernel (7,1,1), pool spectral only
                nn.Conv3d(1, 16, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d((4, 1, 1)),

                # Stage 2 — spectral kernel (5,1,1), pool spectral only
                nn.Conv3d(16, 32, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d((4, 1, 1)),

                # Stage 3 — joint spectral-spatial (3,3,3), pool spectral only
                nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d((4, 1, 1)),
            )

            # auto-compute flattened size via dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                flat  = self.features(dummy).numel()

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    class _Li2017(nn.Module):
        """
        Li (2017) 3D-CNN — faithful replication of Remote Sensing 9(1):67.

        Architecture (Table 8, Indian Pines):
            Input : 1 × B × 5 × 5  (channels-first)
            C1    : 2 kernels, 3×3×7, stride=1, no padding → (2, B-6, 3, 3)
            C2    : 4 kernels, 3×3×3, stride=1, no padding → (8, B-8, 1, 1)
            F1    : fc_units nodes (128 for Indian Pines)
            Out   : num_classes logits

        No pooling layers — explicitly stated in paper Section 2.2.
        No BatchNorm — not in the original paper.
        Optimizer: SGD (set in ModelTrainer.train() when Li2017Config detected).
        """

        def __init__(self, input_shape: tuple, num_classes: int, fc_units: int = 128):
            super().__init__()

            # Valid convolution — no padding, consistent with paper equations
            # Kernel order is (D, H, W) = (spectral, spatial, spatial)
            # Paper's "3×3×7" = spatial 3×3, spectral depth 7 → PyTorch (7, 3, 3)
            # Paper's "3×3×3" = spatial 3×3, spectral depth 3 → PyTorch (3, 3, 3)
            self.conv = nn.Sequential(
                nn.Conv3d(1, 2, kernel_size=(7, 3, 3), padding=0),
                nn.ReLU(),
                nn.Conv3d(2, 4, kernel_size=(3, 3, 3), padding=0),
                nn.ReLU(),
            )

            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                flat  = self.conv(dummy).numel()

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, fc_units),
                nn.ReLU(),
                nn.Linear(fc_units, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.conv(x))


# ─────────────────────────────────────────────
# 5. EarlyStopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Patience-based early stopping with best-weight restore.

    Parameters
    ----------
    patience : int — epochs to wait after last improvement (default 20)
    """

    def __init__(self, patience: int = 20):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = np.inf
        self.best_state = None
        self.stop       = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model: nn.Module) -> None:
        """Load best weights back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────
# 6. ModelTrainer
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    Handles model construction, training loop, evaluation,
    checkpoint saving (with config metadata), and curve plotting.

    Usage
    -----
        trainer = ModelTrainer(config, dataset)
        trainer.build_model()
        trainer.train()
        trainer.evaluate()
        trainer.plot_curves()
        trainer.save()
    """

    def __init__(self, config: TrainingConfig, dataset: HSIDataset):
        self.config  = config
        self.dataset = dataset
        self.model: nn.Module | None = None

        self._train_start: datetime | None = None
        self._train_losses: list[float] = []
        self._val_losses:   list[float] = []
        self._train_accs:   list[float] = []
        self._val_accs:     list[float] = []

        self._train_loader: DataLoader | None = None
        self._val_loader:   DataLoader | None = None

    # ── public ──────────────────────────────

    def build_model(self) -> None:
        """Construct model, move to device, and print parameter count."""
        cfg = self.config
        # Li2017Config fixes architecture to 'li2017' regardless of --model flag
        architecture = "li2017" if isinstance(cfg, Li2017Config) else cfg.model
        self.model = HSIModelFactory.build(
            architecture=architecture,
            input_shape=self.dataset.input_shape,
            num_classes=cfg.num_classes,
        )
        self.model.to(cfg.device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"  Model: {architecture}  |  Device: {cfg.device}  |  "
            f"Trainable params: {total_params:,}"
        )

    def train(self) -> None:
        """
        Split data, build loaders, run training loop with early stopping.

        Optimizer is dispatched based on config type:
          - Li2017Config : SGD, momentum=0.9, weight_decay=0.0005, fixed lr=0.01
                           (Equations 5–6 of Li 2017 paper; no LR scheduler)
          - All others   : Adam lr=1e-3 with ReduceLROnPlateau
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before train().")

        self._train_start = datetime.now()
        cfg = self.config

        X_train, X_val, y_train, y_val = self.dataset.split()

        self._train_loader = DataLoader(
            HSITorchDataset(X_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        self._val_loader = DataLoader(
            HSITorchDataset(X_val, y_val),
            batch_size=cfg.batch_size,
        )

        # ── optimizer dispatch ───────────────────────────────────────────
        if isinstance(cfg, Li2017Config):
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=0.0005,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=0.5
            )
            print("  Optimizer: SGD (momentum=0.9, wd=0.0005, lr=0.01, StepLR ×0.5/100ep) — Li (2017)")
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=True,
            )

        loss_fn = nn.CrossEntropyLoss()
        early   = EarlyStopping(patience=1000)

        for epoch in range(cfg.epochs):
            train_loss, train_acc = self._run_epoch(
                self._train_loader, optimizer, loss_fn, training=True
            )
            val_loss, val_acc = self._run_epoch(
                self._val_loader, optimizer=None, loss_fn=loss_fn, training=False
            )

            if scheduler is not None:
                if isinstance(cfg, Li2017Config):
                    scheduler.step()          # StepLR — unconditional per epoch
                else:
                    scheduler.step(val_loss)  # ReduceLROnPlateau — val loss driven

            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)
            self._train_accs.append(train_acc)
            self._val_accs.append(val_acc)

            print(
                f"  Epoch {epoch + 1:>4}/{cfg.epochs} | "
                f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}"
            )

            early.step(val_loss, self.model)
            if early.stop:
                print(f"  Early stopping triggered at epoch {epoch + 1}.")
                break

        early.restore(self.model)
        print(f"  Best val loss: {early.best_loss:.4f}")

    def evaluate(self) -> dict:
        """
        Evaluate on the validation set using the restored best weights.

        Returns
        -------
        dict with 'loss' and 'accuracy'
        """
        if self.model is None or self._val_loader is None:
            raise RuntimeError("Call train() before evaluate().")

        loss_fn = nn.CrossEntropyLoss()
        val_loss, val_acc = self._run_epoch(
            self._val_loader, optimizer=None, loss_fn=loss_fn, training=False
        )
        print(f"\n  Validation — accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")
        return {"loss": val_loss, "accuracy": val_acc}

    def plot_curves(self) -> None:
        """
        Plot loss and accuracy as side-by-side subplots and save to run_dir.
        """
        epochs = range(1, len(self._train_losses) + 1)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

        ax_loss.plot(epochs, self._train_losses, label="Train Loss")
        ax_loss.plot(epochs, self._val_losses,   label="Val Loss")
        ax_loss.set_title("Loss Curve")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        ax_acc.plot(epochs, self._train_accs, label="Train Acc")
        ax_acc.plot(epochs, self._val_accs,   label="Val Acc")
        ax_acc.set_title("Accuracy Curve")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True)

        fig.suptitle(f"Training Curves — {self.config.run_name}", fontsize=13)
        plt.tight_layout()

        os.makedirs(self.config.run_dir, exist_ok=True)
        plot_path = os.path.join(self.config.run_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"  Curves saved → {plot_path}")

    def save(self) -> str:
        """
        Save full checkpoint and metadata.txt into the structured run directory.

        Checkpoint keys
        ---------------
        model_state_dict : OrderedDict
        config           : dict — all TrainingConfig fields (excluding files/label_map)
                           includes 'bands' for inference compatibility
        train_losses     : list[float]
        val_losses       : list[float]
        train_accs       : list[float]
        val_accs         : list[float]

        Returns
        -------
        Absolute path to the saved .pth file.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        run_dir = self.config.run_dir
        os.makedirs(run_dir, exist_ok=True)

        # ── checkpoint ──────────────────────────────────────────
        config_meta = {
            k: v for k, v in self.config.__dict__.items()
            if k not in ("files", "label_map")
        }
        config_meta["bands"] = self.dataset.input_shape[1]

        model_path = os.path.join(run_dir, self.config.model_filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config":           config_meta,
                "train_losses":     self._train_losses,
                "val_losses":       self._val_losses,
                "train_accs":       self._train_accs,
                "val_accs":         self._val_accs,
            },
            model_path,
        )
        print(f"  Checkpoint saved → {model_path}")

        # ── metadata.txt ────────────────────────────────────────
        metrics  = self.evaluate()
        duration = (datetime.now() - self._train_start).seconds

        meta_path = os.path.join(run_dir, "metadata.txt")
        with open(meta_path, "w") as f:
            f.write("=" * 55 + "\n")
            f.write("HSI TRAINING RUN METADATA\n")
            f.write("=" * 55 + "\n")
            f.write(f"Timestamp      : {self.config._timestamp}\n")
            f.write(f"PyTorch        : {torch.__version__}\n")
            f.write(f"CUDA           : {torch.cuda.is_available()} "
                    f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})\n")
            f.write(f"Python         : {sys.version.split()[0]}\n")
            f.write("\n--- Hyperparameters ---\n")
            for attr in (
                "model", "patch_size", "stride", "epochs",
                "batch_size", "test_size", "seed", "normalize", "datacube_key",
            ):
                f.write(f"  {attr:<15s}: {getattr(self.config, attr)}\n")
            f.write("\n--- Dataset ---\n")
            for cls, path in self.config.files.items():
                f.write(f"  {cls:<10s}: {path}\n")
            f.write("\n--- Results ---\n")
            f.write(f"  val_accuracy   : {metrics['accuracy']:.4f}\n")
            f.write(f"  val_loss       : {metrics['loss']:.4f}\n")
            f.write(f"  training_time  : {duration}s\n")
            f.write("=" * 55 + "\n")

        print(f"  Metadata saved → {meta_path}")
        return model_path

    # ── private ──────────────────────────────

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer | None,
        loss_fn: nn.Module,
        training: bool,
    ) -> tuple[float, float]:
        """Run one full pass over a DataLoader. Returns (mean_loss, accuracy)."""
        self.model.train(training)
        device = self.config.device

        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)

                logits = self.model(Xb)
                loss   = loss_fn(logits, yb)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * Xb.size(0)
                correct    += logits.argmax(dim=1).eq(yb).sum().item()
                total      += yb.size(0)

        return total_loss / total, correct / total


# ─────────────────────────────────────────────
# 7. CLI + main
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HSI 3D-CNN Training Pipeline (PyTorch)")
    parser.add_argument("--model",        type=str,   default="simple",
                        choices=["simple", "li2017"],
                        help=(
                            "'simple'  : spectral-first CNN on v303 RGBP dataset\n"
                            "'li2017'  : Li (2017) replication on Indian Pines "
                            "(patch=25, 16 classes, 10%% val split)"
                        ))
    parser.add_argument("--patch_size",   type=int,   default=3,
                        help="Ignored when --model li2017 (fixed to 25)")
    parser.add_argument("--stride",       type=int,   default=1,
                        help="Ignored when --model li2017 (fixed to 1)")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=8)    
    parser.add_argument("--test_size",    type=float, default=0.2,
                        help="Ignored when --model li2017 (fixed to 0.1)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--datacube_key", type=str,   default=None,
                        help="Explicit .mat variable name; auto-detected if omitted.")
    parser.add_argument("--normalize",    type=str,   default="minmax",
                        choices=["minmax", "max", "none"])
    parser.add_argument("--save",         type=str,   default="hsi_model_default.pth",
                        help="Fallback filename (overridden by auto-generated run_name).")
    parser.add_argument("--output_dir",   type=str,   default="models",
                        help="Ignored when --model li2017 (fixed to models/li2017)")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    print("=" * 60)
    print("HSI 3D-CNN TRAINING PIPELINE  (PyTorch)")
    print("=" * 60)
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 1. Config — dispatch to Li2017Config when --model li2017
    if args.model == "li2017":
        config = Li2017Config(args)
        print("\n  ▶ Li (2017) replication mode — Indian Pines dataset")
        print(f"    patch_size={config.patch_size}, test_size={config.test_size}, "
              f"num_classes={config.num_classes}, normalize={config.normalize}, "
              f"output_dir='{config.output_dir}'")
    else:
        config = TrainingConfig(args)
    print(config)

    # 2. Dataset
    print("\n[1/3] Building dataset...")
    dataset = HSIDataset(config)
    dataset.build()

    # 3. Model
    print("\n[2/3] Building model...")
    trainer = ModelTrainer(config, dataset)
    trainer.build_model()

    # 4. Train → evaluate → plot → save
    print("\n[3/3] Training...")
    trainer.train()
    trainer.evaluate()
    trainer.plot_curves()
    trainer.save()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()