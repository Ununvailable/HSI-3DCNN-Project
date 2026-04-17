# -*- coding: utf-8 -*-
"""
HSI-Train_All_In_One.py
========================
Main training pipeline for HSI 3D-CNN classification.

Classes:
    TrainingConfig   — Hyperparameters, paths, and class definitions
    HSIDataset       — Datacube loading, patch extraction, and dataset assembly
    HSIModelFactory  — Static model builders (simple, li2017)
    ModelTrainer     — Compilation, training, evaluation, and model saving
"""

import datetime
import os
import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
    save          : str   — Output model filename
    files         : dict  — {class_name: path_to_mat}
    label_map     : dict  — {class_name: integer_label}
    num_classes   : int   — Total number of classes
    output_dir    : str   — Directory for saved model
    """

    # Default dataset definition — edit here to add/remove classes
    DEFAULT_FILES = {
        "Red":   "hsi_datasets/v303/Red.mat",
        "Green": "hsi_datasets/v303/Green.mat",
        "Blue":  "hsi_datasets/v303/Blue.mat",
        "Paper": "hsi_datasets/v303/Paper.mat",
    }
    DEFAULT_LABEL_MAP = {"Red": 0, "Green": 1, "Blue": 2, "Paper": 3}

    # DEFAULT_FILES = {
    #     "IndianPinesCorrected": "hsi_datasets/hsi_researches/indian_pines_corrected.mat",
    #     "IndianPinesGT": "hsi_datasets/hsi_researches/indian_pines_gt.mat",
    # }
    # DEFAULT_LABEL_MAP = {
    #     "IndianPinesCorrected": 0,
    #     "IndianPinesGT": 1,
    # }

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

        self.files      = self.DEFAULT_FILES
        self.label_map  = self.DEFAULT_LABEL_MAP
        self.num_classes = len(self.label_map)

        # For timestamped run directories
        from datetime import datetime
        self._timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    @property
    def run_name(self) -> str:
        """Option B directory name: hsi_{arch}_p{patch}-s{stride}-e{epochs}-b{batch}_{normalize}"""
        norm = self.normalize or "none"
        return f"hsi_{self.model}_p{self.patch_size}-s{self.stride}-e{self.epochs}-b{self.batch_size}_{norm}"

    @property
    def run_dir(self) -> str:
        """Full path: output_dir / run_name / timestamp"""
        return os.path.join(self.output_dir, self.run_name, self._timestamp)

    @property
    def model_filename(self) -> str:
        """hsi_{arch}_p{patch}-s{stride}-e{epochs}-b{batch}_{normalize}.keras"""
        return f"{self.run_name}.keras"
    
    def __repr__(self) -> str:
        lines = [f"TrainingConfig("]
        for k, v in self.__dict__.items():
            if k not in ("files", "label_map"):
                lines.append(f"  {k}={v!r}")
        lines.append(")")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 2. HSIDataset
# ─────────────────────────────────────────────

class HSIDataset:
    """
    Loads and assembles a labelled patch dataset from multiple .mat datacubes.

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
        """Load all class files, extract patches, normalize, and store X/y."""
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
        """Returns per-sample input shape (B, P, P, 1) for model construction."""
        if self.X is None:
            raise RuntimeError("Call build() before accessing input_shape.")
        return self.X.shape[1:]

    # ── private helpers ──────────────────────

    def _load_datacube(self, mat_path: str) -> tuple:
        """Load .mat and return cube in (B, H, W) float32."""
        M = loadmat(mat_path)
        key = self.config.datacube_key or self._guess_datacube_key(M)

        cube = np.array(M[key])
        if cube.ndim != 3:
            raise ValueError(
                f"{mat_path} — '{key}' is not 3D, shape={cube.shape}"
            )

        # raw_shape = cube.shape
        # band_axis = int(np.argmin(raw_shape))

        # if band_axis == 0:
        #     cube_bhw = cube
        # elif band_axis == 1:
        #     cube_bhw = np.transpose(cube, (1, 0, 2))
        # else:
        #     cube_bhw = np.transpose(cube, (2, 0, 1))

        raw_shape = cube.shape          # (Y, X, B) as saved by HSI_Conversion
        cube_bhw  = np.transpose(cube, (2, 0, 1))  # → (B, Y, X)
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
        B, H, W = cube_bhw.shape
        r = patch_size // 2

        # Guard: warn if spatial coverage is very limited
        valid_y = len(range(r, H - r, stride))
        valid_x = len(range(r, W - r, stride))
        if valid_y < 5:
            print(f"  ⚠  Very few valid Y positions ({valid_y}) — "
                f"H={H} is small relative to patch_size={patch_size}. "
                f"Consider reducing patch_size.")

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
# 3. HSIModelFactory
# ─────────────────────────────────────────────

class HSIModelFactory:
    """
    Static factory for HSI 3D-CNN architectures.
    Model definitions are intentionally kept as functions (unchanged from original).
    """

    @staticmethod
    def build(architecture: str, input_shape: tuple, num_classes: int) -> keras.Model:
        """
        Dispatch to the requested architecture.

        Parameters
        ----------
        architecture : 'simple' | 'li2017'
        input_shape  : (B, P, P, 1)
        num_classes  : int
        """
        builders = {
            "simple": HSIModelFactory._build_simple,
            "li2017": HSIModelFactory._build_li2017,
        }
        if architecture not in builders:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Choose from: {list(builders.keys())}"
            )
        return builders[architecture](input_shape, num_classes)

    # @staticmethod
    # def _build_simple(input_shape: tuple, num_classes: int) -> keras.Model:
    #     """Practical 3D CNN baseline."""
    #     model = keras.Sequential([
    #         keras.Input(shape=input_shape),

    #         layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
    #         layers.BatchNormalization(),
    #         layers.MaxPooling3D((1, 2, 2)),

    #         layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    #         layers.BatchNormalization(),
    #         layers.MaxPooling3D((1, 2, 2)),

    #         layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    #         layers.BatchNormalization(),
    #         layers.MaxPooling3D((1, 2, 2)),

    #         layers.Flatten(),
    #         layers.Dense(128, activation='relu'),
    #         layers.Dropout(0.6),
    #         layers.Dense(num_classes, activation='softmax'),
    #     ])
    #     return model

    @staticmethod
    def _build_simple(input_shape: tuple, num_classes: int) -> keras.Model:
        model = keras.Sequential([
            keras.Input(shape=input_shape),

            layers.Conv3D(16, (7, 1, 1), activation='relu', padding='same'),  # spectral kernel
            layers.BatchNormalization(),
            layers.MaxPooling3D((4, 1, 1)),   # pool spectral only: 1232→308

            layers.Conv3D(32, (5, 1, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling3D((4, 1, 1)),   # 308→77

            layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),  # joint spectral-spatial
            layers.BatchNormalization(),
            layers.MaxPooling3D((4, 1, 1)),   # 77→19

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])
        return model

    @staticmethod
    def _build_li2017(input_shape: tuple, num_classes: int, fc_units: int = 128) -> keras.Model:
        """Li (2017) style: two Conv3D layers then FC."""
        inp = layers.Input(shape=input_shape)
        x = layers.Conv3D(2,  (3, 3, 7), activation="relu", padding="same")(inp)
        x = layers.Conv3D(4,  (3, 3, 3), activation="relu", padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(fc_units, activation="relu")(x)
        out = layers.Dense(num_classes, activation="softmax")(x)
        return keras.Model(inputs=inp, outputs=out)


# ─────────────────────────────────────────────
# 4. ModelTrainer
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    Handles model compilation, training, evaluation, and persistence.

    Usage
    -----
        trainer = ModelTrainer(config, dataset)
        trainer.build_model()
        trainer.train()
        trainer.evaluate()
        trainer.save()
    """

    def __init__(self, config: TrainingConfig, dataset: HSIDataset):
        self.config  = config
        self.dataset = dataset
        self.model: keras.Model | None = None
        self.history = None

        # Populated by train()
        self._X_train = self._X_val = None
        self._y_train = self._y_val = None

    def build_model(self) -> None:
        """Construct and summarize the model."""
        self.model = HSIModelFactory.build(
            architecture=self.config.model,
            input_shape=self.dataset.input_shape,
            num_classes=self.config.num_classes,
        )
        self.model.summary()

    def train(self) -> None:
        """Split data, compile, and fit the model."""
        from datetime import datetime
        self._train_start = datetime.now()

        if self.model is None:
            raise RuntimeError("Call build_model() before train().")

        cfg = self.config

        self._X_train, self._X_val, self._y_train, self._y_val = (
            self.dataset.split()
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            self._X_train, self._y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_data=(self._X_val, self._y_val),
            callbacks=callbacks,
            verbose=1,
        )

    def evaluate(self) -> dict:
        """
        Evaluate on the validation set.

        Returns
        -------
        dict with 'loss' and 'accuracy'
        """
        if self.model is None or self._X_val is None:
            raise RuntimeError("Call train() before evaluate().")

        loss, acc = self.model.evaluate(self._X_val, self._y_val, verbose=0)
        print(f"\nValidation — accuracy: {acc:.4f}, loss: {loss:.4f}")
        return {"loss": loss, "accuracy": acc}

    def save(self) -> str:
        """Save model and metadata.txt into the structured run directory."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        import sys, time, tensorflow as tf
        from datetime import datetime

        run_dir = self.config.run_dir
        os.makedirs(run_dir, exist_ok=True)

        # ── model ──────────────────────────────────────────────
        model_path = os.path.join(run_dir, self.config.model_filename)
        self.model.save(model_path)
        print(f"Model saved → {model_path}")

        # ── metadata.txt ───────────────────────────────────────
        metrics  = self.evaluate()   # returns {"loss": ..., "accuracy": ...}
        duration = datetime.now() - self._train_start  # timedelta object
        duration = duration.seconds

        meta_path = os.path.join(run_dir, "metadata.txt")
        with open(meta_path, "w") as f:
            f.write("=" * 55 + "\n")
            f.write("HSI TRAINING RUN METADATA\n")
            f.write("=" * 55 + "\n")
            f.write(f"Timestamp      : {self.config._timestamp}\n")
            f.write(f"TensorFlow     : {tf.__version__}\n")
            f.write(f"Python         : {sys.version.split()[0]}\n")
            f.write("\n--- Hyperparameters ---\n")
            for attr in ("model", "patch_size", "stride", "epochs",
                        "batch_size", "test_size", "seed", "normalize",
                        "datacube_key"):
                f.write(f"  {attr:<15s}: {getattr(self.config, attr)}\n")
            f.write("\n--- Dataset ---\n")
            for cls, path in self.config.files.items():
                f.write(f"  {cls:<10s}: {path}\n")
            f.write("\n--- Results ---\n")
            f.write(f"  val_accuracy   : {metrics['accuracy']:.4f}\n")
            f.write(f"  val_loss       : {metrics['loss']:.4f}\n")
            f.write(f"  training_time  : {duration}s\n")
            f.write("=" * 55 + "\n")

        print(f"Metadata saved → {meta_path}")
        return model_path


# ─────────────────────────────────────────────
# 5. CLI + main
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HSI 3D-CNN Training Pipeline")
    parser.add_argument("--model",        type=str,   default="simple",
                        choices=["simple", "li2017"])
    parser.add_argument("--patch_size",   type=int,   default=3)
    parser.add_argument("--stride",       type=int,   default=1)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--datacube_key", type=str,   default=None,
                        help="Explicit .mat variable name; auto-detected if omitted.")
    parser.add_argument("--normalize",    type=str,   default="minmax",
                        choices=["minmax", "max", "none"])
    parser.add_argument("--save",         type=str,   default="hsi_model_default.keras")
    parser.add_argument("--output_dir",   type=str,   default="models")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    print("=" * 60)
    print("HSI 3D-CNN TRAINING PIPELINE")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")

    # Seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 1. Config
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

    # 4. Train + evaluate + save
    print("\n[3/3] Training...")
    trainer.train()
    trainer.evaluate()
    trainer.save()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()