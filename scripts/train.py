"""
Train a Dense classifier head on cached BirdNET embeddings for Bornean
bird species classification.

BirdNET (via birdnetlib) is used purely as a frozen feature extractor —
run scripts/extract_embeddings.py first to cache a 1024-d embedding per
processed clip. This script only trains the small classification head on
top, so no GPU fine-tuning of a vision backbone is required.

Saves the best model to models/checkpoints/best_model.keras.
Saves label_map.json alongside for evaluate.py and export.py.

Run after extract_embeddings.py:
  python scripts/train.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent))
from constants import NUM_SPECIES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT: Path = Path(__file__).parent.parent
CHECKPOINT_DIR: Path = _ROOT / "models" / "checkpoints"
EMBEDDINGS_PATH: Path = CHECKPOINT_DIR / "embeddings_train.npz"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 1024
BATCH_SIZE: int = 32
EPOCHS: int = 60
LEARNING_RATE: float = 1e-3
GROUP_FOLDS: int = 6  # one fold held out for validation => ~1/6 (~17%) of clips, split by recording
RANDOM_STATE: int = 42
WINDOW_SUFFIX: re.Pattern[str] = re.compile(r"_\d+$")  # trailing window index added by preprocess.py
DROPOUT_RATE: float = 0.4
PATIENCE_STOP: int = 10
PATIENCE_LR: int = 4


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load cached BirdNET embeddings produced by extract_embeddings.py.

    Args:
        path: Path to embeddings_train.npz.

    Returns:
        Tuple of (X, y, class_names, paths) where paths[i] is the source clip of row i.
    """
    if not path.exists():
        sys.exit(f"[ERROR] {path} not found. Run extract_embeddings.py first.")
    data = np.load(path, allow_pickle=True)
    if "paths" not in data.files:
        sys.exit(f"[ERROR] {path} has no 'paths' array. Re-run extract_embeddings.py.")
    return data["X"], data["y"], data["class_names"].tolist(), data["paths"]


def recording_ids(paths: np.ndarray) -> np.ndarray:
    """Map each clip path to the id of the source recording it was cut from.

    preprocess.py names clips <recording_stem>_<window_index>.wav inside a
    species folder, so the recording id is species folder + stem without the
    trailing window index.

    Args:
        paths: Clip paths such as dataset/processed/<species>/<stem>_<NN>.wav.

    Returns:
        String array of recording ids, one per clip.
    """
    ids = [f"{Path(p).parent.name}/{WINDOW_SUFFIX.sub('', Path(p).stem)}" for p in paths]
    return np.array(ids)


def grouped_split_indices(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split clips into train/val so no recording appears on both sides.

    Uses one fold of a stratified, shuffled group k-fold, giving a validation
    share of about 1/GROUP_FOLDS while keeping species proportions similar.

    Args:
        y: Integer species labels per clip.
        groups: Recording id per clip (see recording_ids).

    Returns:
        Tuple of (train_indices, val_indices).
    """
    splitter = StratifiedGroupKFold(n_splits=GROUP_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    overlap = set(groups[train_idx]) & set(groups[val_idx])
    if overlap:
        sys.exit(f"[ERROR] {len(overlap)} recordings leak across train/val split.")
    return train_idx, val_idx


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a small Dense classification head on top of BirdNET embeddings.

    Args:
        num_classes: Number of output bird species classes.

    Returns:
        Uncompiled Keras model.
    """
    inputs = tf.keras.Input(shape=(EMBEDDING_DIM,), name="embedding")
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
    return tf.keras.Model(inputs, outputs, name="birdsense")


def make_callbacks(ckpt_path: Path) -> list[Any]:
    """Build standard training callbacks.

    Args:
        ckpt_path: Where to save the best model checkpoint.

    Returns:
        List of Keras callbacks.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE_LR,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(CHECKPOINT_DIR / "logs")),
    ]


def save_training_history(history: dict[str, list[float]], out_dir: Path) -> None:
    """Save per-epoch metrics as JSON and plot loss/accuracy curves as PNG.

    Args:
        history: Keras History.history dict (loss, accuracy, val_loss, val_accuracy, ...).
        out_dir: Directory to write training_history.json and training_curves.png into.
    """
    history_path = out_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    epochs = range(1, len(history["loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "accuracy", "Accuracy")):
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="val")
        ax.axvline(best_epoch, color="grey", linestyle="--", label=f"best val_loss (epoch {best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    curves_path = out_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)

    print(f"\nEpochs run: {len(history['loss'])}  |  best val_loss at epoch {best_epoch}")
    print(f"Final train acc {history['accuracy'][-1]:.4f}  |  final val acc {history['val_accuracy'][-1]:.4f}")
    print(f"History saved: {history_path}\nCurves saved : {curves_path}")


def main() -> None:
    """Entry point: train the Dense head on cached BirdNET embeddings."""
    print("BirdSense Trainer — Dense head on BirdNET embeddings")
    print(f"TensorFlow {tf.__version__}  |  GPU: {tf.config.list_physical_devices('GPU')}")
    print()

    X, y, class_names, paths = load_embeddings(EMBEDDINGS_PATH)
    num_classes = len(class_names)
    print(f"Species: {num_classes}  |  Total clips: {len(X)}")

    if num_classes != NUM_SPECIES:
        print(
            f"[WARN] NUM_SPECIES in constants.py is {NUM_SPECIES} but "
            f"{num_classes} classes found. Update constants.py before export."
        )

    groups = recording_ids(paths)
    train_idx, val_idx = grouped_split_indices(y, groups)
    X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
    print(f"Recordings: {len(set(groups))}  |  Train: {len(y_train)} clips / {len(set(groups[train_idx]))} recordings"
          f"  |  Val: {len(y_val)} clips / {len(set(groups[val_idx]))} recordings  (no recording in both)")

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))
    print(f"Class weight range: {weights.min():.3f} — {weights.max():.3f}\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "best_model.keras"

    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=make_callbacks(ckpt_path),
        verbose=1,
    )
    save_training_history(history.history, CHECKPOINT_DIR)

    # Save label map for evaluate.py and export.py
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    label_map_path = CHECKPOINT_DIR / "label_map.json"
    with label_map_path.open("w") as f:
        json.dump({"class_names": class_names, "label_to_idx": label_to_idx}, f, indent=2)

    print(f"\nLabel map saved : {label_map_path}")
    print(f"Best model saved: {ckpt_path}")
    print("Done. Run evaluate.py to check macro-F1 before exporting.")


if __name__ == "__main__":
    main()
