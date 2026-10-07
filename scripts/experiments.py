"""
Research experiments for BirdSense. Reports, on the same held-out test set:

  1. Split leakage  — random clip-level validation split vs. recording-grouped split.
  2. Baselines      — majority class and logistic regression on the frozen embeddings,
                      vs. the Dense head.
  3. Quantisation   — float32 / float16 / dynamic int8 / full int8 TFLite heads:
                      size, accuracy, agreement with float32, and per-window latency.

Holdout embeddings are cached to models/checkpoints/embeddings_holdout.npz so
BirdNET only runs over the test audio once. Nothing here touches best_model.keras,
label_map.json, eval_report.txt, or models/export/.

Run after extract_embeddings.py and train.py:
  python scripts/experiments.py
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent))
import train
from constants import TOP_K
from evaluate import Embedder, collect_holdout, load_audio, load_label_map, top_k_accuracy

CHECKPOINT_DIR: Path = train.CHECKPOINT_DIR
HOLDOUT_CACHE: Path = CHECKPOINT_DIR / "embeddings_holdout.npz"
REPORT_PATH: Path = CHECKPOINT_DIR / "experiments_report.txt"
JSON_PATH: Path = CHECKPOINT_DIR / "experiments.json"
SEEDS: tuple[int, ...] = (0, 1, 2)  # model-initialisation seeds for the split comparison
LR_MAX_ITER: int = 2000
REPRESENTATIVE_SAMPLES: int = 200  # embeddings used to calibrate full-int8 quantisation
LATENCY_WARMUP: int = 20  # initial invokes excluded from latency timing
SEPARATOR: str = "=" * 70


@dataclass(frozen=True)
class HoldoutData:
    """Cached BirdNET window embeddings for the held-out test files."""

    X: np.ndarray  # (n_windows, 1024) window embeddings
    file_idx: np.ndarray  # (n_windows,) index of the source file for each window
    y_files: np.ndarray  # (n_files,) true class per file
    embed_seconds: float  # wall time BirdNET spent embedding all files


def compute_holdout_embeddings(class_names: list[str], label_to_idx: dict[str, int]) -> HoldoutData:
    """Embed every holdout file with BirdNET, timing the extraction.

    Args:
        class_names: Species names in class-index order.
        label_to_idx: Species name -> class index.

    Returns:
        HoldoutData for all holdout files.
    """
    files, y_files = collect_holdout(class_names, label_to_idx)
    embedder = Embedder()
    chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    elapsed = 0.0
    for i, path in enumerate(files):
        if i % 20 == 0:
            print(f"  embedding holdout {i}/{len(files)}", end="\r", flush=True)
        audio = load_audio(path)
        start = time.perf_counter()
        vectors = embedder.embed(audio)
        elapsed += time.perf_counter() - start
        chunks.append(vectors)
        index_chunks.append(np.full(len(vectors), i, dtype=np.int32))
    print(f"  embedding holdout {len(files)}/{len(files)}")
    return HoldoutData(
        X=np.concatenate(chunks),
        file_idx=np.concatenate(index_chunks),
        y_files=np.array(y_files, dtype=np.int32),
        embed_seconds=elapsed,
    )


def load_holdout(class_names: list[str], label_to_idx: dict[str, int]) -> HoldoutData:
    """Load cached holdout embeddings, computing and caching them if absent.

    Args:
        class_names: Species names in class-index order.
        label_to_idx: Species name -> class index.

    Returns:
        HoldoutData for all holdout files.
    """
    if HOLDOUT_CACHE.exists():
        cached = np.load(HOLDOUT_CACHE)
        return HoldoutData(cached["X"], cached["file_idx"], cached["y_files"], float(cached["embed_seconds"]))
    data = compute_holdout_embeddings(class_names, label_to_idx)
    np.savez(
        HOLDOUT_CACHE, X=data.X, file_idx=data.file_idx, y_files=data.y_files,
        embed_seconds=np.float64(data.embed_seconds),
    )
    return data


def score_files(window_probs: np.ndarray, data: HoldoutData, num_classes: int) -> dict[str, float]:
    """Aggregate per-window probabilities to per-file predictions and score them.

    Ties in the majority vote resolve to the lowest class index, so the majority
    macro-F1 can differ marginally from evaluate.py's arbitrary tie-break.

    Args:
        window_probs: (n_windows, num_classes) class probabilities per holdout window.
        data: Cached holdout embeddings and labels.
        num_classes: Number of species classes.

    Returns:
        Dict with soft_top1, soft_top3, soft_macro_f1, majority_macro_f1.
    """
    n_files = len(data.y_files)
    soft = np.zeros((n_files, num_classes), dtype=np.float32)
    majority = np.zeros(n_files, dtype=np.int64)
    window_pred = window_probs.argmax(axis=1)
    for i in range(n_files):
        rows = data.file_idx == i
        if rows.any():
            soft[i] = window_probs[rows].mean(axis=0)
            majority[i] = np.bincount(window_pred[rows], minlength=num_classes).argmax()
    labels = list(range(num_classes))
    y_true = data.y_files.tolist()
    soft_pred = soft.argmax(axis=1)
    return {
        "soft_top1": float(accuracy_score(y_true, soft_pred)),
        "soft_top3": top_k_accuracy(y_true, soft, TOP_K),
        "soft_macro_f1": float(f1_score(y_true, soft_pred, labels=labels, average="macro", zero_division=0)),
        "majority_macro_f1": float(f1_score(y_true, majority, labels=labels, average="macro", zero_division=0)),
    }


def clip_scores(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Clip-level accuracy and macro-F1 (used for the validation split).

    Args:
        y_true: True class per clip.
        probs: (n_clips, num_classes) predicted probabilities.

    Returns:
        Dict with val_acc and val_macro_f1.
    """
    pred = probs.argmax(axis=1)
    return {
        "val_acc": float(accuracy_score(y_true, pred)),
        "val_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def fit_head(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, seed: int) -> tf.keras.Model:
    """Train the Dense head exactly as train.py does (same hyper-parameters, early stopping on val loss).

    Args:
        X_tr: Training embeddings.
        y_tr: Training labels.
        X_va: Validation embeddings (drives early stopping).
        y_va: Validation labels.
        seed: Model-initialisation seed.

    Returns:
        Trained Keras model with the best-val-loss weights restored.
    """
    tf.keras.utils.set_random_seed(seed)
    model = train.build_model(int(y_tr.max()) + 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(train.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        batch_size=train.BATCH_SIZE,
        epochs=train.EPOCHS,
        class_weight=dict(enumerate(weights)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=train.PATIENCE_STOP, restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=train.PATIENCE_LR, min_lr=1e-7,
            ),
        ],
        verbose=0,
    )
    return model


def mean_std(values: list[float]) -> str:
    """Format a list of floats as 'mean ± std'.

    Args:
        values: Values to summarise.

    Returns:
        Formatted string with 4 decimals.
    """
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def run_split_experiment(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, holdout: HoldoutData, num_classes: int,
) -> dict[str, list[dict[str, float]]]:
    """Compare a random clip-level validation split against a recording-grouped one.

    Args:
        X: Training embeddings.
        y: Training labels.
        groups: Recording id per clip.
        holdout: Cached holdout embeddings.
        num_classes: Number of species classes.

    Returns:
        Mapping split name -> one metrics dict per seed (val_* and test metrics).
    """
    grouped_tr, grouped_va = train.grouped_split_indices(y, groups)
    random_tr, random_va = train_test_split(
        np.arange(len(y)), test_size=len(grouped_va) / len(y), stratify=y, random_state=train.RANDOM_STATE,
    )
    leaked = len(set(groups[random_tr]) & set(groups[random_va]))
    print(f"  random split: {leaked} recordings appear in both train and val")

    results: dict[str, list[dict[str, float]]] = {"random": [], "grouped": []}
    for name, (tr, va) in {"random": (random_tr, random_va), "grouped": (grouped_tr, grouped_va)}.items():
        for seed in SEEDS:
            print(f"  training {name} split, seed {seed}")
            model = fit_head(X[tr], y[tr], X[va], y[va], seed)
            metrics = clip_scores(y[va], model.predict(X[va], verbose=0))
            metrics.update(score_files(model.predict(holdout.X, verbose=0), holdout, num_classes))
            metrics["leaked_recordings"] = float(leaked if name == "random" else 0)
            results[name].append(metrics)
    return results


def run_baselines(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, holdout: HoldoutData, num_classes: int,
) -> dict[str, dict[str, float]]:
    """Fit majority-class and logistic-regression baselines on the grouped split.

    Args:
        X: Training embeddings.
        y: Training labels.
        groups: Recording id per clip.
        holdout: Cached holdout embeddings.
        num_classes: Number of species classes.

    Returns:
        Mapping baseline name -> val and test metrics.
    """
    train_idx, val_idx = train.grouped_split_indices(y, groups)
    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=LR_MAX_ITER, class_weight="balanced"),
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        print(f"  fitting {name}")
        model.fit(X[train_idx], y[train_idx])
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        results[name] = {
            **clip_scores(y[val_idx], _padded(model, classes, X[val_idx], num_classes)),
            **score_files(_padded(model, classes, holdout.X, num_classes), holdout, num_classes),
        }
    return results


def _padded(model: object, classes: np.ndarray, X: np.ndarray, num_classes: int) -> np.ndarray:
    """predict_proba placed into a (n_rows, num_classes) array by class label.

    Args:
        model: Fitted sklearn estimator exposing predict_proba.
        classes: The estimator's class labels, in predict_proba column order.
        X: Embeddings to score.
        num_classes: Total number of species classes.

    Returns:
        (n_rows, num_classes) probability array.
    """
    probs = np.zeros((len(X), num_classes), dtype=np.float32)
    probs[:, classes] = model.predict_proba(X)
    return probs


def convert_variants(model: tf.keras.Model, representative: np.ndarray) -> dict[str, bytes]:
    """Convert a Keras model to TFLite at four precisions.

    Args:
        model: Trained Dense head.
        representative: Embeddings used to calibrate full-int8 quantisation.

    Returns:
        Mapping variant name -> serialised TFLite model.
    """
    def convert(configure) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        configure(converter)
        return converter.convert()

    def to_float16(c: tf.lite.TFLiteConverter) -> None:
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]

    def to_dynamic_int8(c: tf.lite.TFLiteConverter) -> None:
        c.optimizations = [tf.lite.Optimize.DEFAULT]

    def to_full_int8(c: tf.lite.TFLiteConverter) -> None:
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: ([row[None, :].astype(np.float32)] for row in representative)
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    return {
        "float32": convert(lambda c: None),
        "float16": convert(to_float16),
        "int8_dynamic": convert(to_dynamic_int8),
        "int8_full": convert(to_full_int8),
    }


def run_tflite(tflite_bytes: bytes, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run a TFLite model one window at a time on one CPU thread, timing each invoke.

    Args:
        tflite_bytes: Serialised TFLite model.
        X: (n, 1024) float32 embeddings.

    Returns:
        Tuple of (probabilities (n, classes), per-invoke latency in ms after warm-up).
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    outputs: list[np.ndarray] = []
    latencies: list[float] = []
    for row in X.astype(np.float32):
        interpreter.set_tensor(input_index, row[None, :])
        start = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000.0)
        outputs.append(interpreter.get_tensor(output_index)[0].copy())
    return np.stack(outputs), np.array(latencies[LATENCY_WARMUP:])


def run_quantisation(
    model: tf.keras.Model, X: np.ndarray, holdout: HoldoutData, num_classes: int,
) -> dict[str, dict[str, float]]:
    """Compare TFLite precisions on size, test accuracy, float32 agreement and latency.

    Args:
        model: Trained Dense head (best_model.keras).
        X: Training embeddings (source of calibration samples).
        holdout: Cached holdout embeddings.
        num_classes: Number of species classes.

    Returns:
        Mapping variant name -> metrics dict.
    """
    rng = np.random.default_rng(train.RANDOM_STATE)
    representative = X[rng.choice(len(X), size=REPRESENTATIVE_SAMPLES, replace=False)]
    variants = convert_variants(model, representative)
    reference_pred: np.ndarray | None = None
    results: dict[str, dict[str, float]] = {}
    for name, blob in variants.items():
        print(f"  benchmarking {name}")
        probs, latency = run_tflite(blob, holdout.X)
        pred = probs.argmax(axis=1)
        reference_pred = pred if reference_pred is None else reference_pred
        results[name] = {
            "size_kb": len(blob) / 1024.0,
            "agreement_with_float32": float(np.mean(pred == reference_pred)),
            "latency_ms_median": float(np.median(latency)),
            "latency_ms_p95": float(np.percentile(latency, 95)),
            **score_files(probs, holdout, num_classes),
        }
    return results


def format_split_table(results: dict[str, list[dict[str, float]]]) -> str:
    """Render the split-leakage comparison.

    Args:
        results: Output of run_split_experiment.

    Returns:
        Multi-line table.
    """
    lines = [f"{'Split':<9} {'leaked rec.':>11} {'val acc':>16} {'val macro-F1':>16} {'test macro-F1 (soft)':>22}"]
    for name, rows in results.items():
        lines.append(
            f"{name:<9} {int(rows[0]['leaked_recordings']):>11} "
            f"{mean_std([r['val_acc'] for r in rows]):>16} "
            f"{mean_std([r['val_macro_f1'] for r in rows]):>16} "
            f"{mean_std([r['soft_macro_f1'] for r in rows]):>22}"
        )
    return "\n".join(lines)


def format_baseline_table(
    baselines: dict[str, dict[str, float]], head_rows: list[dict[str, float]],
) -> str:
    """Render the baseline comparison, with the Dense head as the last row.

    Args:
        baselines: Output of run_baselines.
        head_rows: Per-seed metrics of the Dense head on the grouped split.

    Returns:
        Multi-line table.
    """
    lines = [f"{'Model':<22} {'val macro-F1':>14} {'test top-1':>11} {'test top-3':>11} {'test macro-F1':>14}"]
    for name, m in baselines.items():
        lines.append(
            f"{name:<22} {m['val_macro_f1']:>14.4f} {m['soft_top1']:>11.4f} {m['soft_top3']:>11.4f} "
            f"{m['soft_macro_f1']:>14.4f}"
        )
    lines.append(
        f"{'dense_head (3 seeds)':<22} {np.mean([r['val_macro_f1'] for r in head_rows]):>14.4f} "
        f"{np.mean([r['soft_top1'] for r in head_rows]):>11.4f} {np.mean([r['soft_top3'] for r in head_rows]):>11.4f} "
        f"{np.mean([r['soft_macro_f1'] for r in head_rows]):>14.4f}"
    )
    return "\n".join(lines)


def format_quant_table(results: dict[str, dict[str, float]], birdnet_ms: float) -> str:
    """Render the quantisation comparison plus BirdNET's per-window cost for context.

    Args:
        results: Output of run_quantisation.
        birdnet_ms: BirdNET embedding time per window in ms (same CPU).

    Returns:
        Multi-line table.
    """
    lines = [
        f"{'Variant':<13} {'size KB':>8} {'test top-1':>11} {'test macro-F1':>14} "
        f"{'agree w/ f32':>13} {'median ms':>10} {'p95 ms':>8}"
    ]
    for name, m in results.items():
        lines.append(
            f"{name:<13} {m['size_kb']:>8.1f} {m['soft_top1']:>11.4f} {m['soft_macro_f1']:>14.4f} "
            f"{m['agreement_with_float32']:>13.4f} {m['latency_ms_median']:>10.3f} {m['latency_ms_p95']:>8.3f}"
        )
    lines.append(f"\nBirdNET embedding (same CPU, includes audio setup): {birdnet_ms:.1f} ms per window")
    return "\n".join(lines)


def main() -> None:
    """Entry point: run the three experiments and save the report."""
    print("BirdSense Experiments\n")
    class_names, label_to_idx = load_label_map(CHECKPOINT_DIR / "label_map.json")
    num_classes = len(class_names)
    X, y, _, paths = train.load_embeddings(train.EMBEDDINGS_PATH)
    groups = train.recording_ids(paths)
    best_model_path = CHECKPOINT_DIR / "best_model.keras"
    if not best_model_path.exists():
        sys.exit(f"[ERROR] {best_model_path} not found. Run train.py first.")

    print("Holdout embeddings...")
    holdout = load_holdout(class_names, label_to_idx)
    n_windows = len(holdout.X)
    print(f"  {len(holdout.y_files)} files, {n_windows} windows\n")

    print("Experiment 1: split leakage")
    split_results = run_split_experiment(X, y, groups, holdout, num_classes)
    print("\nExperiment 2: baselines")
    baselines = run_baselines(X, y, groups, holdout, num_classes)
    print("\nExperiment 3: quantisation")
    model = tf.keras.models.load_model(str(best_model_path))
    quant = run_quantisation(model, X, holdout, num_classes)

    birdnet_ms = holdout.embed_seconds / max(n_windows, 1) * 1000.0
    report = "\n".join([
        SEPARATOR, "1. Validation split: random clips vs. grouped by recording", SEPARATOR,
        f"{len(set(groups))} recordings, {len(y)} clips. Mean ± std over seeds {list(SEEDS)}; "
        f"test = {len(holdout.y_files)}-file holdout.",
        format_split_table(split_results), "",
        SEPARATOR, "2. Baselines (grouped split; test = holdout, soft vote)", SEPARATOR,
        format_baseline_table(baselines, split_results["grouped"]), "",
        SEPARATOR, "3. TFLite quantisation of best_model.keras (1 CPU thread, this machine; head only)", SEPARATOR,
        format_quant_table(quant, birdnet_ms),
    ])
    print("\n" + report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    JSON_PATH.write_text(
        json.dumps({"split": split_results, "baselines": baselines, "quantisation": quant,
                    "birdnet_ms_per_window": birdnet_ms}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {REPORT_PATH}\nSaved: {JSON_PATH}")


if __name__ == "__main__":
    main()
