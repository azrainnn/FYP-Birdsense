"""
Evaluate the trained classifier on the held-out test set.

For each MP3 in dataset/test_holdout/<species>/, extracts a BirdNET
embedding per internal 3-second window (via birdnetlib), classifies each
window with the trained Dense head, and aggregates per file.

Reports:
  - per-species precision / recall / F1 and macro-F1 (majority vote — the export gate)
  - top-1 and top-k accuracy using soft voting (mean softmax over windows, as in the web API)
  - confusion matrix (text in the report + confusion_matrix.png)
  - SNR field test: white Gaussian noise added to the holdout audio at each
    SNR_LEVELS_DB level, with accuracy and macro-F1 per level vs SNR_TARGET_F1

Saves models/checkpoints/eval_report.txt (export.py reads the [PASS] marker),
confusion_matrix.png, and eval_metrics.json.

Run from project root:
  python scripts/evaluate.py
"""

import json
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
import birdnetlib.analyzer as _birdnetlib_analyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from constants import (
    BIRDNET_SAMPLE_RATE,
    SNR_LEVELS_DB,
    SNR_NOISE_SEED,
    SNR_TARGET_F1,
    TARGET_F1,
    TOP_K,
)

# TensorFlow >= 2.17 prunes the intermediate embedding tensor unless all
# tensors are explicitly preserved, which breaks birdnetlib's embedding
# extraction with "Tensor data is null" on every call.
# See: https://github.com/joeweiss/birdnetlib/issues/125
_original_tflite_interpreter = _birdnetlib_analyzer.tflite.Interpreter


def _patched_tflite_interpreter(*args, **kwargs):
    """Force experimental_preserve_all_tensors=True on every TFLite Interpreter."""
    kwargs.setdefault("experimental_preserve_all_tensors", True)
    return _original_tflite_interpreter(*args, **kwargs)


_birdnetlib_analyzer.tflite.Interpreter = _patched_tflite_interpreter

_ROOT: Path = Path(__file__).parent.parent
HOLDOUT_DIR: Path = _ROOT / "dataset" / "test_holdout"
CHECKPOINT_DIR: Path = _ROOT / "models" / "checkpoints"
CLEAN_LABEL: str = "clean"
EMBEDDING_DIM: int = 1024
RECOVER_AFTER_FAILURES: int = 3  # consecutive failures that signal a corrupted TFLite interpreter


class Embedder:
    """BirdNET embedding extractor that recovers from a corrupted shared interpreter."""

    def __init__(self) -> None:
        """Create the underlying BirdNET Analyzer."""
        self._analyzer = Analyzer()
        self._consecutive_failures = 0

    def _try_embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Embed audio once; return None if birdnetlib raises.

        Args:
            audio: Mono float32 waveform at BIRDNET_SAMPLE_RATE.

        Returns:
            Array of shape (n_windows, EMBEDDING_DIM), possibly with zero rows, or None on error.
        """
        try:
            recording = RecordingBuffer(self._analyzer, audio, BIRDNET_SAMPLE_RATE)
            recording.extract_embeddings()
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            return None
        windows = [w["embeddings"] for w in recording.embeddings]
        return np.array(windows, dtype=np.float32).reshape(-1, EMBEDDING_DIM)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Embed audio, reinitialising the analyzer after repeated failures.

        Args:
            audio: Mono float32 waveform at BIRDNET_SAMPLE_RATE.

        Returns:
            Array of shape (n_windows, EMBEDDING_DIM); zero rows if extraction failed.
        """
        vectors = self._try_embed(audio)
        if vectors is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= RECOVER_AFTER_FAILURES:
                print("\n  [RECOVER] Reinitializing BirdNET analyzer after repeated failures...")
                self._analyzer = Analyzer()
                self._consecutive_failures = 0
                vectors = self._try_embed(audio)
        else:
            self._consecutive_failures = 0
        return vectors if vectors is not None else np.empty((0, EMBEDDING_DIM), dtype=np.float32)


def load_label_map(path: Path) -> tuple[list[str], dict[str, int]]:
    """Load class names and label-to-index map saved by train.py.

    Args:
        path: Path to label_map.json.

    Returns:
        Tuple of (class_names, label_to_idx).
    """
    if not path.exists():
        sys.exit(f"[ERROR] {path} not found. Run train.py first.")
    with path.open() as f:
        data = json.load(f)
    return data["class_names"], data["label_to_idx"]


def load_audio(path: Path) -> np.ndarray:
    """Load an audio file as mono float32 at BirdNET's sample rate.

    Args:
        path: Path to an audio file readable by soundfile.

    Returns:
        1-D float32 waveform at BIRDNET_SAMPLE_RATE.
    """
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != BIRDNET_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=BIRDNET_SAMPLE_RATE)
    return mono.astype(np.float32)


def add_noise_at_snr(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of audio with white Gaussian noise added at the given SNR.

    Args:
        audio: Mono waveform.
        snr_db: Target signal-to-noise ratio in dB.
        rng: Random generator (seeded for reproducibility).

    Returns:
        Noisy float32 waveform of the same length.
    """
    signal_rms = float(np.sqrt(np.mean(np.square(audio)))) or 1e-8
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = rng.normal(0.0, noise_rms, size=audio.shape)
    return (audio + noise).astype(np.float32)


def collect_holdout(class_names: list[str], label_to_idx: dict[str, int]) -> tuple[list[Path], list[int]]:
    """List holdout MP3s and their integer labels.

    Args:
        class_names: Species names in class-index order.
        label_to_idx: Species name -> class index.

    Returns:
        Tuple of (files, labels).
    """
    files: list[Path] = []
    labels: list[int] = []
    for species in class_names:
        species_dir = HOLDOUT_DIR / species
        if not species_dir.exists():
            continue
        for mp3 in sorted(species_dir.glob("*.mp3")):
            files.append(mp3)
            labels.append(label_to_idx[species])
    return files, labels


def predict_condition(
    model: tf.keras.Model,
    embedder: Embedder,
    audios: list[np.ndarray],
    snr_db: int | None,
    rng: np.random.Generator,
    num_classes: int,
) -> tuple[list[int], np.ndarray, int]:
    """Classify every file at one noise condition.

    Args:
        model: Trained Dense head.
        embedder: Shared BirdNET embedder.
        audios: Clean waveforms, one per holdout file.
        snr_db: SNR in dB to simulate, or None for clean audio.
        rng: Random generator for noise.
        num_classes: Number of species classes.

    Returns:
        Tuple of (majority-vote predictions, soft-vote probabilities of shape
        (n_files, num_classes), count of files that yielded no embeddings).
    """
    label = CLEAN_LABEL if snr_db is None else f"{snr_db} dB"
    votes: list[int] = []
    mean_probs = np.zeros((len(audios), num_classes), dtype=np.float32)
    empty = 0
    for i, audio in enumerate(audios):
        if i % 20 == 0:
            print(f"  [{label}] {i}/{len(audios)}", end="\r", flush=True)
        noisy = audio if snr_db is None else add_noise_at_snr(audio, snr_db, rng)
        vectors = embedder.embed(noisy)
        if len(vectors) == 0:
            empty += 1
            votes.append(0)
            continue
        probs = model(vectors, training=False).numpy()
        mean_probs[i] = probs.mean(axis=0)
        window_votes = np.argmax(probs, axis=1).tolist()
        votes.append(max(set(window_votes), key=window_votes.count))
    print(f"  [{label}] {len(audios)}/{len(audios)}")
    return votes, mean_probs, empty


def top_k_accuracy(y_true: list[int], probs: np.ndarray, k: int) -> float:
    """Fraction of files whose true class is among the k highest-probability classes.

    Args:
        y_true: True class indices.
        probs: Probabilities of shape (n_files, num_classes).
        k: Number of top classes to consider.

    Returns:
        Top-k accuracy in [0, 1].
    """
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([t in row for t, row in zip(y_true, top_k)]))


def format_confusion_matrix(cm: np.ndarray, names: list[str]) -> str:
    """Render a confusion matrix as text, with a legend mapping index to species.

    Args:
        cm: Confusion matrix (rows = true, columns = predicted).
        names: Species names in class-index order.

    Returns:
        Multi-line string.
    """
    width = max(3, len(str(int(cm.max()))) + 1)
    header = "true\\pred " + "".join(f"{i:>{width}}" for i in range(len(names)))
    rows = [f"{i:>9} " + "".join(f"{v:>{width}}" for v in row) for i, row in enumerate(cm)]
    legend = [f"  {i:>2} = {n}" for i, n in enumerate(names)]
    return "\n".join([header, *rows, "", "Index legend:", *legend])


def save_confusion_matrix_png(cm: np.ndarray, names: list[str], path: Path) -> None:
    """Save a row-normalised confusion matrix heatmap with raw counts annotated.

    Args:
        cm: Confusion matrix (rows = true, columns = predicted).
        names: Species names in class-index order.
        path: Output PNG path.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)), labels=names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), labels=names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — held-out test set (colour = row-normalised recall, text = count)")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            if cm[r, c]:
                colour = "white" if normalised[r, c] > 0.5 else "black"
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=7, color=colour)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_snr_section(
    clean_accuracy: float,
    clean_f1: float,
    snr_rows: list[dict[str, float | int | str | None]],
) -> str:
    """Render the SNR field-test table, clean audio first as the reference.

    Uses MET / NOT MET (never PASS / FAIL) so export.py's [PASS] check is
    driven only by the main macro-F1 gate.

    Args:
        clean_accuracy: Accuracy on clean audio.
        clean_f1: Macro-F1 on clean audio.
        snr_rows: One dict per condition with keys condition, accuracy, macro_f1, target.

    Returns:
        Multi-line string.
    """
    lines = [
        f"{'Condition':<10} {'Accuracy':>9} {'Macro-F1':>9}  Target",
        f"{CLEAN_LABEL:<10} {clean_accuracy:>9.4f} {clean_f1:>9.4f}  (reference)",
    ]
    for row in snr_rows:
        target = row["target"]
        if target is None:
            verdict = "report only"
        else:
            verdict = f">= {target:.2f} {'MET' if row['macro_f1'] >= target else 'NOT MET'}"
        lines.append(f"{row['condition']:<10} {row['accuracy']:>9.4f} {row['macro_f1']:>9.4f}  {verdict}")
    return "\n".join(lines)


def run_snr_test(
    model: tf.keras.Model,
    embedder: Embedder,
    audios: list[np.ndarray],
    y_true: list[int],
    num_classes: int,
) -> list[dict[str, float | int | str | None]]:
    """Evaluate every SNR_LEVELS_DB level with reproducible noise.

    Args:
        model: Trained Dense head.
        embedder: Shared BirdNET embedder.
        audios: Clean holdout waveforms.
        y_true: True class indices.
        num_classes: Number of species classes.

    Returns:
        One dict per SNR level with condition, accuracy, macro_f1 and target.
    """
    rng = np.random.default_rng(SNR_NOISE_SEED)
    rows: list[dict[str, float | int | str | None]] = []
    for snr_db in SNR_LEVELS_DB:
        pred, _, _ = predict_condition(model, embedder, audios, snr_db, rng, num_classes)
        rows.append({
            "condition": f"{snr_db} dB",
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, labels=list(range(num_classes)), average="macro", zero_division=0),
            "target": SNR_TARGET_F1.get(snr_db),
        })
    return rows


def build_gate_message(macro_f1: float) -> str:
    """Build the export-gate line. It carries the only [PASS]/[FAIL] marker in the report.

    Args:
        macro_f1: Clean-audio majority-vote macro-F1.

    Returns:
        Gate message string.
    """
    if macro_f1 >= TARGET_F1:
        return f"[PASS] F1 {macro_f1:.4f} >= {TARGET_F1} — safe to run export.py."
    return (
        f"[FAIL] F1 {macro_f1:.4f} < {TARGET_F1} — do not export. "
        "Consider: more training data, data augmentation, or a larger classifier head."
    )


def main() -> None:
    """Entry point: evaluate on test_holdout/ and save the full report."""
    print("BirdSense Evaluator\n")

    class_names, label_to_idx = load_label_map(CHECKPOINT_DIR / "label_map.json")
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    model_path = CHECKPOINT_DIR / "best_model.keras"
    if not model_path.exists():
        sys.exit(f"[ERROR] {model_path} not found. Run train.py first.")
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    if not HOLDOUT_DIR.exists():
        sys.exit(f"[ERROR] {HOLDOUT_DIR} does not exist. Run split_holdout.py first.")

    files, y_true = collect_holdout(class_names, label_to_idx)
    print(f"Holdout files: {len(files)}\n")
    audios = [load_audio(f) for f in files]

    embedder = Embedder()
    labels = list(range(num_classes))
    rng = np.random.default_rng(SNR_NOISE_SEED)
    clean_pred, clean_probs, clean_empty = predict_condition(model, embedder, audios, None, rng, num_classes)
    snr_rows = run_snr_test(model, embedder, audios, y_true, num_classes)
    print()

    report = classification_report(
        y_true, clean_pred, labels=labels, target_names=class_names, digits=3, zero_division=0,
    )
    macro_f1 = f1_score(y_true, clean_pred, labels=labels, average="macro", zero_division=0)
    clean_accuracy = accuracy_score(y_true, clean_pred)

    soft_pred = np.argmax(clean_probs, axis=1).tolist()
    soft_top1 = accuracy_score(y_true, soft_pred)
    soft_topk = top_k_accuracy(y_true, clean_probs, TOP_K)
    soft_f1 = f1_score(y_true, soft_pred, labels=labels, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, clean_pred, labels=labels)
    cm_png = CHECKPOINT_DIR / "confusion_matrix.png"
    save_confusion_matrix_png(cm, class_names, cm_png)

    separator = "=" * 70
    report_text = "\n".join([
        separator,
        "Classification Report — Held-out Test Set (majority vote over BirdNET windows)",
        separator,
        report,
        f"Macro F1 : {macro_f1:.4f}",
        f"Target   : {TARGET_F1}",
        f"Files with no embeddings (counted as class 0): {clean_empty}",
        "",
        separator,
        "Soft-vote metrics (mean softmax over windows — matches the web API)",
        separator,
        f"Top-1 accuracy : {soft_top1:.4f}",
        f"Top-{TOP_K} accuracy : {soft_topk:.4f}",
        f"Macro F1       : {soft_f1:.4f}",
        "",
        separator,
        "Confusion Matrix (rows = true, columns = predicted; majority vote)",
        separator,
        format_confusion_matrix(cm, class_names),
        f"(image: {cm_png.name})",
        "",
        separator,
        "SNR Field Test (white Gaussian noise added to holdout audio; majority vote)",
        separator,
        format_snr_section(clean_accuracy, macro_f1, snr_rows),
        "",
        build_gate_message(macro_f1),
    ])
    print(report_text)

    report_path = CHECKPOINT_DIR / "eval_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    metrics = {
        "macro_f1_majority": macro_f1,
        "soft_vote": {"top1": soft_top1, f"top{TOP_K}": soft_topk, "macro_f1": soft_f1},
        "snr": snr_rows,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    (CHECKPOINT_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nReport saved: {report_path}\nConfusion matrix image: {cm_png}")


if __name__ == "__main__":
    main()
