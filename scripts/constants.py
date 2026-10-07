##Shared constants for the BirdSense training pipeline.

SAMPLE_RATE: int = 16000          # Hz — all audio resampled to this
DURATION: int = 5                 # seconds — fixed clip length for model input
N_MELS: int = 128                 # mel filterbank bins
HOP_LENGTH: int = 512             # STFT hop size
N_FFT: int = 1024                 # STFT window size
NUM_SPECIES: int = 15             # output classes — update once species list is finalised (target: 25–30)
TARGET_F1: float = 0.80           # minimum macro-F1 required before exporting
SILENCE_THRESHOLD: float = 0.01   # RMS threshold for noise gate; windows below this are dropped
BIRDNET_SAMPLE_RATE: int = 48000  # Hz — BirdNET's native input rate (evaluate.py SNR test)
SNR_LEVELS_DB: tuple[int, ...] = (20, 10, 5)  # simulated field noise: quiet, moderate, loud
SNR_TARGET_F1: dict[int, float] = {20: 0.80, 10: 0.65}  # minimum macro-F1 per SNR level (5 dB: report only)
SNR_NOISE_SEED: int = 42          # fixed seed so SNR results are reproducible
TOP_K: int = 3                    # top-k accuracy reported by evaluate.py (matches the web UI)
