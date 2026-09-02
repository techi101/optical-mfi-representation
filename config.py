"""
config.py
=========
Every "magic number" in this project lives here, in ONE place.

Why a separate config file?
    If a number appears in two files and you change it in only one, your
    results silently become wrong. Keeping constants here means the CNN,
    the baselines and the plots are guaranteed to be talking about the
    same experiment.

Everything below is deliberately explained in plain English, because you
will be asked "why did you choose that value?" in the viva.
"""

import os

# ---------------------------------------------------------------------------
# 0. REPRODUCIBILITY
# ---------------------------------------------------------------------------
# A "seed" fixes the starting point of the random number generator. Random
# numbers on a computer are not truly random -- they are a fixed sequence that
# LOOKS random. Giving it the same seed replays the same sequence, so you get
# byte-for-byte identical results every time you re-run the project.
SEED = 42

# For the multi-seed experiments that produce error bars in the paper.
# Running the same experiment 5 times with different seeds lets us report
# "96.8 +/- 0.2 %" instead of a single number a reviewer cannot trust.
SEED_LIST = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# 1. WHAT WE ARE CLASSIFYING
# ---------------------------------------------------------------------------
# Five modulation formats spanning the range used in real elastic optical
# networks, from the most robust (QPSK) to the most spectrally efficient
# (64-QAM). The number is the constellation order M = points per symbol.
#
# Why five and not three? Published MFI papers typically evaluate 4-6
# formats. Three is enough to demonstrate the idea but makes the task
# artificially easy -- and a reviewer will notice. Adding 8-QAM and 32-QAM
# also adds two NON-SQUARE constellations, which is a genuinely harder and
# more realistic test than three square grids.
MODULATIONS = {
    "QPSK":  4,     # square (2x2)
    "8QAM":  8,     # circular / star -- NOT a square grid
    "16QAM": 16,    # square (4x4)
    "32QAM": 32,    # cross-shaped -- NOT a square grid
    "64QAM": 64,    # square (8x8)
}
CLASS_NAMES = list(MODULATIONS.keys())
N_CLASSES = len(CLASS_NAMES)

# Bits per symbol, used when discussing spectral efficiency in the paper.
BITS_PER_SYMBOL = {"QPSK": 2, "8QAM": 3, "16QAM": 4, "32QAM": 5, "64QAM": 6}


# ---------------------------------------------------------------------------
# 2. THE NOISE SWEEP
# ---------------------------------------------------------------------------
# OSNR = Optical Signal-to-Noise Ratio, in decibels (dB).
# High OSNR = clean signal. Low OSNR = noisy signal.
OSNR_DB_MIN = 5
OSNR_DB_MAX = 25
OSNR_DB_STEP = 1
OSNR_LIST = list(range(OSNR_DB_MIN, OSNR_DB_MAX + 1, OSNR_DB_STEP))
N_OSNR = len(OSNR_LIST)


# ---------------------------------------------------------------------------
# 3. THE OPTICAL LINK
# ---------------------------------------------------------------------------
SYMBOL_RATE = 32e9          # 32 GBaud -- a standard lab/commercial rate
REF_BANDWIDTH = 12.5e9      # 12.5 GHz = the 0.1 nm reference bandwidth at
                            # 1550 nm that OSNR is ALWAYS quoted in
N_POL = 1                   # 1 = single polarisation
WAVELENGTH = 1550e-9        # m, the standard C-band telecom wavelength


# ---------------------------------------------------------------------------
# 4. TRANSCEIVER IMPAIRMENTS
# ---------------------------------------------------------------------------
# Real receivers are not ideal. These model the impairments that survive the
# receiver's DSP and actually reach the constellation diagram. Each can be
# switched off independently, which is what makes the robustness study in the
# paper possible.
#
# "IDEAL"    -- AWGN only. The optimistic case most papers report.
# "REALISTIC"-- what a good commercial coherent receiver actually delivers.
# "SEVERE"   -- a stressed link; used to test where the models break.

IMPAIRMENT_PROFILES = {
    "ideal": dict(
        residual_phase_deg=0.0,      # perfect carrier phase recovery
        laser_linewidth_hz=0.0,      # no laser phase noise
        freq_offset_hz=0.0,          # perfect frequency locking
        iq_gain_imbalance_db=0.0,    # perfect I/Q balance
        iq_phase_skew_deg=0.0,       # perfect 90-degree hybrid
        residual_cd_ps_nm=0.0,       # perfect dispersion compensation
    ),
    "realistic": dict(
        residual_phase_deg=1.0,      # CPE estimation noise, ~1 deg RMS
        laser_linewidth_hz=100e3,    # 100 kHz -- a good external-cavity laser
        freq_offset_hz=10e6,         # 10 MHz left after coarse frequency
                                     # recovery. This is applied BEFORE carrier
                                     # phase recovery, which removes most of
                                     # it; what survives is the drift within
                                     # one 64-symbol CPE block, about +-4 deg.
        iq_gain_imbalance_db=0.3,    # 0.3 dB mismatch between I and Q arms
        iq_phase_skew_deg=2.0,       # 2 deg error in the 90-degree hybrid
        residual_cd_ps_nm=10.0,      # 10 ps/nm left after CD compensation
    ),
    "severe": dict(
        residual_phase_deg=2.0,
        laser_linewidth_hz=500e3,    # a cheaper DFB laser
        freq_offset_hz=50e6,
        iq_gain_imbalance_db=1.0,
        iq_phase_skew_deg=5.0,
        residual_cd_ps_nm=50.0,
    ),
}

# Carrier phase recovery block length, in symbols.
# A real receiver estimates the laser's slowly-drifting phase by averaging
# over a block of symbols and subtracting it. A SHORT block tracks fast phase
# noise well but is noisy; a LONG block is smooth but lags behind. 64 is a
# typical value. This is what turns a laser linewidth into a small RESIDUAL
# phase error rather than a constellation-destroying spiral.
CPE_BLOCK_SYMBOLS = 64

# Which profile the main dataset uses.
DEFAULT_PROFILE = "realistic"

# Backwards-compatible alias used by the original 3-class experiment.
PHASE_NOISE_STD_DEG = IMPAIRMENT_PROFILES["realistic"]["residual_phase_deg"]


# ---------------------------------------------------------------------------
# 5. DATASET SIZE
# ---------------------------------------------------------------------------
N_SYMBOLS = 4096      # symbols per constellation image

# Images per (modulation format, OSNR level).
# Total = N_PER_CLASS_PER_OSNR * N_CLASSES * N_OSNR
#   With 5 classes and 21 OSNR levels:
#     400 ->  42,000 images
#     600 ->  63,000 images
#    1000 -> 105,000 images
N_PER_CLASS_PER_OSNR = 600

# Smaller dataset used for the ablation sweeps, where we train many models and
# only care about RELATIVE differences, not the last 0.1% of accuracy.
N_PER_CLASS_PER_OSNR_ABLATION = 200


# ---------------------------------------------------------------------------
# 6. THE CONSTELLATION IMAGE
# ---------------------------------------------------------------------------
IMG_SIZE = 64        # 64x64 pixels
IQ_RANGE = 2.5       # image covers I,Q in [-2.5, +2.5]
                     # All constellations are normalised to average power 1,
                     # so using the SAME range for every class means the model
                     # cannot cheat by measuring the picture's zoom level.


# ---------------------------------------------------------------------------
# 7. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


# ---------------------------------------------------------------------------
# 8. TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 4
WEIGHT_DECAY = 1e-4      # mild L2 regularisation -- discourages large weights,
                         # which is a second (cheap) defence against overfitting


# ---------------------------------------------------------------------------
# 9. FILE PATHS
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
PAPER_DIR = os.path.join(PROJECT_DIR, "paper")

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, PAPER_DIR):
    os.makedirs(_d, exist_ok=True)


def dataset_paths(tag=""):
    """
    File names for one dataset variant.

    `tag` lets us keep several datasets side by side without overwriting
    each other -- for example the 'ideal' and 'severe' impairment versions
    used in the robustness study.
    """
    suffix = ("_" + tag) if tag else ""
    return dict(
        images=os.path.join(DATA_DIR, "X_images{}.npy".format(suffix)),
        features=os.path.join(DATA_DIR, "X_features{}.npy".format(suffix)),
        iq=os.path.join(DATA_DIR, "X_iq{}.npy".format(suffix)),
        labels=os.path.join(DATA_DIR, "y_labels{}.npy".format(suffix)),
        osnr=os.path.join(DATA_DIR, "osnr_values{}.npy".format(suffix)),
        meta=os.path.join(DATA_DIR, "dataset_meta{}.json".format(suffix)),
    )


# Default (main) dataset paths, kept as module-level names for convenience.
_P = dataset_paths()
F_IMAGES = _P["images"]
F_FEATURES = _P["features"]
F_IQ = _P["iq"]
F_LABELS = _P["labels"]
F_OSNR = _P["osnr"]
F_META = _P["meta"]


# ---------------------------------------------------------------------------
# 10. RAW I/Q REPRESENTATION
# ---------------------------------------------------------------------------
# For the representation study we also feed a 1-D CNN the raw I/Q samples
# directly. Storing all 4096 complex symbols for every image would need
# ~7 GB, so we store a fixed random subset of symbols per example.
# 512 symbols x 2 channels x 4 bytes = 4 kB per example, which is affordable
# and is also a realistic capture length for a monitoring receiver.
N_IQ_SYMBOLS = 512
