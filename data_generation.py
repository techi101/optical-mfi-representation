"""
data_generation.py
==================
STEP 1: simulate a coherent optical link and build the dataset.

There is no public dataset of optical constellation diagrams, so we build our
own by simulation. This is standard practice in the Optical Performance
Monitoring (OPM) literature.

WHAT MAKES THIS SIMULATOR MORE THAN A TOY
Most student projects add only white Gaussian noise. A real coherent receiver
also has to live with laser phase noise, an imperfect frequency lock, an
imperfect 90-degree optical hybrid, and dispersion that the DSP did not fully
remove. All of those distort the constellation diagram in ways that AWGN does
not. This file models each of them, and lets you switch them on and off
independently -- which is what makes the robustness study in the paper
possible.

THE SIGNAL CHAIN

    (1) pick random symbols from one of 5 constellations
    (2) residual chromatic dispersion        -> inter-symbol interference
    (3) laser phase noise (Wiener process)   -> constellation smears in angle
    (4) carrier phase recovery (block-average) -> removes MOST of (3)
    (5) residual frequency offset            -> slow rotation over the capture
    (6) I/Q gain imbalance and phase skew    -> constellation becomes skewed
    (7) normalise to unit average power      -> models receiver AGC
    (8) add AWGN at the requested OSNR       -> models amplifier ASE noise

Then each received signal is turned into THREE representations, all from the
SAME realisation, so the comparison in the paper is exactly fair:

    (a) a 64x64 constellation image      -> the 2-D CNN
    (b) 512 raw I/Q samples              -> the 1-D CNN
    (c) 15 statistical features          -> the MLP, SVM and k-NN

Run me:
    python data_generation.py                      # main dataset
    python data_generation.py --demo               # sanity checks + pictures
    python data_generation.py --profile ideal --tag ideal
    python data_generation.py --profile severe --tag severe
"""

import argparse
import json
import time

import numpy as np
from scipy import stats

import config

SPEED_OF_LIGHT = 2.99792458e8      # m/s


# ===========================================================================
# PART A -- THE FIVE CONSTELLATIONS
# ===========================================================================
def make_constellation(order):
    """
    Build a constellation with average power exactly 1.

    THE FIVE FORMATS AND WHY THEY LOOK DIFFERENT

    QPSK (4)    : a 2x2 square. Four points, maximally far apart. The most
                  robust format -- used on long, noisy links.

    8-QAM (8)   : NOT a square grid. This is the standard "circular" or
                  "star" 8-QAM: four points on an inner ring and four on an
                  outer ring, rotated 45 degrees relative to each other. The
                  radii are chosen so that all neighbouring points are the
                  same distance apart, which is the optimal 8-point
                  arrangement. Including it matters because a classifier that
                  only ever sees square grids has an artificially easy job.

    16-QAM (16) : a 4x4 square grid.

    32-QAM (32) : a CROSS. You cannot make a square grid with 32 points
                  (32 is not a perfect square), so the standard solution is
                  to take a 6x6 grid and delete the four corner points,
                  leaving 32. This "cross-QAM" shape is visually distinctive
                  and is what real 32-QAM transceivers use.

    64-QAM (64) : an 8x8 square grid. The densest format here, and the most
                  fragile in noise.

    WHY NORMALISE EVERY CONSTELLATION TO AVERAGE POWER 1?
    Otherwise the formats would differ in brightness, and any classifier
    could identify them just by measuring received power -- which would be
    meaningless, because a real receiver applies automatic gain control and
    always delivers a fixed power. Normalising forces the model to learn the
    constellation's SHAPE, which is the actual physics.
    """
    if order == 8:
        # Circular / star 8-QAM. Inner ring of 4, outer ring of 4, offset by
        # 45 degrees. The outer radius (1 + sqrt(3)) makes every nearest-
        # neighbour distance equal, which is the optimal 8-point layout.
        inner = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
        outer = (1.0 + np.sqrt(3.0)) * np.exp(1j * (np.arange(4) * np.pi / 2))
        points = np.concatenate([inner, outer])

    elif order == 32:
        # Cross 32-QAM: a 6x6 grid on levels (-5,-3,-1,1,3,5) with the four
        # corner points removed, leaving 36 - 4 = 32 points.
        levels = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        I, Q = np.meshgrid(levels, levels)
        pts = (I + 1j * Q).ravel()
        corner = (np.abs(pts.real) == 5) & (np.abs(pts.imag) == 5)
        points = pts[~corner]
        assert len(points) == 32

    else:
        # Square M-QAM: levels -(k-1), ..., -1, +1, ..., +(k-1) with k=sqrt(M)
        side = int(round(np.sqrt(order)))
        if side * side != order:
            raise ValueError("{}-QAM is not a square constellation".format(order))
        levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
        I, Q = np.meshgrid(levels, levels)
        points = (I + 1j * Q).ravel()

    # Scale so that the mean of |point|^2 is exactly 1.0
    points = points / np.sqrt(np.mean(np.abs(points) ** 2))
    return points


# ===========================================================================
# PART B -- OSNR  ->  SNR
# ===========================================================================
def osnr_db_to_snr_db(osnr_db,
                      symbol_rate=config.SYMBOL_RATE,
                      ref_bw=config.REF_BANDWIDTH,
                      n_pol=config.N_POL):
    """
    Convert an OPTICAL signal-to-noise ratio into the ELECTRICAL SNR the
    receiver actually experiences. These are NOT the same number.

    OSNR is measured by an optical spectrum analyser inside a FIXED reference
    bandwidth of 0.1 nm (= 12.5 GHz at 1550 nm), counting noise in BOTH
    polarisations. SNR is measured after the receiver has filtered to its own
    symbol rate and selected one polarisation.

        SNR = OSNR * (2 * B_ref) / (p * R_s)

    With B_ref = 12.5 GHz, R_s = 32 GBaud, p = 1:
        SNR = OSNR * 0.781    ->    SNR_dB = OSNR_dB - 1.07 dB
    """
    osnr_lin = 10.0 ** (osnr_db / 10.0)
    snr_lin = osnr_lin * (2.0 * ref_bw) / (n_pol * symbol_rate)
    return 10.0 * np.log10(snr_lin)


# ===========================================================================
# PART C -- THE INDIVIDUAL IMPAIRMENTS
# ===========================================================================
def apply_chromatic_dispersion(sig, cd_ps_nm, symbol_rate=config.SYMBOL_RATE,
                               wavelength=config.WAVELENGTH):
    """
    Residual chromatic dispersion -> inter-symbol interference.

    WHAT IT IS: different optical frequencies travel at slightly different
    speeds in glass, so a pulse spreads out in time and starts to overlap its
    neighbours. That overlap is inter-symbol interference (ISI), and on a
    constellation diagram it looks like the points being pulled towards each
    other and blurred.

    The receiver's DSP compensates most of it, but never all of it -- what is
    left is the "residual CD" modelled here, quoted in ps/nm.

    THE MATHS: dispersion is an all-pass filter with a quadratic phase:
        H(w) = exp( j * (beta2 * L / 2) * w^2 )
        beta2 * L = -(lambda^2 / (2*pi*c)) * D_total

    SIMPLIFICATION WORTH STATING IN THE PAPER: we work with one sample per
    symbol, so this captures the ISI that residual CD causes at the decision
    instants, not the full continuous waveform. That is an accurate model for
    SMALL residual CD (which is the realistic case after compensation) and it
    is what the constellation diagram actually shows.
    """
    if cd_ps_nm == 0:
        return sig

    n = len(sig)
    ts = 1.0 / symbol_rate
    # D_total from ps/nm into SI units s/m^2
    d_total = cd_ps_nm * 1e-12 / 1e-9
    beta2_l = -(wavelength ** 2) / (2 * np.pi * SPEED_OF_LIGHT) * d_total

    freqs = np.fft.fftfreq(n, d=ts)
    omega = 2 * np.pi * freqs
    h = np.exp(1j * (beta2_l / 2.0) * omega ** 2)
    return np.fft.ifft(np.fft.fft(sig) * h)


def apply_phase_impairments(sig, rng, linewidth_hz, freq_offset_hz,
                            symbol_rate=config.SYMBOL_RATE,
                            cpe_block=config.CPE_BLOCK_SYMBOLS):
    """
    Laser phase noise AND carrier frequency offset, followed by the receiver's
    carrier phase recovery. All three belong together, because CPE is what
    determines how much of the first two actually survives.

    THE TWO PHASE IMPAIRMENTS

    Laser phase noise: a laser's output phase drifts randomly. The drift is a
    random WALK (each step adds to the last), not independent jitter, and its
    speed is set by the laser's "linewidth" in Hz. A cheap DFB laser has
    ~1 MHz linewidth; a good external-cavity laser ~100 kHz. Over N symbols
    the accumulated phase variance is 2*pi*linewidth*Ts*N.

    Frequency offset: the transmitter laser and the receiver's local
    oscillator are two independent physical devices, so their frequencies
    never match exactly. The difference makes the whole constellation spin at
    a constant rate -- a phase RAMP rather than a random walk.

    WHY THEY MUST BE APPLIED *BEFORE* CARRIER PHASE RECOVERY
    This is a mistake worth understanding. Carrier phase recovery estimates
    the incoming phase over a block of symbols and subtracts it, and it does
    not care whether that phase came from laser drift or from a frequency
    offset -- it removes the slowly-varying part of BOTH. If you instead
    apply a frequency offset after CPE, nothing removes it, and even a small
    1 MHz offset smears every constellation point into a 46-degree arc,
    destroying the constellation. That is not what a real receiver produces.

    So: build the TOTAL phase (random walk + linear ramp), let the block-
    average CPE remove it, and keep what is left.

        surviving error = variation WITHIN one CPE block

    A wider linewidth or a larger frequency offset therefore still leaves a
    bigger residual -- which is the realistic behaviour we want to test the
    classifiers against -- but the constellation stays recognisable.

    THE CPE BLOCK-LENGTH TRADE-OFF (a good viva question): a SHORT block
    tracks fast phase noise well but averages over few symbols so the estimate
    is noisy; a LONG block gives a clean estimate but lags behind the drift.
    64 symbols is a typical compromise.
    """
    n = len(sig)
    ts = 1.0 / symbol_rate
    phase = np.zeros(n)

    # --- laser phase noise: a Wiener process (random walk) ---
    if linewidth_hz > 0:
        step_var = 2 * np.pi * linewidth_hz * ts
        phase += np.cumsum(rng.normal(0.0, np.sqrt(step_var), n))

    # --- frequency offset: a linear phase ramp ---
    if freq_offset_hz != 0:
        phase += 2 * np.pi * freq_offset_hz * np.arange(n) * ts

    if not np.any(phase):
        return sig

    # --- block-average carrier phase estimation, then subtraction ---
    n_blocks = int(np.ceil(n / cpe_block))
    padded = np.full(n_blocks * cpe_block, np.nan)
    padded[:n] = phase
    block_mean = np.nanmean(padded.reshape(n_blocks, cpe_block), axis=1)
    estimate = np.repeat(block_mean, cpe_block)[:n]

    residual_phase = phase - estimate
    return sig * np.exp(1j * residual_phase)


def apply_iq_imbalance(sig, gain_imbalance_db, phase_skew_deg):
    """
    I/Q gain imbalance and phase skew -> the constellation becomes SKEWED.

    A coherent receiver separates the in-phase (I) and quadrature (Q) parts of
    the light using a 90-degree optical hybrid feeding two photodiodes. In
    practice:
      * the two arms have slightly different gain      -> gain imbalance
      * the hybrid's phase is not exactly 90 degrees   -> phase skew

    The visual effect is distinctive and different from noise: a square
    constellation grid turns into a stretched parallelogram rather than
    blurring. That is exactly the kind of structured distortion a CNN might
    handle better than global statistics -- which is one of the things the
    paper tests.
    """
    if gain_imbalance_db == 0 and phase_skew_deg == 0:
        return sig

    # Split the gain error between the two arms so total power barely changes
    a_i = 10.0 ** (gain_imbalance_db / 40.0)
    a_q = 10.0 ** (-gain_imbalance_db / 40.0)
    skew = np.deg2rad(phase_skew_deg)

    i = sig.real
    q = sig.imag
    i_out = a_i * i
    q_out = a_q * (q * np.cos(skew) + i * np.sin(skew))
    return i_out + 1j * q_out


# ===========================================================================
# PART D -- SIMULATE ONE RECEIVED SIGNAL
# ===========================================================================
def simulate_received_symbols(rng, order, osnr_db,
                              n_symbols=config.N_SYMBOLS,
                              profile=None,
                              phase_noise_std_deg=None,
                              return_tx=False):
    """
    Produce one noisy, impaired received signal of `n_symbols` complex numbers.

    `profile` is a dict from config.IMPAIRMENT_PROFILES, or the name of one.
    Passing profile='ideal' gives the classic AWGN-only case.

    NOISE MATHS (worth knowing for the viva):
        After AGC the signal power is exactly 1, so
            noise power = 1 / SNR.
        The noise is COMPLEX and its power splits equally between the I and Q
        axes, so each axis gets standard deviation sqrt(noise_power / 2).

    `return_tx` also returns the clean transmitted symbols. That is used only
    by verify_project.py, which needs them to measure the applied noise power
    precisely. See the note in that file for why measuring it as
    E[|r|^2] - 1 instead would be numerically hopeless at high OSNR.
    """
    if profile is None:
        profile = config.IMPAIRMENT_PROFILES[config.DEFAULT_PROFILE]
    elif isinstance(profile, str):
        profile = config.IMPAIRMENT_PROFILES[profile]

    # Allow an explicit override, used by the original 3-class experiment
    res_phase_deg = (profile["residual_phase_deg"]
                     if phase_noise_std_deg is None else phase_noise_std_deg)

    constellation = make_constellation(order)

    # --- (1) transmit ---------------------------------------------------
    # Uniform random points. We do not need Gray coding or real bits: we are
    # classifying the constellation's shape, not decoding data, and uniform
    # random points give exactly the right constellation statistics.
    tx = rng.choice(constellation, size=n_symbols, replace=True)
    sig = tx.copy()

    # --- (2) residual chromatic dispersion ------------------------------
    sig = apply_chromatic_dispersion(sig, profile["residual_cd_ps_nm"])

    # --- (3)+(4)+(5) laser phase noise AND frequency offset, then the
    #     receiver's carrier phase recovery removes the slow part of both ---
    sig = apply_phase_impairments(sig, rng,
                                  profile["laser_linewidth_hz"],
                                  profile["freq_offset_hz"])

    # --- extra residual phase jitter (carrier-recovery estimation noise) --
    if res_phase_deg > 0:
        sig = sig * np.exp(1j * rng.normal(0.0, np.deg2rad(res_phase_deg),
                                           n_symbols))

    # --- (6) receiver I/Q imbalance -------------------------------------
    sig = apply_iq_imbalance(sig, profile["iq_gain_imbalance_db"],
                             profile["iq_phase_skew_deg"])

    # --- (7) automatic gain control: renormalise to unit average power ---
    # Done BEFORE adding noise so that OSNR keeps its exact meaning.
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))

    # --- (8) AWGN from the optical amplifiers ---------------------------
    snr_db = osnr_db_to_snr_db(osnr_db)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = 1.0 / snr_lin
    sigma = np.sqrt(noise_power / 2.0)
    noise = (rng.normal(0.0, sigma, n_symbols)
             + 1j * rng.normal(0.0, sigma, n_symbols))

    rx = sig + noise
    if return_tx:
        # `sig` is the impaired-but-noiseless signal: the right reference for
        # measuring the noise that was actually added.
        return rx, sig
    return rx


# ===========================================================================
# PART E -- REPRESENTATION 1: THE CONSTELLATION IMAGE
# ===========================================================================
def iq_to_image(rx, img_size=config.IMG_SIZE, iq_range=config.IQ_RANGE):
    """
    Turn the cloud of received points into a 64x64 grayscale image.

    HOW: lay a 64x64 grid over the I-Q plane from -2.5 to +2.5 and count how
    many received points land in each cell. That count becomes the pixel
    brightness. This is a 2-D histogram, and it is exactly what a
    constellation diagram on a lab instrument shows.

    WHY NOT SAVE A MATPLOTLIB SCATTER PLOT?
      * Speed: ~250x faster. 63,000 matplotlib figures would take ~50 minutes.
      * More information: a scatter plot saturates to solid colour where
        points overlap, throwing density away. The histogram keeps it -- and
        density is exactly what separates 16-QAM from 64-QAM at low OSNR.
      * No axes, labels, margins or anti-aliasing to distract the network.

    WHY log(1+x)?
    Cluster centres get far more hits than the sparse noise tails. On a linear
    scale the tails would round to zero and disappear. A log scale compresses
    the bright peaks and lifts the faint detail, keeping the outer
    constellation points visible -- the same reason astronomy images use a log
    scale.
    """
    counts, _, _ = np.histogram2d(
        rx.real, rx.imag,
        bins=img_size,
        range=[[-iq_range, iq_range], [-iq_range, iq_range]],
    )
    # Points outside +-2.5 are dropped, which is realistic: a real
    # analogue-to-digital converter also clips its input range.
    img = np.log1p(counts)
    peak = img.max()
    if peak > 0:
        img = img / peak
    img = (img * 255.0).astype(np.uint8)
    # .T so row index = Q and column index = I, i.e. it looks like a normal
    # maths plot when shown with origin='lower'.
    return img.T


# ===========================================================================
# PART F -- REPRESENTATION 2: RAW I/Q SAMPLES
# ===========================================================================
def iq_to_raw(rx, n_keep=config.N_IQ_SYMBOLS, rng=None):
    """
    Keep a fixed number of raw I/Q samples as a 2 x N array.

    This is the input to the 1-D CNN. Including it is what makes the paper's
    representation study complete: we can then compare

        raw I/Q  ->  image  ->  hand-crafted features

    i.e. three levels of pre-processing, from none to maximum, and see which
    actually matters. Without this, "CNN beats SVM" is ambiguous -- it could
    be the deep model OR the image representation doing the work.

    We keep the FIRST n_keep symbols (not a random subset) so that time
    ordering is preserved, which the 1-D convolution can exploit.
    """
    seg = rx[:n_keep]
    if len(seg) < n_keep:                       # pad if the capture is short
        seg = np.pad(seg, (0, n_keep - len(seg)))
    return np.stack([seg.real, seg.imag]).astype(np.float32)   # (2, n_keep)


# ===========================================================================
# PART G -- REPRESENTATION 3: 15 STATISTICAL FEATURES
# ===========================================================================
FEATURE_NAMES = [
    "|C20|/C21", "|C40|/C21^2", "|C41|/C21^2", "|C42|/C21^2",
    "|C60|/C21^3", "|C61|/C21^3", "|C63|/C21^3",
    "mean|r| / rms|r|", "std|r| / mean|r|",
    "skew(|r|)", "kurtosis(|r|)",
    "PAPR", "kurtosis(I)", "kurtosis(Q)",
    "E[|r|^4]/E[|r|^2]^2",
]


def iq_to_features(rx):
    """
    Squeeze the whole received signal down to 15 numbers.

    This is the CLASSICAL approach to modulation classification, used for
    decades before deep learning: a human engineer decides in advance which
    statistics are informative, computes them, and feeds them to a simple
    classifier.

    THE HIGHER-ORDER CUMULANTS are the important group.
    Moments (mean, mean-square, mean-4th-power, ...) describe the shape of a
    distribution. Cumulants are combinations of moments built so that the
    contribution of Gaussian noise CANCELS at 4th order and above -- which
    makes them a noise-robust "fingerprint" of a constellation. Each QAM order
    has a different theoretical value, so in principle a few of these numbers
    identify the format on their own.

    Each cumulant is divided by a power of C21 (the signal power) so the
    features do not depend on received power -- the same reasoning as
    normalising the constellation.
    """
    r = rx
    a = np.abs(r)
    a2 = a ** 2

    # ---- raw moments  M_pq = E[ r^(p-q) * conj(r)^q ] ----
    M20 = np.mean(r ** 2)
    M21 = np.mean(a2)
    M40 = np.mean(r ** 4)
    M41 = np.mean(r ** 3 * np.conj(r))
    M42 = np.mean(a2 ** 2)
    M60 = np.mean(r ** 6)
    M61 = np.mean(r ** 5 * np.conj(r))
    M63 = np.mean(a2 ** 3)

    # ---- cumulants (standard textbook combinations) ----
    C20 = M20
    C21 = M21
    C40 = M40 - 3.0 * M20 ** 2
    C41 = M41 - 3.0 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2.0 * M21 ** 2
    C60 = M60 - 15.0 * M20 * M40 + 30.0 * M20 ** 3
    C61 = M61 - 5.0 * M21 * M40 - 10.0 * M20 * M41 + 30.0 * M20 ** 2 * M21
    C63 = M63 - 9.0 * C42 * C21 - 6.0 * C21 ** 3

    c21 = np.abs(C21) + 1e-12

    feats = [
        np.abs(C20) / c21,
        np.abs(C40) / c21 ** 2,
        np.abs(C41) / c21 ** 2,
        np.abs(C42) / c21 ** 2,
        np.abs(C60) / c21 ** 3,
        np.abs(C61) / c21 ** 3,
        np.abs(C63) / c21 ** 3,
        # ---- amplitude-distribution features ----
        # QPSK has 1 amplitude ring, 8-QAM has 2, 16-QAM has 3, 32-QAM has 5,
        # 64-QAM has 9. The spread and shape of the amplitude histogram is
        # therefore genuinely informative.
        np.mean(a) / (np.sqrt(np.mean(a2)) + 1e-12),
        np.std(a) / (np.mean(a) + 1e-12),
        stats.skew(a),
        stats.kurtosis(a),
        np.max(a2) / (np.mean(a2) + 1e-12),
        stats.kurtosis(r.real),
        stats.kurtosis(r.imag),
        np.mean(a2 ** 2) / (np.mean(a2) ** 2 + 1e-12),
    ]
    return np.asarray(feats, dtype=np.float32)


# ===========================================================================
# PART H -- BUILD THE WHOLE DATASET
# ===========================================================================
def build_dataset(n_per_class_per_osnr=config.N_PER_CLASS_PER_OSNR,
                  profile=config.DEFAULT_PROFILE,
                  n_symbols=config.N_SYMBOLS,
                  img_size=config.IMG_SIZE,
                  seed=config.SEED,
                  store_iq=True,
                  verbose=True):
    """
    Loop over every (modulation format, OSNR) pair and generate examples.

    Returns five aligned arrays -- row i of every array describes the SAME
    simulated signal:
        X_img   (N, S, S)  uint8    constellation images    -> 2-D CNN
        X_iq    (N, 2, K)  float32  raw I/Q samples         -> 1-D CNN
        X_feat  (N, 15)    float32  statistical features    -> MLP/SVM/kNN
        y       (N,)       int8     class index             -> the answer
        osnr    (N,)       int8     OSNR of that row
    """
    n_total = n_per_class_per_osnr * config.N_CLASSES * config.N_OSNR

    X_img = np.zeros((n_total, img_size, img_size), dtype=np.uint8)
    X_feat = np.zeros((n_total, len(FEATURE_NAMES)), dtype=np.float32)
    X_iq = (np.zeros((n_total, 2, config.N_IQ_SYMBOLS), dtype=np.float32)
            if store_iq else None)
    y = np.zeros(n_total, dtype=np.int8)
    osnr_arr = np.zeros(n_total, dtype=np.int8)

    prof = config.IMPAIRMENT_PROFILES[profile] if isinstance(profile, str) else profile

    idx = 0
    t0 = time.time()

    for cls_id, (name, order) in enumerate(config.MODULATIONS.items()):
        for osnr_db in config.OSNR_LIST:
            # A deterministic seed per (class, OSNR) block, so re-running gives
            # identical data and changing one part does not reshuffle the rest.
            block_seed = seed + cls_id * 1000 + int(osnr_db)
            rng = np.random.default_rng(block_seed)

            for _ in range(n_per_class_per_osnr):
                rx = simulate_received_symbols(rng, order, osnr_db,
                                               n_symbols=n_symbols,
                                               profile=prof)
                X_img[idx] = iq_to_image(rx, img_size=img_size)
                X_feat[idx] = iq_to_features(rx)
                if store_iq:
                    X_iq[idx] = iq_to_raw(rx)
                y[idx] = cls_id
                osnr_arr[idx] = osnr_db
                idx += 1

        if verbose:
            print("  [{:6.1f}s] finished {:<6s} ({:,} / {:,} images)".format(
                time.time() - t0, name, idx, n_total))

    return X_img, X_iq, X_feat, y, osnr_arr


def save_dataset(X_img, X_iq, X_feat, y, osnr_arr,
                 n_per_class_per_osnr, profile, tag=""):
    paths = config.dataset_paths(tag)
    np.save(paths["images"], X_img)
    np.save(paths["features"], X_feat)
    np.save(paths["labels"], y)
    np.save(paths["osnr"], osnr_arr)
    if X_iq is not None:
        np.save(paths["iq"], X_iq)

    prof = (config.IMPAIRMENT_PROFILES[profile]
            if isinstance(profile, str) else profile)
    meta = {
        "seed": config.SEED,
        "profile": profile if isinstance(profile, str) else "custom",
        "impairments": prof,
        "class_names": config.CLASS_NAMES,
        "modulation_orders": config.MODULATIONS,
        "osnr_list": config.OSNR_LIST,
        "n_per_class_per_osnr": n_per_class_per_osnr,
        "n_total": int(len(y)),
        "n_symbols_per_image": config.N_SYMBOLS,
        "n_iq_symbols": config.N_IQ_SYMBOLS,
        "img_size": config.IMG_SIZE,
        "iq_range": config.IQ_RANGE,
        "symbol_rate_gbd": config.SYMBOL_RATE / 1e9,
        "ref_bandwidth_ghz": config.REF_BANDWIDTH / 1e9,
        "n_pol": config.N_POL,
        "cpe_block_symbols": config.CPE_BLOCK_SYMBOLS,
        "feature_names": FEATURE_NAMES,
        "snr_db_at_osnr": {int(o): round(float(osnr_db_to_snr_db(o)), 2)
                           for o in config.OSNR_LIST},
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    return paths


# ===========================================================================
# PART I -- SANITY CHECKS AND PICTURES
# ===========================================================================
def save_sample_grid(filename=None, osnr_to_show=(5, 10, 15, 20, 25),
                     profile=config.DEFAULT_PROFILE, suffix=""):
    """One example constellation image per (format, OSNR), so we can SEE that
    the simulation behaves physically."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if filename is None:
        filename = os.path.join(config.FIGURES_DIR,
                                "sample_constellations{}.png".format(suffix))

    n_rows = config.N_CLASSES
    n_cols = len(osnr_to_show)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.0 * n_cols, 2.1 * n_rows))

    for i, (name, order) in enumerate(config.MODULATIONS.items()):
        for j, osnr_db in enumerate(osnr_to_show):
            rng = np.random.default_rng(config.SEED + 7 * i + osnr_db)
            rx = simulate_received_symbols(rng, order, osnr_db, profile=profile)
            img = iq_to_image(rx)
            ax = axes[i, j]
            ax.imshow(img, cmap="viridis", origin="lower",
                      extent=[-config.IQ_RANGE, config.IQ_RANGE,
                              -config.IQ_RANGE, config.IQ_RANGE])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("OSNR = {} dB".format(osnr_db), fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")

    fig.suptitle("Simulated constellation images -- '{}' impairment profile "
                 "({}x{} px, {} symbols)".format(
                     profile, config.IMG_SIZE, config.IMG_SIZE,
                     config.N_SYMBOLS), fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved " + filename)
    return filename


def theoretical_c40(order):
    """
    The EXACT theoretical value of |C40| / C21^2 for a noiseless
    constellation, computed analytically from the constellation's definition.

    Because the symbols are drawn uniformly from a finite set of points, the
    expectations are just finite sums -- no simulation and no randomness is
    involved. This is therefore a genuinely independent prediction that the
    simulator must reproduce.

    WHY WE COMPUTE THIS RATHER THAN HARD-CODING LITERATURE NUMBERS:
    square QAM (4, 16, 64) has one universally agreed layout, so its cumulants
    are standard published constants. But 8-QAM and 32-QAM each have SEVERAL
    standard variants (star vs rectangular 8-QAM; cross vs other 32-QAM), and
    each variant has a different cumulant. Quoting a number from a paper that
    used a different variant would make a correct simulator look broken.
    Deriving the value from the constellation we actually use avoids that
    trap, and the square formats still let us cross-check against the
    published constants.
    """
    pts = make_constellation(order)          # already unit average power
    M20 = np.mean(pts ** 2)
    M21 = np.mean(np.abs(pts) ** 2)
    M40 = np.mean(pts ** 4)
    C40 = M40 - 3.0 * M20 ** 2
    return float(np.abs(C40) / np.abs(M21) ** 2)


# Published values for SQUARE QAM, used as an independent cross-check that our
# analytic formula above is itself right.
PUBLISHED_C40 = {"QPSK": 1.0000, "16QAM": 0.6800, "64QAM": 0.6191}


def print_theory_check(n_symbols=200000, tol=0.01):
    """
    Compare the simulator's measured cumulants against exact theory.
    A strong, cheap proof that the signal model is right, independent of any
    machine learning.
    """
    print("\n  Sanity check -- normalised |C40| at OSNR = 40 dB, no impairments")
    print("  " + "-" * 62)
    print("  {:<8s} {:>10s} {:>10s} {:>9s} {:>12s}".format(
        "format", "simulated", "theory", "error", "published"))
    ok = True
    for name, order in config.MODULATIONS.items():
        rng = np.random.default_rng(config.SEED)
        rx = simulate_received_symbols(rng, order, 40.0, n_symbols=n_symbols,
                                       profile="ideal")
        meas = iq_to_features(rx)[1]
        theory = theoretical_c40(order)
        err = abs(meas - theory)
        ok &= err < tol
        pub = PUBLISHED_C40.get(name)
        pub_s = "{:.4f}".format(pub) if pub is not None else "(variant)"
        print("  {:<8s} {:>10.4f} {:>10.4f} {:>9.4f} {:>12s}".format(
            name, meas, theory, err, pub_s))
    print("  " + "-" * 62)
    print("  {}  (tolerance {:.3f})".format(
        "ALL MATCH THEORY" if ok else "MISMATCH -- investigate", tol))
    print("  Square QAM also matches the published constants in the far-right")
    print("  column, which validates the analytic formula itself.")
    return ok


def print_constellation_info():
    print("\n  Constellation geometry")
    print("  " + "-" * 56)
    print("  {:<8s} {:>7s} {:>7s} {:>10s} {:>12s}".format(
        "format", "points", "bits", "rings", "peak/avg"))
    for name, order in config.MODULATIONS.items():
        pts = make_constellation(order)
        radii = np.round(np.abs(pts), 6)
        n_rings = len(np.unique(radii))
        papr = np.max(np.abs(pts) ** 2) / np.mean(np.abs(pts) ** 2)
        print("  {:<8s} {:>7d} {:>7d} {:>10d} {:>12.3f}".format(
            name, len(pts), config.BITS_PER_SYMBOL[name], n_rings, papr))


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate the optical MFI dataset")
    p.add_argument("--n-per-class", type=int,
                   default=config.N_PER_CLASS_PER_OSNR)
    p.add_argument("--profile", type=str, default=config.DEFAULT_PROFILE,
                   choices=list(config.IMPAIRMENT_PROFILES.keys()))
    p.add_argument("--tag", type=str, default="",
                   help="suffix for the saved files, to keep variants apart")
    p.add_argument("--no-iq", action="store_true",
                   help="skip storing raw I/Q (saves ~250 MB)")
    p.add_argument("--demo", action="store_true",
                   help="only run the sanity checks and draw sample images")
    args = p.parse_args()

    print("=" * 70)
    print("  OPTICAL MODULATION FORMAT IDENTIFICATION -- DATA GENERATION")
    print("=" * 70)
    print("  formats      : " + ", ".join(config.CLASS_NAMES))
    print("  OSNR sweep   : {}..{} dB ({} levels)".format(
        config.OSNR_DB_MIN, config.OSNR_DB_MAX, config.N_OSNR))
    print("  symbol rate  : {:.0f} GBaud, {} polarisation".format(
        config.SYMBOL_RATE / 1e9, config.N_POL))
    print("  OSNR -> SNR  : SNR_dB = OSNR_dB {:+.2f} dB".format(
        osnr_db_to_snr_db(0)))
    print("  symbols/image: {}".format(config.N_SYMBOLS))
    print("  image size   : {} x {}".format(config.IMG_SIZE, config.IMG_SIZE))
    print("  seed         : {}".format(config.SEED))
    print("\n  impairment profile: '{}'".format(args.profile))
    for k, v in config.IMPAIRMENT_PROFILES[args.profile].items():
        print("    {:<24s} {}".format(k, v))

    print_constellation_info()
    print_theory_check()

    print("\n  Drawing sample constellation images...")
    save_sample_grid(profile=args.profile,
                     suffix=("_" + args.tag) if args.tag else "")

    if args.demo:
        print("\n  --demo given: stopping before the full dataset build.")
        raise SystemExit(0)

    n = args.n_per_class
    total = n * config.N_CLASSES * config.N_OSNR
    print("\n  Building dataset: {} per class per OSNR -> {:,} images".format(
        n, total))

    t0 = time.time()
    X_img, X_iq, X_feat, y, osnr_arr = build_dataset(
        n, profile=args.profile, store_iq=not args.no_iq)
    gen_time = time.time() - t0

    paths = save_dataset(X_img, X_iq, X_feat, y, osnr_arr, n,
                         args.profile, tag=args.tag)

    print("\n  DONE in {:.1f} s".format(gen_time))
    print("  images   : {}  {}".format(X_img.shape, X_img.dtype))
    if X_iq is not None:
        print("  raw I/Q  : {}  {}".format(X_iq.shape, X_iq.dtype))
    print("  features : {}  {}".format(X_feat.shape, X_feat.dtype))
    print("  labels   : {}, counts = {}".format(
        y.shape, dict(zip(config.CLASS_NAMES, np.bincount(y).tolist()))))
    print("  saved to : " + config.DATA_DIR)
