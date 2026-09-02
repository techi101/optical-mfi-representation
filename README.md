# Modulation Format Identification for Optical Performance Monitoring

A complete, reproducible research implementation that identifies the
modulation format of a coherent optical signal (QPSK, 8-QAM, 16-QAM, 32-QAM,
64-QAM) from its constellation diagram — and answers a question the existing
literature leaves open:

> **Does the benefit of deep learning here come from the model, or from the
> input representation?**

Everything runs on a laptop CPU. No GPU required.

---

## Contents

1. [The research question](#1-the-research-question)
2. [The problem in plain English](#2-the-problem-in-plain-english)
3. [How to run everything](#3-how-to-run-everything)
4. [What each file does](#4-what-each-file-does)
5. [The simulator](#5-the-simulator)
6. [The five classifiers](#6-the-five-classifiers)
7. [Every hyperparameter, and why](#7-every-hyperparameter-and-why)
8. [The experiments](#8-the-experiments)
9. [How to check my work](#9-how-to-check-my-work)
10. [Viva questions with answers](#10-viva-questions-with-answers)
11. [Honest limitations](#11-honest-limitations)

---

## 1. The research question

Published work consistently shows that a CNN on constellation images beats an
SVM on hand-crafted features. That's true — but the comparison changes **two
things at once**:

| | Classical | Deep-learning paper |
|---|---|---|
| **Model** | shallow (SVM) | deep (CNN) |
| **Input** | 15 numbers | full 2-D image |

So which one is responsible for the gain? Nobody separates them. This project
does, by adding a **fourth classifier nobody usually includes: an MLP on the
same 15 features.**

```
                     ┌─ CNN    on 64×64 image     ─┐  deep model, rich input
   same signals ─────┼─ MLP    on 15 features     ─┤  deep model, poor input
                     ├─ SVM    on 15 features     ─┤  shallow, poor input
                     ├─ k-NN   on 15 features     ─┤  shallow, poor input
                     └─ IQ-CNN on raw I/Q         ─┘  deep model, raw input
```

- **MLP vs SVM vs k-NN** — same input, three completely different algorithms
  → isolates the effect of the **model**
- **CNN vs MLP** — both trained neural networks, different input
  → isolates the effect of the **representation**

That design is the contribution. It turns your baselines from an obligation
into the core of the paper.

---

## 2. The problem in plain English

### What is a modulation format?

Optical fibre carries data as light. To send more than one bit per pulse, the
transmitter varies both the **amplitude** and the **phase** of the light. Each
distinct (amplitude, phase) combination is a **symbol**, and the set of
allowed symbols is the **modulation format**. Plotting every symbol with its
in-phase component I on the x-axis and quadrature component Q on the y-axis
gives the **constellation diagram**.

| Format | Points | Bits/symbol | Shape |
|---|---|---|---|
| QPSK | 4 | 2 | 2×2 square |
| 8-QAM | 8 | 3 | **star** — 4 inner + 4 outer, rotated 45° |
| 16-QAM | 16 | 4 | 4×4 square |
| 32-QAM | 32 | 5 | **cross** — 6×6 grid minus 4 corners |
| 64-QAM | 64 | 6 | 8×8 square |

Two of the five are deliberately **non-square**. A classifier that only ever
sees square grids of different densities has an artificially easy job, and a
reviewer will say so.

More points = more data per symbol, but the points sit closer together, so
noise corrupts them more easily. That trade-off drives everything here.

### Why identify the format?

Modern networks are **elastic**: the transmitter switches format based on link
conditions — 64-QAM on a short clean link, QPSK on a long noisy one. The
receiver must know which format arrived before it can demodulate anything.
Traditionally this is negotiated over a control channel; **Modulation Format
Identification (MFI)** infers it from the signal directly, making the network
faster to reconfigure and more autonomous. It's a component of **Optical
Performance Monitoring (OPM)**.

### What is OSNR?

**Optical Signal-to-Noise Ratio**, in dB. The dominant noise on a long link is
**ASE (Amplified Spontaneous Emission)** from the erbium-doped fibre
amplifiers placed every ~80 km to counter fibre loss. Each amplifier boosts
the signal but adds a little random light of its own, and that accumulates.

- **High OSNR (25 dB)** — clean, constellation points are tight dots
- **Low OSNR (5 dB)** — noisy, points smear into an indistinct cloud

Decibels are logarithmic: every 3 dB is a factor of 2 in power.

---

## 3. How to run everything

```bash
pip install -r requirements.txt

# 1. sanity checks + sample images         (~1 min)
python data_generation.py --demo

# 2. build the main dataset                (~4.5 min, ~500 MB)
python data_generation.py --n-per-class 600 --profile realistic

# 3. main comparison, 3 seeds              (~60 min)
python train_and_evaluate.py --seeds 42 43 44

# 4. the four extra experiments            (~90 min total)
python experiments.py generalise
python experiments.py ablation
python experiments.py symbols
python experiments.py robustness

# 5. all figures + results summary         (~1 min)
python plots.py

# 6. check everything is correct           (~2 min)
python verify_project.py

# 7. build the paper
python paper/make_macros.py
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

Quick variants:

```bash
python data_generation.py --n-per-class 200      # smaller/faster dataset
python train_and_evaluate.py --seeds 42 --epochs 5   # fast trial run
python cnn_model.py                              # print all 3 architectures
python baselines.py                              # smoke-test SVM and k-NN
```

Everything is seeded — re-running gives identical numbers.

---

## 4. What each file does

```
research paper/
├── config.py               every constant, seed and impairment profile
├── data_generation.py      STEP 1  simulate the optical link → dataset
├── cnn_model.py            STEP 2  the three neural networks
├── baselines.py            STEP 3  the SVM and k-NN
├── train_and_evaluate.py   STEP 4  training harness + main comparison
├── experiments.py          STEP 5  the four extra studies
├── plots.py                STEP 6  all figures + results summary
├── verify_project.py       independent correctness checks
├── requirements.txt        exact package versions
├── paper/
│   ├── paper.tex           the IEEE-format manuscript
│   ├── make_macros.py      generates every number in the paper from JSON
│   └── README.md           how to compile + WHAT YOU MUST CHECK
├── data/                   generated datasets (.npy)
├── results/                JSON results, trained models, summary tables
├── figures/                all .png figures, 300 dpi
└── archive_3class/         the earlier 3-class results, kept for reference
```

**Why a separate `config.py`?** If a constant appeared in two scripts and you
changed it in only one, results would silently become wrong — and you might
not notice until a reviewer did. One source of truth prevents that.

---

## 5. The simulator

### The signal chain

```
random symbols
   → residual chromatic dispersion        (inter-symbol interference)
   → laser phase noise + frequency offset (angular smearing)
   → carrier phase recovery               (removes most of the above)
   → I/Q gain imbalance and phase skew    (skews the constellation)
   → automatic gain control               (normalise to unit power)
   → AWGN at the target OSNR              (amplifier ASE noise)
```

Most student projects add only AWGN. The four extra impairments are what make
this defensible to an optical-communications reviewer — and they enable the
robustness experiment, which is the most interesting result in the project.

### The impairments, and what each does to the picture

| Impairment | Physical cause | Visual effect |
|---|---|---|
| **Residual CD** | different wavelengths travel at different speeds → pulses overlap | points blur into each other |
| **Laser phase noise** | laser phase drifts as a random walk; speed set by linewidth | angular smearing |
| **Frequency offset** | TX laser and local oscillator are different physical devices | constellation rotates |
| **I/Q imbalance** | the two receiver arms have unequal gain; hybrid isn't exactly 90° | grid becomes a skewed parallelogram |
| **AWGN** | ASE noise from the amplifier chain | every point spreads into a cloud |

`figures/impairment_isolation.png` shows each one **in isolation** at 30 dB,
so you can see exactly what each contributes.

### One modelling detail worth knowing (it was a real bug)

The frequency offset must be applied **before** carrier phase recovery, not
after. Removing residual frequency offset is one of CPE's jobs — a real
receiver's phase tracker follows that slow rotation and cancels it. My first
version applied it afterwards, so nothing removed it, and just 1 MHz of offset
smeared every constellation point into a 46° arc. The constellations looked
destroyed at 25 dB, which is not what a working receiver produces.

The fix: build the total phase (random walk + linear ramp), let the
block-average CPE remove the slowly-varying part of **both**, and keep what
survives — the variation *within* one 64-symbol block. A wider linewidth or
bigger offset still leaves a bigger residual, which is the realistic
behaviour, but the constellation stays recognisable.

**This is a good viva story**: it shows you understand what the DSP does, not
just how to call a function.

### Why OSNR ≠ SNR

A favourite examiner question.

**OSNR** is *optical*, measured by an optical spectrum analyser in a fixed
0.1 nm (= 12.5 GHz at 1550 nm) reference bandwidth, counting noise in **both**
polarisations. **SNR** is *electrical*, after the receiver filters to its own
symbol rate and selects one polarisation.

```
SNR = OSNR × (2 · B_ref) / (p · R_s)
```

With B_ref = 12.5 GHz, R_s = 32 GBaud, p = 1:

```
SNR = OSNR × 0.781    →    SNR_dB = OSNR_dB − 1.07 dB
```

So at OSNR 25 dB the receiver really sees ~23.9 dB. Reporting against OSNR
without this conversion overstates your electrical SNR by about 1 dB.

### Proof the simulator is correct

For a noiseless unit-power constellation, the normalised fourth-order cumulant
|C40|/C21² is an **exact finite sum** over the constellation points — no
simulation involved. The simulator reproduces it for all five formats:

| Format | Exact theory | Simulated | Error |
|---|---|---|---|
| QPSK | 1.0000 | 0.9997 | 0.0003 |
| 8-QAM (star) | 1.5274 | 1.5251 | 0.0023 |
| 16-QAM | 0.6800 | 0.6830 | 0.0030 |
| 32-QAM (cross) | 0.1900 | 0.1868 | 0.0032 |
| 64-QAM | 0.6190 | 0.6198 | 0.0007 |

And the three **square** formats independently match the published constants
(1.0000 / 0.6800 / 0.6191), which validates the analytic formula itself.

> **Why compute theory instead of quoting the literature?** Square QAM has one
> agreed layout, so its cumulants are standard constants. But 8-QAM and 32-QAM
> each have *several* standard variants with different cumulants. I initially
> hardcoded literature values for those two and the check "failed" — the
> simulator was right; my reference numbers came from different variants.
> Deriving the value from the constellation actually used avoids that trap.

### The three representations

Every capture is converted into three forms, **all from the same realisation**
so the comparison is exactly fair:

| Representation | Shape | Human pre-processing |
|---|---|---|
| Raw I/Q | 2 × 512 | none |
| Constellation image | 64 × 64 | binning into a 2-D histogram |
| Statistical features | 15 | a lot (chosen cumulants etc.) |

**Why a 2-D histogram, not a matplotlib scatter plot?**

- **Speed**: ~250× faster. 63,000 matplotlib figures ≈ 50 minutes; histograms
  take seconds. That's what makes it practical to iterate.
- **More information**: a scatter plot saturates to solid colour where points
  overlap, throwing away density — and density is exactly what separates
  16-QAM from 64-QAM at low OSNR.
- **No clutter**: no axes, labels, margins or anti-aliasing.

**Why the log(1+x) scaling?** Cluster centres get far more hits than the sparse
noise tails. On a linear scale the tails round to zero and vanish. A log scale
lifts the faint detail so outer constellation points stay visible — the same
reason astronomy images use a log scale.

---

## 6. The five classifiers

### CNN on the constellation image

Three conv blocks (Conv3×3 → BatchNorm → ReLU → MaxPool2) with 16/32/64
channels, then dropout and two dense layers.

| Layer | What it does | Why |
|---|---|---|
| **Conv2d 3×3** | slides a learned filter over the image | detects local patterns anywhere, with shared weights |
| **BatchNorm** | rescales outputs to ~zero mean, unit variance | faster, more stable training |
| **ReLU** | max(0, x) | the non-linearity; without it stacked layers collapse to one linear layer |
| **MaxPool 2×2** | keeps the largest value per 2×2 square | halves the image; a cluster shifted 1 px is still the same cluster |
| **Dropout 0.3** | randomly zeroes 30% during training only | main defence against overfitting |

**Why flatten instead of global average pooling?** GAP discards *where* each
feature was found. For photographs that's good — a cat is a cat anywhere. Here
it would be harmful: the geometric **arrangement** of clusters in the I-Q plane
*is* the signal. Note the 1-D model makes the **opposite** choice, for the
opposite reason. Being able to explain that is a strong viva answer.

### IQ-CNN on raw I/Q

Three 1-D conv blocks (kernels 7, 5, 3) then **global average pooling** —
correct here because the symbol stream is stationary, so absolute position in
the capture carries no information.

**Prediction, stated before running it:** this model should struggle. The
transmitted symbols are i.i.d., so the *order* carries almost no class
information — but a 1-D convolution is built to exploit order. The network must
estimate a *distribution* from a *sequence*. The constellation image hands that
distribution over directly, because a 2-D histogram **is** a density estimate.
This is the clearest statement of why representation matters.

### MLP, SVM, k-NN on 15 features

**The 15 features:** seven normalised higher-order cumulants (|C20|, |C40|,
|C41|, |C42|, |C60|, |C61|, |C63|) plus eight amplitude/phase statistics.

> **What is a cumulant?** Moments (mean, mean-square, mean-4th-power…) describe
> a distribution's shape. Cumulants are combinations of moments built so that
> the contribution of Gaussian noise **cancels** at 4th order and above — a
> noise-robust "fingerprint" of a constellation. Each QAM order has a distinct
> theoretical value. This is the standard pre-deep-learning approach, which is
> exactly why it makes a fair baseline.

> **Why the amplitude statistics help:** QPSK has 1 amplitude ring, 8-QAM has 2,
> 16-QAM has 3, 32-QAM has 5, 64-QAM has 9. The spread and shape of the
> amplitude histogram is genuinely informative.

**Why every model is wrapped in a `Pipeline` with `StandardScaler`:** the
features live on wildly different scales (cumulant ratios ≈ 0.5, PAPR can be
10+). Both k-NN and the RBF-SVM work on **distances**, so a large-range feature
would dominate purely because of its units. Using a Pipeline also matters for
**correctness**: the scaler learns its mean and standard deviation from the
**training data only**. Computing them over the whole dataset would leak test
information into training and inflate the accuracy. **Reviewers check for this** —
it's the single most common silent mistake in ML papers.

**The MLP is the critical control.** It's a trained deep network that sees only
the 15 features. If MLP ≈ SVM, the representation is what matters. If MLP ≈ CNN,
the model is what matters. Either outcome is a real finding.

---

## 7. Every hyperparameter, and why

### Dataset

| Choice | Value | Why |
|---|---|---|
| Formats | 5 | Published MFI papers use 4–6. Three is enough to demo the idea but makes the task artificially easy. Two of ours are non-square. |
| OSNR range | 5–25 dB, 1 dB steps | Covers "hopeless" to "clean"; 21 points draws a smooth curve. |
| Symbols/capture | 4096 | Power of two; enough to populate 64 clusters visibly. |
| Images/class/OSNR | 600 | 63,000 total. ~4.5 min to generate, ~20 min/seed to train. |
| Image size | 64×64 | Must resolve 64-QAM's 8×8 grid → needs ≳40 px. 32×32 can't; 128×128 costs 4× for little gain. |
| I/Q range | ±2.5 | Signal never exceeds ~1.53 after normalisation; rest is noise headroom. **Same range for every class**, so zoom level carries no class information. |
| Symbol rate | 32 GBaud | Standard commercial rate; used in the OSNR→SNR conversion. |
| CPE block | 64 symbols | Short block tracks fast phase noise but is noisy; long block is smooth but lags. 64 is the usual compromise. |

### Split

| Choice | Value | Why |
|---|---|---|
| Train/val/test | 70/15/15 | 9,450 test images; ~90 per (format, OSNR) cell, which matters because we report accuracy per OSNR. |
| Stratified on (class, OSNR) | yes | A plain random split could load the test set with hard low-OSNR cases by chance. Forces each of the 5×21 = 105 cells to be split identically. |

> **Why three sets, not two?** *Train* is what the model learns from.
> *Validation* decides **when to stop** and which weights to keep — because we
> make decisions with it, it's no longer a neutral judge. *Test* is touched
> exactly once, at the end. It's the only honest number, and it's what goes in
> the paper.

### Training

| Choice | Value | Why |
|---|---|---|
| Conv blocks | 3 | After 3 poolings one feature-map pixel covers ~30×30 of the input — the scale of the pattern. The **ablation tests this** rather than asserting it. |
| Channels | 16→32→64 | Standard pyramid: as the image shrinks, allow more feature types. |
| Kernel | 3×3 | Smallest filter with centre-and-surround. Two stacked 3×3 see the same area as one 5×5 with fewer parameters plus an extra non-linearity. Default since VGG (2014). |
| Batch size | 128 | Per-sample updates are noisy; whole-dataset updates give one update per pass. 128 gives a good gradient estimate and ~345 updates/epoch. |
| Optimiser | Adam, lr 1e-3 | Keeps a separate auto-tuned step size per weight; works well without hand-tuning. |
| Weight decay | 1e-4 | Mild L2 — discourages large weights, a cheap second defence against overfitting. |
| LR schedule | halve on plateau | Big steps early to find a good region, small steps later to settle into it. |
| Dropout | 0.3 (CNN), 0.2 (MLP) | Main overfitting defence. 0.5 suits huge dense layers; ours are modest. |
| Early stopping | patience 4 | If validation accuracy hasn't improved in 4 epochs the model is starting to memorise. Stop, keep the best weights. |
| Seeds | 42, 43, 44 | Lets us report mean ± s.d. A single number could be luck. |

---

## 8. The experiments

| Experiment | Question it answers | Why a reviewer cares |
|---|---|---|
| **Main comparison** | Which representation wins? | The paper's central claim |
| **Generalisation** | Does it work at OSNR values never seen in training? | Rules out "it just memorised each noise level" |
| **Ablation** | Do the architecture choices actually matter? | Converts "I chose 3 layers" into "3 was best, and here's the evidence" |
| **Capture length** | How few symbols are enough? | The practical deployment question, rarely answered |
| **Robustness** | Does a model trained on a clean channel survive a dirty one? | The honest answer to "your simulation is idealised" |

The **robustness** experiment is the most valuable. Every simulation-based MFI
paper faces the objection *"you trained and tested on the same idealised
channel."* Training on `ideal` (AWGN only) and testing on `severe` answers it
directly. A large drop is a genuinely useful **negative** result; a small drop
is a strong positive claim. Either way you have something to say.

---

## 8b. The results

All numbers are mean ± standard deviation over 3 seeds. The authoritative,
always-current version is `results/results_summary.txt`; the paper reads the
same numbers from `paper/results_macros.tex`.

### The main comparison

| Model | Input | Accuracy | Params |
|---|---|---|---|
| **CNN** | 64×64 image | **98.73 ± 0.12 %** | 285,941 |
| SVM | 15 features | 93.41 ± 0.10 % | — |
| MLP | 15 features | 93.08 ± 0.10 % | 11,013 |
| k-NN | 15 features | 91.58 ± 0.08 % | — |
| IQ-CNN | raw I/Q | 86.56 ± 4.64 % | 27,781 |

**The central result.** Three fundamentally different algorithms on identical
features span **1.83 points**. Changing the representation to the density
image gains **5.32 points** — about **3×** larger. And the MLP (a deep network)
came out *below* the SVM (a shallow one), so adding depth to the feature
pipeline bought nothing at all.

### Stated as an OSNR margin

| Model | ≥95% accuracy | ≥99% accuracy |
|---|---|---|
| **CNN** | **7 dB** | **10 dB** |
| SVM | 14 dB | 18 dB |
| k-NN | 15 dB | 18 dB |

**A 7 dB OSNR margin.** Quote this rather than the percentage — it converts
directly into reach or amplifier count, which is what an optical engineer
cares about.

### Robustness — the strongest result

Train on one channel, test on another. Bottom row is matched (what most
papers report); the top row is what happens when the real link is worse than
the one you trained on.

| Trained on | CNN → severe | SVM → severe |
|---|---|---|
| clean (AWGN only) | **89.78 %** | **21.84 %** |
| realistic | 91.17 % | 71.98 % |
| severe (matched) | 97.84 % | 94.03 % |

Chance level is 20%. **The feature-based classifier collapses to essentially
random guessing** under a channel it never saw, while the CNN degrades
gracefully. The cumulants are statistics of an *assumed* signal model; change
the impairments and the learned boundaries stop corresponding to anything.

This is a deployment claim, not just an accuracy claim — and it is only
possible *because* the data is simulated.

### The other three experiments

| Experiment | Result |
|---|---|
| **Unseen OSNR** | CNN 97.45 ± 0.16 % on OSNR levels withheld from training — a drop of only 1.27 points. It learned the physics, not the noise levels. |
| **Capture length** | CNN with **512 symbols (94.08%) beats SVM with 4096 (93.82%)** — an 8× shorter capture still wins. |
| **Ablation** | All six variants within **1.26 points**, despite a 9.2× parameter range and 49× compute range. Capacity is not the binding constraint. |

### Reading the ablation honestly

The ablation used one seed, and the non-monotonic behaviour across widths
shows run-to-run noise of the same order as the differences. So the supported
claim is *"capacity is not the limiting factor"*, **not** *"4 channels is
optimal"*. Over-claiming from a single seed is exactly what a reviewer
catches.

---

## 9. How to check my work

```bash
python verify_project.py
```

It runs independent checks and **re-derives** results rather than trusting the
saved numbers:

| Check | What it proves |
|---|---|
| **Recompute accuracy** | Reloads the trained weights and recomputes test accuracy from scratch; must match the reported number to 9 decimals |
| **No data leakage** | Train/val/test share zero samples |
| **Constellation theory** | Simulator matches exact analytic cumulants, and square formats match published constants |
| **Noise power** | Measured SNR matches target to <0.02 dB; noise is Gaussian, zero-mean, equal in I and Q |
| **Stratification** | Every (format, OSNR) cell is split within 2% of 15% |
| **Shuffled-label test** | Train on randomised labels → accuracy collapses to chance |

That last one catches "too good to be true" results. If a model still scores
well after you destroy the labels, you have a leak.

**A note on how this script earned its keep:** it initially *failed* two noise
checks. Investigation showed **the check was wrong, not the data** — I was
measuring noise as `1.004 − 1.000`, a difference of two nearly-equal numbers,
which amplifies random scatter to ±0.87 dB. The fix was to measure noise
directly against the known transmitted symbols. Tolerance then tightened from
0.05 dB to 0.02 dB and everything passed. **When a check fails, find out which
side is broken before assuming.**

---

## 10. Viva questions with answers

**Q: What is OSNR, in one sentence?**
The ratio of optical signal power to noise power in dB, measured in a standard
0.1 nm reference bandwidth. High OSNR = clean signal.

**Q: Why does accuracy drop at low OSNR?**
Because the information genuinely isn't there any more. Noise smears each
constellation point into a cloud; once neighbouring clouds overlap, the grid
structure distinguishing 16-QAM from 64-QAM is physically erased. Look at
`figures/sample_constellations.png` at 5 dB — even a human expert can't tell
them apart. **No classifier can recover information the channel destroyed.**
That's a fundamental limit, not a weakness of the model.

**Q: What's the actual contribution of your paper?**
Existing work shows CNNs on constellation images beat SVMs on cumulants, but
that comparison changes two variables at once — the model becomes deep *and*
the input becomes an image. I separate them by adding an MLP on the same 15
features. Three different algorithms on identical features perform almost
identically, while changing the representation produces a large gain. So the
**representation** is the dominant design variable, not model capacity. That's
a more transferable claim than "CNNs are better."

**Q: Isn't it unfair to give the CNN more information?**
That *is* the finding, not a flaw. Both see the same physical signal; they
differ in how it's represented. Feeding hand-crafted features to a classical
classifier is exactly how MFI was done before deep learning, so it's the right
thing to compare against. The claim isn't "CNNs are better algorithms" — it's
"learning from the constellation image beats summarising it with hand-picked
statistics." Same signals, same training set, same test set.

**Q: Why did the raw-I/Q CNN do worst? It has the most information.**
Because information isn't the same as accessible structure. The symbols are
i.i.d., so their *order* carries almost nothing — but a 1-D convolution is
built to exploit order. The network has to estimate a *distribution* from a
*sequence*. The 2-D histogram already **is** a density estimate, so the binning
step does work the network would otherwise have to learn.

**Q: Which formats get confused, and why?**
The dense square grids — 16-QAM and 64-QAM. QPSK has 4 widely-spaced points and
stays distinguishable deep into noise. 8-QAM (star) and 32-QAM (cross) have
distinctive non-square shapes, which helps them. 16-QAM and 64-QAM differ only
in grid spacing, so once noise blurs the grid they look alike.

**Q: Why normalise every constellation to unit average power?**
Otherwise the formats would differ in brightness and any classifier could
identify them by measuring power alone — meaningless, because a real receiver
applies automatic gain control and delivers fixed power. Normalising forces the
model to learn shape.

**Q: Is OSNR the same as SNR?**
No. `SNR = OSNR × 2·B_ref/(p·R_s)`, which for our 32 GBaud single-polarisation
system means `SNR_dB = OSNR_dB − 1.07 dB`.

**Q: How do you know your simulator is correct?**
Fourth-order cumulants of noiseless QAM are exact finite sums over the
constellation points. The simulator reproduces them for all five formats to
within 0.004, and the three square formats additionally match the published
constants. That validates the signal model independently of any machine
learning.

**Q: Why apply frequency offset before carrier phase recovery?**
Because removing residual frequency offset is one of CPE's jobs — a real
receiver's phase tracker follows that slow rotation and cancels it. If you
apply it afterwards, nothing removes it, and even 1 MHz smears each point into
a 46° arc. I made exactly that mistake initially; the constellations looked
destroyed at 25 dB, which isn't what a working receiver produces.

**Q: Why only 3 convolution layers? Why not ResNet?**
Capacity should match the problem. Our images are synthetic, greyscale,
uncluttered, and the discriminating pattern is a simple repeating grid.
ResNet-50's 25 million parameters are built for real-world photographs. Also,
after 3 poolings one feature-map pixel already covers ~30×30 of the input — the
scale of the pattern. **And the ablation tests this**: 4 blocks gains nothing
over 3.

**Q: What is overfitting, and how did you prevent it?**
Memorising training examples instead of learning the general rule — training
accuracy rises while validation accuracy stalls or falls. Four defences:
dropout, weight decay, early stopping, and a large varied dataset.
`figures/fig5_training_curves.png` is the evidence.

**Q: Why do you report mean ± standard deviation?**
Because a single accuracy number could be luck. Reporting over independent
seeds lets a reviewer judge whether a difference is real. If two models' error
bars overlap heavily, you cannot claim one is better — and saying so honestly
is what makes the rest believable.

**Q: What does "stratified split" mean and why use it?**
Each (format, OSNR) cell is split 70/15/15 individually rather than splitting
the pool at random. Without it, chance could put more hard low-OSNR examples in
the test set, biasing the result. It also guarantees enough test samples at
*every* OSNR, which we need because we report accuracy per OSNR.

**Q: Why PyTorch and not TensorFlow?**
It was already installed; TensorFlow on Windows + Python 3.12 is fragile; and
most importantly PyTorch's training loop is written out explicitly, so every
step (forward → loss → backward → update) is visible in the code rather than
hidden inside `.fit()`.

**Q: Why is there no Softmax at the end of your network?**
`CrossEntropyLoss` applies softmax internally in a numerically more stable way
(it combines log and softmax to avoid overflow). Adding another would apply it
twice and hurt training. For probabilities, apply `torch.softmax()` to the
output manually.

**Q: Could this work on real experimental data?**
The pipeline transfers, but the model needs retraining or fine-tuning on real
captures — and the robustness experiment shows exactly why: a model trained on
a clean channel loses substantial accuracy on a dirtier one. Real signals also
carry impairments we didn't simulate: fibre nonlinearity, PMD, polarisation
multiplexing.

**Q: What would you do next?**
Experimental validation; polarisation-multiplexed transmission; joint
format-and-OSNR estimation (multi-task learning); testing transfer across
symbol rates; and compressing the network for FPGA deployment — the ablation
already shows most of its capacity is unnecessary.

---

## 11. Honest limitations

State these in the paper. Reviewers respect a clear limitations section, and
examiners often ask about them directly.

1. **Simulation only** — no experimental captures.
2. **Single polarisation** — commercial long-haul is polarisation-multiplexed.
3. **Linear channel** — no fibre nonlinearity or PMD. Nonlinear noise is not
   Gaussian, so cumulant-based features would behave differently.
4. **Fixed symbol rate** — all training and testing at 32 GBaud; transfer to
   other rates is untested.
5. **Five formats** — a deployed system might need more, and accuracy
   typically falls as classes are added.
6. **CD modelled at symbol rate** — captures the ISI at decision instants
   rather than the full continuous waveform. Accurate for small residual CD,
   which is the realistic case, but it is a simplification.
7. **SVM trained on a capped subsample** — RBF-SVM cost grows ~quadratically
   with sample count. Its accuracy on 15 low-dimensional features saturates
   quickly, so this costs it very little, but it should be stated.
8. **Citations not verified** — see `paper/README.md`. You must check every
   reference against the publisher before submitting.

---

## Reproducibility

Every random number generator is seeded from `config.SEED = 42`:

- **numpy** — the simulation and the train/val/test split
- **python `random`** — miscellaneous
- **torch** — weight initialisation and dropout

Each (format, OSNR) block gets its own deterministic seed, so changing the
number of OSNR levels doesn't reshuffle everything else. Multi-seed
experiments use 42, 43, 44.

Verified in practice: a standalone benchmark and the full run produced
identical epoch-1 numbers, and `verify_project.py` recomputes the reported
accuracy from the saved weights and confirms it matches.
