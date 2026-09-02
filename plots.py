"""
plots.py
========
STEP 6: turn the saved numbers into the figures for the paper.

This file does NO computation and NO training. It only reads the JSON files
written by train_and_evaluate.py and experiments.py. Keeping plotting separate
means you can restyle a figure for a conference template without re-running an
hour-long experiment.

Figures produced (all in figures/, all 300 dpi):

    fig1_accuracy_vs_osnr.png       THE headline figure: all 5 models
    fig2_representation.png         the paper's central claim, in one chart
    fig3_confusion_matrices.png     CNN vs SVM, where the errors are
    fig4_per_class_vs_osnr.png      which format fails first
    fig5_training_curves.png        evidence that training was healthy
    fig6_ablation_pareto.png        accuracy vs model cost
    fig7_symbols.png                accuracy vs capture length
    fig8_robustness.png             train on one channel, test on another
    fig9_generalisation.png         unseen OSNR levels

Run me:
    python plots.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")        # write files, never try to open a window
import matplotlib.pyplot as plt
import numpy as np

import config

# Conference papers are printed small, so use readable fonts and clear markers
# rather than matplotlib's thin defaults.
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.dpi": 300,          # 300 dpi is the usual print minimum
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour-blind-safe palette, with distinct marker shapes so the figures still
# read correctly when printed in black and white.
STYLE = {
    "cnn":   dict(color="#0072B2", marker="o", ls="-",   lw=2.2, ms=6),
    "iqcnn": dict(color="#CC79A7", marker="v", ls=":",   lw=1.8, ms=5),
    "mlp":   dict(color="#E69F00", marker="D", ls="--",  lw=1.8, ms=5),
    "svm":   dict(color="#D55E00", marker="s", ls="--",  lw=1.8, ms=5),
    "knn":   dict(color="#009E73", marker="^", ls="-.",  lw=1.8, ms=5),
}
ORDER = ["cnn", "iqcnn", "mlp", "svm", "knn"]
LABEL = {"cnn": "CNN (image)", "iqcnn": "IQ-CNN (raw I/Q)",
         "mlp": "MLP (features)", "svm": "SVM (features)",
         "knn": "k-NN (features)"}

CHANCE = 100.0 / config.N_CLASSES


def _load(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    path = os.path.join(config.FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  saved " + path)


# ===========================================================================
# FIGURE 1 -- ACCURACY vs OSNR, WITH ERROR BARS
# ===========================================================================
def fig1_accuracy_vs_osnr(res):
    """
    The single most important figure in the paper.

    HOW TO READ IT: x-axis is signal quality (right = clean), y-axis is how
    often the classifier is right. Every curve MUST fall as you move left,
    because low OSNR genuinely destroys the information -- that is physics,
    not a bug. The paper's claim is that the CNN's curve stays higher for
    longer.

    The shaded band is +/- one standard deviation over the random seeds. If
    two models' bands overlap heavily, you cannot claim one is better than the
    other, and saying so honestly is what makes the rest believable.

    The dotted line is the random-guessing floor: with 5 equally likely
    classes a model that knows nothing still gets 20% right. Always compare
    against that floor, never against 0%.
    """
    agg = res["aggregate"]
    osnr = config.OSNR_LIST
    fig, ax = plt.subplots(figsize=(7.2, 4.9))

    for kind in ORDER:
        if kind not in agg:
            continue
        mean = np.array([100 * agg[kind]["accuracy_per_osnr"][str(o)]["mean"]
                         for o in osnr])
        std = np.array([100 * agg[kind]["accuracy_per_osnr"][str(o)]["std"]
                        for o in osnr])
        ax.plot(osnr, mean, label=LABEL[kind], **STYLE[kind])
        ax.fill_between(osnr, mean - std, mean + std,
                        color=STYLE[kind]["color"], alpha=0.15, lw=0)

    ax.axhline(CHANCE, color="grey", ls=":", lw=1.5)
    ax.text(config.OSNR_DB_MIN + 0.2, CHANCE + 1.5,
            "random guess ({:.0f}%)".format(CHANCE),
            ha="left", va="bottom", fontsize=9, color="grey")

    ax.set_xlabel("OSNR (dB)")
    ax.set_ylabel("Identification accuracy (%)")
    ax.set_title("Modulation format identification accuracy vs OSNR\n"
                 "({} formats, mean $\\pm$ 1 s.d. over {} seeds)".format(
                     config.N_CLASSES, len(res["seeds"])), fontsize=11)
    ax.set_xticks(osnr[::2])
    ax.set_ylim(CHANCE - 8, 102)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig1_accuracy_vs_osnr.png")


# ===========================================================================
# FIGURE 2 -- THE REPRESENTATION STUDY  (the paper's central claim)
# ===========================================================================
def fig2_representation(res):
    """
    The figure that carries the paper's argument.

    Models are grouped by what they SEE, not by what they ARE. Within the
    "15 hand-crafted features" group sit a deep neural network (MLP), a kernel
    method (SVM) and a nearest-neighbour lookup (k-NN) -- three completely
    different learning algorithms. If those three cluster together while the
    image-based CNN sits well above them, then the conclusion is forced:

        what the model is given matters far more than what the model is.

    That is a stronger and more transferable claim than "CNNs beat SVMs", and
    it is the thing this paper contributes.
    """
    agg = res["aggregate"]
    groups = [("Raw I/Q\n(no pre-processing)", ["iqcnn"]),
              ("2-D histogram image\n(binning only)", ["cnn"]),
              ("15 hand-crafted features\n(expert pre-processing)",
               ["mlp", "svm", "knn"])]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs, heights, errs, colors, names = [], [], [], [], []
    pos, ticks, tick_labels = 0, [], []

    for gname, kinds in groups:
        kinds = [k for k in kinds if k in agg]
        if not kinds:
            continue
        start = pos
        for k in kinds:
            xs.append(pos)
            heights.append(100 * agg[k]["accuracy_mean"])
            errs.append(100 * agg[k]["accuracy_std"])
            colors.append(STYLE[k]["color"])
            names.append(LABEL[k].split(" (")[0])
            pos += 1
        ticks.append((start + pos - 1) / 2)
        tick_labels.append(gname)
        pos += 0.8

    ax.bar(xs, heights, yerr=errs, capsize=4, color=colors,
           width=0.75, edgecolor="black", linewidth=0.6)
    for x, h, e, n in zip(xs, heights, errs, names):
        # Clear the error bar's upper cap, not just the bar top, so the value
        # label never sits on top of the whisker.
        ax.text(x, h + e + 1.4, "{:.1f}%".format(h), ha="center", fontsize=9,
                fontweight="bold")
        ax.text(x, 3, n, ha="center", fontsize=8.5, color="white",
                rotation=90, va="bottom", fontweight="bold")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_ylabel("Overall test accuracy (%)")
    ax.set_ylim(0, max(h + e for h, e in zip(heights, errs)) + 10)
    ax.axhline(CHANCE, color="grey", ls=":", lw=1.4)
    # Label on the LEFT, above the line: on the right it collides with the
    # last bar, and below the line it collides with the axis.
    ax.text(xs[0] - 0.45, CHANCE + 1.2,
            "random guess ({:.0f}%)".format(CHANCE),
            ha="left", va="bottom", fontsize=8.5, color="grey")
    ax.set_title("What the model SEES matters more than what the model IS\n"
                 "(three different learning algorithms on identical features "
                 "all perform alike)", fontsize=11)
    ax.grid(axis="x")
    fig.tight_layout()
    _save(fig, "fig2_representation.png")


# ===========================================================================
# FIGURE 3 -- CONFUSION MATRICES
# ===========================================================================
def _draw_confusion(ax, cm, title, show_ylabel=True):
    """
    A confusion matrix answers "when the truth was X, what did the model say?".
    Row = truth, column = prediction. The diagonal is correct answers;
    everything off it is a mistake, and WHICH off-diagonal cell is large tells
    you which two formats get mixed up. Rows are normalised to 100% so classes
    are directly comparable.
    """
    cm = np.asarray(cm, dtype=float)
    pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    n = config.N_CLASSES
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted")
    if show_ylabel:
        ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if pct[i, j] < 0.5:
                continue
            ax.text(j, i, "{:.0f}".format(pct[i, j]), ha="center", va="center",
                    fontsize=8.5,
                    color="white" if pct[i, j] > 55 else "black")


def fig3_confusion(res):
    agg = res["aggregate"]
    kinds = [k for k in ("cnn", "mlp", "svm") if k in agg]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.1 * len(kinds), 4.0))
    if len(kinds) == 1:
        axes = [axes]
    for ax, k in zip(axes, kinds):
        _draw_confusion(ax, agg[k]["confusion_matrix_mean"],
                        "{}\n{:.2f}%".format(LABEL[k],
                                             100 * agg[k]["accuracy_mean"]),
                        show_ylabel=(k == kinds[0]))
    fig.suptitle("Confusion matrices on the same test set "
                 "(row-normalised %, averaged over seeds)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_confusion_matrices.png")


# ===========================================================================
# FIGURE 4 -- PER-CLASS ACCURACY vs OSNR
# ===========================================================================
def fig4_per_class(res):
    """
    Which modulation format fails first as the noise rises.

    This is the figure that EXPLAINS the headline result. QPSK stays accurate
    deep into the noise because its 4 points are far apart; the dense grids
    collapse first because noise blurs them into each other.
    """
    import torch
    from cnn_model import ConstellationCNN
    import train_and_evaluate as te

    seed = res["seeds"][0]
    ckpt = os.path.join(config.RESULTS_DIR, "cnn_best_seed{}.pt".format(seed))
    if not os.path.exists(ckpt):
        print("  (skipping fig4: no saved CNN)")
        return

    data = te.load_data(need_iq=False)
    _, _, idx_test = te.make_splits(data["y"], data["osnr"], seed=seed)
    model = ConstellationCNN()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    X = torch.from_numpy(data["img"]).unsqueeze(1)
    y_pred = te.predict_torch("cnn", model, X, idx_test)
    y_true, o_true = data["y"][idx_test], data["osnr"][idx_test]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    marks = ["o", "s", "^", "D", "v"]
    for c, name in enumerate(config.CLASS_NAMES):
        accs = []
        for o in config.OSNR_LIST:
            m = (o_true == o) & (y_true == c)
            accs.append(100 * float(np.mean(y_pred[m] == y_true[m]))
                        if m.any() else np.nan)
        ax.plot(config.OSNR_LIST, accs, marker=marks[c % 5],
                color=colors[c % 5], lw=2, ms=5, label=name)

    ax.axhline(CHANCE, color="grey", ls=":", lw=1.4)
    ax.set_xlabel("OSNR (dB)")
    ax.set_ylabel("CNN accuracy (%)")
    ax.set_title("CNN accuracy per modulation format")
    ax.set_xticks(config.OSNR_LIST[::2])
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig4_per_class_vs_osnr.png")


# ===========================================================================
# FIGURE 5 -- TRAINING CURVES
# ===========================================================================
def fig5_training_curves(res):
    """
    Evidence that training behaved properly. A reviewer looks for two things:
      * loss goes DOWN and accuracy goes UP (the model is learning);
      * the validation curve tracks the training curve. If training accuracy
        kept climbing while validation accuracy fell, that is OVERFITTING.
        Dropout, weight decay and early stopping exist to prevent exactly that,
        and this figure is where you show they worked.
    """
    seed = str(res["seeds"][0])
    per_seed = res["per_seed"][seed]
    kinds = [k for k in ("cnn", "iqcnn", "mlp") if k in per_seed
             and "history" in per_seed[k]]
    if not kinds:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in kinds:
        h = per_seed[k]["history"]
        ep = h["epoch"]
        axes[0].plot(ep, h["train_loss"], color=STYLE[k]["color"], ls="-",
                     marker="o", ms=3, label="{} train".format(LABEL[k]))
        axes[0].plot(ep, h["val_loss"], color=STYLE[k]["color"], ls="--",
                     marker="s", ms=3, label="{} val".format(LABEL[k]))
        axes[1].plot(ep, [100 * a for a in h["train_acc"]],
                     color=STYLE[k]["color"], ls="-", marker="o", ms=3)
        axes[1].plot(ep, [100 * a for a in h["val_acc"]],
                     color=STYLE[k]["color"], ls="--", marker="s", ms=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss (lower is better)")
    axes[0].legend(fontsize=7.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy (higher is better)\nsolid = train, dashed = validation")
    fig.suptitle("Training history (seed {})".format(seed), fontsize=12)
    fig.tight_layout()
    _save(fig, "fig5_training_curves.png")


# ===========================================================================
# FIGURE 6 -- ABLATION / COMPLEXITY PARETO
# ===========================================================================
def fig6_ablation(abl):
    """
    Accuracy against model cost.

    WHY THIS FIGURE MATTERS: an MFI block would eventually live inside a
    receiver's FPGA or DSP, where compute is scarce. Accuracy alone does not
    decide what to deploy -- accuracy per operation does. The useful reading
    is where the curve flattens: beyond that point you are paying for
    parameters that buy you nothing.
    """
    rows = abl["results"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    accs = [100 * r["accuracy"] for r in rows]

    # ------------------------------------------------------------------
    # AXIS RANGE -- this matters, and getting it wrong is a real hazard.
    # Autoscaling puts these six points on a ~0.25-point y-axis, which makes
    # a negligible difference look like a strong trend and contradicts the
    # conclusion the data actually supports ("capacity is not the binding
    # constraint"). We therefore fix a wider, honest range shared by both
    # panels, so the flatness is visible rather than magnified away.
    # ------------------------------------------------------------------
    lo = min(min(accs) - 2.5, 95.0)
    hi = 100.4

    d = sorted([r for r in rows if r["base_ch"] == 16],
               key=lambda r: r["n_blocks"])
    ax1.plot([r["n_blocks"] for r in d], [100 * r["accuracy"] for r in d],
             "o-", color="#0072B2", lw=2, ms=7)
    for r in d:
        ax1.annotate("{:.2f}%\n{:,} par".format(100 * r["accuracy"],
                                                r["n_parameters"]),
                     (r["n_blocks"], 100 * r["accuracy"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=7.5, color="#444444")
    ax1.set_xlabel("Number of convolution blocks")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("Depth (width fixed at 16 channels)", fontsize=10.5)
    ax1.set_xticks([r["n_blocks"] for r in d])
    ax1.set_xlim(1.6, 4.4)
    ax1.set_ylim(lo, hi)

    w = sorted([r for r in rows if r["n_blocks"] == 3],
               key=lambda r: r["base_ch"])
    ax2.plot([r["macs"] / 1e6 for r in w], [100 * r["accuracy"] for r in w],
             "s-", color="#D55E00", lw=2, ms=7)
    for r in w:
        ax2.annotate("{} ch".format(r["base_ch"]),
                     (r["macs"] / 1e6, 100 * r["accuracy"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=8, color="#444444")
    ax2.set_xlabel("Compute per inference (millions of MACs)")
    ax2.set_title("Width (depth fixed at 3 blocks)", fontsize=10.5)
    ax2.set_xscale("log")
    ax2.set_ylim(lo, hi)

    # Show the full spread across ALL variants, which is the actual finding.
    for ax in (ax1, ax2):
        ax.axhspan(min(accs), max(accs), color="grey", alpha=0.13, lw=0)
    ax2.text(0.97, 0.06,
             "shaded: full spread across all six\nvariants = {:.2f} points"
             .format(max(accs) - min(accs)),
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8, color="#555555")

    fig.suptitle("Architecture ablation: accuracy is insensitive to model "
                 "capacity", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig6_ablation_pareto.png")


# ===========================================================================
# FIGURE 7 -- ACCURACY vs CAPTURE LENGTH
# ===========================================================================
def fig7_symbols(sym):
    """
    How many received symbols are needed for reliable identification.

    A monitoring receiver gets a short capture, so this is the curve a system
    designer needs. The interesting number to quote is the smallest capture
    that still reaches your accuracy target.
    """
    rows = sorted(sym["results"], key=lambda r: r["n_symbols"])
    ns = [r["n_symbols"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ns, [100 * r["cnn_accuracy"] for r in rows], "o-",
            color="#0072B2", lw=2.2, ms=6, label="CNN (image)")
    ax.plot(ns, [100 * r["svm_accuracy"] for r in rows], "s--",
            color="#D55E00", lw=1.8, ms=5, label="SVM (features)")
    ax.axhline(CHANCE, color="grey", ls=":", lw=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Symbols per capture")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("How short a capture is enough?\n"
                 "(all OSNR levels pooled)", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig7_symbols.png")


# ===========================================================================
# FIGURE 8 -- ROBUSTNESS TO UNSEEN IMPAIRMENTS
# ===========================================================================
def fig8_robustness(rob):
    """
    Train on one channel model, test on another.

    HOW TO READ IT: rows are what the model was TRAINED on, columns what it
    was TESTED on. The diagonal is the easy, matched case that most papers
    report. The off-diagonal cells are the honest ones -- they say what
    happens when the real link is dirtier than the simulation you trained on.
    A big drop from the diagonal is a warning that simulation-trained MFI will
    not transfer; a small drop is a genuinely strong claim.
    """
    profs = rob["profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, model in zip(axes, ("cnn", "svm")):
        M = np.array([[100 * rob["matrix"][tr][te][model] for te in profs]
                      for tr in profs])
        im = ax.imshow(M, cmap="RdYlGn", vmin=40, vmax=100)
        ax.set_xticks(range(len(profs))); ax.set_yticks(range(len(profs)))
        ax.set_xticklabels(profs); ax.set_yticklabels(profs)
        ax.set_xlabel("TESTED on"); ax.set_ylabel("TRAINED on")
        ax.set_title(LABEL[model])
        ax.grid(False)
        for i in range(len(profs)):
            for j in range(len(profs)):
                ax.text(j, i, "{:.1f}".format(M[i, j]), ha="center",
                        va="center", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, label="accuracy (%)")
    fig.suptitle("Robustness: does a model trained on one channel survive "
                 "another?", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig8_robustness.png")


# ===========================================================================
# FIGURE 9 -- GENERALISATION TO UNSEEN OSNR
# ===========================================================================
def fig9_generalisation(gen, main=None):
    """
    Accuracy on OSNR levels never seen during training, next to the matched
    case. Small gap = the model learned the physics, not the noise levels.
    """
    kinds = [k for k in ("cnn", "mlp", "svm") if k in gen["summary"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(kinds))
    w = 0.36

    unseen = [100 * gen["summary"][k]["mean"] for k in kinds]
    unseen_e = [100 * gen["summary"][k]["std"] for k in kinds]
    ax.bar(x - w / 2, unseen, w, yerr=unseen_e, capsize=4,
           label="tested on UNSEEN OSNR levels",
           color=[STYLE[k]["color"] for k in kinds], edgecolor="black", lw=0.6)

    matched = None
    if main:
        matched = [100 * main["aggregate"][k]["accuracy_mean"] for k in kinds]
        matched_e = [100 * main["aggregate"][k]["accuracy_std"] for k in kinds]
        ax.bar(x + w / 2, matched, w, yerr=matched_e, capsize=4,
               label="tested on TRAINED OSNR levels (for reference)",
               color=[STYLE[k]["color"] for k in kinds], alpha=0.45,
               edgecolor="black", lw=0.6, hatch="//")

    # Label BOTH bars, and state the drop -- the drop is the actual finding.
    for i, xi in enumerate(x):
        ax.text(xi - w / 2, unseen[i] + 1.2, "{:.1f}".format(unseen[i]),
                ha="center", fontsize=8.5, fontweight="bold")
        if matched:
            ax.text(xi + w / 2, matched[i] + 1.2, "{:.1f}".format(matched[i]),
                    ha="center", fontsize=8.5, color="#555555")
            ax.text(xi, 6, "drop\n{:.2f}".format(matched[i] - unseen[i]),
                    ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in kinds], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Generalisation to OSNR levels never seen in training\n"
                 "(trained on even OSNRs, tested on odd)", fontsize=11,
                 pad=28)
    # Legend ABOVE the axes: inside, it sits on top of the bars.
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.015),
              ncol=1, frameon=False)
    ax.grid(axis="x")
    fig.tight_layout()
    _save(fig, "fig9_generalisation.png")


# ===========================================================================
# TEXT SUMMARY -- numbers to paste straight into the paper
# ===========================================================================
def write_summary(res, gen=None, abl=None, sym=None, rob=None):
    agg = res["aggregate"]
    osnr = config.OSNR_LIST
    L = []
    A = L.append

    A("=" * 78)
    A("  RESULTS SUMMARY")
    A("  CNN-Based Modulation Format Identification for Optical Performance")
    A("  Monitoring")
    A("=" * 78)
    A("")
    A("  Dataset : {:,} constellation images, {} formats, OSNR {}-{} dB".format(
        res["n_total"], config.N_CLASSES, min(osnr), max(osnr)))
    A("  Formats : {}".format(", ".join(config.CLASS_NAMES)))
    A("  Seeds   : {}  (all results are mean +/- s.d. over these)".format(
        res["seeds"]))
    A("")

    A("-" * 78)
    A("  TABLE I -- OVERALL TEST ACCURACY BY INPUT REPRESENTATION")
    A("-" * 78)
    A("  {:<20s} {:<26s} {:>16s} {:>10s}".format(
        "Model", "Input representation", "Accuracy (%)", "Params"))
    reps = {"cnn": "64x64 constellation image", "iqcnn": "512 raw I/Q samples",
            "mlp": "15 statistical features", "svm": "15 statistical features",
            "knn": "15 statistical features"}
    for k in ORDER:
        if k not in agg:
            continue
        a = agg[k]
        pars = "{:,}".format(a["n_parameters"]) if a["n_parameters"] else "n/a"
        A("  {:<20s} {:<26s} {:>9.2f} +/- {:<4.2f} {:>10s}".format(
            LABEL[k], reps[k], 100 * a["accuracy_mean"],
            100 * a["accuracy_std"], pars))
    A("")

    # --- the central claim, stated numerically ---
    if all(k in agg for k in ("cnn", "mlp", "svm", "knn")):
        feat = [100 * agg[k]["accuracy_mean"] for k in ("mlp", "svm", "knn")]
        A("  THE CENTRAL RESULT")
        A("    Three different learning algorithms on the SAME 15 features:")
        A("      MLP (deep)      {:.2f} %".format(feat[0]))
        A("      SVM (kernel)    {:.2f} %".format(feat[1]))
        A("      k-NN (lazy)     {:.2f} %".format(feat[2]))
        A("      spread          {:.2f} percentage points".format(
            max(feat) - min(feat)))
        A("    The SAME kind of network on the constellation IMAGE:")
        A("      CNN             {:.2f} %".format(100 * agg["cnn"]["accuracy_mean"]))
        A("      gain over best feature-based model: {:+.2f} points".format(
            100 * agg["cnn"]["accuracy_mean"] - max(feat)))
        A("")
        A("    Changing the ALGORITHM moves accuracy by {:.1f} points.".format(
            max(feat) - min(feat)))
        A("    Changing the REPRESENTATION moves it by {:.1f} points.".format(
            100 * agg["cnn"]["accuracy_mean"] - max(feat)))
        A("    -> the representation is the dominant factor.")
        A("")

    A("-" * 78)
    A("  TABLE II -- ACCURACY (%) vs OSNR")
    A("-" * 78)
    hdr = "  {:>8s}".format("OSNR")
    for k in ORDER:
        if k in agg:
            hdr += " {:>16s}".format(LABEL[k].split(" (")[0])
    A(hdr)
    for o in osnr:
        line = "  {:>8d}".format(o)
        for k in ORDER:
            if k in agg:
                d = agg[k]["accuracy_per_osnr"][str(o)]
                line += " {:>9.2f}+/-{:<4.2f}".format(100 * d["mean"],
                                                      100 * d["std"])
        A(line)
    A("")

    # --- OSNR thresholds ---
    A("-" * 78)
    A("  TABLE III -- MINIMUM OSNR TO REACH A TARGET ACCURACY")
    A("-" * 78)

    def first_above(k, th):
        for o in osnr:
            if agg[k]["accuracy_per_osnr"][str(o)]["mean"] >= th:
                return o
        return None

    A("  {:<20s} {:>12s} {:>12s}".format("Model", ">= 95%", ">= 99%"))
    for k in ORDER:
        if k not in agg:
            continue
        a95, a99 = first_above(k, 0.95), first_above(k, 0.99)
        A("  {:<20s} {:>12s} {:>12s}".format(
            LABEL[k],
            "{} dB".format(a95) if a95 else "not reached",
            "{} dB".format(a99) if a99 else "not reached"))
    if "cnn" in agg and "svm" in agg:
        c, s = first_above("cnn", 0.95), first_above("svm", 0.95)
        if c and s:
            A("")
            A("  -> the CNN reaches 95% accuracy {} dB earlier than the SVM.".format(s - c))
            A("     In optical terms that is an OSNR margin, which converts")
            A("     directly into longer reach or fewer amplifiers.")
    A("")

    # --- regimes ---
    A("  Mean accuracy by regime (%)")
    A("  {:<20s} {:>14s} {:>14s}".format("Model", "low (5-10 dB)", "high (20-25 dB)"))
    for k in ORDER:
        if k not in agg:
            continue
        lo = np.mean([agg[k]["accuracy_per_osnr"][str(o)]["mean"]
                      for o in osnr if o <= 10])
        hi = np.mean([agg[k]["accuracy_per_osnr"][str(o)]["mean"]
                      for o in osnr if o >= 20])
        A("  {:<20s} {:>14.2f} {:>14.2f}".format(LABEL[k], 100 * lo, 100 * hi))
    A("")

    # --- confusion ---
    if "cnn" in agg:
        A("-" * 78)
        A("  TABLE IV -- CNN CONFUSION MATRIX (row-normalised %, seed-averaged)")
        A("-" * 78)
        cm = np.array(agg["cnn"]["confusion_matrix_mean"], dtype=float)
        pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
        A("  {:<10s}".format("true\\pred")
          + "".join("{:>9s}".format(n) for n in config.CLASS_NAMES))
        for i, n in enumerate(config.CLASS_NAMES):
            A("  {:<10s}".format(n)
              + "".join("{:>9.2f}".format(v) for v in pct[i]))
        A("")

    # --- extra experiments ---
    if gen:
        A("-" * 78)
        A("  TABLE V -- GENERALISATION TO UNSEEN OSNR LEVELS")
        A("-" * 78)
        A("  Trained on even OSNRs {}".format(gen["train_osnr"]))
        A("  Tested  on odd  OSNRs {}".format(gen["test_osnr"]))
        A("")
        for k, v in gen["summary"].items():
            matched = (100 * agg[k]["accuracy_mean"]) if k in agg else float("nan")
            A("  {:<20s} unseen {:.2f} +/- {:.2f} %   (matched {:.2f} %, "
              "drop {:.2f})".format(LABEL[k], 100 * v["mean"], 100 * v["std"],
                                    matched, matched - 100 * v["mean"]))
        A("")

    if abl:
        A("-" * 78)
        A("  TABLE VI -- ARCHITECTURE ABLATION")
        A("-" * 78)
        A("  {:>8s} {:>9s} {:>12s} {:>14s} {:>12s}".format(
            "blocks", "base_ch", "accuracy %", "params", "MACs"))
        for r in abl["results"]:
            A("  {:>8d} {:>9d} {:>12.2f} {:>14,d} {:>12,d}".format(
                r["n_blocks"], r["base_ch"], 100 * r["accuracy"],
                r["n_parameters"], r["macs"]))
        A("")

    if sym:
        A("-" * 78)
        A("  TABLE VII -- ACCURACY vs CAPTURE LENGTH")
        A("-" * 78)
        A("  {:>10s} {:>12s} {:>12s}".format("symbols", "CNN %", "SVM %"))
        for r in sorted(sym["results"], key=lambda r: r["n_symbols"]):
            A("  {:>10d} {:>12.2f} {:>12.2f}".format(
                r["n_symbols"], 100 * r["cnn_accuracy"],
                100 * r["svm_accuracy"]))
        A("")

    if rob:
        A("-" * 78)
        A("  TABLE VIII -- ROBUSTNESS (train profile -> test profile)")
        A("-" * 78)
        profs = rob["profiles"]
        A("  CNN")
        A("  {:<12s}".format("train\\test")
          + "".join("{:>12s}".format(p) for p in profs))
        for tr in profs:
            A("  {:<12s}".format(tr)
              + "".join("{:>12.2f}".format(100 * rob["matrix"][tr][t]["cnn"])
                        for t in profs))
        A("")
        A("  SVM")
        A("  {:<12s}".format("train\\test")
          + "".join("{:>12s}".format(p) for p in profs))
        for tr in profs:
            A("  {:<12s}".format(tr)
              + "".join("{:>12.2f}".format(100 * rob["matrix"][tr][t]["svm"])
                        for t in profs))
        A("")

    text = "\n".join(L)
    path = os.path.join(config.RESULTS_DIR, "results_summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print("  saved " + path)
    return text


# ===========================================================================
if __name__ == "__main__":
    print("Generating figures...")
    main = _load("main_results.json")
    if main is None:
        raise SystemExit("No main_results.json. Run:\n"
                         "    python train_and_evaluate.py")

    fig1_accuracy_vs_osnr(main)
    fig2_representation(main)
    fig3_confusion(main)
    fig4_per_class(main)
    fig5_training_curves(main)

    gen = _load("exp_generalise.json")
    abl = _load("exp_ablation.json")
    sym = _load("exp_symbols.json")
    rob = _load("exp_robustness.json")

    if abl:
        fig6_ablation(abl)
    if sym:
        fig7_symbols(sym)
    if rob:
        fig8_robustness(rob)
    if gen:
        fig9_generalisation(gen, main)

    print("\nWriting results summary...")
    print()
    print(write_summary(main, gen, abl, sym, rob))


# ===========================================================================
# FIGURE 10 -- FAILURE SELF-DETECTION UNDER CHANNEL MISMATCH
# ===========================================================================
def fig10_confidence(conf_ideal, conf_real=None):
    """
    Does the model know when it is failing, and can gating on confidence keep
    it usable?

    LEFT: accuracy against stated confidence on the unseen 'severe' channel.
    A model on the diagonal is honest; one above it is overconfident.

    RIGHT: the practical consequence. If the monitor refuses to answer below a
    confidence threshold, what fraction of inputs does it still answer, and how
    accurate is it on those? A monitor that is only safe when it answers 0.2%
    of the time is not a monitor.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    order = [("cnn", "CNN (image)"), ("mlp", "MLP (features)"),
             ("svm", "SVM (features)")]

    # ---- left: accuracy vs confidence, severe channel ----
    for k, lab in order:
        d = conf_ideal["models"][k]["severe"]
        ax1.scatter(d["mean_confidence"] * 100, d["accuracy"] * 100,
                    s=150, color=STYLE[k]["color"], marker=STYLE[k]["marker"],
                    edgecolor="black", linewidth=0.8, zorder=3, label=lab)
        ax1.annotate("{:+.1f} pts\noverconfident".format(
                         100 * d["overconfidence"]),
                     (d["mean_confidence"] * 100, d["accuracy"] * 100),
                     textcoords="offset points", xytext=(10, -18),
                     fontsize=8, color=STYLE[k]["color"])
    lims = [15, 105]
    ax1.plot(lims, lims, ls=":", color="grey", lw=1.4, zorder=1)
    ax1.text(97, 92, "honest\n(conf = acc)", fontsize=8.5, color="grey",
             ha="right")
    ax1.axhline(CHANCE, color="#B00", ls="--", lw=1.2, zorder=1)
    ax1.text(18, CHANCE + 1.5, "chance ({:.0f}%)".format(CHANCE),
             fontsize=8.5, color="#B00")
    ax1.set_xlim(*lims); ax1.set_ylim(*lims)
    ax1.set_xlabel("Mean stated confidence (%)")
    ax1.set_ylabel("Actual accuracy (%)")
    ax1.set_title("Is the model honest?\n(trained on clean, tested on severe)",
                  fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="upper left")

    # ---- right: coverage vs accuracy under confidence gating ----
    for k, lab in order:
        rows = conf_ideal["thresholding"][k]
        cov = [100 * r["coverage"] for r in rows]
        acc = [100 * r["accuracy_on_kept"] for r in rows]
        ax2.plot(cov, acc, marker=STYLE[k]["marker"], color=STYLE[k]["color"],
                 lw=2, ms=6, label=lab)
        # mark the 90%-confidence operating point
        for r, c, a in zip(rows, cov, acc):
            if abs(r["threshold"] - 0.9) < 1e-9:
                ax2.scatter([c], [a], s=200, facecolor="none",
                            edgecolor=STYLE[k]["color"], linewidth=2.2,
                            zorder=4)
                ax2.annotate("{:.1f}% answered".format(c), (c, a),
                             textcoords="offset points", xytext=(6, -16),
                             fontsize=8, color=STYLE[k]["color"])
    ax2.set_xlabel("Coverage: inputs the monitor still answers (%)")
    ax2.set_ylabel("Accuracy on answered inputs (%)")
    ax2.set_title("Cost of making it safe\n(circles = 90% confidence gate)",
                  fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="lower left")

    fig.suptitle("Failure self-detection under channel mismatch", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig10_confidence.png")


# ===========================================================================
# FIGURE 11 -- IMPAIRMENT TOLERANCE BUDGET
# ===========================================================================
def fig11_tolerance(tol):
    """
    How far can each impairment drift from its assumed value before the
    monitor stops working?

    One panel per impairment. The vertical dashed line marks the value the
    models were trained on; the horizontal line marks a 90% accuracy target.
    Where a curve crosses the target is that model's tolerance limit.

    Read the panels against each other: the two representations do NOT fail
    in the same way, and neither dominates. That asymmetry is the finding.
    """
    sweeps = tol["sweeps"]
    keys = [k for k in ("laser_linewidth_hz", "freq_offset_hz",
                        "iq_gain_imbalance_db", "iq_phase_skew_deg",
                        "residual_cd_ps_nm") if k in sweeps]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        s = sweeps[k]
        pts = sorted(s["points"], key=lambda r: r["value"])
        x = [p["display_value"] for p in pts]
        ax.plot(x, [100 * p["cnn_accuracy"] for p in pts], "o-",
                color=STYLE["cnn"]["color"], lw=2, ms=5, label="CNN (image)")
        ax.plot(x, [100 * p["svm_accuracy"] for p in pts], "s--",
                color=STYLE["svm"]["color"], lw=1.8, ms=5, label="SVM (features)")
        ax.axvline(s["trained_at"] * s["scale"], color="grey", ls=":", lw=1.4)
        ax.axhline(90, color="#B00", ls="-.", lw=1.1, alpha=0.7)
        ax.axhline(CHANCE, color="grey", ls=":", lw=1.0, alpha=0.6)
        ax.set_title("{}\n({})".format(s["label"], s["unit"]), fontsize=9.5)
        ax.set_xlabel(s["unit"])
        ax.set_ylim(10, 103)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].text(0.03, 0.55, "90% target", transform=axes[0].transAxes,
                 fontsize=7.5, color="#B00")

    fig.suptitle("Impairment tolerance: how far each parameter may drift from "
                 "its assumed value\n(dotted vertical line = value the models "
                 "were trained on)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig11_tolerance.png")
