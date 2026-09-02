"""
baselines.py
============
STEP 3: the traditional machine-learning comparison methods.

WHY DO WE NEED BASELINES AT ALL?
A paper that only says "my CNN got 95%" proves nothing. 95% might be easy.
The reviewer's first question is always "compared to what?". So we implement
the methods an engineer would have used BEFORE deep learning, run them on
exactly the same signals, and show the CNN does better. That difference is
the actual contribution of the paper.

THE KEY DIFFERENCE BETWEEN THE TWO APPROACHES
    CNN      : raw 64x64 image  ->  network invents its own features  -> class
    SVM/k-NN : raw I/Q samples  ->  HUMAN-chosen 15 features          -> class

The baselines are not "worse algorithms". They are given a *summary* of the
signal instead of the whole picture. Fifteen numbers cannot capture
everything a 4096-pixel constellation contains, and that information loss is
what the CNN exploits. This is the central argument of the paper.

The 15 features themselves are computed in data_generation.iq_to_features().
"""

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


# ===========================================================================
# WHY EVERY MODEL IS WRAPPED IN A "Pipeline" WITH A StandardScaler
# ===========================================================================
# Our 15 features live on wildly different scales: some cumulant ratios are
# around 0.5, while PAPR can be 10 or more.
#
#   * k-NN measures straight-line distance between feature vectors. A feature
#     with a big numeric range would dominate that distance purely because of
#     its units, drowning out the informative ones.
#   * The SVM's RBF kernel also works on distances, so it has the same issue.
#
# StandardScaler fixes this: for each feature it subtracts the mean and
# divides by the standard deviation, so every feature ends up on a comparable
# scale ("zero mean, unit variance").
#
# Using a Pipeline (rather than scaling by hand) matters for correctness: the
# scaler learns its mean and standard deviation from the TRAINING data only,
# and then applies those same numbers to the test data. Computing them over
# the whole dataset would leak information about the test set into training,
# which would inflate the reported accuracy. Reviewers check for this.


def make_svm(C=10.0, gamma="scale", seed=config.SEED):
    """
    Support Vector Machine with an RBF (Gaussian) kernel.

    WHAT AN SVM DOES, IN ONE PARAGRAPH
        It draws the boundary between classes so that the empty margin either
        side of the boundary is as wide as possible. A wide margin tends to
        generalise well. Only the few training points sitting closest to the
        boundary ("support vectors") actually determine where it goes.

    WHY THE RBF KERNEL?
        A plain SVM can only draw a straight line (a flat plane). Our classes
        are not separable by a straight line in 15-D -- especially at low
        OSNR where the feature clouds curve into each other. The RBF kernel
        implicitly maps the data into a much higher-dimensional space where a
        flat boundary becomes a curved boundary back in the original space.

    THE TWO HYPERPARAMETERS
        C     = how much to penalise misclassified training points.
                Low C  -> smooth, simple boundary, may underfit.
                High C -> boundary bends to catch every training point, may
                          overfit.
                C = 10 is a mild step up from the default of 1, which suits a
                large, noisy dataset like ours where the classes genuinely
                overlap at low OSNR.
        gamma = how far the influence of one training point reaches.
                'scale' sets it automatically to 1/(n_features * variance),
                which is scikit-learn's recommended default and adapts to our
                standardised data.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=C, gamma=gamma, kernel="rbf",
                    cache_size=1000,          # MB of kernel cache -> faster
                    random_state=seed)),
    ])


def make_knn(n_neighbors=5):
    """
    k-Nearest Neighbours.

    WHAT IT DOES
        There is no training at all. To classify a new signal it finds the k
        training examples whose feature vectors are closest, and takes a
        majority vote. It is the simplest sensible classifier that exists,
        which is exactly why it makes a good floor for comparison.

    WHY k = 5?
        k = 1 is very sensitive to a single noisy neighbour.
        Large k over-smooths and blurs the boundary between classes.
        k = 5 is the standard middle-ground default, and being odd it avoids
        ties in a 2-way vote.

    n_jobs=-1 uses all CPU cores, which matters because k-NN does all of its
    work at prediction time.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)),
    ])


# ===========================================================================
# TRAINING HELPER
# ===========================================================================
def fit_baseline(model, X_train, y_train, name="model",
                 max_train=None, seed=config.SEED, verbose=True):
    """
    Fit one baseline model and report how long it took.

    WHY `max_train`?
        An RBF SVM's training cost grows roughly with the SQUARE of the
        number of training samples, because it has to consider pairs of
        points. With 44,000 training samples that becomes slow (many
        minutes). Capping the SVM's training set at a random subsample keeps
        the experiment practical on a laptop.

        This is a fair thing to do, and it must be stated in the paper, but
        note it works AGAINST the CNN's favour only mildly: SVM accuracy on
        15 low-dimensional features saturates quickly with sample count, so
        the subsample costs it very little accuracy. k-NN and the CNN use the
        full training set.
    """
    if max_train is not None and len(y_train) > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_train), size=max_train, replace=False)
        idx.sort()
        X_train, y_train = X_train[idx], y_train[idx]
        if verbose:
            print("    (subsampled to {:,} training points)".format(max_train))

    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    if verbose:
        print("    {} trained on {:,} samples in {:.1f} s".format(
            name, len(y_train), dt))
    return model, dt


def evaluate_per_osnr(model, X_test, y_test, osnr_test):
    """
    Overall accuracy, plus accuracy computed separately at each OSNR level.

    The per-OSNR breakdown is the important one for the paper: a single
    overall number hides the fact that a classifier can be perfect at 25 dB
    and no better than guessing at 5 dB.
    """
    y_pred = model.predict(X_test)
    overall = float(np.mean(y_pred == y_test))

    per_osnr = {}
    for o in config.OSNR_LIST:
        m = (osnr_test == o)
        per_osnr[int(o)] = float(np.mean(y_pred[m] == y_test[m])) if m.any() else float("nan")

    return overall, per_osnr, y_pred


if __name__ == "__main__":
    # A quick standalone smoke test on a small slice of the data, so you can
    # run this file on its own to check it works.
    print("Loading features...")
    X = np.load(config.F_FEATURES)
    y = np.load(config.F_LABELS).astype(np.int64)
    o = np.load(config.F_OSNR).astype(np.int64)
    print("  {} samples, {} features".format(*X.shape))

    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(y))
    tr, te = idx[:8000], idx[8000:11000]

    for name, mk in (("k-NN", make_knn), ("SVM", make_svm)):
        print("\n{}:".format(name))
        model, _ = fit_baseline(mk(), X[tr], y[tr], name=name)
        acc, per, _ = evaluate_per_osnr(model, X[te], y[te], o[te])
        print("    overall acc = {:.3f}".format(acc))
        print("    acc @5dB = {:.3f}   @15dB = {:.3f}   @25dB = {:.3f}".format(
            per[5], per[15], per[25]))
