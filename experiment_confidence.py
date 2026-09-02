"""
experiment_confidence.py
========================
EXPERIMENT 6: can the model tell when it is failing?

THE QUESTION, AND WHY IT IS WORTH ASKING
Experiment 4 (robustness) showed that under channel mismatch a feature-based
classifier collapses to roughly the chance level. But an accuracy number is
something only the experimenter can see. A deployed monitor does not know its
own accuracy -- it only knows what it predicted and how confident it was.

So the operationally important question is not "how much accuracy is lost?"
but:

    when the model is wrong, does it KNOW it is wrong?

A monitor that fails loudly (low confidence) is recoverable: the control plane
can ignore it, fall back, or raise an alarm. A monitor that fails SILENTLY
(high confidence, wrong answer) is worse than no monitor at all, because it
feeds confident nonsense into decisions.

WHAT IS NEW HERE
Uncertainty quantification for modulation classification has been studied in
wireless (Yang & Sahay, 2025). Representation comparison has been studied for
accuracy (Ge et al., 2021). What has not been asked, in either domain, is
whether the INPUT REPRESENTATION determines whether failure is self-detectable.

There is a concrete reason to expect that it might. Cumulants are summary
statistics: an impairment shifts their values, but the shifted values are
still perfectly ordinary-looking numbers, sitting in a region of feature space
the classifier treats as normal. It has no way to notice anything is wrong.
A distorted constellation IMAGE, by contrast, may look visibly unlike anything
seen in training, which a network can in principle register as unfamiliar.

If that holds, then the representation does not merely determine accuracy --
it determines whether failure is DETECTABLE, which is a stronger and more
practically important claim.

WHAT WE MEASURE
For each model, on each channel:
  accuracy          - how often it is right
  mean confidence   - how confident it is on average
  overconfidence    - mean confidence minus accuracy. Large and positive is
                      the danger signal: sure of itself, and wrong.
  AUROC             - can the confidence score separate its own correct
                      predictions from its incorrect ones? 0.5 = no better
                      than a coin toss, so the confidence is useless as a
                      warning signal. 1.0 = perfect self-knowledge.
  ECE               - expected calibration error: across confidence bins, how
                      far the stated confidence sits from the observed
                      accuracy.

Run me:
    python experiment_confidence.py
"""

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import baselines
import config
import train_and_evaluate as te
from cnn_model import ConstellationCNN

PROFILES = ["ideal", "realistic", "severe"]
TRAIN_PROFILE = "realistic"          # what the models are trained on


# ===========================================================================
# CONFIDENCE METRICS
# ===========================================================================
def expected_calibration_error(conf, correct, n_bins=10):
    """
    Expected Calibration Error.

    Split predictions into bins by stated confidence. In each bin, compare the
    average confidence with the actual fraction correct. ECE is the weighted
    average of |confidence - accuracy| over bins.

    A perfectly calibrated model that says "90% sure" is right 90% of the time,
    giving ECE = 0. A model that says "99% sure" while being right 30% of the
    time has a large ECE, and is exactly the failure mode we are looking for.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def confidence_report(y_true, y_pred, conf):
    """Turn predictions plus confidences into the five numbers we report."""
    correct = (y_pred == y_true).astype(int)
    acc = float(correct.mean())
    mean_conf = float(np.mean(conf))

    # AUROC of confidence as a detector of the model's OWN errors.
    # Undefined if every prediction is right or every one is wrong.
    if 0 < correct.sum() < len(correct):
        auroc = float(roc_auc_score(correct, conf))
    else:
        auroc = float("nan")

    return {
        "accuracy": acc,
        "mean_confidence": mean_conf,
        "overconfidence": mean_conf - acc,
        "error_auroc": auroc,
        "ece": expected_calibration_error(conf, correct),
        "n": int(len(y_true)),
    }


# ===========================================================================
# GETTING A CONFIDENCE OUT OF EACH MODEL
# ===========================================================================
@torch.no_grad()
def torch_confidence(kind, model, X, idx, batch_size=512):
    """
    Max softmax probability -- the standard confidence score for a classifier.

    Softmax turns the network's raw scores into numbers that sum to 1, and we
    take the largest. It is the obvious thing a deployed system would threshold
    on, which is why it is the right thing to test.
    """
    model.eval()
    preds = np.zeros(len(idx), dtype=np.int64)
    confs = np.zeros(len(idx), dtype=np.float64)
    for i in range(0, len(idx), batch_size):
        rows = torch.from_numpy(idx[i:i + batch_size])
        logits = model(te.batch_to_float(kind, X[rows]))
        p = torch.softmax(logits, dim=1)
        c, k = p.max(dim=1)
        preds[i:i + batch_size] = k.numpy()
        confs[i:i + batch_size] = c.numpy()
    return preds, confs


def sklearn_confidence(clf, Xf):
    """Max class probability from a scikit-learn model."""
    p = clf.predict_proba(Xf)
    return p.argmax(axis=1), p.max(axis=1)


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================
def main(seed=config.SEED, n_per_class=250, epochs=None,
         train_profile=TRAIN_PROFILE, out_name=None):
    """
    `train_profile` selects the channel the models are trained on.

    Running this for BOTH 'realistic' and 'ideal' matters. Training on
    'realistic' and testing on 'severe' is a moderate shift. Training on
    'ideal' (AWGN only) and testing on 'severe' is the large shift that caused
    the feature-based classifier to collapse to chance in Experiment 4 -- and
    a large accuracy collapse is exactly where silent failure, if it exists,
    would show up.
    """
    globals()["TRAIN_PROFILE"] = train_profile
    out_name = out_name or "exp_confidence_{}.json".format(train_profile)

    print("=" * 74)
    print("  EXPERIMENT 6 -- CAN THE MODEL TELL WHEN IT IS FAILING?")
    print("=" * 74)
    print("  models are trained on the '{}' channel and then tested on all "
          "three".format(train_profile))

    # ---------------- make sure all three datasets exist ----------------
    import data_generation as dg
    data = {}
    for prof in PROFILES:
        paths = config.dataset_paths(prof)
        if not os.path.exists(paths["images"]):
            print("\n  generating '{}' dataset...".format(prof))
            Xi, _, Xf, y, o = dg.build_dataset(n_per_class_per_osnr=n_per_class,
                                               profile=prof, store_iq=False,
                                               verbose=False)
            dg.save_dataset(Xi, None, Xf, y, o, n_per_class, prof, tag=prof)
        data[prof] = te.load_data(prof, need_iq=False)
        print("  loaded '{}': {:,} samples".format(prof, len(data[prof]["y"])))

    d_tr = data[train_profile]
    y, osnr = d_tr["y"], d_tr["osnr"]
    idx_train, idx_val, idx_test = te.make_splits(y, osnr, seed=seed)

    results = {"train_profile": train_profile, "seed": seed, "models": {}}

    # ==================================================================
    # 1. CNN on constellation images
    # ==================================================================
    print("\n  [1/4] CNN (image) -- training on '{}'".format(TRAIN_PROFILE))
    Xt = torch.from_numpy(d_tr["img"]).unsqueeze(1)
    cnn, _ = te.train_torch_model("cnn", Xt, y, idx_train, idx_val,
                                  epochs=epochs, seed=seed, verbose=False)
    results["models"]["cnn"] = {}
    for prof in PROFILES:
        d2 = data[prof]
        idx = idx_test if prof == train_profile else np.arange(len(d2["y"]))
        X2 = torch.from_numpy(d2["img"]).unsqueeze(1)
        pred, conf = torch_confidence("cnn", cnn, X2, idx)
        r = confidence_report(d2["y"][idx], pred, conf)
        results["models"]["cnn"][prof] = r
        print("      {:<10s} acc {:.4f}  conf {:.4f}  overconf {:+.4f}  "
              "AUROC {:.3f}".format(prof, r["accuracy"], r["mean_confidence"],
                                    r["overconfidence"], r["error_auroc"]))

    # ==================================================================
    # 2. MLP on the 15 features -- the controlled comparison.
    #    Same output type (softmax), same training procedure, same data.
    #    ONLY the representation differs.
    # ==================================================================
    print("\n  [2/4] MLP (features) -- the controlled comparison")
    Xf_all, _ = te.prepare_input("mlp", d_tr, idx_train)
    mu = d_tr["feat"][idx_train].mean(axis=0, keepdims=True)
    sd = d_tr["feat"][idx_train].std(axis=0, keepdims=True) + 1e-8
    mlp, _ = te.train_torch_model("mlp", Xf_all, y, idx_train, idx_val,
                                  epochs=epochs, seed=seed, verbose=False)
    results["models"]["mlp"] = {}
    for prof in PROFILES:
        d2 = data[prof]
        idx = idx_test if prof == train_profile else np.arange(len(d2["y"]))
        # standardise the OTHER profiles with the TRAINING profile's statistics
        # -- that is what a deployed system would have to do.
        Xn = torch.from_numpy(((d2["feat"] - mu) / sd).astype(np.float32))
        pred, conf = torch_confidence("mlp", mlp, Xn, idx)
        r = confidence_report(d2["y"][idx], pred, conf)
        results["models"]["mlp"][prof] = r
        print("      {:<10s} acc {:.4f}  conf {:.4f}  overconf {:+.4f}  "
              "AUROC {:.3f}".format(prof, r["accuracy"], r["mean_confidence"],
                                    r["overconfidence"], r["error_auroc"]))

    # ==================================================================
    # 3. SVM on features -- the classical baseline.
    #    probability=True fits Platt scaling internally, which is slow, so we
    #    cap the training set. That is a fair cap: it is the same classifier
    #    family, and its accuracy saturates quickly on 15 features.
    # ==================================================================
    print("\n  [3/4] SVM (features)")
    t0 = time.time()
    svm = baselines.make_svm()
    svm.set_params(svm__probability=True)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(idx_train, min(15000, len(idx_train)),
                             replace=False))
    svm.fit(d_tr["feat"][sub], y[sub])
    print("      fitted with Platt scaling in {:.0f}s".format(time.time() - t0))
    results["models"]["svm"] = {}
    for prof in PROFILES:
        d2 = data[prof]
        idx = idx_test if prof == train_profile else np.arange(len(d2["y"]))
        pred, conf = sklearn_confidence(svm, d2["feat"][idx])
        r = confidence_report(d2["y"][idx], pred, conf)
        results["models"]["svm"][prof] = r
        print("      {:<10s} acc {:.4f}  conf {:.4f}  overconf {:+.4f}  "
              "AUROC {:.3f}".format(prof, r["accuracy"], r["mean_confidence"],
                                    r["overconfidence"], r["error_auroc"]))

    # ==================================================================
    # 4. Can a confidence THRESHOLD save you?
    #    The practical question: if the monitor refuses to answer whenever it
    #    is less than T confident, does what remains become trustworthy?
    # ==================================================================
    print("\n  [4/4] Confidence thresholding on the unseen 'severe' channel")
    print("      (if the monitor abstains below a threshold, is the rest safe?)")
    results["thresholding"] = {}
    for name, getter in (("cnn", None), ("mlp", None), ("svm", None)):
        d2 = data["severe"]
        idx = np.arange(len(d2["y"]))
        if name == "cnn":
            X2 = torch.from_numpy(d2["img"]).unsqueeze(1)
            pred, conf = torch_confidence("cnn", cnn, X2, idx)
        elif name == "mlp":
            Xn = torch.from_numpy(((d2["feat"] - mu) / sd).astype(np.float32))
            pred, conf = torch_confidence("mlp", mlp, Xn, idx)
        else:
            pred, conf = sklearn_confidence(svm, d2["feat"][idx])

        correct = (pred == d2["y"][idx])
        rows = []
        for thr in (0.0, 0.5, 0.7, 0.9, 0.95, 0.99):
            keep = conf >= thr
            rows.append({
                "threshold": thr,
                "coverage": float(keep.mean()),
                "accuracy_on_kept": float(correct[keep].mean()) if keep.any()
                else float("nan"),
            })
        results["thresholding"][name] = rows
        print("      {}:".format(name))
        for r in rows:
            print("        thr {:.2f}  keeps {:5.1f}% of samples, "
                  "accuracy on those {:.4f}".format(
                      r["threshold"], 100 * r["coverage"],
                      r["accuracy_on_kept"]))

    # ---------------- save ----------------
    path = os.path.join(config.RESULTS_DIR, out_name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n  saved -> " + path)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-profile", default="realistic",
                    choices=["ideal", "realistic", "severe"])
    a = ap.parse_args()
    main(train_profile=a.train_profile)
