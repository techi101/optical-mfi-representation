"""
experiment_tolerance.py
=======================
EXPERIMENT 7: how wrong is your channel model allowed to be?

THE QUESTION
Experiments 4 and 6 showed that a model trained on one channel degrades on
another, and that the feature-based classifier degrades in a way that cannot
be recovered by confidence gating. Both used discrete channel profiles
('ideal', 'realistic', 'severe'), which answers "does mismatch hurt?" but not
"how much mismatch can I afford?"

That second question is the one a system designer actually has. Before
deploying a simulation-trained monitor, they need to know how accurately the
channel must be characterised. Nobody has published that number for optical
MFI.

WHAT WE DO
Take a model trained once on the 'realistic' channel. Then vary ONE impairment
at a time -- holding the other four at their training values -- and measure
accuracy as that single impairment moves away from what the model was trained
on. Repeat for each impairment.

The output is a tolerance budget: for each impairment, the largest deviation
from the assumed value at which the monitor still meets an accuracy target.
It also ranks the impairments, which tells a designer which parameter most
needs to be measured accurately.

WHY THIS IS CHEAP
No retraining. The models are fitted once; the cost is generating test sets
and evaluating, which is fast.

Run me:
    python experiment_tolerance.py
"""

import json
import os
import time

import numpy as np
import torch

import baselines
import config
import data_generation as dg
import train_and_evaluate as te

TRAIN_PROFILE = "realistic"

# For each impairment: the sweep values to test. The training value is
# included in each sweep so that the matched case appears on every curve as a
# reference point.
SWEEPS = {
    "laser_linewidth_hz":   [0.0, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6],
    "freq_offset_hz":       [0.0, 5e6, 10e6, 25e6, 50e6, 100e6, 200e6],
    "iq_gain_imbalance_db": [0.0, 0.15, 0.3, 0.6, 1.0, 1.5, 2.5],
    "iq_phase_skew_deg":    [0.0, 1.0, 2.0, 4.0, 6.0, 9.0, 14.0],
    "residual_cd_ps_nm":    [0.0, 5.0, 10.0, 25.0, 50.0, 90.0, 150.0],
}

PRETTY = {
    "laser_linewidth_hz":   ("laser linewidth", "kHz", 1e-3),
    "freq_offset_hz":       ("residual frequency offset", "MHz", 1e-6),
    "iq_gain_imbalance_db": ("I/Q gain imbalance", "dB", 1.0),
    "iq_phase_skew_deg":    ("I/Q phase skew", "deg", 1.0),
    "residual_cd_ps_nm":    ("residual dispersion", "ps/nm", 1.0),
}


def _save(obj, name="exp_tolerance.json"):
    path = os.path.join(config.RESULTS_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    return path


def _resume(name="exp_tolerance.json"):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f).get("sweeps", {})
    except (ValueError, OSError):
        return {}


def main(seed=config.SEED, n_per_class=60, epochs=None):
    print("=" * 74)
    print("  EXPERIMENT 7 -- IMPAIRMENT TOLERANCE BUDGET")
    print("=" * 74)
    base = dict(config.IMPAIRMENT_PROFILES[TRAIN_PROFILE])
    print("  models trained on the '{}' channel:".format(TRAIN_PROFILE))
    for k, v in base.items():
        print("     {:<24s} {}".format(k, v))
    print("\n  each impairment is then swept alone, others held at the above")

    # ---------------- train once ----------------
    d_tr = te.load_data(TRAIN_PROFILE, need_iq=False)
    y, osnr = d_tr["y"], d_tr["osnr"]
    idx_train, idx_val, idx_test = te.make_splits(y, osnr, seed=seed)

    print("\n  training the CNN once ({:,} images)...".format(len(idx_train)))
    Xt = torch.from_numpy(d_tr["img"]).unsqueeze(1)
    cnn, _ = te.train_torch_model("cnn", Xt, y, idx_train, idx_val,
                                  epochs=epochs, seed=seed, verbose=False)

    print("  fitting the SVM once...")
    svm, _ = baselines.fit_baseline(baselines.make_svm(),
                                    d_tr["feat"][idx_train], y[idx_train],
                                    name="svm", max_train=45000, verbose=False)

    results = {"train_profile": TRAIN_PROFILE, "base": base, "seed": seed,
               "n_per_class": n_per_class, "sweeps": _resume()}
    if results["sweeps"]:
        print("  resuming: already have {}".format(list(results["sweeps"])))

    # ---------------- sweep ----------------
    for param, values in SWEEPS.items():
        if param in results["sweeps"]:
            continue
        label, unit, scale = PRETTY[param]
        print("\n  --- sweeping {} ---".format(label))
        rows = []
        for v in values:
            t0 = time.time()
            prof = dict(base)
            prof[param] = v

            X_img, _, X_feat, yy, oo = dg.build_dataset(
                n_per_class_per_osnr=n_per_class, profile=prof,
                store_iq=False, verbose=False)
            yy = yy.astype(np.int64)
            idx = np.arange(len(yy))

            X2 = torch.from_numpy(X_img).unsqueeze(1)
            acc_cnn = float(np.mean(
                te.predict_torch("cnn", cnn, X2, idx) == yy))
            acc_svm = float(np.mean(svm.predict(X_feat) == yy))

            rows.append({"value": float(v),
                         "display_value": float(v) * scale,
                         "cnn_accuracy": acc_cnn,
                         "svm_accuracy": acc_svm})
            marker = "  <- trained here" if v == base[param] else ""
            print("      {:>10.3f} {:<6s} CNN {:.4f}   SVM {:.4f}   "
                  "({:.0f}s){}".format(v * scale, unit, acc_cnn, acc_svm,
                                       time.time() - t0, marker))
            del X_img, X_feat, X2

            results["sweeps"][param] = {"label": label, "unit": unit,
                                        "scale": scale,
                                        "trained_at": float(base[param]),
                                        "points": rows}
            _save(results)

    # ---------------- the budget table ----------------
    print("\n" + "=" * 74)
    print("  TOLERANCE BUDGET -- largest tested deviation still meeting the "
          "target")
    print("=" * 74)
    budget = {}
    for target in (0.95, 0.90):
        print("\n  target accuracy {:.0f}%".format(100 * target))
        print("  {:<28s} {:>16s} {:>16s}".format("impairment", "CNN limit",
                                                 "SVM limit"))
        budget[str(target)] = {}
        for param, s in results["sweeps"].items():
            pts = sorted(s["points"], key=lambda r: r["value"])
            lim = {}
            for m in ("cnn", "svm"):
                ok = [p for p in pts if p[m + "_accuracy"] >= target]
                lim[m] = (max(p["display_value"] for p in ok) if ok
                          else float("nan"))
            budget[str(target)][param] = lim
            fmt = lambda x: ("--" if np.isnan(x)
                             else "{:.1f} {}".format(x, s["unit"]))
            print("  {:<28s} {:>16s} {:>16s}".format(
                s["label"], fmt(lim["cnn"]), fmt(lim["svm"])))

    results["budget"] = budget
    path = _save(results)
    print("\n  saved -> " + path)
    return results


if __name__ == "__main__":
    main()
