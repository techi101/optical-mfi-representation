"""
experiments.py
==============
STEP 5: the experiments that turn a working implementation into a paper.

The main comparison lives in train_and_evaluate.py. This file contains the
four studies that answer the questions a reviewer will actually ask.

    generalise   Can the model handle OSNR values it never saw in training?
                 (Trained on even OSNRs, tested on odd ones. If accuracy
                 holds, the network learned the physics rather than
                 memorising what each OSNR level looks like.)

    ablation     Do the architecture choices actually matter? Sweeps the
                 number of conv blocks and the number of channels. This
                 converts "I chose 3 layers because it seemed right" into
                 "I chose 3 layers because 2 was worse and 4 gained nothing".

    symbols      How few received symbols are enough? A monitoring receiver
                 wants an answer from a SHORT capture, so accuracy-vs-capture
                 -length is the practically important curve, and it is
                 rarely reported.

    robustness   Does a model trained on a clean channel survive a dirty one?
                 Trains on the 'ideal' (AWGN-only) dataset and tests on
                 'realistic' and 'severe'. This is the experiment that tells
                 you whether simulation-trained MFI would work on a real link.

Run me:
    python experiments.py generalise
    python experiments.py ablation
    python experiments.py symbols
    python experiments.py robustness
    python experiments.py all
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

import baselines
import config
import data_generation as dg
import train_and_evaluate as te
from cnn_model import (MODEL_INPUT_SHAPE, build_model, count_flops,
                       count_parameters)


def _save(name, obj):
    path = os.path.join(config.RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("  saved -> " + path)
    return path


# ---------------------------------------------------------------------------
# CHECKPOINTING
# ---------------------------------------------------------------------------
# These experiments loop over several settings, each taking many minutes. If
# the process is killed part-way -- which happens on a laptop when memory runs
# short -- writing results only at the end throws away everything.
#
# So we write the file after EVERY completed setting, and on startup we read
# back whatever is already there and skip those settings. A killed run then
# costs at most one setting instead of the whole experiment.
# ---------------------------------------------------------------------------
def _save_incremental(name, obj):
    """Write results so far, atomically (write to .tmp, then replace)."""
    path = os.path.join(config.RESULTS_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)          # atomic: never leaves a half-written file
    print("    [checkpoint saved: {} result(s)]".format(
        len(obj.get("results", []))))
    return path


def _resume_rows(name, key=None):
    """Return the results already stored in `name`, or [] if there are none."""
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get("results", [])
    except (ValueError, OSError):
        return []


# ===========================================================================
# EXPERIMENT 1 -- GENERALISATION TO UNSEEN OSNR
# ===========================================================================
def experiment_generalise(seeds=(42, 43, 44), epochs=None):
    """
    Train only on EVEN OSNR values, test only on ODD ones.

    WHY THIS IS WORTH DOING
    Our main experiment trains and tests on the same 21 OSNR levels. A
    sceptical reviewer can argue the network simply memorised what each level
    looks like, which would be useless in a real network where OSNR drifts
    continuously. Holding out entire OSNR levels removes that possibility: at
    test time the model sees noise levels it has never encountered.

    If accuracy stays close to the main result, the model has learned the
    underlying constellation geometry and interpolates across noise levels --
    a much stronger claim, and cheap to demonstrate.
    """
    print("=" * 72)
    print("  EXPERIMENT 1 -- GENERALISATION TO UNSEEN OSNR LEVELS")
    print("=" * 72)

    data = te.load_data()
    y, osnr = data["y"], data["osnr"]

    train_osnr = [o for o in config.OSNR_LIST if o % 2 == 0]
    test_osnr = [o for o in config.OSNR_LIST if o % 2 == 1]
    print("  train OSNR : {}".format(train_osnr))
    print("  test  OSNR : {}  (never seen during training)".format(test_osnr))

    is_train_osnr = np.isin(osnr, train_osnr)
    pool_train = np.where(is_train_osnr)[0]
    pool_test = np.where(~is_train_osnr)[0]

    out = {"train_osnr": train_osnr, "test_osnr": test_osnr, "per_seed": {}}

    for seed in seeds:
        print("\n  --- seed {} ---".format(seed))
        rng = np.random.default_rng(seed)
        # carve a validation set out of the TRAINING OSNRs only
        perm = rng.permutation(len(pool_train))
        n_val = int(0.15 * len(pool_train))
        idx_val = np.sort(pool_train[perm[:n_val]])
        idx_train = np.sort(pool_train[perm[n_val:]])
        idx_test = pool_test

        seed_res = {}
        for kind in ("cnn", "mlp"):
            X, _ = te.prepare_input(kind, data, idx_train)
            model, _ = te.train_torch_model(kind, X, y, idx_train, idx_val,
                                            epochs=epochs, seed=seed,
                                            verbose=False)
            y_pred = te.predict_torch(kind, model, X, idx_test)
            r = te.evaluate(y[idx_test], y_pred, osnr[idx_test],
                            osnr_list=test_osnr)
            seed_res[kind] = r
            print("    {:<16s} unseen-OSNR accuracy = {:.4f}".format(
                te.MODEL_LABEL[kind], r["overall_accuracy"]))

        # classical baseline for reference
        clf, _ = baselines.fit_baseline(baselines.make_svm(),
                                        data["feat"][idx_train], y[idx_train],
                                        name="svm", max_train=45000,
                                        verbose=False)
        y_pred = clf.predict(data["feat"][idx_test])
        seed_res["svm"] = te.evaluate(y[idx_test], y_pred, osnr[idx_test],
                                      osnr_list=test_osnr)
        print("    {:<16s} unseen-OSNR accuracy = {:.4f}".format(
            "SVM (features)", seed_res["svm"]["overall_accuracy"]))

        out["per_seed"][str(seed)] = seed_res

    # summarise
    out["summary"] = {}
    for kind in ("cnn", "mlp", "svm"):
        vals = [out["per_seed"][str(s)][kind]["overall_accuracy"] for s in seeds]
        out["summary"][kind] = {"mean": float(np.mean(vals)),
                                "std": float(np.std(vals))}
        print("\n  {:<16s} {:.2f} +/- {:.2f} %".format(
            te.MODEL_LABEL[kind], 100 * np.mean(vals), 100 * np.std(vals)))

    return _save("exp_generalise.json", out)


# ===========================================================================
# EXPERIMENT 2 -- ARCHITECTURE ABLATION
# ===========================================================================
def experiment_ablation(seed=42, epochs=12, train_subsample=20000):
    """
    Sweep the CNN's depth and width, and report accuracy against cost.

    WHY A PAPER NEEDS THIS
    Saying "I used 3 convolution blocks with 16 base channels" invites the
    question "why?". An ablation answers it with evidence: it shows what
    happens with 2 blocks and with 4, and with narrower and wider layers. It
    also produces the accuracy-vs-parameters trade-off curve, which is the
    figure a systems reviewer actually cares about, because an MFI block has
    to fit inside a receiver's limited compute budget.

    WHY WE TRAIN ON A SUBSAMPLE HERE
    Six variants on the full 44,100-image training set would take about 100
    minutes. But an ablation only needs RELATIVE differences -- "is 4 blocks
    better than 3?" -- not the last fraction of a percent of absolute
    accuracy. Training every variant on the same 20,000-image subsample keeps
    the comparison fair (identical data for every variant) and cuts the cost
    by more than half. The absolute numbers here will sit slightly below the
    main result, and the paper should say so rather than presenting them as
    comparable to Table I.
    """
    print("=" * 72)
    print("  EXPERIMENT 2 -- ARCHITECTURE ABLATION")
    print("=" * 72)

    data = te.load_data(need_iq=False)
    y, osnr = data["y"], data["osnr"]
    idx_train, idx_val, idx_test = te.make_splits(y, osnr, seed=seed)

    if train_subsample and len(idx_train) > train_subsample:
        rng = np.random.default_rng(seed)
        idx_train = np.sort(rng.choice(idx_train, train_subsample,
                                       replace=False))
        print("  training each variant on a fixed {:,}-image subsample".format(
            train_subsample))

    X, _ = te.prepare_input("cnn", data, idx_train)

    variants = []
    for n_blocks in (2, 3, 4):
        variants.append(dict(n_blocks=n_blocks, base_ch=16))
    for base_ch in (4, 8, 32):
        variants.append(dict(n_blocks=3, base_ch=base_ch))

    # Pick up anything a previous (possibly interrupted) run already finished,
    # so a killed run does not throw away hours of work. See the note in
    # _save_incremental below.
    rows = _resume_rows("exp_ablation.json",
                        key=lambda r: (r["n_blocks"], r["base_ch"]))
    done = {(r["n_blocks"], r["base_ch"]) for r in rows}
    if done:
        print("  resuming: {} variant(s) already complete".format(len(done)))

    for v in variants:
        if (v["n_blocks"], v["base_ch"]) in done:
            continue
        t0 = time.time()
        model, hist = te.train_torch_model("cnn", X, y, idx_train, idx_val,
                                           epochs=epochs, seed=seed,
                                           verbose=False, model_kwargs=v)
        y_pred = te.predict_torch("cnn", model, X, idx_test)
        r = te.evaluate(y[idx_test], y_pred, osnr[idx_test])
        n_par = count_parameters(model)
        macs = count_flops(build_model("cnn", **v), MODEL_INPUT_SHAPE["cnn"])
        row = dict(v, accuracy=r["overall_accuracy"], n_parameters=int(n_par),
                   macs=int(macs), epochs_run=len(hist["epoch"]),
                   train_time_sec=time.time() - t0,
                   accuracy_per_osnr=r["accuracy_per_osnr"])
        rows.append(row)
        print("  blocks={n_blocks}  base_ch={base_ch:<3d}  "
              "acc={accuracy:.4f}  params={n_parameters:>8,}  "
              "MACs={macs:>11,}".format(**row))

        # Write after EVERY variant, not at the end.
        _save_incremental("exp_ablation.json",
                          {"seed": seed, "epochs": epochs,
                           "train_subsample": train_subsample,
                           "n_train": int(len(idx_train)), "results": rows})
        del model
        gc.collect()

    return os.path.join(config.RESULTS_DIR, "exp_ablation.json")


# ===========================================================================
# EXPERIMENT 3 -- HOW MANY SYMBOLS DO YOU NEED?
# ===========================================================================
def experiment_symbols(symbol_counts=(128, 256, 512, 1024, 2048, 4096),
                       n_per_class=150, seed=42, epochs=12):
    """
    Accuracy as a function of how many received symbols go into the image.

    WHY THIS IS THE MOST PRACTICALLY USEFUL EXPERIMENT HERE
    A performance monitor does not get to stare at the signal forever. It gets
    a short capture, and shorter captures mean faster monitoring and cheaper
    memory. So "how many symbols do I actually need for reliable
    identification?" is the question a system designer asks first, and it is
    rarely answered in the MFI literature.

    We build a small dedicated dataset for each symbol count. It is smaller
    than the main one because we only need to compare settings against each
    other, not to squeeze out the last 0.1% of accuracy.
    """
    print("=" * 72)
    print("  EXPERIMENT 3 -- ACCURACY vs NUMBER OF SYMBOLS")
    print("=" * 72)

    rows = _resume_rows("exp_symbols.json")
    done = {r["n_symbols"] for r in rows}
    if done:
        print("  resuming: already have {}".format(sorted(done)))

    for n_sym in symbol_counts:
        if n_sym in done:
            continue
        t0 = time.time()
        print("\n  --- {} symbols per image ---".format(n_sym))
        X_img, _, X_feat, y, osnr = dg.build_dataset(
            n_per_class_per_osnr=n_per_class, profile=config.DEFAULT_PROFILE,
            n_symbols=n_sym, store_iq=False, verbose=False)
        data = {"img": X_img, "feat": X_feat, "y": y.astype(np.int64),
                "osnr": osnr.astype(np.int64)}
        y64 = data["y"]
        idx_train, idx_val, idx_test = te.make_splits(y64, data["osnr"], seed=seed)

        X, _ = te.prepare_input("cnn", data, idx_train)
        model, _ = te.train_torch_model("cnn", X, y64, idx_train, idx_val,
                                        epochs=epochs, seed=seed, verbose=False)
        y_pred = te.predict_torch("cnn", model, X, idx_test)
        r_cnn = te.evaluate(y64[idx_test], y_pred, data["osnr"][idx_test])

        clf, _ = baselines.fit_baseline(baselines.make_svm(),
                                        X_feat[idx_train], y64[idx_train],
                                        name="svm", verbose=False)
        r_svm = te.evaluate(y64[idx_test], clf.predict(X_feat[idx_test]),
                            data["osnr"][idx_test])

        row = {"n_symbols": n_sym,
               "cnn_accuracy": r_cnn["overall_accuracy"],
               "svm_accuracy": r_svm["overall_accuracy"],
               "cnn_per_osnr": r_cnn["accuracy_per_osnr"],
               "svm_per_osnr": r_svm["accuracy_per_osnr"],
               "seconds": time.time() - t0}
        rows.append(row)
        print("    CNN {:.4f}   SVM {:.4f}   ({:.0f}s)".format(
            row["cnn_accuracy"], row["svm_accuracy"], row["seconds"]))

        _save_incremental("exp_symbols.json",
                          {"seed": seed, "n_per_class": n_per_class,
                           "results": rows})
        # These arrays are large; free them before building the next dataset.
        del X_img, X_feat, data, X, model
        gc.collect()

    return os.path.join(config.RESULTS_DIR, "exp_symbols.json")


# ===========================================================================
# EXPERIMENT 4 -- ROBUSTNESS TO UNSEEN IMPAIRMENTS
# ===========================================================================
def experiment_robustness(seed=42, epochs=None, n_per_class=250):
    """
    Train on one impairment profile, test on the others.

    WHY THIS IS THE MOST IMPORTANT EXPERIMENT IN THE PAPER
    Every simulation-based MFI paper faces the same objection: "your model was
    trained and tested on the same idealised channel, so of course it works."
    The honest question is whether a model trained on a clean simulated link
    still works when the link is dirtier than expected -- because that is what
    happens when you deploy it.

    We train on 'ideal' (AWGN only) and test on 'realistic' and 'severe'
    (which add laser phase noise, frequency offset, I/Q imbalance and residual
    dispersion). A large drop means simulation-trained MFI does not transfer,
    which is a genuinely useful negative result. A small drop is a strong
    positive claim.

    We also train on 'realistic' and test on 'severe', which is the more
    practical question: how much margin does training on a reasonable channel
    model buy you?
    """
    print("=" * 72)
    print("  EXPERIMENT 4 -- ROBUSTNESS TO UNSEEN IMPAIRMENTS")
    print("=" * 72)

    profiles = ["ideal", "realistic", "severe"]
    datasets = {}
    for prof in profiles:
        tag = prof
        paths = config.dataset_paths(tag)
        if not os.path.exists(paths["images"]):
            print("\n  generating '{}' dataset ({} per class per OSNR)...".format(
                prof, n_per_class))
            X_img, X_iq, X_feat, y, osnr = dg.build_dataset(
                n_per_class_per_osnr=n_per_class, profile=prof,
                store_iq=False, verbose=False)
            dg.save_dataset(X_img, None, X_feat, y, osnr, n_per_class,
                            prof, tag=tag)
        datasets[prof] = te.load_data(tag, need_iq=False)
        print("  loaded '{}': {:,} samples".format(prof, len(datasets[prof]["y"])))

    out = {"profiles": profiles, "matrix": {}}

    for train_prof in profiles:
        d = datasets[train_prof]
        y, osnr = d["y"], d["osnr"]
        idx_train, idx_val, idx_test = te.make_splits(y, osnr, seed=seed)

        # --- train the CNN once on this profile ---
        X, _ = te.prepare_input("cnn", d, idx_train)
        cnn, _ = te.train_torch_model("cnn", X, y, idx_train, idx_val,
                                      epochs=epochs, seed=seed, verbose=False)
        # --- and the SVM, for comparison ---
        svm, _ = baselines.fit_baseline(baselines.make_svm(),
                                        d["feat"][idx_train], y[idx_train],
                                        name="svm", max_train=45000,
                                        verbose=False)
        # feature standardisation is inside the sklearn Pipeline already

        out["matrix"][train_prof] = {}
        for test_prof in profiles:
            d2 = datasets[test_prof]
            # Use the SAME held-out indices when the profile matches, and the
            # whole of the other datasets otherwise (they are independent).
            idx_eval = idx_test if test_prof == train_prof else np.arange(len(d2["y"]))

            X2 = torch.from_numpy(d2["img"]).unsqueeze(1)
            y_pred = te.predict_torch("cnn", cnn, X2, idx_eval)
            r_cnn = te.evaluate(d2["y"][idx_eval], y_pred, d2["osnr"][idx_eval])

            r_svm = te.evaluate(d2["y"][idx_eval],
                                svm.predict(d2["feat"][idx_eval]),
                                d2["osnr"][idx_eval])

            out["matrix"][train_prof][test_prof] = {
                "cnn": r_cnn["overall_accuracy"],
                "svm": r_svm["overall_accuracy"],
                "cnn_per_osnr": r_cnn["accuracy_per_osnr"],
                "svm_per_osnr": r_svm["accuracy_per_osnr"],
            }
            print("  train={:<10s} test={:<10s}  CNN {:.4f}   SVM {:.4f}".format(
                train_prof, test_prof, r_cnn["overall_accuracy"],
                r_svm["overall_accuracy"]))

    return _save("exp_robustness.json", out)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["generalise", "ablation", "symbols",
                                     "robustness", "all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    t0 = time.time()
    if args.which in ("generalise", "all"):
        experiment_generalise(seeds=args.seeds, epochs=args.epochs)
    if args.which in ("ablation", "all"):
        experiment_ablation(seed=args.seeds[0], epochs=args.epochs or 12)
    if args.which in ("symbols", "all"):
        experiment_symbols(seed=args.seeds[0], epochs=args.epochs or 12)
    if args.which in ("robustness", "all"):
        experiment_robustness(seed=args.seeds[0], epochs=args.epochs)
    print("\n  total time: {:.1f} min".format((time.time() - t0) / 60))
