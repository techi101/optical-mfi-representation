"""
train_and_evaluate.py
=====================
STEP 4: the main experiment, and the shared training machinery that every
other experiment reuses.

It compares FIVE classifiers on exactly the same signals:

    model      input                        pre-processing by a human
    ---------  ---------------------------  --------------------------
    CNN        64x64 constellation image    binning into a 2-D histogram
    IQ-CNN     512 raw I/Q samples          none
    MLP        15 statistical features      a lot
    SVM        15 statistical features      a lot
    k-NN       15 statistical features      a lot

Rows 1-3 are neural networks, rows 3-5 share an input. That layout is
deliberate: comparing CNN vs MLP isolates the effect of the REPRESENTATION
(both are trained neural networks), and comparing MLP vs SVM/k-NN isolates the
effect of the MODEL (all three see identical inputs). A plain "CNN beats SVM"
comparison confounds the two.

Everything is run for several random seeds so the paper can report
"96.8 +/- 0.2 %" instead of a single number a reviewer cannot trust.

Run me:
    python train_and_evaluate.py                     # 3 seeds, all models
    python train_and_evaluate.py --seeds 42          # quick single-seed run
    python train_and_evaluate.py --epochs 5          # quick trial
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import baselines
import config
from cnn_model import (MODEL_INPUT_SHAPE, build_model, count_flops,
                       count_parameters, set_all_seeds)


# ===========================================================================
# 1. LOADING AND SPLITTING
# ===========================================================================
def load_data(tag="", need_iq=True):
    """Load one dataset variant. `tag` selects e.g. the 'severe' version."""
    paths = config.dataset_paths(tag)
    if not os.path.exists(paths["images"]):
        raise SystemExit(
            "Dataset '{}' not found. Generate it first:\n"
            "    python data_generation.py --n-per-class 600{}".format(
                tag or "(main)",
                " --profile {} --tag {}".format(tag, tag) if tag else ""))

    data = {
        "img": np.load(paths["images"]),
        "feat": np.load(paths["features"]),
        "y": np.load(paths["labels"]).astype(np.int64),
        "osnr": np.load(paths["osnr"]).astype(np.int64),
    }
    if need_iq and os.path.exists(paths["iq"]):
        data["iq"] = np.load(paths["iq"])
    return data


def make_splits(y, osnr, seed=config.SEED):
    """
    Split into 70% train / 15% validation / 15% test.

    WHY THREE SETS AND NOT TWO?
        train      -- the model learns from these.
        validation -- checked after every epoch to decide WHEN TO STOP and
                      which epoch's weights to keep. Because we make
                      decisions with it, it is no longer a neutral judge.
        test       -- touched exactly once, at the very end. The only honest
                      estimate of real-world performance, and the number that
                      goes in the paper.

    WHY STRATIFY ON (class, OSNR)?
        A plain random split could, by bad luck, put more low-OSNR 64-QAM
        images in the test set than in training, making the test look
        artificially hard. Stratifying on the pair forces each of the
        5 x 21 = 105 groups to be split 70/15/15 individually. That matters
        especially because we report accuracy PER OSNR, so we need a decent
        number of test samples at every level.
    """
    idx = np.arange(len(y))
    strat = y.astype(np.int64) * 1000 + osnr.astype(np.int64)

    idx_train, idx_tmp = train_test_split(
        idx, test_size=(config.VAL_FRAC + config.TEST_FRAC),
        random_state=seed, shuffle=True, stratify=strat)
    rel_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=rel_test,
        random_state=seed, shuffle=True, stratify=strat[idx_tmp])

    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


# ===========================================================================
# 2. PREPARING EACH MODEL'S INPUT
# ===========================================================================
def prepare_input(kind, data, idx_train):
    """
    Turn the stored arrays into the tensor each network expects, and work out
    any normalisation from the TRAINING SET ONLY.

    Why "training set only" matters: if we computed the mean and standard
    deviation over the whole dataset, information about the test set would
    leak into training and inflate the reported accuracy. Reviewers check for
    this, and it is the single most common silent mistake in ML papers.
    """
    if kind == "cnn":
        # uint8 images; we divide by 255 per batch to keep memory down.
        return torch.from_numpy(data["img"]).unsqueeze(1), None

    if kind == "iqcnn":
        return torch.from_numpy(data["iq"]), None

    if kind == "mlp":
        X = data["feat"]
        mu = X[idx_train].mean(axis=0, keepdims=True)
        sd = X[idx_train].std(axis=0, keepdims=True) + 1e-8
        Xn = ((X - mu) / sd).astype(np.float32)
        return torch.from_numpy(Xn), (mu, sd)

    raise ValueError(kind)


def batch_to_float(kind, xb):
    """Per-batch conversion. Only the image model needs the /255 rescale."""
    if kind == "cnn":
        # Neural networks train much better on small, centred inputs than on
        # raw 0-255 integers.
        return xb.float().div_(255.0)
    return xb.float()


# ===========================================================================
# 3. THE TRAINING LOOP
# ===========================================================================
def iterate_minibatches(n, batch_size, rng, shuffle=True):
    """
    Yield arrays of row-indices, `batch_size` at a time.

    WHY MINI-BATCHES?
        Updating the weights after every single sample is noisy and slow.
        Updating only after all 44,100 would give one update per pass. A
        mini-batch of 128 is the compromise: the average gradient over 128
        samples is a good estimate of the true gradient, and we still get
        ~345 weight updates per epoch.

    WHY SHUFFLE EVERY EPOCH?
        The dataset is stored in class order. Without shuffling, every batch
        would contain a single class and the model would lurch back and forth
        instead of learning. Shuffling makes each batch a representative mix.
    """
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


def train_torch_model(kind, X, y, idx_train, idx_val,
                      epochs=None, batch_size=None, lr=None,
                      patience=None, seed=config.SEED, verbose=True,
                      model_kwargs=None):
    """
    Train one neural network and return (best model, history).

    THE TRAINING LOOP IN PLAIN ENGLISH -- for each mini-batch:
        1. forward  : push the inputs through the network to get class scores
        2. loss     : compare the scores with the true labels. Cross-entropy
                      is small when the network puts a high score on the
                      correct class, and large when it is confidently wrong.
        3. backward : work out, for every weight, which direction would
                      reduce the loss. This is backpropagation.
        4. step     : nudge every weight a little in that direction.
    """
    epochs = config.EPOCHS if epochs is None else epochs
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    lr = config.LEARNING_RATE if lr is None else lr
    patience = config.EARLY_STOP_PATIENCE if patience is None else patience

    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(os.cpu_count() or 4)

    model = build_model(kind, **(model_kwargs or {}))

    # CrossEntropyLoss: the standard multi-class loss. It applies softmax
    # internally, in a numerically safer way than doing it in the model.
    criterion = nn.CrossEntropyLoss()
    # Adam keeps a separate, automatically tuned step size per weight, which
    # is why it works well without careful hand-tuning of the learning rate.
    # weight_decay adds mild L2 regularisation: it discourages large weights,
    # a cheap second defence against overfitting.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=config.WEIGHT_DECAY)
    # Halve the learning rate when validation accuracy plateaus: big steps
    # early to find a good region, small steps later to settle into it.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    yt = torch.from_numpy(y)
    tr = torch.from_numpy(idx_train)
    va = torch.from_numpy(idx_val)

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": [], "seconds": []}
    best_val, best_state, stale = -1.0, None, 0

    if verbose:
        print("    {:<6s} {:>10s} {:>9s} {:>10s} {:>9s} {:>7s}".format(
            "epoch", "tr_loss", "tr_acc", "val_loss", "val_acc", "sec"))

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()      # Dropout ON, BatchNorm updating its statistics
        tl = tc = tn = 0
        for bi in iterate_minibatches(len(tr), batch_size, rng):
            rows = tr[bi]
            xb = batch_to_float(kind, X[rows])
            yb = yt[rows]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(rows)
            tc += (out.argmax(1) == yb).sum().item()
            tn += len(rows)

        model.eval()       # Dropout OFF, BatchNorm frozen
        vl = vc = vn = 0
        with torch.no_grad():
            for bi in iterate_minibatches(len(va), 512, rng, shuffle=False):
                rows = va[bi]
                xb = batch_to_float(kind, X[rows])
                yb = yt[rows]
                out = model(xb)
                vl += criterion(out, yb).item() * len(rows)
                vc += (out.argmax(1) == yb).sum().item()
                vn += len(rows)

        train_loss, train_acc = tl / tn, tc / tn
        val_loss, val_acc = vl / vn, vc / vn
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        dt = time.time() - t0

        for k, v in zip(history, (epoch, train_loss, train_acc,
                                  val_loss, val_acc, cur_lr, dt)):
            history[k].append(v)

        mark = ""
        if val_acc > best_val:
            best_val = val_acc
            # Keep the best weights: if later epochs overfit, we still return
            # the best model we ever saw.
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            stale, mark = 0, "  <- best"
        else:
            stale += 1

        if verbose:
            print("    {:<6d} {:>10.4f} {:>9.4f} {:>10.4f} {:>9.4f} "
                  "{:>7.1f}{}".format(epoch, train_loss, train_acc,
                                      val_loss, val_acc, dt, mark))

        # Early stopping: if validation accuracy has not improved for
        # `patience` epochs, the model has stopped learning anything
        # generalisable and is starting to memorise. Stop, keep the best.
        if stale >= patience:
            if verbose:
                print("    early stop (no improvement for {} epochs)".format(patience))
            break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_torch(kind, model, X, idx, batch_size=512):
    model.eval()
    out = np.zeros(len(idx), dtype=np.int64)
    for i in range(0, len(idx), batch_size):
        rows = torch.from_numpy(idx[i:i + batch_size])
        out[i:i + batch_size] = model(batch_to_float(kind, X[rows])).argmax(1).numpy()
    return out


# ===========================================================================
# 4. EVALUATION
# ===========================================================================
def evaluate(y_true, y_pred, osnr_true, osnr_list=None):
    """Overall accuracy, per-OSNR accuracy, and the confusion matrix."""
    osnr_list = config.OSNR_LIST if osnr_list is None else osnr_list
    per = {}
    for o in osnr_list:
        m = (osnr_true == o)
        per[int(o)] = float(np.mean(y_pred[m] == y_true[m])) if m.any() else float("nan")
    return {
        "overall_accuracy": float(np.mean(y_pred == y_true)),
        "accuracy_per_osnr": per,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(config.N_CLASSES))).tolist(),
    }


# ===========================================================================
# 5. RUN EVERY MODEL FOR ONE SEED
# ===========================================================================
NEURAL = ("cnn", "iqcnn", "mlp")
CLASSICAL = ("svm", "knn")
MODEL_LABEL = {"cnn": "CNN (image)", "iqcnn": "IQ-CNN (raw I/Q)",
               "mlp": "MLP (features)", "svm": "SVM (features)",
               "knn": "k-NN (features)"}


def run_one_seed(data, seed, epochs=None, models=None, verbose=True):
    """Train and evaluate every model for a single random seed."""
    models = models or (NEURAL + CLASSICAL)
    y, osnr = data["y"], data["osnr"]
    idx_train, idx_val, idx_test = make_splits(y, osnr, seed=seed)
    y_test, osnr_test = y[idx_test], osnr[idx_test]

    out = {}

    for kind in models:
        t0 = time.time()

        if kind in NEURAL:
            if kind == "iqcnn" and "iq" not in data:
                continue
            if verbose:
                print("\n  [{}] {}".format(seed, MODEL_LABEL[kind]))
            X, _ = prepare_input(kind, data, idx_train)
            model, hist = train_torch_model(kind, X, y, idx_train, idx_val,
                                            epochs=epochs, seed=seed,
                                            verbose=verbose)
            y_pred = predict_torch(kind, model, X, idx_test)
            res = evaluate(y_test, y_pred, osnr_test)
            res["n_parameters"] = int(count_parameters(model))
            res["macs"] = int(count_flops(build_model(kind),
                                          MODEL_INPUT_SHAPE[kind]))
            res["epochs_run"] = len(hist["epoch"])
            res["history"] = hist
            if kind == "cnn":
                torch.save(model.state_dict(),
                           os.path.join(config.RESULTS_DIR,
                                        "cnn_best_seed{}.pt".format(seed)))
        else:
            if verbose:
                print("\n  [{}] {}".format(seed, MODEL_LABEL[kind]))
            maker = baselines.make_svm if kind == "svm" else baselines.make_knn
            cap = 45000 if kind == "svm" else None
            clf, _ = baselines.fit_baseline(maker(), data["feat"][idx_train],
                                            y[idx_train], name=kind,
                                            max_train=cap, verbose=verbose)
            y_pred = clf.predict(data["feat"][idx_test])
            res = evaluate(y_test, y_pred, osnr_test)
            res["n_parameters"] = None
            res["macs"] = None

        res["train_time_sec"] = time.time() - t0
        out[kind] = res
        if verbose:
            print("    TEST ACCURACY = {:.4f}".format(res["overall_accuracy"]))

    return out


def aggregate_seeds(per_seed):
    """
    Combine several seeds into mean and standard deviation.

    WHY THIS MATTERS FOR THE PAPER: a single accuracy number could be luck.
    Reporting mean +/- std over independent seeds is what lets a reviewer
    judge whether a difference between two models is real. If two models'
    error bars overlap heavily, you cannot claim one is better.
    """
    agg = {}
    kinds = sorted({k for s in per_seed.values() for k in s})
    for kind in kinds:
        runs = [s[kind] for s in per_seed.values() if kind in s]
        accs = np.array([r["overall_accuracy"] for r in runs])
        per_osnr = {}
        for o in config.OSNR_LIST:
            vals = np.array([r["accuracy_per_osnr"][o] for r in runs])
            per_osnr[int(o)] = {"mean": float(vals.mean()),
                                "std": float(vals.std(ddof=0))}
        cms = np.array([r["confusion_matrix"] for r in runs], dtype=float)
        agg[kind] = {
            "label": MODEL_LABEL[kind],
            "n_seeds": len(runs),
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=0)),
            "accuracy_all_seeds": accs.tolist(),
            "accuracy_per_osnr": per_osnr,
            "confusion_matrix_mean": cms.mean(axis=0).tolist(),
            "n_parameters": runs[0].get("n_parameters"),
            "macs": runs[0].get("macs"),
            "train_time_sec_mean": float(np.mean(
                [r["train_time_sec"] for r in runs])),
        }
    return agg


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--models", type=str, nargs="+",
                   default=list(NEURAL + CLASSICAL))
    p.add_argument("--out", type=str, default="main_results.json")
    args = p.parse_args()

    print("=" * 72)
    print("  MODULATION FORMAT IDENTIFICATION -- MAIN COMPARISON")
    print("=" * 72)

    data = load_data(args.tag)
    print("  dataset  : {:,} samples, {} classes, {} OSNR levels".format(
        len(data["y"]), config.N_CLASSES, config.N_OSNR))
    print("  classes  : {}".format(", ".join(config.CLASS_NAMES)))
    print("  seeds    : {}".format(args.seeds))
    print("  models   : {}".format(", ".join(args.models)))

    per_seed = {}
    t_start = time.time()
    for seed in args.seeds:
        print("\n" + "=" * 72)
        print("  SEED {}".format(seed))
        print("=" * 72)
        per_seed[seed] = run_one_seed(data, seed, epochs=args.epochs,
                                      models=args.models)

    agg = aggregate_seeds(per_seed)

    results = {
        "experiment": "main_comparison",
        "tag": args.tag or "realistic",
        "class_names": config.CLASS_NAMES,
        "osnr_list": config.OSNR_LIST,
        "seeds": args.seeds,
        "n_total": int(len(data["y"])),
        "total_runtime_sec": time.time() - t_start,
        "aggregate": agg,
        "per_seed": {str(k): v for k, v in per_seed.items()},
    }
    path = os.path.join(config.RESULTS_DIR, args.out)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print("  OVERALL TEST ACCURACY  (mean +/- std over {} seeds)".format(
        len(args.seeds)))
    print("=" * 72)
    print("  {:<20s} {:>18s} {:>12s} {:>12s}".format(
        "model", "accuracy", "params", "MACs"))
    for kind in args.models:
        if kind not in agg:
            continue
        a = agg[kind]
        pars = "{:,}".format(a["n_parameters"]) if a["n_parameters"] else "-"
        macs = "{:,}".format(a["macs"]) if a["macs"] else "-"
        print("  {:<20s} {:>10.2f} +/- {:<4.2f} {:>12s} {:>12s}".format(
            a["label"], 100 * a["accuracy_mean"], 100 * a["accuracy_std"],
            pars, macs))

    print("\n  saved -> {}".format(path))
    print("  runtime: {:.1f} min".format(results["total_runtime_sec"] / 60))
    print("\n  Next:  python plots.py")


if __name__ == "__main__":
    main()
