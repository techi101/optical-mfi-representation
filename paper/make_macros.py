"""
make_macros.py
==============
Generate results_macros.tex from the experiment JSON files.

WHY DO THIS INSTEAD OF TYPING THE NUMBERS INTO THE PAPER?
Because hand-copying numbers from a results file into a manuscript is where
papers get their embarrassing errors. If you re-run an experiment and forget
to update one number in the text, the paper silently contradicts its own
table -- and reviewers do notice.

Here every number in the manuscript is a LaTeX macro such as \\CNNacc, and
this script writes those macros straight from the JSON. Re-run the
experiments, re-run this, recompile: the paper is guaranteed to be
consistent with the results.

Run me (from the project root):
    python paper/make_macros.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config    # noqa: E402


def load(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=2):
    return "{:.{}f}".format(x, nd)


def main():
    main_res = load("main_results.json")
    if main_res is None:
        raise SystemExit("results/main_results.json not found. Run "
                         "train_and_evaluate.py first.")
    gen = load("exp_generalise.json")
    abl = load("exp_ablation.json")
    sym = load("exp_symbols.json")
    rob = load("exp_robustness.json")

    agg = main_res["aggregate"]
    osnr = config.OSNR_LIST
    M = {}      # macro name -> value

    # ---------------- dataset ----------------
    M["NumFormats"] = str(config.N_CLASSES)
    M["FormatList"] = ", ".join(config.CLASS_NAMES).replace("QAM", "-QAM")
    M["NumImages"] = "{:,}".format(main_res["n_total"]).replace(",", ",")
    M["OSNRmin"] = str(config.OSNR_DB_MIN)
    M["OSNRmax"] = str(config.OSNR_DB_MAX)
    M["NumOSNR"] = str(config.N_OSNR)
    M["NumSymbols"] = str(config.N_SYMBOLS)
    M["ImgSize"] = str(config.IMG_SIZE)
    M["SymbolRate"] = "{:.0f}".format(config.SYMBOL_RATE / 1e9)
    M["NumSeeds"] = str(len(main_res["seeds"]))
    # Drop the decimal when the chance level is a whole number (5 classes ->
    # "20%", not "20.0%"), but keep it when it is not (3 classes -> "33.3%").
    chance = 100.0 / config.N_CLASSES
    M["ChanceLevel"] = fmt(chance, 0 if abs(chance - round(chance)) < 1e-9 else 1)

    # ---------------- headline accuracies ----------------
    short = {"cnn": "CNN", "iqcnn": "IQCNN", "mlp": "MLP",
             "svm": "SVM", "knn": "KNN"}
    for k, s in short.items():
        if k not in agg:
            continue
        M[s + "acc"] = fmt(100 * agg[k]["accuracy_mean"])
        M[s + "std"] = fmt(100 * agg[k]["accuracy_std"])
        if agg[k]["n_parameters"]:
            M[s + "params"] = "{:,}".format(agg[k]["n_parameters"])
            # Report in millions, but keep enough decimals for the small
            # models: the MLP is ~0.01 M MACs and rounding to one decimal
            # printed a meaningless "0.0".
            macs_m = agg[k]["macs"] / 1e6
            M[s + "macs"] = fmt(macs_m, 1 if macs_m >= 1 else 3)

    # ---------------- the central claim ----------------
    feat_keys = [k for k in ("mlp", "svm", "knn") if k in agg]
    if feat_keys and "cnn" in agg:
        feat = [100 * agg[k]["accuracy_mean"] for k in feat_keys]
        M["FeatSpread"] = fmt(max(feat) - min(feat))
        M["FeatBest"] = fmt(max(feat))
        M["FeatWorst"] = fmt(min(feat))
        M["RepGain"] = fmt(100 * agg["cnn"]["accuracy_mean"] - max(feat))
        M["RepRatio"] = fmt((100 * agg["cnn"]["accuracy_mean"] - max(feat))
                            / max(max(feat) - min(feat), 1e-9), 1)

    # ---------------- OSNR thresholds ----------------
    def first_above(k, th):
        for o in osnr:
            if agg[k]["accuracy_per_osnr"][str(o)]["mean"] >= th:
                return o
        return None

    for k, s in short.items():
        if k not in agg:
            continue
        for th, tag in ((0.95, "NinetyFive"), (0.99, "NinetyNine")):
            v = first_above(k, th)
            M[s + "OSNR" + tag] = str(v) if v is not None else "--"

    if "cnn" in agg and "svm" in agg:
        c, s_ = first_above("cnn", 0.95), first_above("svm", 0.95)
        if c and s_:
            M["OSNRMargin"] = str(s_ - c)

    # ---------------- regimes ----------------
    for k, s in short.items():
        if k not in agg:
            continue
        lo = np.mean([agg[k]["accuracy_per_osnr"][str(o)]["mean"]
                      for o in osnr if o <= 10])
        hi = np.mean([agg[k]["accuracy_per_osnr"][str(o)]["mean"]
                      for o in osnr if o >= 20])
        M[s + "accLow"] = fmt(100 * lo)
        M[s + "accHigh"] = fmt(100 * hi)

    # ---------------- generalisation ----------------
    if gen:
        for k, s in short.items():
            if k in gen.get("summary", {}):
                M[s + "GenAcc"] = fmt(100 * gen["summary"][k]["mean"])
                M[s + "GenStd"] = fmt(100 * gen["summary"][k]["std"])
                if k in agg:
                    M[s + "GenDrop"] = fmt(
                        100 * agg[k]["accuracy_mean"]
                        - 100 * gen["summary"][k]["mean"])

    # ---------------- ablation ----------------
    if abl:
        rows = abl["results"]
        best = max(rows, key=lambda r: r["accuracy"])
        M["AblBestBlocks"] = str(best["n_blocks"])
        M["AblBestCh"] = str(best["base_ch"])
        M["AblBestAcc"] = fmt(100 * best["accuracy"])
        tiny = min(rows, key=lambda r: r["n_parameters"])
        M["AblTinyParams"] = "{:,}".format(tiny["n_parameters"])
        M["AblTinyAcc"] = fmt(100 * tiny["accuracy"])
        M["AblTinyCh"] = str(tiny["base_ch"])
        # The DEPTH comparison must hold width fixed, otherwise it is not a
        # depth comparison at all. All three of these are base_ch = 16.
        for n_blocks, tag in ((2, "Two"), (3, "Three"), (4, "Four")):
            hit = [r for r in rows
                   if r["n_blocks"] == n_blocks and r["base_ch"] == 16]
            if hit:
                M["Abl{}BlockAcc".format(tag)] = fmt(100 * hit[0]["accuracy"])

        # The honest summary of the ablation: the full spread across every
        # variant. With a single seed the run-to-run noise is of the same
        # order, so the correct claim is "all within noise", not "X is best".
        accs = [100 * r["accuracy"] for r in rows]
        M["AblSpread"] = fmt(max(accs) - min(accs))
        M["AblMin"] = fmt(min(accs))
        M["AblMax"] = fmt(max(accs))
        biggest = max(rows, key=lambda r: r["n_parameters"])
        M["AblBigParams"] = "{:,}".format(biggest["n_parameters"])
        M["AblBigAcc"] = fmt(100 * biggest["accuracy"])
        M["AblParamRatio"] = fmt(
            biggest["n_parameters"] / tiny["n_parameters"], 1)
        M["AblMacRatio"] = fmt(biggest["macs"] / tiny["macs"], 0)
        M["AblTinyMacs"] = fmt(tiny["macs"] / 1e6, 2)

    # ---------------- capture length ----------------
    if sym:
        rows = sorted(sym["results"], key=lambda r: r["n_symbols"])
        M["SymMin"] = str(rows[0]["n_symbols"])
        M["SymMax"] = str(rows[-1]["n_symbols"])
        M["SymMinAcc"] = fmt(100 * rows[0]["cnn_accuracy"])
        M["SymMaxAcc"] = fmt(100 * rows[-1]["cnn_accuracy"])
        # smallest capture that still reaches 95% of the full-capture accuracy
        target = 0.95 * rows[-1]["cnn_accuracy"]
        ok = [r for r in rows if r["cnn_accuracy"] >= target]
        M["SymEnough"] = str(ok[0]["n_symbols"]) if ok else "--"
        M["SymEnoughAcc"] = fmt(100 * ok[0]["cnn_accuracy"]) if ok else "--"

    # ---------------- robustness ----------------
    if rob:
        mt = rob["matrix"]
        M["RobIdealIdealCNN"] = fmt(100 * mt["ideal"]["ideal"]["cnn"])
        M["RobIdealRealCNN"] = fmt(100 * mt["ideal"]["realistic"]["cnn"])
        M["RobIdealSevereCNN"] = fmt(100 * mt["ideal"]["severe"]["cnn"])
        M["RobRealRealCNN"] = fmt(100 * mt["realistic"]["realistic"]["cnn"])
        M["RobRealSevereCNN"] = fmt(100 * mt["realistic"]["severe"]["cnn"])
        M["RobIdealIdealSVM"] = fmt(100 * mt["ideal"]["ideal"]["svm"])
        M["RobIdealSevereSVM"] = fmt(100 * mt["ideal"]["severe"]["svm"])
        M["RobRealSevereSVM"] = fmt(100 * mt["realistic"]["severe"]["svm"])
        M["RobCNNDrop"] = fmt(100 * (mt["ideal"]["ideal"]["cnn"]
                                     - mt["ideal"]["severe"]["cnn"]))
        M["RobSVMDrop"] = fmt(100 * (mt["ideal"]["ideal"]["svm"]
                                     - mt["ideal"]["severe"]["svm"]))


    # ---------------- failure self-detection (experiment 6) ----------------
    conf_i = load("exp_confidence_ideal.json")
    conf_r = load("exp_confidence_realistic.json")
    if conf_i:
        m = conf_i["models"]
        for k, tag in (("cnn", "ConfCNN"), ("mlp", "ConfMLP"), ("svm", "ConfSVM")):
            d = m[k]["severe"]
            M[tag + "sevAcc"] = fmt(100 * d["accuracy"])
            M[tag + "sevConf"] = fmt(100 * d["mean_confidence"])
            M[tag + "sevOver"] = fmt(100 * d["overconfidence"])
            M[tag + "sevAuroc"] = fmt(d["error_auroc"], 3)
            # coverage and accuracy at the 90% confidence gate
            row = [r for r in conf_i["thresholding"][k]
                   if abs(r["threshold"] - 0.9) < 1e-9][0]
            M[tag + "Cov90"] = fmt(100 * row["coverage"], 1)
            M[tag + "Acc90"] = fmt(100 * row["accuracy_on_kept"])
        cov_cnn = [r for r in conf_i["thresholding"]["cnn"]
                   if abs(r["threshold"] - 0.9) < 1e-9][0]["coverage"]
        cov_svm = [r for r in conf_i["thresholding"]["svm"]
                   if abs(r["threshold"] - 0.9) < 1e-9][0]["coverage"]
        M["CovRatio"] = fmt(cov_cnn / max(cov_svm, 1e-9), 0)
    if conf_r:
        m = conf_r["models"]
        M["ConfSVMmodSevAcc"] = fmt(100 * m["svm"]["severe"]["accuracy"])
        M["ConfSVMmodSevOver"] = fmt(100 * m["svm"]["severe"]["overconfidence"])
        row = [r for r in conf_r["thresholding"]["svm"]
               if abs(r["threshold"] - 0.9) < 1e-9][0]
        M["ConfSVMmodCov90"] = fmt(100 * row["coverage"], 1)


    # ---------------- tolerance budget (experiment 7) ----------------
    tol = load("exp_tolerance.json")
    if tol:
        sw = tol["sweeps"]

        def at(param, value, model):
            pts = [q for q in sw[param]["points"]
                   if abs(q["value"] - value) < 1e-9]
            return 100 * pts[0][model + "_accuracy"] if pts else float("nan")

        M["TolSVMgainLo"] = fmt(at("iq_gain_imbalance_db", 0.3, "svm"))
        M["TolSVMgainHi"] = fmt(at("iq_gain_imbalance_db", 2.5, "svm"))
        M["TolCNNgainHi"] = fmt(at("iq_gain_imbalance_db", 2.5, "cnn"))
        M["TolSVMskewLo"] = fmt(at("iq_phase_skew_deg", 2.0, "svm"))
        M["TolSVMskewHi"] = fmt(at("iq_phase_skew_deg", 14.0, "svm"))
        M["TolCNNskewHi"] = fmt(at("iq_phase_skew_deg", 14.0, "cnn"))
        M["TolSVMcdHi"] = fmt(at("residual_cd_ps_nm", 150.0, "svm"))
        M["TolCNNcdHi"] = fmt(at("residual_cd_ps_nm", 150.0, "cnn"))
        b90 = tol.get("budget", {}).get("0.9", {})
        for key, tag in (("iq_gain_imbalance_db", "Gain"),
                         ("iq_phase_skew_deg", "Skew"),
                         ("residual_cd_ps_nm", "CD"),
                         ("freq_offset_hz", "Foff"),
                         ("laser_linewidth_hz", "Lw")):
            if key in b90:
                for m, mt in (("cnn", "CNN"), ("svm", "SVM")):
                    v = b90[key][m]
                    M["Budget" + mt + tag] = ("--" if v != v
                                              else fmt(v, 1).rstrip("0").rstrip("."))

    # ---------------- write ----------------
    out = [
        "% results_macros.tex -- AUTO-GENERATED by paper/make_macros.py",
        "% Do not edit by hand: re-run the script instead.",
        "% Every number in the manuscript comes from here, so the text can",
        "% never disagree with the experiments.",
        "",
    ]
    for k in sorted(M):
        out.append("\\newcommand{{\\{}}}{{{}}}".format(k, M[k]))

    path = os.path.join(config.PAPER_DIR, "results_macros.tex")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote {} macros -> {}".format(len(M), path))

    # also dump a plain-text version so you can read the numbers directly
    txt = os.path.join(config.PAPER_DIR, "results_macros.txt")
    with open(txt, "w") as f:
        for k in sorted(M):
            f.write("{:<24s} {}\n".format(k, M[k]))
    print("wrote plain-text copy   -> {}".format(txt))


if __name__ == "__main__":
    main()
