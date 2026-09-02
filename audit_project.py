"""
audit.py -- full consistency audit of the project.

Checks mechanically, not by eye:
  A. config sanity
  B. all scripts compile
  C. dataset integrity
  D. every results file present and parseable
  E. every macro traceable to its source JSON
  F. HARD-CODED numbers in paper.tex that should be macros (drift risk)
  G. citations resolve, figures exist, no control characters
  H. README / guide consistency with actual results
"""
import json
import os
import re
import sys

ROOT = r"D:\research paper"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import config  # noqa

FAIL, WARN, OK = [], [], []


def fail(m): FAIL.append(m); print("  [FAIL] " + m)
def warn(m): WARN.append(m); print("  [warn] " + m)
def ok(m):   OK.append(m);   print("  [ ok ] " + m)
def head(t): print("\n" + "=" * 70 + "\n  " + t + "\n" + "=" * 70)


def load(n):
    p = os.path.join(config.RESULTS_DIR, n)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except ValueError:
        fail("results/{} is not valid JSON".format(n))
        return None


# ---------------------------------------------------------------- A
head("A. CONFIG SANITY")
ok("N_CLASSES = {} ({})".format(config.N_CLASSES, ", ".join(config.CLASS_NAMES)))
if config.N_CLASSES != len(config.MODULATIONS):
    fail("N_CLASSES disagrees with MODULATIONS")
if len(config.OSNR_LIST) != config.N_OSNR:
    fail("OSNR_LIST length disagrees with N_OSNR")
else:
    ok("OSNR sweep {}..{} = {} levels".format(
        config.OSNR_DB_MIN, config.OSNR_DB_MAX, config.N_OSNR))
if abs(config.TRAIN_FRAC + config.VAL_FRAC + config.TEST_FRAC - 1.0) > 1e-9:
    fail("split fractions do not sum to 1")
else:
    ok("split fractions sum to 1")

# ---------------------------------------------------------------- B
head("B. SCRIPTS COMPILE")
import py_compile
scripts = ["config.py", "data_generation.py", "cnn_model.py", "baselines.py",
           "train_and_evaluate.py", "experiments.py", "experiment_confidence.py",
           "experiment_tolerance.py", "plots.py", "verify_project.py",
           "run_remaining.py", os.path.join("paper", "make_macros.py"),
           os.path.join("paper", "render_preview.py")]
for s in scripts:
    if not os.path.exists(s):
        fail("missing script: " + s)
        continue
    try:
        py_compile.compile(s, doraise=True)
    except Exception as e:
        fail("{} does not compile: {}".format(s, e))
ok("{} scripts checked".format(len(scripts)))

# ---------------------------------------------------------------- C
head("C. DATASET INTEGRITY")
import numpy as np
for tag in ("", "ideal", "severe"):
    p = config.dataset_paths(tag)
    name = tag or "main"
    if not os.path.exists(p["labels"]):
        warn("dataset '{}' absent".format(name))
        continue
    y = np.load(p["labels"])
    o = np.load(p["osnr"])
    cnt = np.bincount(y, minlength=config.N_CLASSES)
    if len(set(cnt.tolist())) != 1:
        fail("dataset '{}' class counts unbalanced: {}".format(name, cnt))
    elif len(cnt) != config.N_CLASSES:
        fail("dataset '{}' has {} classes, config says {}".format(
            name, len(cnt), config.N_CLASSES))
    else:
        ok("dataset '{}': {:,} samples, {} classes balanced ({} each)".format(
            name, len(y), len(cnt), cnt[0]))
    if sorted(np.unique(o).tolist()) != config.OSNR_LIST:
        fail("dataset '{}' OSNR levels do not match config".format(name))

# ---------------------------------------------------------------- D
head("D. RESULTS FILES")
need = {"main_results.json": "main comparison",
        "exp_generalise.json": "generalisation",
        "exp_ablation.json": "ablation",
        "exp_symbols.json": "capture length",
        "exp_robustness.json": "robustness",
        "exp_confidence_ideal.json": "confidence (ideal-trained)",
        "exp_confidence_realistic.json": "confidence (realistic-trained)",
        "exp_tolerance.json": "tolerance budget"}
data = {}
for f, desc in need.items():
    d = load(f)
    data[f] = d
    if d is None:
        fail("missing results: {} ({})".format(f, desc))
    else:
        ok("{:<34s} {}".format(f, desc))

# ---------------------------------------------------------------- E
head("E. MACROS TRACEABLE TO SOURCE")
mac_path = os.path.join(config.PAPER_DIR, "results_macros.tex")
macros = dict(re.findall(r"newcommand\{\\(\w+)\}\{([^}]*)\}",
                         open(mac_path).read()))
ok("{} macros defined".format(len(macros)))

main = data["main_results.json"]
if main:
    agg = main["aggregate"]
    spot = {
        "CNNacc": 100 * agg["cnn"]["accuracy_mean"],
        "SVMacc": 100 * agg["svm"]["accuracy_mean"],
        "MLPacc": 100 * agg["mlp"]["accuracy_mean"],
        "KNNacc": 100 * agg["knn"]["accuracy_mean"],
    }
    for k, v in spot.items():
        if k not in macros:
            fail("macro {} missing".format(k)); continue
        if abs(float(macros[k]) - v) > 0.005:
            fail("macro {} = {} but JSON says {:.2f}".format(k, macros[k], v))
    ok("headline accuracy macros match main_results.json")

    feat = [100 * agg[k]["accuracy_mean"] for k in ("mlp", "svm", "knn")]
    if abs(float(macros["FeatSpread"]) - (max(feat) - min(feat))) > 0.005:
        fail("FeatSpread inconsistent")
    if abs(float(macros["RepGain"]) -
           (100 * agg["cnn"]["accuracy_mean"] - max(feat))) > 0.005:
        fail("RepGain inconsistent")
    ok("FeatSpread / RepGain recomputed and match")

ci = data["exp_confidence_ideal.json"]
if ci:
    for k, tag in (("cnn", "ConfCNN"), ("svm", "ConfSVM")):
        v = 100 * ci["models"][k]["severe"]["accuracy"]
        if abs(float(macros[tag + "sevAcc"]) - v) > 0.005:
            fail("{}sevAcc = {} but JSON says {:.2f}".format(
                tag, macros[tag + "sevAcc"], v))
    ok("confidence macros match exp_confidence_ideal.json")

# ---------------------------------------------------------------- F
head("F. HARD-CODED NUMBERS IN paper.tex (drift risk)")
tex = open(os.path.join(config.PAPER_DIR, "paper.tex"), encoding="utf-8").read()
body = tex.split(r"\begin{thebibliography}")[0]
body_nc = re.sub(r"(?m)^\s*%.*$", "", body)
# percentages written literally, e.g. 93.1\%  or  20.4\%
hard = sorted(set(re.findall(r"(?<![\d.])(\d{1,3}\.\d)\\%", body_nc)))
if hard:
    warn("{} hard-coded percentages found: {}".format(len(hard), ", ".join(hard)))
    warn("  these will NOT update if experiments are re-run -- verify each")
else:
    ok("no hard-coded percentages")

tol = data["exp_tolerance.json"]
if tol:
    print("\n  verifying the hard-coded tolerance numbers against JSON:")
    sw = tol["sweeps"]
    checks = [
        ("iq_gain_imbalance_db", 0.3, "svm", 93.1),
        ("iq_gain_imbalance_db", 2.5, "svm", 20.4),
        ("iq_phase_skew_deg", 2.0, "svm", 93.1),
        ("iq_phase_skew_deg", 14.0, "svm", 20.1),
        ("iq_gain_imbalance_db", 2.5, "cnn", 95.5),
        ("iq_phase_skew_deg", 14.0, "cnn", 96.4),
        ("residual_cd_ps_nm", 150.0, "svm", 80.1),
        ("residual_cd_ps_nm", 150.0, "cnn", 54.9),
    ]
    for param, val, model, claimed in checks:
        pts = [p for p in sw[param]["points"] if abs(p["value"] - val) < 1e-9]
        if not pts:
            fail("no sweep point {}={} to check".format(param, val)); continue
        actual = 100 * pts[0][model + "_accuracy"]
        status = "ok " if abs(actual - claimed) < 0.06 else "FAIL"
        line = "    [{}] {:<22s} {:>7} {} : paper {:.1f}  actual {:.2f}".format(
            status, param, val, model, claimed, actual)
        print(line)
        if status == "FAIL":
            fail("paper says {:.1f}% for {}={} ({}) but data says {:.2f}%"
                 .format(claimed, param, val, model, actual))

# ---------------------------------------------------------------- G
head("G. PAPER STRUCTURE")
cited = set()
for m in re.findall(r"cite\{([^}]*)\}", tex):
    cited.update(x.strip() for x in m.split(","))
defined = set(re.findall(r"bibitem\{([^}]*)\}", tex))
if cited - defined:
    fail("cited but undefined: " + ", ".join(sorted(cited - defined)))
elif defined - cited:
    warn("defined but never cited: " + ", ".join(sorted(defined - cited)))
else:
    ok("{} references, all cited and all defined".format(len(defined)))

figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
missing = [f for f in figs if not os.path.exists(
    os.path.join(config.FIGURES_DIR, f))]
if missing:
    fail("missing figure files: " + ", ".join(missing))
else:
    ok("{} figures referenced, all present".format(len(figs)))

labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
if refs - labels:
    fail("\\ref to undefined labels: " + ", ".join(sorted(refs - labels)))
else:
    ok("all \\ref targets defined")

raw = open(os.path.join(config.PAPER_DIR, "paper.tex"), "rb").read()
ctrl = {c: raw.count(bytes([c])) for c in (7, 8, 11, 12) if raw.count(bytes([c]))}
if ctrl:
    fail("control characters in paper.tex: {}".format(ctrl))
else:
    ok("no control characters")

undef = sorted(u for u in set(re.findall(r"\\([A-Z]\w+)", tex))
               if u not in macros and u not in {
                   "IEEEauthorblockN", "IEEEauthorblockA", "IEEEkeywords",
                   "IEEEoverridecommandlockouts", "Delta"})
if undef:
    fail("capitalised commands with no macro: " + ", ".join(undef))
else:
    ok("every capitalised command resolves to a macro")

for ph in ("[Author Name]", "[Institution]", "[City]", "[email]",
           "[repository URL]"):
    if ph in tex:
        warn("placeholder still present: " + ph)

# ---------------------------------------------------------------- H
head("H. README / GUIDE CONSISTENCY")
for fn in ("README.md", "PROJECT_GUIDE.html"):
    if not os.path.exists(fn):
        warn("missing " + fn); continue
    t = open(fn, encoding="utf-8").read()
    stale = []
    if "33.3%" in t or "3 classes" in t or "three formats" in t.lower():
        stale.append("mentions 3 classes / 33.3% chance")
    if main:
        head_acc = "{:.2f}".format(100 * main["aggregate"]["cnn"]["accuracy_mean"])
        if head_acc not in t:
            stale.append("does not contain current CNN accuracy " + head_acc)
    if stale:
        for s in stale:
            warn("{}: {}".format(fn, s))
    else:
        ok(fn + " consistent with current results")

# ---------------------------------------------------------------- summary
head("SUMMARY")
print("  passed : {}".format(len(OK)))
print("  warns  : {}".format(len(WARN)))
print("  FAILS  : {}".format(len(FAIL)))
if FAIL:
    print("\n  MUST FIX:")
    for f in FAIL:
        print("    - " + f)
if WARN:
    print("\n  review:")
    for w in WARN:
        print("    - " + w)
sys.exit(1 if FAIL else 0)
