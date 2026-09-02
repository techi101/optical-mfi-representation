# Where we left off

**Last updated:** 2 September 2026
**Author:** Nitesh Kumar Yadav — niteshkumar88449@gmail.com
**Repo:** https://github.com/techi101/optical-mfi-representation *(private)*

---

## How to resume this conversation

Open a terminal in `D:\research paper` and run:

```bash
claude --continue
```

That picks up this exact conversation with full context. If it does not, use
`claude --resume` and choose this session from the list.

If neither works, just start `claude` and say: *"read HANDOFF.md and continue"*.

---

## Status: the research is finished

| | |
|---|---|
| Experiments | **7 of 7 complete** |
| Figures | 11, all inspected |
| Manuscript | complete, IEEE format, 10 references |
| Numbers in paper | 135, all auto-generated from the results |
| Audit | 27 passed, 0 failures |
| GitHub | pushed, private, 4 commits |

Run `python audit_project.py` any time to re-verify everything.

---

## The paper, in one paragraph

Everyone knows CNNs on constellation images beat classical methods for
modulation format identification. But that comparison changes two things at
once — the model becomes deep *and* the input becomes an image. This paper
separates them by adding an MLP trained on the same 15 hand-crafted features.
The representation turns out to matter about three times more than the
algorithm, the gap widens enormously under channel mismatch, and the
feature-based classifier cannot reliably tell you when it has failed.

## Headline numbers

| Model | Input | Accuracy |
|---|---|---|
| **CNN** | 64×64 image | **98.73 ± 0.12 %** |
| SVM | 15 features | 93.41 ± 0.10 % |
| MLP | 15 features | 93.08 ± 0.10 % |
| k-NN | 15 features | 91.58 ± 0.08 % |
| IQ-CNN | raw I/Q | 86.56 ± 4.64 % *(high variance — do not over-claim)* |

- **Algorithm effect: 1.83 points. Representation effect: 5.32 points.** ~3×.
- **7 dB OSNR margin** — CNN reaches 95% at 7 dB, SVM needs 14 dB.
- **Robustness:** trained clean, tested severe — CNN 89.8%, SVM **21.8%** (chance = 20%).
- **Confidence:** at a 90% gate the CNN answers 86.9% of inputs, the SVM 0.2%. **447×.**
- **Tolerance:** I/Q imbalance is what kills cumulants. But the SVM beats the
  CNN on chromatic dispersion (80.1% vs 54.9%) — reported honestly.

Full numbers: `results/results_summary.txt` and `paper/results_macros.txt`.

---

## THE ONE THING STILL MISSING

`paper/paper.tex` line ~39 still says **`[Institution]`**. Replace it with your
college's full official name. That is the only placeholder left.

---

## What you need to do (not me)

1. **Ask your professor whether they should be a co-author.** Most colleges
   expect the project guide. Do this early — adding an author later is awkward.
2. **Compile on Overleaf** for real IEEE two-column formatting. Upload
   `paper.tex`, `results_macros.tex` and the `figures/` folder. Set the
   compiler to pdfLaTeX. *(LaTeX is not installed on this machine, which is why
   `paper/paper_preview.pdf` is a browser-rendered approximation.)*
3. **Read the paper aloud** and rewrite anything that does not sound like you.
   This does more for authenticity than any editing, and doubles as viva prep.
4. **Trim to 6 pages.** You have more material than fits. Convert the ablation
   and generalisation figures to compact tables first — that saves about
   two-thirds of a page with almost no loss.
5. **Sort out the publication fee.** Ask your department. If they will not fund
   it, arXiv is free and IEEE *journals* charge nothing for traditional
   (non-open-access) submission.
6. **Send the data-request emails** in `OUTREACH_EMAILS.md` if you want to try
   for experimental validation. Replies take weeks — start soon.

---

## Target venue

**QPAIN 2027** — IEEE 3rd Int. Conf. on Quantum Photonics, AI & Networking

- **Deadline: 30 November 2026** (before your 10 December college deadline)
- 8–10 April 2027, Chattogram, Bangladesh
- IEEE Photonics Society sponsored, **IEEE Xplore + Scopus indexed**
- 4–6 pages, IEEE template — your paper already fits
- ⚠️ **Registration fee not yet checked.** Find this before committing.

Backup: **PHOTOPTICS** (22 October deadline, tighter).

---

## Decisions already made — do not relitigate

- **PyTorch, not TensorFlow.** Already installed; explicit training loop is
  easier to defend in a viva.
- **Simulation, not measured data.** No public optical constellation dataset
  exists (checked). A real testbed costs crores. Simulation also gives exact
  controlled OSNR sweeps, which is what the paper needs.
- **No topic change to quantum.** Checked the literature; every adjacent area
  is crowded or out of reach, and there is not enough time to learn CV-QKD
  security analysis safely.
- **OSNR-estimation experiment deferred and then dropped.** The paper is
  already over-full for 6 pages. It would make a better second paper.
- **Novelty is narrower than first assumed.** Ge et al. (2021, *Sensors*) did
  the representation comparison in *wireless* AMC. The paper cites them and
  positions honestly: optical domain, both directions controlled, and a
  different conclusion. The genuinely new parts are the cross-channel
  robustness and the representation × self-detectability interaction.

---

## Files

```
config.py                  all constants and the random seed
data_generation.py         the optical link simulator
cnn_model.py               the three neural networks
baselines.py               SVM and k-NN
train_and_evaluate.py      training harness + main comparison
experiments.py             generalise / ablation / symbols / robustness
experiment_confidence.py   failure self-detection
experiment_tolerance.py    impairment tolerance budget
plots.py                   all 11 figures
verify_project.py          independent correctness checks
audit_project.py           full consistency audit  ← run after any change
run_remaining.py           unattended pipeline runner

paper/paper.tex            the manuscript
paper/make_macros.py       generates every number in the paper from JSON
paper/render_preview.py    tex → readable HTML/PDF (no LaTeX needed)
paper/README.md            how to compile, and what to check

README.md                  full project documentation, 25 viva Q&As
PROJECT_GUIDE.html         reading guide
CONSTELLATION_LAB.html     interactive teaching page, 4 live simulators
OUTREACH_EMAILS.md         drafted data-request emails (not in git)
```

**Not in git:** `data/` (500 MB, regenerates from the seed in ~5 min) and
`results/*.pt` (model weights).

## To rebuild everything from scratch

```bash
python data_generation.py --n-per-class 600 --profile realistic   # ~5 min
python train_and_evaluate.py --seeds 42 43 44                     # ~65 min
python experiments.py generalise                                   # ~33 min
python experiments.py ablation                                     # ~44 min
python experiments.py symbols                                      # ~23 min
python experiments.py robustness                                   # ~30 min
python experiment_confidence.py --train-profile ideal              # ~12 min
python experiment_tolerance.py                                     # ~70 min
python plots.py && python paper/make_macros.py
python verify_project.py && python audit_project.py
```

⚠️ **Close browser tabs before long runs.** Two runs were killed by Windows
under memory pressure when Chrome/Brave held ~7 GB. The experiments now
checkpoint after every step, so a kill costs one step rather than the run.

⚠️ **Use the right Python.** `python` on PATH is an MSYS2 3.14 build with no
pip. Use `C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe`.

---

## Your three links

- **Paper preview** — https://claude.ai/code/artifact/c222c074-cede-40e0-9d00-7205e281e5dc
- **Interactive lab** — https://claude.ai/code/artifact/04457caf-6e50-4300-b3e3-22bd5034c525
- **Reading guide** — https://claude.ai/code/artifact/9af5fbf1-7706-4678-a468-ddc7b19e5d14

Find them again any time with `/artifacts` in the terminal.
