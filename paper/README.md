# The manuscript

`paper.tex` is a complete IEEE-conference-format draft of the paper.

## How to build it

```bash
# 1. regenerate every number in the paper from the experiment results
python paper/make_macros.py

# 2. compile (from inside the paper/ directory)
cd paper
pdflatex paper.tex
pdflatex paper.tex        # twice, so cross-references resolve
```

You need a LaTeX distribution. On Windows, **MiKTeX** (<https://miktex.org>)
is the easiest option — it downloads missing packages automatically the first
time you compile. **Overleaf** (<https://overleaf.com>) works too: upload
`paper.tex`, `results_macros.tex` and the `figures/` folder, and set the
compiler to pdfLaTeX.

## How the numbers work — read this

**No number is typed into `paper.tex` by hand.** Every result in the text is a
LaTeX macro (`\CNNacc`, `\OSNRMargin`, …) defined in `results_macros.tex`,
which `make_macros.py` generates directly from the experiment JSON files.

Why this matters: hand-copying numbers from results into a manuscript is where
papers get their embarrassing errors. If you re-run an experiment and forget
to update one sentence, the prose silently contradicts your own table — and
reviewers do notice. With this setup, re-running `make_macros.py` and
recompiling guarantees the paper agrees with the experiments.

`results_macros.txt` is a plain-text copy of the same numbers, so you can read
them without compiling anything.

---

## ⚠️ Two things you MUST do before submitting

### 1. Verify every citation yourself

The reference list contains six papers that are genuinely well known in this
field, and they are the right works to cite. **But I wrote the bibliographic
details from memory, and I cannot guarantee the volume numbers, page numbers,
or years are exact.** Some may be slightly wrong.

Before submitting, look up each one on IEEE Xplore, Optica Publishing, or
Google Scholar and replace the entry with the publisher's official citation.
This takes about fifteen minutes and it is not optional — a reviewer who
finds a wrong page number will assume you did not read the paper.

The six references are:

| Key | Paper |
|---|---|
| `khan2019perspective` | Khan, Fan, Lu, Lau — *An optical communication's perspective on machine learning and its applications*, J. Lightwave Technol. |
| `swami2000cumulants` | Swami & Sadler — *Hierarchical digital modulation classification using cumulants*, IEEE Trans. Commun. |
| `dobre2007survey` | Dobre, Abdi, Bar-Ness, Su — *Survey of automatic modulation classification techniques*, IET Commun. |
| `wang2017analyzer` | Wang et al. — *Intelligent constellation diagram analyzer using CNN-based deep learning*, Optics Express |
| `khan2016deep` | Khan et al. — *Modulation format identification in coherent receivers using deep machine learning*, IEEE Photon. Technol. Lett. |
| `oshea2016radio` | O'Shea, Corgan, Clancy — *Convolutional radio modulation recognition networks*, EANN |

### 2. Do a literature check of your own

I have not searched the recent literature. Before you submit, spend an hour
on Google Scholar searching for *"modulation format identification CNN
constellation"* and check:

- **Has someone already done the representation-vs-architecture comparison?**
  This is the paper's core novelty claim. If a 2022–2025 paper already
  separates these two factors, you must cite it and reposition your
  contribution (e.g. as extending it to non-square formats and to
  cross-channel robustness).
- **What accuracies do recent papers report?** A related-work table comparing
  your numbers against published ones strengthens the paper considerably, and
  reviewers often expect it.
- **Are there newer standard baselines** you should compare against?

This is genuinely your job rather than something to delegate — you need to
know this literature to defend the paper in review and in your viva.

---

## Fill in before submitting

Search `paper.tex` for square brackets:

- `[Author Name]`, `[Institution]`, `[City]`, `[email]`
- `[repository URL]` in the Reproducibility section — upload the code to
  GitHub and put the link here. Reproducibility statements are increasingly
  expected and cost you nothing.

## Figures used

`paper.tex` pulls figures from `../figures/`:

| File | Used as |
|---|---|
| `sample_constellations.png` | Fig. 1 — example inputs |
| `fig2_representation.png` | Fig. 2 — the central claim |
| `fig1_accuracy_vs_osnr.png` | Fig. 3 — accuracy vs OSNR |
| `fig9_generalisation.png` | Fig. 4 — unseen OSNR |
| `fig6_ablation_pareto.png` | Fig. 5 — ablation |
| `fig7_symbols.png` | Fig. 6 — capture length |
| `fig8_robustness.png` | Fig. 7 — cross-channel robustness |

If a figure is missing, the corresponding experiment has not been run yet —
see the main `README.md` for the run order.

## If you need Word instead of LaTeX

Many Indian conferences accept Word. Options, best first:

1. **Overleaf → download PDF**, and submit the PDF if permitted. Cleanest.
2. **Pandoc**: `pandoc paper.tex -o paper.docx` — converts text and structure
   but loses IEEE formatting and mangles equations. You will need to fix the
   equations and re-apply the IEEE Word template by hand.
3. **Retype into the IEEE Word template**, using `results_macros.txt` for the
   numbers. Tedious but gives the most reliable formatting.
