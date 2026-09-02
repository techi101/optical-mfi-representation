# Data request emails — ready to send

**From:** niteshkumar88449@gmail.com

Send these **one at a time**, a few days apart. Do not BCC them together — it
reads as a mass mailing and gets ignored.

Find current email addresses on the papers themselves (usually on the first
page, next to the corresponding author's name) or on the group's university
page. Search: *"Xiangjun Xin BUPT"*, *"Danshi Wang BUPT"*,
*"Faisal Nadeem Khan Tsinghua"*.

---

## Email 1 — Prof. Xiangjun Xin / Prof. Qi Zhang (BUPT) — BEST TARGET

**Subject:** Request for a small sample of experimental constellation data — undergraduate MFI study

Dear Professor Xin,

I am a final-year undergraduate in Electronics and Communication Engineering
in India, working on a solo research project on modulation format
identification for optical performance monitoring.

I read your 2022 *Photonics* paper, "Modulation format identification based on
signal constellation diagrams and support vector machine", with interest —
particularly the comparison between constellation-image features and
higher-order-statistics features.

My study compares input representations for MFI: a CNN on constellation
density images against SVM, k-NN and an MLP on higher-order cumulants, over
five formats and OSNR from 5 to 25 dB. One result is that the cumulant-based
classifier loses almost all accuracy under I/Q imbalance while the image-based
model does not.

All of my data is simulated, which is the main limitation of the work. I would
like to check whether the conclusion survives on real measurements.

Would you be willing to share a small sample of experimental data from your
coherent testbed? Specifically:

- received I/Q samples after standard DSP (CD compensation, timing recovery,
  carrier phase recovery)
- any two or three formats among QPSK, 16-QAM and 64-QAM
- roughly 4,000 symbols per capture, and only about 20–50 captures in total
- the format label, and the OSNR if it was recorded
- any convenient format (.mat, .npy or .csv)

This would be a few megabytes at most. I would cite your work, acknowledge
your group explicitly, and I am happy to share my code, results and manuscript
with you beforehand. If you would prefer any other form of acknowledgement or
involvement, I would be glad to discuss it.

I understand you are busy, and I appreciate your time either way.

With respect,

Nitesh Kumar
B.Tech, Electronics and Communication Engineering (AI/ML)
niteshkumar88449@gmail.com

---

## Email 2 — Prof. Danshi Wang (BUPT)

**Subject:** Small experimental data request — undergraduate study on constellation-based MFI

Dear Professor Wang,

I am a final-year undergraduate in Electronics and Communication Engineering
in India, working on a solo project on modulation format identification.

Your 2017 *Optics Express* paper, "Intelligent constellation diagram analyzer
using convolutional neural network-based deep learning", is a central
reference for my work.

My study asks a narrower question than yours: whether the reported advantage
of CNNs on constellation diagrams comes from the deep model or from the image
representation itself. To separate the two, I train an MLP on the same
hand-crafted features given to the SVM. The representation turns out to matter
several times more than the choice of algorithm, and the gap widens sharply
when the channel differs from the one used for training.

My evaluation is entirely simulation-based, which is its main weakness. I
would like to test whether the finding holds on measured data.

Would you be willing to share a small sample from your experimental setup?
About 20–50 captures of roughly 4,000 received I/Q symbols each, after
standard DSP, for two or three modulation formats, with format labels and
OSNR values if available. Any file format is fine, and the total would be only
a few megabytes.

I would cite your work, acknowledge your group, and am happy to share my code
and manuscript in advance.

Thank you for your time.

With respect,

Nitesh Kumar
B.Tech, Electronics and Communication Engineering (AI/ML)
niteshkumar88449@gmail.com

---

## Email 3 — Prof. Faisal Nadeem Khan (Tsinghua Shenzhen)

**Subject:** Undergraduate request: small sample of experimental MFI data

Dear Professor Khan,

I am a final-year undergraduate in Electronics and Communication Engineering
in India, working on a solo research project on modulation format
identification for optical performance monitoring. Your 2016 *IEEE Photonics
Technology Letters* paper and your 2019 *JLT* review are both central
references for my work.

I have built a simulation study comparing input representations for MFI across
five formats and OSNR from 5 to 25 dB. The finding is that the input
representation matters considerably more than the choice of classifier, and
that cumulant-based classifiers fail in a specific way under I/Q imbalance,
which image-based classifiers do not.

The evaluation is simulation-only, which is its main limitation. I would like
to know whether the conclusion holds on measured data.

Would you, or anyone in your group, be able to share a small sample of
experimental coherent receiver data? Around 20–50 captures of roughly 4,000
I/Q symbols each, after standard DSP, for two or three formats, with labels.
A few megabytes in any format would be enough.

I would cite and acknowledge your group, and would be glad to share the code
and manuscript first.

Thank you for considering this.

With respect,

Nitesh Kumar
B.Tech, Electronics and Communication Engineering (AI/ML)
niteshkumar88449@gmail.com

---

## Before you send

1. **Check the names and titles.** Look up each professor's current
   institution and preferred form of address. Getting this wrong loses you the
   email immediately.
2. **Do not attach anything** to the first message. Attachments from unknown
   senders get filtered.
3. **Send from a signature-bearing account** if you can. If your college gives
   you an institutional address, use it instead of Gmail — a `.edu` or
   `.ac.in` address is taken far more seriously than a personal one.
4. **Ask your project guide to be copied in**, or to send a one-line
   endorsement. A supervisor's name transforms the response rate.
5. **Wait two weeks** before a single polite follow-up. One follow-up only.

## Realistic expectations

Cold emails to research groups for data have a low hit rate — perhaps one
reply in three or four, and fewer that result in actual data. Sharing
experimental data involves institutional permissions that are often not the
professor's alone to give.

So: send all three, expect nothing, and continue as though the answer is no.
If data does arrive, it is a substantial upgrade to the paper. If it does not,
the paper stands on its own with simulation-only stated plainly as a
limitation.

**Start now.** Replies take weeks, and your deadline is 30 November.
