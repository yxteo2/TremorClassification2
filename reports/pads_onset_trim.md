# PADS carries a class-ordered arm-raising onset, and removing it changes nothing

## What is in the data

`extract_pads --trim-start` defaults to 0 s on one early, unpaired PADS-only
ET-F1 comparison. The onset it exists to remove is real, and it is not noise.

Robust-z outliers (>10 MAD, any axis) in PADS StretchHold sit almost entirely in
the first third of each recording — N 0.98, PD 0.89, ET 0.92 of all outlier
samples — and they are multi-sample events (median run 3 samples, 24 % of runs
≥ 5, max 221 = 2.2 s), so they are movement, not sensor glitches.

In absolute terms, in-band (3–15 Hz) RMS of the first 1.5 s over the remainder:

| cohort | N | PD | ET |
|---|---|---|---|
| **PADS** | **1.39** (31 % > 2×) | **1.33** (18 %) | **1.06** (11 %) |
| 2015 | 0.96 | 0.97 | 0.93 |
| NewData | 1.03 | 0.99 | 1.02 |

**Only PADS has an onset excess, and within PADS it is ordered N > PD > ET.**
Healthy controls raise the arm against almost no tremor, so the transient
dominates their first second; ET patients tremble from the moment the posture is
adopted, so it barely registers. The onset adds in-band power preferentially to
N and PD — a broadband signature ET lacks, and one that would *help* a
classifier within PADS while being absent from every in-house recording.

That is the shape of an artifact that could explain the measured fact that PADS
does not transfer to in-house patients. So it was tested properly, with a
length-matched control: drop the first 1.5 s, or drop the *last* 1.5 s (same
length, same frame count, onset left in). Only PADS is modified.

## (A) Mechanism — held

| arm | N | PD | ET |
|---|---|---|---|
| untrimmed | 1.39 / Q 1.73 | 1.33 / Q 2.32 | 1.06 / Q 3.77 |
| **trim-start** | **1.10 / Q 1.95** | **1.04 / Q 2.50** | **0.96 / Q 3.81** |
| trim-end (control) | 1.40 / Q 1.74 | 1.33 / Q 2.36 | 1.05 / Q 3.69 |

Trimming the start collapses the ordering toward 1.0 and sharpens the N and PD
peaks, exactly as predicted; trimming the end does neither. The trim removes
what it was meant to remove.

## (C) Transfer, PADS → in-house PD-vs-ET — the verdict arm — failed

Logistic regression on the 16-bin spectrum, fitted on PADS PD/ET, scored on all
119 in-house PD/ET patients (21 ET), paired subject bootstrap:

| arm | AUC | vs untrimmed |
|---|---|---|
| untrimmed | 0.578 [0.447, 0.701] | — |
| trim-start | 0.563 | −0.015 [−0.042, +0.009] |
| trim-end | 0.592 | +0.015 [−0.004, +0.036] |

**Every arm sits below the 0.655 detection floor.** All three are
indistinguishable from chance and the differences between them are noise. The
spectrum-only PD-vs-ET model does not transfer PADS → in-house with or without
the onset; the onset is a second-order contamination (~5 % of in-band energy)
sitting on a cohort gap far larger than it. This test cannot exclude a small
effect — nothing below the floor can — but it excludes the onset as the *binding*
reason for non-transfer.

## (B) Mixed-cohort headline, 20 splits — null, as predicted

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **untrimmed (current)** | 0.650 | 0.652 | 0.654 | **0.652** | **0.602** |
| trim-start 1.5 s | 0.645 | 0.658 | 0.634 | 0.646 | 0.591 |
| trim-end 1.5 s (control) | 0.613 | 0.665 | 0.678 | 0.652 | 0.590 |

paired vs untrimmed:

    trim-start   precET -0.020 [-0.093, +0.055]   macroP -0.006 [-0.031, +0.022]
    trim-end     precET +0.024 [-0.025, +0.071]   macroP -0.000 [-0.017, +0.018]
                 precN  -0.037 [-0.058, -0.019] *

Removing the onset is null on every column. The one significant cell is in the
*control*: cutting the last 1.5 s costs N precision. So the tail of a PADS
recording carries something the model uses for N, while the head — onset and
all — does not. Consistent with the model having learned to discount the onset.

A limitation stated in advance: `DESC` and `TRAJ` come from `build()` and are
untrimmed in every arm, so (B) leaves part of the onset in the model. That
biases (B) toward null, and it *is* null — but it is not a fully trimmed test of
the reported model. (C) used the spectrum alone and was fully trimmed.

## Standing

* **Leave `--trim-start` at 0.** Removing the onset does not change the reported
  model (macroP −0.006) and does not lift transfer out of the chance region.
* **The onset is a documented cohort inconsistency, not a harm.** Record it in
  the paper's preprocessing section; do not claim it as a confound that was
  fixed, because fixing it measured nothing.
* **Do not trim the end.** It is the one operation here that significantly hurts
  (precN −0.037 *).
* The predictions on record: mechanism **held**, headline-small **held**,
  transfer-improves **failed**. Three predictions, two derived from measurements
  of the data, one from a mechanism story about why non-transfer happens — and
  the mechanism story is the one that failed, again.
