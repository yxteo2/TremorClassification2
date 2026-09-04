# Cropped training: 9.2× the rows, and the win rate says nothing happened

## What this is

**Cropped training** from the EEG deep-learning literature (Schirrmeister et
al., *Human Brain Mapping* 38(11) 2017) — the standard answer to a data-limited
decoder. Train on many overlapping crops of each trial, score per crop,
aggregate at inference. Here: 4 s windows at 2 s hop, patient-level scoring by
averaging a patient's window probabilities.

    patients 404   windows 3705   median 8 per patient (min 4, max 19)
    training rows: 3705 windows vs 404 patients = 9.2x more

It is a **training device, not a representation change**, which is what
separates it from `window_vs_patient_level.md` (which scored each window as its
own case and lost) and from `tf_window_length.md` (which changed the spectrum).
Every metric below is patient-level. Splits stay patient-disjoint: all of a
patient's windows sit in one fold.

The experiment had been on disk and unrun since August — one of the orphans
`experiments/INDEX.md` names. It is reported here.

## Result — 20 splits, same architecture, same features, same splits

| | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| patient-trained (control) | 0.651 | 0.628 | 0.547 | 0.609 | 0.569 |
| **window-trained, patient-scored** | 0.657 | 0.644 | 0.574 | 0.625 | 0.582 |
| sd (window arm) | 0.061 | 0.057 | **0.178** | 0.072 | 0.062 |

**Paired, window − patient:**

| | delta | 95 % CI | win rate |
|---|---|---|---|
| precN | +0.006 | [−0.024, +0.036] | 0.50 |
| precPD | +0.016 | [−0.022, +0.056] | 0.70 |
| precET | +0.026 | [−0.051, +0.128] | 0.50 (0.10 tied) |
| **macroP** | **+0.016** | [−0.016, +0.056] | **0.45** |
| macroF1 | +0.014 | [−0.016, +0.047] | 0.45 |

## The win rate is what decides this

Every column is positive and every column is null, which on its own would read
as "promising, needs more splits". **The win rates say otherwise: macroP is
positive on average while losing 11 of 20 splits.** That is the exact pattern
invariant 5 exists for — a positive mean carried by a few favourable folds
rather than a method that helps.

precET is worse than null, it is *uninformative*: mean +0.026 against a CI half
width of ~0.09 and a per-split sd of 0.178, the noisiest column measured in this
project. Twenty splits cannot resolve anything here.

## The recorded prediction failed

> **Negative on precET**, because `window_vs_patient_level.md` already measured
> that aggregating windows to patients *helps* — averaging the spectrum over a
> whole recording is a denoising step, and window training makes the network fit
> the un-denoised rows.

It came out **+0.026, positive and null**. The denoising argument did not
transfer from evaluation to training, and the reason is visible in hindsight:
the window arm still aggregates to patients *at inference*, so it gets the
denoising too. Only its gradient signal is noisier. That is a much weaker
handicap than the one predicted.

**The sub-prediction was designed so either outcome was informative** (the
lesson from prediction 12) and it did not pay out either. If training rows were
the binding constraint the gain should be largest for the scarcest class. ET is
scarcest (49 patients) and does show the largest gain (+0.026), but the rest of
the ordering is not monotone in class size — PD is the *most* common class
(188) and gains second-most (+0.016), ahead of N (167, +0.006). With every
interval spanning zero, this cannot distinguish "rows bind" from noise. **The
question is left open, not answered.**

## Limit worth stating

The control here is a single `Spectrum1DCNN` on welch log-bins — macroP 0.609
against the reported model's 0.652. No descriptors, no trajectory stream, no
TCN, no six-member ensemble. So this measures cropped training **inside a weaker
pipeline than the reported one**, and whether a null there implies a null in the
full model is untested. Given the win rates, re-running it inside the full
pipeline is not a priority.

## Standing

* **Do not adopt cropped training.** Positive means, sub-0.5 win rate on the
  headline column, nothing significant.
* **The 9.2× is 9.2× of correlated rows.** A patient contributes ~8 windows of
  one recording; the effective number of independent units is still 404
  patients. This is consistent with — though not proof of — why capacity,
  augmentation and resampling arms have all failed here.
* **Report the win rate next to every paired mean.** This experiment would have
  been written up as "+0.016 macroP, promising" on the means alone.
